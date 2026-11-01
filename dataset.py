"""
dataset.py — clip dataset, clip-consistent augmentation, track-capped sampler.

A sample = T crops of ONE person-track, read from the cache made by build_clips.py.
Dataset returns uint8 tensors (T,3,S,S); normalisation is done on the GPU in train.py.

Design notes
------------
* Spatial augmentations are applied with the SAME parameters to all T frames of a
  clip, so temporal consistency is preserved (a "clip" is one continuous video segment).
* Random erasing is intentionally conservative: a phone occupies only ~1-3% of the
  crop area. Aggressive erasing (p>0.2, area>0.05) destroys the label signal and
  was found to hurt performance in ablation.
* Temporal reversal is NOT used: "raising" vs "lowering" a phone is semantically
  different and reversing the clip breaks label meaning.
* Temporal dropout is NOT used: replacing frames with neighbours can hide the exact
  moment the phone is visible, adding label noise.
* Feature-level augmentation (mixup/cutmix on frozen features) was removed: it was
  never used and its semantics were ambiguous.

Augmentation levels (pass to ClipDataset via aug_level):
    "none"       : eval transform only (resize, no augmentation)
    "basic"      : crop + flip + photometric + gamma + blur + noise
    "clean"      : basic + random affine (RECOMMENDED DEFAULT)
    "full"       : clean + light JPEG + very small random erasing
    "aggressive" : full with stronger magnitudes (upper bound for sensitivity analysis)
"""
import io
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, Sampler

cv2.setNumThreads(0)


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def _gauss_blur(x, sigma):
    """Separable Gaussian blur on (T,C,H,W) float32. Same sigma for all frames."""
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.arange(-r, r + 1, dtype=x.dtype)
    k = torch.exp(-(k ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kh, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kv, groups=c)


def _random_affine(x, max_rot=10.0, max_translate=0.05):
    """Same rotation + translation applied to all frames of a clip.

    x: (T,C,H,W) float32 in [0,1].
    """
    T, C, H, W = x.shape
    angle = random.uniform(-max_rot, max_rot)
    tx = random.uniform(-max_translate, max_translate) * 2
    ty = random.uniform(-max_translate, max_translate) * 2

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    theta = torch.tensor([[
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
    ]], dtype=x.dtype)                                   # (1, 2, 3)

    grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)
    xf = x.permute(1, 0, 2, 3).reshape(1, T * C, H, W)   # flatten T into channels
    out = F.grid_sample(xf, grid, mode="bilinear",
                        padding_mode="reflection", align_corners=False)
    return out.reshape(C, T, H, W).permute(1, 0, 2, 3)


def _random_erase_clip(x, p=0.15, area=(0.005, 0.03), ratio=(0.3, 3.3),
                       fill_value=0.5):
    """Small random erasing with the SAME rectangle across all frames.

    Conservative by design: phone occupies ~1-3% of the crop, so we keep
    `area` and `p` low. Fills with a fixed gray value (0.5) rather than noise
    to avoid introducing spurious high-frequency patterns.

    x: (T,C,H,W) float32 in [0,1].
    """
    if random.random() >= p:
        return x
    T, C, H, W = x.shape
    total = H * W
    for _ in range(20):
        target = random.uniform(*area) * total
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        h = int(round(math.sqrt(target / ar)))
        w = int(round(math.sqrt(target * ar)))
        if 0 < h < H and 0 < w < W:
            break
    else:
        return x
    top = random.randint(0, H - h)
    left = random.randint(0, W - w)
    x = x.clone()
    x[:, :, top:top + h, left:left + w] = fill_value
    return x


def _jpeg_compress_clip(x_uint8, quality):
    """Per-frame JPEG compression. Simulates video-compression artefacts.

    Note: JPEG is applied per-frame (not clip-consistent) on purpose — real
    video codecs also vary between I/P/B frames.
    """
    out = []
    for t in range(x_uint8.shape[0]):
        arr = x_uint8[t].permute(1, 2, 0).numpy()        # H,W,3
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img2 = Image.open(buf).convert("RGB")
        out.append(torch.from_numpy(np.array(img2)).permute(2, 0, 1))
    return torch.stack(out).to(torch.uint8)


# --------------------------------------------------------------------------- #
# main augmentation
# --------------------------------------------------------------------------- #
def augment_clip(x, size, level="clean", scale=(0.8, 1.0), ratio=(0.9, 1.1)):
    """Apply clip-consistent augmentation.

    x: uint8 (T,3,H,W)
    returns: uint8 (T,3,size,size)

    Level semantics:
      none       : eval transform only
      basic      : crop + flip + photometric + gamma + blur + noise
      clean      : basic + affine                        <-- default
      full       : clean + light jpeg + small erasing
      aggressive : full with stronger magnitudes
    """
    if level == "none":
        return resize_clip(x, size)

    # ---- level flags ----
    is_clean = level in ("clean", "full", "aggressive")
    is_full  = level in ("full", "aggressive")
    is_aggr  = level == "aggressive"

    T, _, H, W = x.shape
    x = x.float().div_(255.0)

    # -------- 1. random resized crop (tracker-box noise) ------------------- #
    s = random.uniform(*scale)
    r = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
    w = min(W, max(8, int(round(math.sqrt(H * W * s * r)))))
    h = min(H, max(8, int(round(math.sqrt(H * W * s / r)))))
    top, left = random.randint(0, H - h), random.randint(0, W - w)
    x = F.interpolate(x[:, :, top:top + h, left:left + w], size=(size, size),
                      mode="bilinear", antialias=True, align_corners=False)

    # -------- 2. random affine (clean and above) -------------------------- #
    if is_clean:
        max_rot = 15.0 if is_aggr else 10.0
        x = _random_affine(x, max_rot=max_rot, max_translate=0.05)

    # -------- 3. horizontal flip ------------------------------------------ #
    if random.random() < 0.5:
        x = x.flip(-1)

    # -------- 4. photometric: brightness / contrast / saturation ---------- #
    b   = (0.5, 1.5) if is_aggr else (0.6, 1.4)
    c   = (0.6, 1.4) if is_aggr else (0.7, 1.3)
    sat = (0.6, 1.4) if is_aggr else (0.7, 1.3)
    x = x * random.uniform(*b)
    m = x.mean()
    x = (x - m) * random.uniform(*c) + m
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    x = gray + (x - gray) * random.uniform(*sat)
    x = x.clamp(0, 1)

    # -------- 5. per-channel color jitter (small, helps with color temp) -- #
    # Small independent gain per RGB channel — models different lighting
    # (tungsten, fluorescent, daylight) which is common across scenes.
    if is_clean:
        gain_range = 0.10 if is_aggr else 0.06
        gain = 1.0 + (torch.rand(1, 3, 1, 1) * 2 - 1) * gain_range
        x = (x * gain).clamp(0, 1)

    # -------- 6. gamma ---------------------------------------------------- #
    p_gamma = 0.4 if is_clean else 0.3
    if random.random() < p_gamma:
        g_range = (0.5, 1.8) if is_aggr else (0.7, 1.6)
        x = x ** random.uniform(*g_range)

    # -------- 7. gaussian blur -------------------------------------------- #
    p_blur = 0.3 if is_clean else 0.2
    if random.random() < p_blur:
        s_range = (0.3, 2.0) if is_aggr else (0.3, 1.5)
        x = _gauss_blur(x, random.uniform(*s_range))

    # -------- 8. sensor noise --------------------------------------------- #
    p_noise = 0.3 if is_clean else 0.2
    if random.random() < p_noise:
        n_range = (0.005, 0.04) if is_aggr else (0.005, 0.03)
        x = x + torch.randn_like(x) * random.uniform(*n_range)

    x = x.clamp(0, 1)

    # -------- 9. random erasing (full only, SMALL) ------------------------ #
    if is_full:
        p_erase = 0.25 if is_aggr else 0.15
        area    = (0.005, 0.05) if is_aggr else (0.005, 0.03)
        x = _random_erase_clip(x, p=p_erase, area=area, fill_value=0.5)

    # -------- 10. jpeg compression (full only, light) --------------------- #
    x_uint8 = x.mul_(255.0).round_().to(torch.uint8)
    if is_full and random.random() < 0.15:
        q = random.randint(60, 90) if is_aggr else random.randint(75, 95)
        x_uint8 = _jpeg_compress_clip(x_uint8, q)

    return x_uint8


def resize_clip(x, size):
    """Eval transform: full crop, resize to square. uint8 -> uint8."""
    x = x.float().div_(255.0)
    x = F.interpolate(x, size=(size, size), mode="bilinear",
                      antialias=True, align_corners=False)
    return x.clamp(0, 1).mul_(255.0).round_().to(torch.uint8)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class ClipDataset(Dataset):
    def __init__(self, cache_root, clips: pd.DataFrame, train: bool, size: int = 224,
                 aug_level: str = "clean"):
        self.root = Path(cache_root)
        self.df = clips.reset_index(drop=True)
        self.train, self.size = train, size
        self.aug_level = aug_level if train else "none"
        self.scene = self.df.scene.to_numpy()
        self.video = self.df.video.to_numpy()
        self.track = self.df.track.to_numpy().astype(int)
        self.start = self.df.start.to_numpy().astype(int)
        self.stride = self.df.stride.to_numpy().astype(int)
        self.T = int(self.df["T"].iloc[0])
        self.label = self.df.label.to_numpy().astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        base = self.root / "crops" / self.scene[i] / self.video[i] / str(self.track[i])
        frames = []
        for k in range(self.T):
            p = base / f"{self.start[i] + k * self.stride[i]:05d}.jpg"
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(p)
            frames.append(im[:, :, ::-1])                    # BGR -> RGB
        x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))) \
                 .permute(0, 3, 1, 2).contiguous()
        if self.train:
            x = augment_clip(x, self.size, level=self.aug_level)
        else:
            x = resize_clip(x, self.size)
        return x, int(self.label[i]), i


class TrackCappedSampler(Sampler):
    """Each epoch: at most `cap` random clips per track; positives oversampled
    to `pos_frac`."""
    def __init__(self, clips: pd.DataFrame, cap=24, pos_frac=0.4, seed=0):
        self.label = clips.label.to_numpy().astype(int)
        key = clips.video.astype(str) + "#" + clips.track.astype(str)
        self.groups = [g.to_numpy() for _, g in
                       pd.Series(np.arange(len(clips))).groupby(key.to_numpy())]
        self.cap, self.pos_frac, self.seed, self.epoch = cap, pos_frac, seed, 0
        self.n = len(self._draw(random.Random(seed)))

    def set_epoch(self, e):
        self.epoch = e

    def _draw(self, rng):
        picks = []
        for g in self.groups:
            picks += rng.sample(list(g), min(self.cap, len(g)))
        pos = [i for i in picks if self.label[i] == 1]
        neg = [i for i in picks if self.label[i] == 0]
        if self.pos_frac and pos and neg:
            target = int(len(neg) * self.pos_frac / (1 - self.pos_frac))
            if len(pos) < target:
                pos = pos + rng.choices(pos, k=target - len(pos))
        picks = pos + neg
        rng.shuffle(picks)
        return picks

    def __iter__(self):
        picks = self._draw(random.Random(self.seed * 1000 + self.epoch))
        if len(picks) < self.n:
            picks = picks + random.Random(self.epoch).choices(
                picks, k=self.n - len(picks))
        return iter(picks[:self.n])

    def __len__(self):
        return self.n
