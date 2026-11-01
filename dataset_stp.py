#!/usr/bin/env python3
"""
dataset_stp.py

Sliding-window dataset for the STP branch.

Design
------
- Source of truth for frames:   phone_cache_v2/frames.csv
- Source of truth for FPS:      phone_cache_v2/videos_meta.csv
- Crop images:                  phone_cache_v2/crops/...

Unlike dataset.py, this file does NOT rely on clips_train.csv / clips_eval.csv.
Windows are built at index time so that `hop` is a first-class parameter
expressed in "sampled steps" (units of target_dt seconds).

Time-grid sampling
------------------
For each track we map time t (seconds, starting at 0) to a raw video frame via
    frame = round(t * fps)
and take T samples at times:
    t_start + i * target_dt,   i = 0..T-1
with target_dt default 0.1 s. This keeps the temporal spacing honest across
videos with different FPS (12.5, 20, 24, 25, 30 in this dataset).

Roles and hop
-------------
- train : hop = args.hop_train sampled steps   (default 4)
- val   : hop = args.hop_eval  sampled steps   (default 1)
- test  : hop = args.hop_eval  sampled steps   (default 1)

Train-only filtering
--------------------
Windows with purity < min_purity_train (default 0.8) are dropped ONLY during
training; validation and test keep all windows so the evaluation distribution
matches deployment.

Split
-----
Scene-level K-fold from folds.json: {"K": int, "fold_of_scene": {scene: fold}}.
Role of a scene s for a given test fold f:
    test  if fold_of_scene[s] == f
    val   if fold_of_scene[s] == (f + 1) % K
    train otherwise
"""
from __future__ import annotations

# ---- path bootstrap: make `dataset.py` importable ------------------------- #
import sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_MAIN = _HERE.parent / "yolo_cvat_tracking"
if _MAIN.exists() and str(_MAIN) not in sys.path:
    sys.path.insert(0, str(_MAIN))
# --------------------------------------------------------------------------- #

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import augment_clip, resize_clip, TrackCappedSampler


# =========================================================================== #
# Cache loading
# =========================================================================== #
def load_stp_cache(cache_root: str | Path):
    """Return (frames, tracks, meta) DataFrames from phone_cache_v2."""
    cache_root = Path(cache_root)
    frames = pd.read_csv(cache_root / "frames.csv")
    tracks = pd.read_csv(cache_root / "tracks.csv")
    meta   = pd.read_csv(cache_root / "videos_meta.csv")
    return frames, tracks, meta


# =========================================================================== #
# Scene-level fold split
# =========================================================================== #
def _role_of(scene: str, fold_of_scene: dict, test_fold: int, K: int) -> str:
    f = int(fold_of_scene[scene])
    if f == test_fold:
        return "test"
    if f == (test_fold + 1) % K:
        return "val"
    return "train"


# =========================================================================== #
# Main dataset
# =========================================================================== #
class ClipDatasetSTP(Dataset):
    """Track-level sliding-window clip dataset for STP.

    Parameters
    ----------
    cache_root : str | Path
        phone_cache_v2 root directory.
    frames_df : pd.DataFrame
        Subset of frames.csv containing only the tracks for this role.
        Must have columns: scene, video, track, frame, label, path.
    meta_df : pd.DataFrame
        Full videos_meta.csv (used for FPS).
    role : {"train", "val", "test"}
    size : int
        Output crop size (224 for STP v1).
    aug_level : str
        Augmentation level name understood by dataset.augment_clip.
    T : int
        Number of temporal samples per window.
    target_dt : float
        Temporal spacing in seconds between two consecutive samples.
    hop_samples : int
        Number of sampled steps between consecutive window starts.
        train default 4, val/test default 1.
    min_purity_train : float | None
        Drop windows with purity < this value during training only.
        Pass None to disable.
    """

    def __init__(
        self,
        cache_root: str | Path,
        frames_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        role: str = "train",
        size: int = 224,
        aug_level: str = "clean",
        T: int = 16,
        target_dt: float = 0.1,
        hop_samples: int = 1,
        min_purity_train: Optional[float] = 0.8,
        load_fpi_feats: bool = False
    ):
        assert role in ("train", "val", "test"), role
        assert hop_samples >= 1, hop_samples

        self.cache_root = Path(cache_root)
        self.role = role
        self.train = (role == "train")
        self.size = int(size)
        self.aug_level = aug_level if self.train else "none"
        self.T = int(T)
        self.target_dt = float(target_dt)
        self.hop_samples = int(hop_samples)
        self.min_purity_train = min_purity_train
        self.load_fpi_feats = bool(load_fpi_feats)
        self.fpi_root = Path(cache_root) / "fpi_feats"

        # ---- FPS lookup ----------------------------------------------------- #
        meta_df = meta_df.copy()
        meta_df["key"] = list(zip(meta_df["scene"], meta_df["stem"]))
        self._fps = dict(zip(meta_df["key"], meta_df["fps"].astype(float)))

        # ---- per-track frame arrays ---------------------------------------- #
        # track_key = (scene, video, track_id)
        self.track_frames: dict[tuple, np.ndarray] = {}
        self.track_labels: dict[tuple, np.ndarray] = {}
        self.track_paths:  dict[tuple, list]      = {}
        self.track_fps:    dict[tuple, float]     = {}

        frames_df = frames_df.sort_values(["scene", "video", "track", "frame"])
        self.frames_df = frames_df
        for (scene, video, track), grp in frames_df.groupby(
            ["scene", "video", "track"], sort=False
        ):
            key = (scene, video, int(track))
            self.track_frames[key] = grp["frame"].to_numpy()
            self.track_labels[key] = grp["label"].to_numpy().astype(int)
            self.track_paths[key]  = grp["path"].tolist()

            fps = self._fps.get((scene, video), None)
            if fps is None or fps <= 0:
                raise ValueError(f"Missing/invalid FPS for ({scene}, {video})")
            self.track_fps[key] = float(fps)

        # ---- build window index -------------------------------------------- #
        self.samples = self._build_index()

        # ------------------------------------------------------------------ #
    def _build_index(self) -> list:
        """Return list of (track_key, positions, label, purity, start_sample)."""
        samples = []
        skipped_no_frame = 0
        skipped_purity = 0

        for key, frame_ids in self.track_frames.items():
            fps = self.track_fps[key]
            stride_f = self.target_dt * fps
            first_frame = int(frame_ids.min())
            last_frame = int(frame_ids.max())

            span_f = (self.T - 1) * stride_f
            hop_f = self.hop_samples * stride_f

            if last_frame - first_frame < span_f:
                continue

            id_to_pos = {int(fid): i for i, fid in enumerate(frame_ids)}
            labels = self.track_labels[key]
            max_gap = max(2, int(round(stride_f)))

            n_windows = int((last_frame - first_frame - span_f) // hop_f) + 1

            for w in range(n_windows):
                start_f = int(round(first_frame + w * hop_f))
                frame_idx = [int(round(start_f + k * stride_f)) for k in range(self.T)]

                positions = []
                bad = False
                for f in frame_idx:
                    p = id_to_pos.get(f)
                    if p is None:
                        nearest_pos = int(np.argmin(np.abs(frame_ids - f)))
                        if abs(int(frame_ids[nearest_pos]) - f) > max_gap:
                            bad = True
                            break
                        p = nearest_pos
                    positions.append(p)
                if bad:
                    skipped_no_frame += 1
                    continue

                win_labels = labels[positions]
                pos_count = int((win_labels == 1).sum())
                neg_count = self.T - pos_count
                label = 1 if pos_count > neg_count else 0
                purity = max(pos_count, neg_count) / self.T

                if (self.train
                        and self.min_purity_train is not None
                        and purity < self.min_purity_train):
                    skipped_purity += 1
                    continue

                samples.append((key, positions, label, purity, w))

        if self.train and skipped_purity:
            print(f"[ClipDatasetSTP/{self.role}] dropped {skipped_purity} low-purity windows")
        if skipped_no_frame:
            print(f"[ClipDatasetSTP/{self.role}] dropped {skipped_no_frame} windows (missing frames)")

        return samples

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        key, positions, label, _purity, _start = self.samples[i]
        scene, video, track = key
        rel_paths = self.track_paths[key]

        frames = []
        for p in positions:
            full = self.cache_root / rel_paths[p]
            im = cv2.imread(str(full), cv2.IMREAD_COLOR)
            if im is None:
                raise FileNotFoundError(full)
            frames.append(im[:, :, ::-1])

        x = torch.from_numpy(np.ascontiguousarray(np.stack(frames))) \
                 .permute(0, 3, 1, 2).contiguous()

        if self.train:
            x = augment_clip(x, self.size, level=self.aug_level)
        else:
            x = resize_clip(x, self.size)

        if self.load_fpi_feats:
            frame_ids = self.track_frames[key][positions]      # (T,)
            fpi = np.zeros((self.T, 11), dtype=np.float32)
            track_dir = self.fpi_root / str(scene) / str(video) / str(track)
            for k, fid in enumerate(frame_ids):
                fp = track_dir / f"{int(fid):05d}.npy"
                if fp.exists():
                    fpi[k] = np.load(fp)
            fpi_t = torch.from_numpy(fpi).float()
            has_fpi = (fpi_t[:, 8] > 0).float()
            return x, int(label), i, fpi_t, has_fpi

        return x, int(label), i

    # ------------------------------------------------------------------ #
    def window_info(self, i: int) -> dict:
        """Return window metadata for visualization / debugging."""
        key, positions, label, purity, start_sample = self.samples[i]
        scene, video, track = key
        fps = self.track_fps[key]
        frame_ids = [int(self.track_frames[key][p]) for p in positions]
        times = [round(f / fps, 4) for f in frame_ids]
        return {
            "scene": scene,
            "video": video,
            "track": track,
            "start_sample": start_sample,
            "frame_ids": frame_ids,
            "times": times,
            "fps": fps,
            "label": label,
            "purity": purity,
        }

    # ------------------------------------------------------------------ #
    def to_dataframe(self) -> pd.DataFrame:
        """Export all windows as a DataFrame for audit / aggregation."""
        # lookup box_h از frames_df
        has_box_h = "box_h" in self.frames_df.columns
        box_h_lookup = {}
        if has_box_h:
            for row in self.frames_df.itertuples(index=False):
                key = (row.scene, row.video, int(row.track), int(row.frame))
                box_h_lookup[key] = float(row.box_h)

        rows = []
        for i in range(len(self)):
            info = self.window_info(i)
            bh = None
            if has_box_h:
                bhs = [
                    box_h_lookup.get((info["scene"], info["video"], info["track"], fid))
                    for fid in info["frame_ids"]
                ]
                bhs = [b for b in bhs if b is not None]
                bh = float(np.median(bhs)) if bhs else None
            rows.append({
                "idx": i,
                "scene": info["scene"],
                "video": info["video"],
                "track": info["track"],
                "start_sample": info["start_sample"],
                "label": info["label"],
                "purity": info["purity"],
                "fps": info["fps"],
                "first_frame": info["frame_ids"][0],
                "last_frame": info["frame_ids"][-1],
                "box_h": bh,
            })
        return pd.DataFrame(rows)


# =========================================================================== #
# Factory: build train / val / test for one fold
# =========================================================================== #
def make_stp_datasets(
    cache_root: str | Path,
    folds_json: str | Path,
    fold: int,
    size: int = 224,
    aug_level: str = "clean",
    T: int = 16,
    target_dt: float = 0.1,
    hop_train: int = 4,
    hop_eval: int = 1,
    min_purity_train: Optional[float] = 0.8,
    verbose: bool = True,
    hop_val: Optional[int] = None,
    load_fpi_feats: bool = False,
):
    """Return (train_ds, val_ds, test_ds) for the given scene-level fold."""
    frames, _tracks, meta = load_stp_cache(cache_root)

    fj = json.loads(Path(folds_json).read_text(encoding="utf-8"))
    K = int(fj["K"])
    fold_of_scene = fj["fold_of_scene"]

    frames = frames.copy()
    frames["__role__"] = frames["scene"].map(
        lambda s: _role_of(s, fold_of_scene, fold, K)
    )

    role_frames = {
        r: frames[frames["__role__"] == r].drop(columns="__role__").reset_index(drop=True)
        for r in ("train", "val", "test")
    }

    train_ds = ClipDatasetSTP(
        cache_root, role_frames["train"], meta, role="train",
        size=size, aug_level=aug_level, T=T, target_dt=target_dt,
        hop_samples=hop_train, min_purity_train=min_purity_train, load_fpi_feats=load_fpi_feats
    )
    val_ds = ClipDatasetSTP(
        cache_root, role_frames["val"], meta, role="val",
        size=size, aug_level="none", T=T, target_dt=target_dt,
        hop_samples=(hop_val if hop_val is not None else hop_eval), min_purity_train=None, load_fpi_feats=load_fpi_feats
    )
    test_ds = ClipDatasetSTP(
        cache_root, role_frames["test"], meta, role="test",
        size=size, aug_level="none", T=T, target_dt=target_dt,
        hop_samples=hop_eval, min_purity_train=None, load_fpi_feats=load_fpi_feats
    )

    if verbose:
        for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            n_pos = sum(1 for s in ds.samples if s[2] == 1)
            n_neg = len(ds.samples) - n_pos
            prev = n_pos / max(1, len(ds.samples))
            print(f"[fold {fold}] {name:5s}  n={len(ds):6d}  "
                  f"pos={n_pos:5d}  neg={n_neg:5d}  prevalence={prev:.3f}  "
                  f"tracks={len(ds.track_frames)}  hop={ds.hop_samples}  T={ds.T}")

    return train_ds, val_ds, test_ds


# =========================================================================== #
# Sampler helper
# =========================================================================== #
def stp_meta_df(dataset: ClipDatasetSTP) -> pd.DataFrame:
    """Build a DataFrame compatible with dataset.TrackCappedSampler.

    `video` field is unique-per-scene by concatenating scene||video.
    """
    rows = []
    for key, _positions, label, purity, _start in dataset.samples:
        scene, video, track = key
        rows.append({
            "scene": scene,
            "video": f"{scene}||{video}",
            "track": track,
            "label": label,
            "purity": purity,
        })
    return pd.DataFrame(rows)


def make_stp_sampler(
    dataset: ClipDatasetSTP,
    cap_per_track: int = 24,
    pos_frac: float = 0.4,
    seed: int = 0,
) -> TrackCappedSampler:
    """Wrap TrackCappedSampler for a ClipDatasetSTP train split."""
    meta = stp_meta_df(dataset)
    return TrackCappedSampler(meta, cap=cap_per_track, pos_frac=pos_frac, seed=seed)


# =========================================================================== #
# Self-audit — run as `python dataset_stp.py --cache ... --folds ... --fold 0`
# =========================================================================== #
def _audit():
    import argparse
    import time

    p = argparse.ArgumentParser(description="Audit STP sliding-window dataset")
    p.add_argument("--cache", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--target-dt", type=float, default=0.1)
    p.add_argument("--hop-train", type=int, default=4)
    p.add_argument("--hop-eval", type=int, default=1)
    p.add_argument("--min-purity-train", type=float, default=0.8)
    args = p.parse_args()

    print("=" * 72)
    print(f"STP dataset audit  fold={args.fold}  T={args.T}  Δt={args.target_dt}s")
    print("=" * 72)

    t0 = time.time()
    train_ds, val_ds, test_ds = make_stp_datasets(
        args.cache, args.folds, args.fold,
        size=args.size, aug_level="clean",
        T=args.T, target_dt=args.target_dt,
        hop_train=args.hop_train, hop_eval=args.hop_eval,
        min_purity_train=args.min_purity_train,
    )
    print(f"\n[built in {time.time() - t0:.2f}s]\n")

    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        print(f"--- {name} ---")
        print(f"  n_windows     : {len(ds)}")
        print(f"  n_tracks      : {len(ds.track_frames)}")
        print(f"  hop_samples   : {ds.hop_samples}")
        if len(ds) == 0:
            continue

        # class balance
        labels = np.array([s[2] for s in ds.samples])
        purities = np.array([s[3] for s in ds.samples])
        print(f"  prevalence    : {labels.mean():.3f}")
        print(f"  purity mean   : {purities.mean():.3f}  min={purities.min():.3f}")
        print(f"  purity < 0.8  : {(purities < 0.8).mean() * 100:.2f}%")

        # show a couple of windows
        for i in [0, len(ds) // 2, len(ds) - 1]:
            info = ds.window_info(i)
            t0_s = info["times"][0]
            t1_s = info["times"][-1]
            print(f"  ex[{i:6d}]  {info['video']:>30s}  tr={info['track']:2d}  "
                  f"label={info['label']}  purity={info['purity']:.2f}  "
                  f"t=[{t0_s:.2f}..{t1_s:.2f}]s  Δt≈{(t1_s - t0_s) / (args.T - 1):.3f}s")

        # actually load one sample to verify I/O
        x, y, idx = ds[0]
        print(f"  sample[0] shape={tuple(x.shape)} dtype={x.dtype} label={y}")
        print()

    print("=" * 72)
    print("AUDIT OK")
    print("=" * 72)


if __name__ == "__main__":
    _audit()
