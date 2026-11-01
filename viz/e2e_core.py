"""viz/e2e_core.py — pure logic of the deployed pipeline (no torch / ultralytics imports).

Everything that decides *what the user sees* lives here and is unit-testable:

  crop        : byte-for-byte copy of build_clips.py::crop_square (+ top_frac)
  stride      : same rule as training  -> stride = max(1, round(fps * 0.1))
  Stitcher    : keeps a logical id when the tracker re-issues a new id for the same person
  TrackVerdict: EMA + hysteresis + hold times (causal, replayable from cached raw probabilities)
  score_stream: crops -> (optional FPI) -> model -> raw probability series per logical id
  evaluation  : match to CVAT ground truth, compute frame-level / flip / latency / id-switch metrics

Self-test:  python -m viz.e2e_core
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from math import hypot

import cv2
import numpy as np
import pandas as pd

GRAY_BGR = (104, 116, 124)
KEEP = {"using_phone": 1, "not_using_phone": 0}
PRIOR_P = 0.25            # train prevalence ~0.29 / test ~0.21 -> used only to fill "no prediction yet"


# ====================================================================== #
# 1. CROP  (identical to build_clips.py)
# ====================================================================== #
@dataclass
class CropCfg:
    margin: float = 1.25
    size: int = 384           # size of the *cache* crops (FPI features were computed on this size)
    mask: bool = True
    box_margin: float = 0.15
    top_frac: float = 1.0


def crop_square(img, box, margin, size, mask=False, box_margin=0.15):
    """VERBATIM copy of build_clips.py::crop_square."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = (x2 - x1) * (1 + (box_margin if mask else 0.0))
    bh = (y2 - y1) * (1 + (box_margin if mask else 0.0))
    s = max(8, int(round(max(bw, bh) * margin)))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    canvas = np.empty((s, s, 3), np.uint8)
    canvas[:] = GRAY_BGR
    ax0, ay0, ax1, ay1 = x0, y0, x0 + s, y0 + s
    if mask:
        ax0, ay0 = max(ax0, int(round(cx - bw / 2))), max(ay0, int(round(cy - bh / 2)))
        ax1, ay1 = min(ax1, int(round(cx + bw / 2))), min(ay1, int(round(cy + bh / 2)))
    sx0, sy0, sx1, sy1 = max(0, ax0), max(0, ay0), min(W, ax1), min(H, ay1)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)


def make_crop(img, box, cfg: CropCfg):
    """Same call pattern as build_clips.py::process (top_frac applied to the box first)."""
    if cfg.top_frac < 1:
        box = (box[0], box[1], box[2], box[1] + (box[3] - box[1]) * cfg.top_frac)
    return crop_square(img, box, cfg.margin, cfg.size, cfg.mask, cfg.box_margin)


def frame_stride(fps, target_dt=0.1):
    """Same rule as build_clips.make_clips (Python round -> 25 fps gives 2, 30 fps gives 3)."""
    return max(1, int(round(fps * target_dt)))


# ====================================================================== #
# 2. FPI features (copy of cache_fpi_features / video_demo_online)
# ====================================================================== #
def fpi_features(xyxy, cls, conf, w, h):
    feats = np.zeros(11, dtype=np.float32)
    if len(xyxy) == 0:
        return feats
    FACE_CLASS, PHONE_CLASS = 1, 0
    face, phone = None, None
    for i, c in enumerate(cls):
        if c == FACE_CLASS and (face is None or conf[i] > face[4]):
            face = (*xyxy[i], conf[i])
        if c == PHONE_CLASS and (phone is None or conf[i] > phone[4]):
            phone = (*xyxy[i], conf[i])
    if face is not None:
        x1, y1, x2, y2, _ = face
        feats[0] = (x1 + x2) / 2.0 / w
        feats[1] = (y1 + y2) / 2.0 / h
        feats[2] = (x2 - x1) / w
        feats[3] = (y2 - y1) / h
    if phone is not None:
        x1, y1, x2, y2, cf = phone
        feats[4] = (x1 + x2) / 2.0 / w
        feats[5] = (y1 + y2) / 2.0 / h
        feats[6] = (x2 - x1) / w
        feats[7] = (y2 - y1) / h
        feats[8] = cf
    if face is not None and phone is not None:
        fx = (face[0] + face[2]) / 2
        fy = (face[1] + face[3]) / 2
        px = (phone[0] + phone[2]) / 2
        py = (phone[1] + phone[3]) / 2
        feats[9] = np.sqrt((fx - px) ** 2 + (fy - py) ** 2) / np.sqrt(w * h + 1e-6)
        ix1, iy1 = max(face[0], phone[0]), max(face[1], phone[1])
        ix2, iy2 = min(face[2], phone[2]), min(face[3], phone[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        fa = (face[2] - face[0]) * (face[3] - face[1])
        pa = (phone[2] - phone[0]) * (phone[3] - phone[1])
        feats[10] = inter / (fa + pa - inter + 1e-6)
    return feats


# ====================================================================== #
# 3. geometry
# ====================================================================== #
def iou_matrix(A, B):
    A = np.asarray(A, float).reshape(-1, 4)
    B = np.asarray(B, float).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0])
    iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2])
    iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


# ====================================================================== #
# 4. ID stitching
# ====================================================================== #
class Stitcher:
    """Maps raw tracker ids -> stable logical ids.

    A *new* raw id is linked to a *recently lost* logical track when the boxes are compatible
    (overlap or close, similar size). Linking is refused when the lost box touched the frame border
    (person probably left) unless the new box appears almost exactly there.
    Appearance is NOT used, so two people crossing paths can still be confused - this only fixes the
    common case 'tracker lost the person for a moment and issued a new id'.
    """

    def __init__(self, fps, frame_wh, max_gap_s=2.0, iou_min=0.10, dist_base=0.6, dist_per_s=0.4,
                 area_ratio=(0.4, 2.5), edge_frac=0.02, enabled=True):
        self.fps, self.W, self.H = float(fps), float(frame_wh[0]), float(frame_wh[1])
        self.max_gap = int(round(max_gap_s * self.fps))
        self.iou_min, self.dist_base, self.dist_per_s = iou_min, dist_base, dist_per_s
        self.area_ratio, self.edge = area_ratio, edge_frac
        self.enabled = enabled
        self.raw2log, self.last, self.next_lid = {}, {}, 1

    def _touches_border(self, b):
        return (b[0] <= self.edge * self.W or b[1] <= self.edge * self.H or
                b[2] >= (1 - self.edge) * self.W or b[3] >= (1 - self.edge) * self.H)

    def _score(self, f, lid, box):
        lf, lbox, _ = self.last[lid]
        gap_s = (f - lf) / self.fps
        a, b = np.asarray(lbox, float), np.asarray(box, float)
        area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
        area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        r = area_b / area_a
        if not (self.area_ratio[0] <= r <= self.area_ratio[1]):
            return None
        iou = float(iou_matrix([a], [b])[0, 0])
        d = hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)
        d /= max(1.0, a[3] - a[1], b[3] - b[1])
        allow = self.dist_base + self.dist_per_s * gap_s
        if self._touches_border(a):
            allow = min(allow, 0.3)
            if iou < self.iou_min and d > allow:
                return None
        if iou >= self.iou_min or d <= allow:
            return iou - 0.5 * d
        return None

    def update(self, f, dets):
        """dets: list of (raw_id, box_xyxy, conf).  -> list of (lid, raw_id, box, conf)."""
        if not self.enabled:                       # identity mapping: logical id == raw id
            out = [(rid, rid, tuple(box), conf) for rid, box, conf in dets]
            for lid, rid, box, _ in out:
                self.last[lid] = (f, box, rid)
            return out
        out, active = [], set()
        by_lid = defaultdict(list)
        pending = []
        for d in dets:
            lid = self.raw2log.get(d[0]) if self.enabled else None
            if lid is None:
                pending.append(d)
            else:
                by_lid[lid].append(d)
        for lid, ds in by_lid.items():
            last_raw = self.last.get(lid, (None, None, None))[2]
            ds = sorted(ds, key=lambda d: (d[0] != last_raw, -d[2]))
            out.append((lid, ds[0][0], ds[0][1], ds[0][2]))
            active.add(lid)
            for d in ds[1:]:                       # two raw ids claim the same logical track
                self.raw2log.pop(d[0], None)
                pending.append(d)

        if pending:
            cands = [lid for lid, (lf, _, _) in self.last.items()
                     if lid not in active and 0 < f - lf <= self.max_gap] if self.enabled else []
            pairs = []
            for pi, d in enumerate(pending):
                for lid in cands:
                    s = self._score(f, lid, d[1])
                    if s is not None:
                        pairs.append((s, pi, lid))
            pairs.sort(reverse=True)
            used_p, used_l = set(), set()
            assign = {}
            for s, pi, lid in pairs:
                if pi in used_p or lid in used_l:
                    continue
                used_p.add(pi); used_l.add(lid); assign[pi] = lid
            for pi, d in enumerate(pending):
                lid = assign.get(pi)
                if lid is None:
                    lid = self.next_lid
                    self.next_lid += 1
                self.raw2log[d[0]] = lid
                active.add(lid)
                out.append((lid, d[0], d[1], d[2]))

        for lid, rid, box, _ in out:
            self.last[lid] = (f, tuple(box), rid)
        for lid in [l for l, (lf, _, _) in self.last.items() if f - lf > self.max_gap]:
            del self.last[lid]
        return out


# ====================================================================== #
# 5. temporal verdict (causal)
# ====================================================================== #
@dataclass
class FilterCfg:
    alpha: float = 0.7        # EMA memory (offline results were validated with alpha=0.7)
    on: float = 0.65          # EMA must reach this to turn USING   (optimum threshold of the model ~0.62)
    off: float = 0.50         # EMA must fall to this to turn NOT USING
    min_real: int = 2         # real (non-padded) crops needed before a prediction is trusted
    init_trust: float = 0.6   # first trusted sample moves the EMA this far from the prior (1.0 = fully trusted)
    on_hold_s: float = 0.2    # ... and stay beyond `on` for this long
    off_hold_s: float = 0.6   # ... and stay below `off` for this long
    init_state: bool = False  # verdict before the first trusted prediction (False = NOT USING)
    min_phone: float = 0.0    # gate: turning USING needs EMA(share of window crops with a detected phone) >= this (0 = off)


class TrackVerdict:
    def __init__(self, cfg: FilterCfg):
        self.cfg, self.p, self.state, self.flips = cfg, None, cfg.init_state, 0
        self.ph = None
        self._cand = None

    def update(self, t_s, p_raw, n_real, phone=None):
        c = self.cfg
        if n_real < c.min_real:
            return self.state, self.p
        if phone is not None and phone == phone:
            self.ph = phone if self.ph is None else c.alpha * self.ph + (1 - c.alpha) * phone
        if self.p is None:
            self.p = PRIOR_P + c.init_trust * (p_raw - PRIOR_P)
        else:
            self.p = c.alpha * self.p + (1 - c.alpha) * p_raw
        if not self.state:
            gate_ok = c.min_phone <= 0 or self.ph is None or self.ph >= c.min_phone
            if self.p >= c.on and gate_ok:
                self._cand = t_s if self._cand is None else self._cand
                if t_s - self._cand >= c.on_hold_s:
                    self.state, self._cand = True, None
                    self.flips += 1
            else:
                self._cand = None
        else:
            if self.p <= c.off:
                self._cand = t_s if self._cand is None else self._cand
                if t_s - self._cand >= c.off_hold_s:
                    self.state, self._cand = False, None
                    self.flips += 1
            else:
                self._cand = None
        return self.state, self.p


def run_filter_all(praw: pd.DataFrame, fcfg: FilterCfg, fps: float):
    """praw columns: frame, lid, p_raw, n_real.  -> {lid: (frames, p_ema, state, flips)}"""
    out = {}
    for lid, g in praw.groupby("lid"):
        g = g.sort_values("frame")
        fr, pr, nr = g.frame.to_numpy(), g.p_raw.to_numpy(), g.n_real.to_numpy()
        ph = g.phone_frac.to_numpy() if "phone_frac" in g.columns else None
        v = TrackVerdict(fcfg)
        P, S = np.full(len(fr), np.nan), np.zeros(len(fr), bool)
        for k in range(len(fr)):
            s, p = v.update(fr[k] / fps, float(pr[k]), int(nr[k]), None if ph is None else float(ph[k]))
            S[k], P[k] = s, (np.nan if p is None else p)
        out[int(lid)] = (fr, P, S, v.flips)
    return out


def lookup(series, lid, frames, init_state=False):
    """State/prob of logical track `lid` at the given frames (causal: last update with frame <= f)."""
    if lid not in series:
        return np.full(len(frames), np.nan), np.full(len(frames), init_state)
    fr, P, S, _ = series[lid]
    idx = np.searchsorted(fr, frames, side="right") - 1
    ok = idx >= 0
    p = np.where(ok, P[np.clip(idx, 0, None)], np.nan)
    s = np.where(ok, S[np.clip(idx, 0, None)], init_state)
    return p, s


# ====================================================================== #
# 6. scoring: crops -> model -> raw probability series
# ====================================================================== #
def score_stream(frames, dets_by_frame, fps, frame_wh, predict_fn, fpi_fn=None,
                 crop_cfg: CropCfg = CropCfg(), T=16, target_dt=0.1, infer_ticks=2, early_ticks=4,
                 stitch=True, stitch_kw=None, progress_every=500):
    """frames: iterable of BGR frames.   dets_by_frame: {frame: [(raw_id, box, conf), ...]}
    predict_fn(clip_rgb[T,S,S,3] uint8, fpi[T,11] | None, has_fpi[T] | None) -> float prob(using)
    fpi_fn(crop_bgr) -> float32[11]      (BGR is what ultralytics expects for numpy arrays)

    Window = last T crops of a logical track, `stride` video frames apart (training rule); the window
    always *ends at the current frame*, so everything is causal.  If fewer than T crops exist the
    window is left-padded with the OLDEST crop (a "person has not moved yet" assumption) and n_real
    reports how many crops are real.  Predictions run on the first `early_ticks` ticks, then every
    `infer_ticks` ticks.
    """
    stride = frame_stride(fps, target_dt)
    st = Stitcher(fps, frame_wh, enabled=stitch, **(stitch_kw or {}))
    bufs, n_ticks = defaultdict(lambda: deque(maxlen=T)), defaultdict(int)
    box_rows, praw_rows = [], []
    for f, img in enumerate(frames):
        assigned = st.update(f, dets_by_frame.get(f, []))
        for lid, rid, box, conf in assigned:
            box_rows.append((f, lid, rid, *[float(v) for v in box], float(conf)))
        if f % stride == 0:
            for lid, rid, box, conf in assigned:
                crop = make_crop(img, box, crop_cfg)
                fpi = fpi_fn(crop) if fpi_fn is not None else None
                bufs[lid].append((crop, fpi))
                n_ticks[lid] += 1
                if n_ticks[lid] <= early_ticks or n_ticks[lid] % infer_ticks == 0:
                    items = list(bufs[lid])
                    n_real = len(items)
                    items = [items[0]] * (T - n_real) + items
                    clip = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c, _ in items])
                    fpi_arr = has = None
                    if fpi_fn is not None:
                        fpi_arr = np.stack([x for _, x in items]).astype(np.float32)
                        has = (fpi_arr[:, 8] > 0).astype(np.float32)
                    pf = float(has[T - n_real:].mean()) if has is not None else float("nan")
                    praw_rows.append((f, lid, float(predict_fn(clip, fpi_arr, has)), n_real, pf))
        if progress_every and f % progress_every == 0 and f:
            print(f"  [score] frame {f}  tracks so far={st.next_lid - 1}  predictions={len(praw_rows)}")
    boxes = pd.DataFrame(box_rows, columns=["frame", "lid", "rid", "x1", "y1", "x2", "y2", "conf"])
    praw = pd.DataFrame(praw_rows, columns=["frame", "lid", "p_raw", "n_real", "phone_frac"])
    return boxes, praw


# ====================================================================== #
# 7. ground truth + evaluation
# ====================================================================== #
def parse_xml(xml_path):
    """Same parsing as build_clips.py.  -> n_frames, {frame: [(tid, box, label)]}"""
    r = ET.parse(xml_path).getroot()
    n_frames = int(r.findtext("meta/task/size"))
    per_frame = defaultdict(list)
    for t in r.findall("track"):
        tid = int(t.get("id"))
        for b in t.findall("box"):
            if b.get("outside") == "1":
                continue
            a = b.find("attribute[@name='phone_activity']")
            lab = a.text if a is not None else "MISSING"
            if lab not in KEEP:
                continue
            box = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            per_frame[int(b.get("frame"))].append((tid, box, KEEP[lab]))
    return n_frames, per_frame


def gt_as_dets(per_frame):
    """Oracle mode: ground-truth boxes as if they came from a perfect detector+tracker."""
    d = defaultdict(list)
    for f, rows in per_frame.items():
        for tid, box, _ in rows:
            d[f].append((int(tid), box, 1.0))
    return d


def match_gt_pred(gt_by_frame, boxes: pd.DataFrame, iou_thr=0.5):
    from scipy.optimize import linear_sum_assignment
    groups = {f: g for f, g in boxes.groupby("frame")}
    rows = []
    for f, gts in gt_by_frame.items():
        g = groups.get(f)
        if g is None:
            continue
        G = np.array([b for _, b, _ in gts])
        M = iou_matrix(G, g[["x1", "y1", "x2", "y2"]].to_numpy())
        r, c = linear_sum_assignment(-M)
        lids, rids = g.lid.to_numpy(), g.rid.to_numpy()
        for i, j in zip(r, c):
            if M[i, j] >= iou_thr:
                rows.append((f, gts[i][0], gts[i][2], int(lids[j]), int(rids[j]), float(M[i, j])))
    return pd.DataFrame(rows, columns=["frame", "gt_tid", "gt_label", "lid", "rid", "iou"])


def attach_predictions(matches: pd.DataFrame, series, init_state=False):
    p = np.full(len(matches), np.nan)
    s = np.full(len(matches), init_state)
    for lid, idx in matches.groupby("lid").indices.items():
        pp, ss = lookup(series, int(lid), matches.frame.to_numpy()[idx], init_state)
        p[idx], s[idx] = pp, ss
    out = matches.copy()
    out["p"], out["state"] = p, s
    return out


def summarize(M: pd.DataFrame, fps_map, gt_len, flips):
    """M: matched rows over all videos with columns vid, frame, gt_tid, gt_label, lid, rid, p, state.
    fps_map {vid: fps};  gt_len {(vid, gt_tid): n GT frames};  flips {(vid, lid): n flips}."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    res = {"n_rows": int(len(M))}
    if len(M) == 0:
        return res
    y = M.gt_label.to_numpy().astype(int)
    s = M.state.to_numpy().astype(int)
    p = np.nan_to_num(M.p.to_numpy(), nan=PRIOR_P)
    tp, fp = int(((s == 1) & (y == 1)).sum()), int(((s == 1) & (y == 0)).sum())
    fn, tn = int(((s == 0) & (y == 1)).sum()), int(((s == 0) & (y == 0)).sum())
    res.update(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
               f1=2 * tp / max(1, 2 * tp + fp + fn), acc=(tp + tn) / len(M), fp=fp, fn=fn)
    if len(set(y)) == 2:
        res["auprc"] = float(average_precision_score(y, p))
        res["auroc"] = float(roc_auc_score(y, p))
    fps_row = M.vid.map(fps_map).to_numpy()
    first = M.groupby(["vid", "lid"]).frame.transform("min").to_numpy()
    early = (M.frame.to_numpy() - first) < fps_row
    res["early_acc_1s"] = float((s[early] == y[early]).mean()) if early.any() else float("nan")
    res["late_acc"] = float((s[~early] == y[~early]).mean()) if (~early).any() else float("nan")

    # flips per minute on matched logical tracks
    keys = M.groupby(["vid", "lid"]).size()
    minutes = sum(n / fps_map[v] / 60.0 for (v, _), n in keys.items())
    res["flips_per_min"] = float(sum(flips.get((v, l), 0) for (v, l) in keys.index) / max(minutes, 1e-9))

    settle, wrong_any, idsw_raw, idsw_log, cover, extra = [], [], [], [], [], []
    for (v, g), d in M.groupby(["vid", "gt_tid"]):
        fps = fps_map[v]
        idsw_raw.append(d.rid.nunique() - 1)
        idsw_log.append(d.lid.nunique() - 1)
        cover.append(len(d) / max(1, gt_len.get((v, g), len(d))))
        if d.gt_label.nunique() == 1:                       # constant-label person (89% of tracks)
            d = d.sort_values("frame")
            st_ = d.state.to_numpy().astype(int)
            wrong = st_ != d.gt_label.to_numpy()
            wrong_any.append(bool(wrong.any()))
            changes = int((np.diff(st_) != 0).sum())
            needed = 1 if st_[0] != int(d.gt_label.iloc[0]) else 0
            extra.append(max(0, changes - needed))          # state changes beyond the one needed
            fr = d.frame.to_numpy()
            settle.append(((fr[wrong].max() - fr.min() + 1) / fps) if wrong.any() else 0.0)
    per_lid = M.groupby(["vid", "lid"]).gt_tid.nunique()
    res["merges_per_lid"] = float((per_lid - 1).mean())        # >0: one logical id covered several GT people
    res.update(settle_med_s=float(np.median(settle)) if settle else float("nan"),
               settle_p90_s=float(np.percentile(settle, 90)) if settle else float("nan"),
               ever_wrong_frac=float(np.mean(wrong_any)) if wrong_any else float("nan"),
               extra_flips_per_track=float(np.mean(extra)) if extra else float("nan"),
               idsw_raw_per_gt=float(np.mean(idsw_raw)), idsw_lid_per_gt=float(np.mean(idsw_log)),
               coverage=float(np.mean(cover)), n_gt_tracks=len(idsw_raw))
    return res


def default_grid():
    g = []
    for alpha in (0.5, 0.7, 0.85):
        for on in (0.55, 0.62, 0.70, 0.80):
            for off in (0.35, 0.45, 0.55):
                if off >= on:
                    continue
                for min_real in (1, 3, 6):
                    for init_trust in (0.5, 1.0):
                        for on_h, off_h in ((0.2, 0.4), (0.4, 1.0)):
                            for mp in (0.0, 0.15):
                                g.append(FilterCfg(alpha, on, off, min_real, init_trust, on_h, off_h, min_phone=mp))
    return g


def objective(m):
    """frame-level F1 minus a penalty for every unnecessary state change per constant-label person."""
    ex = m.get("extra_flips_per_track", 0.0)
    return m.get("f1", 0.0) - 0.05 * (0.0 if ex != ex else ex)


# ====================================================================== #
# 8. self-test
# ====================================================================== #
def _selftest():
    import importlib.util, sys
    rng = np.random.default_rng(0)

    # --- crop parity with the real build_clips.py (if present next to us / in uploads) ---
    for cand in ("build_clips.py", "/mnt/user-data/uploads/1792352482842_build_clips.py"):
        try:
            spec = importlib.util.spec_from_file_location("bc", cand)
            bc = importlib.util.module_from_spec(spec); sys.modules["bc"] = bc
            spec.loader.exec_module(bc)
            break
        except Exception:
            bc = None
    if bc is not None:
        for _ in range(200):
            img = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
            x1, y1 = rng.uniform(-50, 600), rng.uniform(-50, 330)
            box = (x1, y1, x1 + rng.uniform(5, 300), y1 + rng.uniform(5, 300))
            cfg = CropCfg(size=96, mask=bool(rng.integers(2)), top_frac=float(rng.choice([1.0, 0.65])))
            a = make_crop(img, box, cfg)
            b0 = box if cfg.top_frac >= 1 else (box[0], box[1], box[2], box[1] + (box[3] - box[1]) * cfg.top_frac)
            b = bc.crop_square(img, b0, cfg.margin, cfg.size, cfg.mask, cfg.box_margin)
            assert np.array_equal(a, b), "crop mismatch vs build_clips"
        print("ok  crop == build_clips.crop_square (200 random cases)")
    assert frame_stride(25) == 2 and frame_stride(30) == 3 and frame_stride(12.5) == 1 and frame_stride(20) == 2
    print("ok  stride rule")

    # --- stitcher: id switch in the middle, a second person that must keep its own id ---
    st = Stitcher(25, (640, 360))
    ids = []
    for f in range(120):
        a = (100 + f, 100, 180 + f, 300)
        dets = [(7 if f < 60 else 19, a, 0.9)]            # person A: tracker swaps id 7 -> 19 at f=60
        if f >= 20:
            dets.append((8, (400, 80, 480, 280), 0.9))    # person B enters at f=20 (different place)
        if 60 <= f < 64:
            dets = [d for d in dets if d[0] != 19]        # short loss around the switch
        out = st.update(f, dets)
        ids.append({rid: lid for lid, rid, _, _ in out})
    assert ids[10][7] == ids[100][19], "stitching failed"
    assert ids[30][8] != ids[10][7], "different people merged"
    print("ok  stitcher keeps identity across id switch, keeps people apart")
    st2 = Stitcher(25, (640, 360))             # person leaves at the right border, another enters at left
    st2.update(0, [(1, (560, 100, 639, 300), .9)])
    out = st2.update(10, [(2, (0, 100, 70, 300), .9)])
    assert out[0][0] != 1, "stitched across the frame (exit/enter)"
    print("ok  no stitching after exit at border")

    st3 = Stitcher(25, (640, 360), enabled=False)
    for f in range(5):
        o = st3.update(f, [(4, (10, 10, 50, 120), .9)])
        assert o[0][0] == 4, "disabled stitcher must keep raw ids"
    print("ok  disabled stitcher = identity")

    vg = TrackVerdict(FilterCfg(min_phone=0.15))
    for i in range(12):
        vg.update(i * 0.2, 0.95, 16, 0.0)                     # model says USING but no phone ever seen
    assert vg.state is False, "phone gate failed to block"
    for i in range(12, 24):
        vg.update(i * 0.2, 0.95, 16, 0.6)
    assert vg.state is True, "phone gate failed to release"
    print("ok  phone gate blocks USING without phone evidence, releases with it")

    # --- verdict logic ---
    v = TrackVerdict(FilterCfg())
    states = [v.update(i * 0.2, 0.9, 16)[0] for i in range(6)]
    assert states[:2] == [False, False] and states[2] is True, states   # obvious user confirmed on 3rd tick (~0.4s)
    assert states[-1] is True
    s = [v.update(10 + i * 0.2, 0.05, 16)[0] for i in range(12)]
    assert s[-1] is False and v.flips == 2
    v2 = TrackVerdict(FilterCfg())
    for i in range(40):
        v2.update(i * 0.2, 0.9 if i % 2 == 0 else 0.1, 16)     # alternating noise must not flip
    assert v2.flips == 0, v2.flips
    v3 = TrackVerdict(FilterCfg(min_real=4)); v3.update(0, 0.99, 2)
    assert v3.p is None and v3.state is False
    print("ok  verdict: hysteresis, noise rejection, warm-up guard")

    # --- score_stream with a fake model (checks padding, causality, logical ids) ---
    H, W, N = 120, 160, 140
    frames = [np.full((H, W, 3), 60, np.uint8) for _ in range(N)]
    dets = defaultdict(list)
    for f in range(N):
        dets[f].append((3 if f < 70 else 11, (20 + f * 0.3, 20, 70 + f * 0.3, 110), .9))
    seen = []
    def fake_predict(clip, fpi, has):
        seen.append((clip.shape, None if fpi is None else fpi.shape))
        return 0.8
    boxes, praw = score_stream(frames, dets, 25, (W, H), fake_predict,
                               fpi_fn=lambda c: np.zeros(11, np.float32),
                               crop_cfg=CropCfg(size=32), progress_every=0)
    assert boxes.lid.nunique() == 1, boxes.lid.unique()
    assert all(s_[0] == (16, 32, 32, 3) and s_[1] == (16, 11) for s_ in seen)
    assert praw.n_real.iloc[0] == 1 and praw.n_real.max() == 16
    assert (praw.frame % 2 == 0).all()
    print(f"ok  score_stream: {len(praw)} predictions, n_real {praw.n_real.min()}..{praw.n_real.max()}, 1 logical id")

    # --- evaluation plumbing with a synthetic perfect / noisy predictor ---
    gt = defaultdict(list)
    for f in range(200):
        gt[f].append((0, (10, 10, 60, 120), 1))
        gt[f].append((1, (200, 10, 260, 120), 0))
    det = {f: [(5, (11, 11, 61, 121), 1.0), (6, (201, 9, 259, 119), 1.0)] for f in range(200)}
    boxes = pd.DataFrame([(f, 5 if r == 5 else 6, r, *b, 1.0) for f, ds in det.items() for r, b, _ in ds],
                         columns=["frame", "lid", "rid", "x1", "y1", "x2", "y2", "conf"])
    pr = []
    for f in range(0, 200, 4):
        pr.append((f, 5, float(np.clip(0.85 + rng.normal(0, .08), 0, 1)), 16))
        pr.append((f, 6, float(np.clip(0.10 + rng.normal(0, .08), 0, 1)), 16))
    praw = pd.DataFrame(pr, columns=["frame", "lid", "p_raw", "n_real"])
    m = match_gt_pred(gt, boxes)
    assert len(m) == 400 and set(m.groupby("gt_tid").lid.nunique()) == {1}
    series = run_filter_all(praw, FilterCfg(), 25)
    M = attach_predictions(m, series); M["vid"] = "v"
    fl = {("v", l): series[l][3] for l in series}
    r = summarize(M, {"v": 25}, {("v", 0): 200, ("v", 1): 200}, fl)
    assert r["f1"] > 0.9 and r["extra_flips_per_track"] == 0 and r["idsw_lid_per_gt"] == 0 and r["coverage"] == 1.0, r
    print("ok  evaluation: f1=%.3f acc=%.3f settle_med=%.2fs flips/min=%.2f" %
          (r["f1"], r["acc"], r["settle_med_s"], r["flips_per_min"]))
    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
