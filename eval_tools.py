"""eval_tools.py — one evaluation protocol for every run (no torch needed).

Unit of evaluation = 1-second bins inside a track (mean of window probs),
optionally causally smoothed with a time-aware EMA. Uncertainty = bootstrap
over TRACKS (windows/bins of one track are not independent).
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


# --------------------------------------------------------------------------- #
def ap(y, p):
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def f1_at(y, p, thr):
    y = np.asarray(y).astype(bool); pr = np.asarray(p) >= thr
    tp = (y & pr).sum(); d = pr.sum() + y.sum()
    return float(2 * tp / d) if d else float("nan")


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / (pr + rc + 1e-12)
    i = int(np.nanargmax(f[:-1]))
    return float(th[i]), float(f[i])


# --------------------------------------------------------------------------- #
def load_preds(path, fps_fallback_from_stride=True) -> pd.DataFrame:
    """Normalise preds_*.csv of ANY run family to: key, scene, video, track,
    wid, t (sec, window centre), label, prob [, box_h]."""
    d = pd.read_csv(path)
    if "first_frame" in d and "fps" in d:                       # STP / v3 format
        d["t"] = (d.first_frame + d.last_frame) / 2.0 / d.fps
        d["wid"] = d.first_frame.astype(int)
    elif "start" in d and "stride" in d:                        # old LoRA/CNN format
        T = d["T"] if "T" in d else 16
        c = d.start + (T - 1) / 2.0 * d.stride
        if fps_fallback_from_stride:
            warnings.warn("old-format preds: fps unknown, t ~ frame/(10*stride) (approximate)")
        d["t"] = c / (d.stride * 10.0)
        d["wid"] = d.start.astype(int)
    else:
        raise ValueError(f"unknown preds format: {list(d.columns)}")
    d["key"] = d.scene.astype(str) + "|" + d.video.astype(str) + "|" + d.track.astype(str)
    keep = ["key", "scene", "video", "track", "wid", "t", "label", "prob"]
    if "box_h" in d:
        keep.append("box_h")
    return d[keep].copy()


def to_bins(d: pd.DataFrame, w: float = 1.0) -> pd.DataFrame:
    d = d.copy(); d["bin"] = np.floor(d.t / w).astype(int)
    agg = dict(prob=("prob", "mean"), label=("label", "mean"), scene=("scene", "first"))
    if "box_h" in d:
        agg["box_h"] = ("box_h", "median")
    g = d.groupby(["key", "bin"], sort=True).agg(**agg).reset_index()
    g["label"] = (g.label >= 0.5).astype(int)
    g["t"] = (g.bin + 0.5) * w
    return g


def ema_time(d: pd.DataFrame, alpha_per_sec: float) -> pd.DataFrame:
    """Causal EMA per track; the decay is time-aware: a = alpha**dt, so the same
    alpha means the same thing for hop=1 windows, 1-s bins or old hop-8 windows."""
    if alpha_per_sec <= 0 or alpha_per_sec >= 1.0:
        return d.copy()
    d = d.sort_values(["key", "t"]).copy()
    out = np.empty(len(d)); k = 0
    for _, g in d.groupby("key", sort=False):
        t = g.t.values; p = g.prob.values; e = p[0]; out[k] = e
        for i in range(1, len(g)):
            a = alpha_per_sec ** max(t[i] - t[i - 1], 1e-3)
            e = a * e + (1 - a) * p[i]; out[k + i] = e
        k += len(g)
    d["prob"] = out
    return d


def smoothed_bin_ap(d: pd.DataFrame, alpha: float = 0.7) -> float:
    """Model-selection score used during training (val)."""
    return ap(*(lambda b: (b.label, b.prob))(ema_time(to_bins(d), alpha)))


def track_bootstrap(d: pd.DataFrame, fn, n=500, seed=0, label="label", prob="prob"):
    """fn(y,p)->float. Returns (point, lo, hi) with tracks resampled."""
    rng = np.random.default_rng(seed)
    groups = [(g[label].values, g[prob].values) for _, g in d.groupby("key")]
    point = fn(d[label].values, d[prob].values)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        y = np.concatenate([groups[i][0] for i in idx]); p = np.concatenate([groups[i][1] for i in idx])
        v = fn(y, p)
        if not np.isnan(v):
            vals.append(v)
            
    if not vals:
        return point, float("nan"), float("nan")
        
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_track_bootstrap(a: pd.DataFrame, b: pd.DataFrame, fn=ap, n=500, seed=0):
    """a,b: frames with key,bin,label,prob on identical (key,bin). Returns delta b-a."""
    m = a.merge(b, on=["key", "bin"], suffixes=("_a", "_b"))
    rng = np.random.default_rng(seed)
    groups = [(g.label_a.values, g.prob_a.values, g.prob_b.values) for _, g in m.groupby("key")]
    point = fn(m.label_a, m.prob_b) - fn(m.label_a, m.prob_a)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        y = np.concatenate([groups[i][0] for i in idx])
        pa = np.concatenate([groups[i][1] for i in idx]); pb = np.concatenate([groups[i][2] for i in idx])
        v = fn(y, pb) - fn(y, pa)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float(point), float("nan"), float("nan"), len(m)
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), len(m)
