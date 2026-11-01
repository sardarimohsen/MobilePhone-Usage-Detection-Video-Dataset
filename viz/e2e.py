"""viz/e2e.py — deployed pipeline (detect+track -> crop/FPI/model -> causal verdict -> overlay) and its evaluation.

Stages are separate and cached so that experiments are replayable and change ONE thing at a time:

  track   video  -> <base>.tracks.json   (person detector + tracker, run once)
  score   tracks -> <base>.boxes.csv / <base>.praw.csv   (stitching, crops, FPI, model: raw probability series)
  render  boxes+praw -> overlay mp4      (EMA + hysteresis; cheap, re-run with any filter setting)

Commands
  run     all three stages on one video
  render  only stage 3 from cache
  eval    labelled videos: grid-search the filter on VAL, report on TEST (det mode and/or oracle-GT-box mode)
  parity  verify crops / FPI features of this code against the training cache

Examples
  python -m viz.e2e parity --n 40 --fpi-model /home/meybod/vault-open/runs_yolo/fpi_yolo26x_832/weights/best.pt
  python -m viz.e2e run  --video my.mp4 --out presentation/video/my_e2e.mp4 \
        --person-model /home/meybod/vault-open/yolo26x.pt \
        --fpi-model /home/meybod/vault-open/runs_yolo/fpi_yolo26x_832/weights/best.pt
  python -m viz.e2e eval --run C_s384_mid --fold 0 --seed 0 --modes gt det --out presentation/e2e_eval \
        --person-model yolo11n.pt --fpi-model <fpi best.pt>
"""
import argparse
import json
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from . import e2e_core as C
from .config import RUNS, VAULT

DATASET_ROOT = VAULT / "Dataset/Dataset"
ANN_ROOT = VAULT / "Dataset_annotations"
CACHE_V3 = VAULT / "phone_cache_v3"

RED, GREEN, YELLOW = (60, 60, 220), (60, 220, 60), (40, 220, 240)      # BGR


# ====================================================================== #
# helpers
# ====================================================================== #
def _dev():
    import torch
    return (0, torch.device("cuda")) if torch.cuda.is_available() else ("cpu", torch.device("cpu"))


def parse_ckpts(s):
    return [tuple(int(x) for x in t.split(":")) for t in s.split(",")]


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        print(f"[warn] bad fps for {path}; assuming 25")
        fps = 25.0
    info = dict(fps=float(fps), W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def frame_iter(path, max_frames=0):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, img = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        yield img
        n += 1
    cap.release()


def load_tracks(path):
    d = json.load(open(path, encoding="utf-8"))
    by = {}
    for f, tid, x1, y1, x2, y2, cf in d["dets"]:
        by.setdefault(int(f), []).append((int(tid), (x1, y1, x2, y2), cf))
    return by, d


# ====================================================================== #
# stage 1: detection + tracking
# ====================================================================== #
def build_tracker_yaml(kind, fps, a, path):
    from ultralytics.utils import ROOT
    cfg = yaml.safe_load((ROOT / "cfg" / "trackers" / f"{kind}.yaml").read_text(encoding="utf-8"))
    cfg.update(track_high_thresh=a.track_high, track_low_thresh=a.track_low,
               new_track_thresh=a.new_track, track_buffer=max(1, int(round(fps * a.buffer_s))),
               match_thresh=a.match_thresh)
    if kind == "botsort":
        cfg.update(gmc_method=a.gmc, with_reid=bool(a.reid))
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def track_video(video, out_json, a):
    from ultralytics import YOLO
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    info = video_info(video)
    yml = out_json.with_suffix(".tracker.yaml")
    tcfg = build_tracker_yaml(a.tracker, info["fps"], a, yml)
    model = YOLO(a.person_model)            # fresh object per video: tracker state must not leak between videos
    dev, _ = _dev()
    dets, n = [], 0
    t0 = time.time()
    stream = model.track(source=str(video), stream=True, persist=True, tracker=str(yml), classes=[0],
                         conf=a.det_conf, iou=a.iou, imgsz=a.imgsz, device=dev, max_det=300, verbose=False)
    for f, res in enumerate(stream):
        if a.max_frames and f >= a.max_frames:
            break
        b = res.boxes
        if b is not None and len(b) > 0 and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy()
            for box, tid, c in zip(xyxy, ids, cf):
                dets.append([f, int(tid), *[round(float(v), 2) for v in box], round(float(c), 3)])
        n = f + 1
        if f % 300 == 0 and f:
            print(f"  [track] frame {f}  dets={len(dets)}  {f / (time.time() - t0):.1f} fps")
    json.dump(dict(video=str(video), fps=info["fps"], W=info["W"], H=info["H"], n_frames=n,
                   tracker=tcfg, person_model=str(a.person_model), det_conf=a.det_conf, imgsz=a.imgsz,
                   dets=dets), open(out_json, "w"))
    if info["n"] and abs(info["n"] - n) > 2 and not a.max_frames:
        print(f"[warn] tracker saw {n} frames, container says {info['n']}")
    print(f"[track] {Path(video).name}: {n} frames, {len(dets)} boxes -> {out_json.name}")


def gt_tracks_json(video, xml, out_json):
    """Oracle: ground-truth boxes written in the same format as stage 1."""
    info = video_info(video)
    n_xml, pf = C.parse_xml(xml)
    dets = [[f, tid, *[round(v, 2) for v in box], 1.0] for f, rows in sorted(pf.items()) for tid, box, _ in rows]
    json.dump(dict(video=str(video), fps=info["fps"], W=info["W"], H=info["H"], n_frames=info["n"],
                   oracle=True, dets=dets), open(out_json, "w"))


# ====================================================================== #
# stage 2: scoring
# ====================================================================== #
def load_models(a):
    from .storyboard import _load_model
    _, dev = _dev()
    Ms = [_load_model(a.run, fold=f, seed=s, device=dev) for f, s in parse_ckpts(a.ckpts)]
    T = Ms[0]["T"]
    assert all(m["T"] == T for m in Ms), "ensemble members must share T"
    print(f"[models] {a.run} {a.ckpts}  T={T} size={Ms[0]['size']} use_fpi={[m['use_fpi'] for m in Ms]}")
    return Ms, dev


def load_fpi_fn(a, Ms, crop_size):
    if not any(m["use_fpi"] for m in Ms):
        return None
    if not a.fpi_model:
        raise SystemExit("this checkpoint was trained WITH FPI features: pass --fpi-model "
                         "(zero features silently degrade accuracy)")
    from ultralytics import YOLO
    fpi = YOLO(a.fpi_model)
    dev, _ = _dev()

    def fn(crop_bgr):                       # ultralytics expects BGR for numpy input -> pass BGR
        r = fpi.predict(crop_bgr, imgsz=832, conf=0.15, verbose=False, device=dev, max_det=50)
        if r and len(r[0].boxes) > 0:
            b = r[0].boxes
            return C.fpi_features(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                                  b.conf.cpu().numpy(), crop_size, crop_size)
        return np.zeros(11, np.float32)
    return fn


def make_predict(Ms):
    import torch
    from .storyboard import _forward

    def predict(clip, fpi, has):
        ps = []
        for M in Ms:
            c = clip
            if M["size"] != clip.shape[1]:
                c = np.stack([cv2.resize(x, (M["size"], M["size"]), interpolation=cv2.INTER_AREA) for x in clip])
            ft = ht = None
            if M["use_fpi"]:
                ft, ht = torch.from_numpy(fpi).float(), torch.from_numpy(has).float()
            probs, _ = _forward(M, c, fpi_feats=ft, has_fpi=ht)
            ps.append(float(probs[1]))
        return float(np.mean(ps))
    return predict


def score_video(video, tracks_json, base, a, models=None):
    by, meta = load_tracks(tracks_json)
    Ms, _ = models or load_models(a)
    ccfg = C.CropCfg(margin=a.margin, size=a.crop_size, mask=True, box_margin=a.box_margin, top_frac=a.top_frac)
    fpi_fn = load_fpi_fn(a, Ms, ccfg.size) if not hasattr(a, "_fpi_cache") else a._fpi_cache
    a._fpi_cache = fpi_fn
    oracle = bool(meta.get("oracle"))
    stitch = (not a.no_stitch) and not oracle
    n = meta["n_frames"] if not a.max_frames else min(a.max_frames, meta["n_frames"])
    frames = islice(frame_iter(video, a.max_frames), n)
    t0 = time.time()
    boxes, praw = C.score_stream(frames, by, meta["fps"], (meta["W"], meta["H"]), make_predict(Ms), fpi_fn,
                                 crop_cfg=ccfg, T=Ms[0]["T"], target_dt=a.target_dt, infer_ticks=a.infer_ticks,
                                 early_ticks=a.early_ticks, stitch=stitch,
                                 stitch_kw=dict(max_gap_s=a.stitch_gap_s))
    boxes.to_csv(f"{base}.boxes.csv", index=False)
    praw.to_csv(f"{base}.praw.csv", index=False)
    print(f"[score] {Path(video).name}: {len(boxes)} boxes, {boxes.lid.nunique()} logical ids "
          f"({boxes.rid.nunique()} raw ids), {len(praw)} predictions, {time.time() - t0:.0f}s")


# ====================================================================== #
# stage 3: render
# ====================================================================== #
def draw_box(img, box, lid, p, state, warm=False):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    color = YELLOW if (warm and p is None) else (RED if state else GREEN)
    label = f"ID {lid}  " + ("USING PHONE" if state else "NOT USING")
    sub = "p=--" if p is None else f"p={p:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ly = y1 - th - 12 if y1 - th - 12 >= 0 else y1 + 2
    cv2.rectangle(img, (x1, ly), (x1 + tw + 12, ly + th + 12), color, -1)
    cv2.putText(img, label, (x1 + 6, ly + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y2), (x1 + sw + 12, y2 + sh + 10), color, -1)
    cv2.putText(img, sub, (x1 + 6, y2 + sh + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def render_video(video, boxes, praw, out_path, fcfg, box_alpha=0.5, neutral_warmup=False):
    info = video_info(video)
    fps = info["fps"]
    series = C.run_filter_all(praw, fcfg, fps)
    by = {f: g for f, g in boxes.groupby("frame")}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (info["W"], info["H"]))
    smooth, rows, f = {}, [], 0
    for img in frame_iter(video):
        g = by.get(f)
        if g is not None:
            for lid, rid, x1, y1, x2, y2 in zip(g.lid, g.rid, g.x1, g.y1, g.x2, g.y2):
                p, s = C.lookup(series, int(lid), np.array([f]), fcfg.init_state)
                p, s = (None if np.isnan(p[0]) else float(p[0])), bool(s[0])
                raw = np.array([x1, y1, x2, y2], float)
                prev = smooth.get(int(lid))                      # display-only smoothing (crops use raw boxes)
                sm = raw if (prev is None or f - prev[0] > 5) else box_alpha * prev[1] + (1 - box_alpha) * raw
                smooth[int(lid)] = (f, sm)
                draw_box(img, sm, int(lid), p, s, warm=neutral_warmup)
                rows.append((f, int(lid), int(rid), p, int(s)))
        cv2.putText(img, f"frame {f}  |  tracks: {0 if g is None else len(g)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        wr.write(img)
        f += 1
    wr.release()
    pd.DataFrame(rows, columns=["frame", "lid", "rid", "p_ema", "using"]).to_csv(
        out_path.with_suffix(".verdicts.csv"), index=False)
    print(f"[render] wrote {out_path}  ({f} frames)")


# ====================================================================== #
# eval
# ====================================================================== #
def run_list(run_key, fold, seed, split):
    cfg = next((r for r in RUNS if r["key"] == run_key), None)
    if cfg is None:
        raise KeyError(run_key)
    for pat in [f"{cfg['tag']}_f{fold}_s{seed}", f"{cfg['tag']}_f{fold}"]:
        p = Path(cfg["root"]) / pat / f"preds_{split}.csv"
        if p.exists():
            df = pd.read_csv(p, usecols=["scene", "video"]).drop_duplicates()
            return [(str(r.scene), str(r.video)) for r in df.itertuples()]
    raise FileNotFoundError(f"preds_{split}.csv for {run_key} f{fold} s{seed}")


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def prepare_videos(vids, mode, split, a, models):
    """Run stages 1+2 for every video (cached). Returns list of dicts with the loaded data."""
    data = []
    for k, (scene, video) in enumerate(vids, 1):
        vpath = DATASET_ROOT / scene / f"{video}.mp4"
        xml = ANN_ROOT / scene / video / "annotations.xml"
        if not vpath.is_file() or not xml.is_file():
            print(f"[skip] {scene}/{video}: missing video or annotations")
            continue
        base = Path(a.out) / mode / split / safe(scene) / video
        base.parent.mkdir(parents=True, exist_ok=True)
        tj = Path(f"{base}.tracks.json")
        print(f"\n=== [{mode}/{split}] {k}/{len(vids)}  {scene} / {video}")
        if a.force or not tj.exists():
            gt_tracks_json(vpath, xml, tj) if mode == "gt" else track_video(vpath, tj, a)
        if a.force or a.rescore or not Path(f"{base}.praw.csv").exists():
            score_video(vpath, tj, str(base), a, models)
        meta = json.load(open(tj, encoding="utf-8"))
        n_xml, gt = C.parse_xml(xml)
        gt_len = {}
        for rows in gt.values():
            for tid, _, _ in rows:
                gt_len[tid] = gt_len.get(tid, 0) + 1
        boxes = pd.read_csv(f"{base}.boxes.csv")
        praw = pd.read_csv(f"{base}.praw.csv")
        data.append(dict(vid=video, scene=scene, fps=meta["fps"], boxes=boxes, praw=praw, gt=gt, gt_len=gt_len,
                         matches=C.match_gt_pred(gt, boxes) if len(boxes) else
                         pd.DataFrame(columns=["frame", "gt_tid", "gt_label", "lid", "rid", "iou"])))
    return data


def evaluate_cfg(data, fcfg, by_scene=False):
    rows, flips, fps_map, gt_len = [], {}, {}, {}
    for d in data:
        series = C.run_filter_all(d["praw"], fcfg, d["fps"]) if len(d["praw"]) else {}
        M = C.attach_predictions(d["matches"], series, fcfg.init_state)
        M["vid"], M["scene"] = d["vid"], d["scene"]
        rows.append(M)
        fps_map[d["vid"]] = d["fps"]
        for l, s in series.items():
            flips[(d["vid"], l)] = s[3]
        for g, n in d["gt_len"].items():
            gt_len[(d["vid"], g)] = n
    M = pd.concat(rows, ignore_index=True)
    res = C.summarize(M, fps_map, gt_len, flips)
    if not by_scene:
        return res
    per = {}
    for sc, g in M.groupby("scene"):
        per[sc] = C.summarize(g.reset_index(drop=True), fps_map, gt_len, flips)
    return res, per


COLS = ["f1", "precision", "recall", "acc", "auprc", "auroc", "early_acc_1s", "late_acc", "flips_per_min",
        "extra_flips_per_track", "settle_med_s", "settle_p90_s", "ever_wrong_frac",
        "idsw_raw_per_gt", "idsw_lid_per_gt", "merges_per_lid", "coverage", "n_rows", "n_gt_tracks"]


def fmt(df):
    return df[[c for c in COLS if c in df.columns]].round(3).to_string()


OLD_FILTER = C.FilterCfg(alpha=0.3, on=0.5, off=0.5, min_real=16, init_trust=1.0,
                         on_hold_s=0.0, off_hold_s=0.0, init_state=True)   # emulation of the old robust-script verdict


def cmd_eval(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = load_models(a)
    summary = {}
    for mode in a.modes:
        print(f"\n######## MODE: {mode} ########")
        val = prepare_videos(run_list(a.run, a.fold, a.seed, "val")[::a.every], mode, "val", a, models) \
            if not a.no_grid else []
        test = prepare_videos(run_list(a.run, a.fold, a.seed, "test")[::a.every], mode, "test", a, models)
        best = C.FilterCfg()
        if val:
            print("\n--- grid search on VAL ---")
            res, cfgs = [], C.default_grid()
            if not all("phone_frac" in d["praw"].columns for d in val):
                print("[note] cached praw has no phone_frac -> phone gate not searched (use --rescore to add it)")
                cfgs = [c for c in cfgs if c.min_phone == 0]
            for fc in cfgs:
                m = evaluate_cfg(val, fc)
                res.append({**asdict(fc), **m, "obj": C.objective(m)})
            g = pd.DataFrame(res).sort_values("obj", ascending=False)
            g.to_csv(out / f"grid_val_{mode}.csv", index=False)
            top = g.head(10)
            print(top[["alpha", "on", "off", "min_real", "init_trust", "on_hold_s", "off_hold_s", "f1",
                       "extra_flips_per_track", "settle_med_s", "obj"]].round(3).to_string(index=False))
            best = cfgs[int(np.argmax([r["obj"] for r in res]))]      # typed FilterCfg, no float casts
            json.dump(asdict(best), open(out / f"best_filter_{mode}.json", "w"), indent=1)
        print("\n--- TEST ---")
        table = {}
        for name, fc in (("best_on_val", best), ("defaults", C.FilterCfg()), ("old_filter_emulation", OLD_FILTER)):
            table[name] = evaluate_cfg(test, fc)
        t = pd.DataFrame(table).T
        print(fmt(t))
        t.to_csv(out / f"test_{mode}.csv")
        r, per = evaluate_cfg(test, best, by_scene=True)
        ps = pd.DataFrame(per).T
        print("\nper scene (best_on_val):")
        print(fmt(ps))
        ps.to_csv(out / f"test_by_scene_{mode}.csv")
        summary[mode] = dict(best=asdict(best), test=table)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    print(f"\n[done] reports in {out}")


# ====================================================================== #
# parity (crops + FPI vs training cache)
# ====================================================================== #
def robust_old_crop(img, box, margin, size, box_margin=0.15, **_):
    """crop function of video_demo_robust.py (kept only to show that it differs from training)."""
    GRAY = C.GRAY_BGR
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    s = max(8, int(round(max(x2 - x1, y2 - y1) * margin)))
    x0 = int(round((x1 + x2) / 2 - s / 2)); y0 = int(round((y1 + y2) / 2 - s / 2))
    canvas = np.empty((s, s, 3), np.uint8); canvas[:] = GRAY
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    bx1 = max(0.0, x1 - box_margin * (x2 - x1)); by1 = max(0.0, y1 - box_margin * (y2 - y1))
    bx2 = min(float(W), x2 + box_margin * (x2 - x1)); by2 = min(float(H), y2 + box_margin * (y2 - y1))
    cx1, cy1 = int(round(bx1 - x0)), int(round(by1 - y0)); cx2, cy2 = int(round(bx2 - x0)), int(round(by2 - y0))
    cy1c, cy2c, cx1c, cx2c = max(0, cy1), max(0, cy2), max(0, cx1), max(0, cx2)
    canvas[:cy1c, :] = GRAY; canvas[cy2c:, :] = GRAY; canvas[:, :cx1c] = GRAY; canvas[:, cx2c:] = GRAY
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


def cmd_parity(a):
    cache = Path(a.cache)
    fr = pd.read_csv(cache / "frames.csv")
    smp = fr.sample(n=min(a.n, len(fr)), random_state=0).sort_values(["scene", "video", "frame"])
    cands = {f"mask={m} top_frac={t}": C.CropCfg(margin=1.25, size=a.size, mask=m, top_frac=t)
             for m in (True, False) for t in (1.0, 0.65)}
    diffs = {k: [] for k in cands}
    diffs["robust_script_crop (old)"] = []
    pairs = []
    for (scene, video), g in smp.groupby(["scene", "video"]):
        want = {int(r.frame): r for r in g.itertuples()}
        for idx, img in enumerate(frame_iter(DATASET_ROOT / scene / f"{video}.mp4")):
            if idx > max(want):
                break
            r = want.get(idx)
            if r is None:
                continue
            ref = cv2.imread(str(cache / r.path))
            if ref is None:
                continue
            box = (r.x1, r.y1, r.x2, r.y2)
            for k, cfg in cands.items():
                c = C.make_crop(img, box, C.CropCfg(**{**asdict(cfg), "size": ref.shape[0]}))
                diffs[k].append(float(np.abs(c.astype(np.int16) - ref.astype(np.int16)).mean()))
            c = robust_old_crop(img, box, 1.25, ref.shape[0])
            diffs["robust_script_crop (old)"].append(float(np.abs(c.astype(np.int16) - ref.astype(np.int16)).mean()))
            pairs.append((r, ref))
    print("\nCROP PARITY  (mean |pixel diff| vs cached training crop; JPEG noise alone is ~1-3)")
    for k, v in sorted(diffs.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else 1e9):
        print(f"  {k:34s} {np.mean(v):7.2f}   (n={len(v)})")
    if a.fpi_model and pairs:
        from ultralytics import YOLO
        fpi = YOLO(a.fpi_model)
        dev, _ = _dev()
        rows = []
        for r, ref in pairs[:a.n]:
            f = cache / "fpi_feats" / str(r.scene) / str(r.video) / str(r.track) / f"{int(r.frame):05d}.npy"
            if not f.exists():
                continue
            cached = np.load(f)
            out = {}
            for name, im in (("BGR (correct)", ref), ("RGB-swapped", ref[..., ::-1].copy())):
                res = fpi.predict(im, imgsz=832, conf=0.15, verbose=False, device=dev, max_det=50)
                if res and len(res[0].boxes) > 0:
                    b = res[0].boxes
                    v = C.fpi_features(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int),
                                       b.conf.cpu().numpy(), ref.shape[1], ref.shape[0])
                else:
                    v = np.zeros(11, np.float32)
                out[name] = (float(np.abs(v - cached).mean()), bool((v[8] > 0) == (cached[8] > 0)))
            rows.append(out)
        print("\nFPI PARITY vs cached fpi_feats  (mean |feature diff|, phone-presence agreement)")
        for name in ("BGR (correct)", "RGB-swapped"):
            print(f"  {name:16s} diff={np.mean([x[name][0] for x in rows]):.4f}  "
                  f"phone-flag agreement={np.mean([x[name][1] for x in rows]):.3f}   (n={len(rows)})")


# ====================================================================== #
# CLI
# ====================================================================== #
def add_common(ap):
    g = ap.add_argument_group("model")
    g.add_argument("--run", default="C_s384_mid")
    g.add_argument("--ckpts", default="0:0", help="fold:seed list, e.g. 0:0,0:1,0:2 (probabilities are averaged)")
    g.add_argument("--fpi-model", default=None)
    g = ap.add_argument_group("crop / window (must match training)")
    g.add_argument("--crop-size", type=int, default=384)
    g.add_argument("--margin", type=float, default=1.25)
    g.add_argument("--box-margin", type=float, default=0.15)
    g.add_argument("--top-frac", type=float, default=1.0)
    g.add_argument("--target-dt", type=float, default=0.1)
    g.add_argument("--infer-ticks", type=int, default=2, help="predict every N ticks (tick = target-dt)")
    g.add_argument("--early-ticks", type=int, default=4, help="predict on every tick for the first N ticks of a track")
    g.add_argument("--no-stitch", action="store_true")
    g.add_argument("--stitch-gap-s", type=float, default=10.0)
    g = ap.add_argument_group("detector / tracker")
    g.add_argument("--person-model", default="yolo11n.pt")
    g.add_argument("--imgsz", type=int, default=1280)
    g.add_argument("--det-conf", type=float, default=0.10, help="MUST be <= track-low, or BYTE's 2nd association is dead")
    g.add_argument("--iou", type=float, default=0.7)
    g.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort")
    g.add_argument("--track-high", type=float, default=0.40)
    g.add_argument("--track-low", type=float, default=0.10)
    g.add_argument("--new-track", type=float, default=0.60)
    g.add_argument("--match-thresh", type=float, default=0.80)
    g.add_argument("--buffer-s", type=float, default=10.0, help="track_buffer in seconds (converted to frames)")
    g.add_argument("--gmc", default="none", help="botsort camera-motion compensation (none for fixed cameras)")
    g.add_argument("--reid", action="store_true")
    g.add_argument("--max-frames", type=int, default=0)
    g.add_argument("--force", action="store_true", help="recompute cached stages")
    g.add_argument("--rescore", action="store_true", help="recompute only stage 2 (keeps tracks.json)")


def add_filter(ap):
    g = ap.add_argument_group("verdict filter")
    d = C.FilterCfg()
    g.add_argument("--alpha", type=float, default=d.alpha)
    g.add_argument("--on", type=float, default=d.on)
    g.add_argument("--off", type=float, default=d.off)
    g.add_argument("--min-real", type=int, default=d.min_real)
    g.add_argument("--init-trust", type=float, default=d.init_trust)
    g.add_argument("--on-hold", type=float, default=d.on_hold_s)
    g.add_argument("--off-hold", type=float, default=d.off_hold_s)
    g.add_argument("--min-phone", type=float, default=0.0, help="phone-evidence gate for USING (0 = off)")
    g.add_argument("--filter-json", default=None, help="e.g. e2e_eval/best_filter_det.json (overrides the above)")
    g.add_argument("--box-alpha", type=float, default=0.5, help="display-only box smoothing")
    g.add_argument("--neutral-warmup", action="store_true", help="yellow box until the first trusted prediction")


def filter_from_args(a):
    if a.filter_json:
        return C.FilterCfg(**json.load(open(a.filter_json)))
    return C.FilterCfg(alpha=a.alpha, on=a.on, off=a.off, min_real=a.min_real, init_trust=a.init_trust,
                       on_hold_s=a.on_hold, off_hold_s=a.off_hold, min_phone=a.min_phone)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="track + score + render one video")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="output mp4 (cache files are written next to it)")
    add_common(p); add_filter(p)

    p = sub.add_parser("render", help="re-render from cached stages with another filter setting")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    add_filter(p)

    p = sub.add_parser("eval", help="evaluate on labelled videos")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="presentation/e2e_eval")
    p.add_argument("--modes", nargs="+", default=["det"], choices=["det", "gt"],
                   help="det = detector+tracker (deployment), gt = oracle boxes (upper bound)")
    p.add_argument("--every", type=int, default=1, help="use every N-th video (quick runs)")
    p.add_argument("--no-grid", action="store_true")
    add_common(p)

    p = sub.add_parser("parity", help="compare crops/FPI with the training cache")
    p.add_argument("--cache", default=str(CACHE_V3))
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--fpi-model", default=None)

    a = ap.parse_args()
    if a.cmd == "parity":
        cmd_parity(a)
    elif a.cmd == "eval":
        cmd_eval(a)
    elif a.cmd == "render":
        base = Path(a.out).with_suffix("")
        render_video(a.video, pd.read_csv(f"{base}.boxes.csv"), pd.read_csv(f"{base}.praw.csv"), a.out,
                     filter_from_args(a), a.box_alpha, a.neutral_warmup)
    elif a.cmd == "run":
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        base = out.with_suffix("")
        tj = Path(f"{base}.tracks.json")
        if a.force or not tj.exists():
            track_video(a.video, tj, a)
        if a.force or a.rescore or not Path(f"{base}.praw.csv").exists():
            score_video(a.video, tj, str(base), a)
        render_video(a.video, pd.read_csv(f"{base}.boxes.csv"), pd.read_csv(f"{base}.praw.csv"), out,
                     filter_from_args(a), a.box_alpha, a.neutral_warmup)


if __name__ == "__main__":
    main()
