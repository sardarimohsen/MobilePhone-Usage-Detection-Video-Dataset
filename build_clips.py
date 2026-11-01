#!/usr/bin/env python3
"""
build_clips.py — Phase 2: crop cache + clip index (uses your frozen split_map.json).

For every video:  read annotations.xml  ->  crop each person box from the video
(square, with context margin)  ->  save jpg  ->  write frame/clip indexes.

Outputs in --out:
    crops/<scene>/<stem>/<track>/<frame:05d>.jpg
    frames.csv   one row per (video, track, frame)
    tracks.csv   one row per track (static_ratio, label mix, ...)
    clips.csv    one row per training/eval window (T frames, stride)
    stats.txt    per-split / per-scene summary (read it before training!)

Example:
    python build_clips.py \
        --dataset-root /home/meybod/Dataset/Dataset \
        --ann-root     /home/meybod/Dataset_annotations \
        --split-map    split_map.json \
        --out          /home/meybod/phone_cache --workers 8
"""
import argparse, json
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

KEEP = {"using_phone": 1, "not_using_phone": 0}


GRAY_BGR = (104, 116, 124)      # = ImageNet mean -> becomes ~0 after normalisation (neutral for the net)


def crop_square(img, box, margin, size, mask=False, box_margin=0.15):
    """Square crop centred on the person box, zero-information (gray) padding outside the frame.

    mask=False : everything inside the square is kept (context + neighbours, old behaviour).
    mask=True  : only the target box (enlarged by box_margin) is kept; the rest of the square is
                 neutral gray -> neighbours and scene background are removed, so the label
                 ("THIS person uses a phone") is unambiguous and scene identity cannot be memorised.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = (x2 - x1) * (1 + (box_margin if mask else 0.0))
    bh = (y2 - y1) * (1 + (box_margin if mask else 0.0))
    s = max(8, int(round(max(bw, bh) * margin)))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    canvas = np.empty((s, s, 3), np.uint8)
    canvas[:] = GRAY_BGR
    # region that is allowed to be copied: square ∩ frame (∩ enlarged box if mask)
    ax0, ay0, ax1, ay1 = x0, y0, x0 + s, y0 + s
    if mask:
        ax0, ay0 = max(ax0, int(round(cx - bw / 2))), max(ay0, int(round(cy - bh / 2)))
        ax1, ay1 = min(ax1, int(round(cx + bw / 2))), min(ay1, int(round(cy + bh / 2)))
    sx0, sy0, sx1, sy1 = max(0, ax0), max(0, ay0), min(W, ax1), min(H, ay1)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interp)


def parse_xml(xml_path):
    r = ET.parse(xml_path).getroot()
    n_frames = int(r.findtext("meta/task/size"))
    per_frame = defaultdict(list)          # frame -> [(track, box, label)]
    dropped = 0
    for t in r.findall("track"):
        tid = int(t.get("id"))
        for b in t.findall("box"):
            if b.get("outside") == "1":
                continue
            a = b.find("attribute[@name='phone_activity']")
            lab = a.text if a is not None else "MISSING"
            if lab not in KEEP:
                dropped += 1
                continue
            box = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            per_frame[int(b.get("frame"))].append((tid, box, KEEP[lab]))
    return n_frames, per_frame, dropped


def process(job):
    cv2.setNumThreads(1)
    scene, stem, split, a = job
    video = Path(a["dataset_root"]) / scene / f"{stem}.mp4"
    xml = Path(a["ann_root"]) / scene / stem / "annotations.xml"
    if not video.is_file() or not xml.is_file():
        return dict(scene=scene, stem=stem, error=f"missing file (video={video.is_file()}, xml={xml.is_file()})")

    n_xml, per_frame, dropped = parse_xml(xml)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows, idx = [], 0
    ok, img = cap.read()
    while ok:
        for tid, box, lab in per_frame.get(idx, []):
            cbox = box if a["top_frac"] >= 1 else (box[0], box[1], box[2], box[1] + (box[3] - box[1]) * a["top_frac"])
            c = crop_square(img, cbox, a["margin"], a["size"], a["mask_outside"], a["box_margin"])
            rel = Path("crops") / scene / stem / str(tid) / f"{idx:05d}.jpg"
            p = Path(a["out"]) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), c, [cv2.IMWRITE_JPEG_QUALITY, 95])
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            rows.append((split, scene, stem, tid, idx, lab, *box, box[3] - box[1],
                         round(float(g.mean()), 2),
                         round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2), str(rel)))
        idx += 1
        ok, img = cap.read()
    cap.release()
    return dict(scene=scene, stem=stem, split=split, rows=rows, n_decoded=idx, n_xml=n_xml,
                fps=fps, W=W, H=H, dropped=dropped)


def make_clips(frames, T, stride, hop_train, hop_eval, min_purity_train, fps_map=None, target_dt=0.0, role=None):
    """If target_dt > 0 the frame stride is chosen per video so that consecutive frames of a
    window are ~target_dt seconds apart (videos here range from 12.5 to 30 FPS).
    role: None -> use the split column ("train" scenes get dense/pure windows, others eval-style);
          "train" / "eval" -> force that style for every track (needed for scene-fold CV)."""
    clips, tracks, short = [], [], 0
    for (split, scene, video, tid), g in frames.groupby(["split", "scene", "video", "track"], sort=False):
        stride_v = stride
        if target_dt > 0 and fps_map is not None:
            stride_v = max(1, int(round(fps_map[(scene, video)] * target_dt)))
        span = (T - 1) * stride_v + 1
        g = g.sort_values("frame")
        fr, lab = g.frame.to_numpy(), g.label.to_numpy()
        # static_ratio: share of consecutive frames with identical box (box never moves)
        bx = g[["x1", "y1", "x2", "y2"]].to_numpy()
        static = float((bx[1:] == bx[:-1]).all(1).mean()) if len(g) > 1 else 1.0
        tracks.append((split, scene, video, tid, len(g), int(lab.sum()), round(static, 3),
                       int(np.median(g.box_h))))
        # contiguous runs (a gap in frames breaks the window)
        cuts = np.where(np.diff(fr) != 1)[0] + 1
        style = role or ("train" if split == "train" else "eval")
        hop = hop_train if style == "train" else hop_eval
        made = False
        for run in np.split(np.arange(len(fr)), cuts):
            for s in range(0, len(run) - span + 1, hop):
                ids = run[s:s + span:stride_v]
                l = lab[ids]
                pos = float(l.mean())
                label, purity = int(pos >= 0.5), max(pos, 1 - pos)
                if style == "train" and purity < min_purity_train:
                    continue                       # drop transition-heavy windows in train only
                clips.append((split, scene, video, tid, int(fr[ids[0]]), stride_v, T, label,
                              round(purity, 3), int(np.median(g.box_h.to_numpy()[ids])),
                              round(float(g.lum.to_numpy()[ids].mean()), 1)))
                made = True
        if not made:
            short += 1
    return (pd.DataFrame(clips, columns=["split", "scene", "video", "track", "start", "stride", "T",
                                          "label", "purity", "box_h", "lum"]),
            pd.DataFrame(tracks, columns=["split", "scene", "video", "track", "n_frames",
                                           "n_pos_frames", "static_ratio", "box_h"]), short)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--ann-root", required=True)
    ap.add_argument("--split-map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256, help="stored crop size (train uses 224)")
    ap.add_argument("--margin", type=float, default=1.25, help="square side = max(w,h)*margin")
    ap.add_argument("--mask-outside", action="store_true",
                    help="gray-out everything outside the target box (removes neighbours/background)")
    ap.add_argument("--top-frac", type=float, default=1.0,
                    help="keep only the top fraction of the box height (e.g. 0.65 = head to hips): "
                         "more pixels on face/hands. box_h in the csv files stays the FULL height.")
    ap.add_argument("--box-margin", type=float, default=0.15,
                    help="with --mask-outside: enlarge the kept box by this fraction")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2, help="fixed frame step (used if --target-dt 0)")
    ap.add_argument("--target-dt", type=float, default=0.1,
                    help="seconds between frames inside a window; stride=round(fps*dt) per video")
    ap.add_argument("--hop-train", type=int, default=4)
    ap.add_argument("--hop-eval", type=int, default=8)
    ap.add_argument("--min-purity-train", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-crops", action="store_true", help="only rebuild clips from frames.csv")
    a = vars(ap.parse_args())
    out = Path(a["out"]); out.mkdir(parents=True, exist_ok=True)

    log = []
    def say(s=""):
        print(s); log.append(s)

    if not a["skip_crops"]:
        smap = json.load(open(a["split_map"], encoding="utf-8"))
        jobs = [(Path(k).parent.name, Path(k).name, v, a) for k, v in smap.items()]
        say(f"videos in split map: {len(jobs)}")
        frames, meta = [], []
        with Pool(a["workers"]) as pool:
            for i, r in enumerate(pool.imap_unordered(process, jobs), 1):
                if "error" in r:
                    say(f"[WARN] {r['scene']}/{r['stem']}: {r['error']}"); continue
                frames += r.pop("rows"); meta.append(r)
                if r["n_decoded"] != r["n_xml"]:
                    say(f"[WARN] frame mismatch {r['scene']}/{r['stem']}: video={r['n_decoded']} xml={r['n_xml']}")
                if i % 10 == 0:
                    print(f"  {i}/{len(jobs)} videos done")
        df = pd.DataFrame(frames, columns=["split", "scene", "video", "track", "frame", "label",
                                           "x1", "y1", "x2", "y2", "box_h", "lum", "blur", "path"])
        df.to_csv(out / "frames.csv", index=False)
        pd.DataFrame(meta).to_csv(out / "videos_meta.csv", index=False)
        say(f"fps values seen: {sorted(set(round(m['fps'], 2) for m in meta))}")
        say(f"frames per video: min={min(m['n_decoded'] for m in meta)} "
            f"max={max(m['n_decoded'] for m in meta)}")
        say(f"dropped (uncertain/missing label) boxes: {sum(m['dropped'] for m in meta)}")
    else:
        df = pd.read_csv(out / "frames.csv")

    meta_df = pd.read_csv(out / "videos_meta.csv")
    fps_map = {(r.scene, r.stem): r.fps for r in meta_df.itertuples()}
    clips, tracks, short = make_clips(df, a["T"], a["stride"], a["hop_train"], a["hop_eval"],
                                      a["min_purity_train"], fps_map, a["target_dt"])
    clips.to_csv(out / "clips.csv", index=False)
    for r_ in ("train", "eval"):          # role-specific sets for scene-fold CV (train.py --folds)
        c_, _, _ = make_clips(df, a["T"], a["stride"], a["hop_train"], a["hop_eval"],
                              a["min_purity_train"], fps_map, a["target_dt"], role=r_)
        c_.to_csv(out / f"clips_{r_}.csv", index=False)
    tracks.to_csv(out / "tracks.csv", index=False)

    say("\n" + "=" * 70 + "\nSPLIT SUMMARY (check prevalence before training!)\n" + "=" * 70)
    for sp in ("train", "val", "test"):
        f, t, c = df[df.split == sp], tracks[tracks.split == sp], clips[clips.split == sp]
        say(f"{sp:5s} videos={f.video.nunique():3d} tracks={len(t):5d} "
            f"(tracks with any using_phone={int((t.n_pos_frames > 0).sum())}) "
            f"frames={len(f):7d} pos_frames={f.label.mean():.3f} "
            f"clips={len(c):6d} pos_clips={c.label.mean() if len(c) else float('nan'):.3f}")
    say(f"tracks too short for a window (T={a['T']}, target_dt={a['target_dt']}s): {short}")
    say("stride used per video: " + str(clips.groupby("stride").video.nunique().to_dict()))
    say("\nPer-scene:")
    sc = tracks.groupby(["split", "scene"]).agg(
        videos=("video", "nunique"), tracks=("track", "size"),
        pos_tracks=("n_pos_frames", lambda s: int((s > 0).sum())),
        static_tracks=("static_ratio", lambda s: int((s > 0.95).sum())),
        med_box_h=("box_h", "median"))
    say(sc.to_string())
    say("\n(static_tracks = tracks whose box never moves. High numbers in scenes where "
        "people walk (corridors, parking) would mean boxes did not follow the people.)")
    (out / "stats.txt").write_text("\n".join(log), encoding="utf-8")


if __name__ == "__main__":
    main()
