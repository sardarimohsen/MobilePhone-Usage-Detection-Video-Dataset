#!/usr/bin/env python3
"""train_v3.py -- training for phone-usage detection (v3.3).

Supports ablation via:
  --head {simple, stt, cnn, mm_simple}
  --no-motion           (disable motion stream in STTHead)
  --use-motion          (enable motion stream in SimpleHead)
  --use-fpi-feats       (load FPI features from cache/fpi_feats/ and pass to head)
  --backbone <timm name>  (auto-detects CNN vs ViT)
"""
from __future__ import annotations
import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_STP = _HERE.parent / "STP_phone_usage_v1"
_MAIN = Path(os.environ.get("STP_MAIN", _HERE.parent / "yolo_cvat_tracking"))
for _p in (_MAIN, _STP, _HERE):
    if _p.exists():
        sp = str(_p)
        if sp in sys.path:
            sys.path.remove(sp)
        sys.path.insert(0, sp)

import argparse, contextlib, json, math, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_stp import make_stp_datasets, make_stp_sampler
from eval_tools import ap, auc, f1_at, smoothed_bin_ap, to_bins, ema_time
from models_v3 import (IMAGENET_MEAN, IMAGENET_STD, SimpleHead, STTHead, CNNHead,
                       extract_tokens, extract_global_feature,
                       is_cnn_backbone, make_backbone, build_head)
import timm
from peft import LoraConfig, inject_adapter_in_model


# =========================================================================== #
# Losses
# =========================================================================== #
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.alpha, self.gamma, self.ls = float(alpha), float(gamma), float(label_smoothing)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.ls)
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(ce, self.alpha),
                              torch.full_like(ce, 1.0 - self.alpha))
        return (alpha_t * (1.0 - pt).pow(self.gamma) * ce).mean()


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos: float = 1.0, gamma_neg: float = 3.0,
                 alpha: float = 0.71, label_smoothing: float = 0.05,
                 clip: float = 0.05):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.alpha = float(alpha)
        self.ls = float(label_smoothing)
        self.clip = float(clip)

    def forward(self, logits, targets):
        n_classes = logits.shape[-1]
        targets_soft = torch.full_like(logits, self.ls / n_classes)
        targets_soft.scatter_(1, targets.unsqueeze(1),
                              1.0 - self.ls + self.ls / n_classes)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        p_pos = probs[:, 1]
        p_neg_shift = (p_pos + self.clip).clamp(max=1.0) if self.clip > 0 else p_pos

        focal_pos = (1.0 - p_pos).clamp(min=1e-7).pow(self.gamma_pos)
        focal_neg = p_neg_shift.clamp(min=1e-7).pow(self.gamma_neg)

        loss = -(
            targets_soft[:, 1] * self.alpha * focal_pos * log_probs[:, 1]
            + targets_soft[:, 0] * (1.0 - self.alpha) * focal_neg * log_probs[:, 0]
        )
        return loss.mean()


# =========================================================================== #
# Args
# =========================================================================== #
def build_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--cache", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs_v3")
    p.add_argument("--tag", default="v3")

    # backbone / head
    p.add_argument("--backbone", default="vit_small_patch14_dinov2.lvd142m")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--head", choices=["simple", "stt", "cnn", "mm_simple"],
                   default="stt")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--drop", type=float, default=0.2)
    p.add_argument("--in-drop", type=float, default=0.1)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--n-spatial-layers", type=int, default=1)
    p.add_argument("--no-motion", action="store_true",
                   help="STTHead: disable motion stream (config B)")
    p.add_argument("--use-motion", action="store_true",
                   help="SimpleHead: enable motion stream (config C)")

    # LoRA (ViT only)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target", default="qkv")
    p.add_argument("--grad-checkpoint", action="store_true")

    # loss
    p.add_argument("--loss", choices=["asl", "focal", "ce"], default="asl")
    p.add_argument("--asl-gamma-pos", type=float, default=1.0)
    p.add_argument("--asl-gamma-neg", type=float, default=3.0)
    p.add_argument("--asl-alpha", type=float, default=0.71)
    p.add_argument("--asl-clip", type=float, default=0.05)
    p.add_argument("--focal-alpha", type=float, default=0.71)
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)

    # optimization
    p.add_argument("--aug-level", default="clean",
                   choices=["none", "basic", "clean", "full", "aggressive"])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-cnn", type=float, default=1e-5,
                   help="LR for full-finetune CNN backbones")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.10)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--sel-alpha", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=8)

    # data / windowing
    p.add_argument("--cap-per-track", type=int, default=24)
    p.add_argument("--pos-frac", type=float, default=0.4)
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--target-dt", type=float, default=0.1)
    p.add_argument("--hop-train", type=int, default=4)
    p.add_argument("--hop-val", type=int, default=2)
    p.add_argument("--hop-eval", type=int, default=1)
    p.add_argument("--min-purity-train", type=float, default=0.8)

    # FPI multimodal
    p.add_argument("--use-fpi-feats", action="store_true",
                   help="load FPI features from cache/fpi_feats/ and pass to head")

    # runtime
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# =========================================================================== #
# Utils
# =========================================================================== #
def seed_all(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker_init(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


class EMA:
    """EMA over all trainable params (LoRA / CNN / head)."""
    def __init__(self, modules, decay):
        self.decay, self.n = decay, 0
        self.params = [p for m in modules for p in m.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach(), alpha=1 - d)

    @contextlib.contextmanager
    def swap(self):
        backup = [p.detach().clone() for p in self.params]
        with torch.no_grad():
            for s, p in zip(self.shadow, self.params):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for b, p in zip(backup, self.params):
                    p.copy_(b)


def normalize(x, device):
    x = x.to(device, non_blocking=True).float().div_(255.0)
    m = IMAGENET_MEAN.to(device).view(1, 1, 3, 1, 1)
    s = IMAGENET_STD.to(device).view(1, 1, 3, 1, 1)
    return (x - m) / s


def forward_clip(backbone, head, x, device, is_cnn=False,
                 fpi_feats=None, has_fpi=None):
    """Unified forward: detects FPI-capable heads via attribute presence."""
    b, t = x.shape[:2]
    x = normalize(x, device)
    flat = x.reshape(b * t, 3, x.shape[-2], x.shape[-1]).contiguous(
        memory_format=torch.channels_last)

    if is_cnn:
        feats = extract_global_feature(backbone, flat)
        return head(feats.view(b, t, -1))

    cls, patches = extract_tokens(backbone, flat)
    D = cls.shape[-1]

    # MultiModalSimpleHead exposes `accepts_fpi = True`; SimpleHead/STTHead do not.
    accepts_fpi = getattr(head, "accepts_fpi", False) and fpi_feats is not None
    if accepts_fpi:
        return head(cls.view(b, t, D),
                    patches.view(b, t, patches.shape[1], D),
                    fpi_feats=fpi_feats, has_fpi=has_fpi)

    return head(cls.view(b, t, D),
                patches.view(b, t, patches.shape[1], D))


@torch.no_grad()
def predict(backbone, head, loader, device, use_amp, is_cnn=False):
    """Run inference over a loader; supports both 3-tuple and 5-tuple batches."""
    backbone.eval(); head.eval()
    P, Y, I = [], [], []
    for batch in loader:
        if len(batch) == 5:
            x, y, i, fpi, has_fpi = batch
            fpi = fpi.to(device, non_blocking=True)
            has_fpi = has_fpi.to(device, non_blocking=True)
        else:
            x, y, i = batch
            fpi, has_fpi = None, None

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = forward_clip(backbone, head, x, device, is_cnn=is_cnn,
                                  fpi_feats=fpi, has_fpi=has_fpi)
        P.append(F.softmax(logits.float(), 1)[:, 1].cpu())
        Y.append(y.cpu()); I.append(i.cpu())
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(I).numpy()


def window_meta(ds):
    m = ds.to_dataframe().set_index("idx")
    m["t"] = (m.first_frame + m.last_frame) / 2.0 / m.fps
    m["key"] = m.scene.astype(str) + "|" + m.video.astype(str) + "|" + m.track.astype(str)
    return m


def score_frame(meta, p, y, ids):
    m = meta.loc[ids]
    return pd.DataFrame({"key": m.key.values, "scene": m.scene.values,
                         "t": m.t.values, "label": y, "prob": p})


def split_summary(meta, p, y, ids, alpha):
    d = score_frame(meta, p, y, ids)
    b = to_bins(d); be = ema_time(b, alpha)
    return dict(n=int(len(d)), prevalence=float(d.label.mean()),
                win_auprc=ap(d.label, d.prob), win_auroc=auc(d.label, d.prob),
                bin_auprc=ap(b.label, b.prob), bin_auroc=auc(b.label, b.prob),
                bin_ema_auprc=ap(be.label, be.prob),
                bin_ema_auroc=auc(be.label, be.prob),
                win_f1_at_0p5=f1_at(d.label, d.prob, 0.5))


def save_preds(meta, ids, p, path):
    m = meta.loc[ids].reset_index().copy(); m["prob"] = p
    m.to_csv(path, index=False)


def snapshot(backbone, head, is_cnn):
    """Return head state + (LoRA-only for ViT, full for CNN)."""
    bb_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    if not is_cnn:
        bb_state = {k: v for k, v in bb_state.items() if "lora_" in k}
    return dict(
        head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        backbone=bb_state)


# =========================================================================== #
# Backbone construction
# =========================================================================== #
def build_backbone_and_head(args, device):
    """Return (backbone, head, is_cnn, lora_cfg, trainable_backbone_params, N_patches)."""
    is_cnn = (args.head == "cnn") or is_cnn_backbone(args.backbone)

    if is_cnn:
        # Full fine-tune CNN
        bb = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
        for p in bb.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, args.size, args.size, device=device)
            D = extract_global_feature(bb, dummy).shape[1]
        N = 0
        lora_cfg = None
        trainable = list(bb.parameters())
        print(f"[backbone] CNN {args.backbone}  D={D}  full-finetune")
    else:
        lora_cfg = dict(r=args.lora_r, alpha=args.lora_alpha,
                        dropout=args.lora_dropout, target=args.lora_target)
        bb = make_backbone(args.backbone, args.size, lora_cfg, pretrained=True,
                           grad_checkpoint=args.grad_checkpoint).to(device)
        with torch.no_grad():
            cls, patches = extract_tokens(
                bb, torch.zeros(1, 3, args.size, args.size, device=device))
        D = int(cls.shape[-1]); N = int(patches.shape[1])
        trainable = [p for p in bb.parameters() if p.requires_grad]
        print(f"[backbone] ViT {args.backbone}  D={D}  patches/frame={N}  "
              f"LoRA={sum(p.numel() for p in trainable)/1e6:.3f}M")

    # Build head
    hc = dict(T=args.T, d_model=args.d_model, layers=args.layers, heads=args.heads,
              dropout=args.drop, in_drop=args.in_drop,
              n_queries=args.n_queries, n_spatial_layers=args.n_spatial_layers,
              use_motion=not args.no_motion)

    if args.head == "simple":
        head = SimpleHead(D, T=args.T, d_model=args.d_model, layers=args.layers,
                          heads=args.heads, dropout=args.drop, in_drop=args.in_drop,
                          use_motion=args.use_motion)
    elif args.head == "stt":
        head = STTHead(D, T=args.T, d_model=args.d_model, layers=args.layers,
                       heads=args.heads, dropout=args.drop, in_drop=args.in_drop,
                       n_queries=args.n_queries,
                       n_spatial_layers=args.n_spatial_layers,
                       use_motion=not args.no_motion)
    elif args.head == "mm_simple":
        from models_multimodal import MultiModalSimpleHead
        head = MultiModalSimpleHead(
            D, T=args.T, d_model=args.d_model, layers=args.layers,
            heads=args.heads, dropout=args.drop, in_drop=args.in_drop,
            fpi_dim=11, use_motion=args.use_motion,
            gate_bias_init=-2.0,
        )
    else:  # cnn
        head = CNNHead(D, T=args.T, d_model=args.d_model, layers=args.layers,
                       heads=args.heads, dropout=args.drop, in_drop=args.in_drop)

    head = head.to(device)
    print(f"[head] {head.arch_version}  params="
          f"{sum(p.numel() for p in head.parameters())/1e6:.3f}M")
    return bb, head, is_cnn, lora_cfg, trainable, N


# =========================================================================== #
# Main
# =========================================================================== #
def main():
    args = build_args()
    if args.smoke:
        args.epochs, args.workers, args.patience = 1, 2, 1

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # data
    train_ds, val_ds, test_ds = make_stp_datasets(
        args.cache, args.folds, args.fold, size=args.size,
        aug_level=args.aug_level, T=args.T, target_dt=args.target_dt,
        hop_train=args.hop_train, hop_eval=args.hop_eval,
        hop_val=args.hop_val, min_purity_train=args.min_purity_train,
        load_fpi_feats=args.use_fpi_feats)

    if args.smoke:
        train_ds.samples = train_ds.samples[::40]
        val_ds.samples = val_ds.samples[::60]
        test_ds.samples = test_ds.samples[::120]

    T = train_ds.T
    sampler = make_stp_sampler(train_ds, cap_per_track=args.cap_per_track,
                               pos_frac=args.pos_frac, seed=args.seed)
    common = dict(num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init)
    train_dl = DataLoader(train_ds, args.batch, sampler=sampler, drop_last=True,
                          persistent_workers=args.workers > 0,
                          **({"prefetch_factor": 4} if args.workers > 0 else {}), **common)
    ev = dict(num_workers=min(args.workers, 4), pin_memory=True,
              worker_init_fn=worker_init)
    val_dl = DataLoader(val_ds, args.batch * 2, shuffle=False, **ev)
    test_dl = DataLoader(test_ds, args.batch * 2, shuffle=False, **ev)
    val_meta, test_meta = window_meta(val_ds), window_meta(test_ds)

    run = Path(args.out) / f"{args.tag}_f{args.fold}_s{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\n=== run: {run}\n    head={args.head} loss={args.loss} "
          f"T={T} size={args.size} use_motion={args.use_motion} "
          f"no_motion={args.no_motion} use_fpi={args.use_fpi_feats} ===\n")

    # model
    backbone, head, is_cnn, lora_cfg, trainable_bb, N = build_backbone_and_head(args, device)
    arch = head.arch_version

    # optimizer
    bb_lr = args.lr_cnn if is_cnn else args.lr_lora
    opt = torch.optim.AdamW([
        {"params": trainable_bb, "lr": bb_lr, "weight_decay": args.wd},
        {"params": head.parameters(), "lr": args.lr_head, "weight_decay": args.wd},
    ])

    # loss
    if args.loss == "asl":
        criterion = AsymmetricFocalLoss(
            gamma_pos=args.asl_gamma_pos, gamma_neg=args.asl_gamma_neg,
            alpha=args.asl_alpha, label_smoothing=args.label_smoothing,
            clip=args.asl_clip)
        print(f"[loss] ASL(gamma_pos={args.asl_gamma_pos}, "
              f"gamma_neg={args.asl_gamma_neg}, alpha={args.asl_alpha})")
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma,
                              label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # scheduler
    steps_ep = max(1, len(train_dl) // args.accum)
    total = steps_ep * args.epochs
    warm = max(1, int(args.warmup_frac * total))

    def lr_scale(s):
        if s < warm:
            return (s + 1) / warm
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) /
                                                max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    ema = EMA([backbone, head], args.ema_decay)

    # train loop
    best, best_ep, bad, best_state, logs = -1.0, -1, 0, None, []
    for epoch in range(args.epochs):
        train_dl.sampler.set_epoch(epoch)
        backbone.train(); head.train()
        t0, loss_sum, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, batch in enumerate(train_dl):
            if args.smoke and it >= 20:
                break
            if len(batch) == 5:
                x, y, _, fpi, has_fpi = batch
                fpi = fpi.to(device, non_blocking=True)
                has_fpi = has_fpi.to(device, non_blocking=True)
            else:
                x, y, _ = batch
                fpi, has_fpi = None, None
            y = y.to(device, non_blocking=True)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = forward_clip(backbone, head, x, device, is_cnn=is_cnn,
                                      fpi_feats=fpi, has_fpi=has_fpi)
                loss = criterion(logits.float(), y)
            (loss / args.accum).backward()
            is_last = (it + 1) == len(train_dl)
            if (it + 1) % args.accum == 0 or is_last:
                # rescale if last batch is smaller
                if is_last and (it + 1) % args.accum != 0:
                    actual = (it + 1) % args.accum
                    scale = args.accum / actual
                    for p in trainable_bb + list(head.parameters()):
                        if p.grad is not None:
                            p.grad.mul_(scale)
                nn.utils.clip_grad_norm_(
                    trainable_bb + list(head.parameters()), 1.0)
                opt.step(); sched.step(); ema.update()
                opt.zero_grad(set_to_none=True)
            loss_sum += float(loss.item()) * len(y); seen += len(y)
            if (it + 1) % 50 == 0:
                print(f"ep={epoch:02d} it={it+1:04d}/{len(train_dl)} "
                      f"loss={loss_sum/max(1,seen):.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}")

        # validation with EMA weights
        with ema.swap():
            pv, yv, iv = predict(backbone, head, val_dl, device, use_amp, is_cnn=is_cnn)
            sel = smoothed_bin_ap(score_frame(val_meta, pv, yv, iv), args.sel_alpha)
            win_ap, win_auc = ap(yv, pv), auc(yv, pv)
            if sel > best:
                best, best_ep, bad = sel, epoch, 0
                best_state = snapshot(backbone, head, is_cnn)
            else:
                bad += 1

        row = dict(epoch=epoch, train_loss=loss_sum / max(1, seen),
                   val_sel_auprc=sel, val_win_auprc=win_ap,
                   val_win_auroc=win_auc,
                   minutes=(time.time() - t0) / 60)
        logs.append(row)
        pd.DataFrame(logs).to_csv(run / "log.csv", index=False)
        print(row)
        if bad >= args.patience:
            print(f"early stop at epoch {epoch}; best={best_ep}")
            break

    # restore best EMA snapshot
    head.load_state_dict(best_state["head"])
    cur = backbone.state_dict()
    for k, v in best_state["backbone"].items():
        cur[k] = v.to(device)
    backbone.load_state_dict(cur)

    pv, yv, iv = predict(backbone, head, val_dl, device, use_amp, is_cnn=is_cnn)
    pt, yt, it_ = predict(backbone, head, test_dl, device, use_amp, is_cnn=is_cnn)

    res = dict(name=run.name, arch_version=arch, best_epoch=best_ep,
               is_cnn=is_cnn,
               use_motion=not args.no_motion if args.head == "stt" else args.use_motion,
               use_fpi_feats=args.use_fpi_feats,
               selection_auprc=best,
               val=split_summary(val_meta, pv, yv, iv, args.sel_alpha),
               test=split_summary(test_meta, pt, yt, it_, args.sel_alpha))
    (run / "results.json").write_text(json.dumps(res, indent=2))
    save_preds(val_meta, iv, pv, run / "preds_val.csv")
    save_preds(test_meta, it_, pt, run / "preds_test.csv")

    torch.save(dict(
        arch_version=arch,
        backbone_name=args.backbone,
        size=args.size,
        is_cnn=is_cnn,
        use_fpi_feats=args.use_fpi_feats,
        feature_dim=int(head.feature_dim if hasattr(head, "feature_dim") else 0),
        patch_tokens=N,
        head_state_dict=best_state["head"],
        lora_state_dict=best_state["backbone"],   # full state for CNN
        lora_config=lora_cfg,                     # None for CNN
        head_config=dict(
            T=T, d_model=args.d_model, layers=args.layers, heads=args.heads,
            dropout=args.drop, in_drop=args.in_drop,
            feature_dim=int(head.feature_dim if hasattr(head, "feature_dim") else 0),
            n_queries=args.n_queries,
            n_spatial_layers=args.n_spatial_layers,
            use_motion=(not args.no_motion) if args.head == "stt" else args.use_motion,
            fpi_dim=11,
        ),
        val_threshold=0.5,
        best_epoch=best_ep,
    ), run / "head.pt")

    t = res["test"]
    print(f"\nTEST  bin+EMA AUPRC={t['bin_ema_auprc']:.4f} "
          f"AUROC={t['bin_ema_auroc']:.4f} | "
          f"win AUPRC={t['win_auprc']:.4f}   best_epoch={best_ep}")
    print(f"saved: {run}")


if __name__ == "__main__":
    main()
