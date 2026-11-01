"""models_v3.py -- shared by training (train_v3.py) and inference (v3_inference.py).

Heads share the call signature:
    ViT path : head(frame_cls, patch_tokens)
    CNN path : head(frame_features)

  simple : concat(CLS, mean(patches)) [+ optional motion] -> temporal Transformer
  stt    : STTHead (spatial cross-attention + motion + gated fusion)
  cnn    : CNNHead (global-pooled per-frame features + temporal Transformer)

Backbones may be ViT (DINOv2 + LoRA) or CNN (ResNet-50, full fine-tune).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, inject_adapter_in_model
from models_multimodal import MultiModalSimpleHead

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


# --------------------------------------------------------------------------- #
# Token / feature extractors
# --------------------------------------------------------------------------- #
def extract_tokens(backbone: nn.Module, x: torch.Tensor):
    """Extract (cls, patches) from a ViT. Returns (N, D), (N, P, D)."""
    f = backbone.forward_features(x)
    if isinstance(f, dict):
        if "x_norm_clstoken" in f and "x_norm_patchtokens" in f:
            return f["x_norm_clstoken"], f["x_norm_patchtokens"]
        if "last_hidden_state" in f:
            f = f["last_hidden_state"]
        else:
            raise TypeError(f"Unsupported forward_features keys: {list(f.keys())}")
    if not torch.is_tensor(f) or f.ndim != 3:
        raise ValueError(f"Expected [N,L,D] tokens, got {type(f)}")
    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    return f[:, 0], f[:, prefix:]


def extract_global_feature(backbone: nn.Module, x: torch.Tensor):
    """Global-pooled features from a CNN backbone. Returns (N, D)."""
    f = backbone.forward_features(x)
    if isinstance(f, dict):
        for k in ("global_pool", "pool", "features", "last_hidden_state"):
            if k in f:
                f = f[k]
                break
        else:
            raise TypeError(f"Unsupported CNN feature keys: {list(f.keys())}")
    if f.ndim == 4:
        f = f.mean(dim=(2, 3))
    elif f.ndim == 3:
        f = f.mean(dim=1)
    return f


# --------------------------------------------------------------------------- #
# SimpleHead  (configs A, C)
# --------------------------------------------------------------------------- #
class SimpleHead(nn.Module):
    """Concat(CLS, mean(patches) [, motion]) -> proj -> temporal Transformer.

    Config A: use_motion=False -> no explicit temporal-difference stream.
    Config C: use_motion=True  -> add a first-difference motion stream.
    """
    arch_version = "v3_simple"

    def __init__(self, feature_dim, T=16, d_model=256, layers=2, heads=4,
                 dropout=0.2, in_drop=0.2, use_motion=False, **_ignored):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.T = int(T)
        self.use_motion = bool(use_motion)

        in_dim = 2 * feature_dim + (feature_dim if use_motion else 0)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Dropout(in_drop),
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model))

        if use_motion:
            self.motion_proj = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim))

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, frame_cls, patch_tokens, return_attention=False):
        b, t, _ = frame_cls.shape
        if t != self.T:
            raise ValueError(f"Expected T={self.T}, got {t}")

        parts = [frame_cls, patch_tokens.mean(dim=2)]
        if self.use_motion:
            if t > 1:
                motion = frame_cls[:, 1:] - frame_cls[:, :-1]
                motion = F.pad(motion, (0, 0, 1, 0))
            else:
                motion = torch.zeros_like(frame_cls)
            parts.append(self.motion_proj(motion))

        z = self.proj(torch.cat(parts, dim=-1))
        z = torch.cat([self.cls.expand(b, -1, -1), z], dim=1) + self.pos[:, :t + 1]
        logits = self.classifier(self.enc(z)[:, 0])
        return (logits, {}) if return_attention else logits


# --------------------------------------------------------------------------- #
# STTHead  (configs B, D)
# --------------------------------------------------------------------------- #
class STTHead(nn.Module):
    """Spatio-Temporal Transformer Head.

    Pipeline:
        1. Project CLS + patches to d_model
        2. Per-frame cross-attention: n_queries learnable queries attend to (CLS, patches)
        3. Motion stream: first-difference of CLS features across time
        4. Fuse [CLS, pooled_queries, motion] -> temporal Transformer with CLS
        5. Classify from temporal CLS token

    Config B: use_motion=False (spatial attention only)
    Config D: use_motion=True  (full model)
    """
    arch_version = "stt_v1"

    def __init__(self, feature_dim, T=16, d_model=256, layers=2, heads=4,
                 dropout=0.2, in_drop=0.1, n_queries=8, n_spatial_layers=1,
                 use_motion=True, **_ignored):
        super().__init__()
        self.T = int(T)
        self.feature_dim = int(feature_dim)
        self.n_queries = int(n_queries)
        self.n_spatial_layers = int(n_spatial_layers)
        self.use_motion = bool(use_motion)

        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(in_drop),
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model))

        self.spatial_queries = nn.Parameter(torch.zeros(1, n_queries, d_model))
        nn.init.trunc_normal_(self.spatial_queries, std=0.02)

        spatial_layer = nn.TransformerDecoderLayer(
            d_model, heads, d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.spatial_decoder = nn.TransformerDecoder(
            spatial_layer, n_spatial_layers)

        if use_motion:
            self.motion_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model))

        d_fuse = d_model * (3 if use_motion else 2)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_fuse), nn.Dropout(dropout),
            nn.Linear(d_fuse, d_model), nn.LayerNorm(d_model))

        self.temporal_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d_model))
        nn.init.trunc_normal_(self.temporal_cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer, layers, enable_nested_tensor=False)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2))

    # ---- manual unroll to expose cross-attention weights ---------------- #
    def _spatial_with_attn(self, tokens_flat, queries):
        """Single-layer unroll. Returns (out, cross_attn_weights).

        cross_attn_weights: (BT, heads, n_queries, N+1)
        For n_spatial_layers > 1, extra layers run in standard mode
        and only the first layer's attention is returned.
        """
        layer = self.spatial_decoder.layers[0]

        # self-attention (queries attend to themselves)
        x = queries
        xn = layer.norm1(x)
        sa_out, _ = layer.self_attn(xn, xn, xn, need_weights=False)
        x = x + layer.dropout1(sa_out)

        # cross-attention (queries attend to tokens)
        xn = layer.norm2(x)
        ca_out, ca_w = layer.multihead_attn(
            xn, tokens_flat, tokens_flat,
            need_weights=True, average_attn_weights=False)
        x = x + layer.dropout2(ca_out)

        # feed-forward
        xn = layer.norm3(x)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(xn))))
        x = x + layer.dropout3(ff)

        # remaining layers (if any) run in standard mode
        if self.n_spatial_layers > 1:
            rest = nn.TransformerDecoder(
                nn.ModuleList(list(self.spatial_decoder.layers[1:])),
                self.n_spatial_layers - 1)
            x = rest(x, tokens_flat)

        return x, ca_w

    def forward(self, frame_cls, patch_tokens, return_attention=False):
        B, T, D = frame_cls.shape
        N = patch_tokens.shape[2]
        if T != self.T:
            raise ValueError(f"Expected T={self.T}, got {T}")

        tokens = torch.cat([frame_cls.unsqueeze(2), patch_tokens], dim=2)
        tokens = self.feat_proj(tokens)
        BT = B * T
        tokens_flat = tokens.reshape(BT, N + 1, -1)
        queries = self.spatial_queries.expand(BT, -1, -1)

        attn_maps = None
        if return_attention:
            spatial_out, ca_w = self._spatial_with_attn(tokens_flat, queries)
            attn_maps = ca_w.mean(dim=1).reshape(B, T, self.n_queries, N + 1)
        else:
            spatial_out = self.spatial_decoder(queries, tokens_flat)

        spatial_pooled = spatial_out.mean(dim=1).reshape(B, T, -1)
        cls_stream = tokens[:, :, 0, :]

        if self.use_motion:
            if T > 1:
                motion = cls_stream[:, 1:] - cls_stream[:, :-1]
                motion = F.pad(motion, (0, 0, 1, 0))
            else:
                motion = torch.zeros_like(cls_stream)
            motion = self.motion_proj(motion)
            fused = torch.cat([cls_stream, spatial_pooled, motion], dim=-1)
        else:
            fused = torch.cat([cls_stream, spatial_pooled], dim=-1)

        z = self.fuse(fused)
        z = torch.cat([self.temporal_cls.expand(B, -1, -1), z], dim=1) + self.pos[:, :T + 1]
        z = self.temporal_encoder(z)
        logits = self.classifier(z[:, 0])

        if return_attention:
            return logits, {"spatial_attn": attn_maps}
        return logits


# --------------------------------------------------------------------------- #
# CNNHead  (config E -- ResNet-50 baseline)
# --------------------------------------------------------------------------- #
class CNNHead(nn.Module):
    """Per-frame global CNN feature + temporal Transformer. No spatial attention."""
    arch_version = "v3_cnn"

    def __init__(self, feature_dim, T=16, d_model=256, layers=2, heads=4,
                 dropout=0.2, in_drop=0.1, **_ignored):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.T = int(T)

        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Dropout(in_drop),
            nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model))

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 2))

    def forward(self, features, patch_tokens=None, return_attention=False):
        # features: (B, T, D). patch_tokens is ignored.
        b, t, _ = features.shape
        if t != self.T:
            raise ValueError(f"Expected T={self.T}, got {t}")
        z = self.proj(features)
        z = torch.cat([self.cls.expand(b, -1, -1), z], dim=1) + self.pos[:, :t + 1]
        logits = self.classifier(self.enc(z)[:, 0])
        return (logits, {}) if return_attention else logits


# --------------------------------------------------------------------------- #
# build_head
# --------------------------------------------------------------------------- #
def build_head(arch_version: str, feature_dim: int, hc: dict) -> nn.Module:
    if arch_version == "v3_simple":
        return SimpleHead(
            feature_dim, T=hc["T"], d_model=hc["d_model"],
            layers=hc["layers"], heads=hc["heads"],
            dropout=hc["dropout"], in_drop=hc.get("in_drop", 0.2),
            use_motion=hc.get("use_motion", False))

    if arch_version == "stt_v1":
        return STTHead(
            feature_dim, T=hc["T"], d_model=hc["d_model"],
            layers=hc["layers"], heads=hc["heads"],
            dropout=hc["dropout"], in_drop=hc.get("in_drop", 0.1),
            n_queries=hc.get("n_queries", 8),
            n_spatial_layers=hc.get("n_spatial_layers", 1),
            use_motion=hc.get("use_motion", True))

    if arch_version == "v3_cnn":
        return CNNHead(
            feature_dim, T=hc["T"], d_model=hc["d_model"],
            layers=hc["layers"], heads=hc["heads"],
            dropout=hc["dropout"], in_drop=hc.get("in_drop", 0.1))

    if arch_version in ("mm_simple_v1", "mm_simple_v2"):
        return MultiModalSimpleHead(
            feature_dim,
            T=hc["T"],
            d_model=hc["d_model"],
            layers=hc["layers"],
            heads=hc["heads"],
            dropout=hc["dropout"],
            in_drop=hc.get("in_drop", 0.1),
            use_motion=hc.get("use_motion", False),
        )
    raise ValueError(f"unknown arch_version {arch_version!r}")


# --------------------------------------------------------------------------- #
# Backbone factory
# --------------------------------------------------------------------------- #
def is_cnn_backbone(name: str) -> bool:
    """Heuristic: names containing these substrings are CNN backbones."""
    n = name.lower()
    return any(k in n for k in ("resnet", "efficientnet", "convnext",
                                "regnet", "mobilenet", "vgg", "densenet"))


def make_backbone(name: str, size: int, lora: dict | None, pretrained: bool = True,
                  grad_checkpoint: bool = False):
    kwargs = dict(pretrained=pretrained, num_classes=0)
    if not is_cnn_backbone(name):
        kwargs["img_size"] = size          # only ViT/Swin accept img_size
    bb = timm.create_model(name, **kwargs)
    for p in bb.parameters():
        p.requires_grad_(False)
    if lora is not None:
        targets = [t.strip() for t in lora["target"].split(",") if t.strip()]
        cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                         lora_dropout=lora["dropout"],
                         target_modules=targets, bias="none")
        bb = inject_adapter_in_model(cfg, bb)
    if grad_checkpoint and hasattr(bb, "set_grad_checkpointing"):
        bb.set_grad_checkpointing(True)
    return bb


def load_lora_state(backbone: nn.Module, lora_state: dict, device) -> None:
    cur = backbone.state_dict()
    missing = [k for k in lora_state if k not in cur]
    if missing:
        raise KeyError(f"{len(missing)} LoRA keys not found, e.g. {missing[:2]}")
    for k, v in lora_state.items():
        cur[k] = v.to(device)
    backbone.load_state_dict(cur)


def load_backbone_state(backbone: nn.Module, full_state: dict, device) -> None:
    """Load a full backbone state dict (used for CNN baselines)."""
    cur = backbone.state_dict()
    for k, v in full_state.items():
        if k in cur:
            cur[k] = v.to(device)
    backbone.load_state_dict(cur)
