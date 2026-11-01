"""models_multimodal.py — Multi-modal head combining DINOv2 + FPI features.

Architecture (v2 — AdaptiveGateFusion):
    DINOv2 CLS (B,T,D) ─┐
                         ├─→ AdaptiveGateFusion → fused CLS (B,T,D)
    FPI token (B,T,D) ──┘   (content-aware per-frame gate)

The old design used a single learnable query over 2 keys, which is
mathematically equivalent to a fixed scalar interpolation. This file
replaces it with a content-aware adaptive gate that produces a per-frame
weight g ∈ (0,1) computed from the raw FPI features themselves:

    fused[b,t] = (1 - g[b,t]) * dino_cls[b,t] + g[b,t] * fpi_tok[b,t]

Properties:
  - g is content-aware: high when phone is confidently detected,
    low when phone is absent (because g is also masked by has_fpi).
  - Params: ~15K (vs ~750K for the old attention head).
  - Starts conservative: gate_bias = -2.0 → sigmoid(-2) ≈ 0.12,
    so at initialization fused ≈ 0.88 * dino + 0.12 * fpi.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# =========================================================================== #
# Adaptive gate fusion — the new core
# =========================================================================== #
class AdaptiveGateFusion(nn.Module):
    """Content-aware adaptive gate between DINOv2 CLS and FPI features.

    Pipeline:
        1. Project 11-dim FPI features to feature_dim (identity-like MLP).
        2. Replace frames with has_fpi=0 by a learnable null token.
        3. Compute per-frame scalar gate g ∈ (0,1) from raw FPI features.
        4. Zero out g for frames without detection (force trust in DINOv2).
        5. Fuse: fused = (1 - g) * dino_cls + g * fpi_tok.

    Compared to the old `FrameFusionAttention`:
        - old: 1 fixed query × 2 keys  → α is essentially a constant scalar
        - new: gate depends on fpi_feats → truly adaptive per frame
        - old params: ~750K; new params: ~15K
    """

    def __init__(self, fpi_dim: int = 11, feature_dim: int = 384,
                 dropout: float = 0.1, gate_bias_init: float = -2.0):
        super().__init__()
        self.fpi_dim = int(fpi_dim)
        self.feature_dim = int(feature_dim)

        # -- FPI → feature_dim (identity-like, with GELU nonlinearity) --
        self.fpi_proj = nn.Sequential(
            nn.LayerNorm(fpi_dim),
            nn.Linear(fpi_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # -- Null token used for frames without FPI detection --
        # Shape (1, 1, D) so expand_as works on (B, T, D).
        self.null_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.null_token, std=0.02)

        # -- Per-frame gate: raw FPI (11-dim) → scalar in (0,1) --
        self.gate = nn.Sequential(
            nn.LayerNorm(fpi_dim),
            nn.Linear(fpi_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # -- Bias so that at init the gate is close to zero → trust DINOv2 --
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))

    def forward(self, dino_cls, fpi_feats, has_fpi=None, return_gate=False):
        """
        Args:
            dino_cls:  (B, T, D)   — DINOv2 CLS tokens (already in feature_dim)
            fpi_feats: (B, T, 11)  — raw FPI features
            has_fpi:   (B, T)      — 0/1 mask; computed from fpi_feats if None
        Returns:
            fused: (B, T, D)
            gate (optional): (B, T, 1) — how much FPI was used per frame
        """
        B, T, _ = dino_cls.shape

        if has_fpi is None:
            # Convention: a frame has FPI iff phone_conf > 0 (index 8)
            has_fpi = (fpi_feats[..., 8] > 0).float()   # (B, T)

        # -- 1. Project FPI and substitute null on missing frames --
        z = self.fpi_proj(fpi_feats)                    # (B, T, D)
        null = self.null_token.expand_as(z)             # (B, T, D)
        mask = has_fpi.unsqueeze(-1).bool()             # (B, T, 1)
        fpi_tok = torch.where(mask, z, null)            # (B, T, D)

        # -- 2. Per-frame gate from raw FPI features --
        g = torch.sigmoid(self.gate(fpi_feats) + self.gate_bias)   # (B, T, 1)

        # -- 3. Zero out g on frames without detection --
        # (Semantics: "if there is no phone in this frame, trust DINOv2 entirely.")
        g = g * has_fpi.unsqueeze(-1)                   # (B, T, 1)

        # -- 4. Fuse --
        fused = (1.0 - g) * dino_cls + g * fpi_tok      # (B, T, D)

        if return_gate:
            return fused, g
        return fused


# =========================================================================== #
# Multi-modal head
# =========================================================================== #
class MultiModalSimpleHead(nn.Module):
    """Fuse DINOv2 CLS with FPI features via AdaptiveGateFusion, then SimpleHead.

    Marker `accepts_fpi = True` allows train_v3.py to detect multimodal heads
    without relying on `hasattr(head, 'fpi_proj')`.
    """

    arch_version = "mm_simple_v2"
    accepts_fpi = True                              # ← marker for train_v3.py

    def __init__(
        self,
        feature_dim: int,
        T: int = 16,
        d_model: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        in_drop: float = 0.1,
        fpi_dim: int = 11,
        use_motion: bool = False,
        gate_bias_init: float = -2.0,
        **_ignored,
    ):
        super().__init__()
        self.T = int(T)
        self.feature_dim = int(feature_dim)
        self.fpi_dim = int(fpi_dim)

        # Core fusion module
        self.fusion = AdaptiveGateFusion(
            fpi_dim=fpi_dim,
            feature_dim=feature_dim,
            dropout=in_drop,
            gate_bias_init=gate_bias_init,
        )

        # The classification backbone head (unchanged from v3_simple)
        from models_v3 import SimpleHead
        self.simple_head = SimpleHead(
            feature_dim,
            T=T,
            d_model=d_model,
            layers=layers,
            heads=heads,
            dropout=dropout,
            in_drop=in_drop,
            use_motion=use_motion,
        )

    def forward(
        self,
        frame_cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        fpi_feats: torch.Tensor | None = None,
        has_fpi: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        Args:
            frame_cls:    (B, T, D)   — DINOv2 CLS
            patch_tokens: (B, T, N, D) — DINOv2 patch tokens
            fpi_feats:    (B, T, 11) or None
            has_fpi:      (B, T) or None
        """
        gate_values = None

        if fpi_feats is not None:
            if has_fpi is None:
                has_fpi = (fpi_feats[..., 8] > 0).float()
            if return_attention:
                frame_cls, gate_values = self.fusion(
                    frame_cls, fpi_feats, has_fpi, return_gate=True
                )
            else:
                frame_cls = self.fusion(frame_cls, fpi_feats, has_fpi)

        if return_attention:
            logits, head_aux = self.simple_head(
                frame_cls, patch_tokens, return_attention=True
            )
            aux = dict(head_aux or {})
            aux["fusion_gate"] = gate_values       # (B, T, 1) per-frame weight on FPI
            return logits, aux

        return self.simple_head(frame_cls, patch_tokens)
