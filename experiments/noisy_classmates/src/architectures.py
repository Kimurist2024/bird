"""Classmate architectures for Noisy Classmates v1.

All take input shape (B, T=12, D=1536) Perch embeddings, optionally site/hour,
and return logits (B, T, 234). This keeps the co-evolutionary loss simple:
all classmates produce the same output shape so pseudo-labels are interchangeable.

Architecture diversity is the whole point of the technique — these three have
different inductive biases:
- ProtoSSM (existing): bidirectional SSM + prototypes — temporal continuity
- MLPMixer: per-window MLP + cross-window MLP — Transformer-like w/o attention
- AttnPool: lightweight self-attention over windows — local-global pooling
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the ProtoSSMv2 we already extracted
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "single_proto_ssm" / "src"))
from proto_ssm_model import ProtoSSMv2, PRODUCTION_CONFIG, load_proto_ssm  # noqa: E402

__all__ = ["ProtoSSMv2", "MLPMixerHead", "AttnPoolHead", "build_classmate", "load_proto_ssm"]


class MLPMixerHead(nn.Module):
    """Per-window MLP + cross-window MLP, no attention.

    Cheaper and more diverse from ProtoSSM than another SSM variant.
    Outputs (B, T, n_classes) so it slots into the same co-distillation loss.
    """

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 384,
        n_windows: int = 12,
        n_classes: int = 234,
        depth: int = 4,
        dropout: float = 0.1,
        expansion: int = 2,
        n_sites: int = 24,
        meta_dim: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_windows = n_windows

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(_MixerBlock(d_model, n_windows, expansion, dropout))

        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ):
        h = self.input_proj(emb)
        h = h + self.pos[:, : h.shape[1], :]
        if site_ids is not None and hours is not None:
            s = self.site_emb(site_ids)
            t = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s, t], dim=-1))
            h = h + meta[:, None, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        logits = self.cls(h)
        return logits, None, h


class _MixerBlock(nn.Module):
    def __init__(self, d_model: int, n_windows: int, expansion: int, dropout: float) -> None:
        super().__init__()
        # Token (cross-window) mixing
        self.norm_t = nn.LayerNorm(d_model)
        self.token_mix = nn.Sequential(
            nn.Linear(n_windows, n_windows * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_windows * expansion, n_windows),
            nn.Dropout(dropout),
        )
        # Channel (per-window) mixing
        self.norm_c = nn.LayerNorm(d_model)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = self.norm_t(x).transpose(1, 2)  # (B, D, T)
        y = self.token_mix(y).transpose(1, 2)
        x = x + y
        y = self.norm_c(x)
        x = x + self.channel_mix(y)
        return x


class AttnPoolHead(nn.Module):
    """Lightweight Transformer encoder over the 12 windows.

    Different inductive bias from ProtoSSM (no SSM state, global attention)
    and from MLPMixer (no fixed-shape MLP over time)."""

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 320,
        n_windows: int = 12,
        n_classes: int = 234,
        n_heads: int = 8,
        depth: int = 3,
        dropout: float = 0.1,
        n_sites: int = 24,
        meta_dim: int = 16,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ):
        h = self.input_proj(emb)
        h = h + self.pos[:, : h.shape[1], :]
        if site_ids is not None and hours is not None:
            s = self.site_emb(site_ids)
            t = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s, t], dim=-1))
            h = h + meta[:, None, :]
        h = self.enc(h)
        h = self.norm(h)
        return self.cls(h), None, h


# Registry for config-driven construction
ARCHITECTURES = {
    "proto_ssm": lambda **kw: ProtoSSMv2(**{**PRODUCTION_CONFIG, **kw}),
    "mlp_mixer": lambda **kw: MLPMixerHead(**kw),
    "attn_pool": lambda **kw: AttnPoolHead(**kw),
}


def build_classmate(arch: str, **kw) -> nn.Module:
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch '{arch}'; available: {list(ARCHITECTURES)}")
    return ARCHITECTURES[arch](**kw)
