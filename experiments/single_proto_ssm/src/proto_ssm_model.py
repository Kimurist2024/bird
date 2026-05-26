"""ProtoSSMv2 single-model architecture, extracted verbatim from
nina2025/birdclef-2026-eos-5 (Model_2 cell). Use it to load the trained weights
at experiments/single_proto_ssm/models/proto_ssm_best.pt.

Config used at training time (from v17_logs.json):
    ProtoSSMv2(d_input=1536, d_model=320, d_state=32, n_ssm_layers=4,
               n_classes=234, n_windows=12, dropout=0.12,
               n_sites=20, meta_dim=24,
               use_cross_attn=True, cross_attn_heads=8)

Inputs at inference time:
    emb           (B, T, 1536)   Perch v2 embeddings per 5-second window
    perch_logits  (B, T, 234)    Perch v2 classification logits (optional, for gated fusion)
    site_ids      (B,)           recording site index (LongTensor)
    hours         (B,)           hour of day in UTC, 0..23

Outputs:
    species_logits  (B, T, 234)  per-window per-class logits → sigmoid for probabilities
    family_logits   None (taxonomic aux head not attached at inference)
    h_temporal      (B, T, 320)  intermediate features (for downstream models)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Mamba-style selective state-space model — input-dependent discretization."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, _z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out
        residual = x
        x = self.norm2(x)
        return residual + self.ffn(x)


class ProtoSSMv2(nn.Module):
    """Bidirectional SSM + cross-attention + per-class prototype head."""

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 192,
        d_state: int = 16,
        n_ssm_layers: int = 2,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.2,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(d_model, n_heads=cross_attn_heads, dropout=dropout)

        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        self.n_families = 0
        self.family_head: nn.Linear | None = None

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ):
        B, T, _ = emb.shape
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        for fwd, bwd, merge, norm in zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        return species_logits, None, h_temporal


# Configuration that matches the saved weights at models/proto_ssm_best.pt
PRODUCTION_CONFIG = dict(
    d_input=1536, d_model=320, d_state=32, n_ssm_layers=4,
    n_classes=234, n_windows=12, dropout=0.12,
    n_sites=20, meta_dim=24,
    use_cross_attn=True, cross_attn_heads=8,
)


def load_proto_ssm(weights_path: str, device: str = "cpu") -> ProtoSSMv2:
    """Instantiate ProtoSSMv2 with production config and load weights."""
    model = ProtoSSMv2(**PRODUCTION_CONFIG)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_proto_ssm] missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load_proto_ssm] unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    model.eval().to(device)
    return model
