from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, T, D]
        return out.mean(dim=1)


# ------------------------- building blocks -------------------------

class MeanPool(nn.Module):
    def forward(self, h):            # [B, T, D] -> [B, D]
        return h.mean(dim=1)


class MaxPool(nn.Module):
    def forward(self, h):
        return h.amax(dim=1)


class MultiScalePool(nn.Module):
    """Attention + mean + max → d_model."""
    def __init__(self, d_model, attn_pool_cls):
        super().__init__()
        self.attn = attn_pool_cls(d_model)
        self.proj = nn.Linear(d_model * 3, d_model)

    def forward(self, h):
        a = self.attn(h)
        m = h.mean(dim=1)
        x = h.amax(dim=1)
        return self.proj(torch.cat([a, m, x], dim=-1))


def build_pool(mode: str, d_model: int, attn_pool_cls):
    if mode == "attn":
        return attn_pool_cls(d_model)
    if mode == "mean":
        return MeanPool()
    if mode == "max":
        return MaxPool()
    if mode == "multi":
        return MultiScalePool(d_model, attn_pool_cls)
    raise ValueError(f"unknown pool_mode: {mode}")


class ConcatFusion(nn.Module):
    """[B, D] * N  ->  [B, D]  (Linear projection)."""
    def __init__(self, d_model, n_branches):
        super().__init__()
        self.proj = nn.Linear(d_model * n_branches, d_model)

    def forward(self, branches):
        return self.proj(torch.cat(branches, dim=-1)), None


class GatedFusion(nn.Module):
    """Per-sample softmax gating over branch summaries."""
    def __init__(self, d_model, n_branches):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * n_branches, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_branches),
        )

    def forward(self, branches):
        stacked = torch.stack(branches, dim=1)            # [B, N, D]
        concat = torch.cat(branches, dim=-1)              # [B, N*D]
        w = F.softmax(self.gate(concat), dim=-1)          # [B, N]
        fused = (stacked * w.unsqueeze(-1)).sum(dim=1)    # [B, D]
        return fused, w


class ConvAttnFusion(nn.Module):
    """Convolutional attention fusion over branch summaries."""
    def __init__(self, d_model, n_branches):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, branches):
        stacked = torch.stack(branches, dim=1)            # [B, N, D]
        attn = self.conv(stacked.transpose(1, 2)).transpose(1, 2)  # [B, N, D]
        attn = torch.softmax(attn, dim=1)                # [B, N, D]
        fused = (stacked * attn).sum(dim=1)             # [B, D]
        fused += stacked.mean(dim=1)                          # residual connection
        return fused, attn.mean(dim=-1)                 # return mean attention weights


def build_fusion(mode: str, d_model: int, n_branches: int):
    if n_branches == 1:
        return nn.Identity()   # 단일 branch일 땐 fusion 자체가 no-op
    if mode == "concat":
        return ConcatFusion(d_model, n_branches)
    if mode == "gated":
        return GatedFusion(d_model, n_branches)
    if mode == 'ConvAttn':
        return ConvAttnFusion(d_model, n_branches)
    raise ValueError(f"unknown fusion_mode: {mode}")


# ------------------------- main model -------------------------

class IROBOT(nn.Module):
    """
    Dual-branch BiGRU (raw + 1st-order difference) with
    optional bidirectional cross-attention, configurable pooling, and configurable fusion.
    Designed so that ablations can be run by flipping flags.

    Ablation axes:
        use_diff        : dual-branch vs raw-only
        diff_only       : diff-branch only (no raw branch); forces use_diff=True
        use_cross_attn  : with/without cross-branch attention
        pool_mode       : "attn" | "mean" | "max" | "multi"
        fusion_mode     : "concat" | "gated" | "ConvAttn"
    """
    def __init__(
        self,
        input_dim: int,
        attn_pool_cls,                    # 기존 AttentionPooling 주입
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_diff: bool = True,
        diff_only: bool = False,
        use_cross_attn: bool = True,
        pool_mode: str = "multi",
        fusion_mode: str = "gated",
    ):
        super().__init__()
        if diff_only:
            use_diff = True
        self.use_diff = use_diff
        self.diff_only = diff_only
        self.use_cross_attn = use_cross_attn and use_diff and not diff_only
        self.pool_mode = pool_mode
        self.fusion_mode = fusion_mode

        d_model = hidden_dim * 2
        self.d_model = d_model

        # Raw branch (skipped when diff_only=True)
        if not diff_only:
            self.raw_gru = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=2, batch_first=True,
                bidirectional=True, dropout=dropout,
            )
            self.raw_norm = nn.LayerNorm(d_model)
            self.raw_pool = build_pool(pool_mode, d_model, attn_pool_cls)

        # Diff branch (always present when use_diff or diff_only)
        if use_diff:
            self.diff_gru = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=2, batch_first=True,
                bidirectional=True, dropout=dropout,
            )
            self.diff_norm = nn.LayerNorm(d_model)
            self.diff_pool = build_pool(pool_mode, d_model, attn_pool_cls)

        # Fusion: 2 branches normally, 1 branch when diff_only or raw-only
        if diff_only:
            n_branches = 1
        else:
            n_branches = 2 if use_diff else 1
        self.fusion = build_fusion(fusion_mode, d_model, n_branches)

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _first_diff(x):
        d = x[:, 1:, :] - x[:, :-1, :]
        z = torch.zeros_like(x[:, :1, :])
        return torch.cat([z, d], dim=1)

    def forward(self, x, return_features: bool = False, return_gates: bool = False):
        if self.diff_only:
            # Diff-only: skip raw branch entirely
            diff_in = self._first_diff(x)
            diff_h, _ = self.diff_gru(diff_in)
            diff_h = self.diff_norm(diff_h)
            diff_pooled = self.diff_pool(diff_h)
            fused, gates = diff_pooled, None   # single branch — no fusion
        elif self.use_diff:
            # Dual branch: raw + diff
            raw_h, _ = self.raw_gru(x)
            raw_h = self.raw_norm(raw_h)
            diff_in = self._first_diff(x)
            diff_h, _ = self.diff_gru(diff_in)
            diff_h = self.diff_norm(diff_h)
            raw_pooled = self.raw_pool(raw_h)
            diff_pooled = self.diff_pool(diff_h)
            fused, gates = self.fusion([raw_pooled, diff_pooled])
        else:
            # Raw-only
            raw_h, _ = self.raw_gru(x)
            raw_h = self.raw_norm(raw_h)
            raw_pooled = self.raw_pool(raw_h)
            fused, gates = raw_pooled, None   # single branch — no fusion

        logits = self.head(fused).squeeze(-1)

        outs = [logits]
        if return_features: outs.append(fused)
        if return_gates:    outs.append(gates)
        return outs[0] if len(outs) == 1 else tuple(outs)


def build_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    **_,
) -> nn.Module:
    name = model_name.lower()

    # ---------- IROBOT family ----------
    irobot_variants = {
        # incremental ablation
        "irobot_raw":               dict(use_diff=False, use_cross_attn=False, pool_mode="mean",  fusion_mode="ConvAttn"),
        "irobot_dual":              dict(use_diff=True,  use_cross_attn=False, pool_mode="mean",  fusion_mode="concat"),
        # pooling ablation
        "irobot_pool_conv":        dict(use_diff=True,  use_cross_attn=False, pool_mode="mean",  fusion_mode="ConvAttn"),
        "irobot_pool_conv_diff":   dict(diff_only=True,use_cross_attn=False, pool_mode="mean",  fusion_mode="ConvAttn"),
    }
    if name in irobot_variants:
        return IROBOT(
            input_dim=input_dim,
            attn_pool_cls=AttentionPooling,   # factory가 주입
            hidden_dim=hidden_dim,
            dropout=dropout,
            **irobot_variants[name],
        )
    raise ValueError(
        f"Unknown model_name={model_name!r}. "
        f"Available iRobot variants: {', '.join(irobot_variants)}"
    )
