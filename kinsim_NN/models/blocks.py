"""Building blocks: DiTBlock with AdaLN-Zero, MultiHeadAttention, FFN, helpers.

DiT (Peebles & Xie, 2022) introduces AdaLN-Zero — a layer norm whose
shift/scale/gate are all modulated by a per-sample conditioning vector
(here, ``z_emb`` derived from the latent noise + a global pooled
representation of the condition). The gate is initialised to zero so
the block starts as an identity — improves training stability.

We also expose :func:`maybe_spectral_norm` so the discriminator can
spectrally normalise every Linear / qkv projection without polluting
the generator.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


_LEGACY_SN_WARNED = False


def maybe_spectral_norm(module: nn.Module, apply: bool = True) -> nn.Module:
    """Apply spectral normalisation in-place if ``apply`` is True.

    Prefers the modern :func:`torch.nn.utils.parametrizations.spectral_norm`
    API (PyTorch ≥ 1.12) which composes cleanly with double-backward (needed
    for WGAN-GP's gradient penalty). The legacy
    :func:`torch.nn.utils.spectral_norm` is known to leak / break with
    create_graph=True on some torch versions — we use it only as a last
    resort and warn loudly so a silent fallback can't ship to a training
    run unnoticed.
    """
    if not apply:
        return module
    try:
        from torch.nn.utils.parametrizations import spectral_norm as _sn
        return _sn(module)
    except ImportError:
        global _LEGACY_SN_WARNED
        if not _LEGACY_SN_WARNED:
            import logging
            logging.getLogger(__name__).warning(
                "torch.nn.utils.parametrizations.spectral_norm is unavailable "
                "(torch < 1.12). Falling back to legacy nn.utils.spectral_norm — "
                "this is known to be flaky with WGAN-GP's double-backward. "
                "Upgrade torch ≥ 1.12 if D loss diverges in the first ~100 steps."
            )
            _LEGACY_SN_WARNED = True
        return nn.utils.spectral_norm(module)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: scale and shift on the last (channel) dim.

    ``x`` shape: (B, K, d). ``shift``/``scale`` shape: (B, d). Broadcast.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Multi-head self-attention (with optional spectral norm)
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        spectral_norm: bool = False,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = maybe_spectral_norm(nn.Linear(d_model, d_model * 3, bias=True), spectral_norm)
        self.out = maybe_spectral_norm(nn.Linear(d_model, d_model, bias=True), spectral_norm)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, D = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, K, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, K, K)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                   # (B, H, K, d_head)
        out = out.transpose(1, 2).reshape(B, K, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0,
                 spectral_norm: bool = False, drop_rate: float = 0.0,
                 activation: Callable = F.gelu):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = maybe_spectral_norm(nn.Linear(d_model, hidden, bias=True), spectral_norm)
        self.fc2 = maybe_spectral_norm(nn.Linear(hidden, d_model, bias=True), spectral_norm)
        self.act = activation
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# AdaLN-Zero DiT block (used by Generator)
# ---------------------------------------------------------------------------


class AdaLNZeroBlock(nn.Module):
    """DiT-style block with AdaLN-Zero modulation conditioned on ``z_emb``.

    z_emb: per-sample conditioning vector (B, d_cond) — at construction
    time ``d_cond`` matches the embedding output of the noise z.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_cond: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, spectral_norm=False,
                                           drop_rate=drop_rate)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = FFN(d_model, mlp_ratio, spectral_norm=False, drop_rate=drop_rate)
        # 6 modulation parameters per block: shift/scale/gate × {attn, ffn}
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model, bias=True),
        )
        # AdaLN-Zero init: zero so the block starts as identity.
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x: torch.Tensor, z_emb: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = \
            self.mod(z_emb).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_ffn.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift_ffn, scale_ffn))
        return x


# ---------------------------------------------------------------------------
# Plain transformer block (used by Discriminator — no AdaLN, has spectral norm)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-LN transformer block. Used by the critic (discriminator)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        spectral_norm: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads,
                                           spectral_norm=spectral_norm,
                                           drop_rate=drop_rate)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mlp_ratio, spectral_norm=spectral_norm,
                       drop_rate=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Sinusoidal time / noise embedding (DiT-style, learned MLP on top)
# ---------------------------------------------------------------------------


def sinusoidal_embed(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal positional / continuous-value embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=x.device) / half
    )
    args = x.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


__all__ = [
    "maybe_spectral_norm",
    "modulate",
    "MultiHeadSelfAttention",
    "FFN",
    "AdaLNZeroBlock",
    "TransformerBlock",
    "sinusoidal_embed",
]
