"""TransformerGenerator — same family as kinsim_NN v6 generator, smaller.

Six AdaLN-Zero blocks, d_model = 192, six heads. ~3 M parameters. The
size reduction relative to v6's 10 M is deliberate: without a critic
forcing the model to chase ever-more-discriminating features, the
representational pressure on the generator is lower, and a smaller
backbone is enough to fit the target distribution under a direct loss.

Input per position (K = 21): one-hot fwd base, one-hot rev base, one-hot
fwd methylation, one-hot rev methylation, learned positional embedding.
Plus a per-sample latent z ∈ ℝ^{z_dim} (GeneratorConfig.z_dim; default 64,
96 in the post-audit config) for stochasticity.

Output: kinetic tile (B, K, 4) = (IPD_fwd, PW_fwd, IPD_rev, PW_rev) in
log1p(frames) space. At generation time the output is exponentiated,
clamped to [1, 255] and CodecV1-encoded before writing the BAM tags.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN affine: y = x * (1 + scale) + shift, broadcast over positions."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, drop_rate: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, D = x.shape
        qkv = self.qkv(x).reshape(B, K, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, K, D)
        return self.out(out)


class FFN(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, drop_rate: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class AdaLNZeroBlock(nn.Module):
    """One transformer block with AdaLN-Zero conditioning (DiT-style).

    Six modulation channels per block: (shift, scale, gate) for the
    attention sub-block and the FFN sub-block. All driven from a shared
    per-sample conditioning vector via a single Linear. The gate is
    zero-initialised so the block starts as the identity transform.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, drop_rate=drop_rate)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.ffn = FFN(d_model, mlp_ratio=mlp_ratio, drop_rate=drop_rate)
        # SiLU before the modulation Linear: matches DiT / v6 — without it
        # the cond → (shift, scale, gate) mapping is purely linear.
        self.mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s1, sc1, g1, s2, sc2, g2 = self.mod(c).chunk(6, dim=-1)
        x = x + g1.unsqueeze(1) * self.attn(_modulate(self.norm1(x), s1, sc1))
        x = x + g2.unsqueeze(1) * self.ffn(_modulate(self.norm2(x), s2, sc2))
        return x


@dataclass
class GeneratorConfig:
    k: int = 21
    n_channels: int = 4
    n_meth_types: int = 4
    d_model: int = 192
    n_layers: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    z_dim: int = 64
    pos_embed_dim: int = 24
    drop_rate: float = 0.0


class TransformerGenerator(nn.Module):
    """Conditional transformer generator. Stochastic via latent z.

    forward(base_fwd, base_rev, meth_fwd, meth_rev, z) → signal
        input one-hots: (B, K, 4) for bases, (B, K, M) for meth
        z: (B, z_dim)
        signal: (B, K, n_channels) in log1p(frames) space
    """

    def __init__(self, cfg: GeneratorConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 4 + 4 + cfg.n_meth_types + cfg.n_meth_types + cfg.pos_embed_dim
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.randn(cfg.k, cfg.pos_embed_dim) * 0.02)
        self.z_mlp = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.cond_pool_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.drop_rate)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=True),
        )
        nn.init.zeros_(self.final_mod[1].weight)
        nn.init.zeros_(self.final_mod[1].bias)
        self.head = nn.Linear(cfg.d_model, cfg.n_channels)
        # Zero-init the head so the model starts emitting ~0 (i.e. log1p(0))
        # and learns to push values up from there.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _build_tokens(
        self,
        base_fwd: torch.Tensor,
        base_rev: torch.Tensor,
        meth_fwd: torch.Tensor,
        meth_rev: torch.Tensor,
    ) -> torch.Tensor:
        """Concat one-hots + pos_embed → (B, K, d_model)."""
        B, K, _ = base_fwd.shape
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)  # (B, K, pos_embed_dim)
        x = torch.cat([base_fwd, base_rev, meth_fwd, meth_rev, pos], dim=-1)
        return self.input_proj(x)

    def forward(
        self,
        base_fwd_onehot: torch.Tensor,
        base_rev_onehot: torch.Tensor,
        meth_fwd_onehot: torch.Tensor,
        meth_rev_onehot: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._build_tokens(base_fwd_onehot, base_rev_onehot,
                                    meth_fwd_onehot, meth_rev_onehot)
        z_e = self.z_mlp(z)
        cond_e = self.cond_pool_proj(tokens.mean(dim=1))
        cond = z_e + cond_e                               # (B, d_model)

        x = tokens
        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.final_mod(cond).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        return self.head(x)                               # (B, K, n_channels)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
