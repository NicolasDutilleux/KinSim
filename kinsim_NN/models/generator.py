"""Transformer generator for kinsim_NN.

Inputs (conditioning, per token):
    base_fwd_onehot  (B, K, 4)
    base_rev_onehot  (B, K, 4)
    meth_fwd_onehot  (B, K, M)
    meth_rev_onehot  (B, K, M)

Latent noise:
    z                (B, z_dim)

Output:
    signal           (B, K, 4)   log1p(frames) space, channels = (IPD_fwd, PW_fwd, IPD_rev, PW_rev)

Architecture: K=21 tokens × `n_layers` AdaLN-Zero blocks. Conditioning
vector ``z_emb`` is computed once at the top (z + mean-pooled token
features), then injected into every block as AdaLN modulation. This
way z influences every layer and the model can carry stochasticity all
the way to the output.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AdaLNZeroBlock


class TransformerGenerator(nn.Module):
    def __init__(
        self,
        k: int = 21,
        n_meth_types: int = 4,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 6,
        z_dim: int = 64,
        pos_embed_dim: int = 16,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.n_meth_types = n_meth_types
        self.d_model = d_model
        self.z_dim = z_dim

        # Per-token feature dimensions (concat then project)
        n_base = 4
        in_features = (
            n_base                 # base_fwd
            + n_base               # base_rev
            + n_meth_types         # meth_fwd
            + n_meth_types         # meth_rev
            + pos_embed_dim
        )
        self.input_proj = nn.Linear(in_features, d_model, bias=True)

        # Learned positional embedding (K positions) — trunc-normal init.
        self.pos_embed = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(k, pos_embed_dim), std=0.02)
        )

        # z → conditioning vector for AdaLN
        # We give a fairly rich z embedding: sinusoidal of each z dim
        # then MLP to d_cond. Final d_cond = d_model so AdaLN modulation
        # has matching shape per block.
        d_cond = d_model
        self.z_mlp = nn.Sequential(
            nn.Linear(z_dim, d_cond),
            nn.SiLU(),
            nn.Linear(d_cond, d_cond),
        )

        # Optionally pool the condition tokens into z_emb so the
        # adaptive LN also sees the conditioning context. This breaks
        # the "z is the ONLY randomness" symmetry — desirable because
        # the per-position meth context should also modulate.
        self.cond_pool_proj = nn.Linear(d_model, d_cond)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(d_model, n_heads, d_cond, mlp_ratio=4.0, drop_rate=drop_rate)
            for _ in range(n_layers)
        ])

        # Final norm + head: per-token 4 channels
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 2 * d_model),
        )
        # AdaLN-Zero init for final modulation
        nn.init.zeros_(self.final_mod[1].weight)
        nn.init.zeros_(self.final_mod[1].bias)
        self.head = nn.Linear(d_model, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _build_tokens(
        self,
        base_fwd_onehot: torch.Tensor,
        base_rev_onehot: torch.Tensor,
        meth_fwd_onehot: torch.Tensor,
        meth_rev_onehot: torch.Tensor,
    ) -> torch.Tensor:
        B = base_fwd_onehot.size(0)
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)
        feats = torch.cat(
            [base_fwd_onehot, base_rev_onehot, meth_fwd_onehot, meth_rev_onehot, pos],
            dim=-1,
        )
        return self.input_proj(feats)  # (B, K, d_model)

    def forward(
        self,
        z: torch.Tensor,
        base_fwd_onehot: torch.Tensor,
        base_rev_onehot: torch.Tensor,
        meth_fwd_onehot: torch.Tensor,
        meth_rev_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = self._build_tokens(base_fwd_onehot, base_rev_onehot,
                               meth_fwd_onehot, meth_rev_onehot)  # (B, K, d)

        # z embedding + conditioning context pool
        z_e = self.z_mlp(z)                                       # (B, d_cond)
        cond_e = self.cond_pool_proj(x.mean(dim=1))               # (B, d_cond)
        cond_emb = z_e + cond_e                                   # (B, d_cond)

        for block in self.blocks:
            x = block(x, cond_emb)

        shift, scale = self.final_mod(cond_emb).chunk(2, dim=-1)
        x = self.final_norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.head(x)                                       # (B, K, 4)

    def sample_z(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Standard-normal noise sampler matching this G's z_dim."""
        return torch.randn(batch_size, self.z_dim, device=device)


__all__ = ["TransformerGenerator"]
