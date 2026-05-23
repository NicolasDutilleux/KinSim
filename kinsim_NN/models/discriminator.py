"""Transformer discriminator (critic) for kinsim_NN.

Receives both the signal (real or generated) AND the conditioning
context. Outputs a single scalar (Wasserstein critic score).

Architecture: K=21 tokens + 1 prepended CLS token → `n_layers` plain
pre-LN transformer blocks with spectral norm on every Linear. Final
critic score = ``head(CLS_token_output)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import TransformerBlock, maybe_spectral_norm


class TransformerDiscriminator(nn.Module):
    def __init__(
        self,
        k: int = 21,
        n_meth_types: int = 4,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        pos_embed_dim: int = 16,
        spectral_norm: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.k = k
        self.n_meth_types = n_meth_types
        self.d_model = d_model

        n_base = 4
        in_features = (
            n_base                       # base_fwd
            + n_base                     # base_rev
            + n_meth_types               # meth_fwd
            + n_meth_types               # meth_rev
            + 4                          # signal channels (IPD_fwd, PW_fwd, IPD_rev, PW_rev)
            + pos_embed_dim
        )
        self.input_proj = maybe_spectral_norm(
            nn.Linear(in_features, d_model, bias=True), spectral_norm,
        )

        # Learned positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(k, pos_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CLS token (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio=4.0,
                             spectral_norm=spectral_norm, drop_rate=drop_rate)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = maybe_spectral_norm(nn.Linear(d_model, 1, bias=True), spectral_norm)

    def _build_tokens(
        self,
        base_fwd_onehot: torch.Tensor,
        base_rev_onehot: torch.Tensor,
        meth_fwd_onehot: torch.Tensor,
        meth_rev_onehot: torch.Tensor,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        B = base_fwd_onehot.size(0)
        pos = self.pos_embed.unsqueeze(0).expand(B, -1, -1)
        feats = torch.cat(
            [base_fwd_onehot, base_rev_onehot,
             meth_fwd_onehot, meth_rev_onehot,
             signal, pos],
            dim=-1,
        )
        return self.input_proj(feats)  # (B, K, d_model)

    def forward(
        self,
        signal: torch.Tensor,
        base_fwd_onehot: torch.Tensor,
        base_rev_onehot: torch.Tensor,
        meth_fwd_onehot: torch.Tensor,
        meth_rev_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = self._build_tokens(base_fwd_onehot, base_rev_onehot,
                               meth_fwd_onehot, meth_rev_onehot, signal)  # (B, K, d)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)                            # (B, 1, d)
        x = torch.cat([cls, x], dim=1)                                    # (B, K+1, d)
        for block in self.blocks:
            x = block(x)
        cls_out = self.final_norm(x[:, 0])                                # (B, d)
        return self.head(cls_out).squeeze(-1)                             # (B,)


__all__ = ["TransformerDiscriminator"]
