"""Bilateral ConvPredictor — predicts (ipd_fwd, pw_fwd, ipd_rev, pw_rev) jointly.

Forward kmer + reverse kmer (revcomp, derived) both embedded; 2 strand-
specific conv backbones learn sequence motifs unconditioned on
methylation; cross-meth FiLM modulators (both meth contexts feed both
branches) then shift/scale the conv FEATURE maps; merged readout ->
8 outputs (4 mu + 4 log sigma).

Outputs are reference-strand-relative; ``generate.py`` routes them back
to BAM fi/fp/ri/rp tags based on ``read.is_reverse``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..data.dataset import inv_log_transform
from ..utils.encoding import KMER_PRED_IDX
from ..utils.encoding import K as _DEFAULT_K

log = logging.getLogger(__name__)


_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def _build_meth_compat_buffer(num_meth_types: int) -> torch.Tensor:
    """Return ``(4, num_meth_types)`` ``compat[template_base, meth_id]``."""
    compat = torch.zeros(4, num_meth_types, dtype=torch.float32)
    compat[:, 0] = 1.0
    try:
        from ..utils.config import get_modified_base_map
        from ..utils.encoding import get_meth_ids
        base_map = get_modified_base_map()
        meth_id_map = get_meth_ids()
        for mname, base in base_map.items():
            mid = int(meth_id_map.get(mname, 0))
            bid = _BASE_TO_ID.get(str(base).upper())
            if mid > 0 and 0 <= mid < num_meth_types and bid is not None:
                compat[bid, mid] = 1.0
    except (ImportError, OSError, ValueError, KeyError) as exc:
        log.warning("biology_mask: could not build from YAML (%s) — using all-ones.", exc)
        compat.fill_(1.0)
    return compat


class ConvPredictor(nn.Module):
    """Bilateral 1D-conv predictor with cross-meth FiLM."""

    def __init__(
        self,
        base_embed_dim: int = 16,
        num_meth_types: int = 4,
        meth_proj_dim: int = 8,
        conv_dim: int = 128,
        n_conv_layers: int = 3,
        kernel_size: int = 3,
        head_dim: int = 128,
        dropout: float = 0.1,
        kmer_size: int = _DEFAULT_K,
        kmer_aware_film: bool = True,
        biology_mask: bool = False,
        log_sigma_clamp_min: float = -6.0,
        log_sigma_clamp_max: float = 1.5,
        active_site_index: int | None = None,
    ):
        super().__init__()
        if active_site_index is None:
            active_site_index = KMER_PRED_IDX
        if not 0 <= active_site_index < kmer_size:
            raise ValueError(
                f"ConvPredictor: active_site_index ({active_site_index}) "
                f"must satisfy 0 <= idx < kmer_size ({kmer_size})."
            )

        self.base_embed_dim = base_embed_dim
        self.num_meth_types = num_meth_types
        self.meth_proj_dim = meth_proj_dim
        self.conv_dim = conv_dim
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.head_dim = head_dim
        self.dropout_p = dropout
        self.kmer_size = kmer_size
        self.active_site_index = int(active_site_index)
        self.kmer_aware_film = kmer_aware_film
        self.biology_mask = bool(biology_mask)
        self.log_sigma_clamp_min = float(log_sigma_clamp_min)
        self.log_sigma_clamp_max = float(log_sigma_clamp_max)

        self.base_embed = nn.Embedding(4, base_embed_dim)
        self.pos_embed_fwd = nn.Parameter(torch.zeros(1, kmer_size, base_embed_dim))
        self.pos_embed_rev = nn.Parameter(torch.zeros(1, kmer_size, base_embed_dim))

        meth_input_dim = 2 * kmer_size * num_meth_types
        self.meth_proj = nn.Linear(meth_input_dim, meth_proj_dim, bias=False)
        # FiLM modulates the conv backbone OUTPUT, not the input embedding —
        # conv first learns sequence motifs unconditionally, then meth context
        # shifts/scales the high-level features (standard FiLM placement,
        # Perez et al. 2018). film_gamma/beta therefore output ``conv_dim``
        # channels (was ``base_embed_dim`` in the pre-conv variant).
        # kmer-aware FiLM concatenates the pooled conv output (also conv_dim).
        film_in_dim = meth_proj_dim + (conv_dim if kmer_aware_film else 0)
        self.film_gamma_fwd = nn.Linear(film_in_dim, conv_dim)
        self.film_beta_fwd = nn.Linear(film_in_dim, conv_dim)
        self.film_gamma_rev = nn.Linear(film_in_dim, conv_dim)
        self.film_beta_rev = nn.Linear(film_in_dim, conv_dim)

        def _make_conv_stack() -> nn.Sequential:
            layers: list[nn.Module] = []
            in_ch = base_embed_dim
            for _ in range(n_conv_layers):
                layers.extend([
                    nn.Conv1d(in_ch, conv_dim, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(conv_dim),
                    nn.GELU(),
                ])
                in_ch = conv_dim
            return nn.Sequential(*layers)

        self.conv_fwd = _make_conv_stack()
        self.conv_rev = _make_conv_stack()

        readout_dim = 4 * conv_dim  # center_fwd + mean_fwd + center_rev + mean_rev
        self.head = nn.Sequential(
            nn.Linear(readout_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 8),
        )

        self.register_buffer(
            "_shifts",
            torch.arange(kmer_size - 1, -1, -1) * 2,
        )
        self.register_buffer(
            "_meth_compat",
            _build_meth_compat_buffer(num_meth_types),
            persistent=False,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for layer in (
            self.film_gamma_fwd, self.film_beta_fwd,
            self.film_gamma_rev, self.film_beta_rev,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.pos_embed_fwd, std=0.02)
        nn.init.normal_(self.pos_embed_rev, std=0.02)

    def _decode_kmer_ids(self, kmer_ids: torch.Tensor) -> torch.Tensor:
        return (kmer_ids.unsqueeze(1) >> self._shifts.unsqueeze(0)) & 3

    @staticmethod
    def _revcomp_bases(bases_fwd: torch.Tensor) -> torch.Tensor:
        """Reverse-complement (B, K) base int tensor: complement then flip."""
        return (bases_fwd ^ 3).flip(dims=[1])

    def _forward_conv(
        self,
        kmer_ids: torch.Tensor,
        meth_ctx_fwd: torch.Tensor,
        meth_ctx_rev: torch.Tensor,
    ) -> torch.Tensor:
        bases_fwd = self._decode_kmer_ids(kmer_ids)
        bases_rev = self._revcomp_bases(bases_fwd)

        if self.biology_mask:
            template_fwd = bases_fwd ^ 3
            template_rev = bases_rev ^ 3
            compat_fwd = self._meth_compat[template_fwd]
            compat_rev = self._meth_compat[template_rev]
            meth_ctx_fwd = meth_ctx_fwd * compat_fwd
            meth_ctx_rev = meth_ctx_rev * compat_rev

        x_fwd = self.base_embed(bases_fwd) + self.pos_embed_fwd
        x_rev = self.base_embed(bases_rev) + self.pos_embed_rev

        # Conv on unconditioned embeddings — backbone learns sequence motifs
        # independently of methylation. FiLM (below) modulates the high-level
        # features.
        h_fwd = self.conv_fwd(x_fwd.transpose(1, 2))  # (B, conv_dim, K)
        h_rev = self.conv_rev(x_rev.transpose(1, 2))  # (B, conv_dim, K)

        meth_concat = torch.cat([meth_ctx_fwd, meth_ctx_rev], dim=1)
        meth_flat = meth_concat.reshape(meth_concat.shape[0], -1)
        meth_feat = self.meth_proj(meth_flat)

        if self.kmer_aware_film:
            film_in_fwd = torch.cat([meth_feat, h_fwd.mean(dim=2)], dim=-1)
            film_in_rev = torch.cat([meth_feat, h_rev.mean(dim=2)], dim=-1)
        else:
            film_in_fwd = meth_feat
            film_in_rev = meth_feat

        # Broadcast (B, conv_dim) -> (B, conv_dim, 1) so FiLM applies the same
        # scale/shift at every K position of the conv feature map.
        gamma_fwd = self.film_gamma_fwd(film_in_fwd).unsqueeze(2)
        beta_fwd = self.film_beta_fwd(film_in_fwd).unsqueeze(2)
        gamma_rev = self.film_gamma_rev(film_in_rev).unsqueeze(2)
        beta_rev = self.film_beta_rev(film_in_rev).unsqueeze(2)

        h_fwd = (1.0 + gamma_fwd) * h_fwd + beta_fwd
        h_rev = (1.0 + gamma_rev) * h_rev + beta_rev

        center_fwd = h_fwd[:, :, self.active_site_index]
        pool_fwd = h_fwd.mean(dim=2)
        center_rev = h_rev[:, :, self.active_site_index]
        pool_rev = h_rev.mean(dim=2)
        readout = torch.cat([center_fwd, pool_fwd, center_rev, pool_rev], dim=1)
        return self.head(readout)

    def forward(
        self,
        kmer_ids: torch.Tensor,
        meth_ctx_fwd: torch.Tensor,
        meth_ctx_rev: torch.Tensor,
    ) -> torch.Tensor:
        """Predict bilateral Gaussian parameters.

        Args:
            kmer_ids:     (B,) Long — forward-strand kmer IDs (2K-bit encoded).
            meth_ctx_fwd: (B, K, M) — + strand meth context per position.
            meth_ctx_rev: (B, K, M) — - strand meth context per position.

        Returns:
            (B, 8): [μ_fi, μ_fp, μ_ri, μ_rp, logσ_fi, logσ_fp, logσ_ri, logσ_rp].
        """
        if meth_ctx_fwd.dim() != 3 or meth_ctx_rev.dim() != 3:
            raise ValueError(
                f"meth contexts must be 3-D (B, K, M); got "
                f"{tuple(meth_ctx_fwd.shape)}, {tuple(meth_ctx_rev.shape)}."
            )
        return self._forward_conv(kmer_ids, meth_ctx_fwd, meth_ctx_rev)

    @torch.no_grad()
    def sample(
        self,
        kmer_ids: torch.Tensor,
        meth_ctx_fwd: torch.Tensor,
        meth_ctx_rev: torch.Tensor,
    ) -> torch.Tensor:
        """Sample (fi, fp, ri, rp) from the predicted Gaussian. (B, 4) raw uint8."""
        params = self.forward(kmer_ids, meth_ctx_fwd, meth_ctx_rev)
        mu = params[:, :4]
        log_sig = torch.clamp(params[:, 4:], self.log_sigma_clamp_min, self.log_sigma_clamp_max)
        sigma = torch.exp(log_sig)
        z = torch.randn_like(mu)
        return inv_log_transform(mu + sigma * z)

    @torch.no_grad()
    def predict_mean(
        self,
        kmer_ids: torch.Tensor,
        meth_ctx_fwd: torch.Tensor,
        meth_ctx_rev: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted means (fi, fp, ri, rp). (B, 4) in raw uint8 space."""
        params = self.forward(kmer_ids, meth_ctx_fwd, meth_ctx_rev)
        return inv_log_transform(params[:, :4])

    def get_config(self) -> dict:
        return {
            "kmer_size": self.kmer_size,
            "active_site_index": self.active_site_index,
            "base_embed_dim": self.base_embed_dim,
            "num_meth_types": self.num_meth_types,
            "meth_proj_dim": self.meth_proj_dim,
            "conv_dim": self.conv_dim,
            "n_conv_layers": self.n_conv_layers,
            "kernel_size": self.kernel_size,
            "head_dim": self.head_dim,
            "dropout": self.dropout_p,
            "kmer_aware_film": self.kmer_aware_film,
            "biology_mask": self.biology_mask,
            "log_sigma_clamp_min": self.log_sigma_clamp_min,
            "log_sigma_clamp_max": self.log_sigma_clamp_max,
            "architecture": "conv_bilateral_v2_postfilm",
        }


def create_from_config(config: dict) -> nn.Module:
    return ConvPredictor(
        base_embed_dim=config.get("base_embed_dim", 16),
        num_meth_types=config.get("num_meth_types", 4),
        meth_proj_dim=config.get("meth_proj_dim", 8),
        conv_dim=config.get("conv_dim", 128),
        n_conv_layers=config.get("n_conv_layers", 3),
        kernel_size=config.get("kernel_size", 3),
        head_dim=config.get("head_dim", 128),
        dropout=config.get("dropout", 0.1),
        kmer_size=config.get("kmer_size", _DEFAULT_K),
        kmer_aware_film=config.get("kmer_aware_film", True),
        biology_mask=config.get("biology_mask", False),
        log_sigma_clamp_min=config.get("log_sigma_clamp_min", -6.0),
        log_sigma_clamp_max=config.get("log_sigma_clamp_max", 1.5),
        active_site_index=config.get("active_site_index", KMER_PRED_IDX),
    )


def load_state_dict_from_ckpt(ckpt_path) -> dict:
    """Load a state_dict from a Lightning .ckpt or legacy .pt."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        return {
            k[len("model.") :]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")
        }
    if "model" in ckpt:
        return ckpt["model"]
    raise ValueError(
        f"Unrecognized checkpoint format in {ckpt_path}: "
        f"expected 'state_dict' (Lightning) or 'model' (legacy) key."
    )
