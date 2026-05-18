"""ConvPredictor — 1D-conv + FiLM kinetic predictor.

Per-base embedding (4 bases x 16-dim) + 1D convolutions learn spatial
patterns across the kmer window.  FiLM conditioning from methylation
probabilities modulates the base representations at each position.

~140K parameters. The model is forced to learn compositional rules:
  - "A at offset -3 from the active site shifts IPD by X"
  - "m6A at the center amplifies signal by Y"

Output (all in log1p space):
    [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]

Use ``create_from_config(config_dict)`` to reconstruct from a saved
``model_config.json``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..data.dataset import inv_log_transform, log_transform  # noqa: F401
from ..utils.encoding import KMER_PRED_IDX
from ..utils.encoding import K as _DEFAULT_K
from ..utils.sample_layout import REV_METH_LEN as _REV_METH_LEN

log = logging.getLogger(__name__)


# =========================================================================
# Biology compatibility mask (architectural enforcement)
# =========================================================================
#
# Forbids the model from ever seeing an impossible (base, meth_id) pair —
# e.g. m5C on an A. The mask is multiplied into ``meth_full`` at the kmer
# positions before FiLM. Built from kinsim_config.yaml's
# ``kinetic_signatures.<name>.modified_base`` field.

_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def _build_meth_compat_buffer(num_meth_types: int) -> torch.Tensor:
    """Return a (4, num_meth_types) float buffer.

    ``compat[base_id, meth_id] = 1.0`` iff the meth can biologically
    occur on that base. ``meth_id = 0`` (none) is always compatible
    with every base.
    """
    compat = torch.zeros(4, num_meth_types, dtype=torch.float32)
    compat[:, 0] = 1.0  # "none" is always allowed on any base
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
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("biology_mask: could not build from YAML (%s) — "
                    "falling back to all-ones (no constraint).", exc)
        compat.fill_(1.0)
    return compat


# =========================================================================
# ConvPredictor
# =========================================================================


class ConvPredictor(nn.Module):
    """1D-convolutional predictor with FiLM methylation conditioning.

    Architecture::

        bases (B, K) int      -> Embedding(4, 16)   -> (B, K, 16)
                                   + positional embed     (learnable)
        meth  (B, K+R, M)     -> Linear((K+R)*M, 8) -> (B, 8)
                                   -> FiLM gamma, beta -> modulate base

        (B, 16, K)  -- Conv1d x3 (k=3, BN, GELU)  -> (B, conv_dim, K)

        Readout:  bases[:, :, active_site]  ||  mean(dim=2)
        Head:     Linear -> LayerNorm -> GELU -> Dropout -> Linear(4)

    Output (log1p space): [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]
    """

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
        biology_mask: bool = True,
        log_sigma_clamp_min: float = -6.0,
        log_sigma_clamp_max: float = 1.5,
        active_site_index: int | None = None,
        n_rev_meth: int = _REV_METH_LEN,
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
        self.n_rev_meth = int(n_rev_meth)
        self.kmer_aware_film = kmer_aware_film
        self.biology_mask = bool(biology_mask)
        self.log_sigma_clamp_min = float(log_sigma_clamp_min)
        self.log_sigma_clamp_max = float(log_sigma_clamp_max)

        self.base_embed = nn.Embedding(4, base_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, kmer_size, base_embed_dim))

        self._meth_positions = kmer_size + self.n_rev_meth
        self.meth_proj = nn.Linear(
            self._meth_positions * num_meth_types, meth_proj_dim, bias=False
        )

        film_in_dim = meth_proj_dim + base_embed_dim if kmer_aware_film else meth_proj_dim
        self.film_gamma = nn.Linear(film_in_dim, base_embed_dim)
        self.film_beta = nn.Linear(film_in_dim, base_embed_dim)

        conv_layers: list[nn.Module] = []
        in_ch = base_embed_dim
        for _ in range(n_conv_layers):
            conv_layers.extend([
                nn.Conv1d(in_ch, conv_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(conv_dim),
                nn.GELU(),
            ])
            in_ch = conv_dim
        self.conv = nn.Sequential(*conv_layers)

        readout_dim = conv_dim * 2
        self.head = nn.Sequential(
            nn.Linear(readout_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 4),
        )

        self.register_buffer(
            "_shifts",
            torch.arange(kmer_size - 1, -1, -1) * 2,
        )

        # Rebuilt from YAML at construction, not persisted.
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

        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _decode_kmer_ids(self, kmer_ids: torch.Tensor) -> torch.Tensor:
        return (kmer_ids.unsqueeze(1) >> self._shifts.unsqueeze(0)) & 3

    def _forward_conv(
        self,
        bases: torch.Tensor,
        meth_full: torch.Tensor,
    ) -> torch.Tensor:
        # Biology gate: zero impossible (template_base, meth_id) pairs.
        # Bases are SYNTHESIZED-strand; meth sits on the TEMPLATE — complement
        # via 2-bit XOR (A↔T, C↔G).
        if self.biology_mask:
            kmer_len = bases.shape[1]
            template_bases = bases ^ 3
            compat_at_pos = self._meth_compat[template_bases]
            meth_full = meth_full.clone()
            meth_full[:, :kmer_len, :] = meth_full[:, :kmer_len, :] * compat_at_pos

        x = self.base_embed(bases) + self.pos_embed

        meth_flat = meth_full.reshape(meth_full.shape[0], -1)
        meth_feat = self.meth_proj(meth_flat)
        if self.kmer_aware_film:
            kmer_summary = x.mean(dim=1)
            film_in = torch.cat([meth_feat, kmer_summary], dim=-1)
        else:
            film_in = meth_feat
        gamma = self.film_gamma(film_in).unsqueeze(1)
        beta = self.film_beta(film_in).unsqueeze(1)
        x = (1.0 + gamma) * x + beta

        x = x.transpose(1, 2)
        x = self.conv(x)

        center = x[:, :, self.active_site_index]
        global_pool = x.mean(dim=2)
        readout = torch.cat([center, global_pool], dim=1)

        return self.head(readout)

    def forward(
        self,
        kmer_ids: torch.Tensor,
        meth_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Gaussian parameters from packed k-mer IDs and methylation.

        Args:
            kmer_ids:   (B,) Long tensor of 2K-bit encoded kmers.
            meth_probs: (B, K+n_rev_meth, M) per-position methylation tensor.

        Returns:
            (B, 4): [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw] in log1p space.
        """
        if meth_probs.dim() != 3:
            raise ValueError(
                f"ConvPredictor expects meth_probs of shape (B, K+n_rev_meth, M); "
                f"got shape {tuple(meth_probs.shape)}."
            )
        bases = self._decode_kmer_ids(kmer_ids)
        return self._forward_conv(bases, meth_probs)

    def forward_positional(
        self,
        bases: torch.Tensor,
        meth_full: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with explicit per-position inputs."""
        return self._forward_conv(bases, meth_full)

    @torch.no_grad()
    def sample(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        """Sample (IPD, PW) from the predicted Gaussian. Stochastic."""
        params = self.forward(kmer_ids, meth_probs)
        mu = params[:, :2]
        log_sig = torch.clamp(
            params[:, 2:], self.log_sigma_clamp_min, self.log_sigma_clamp_max,
        )
        sigma = torch.exp(log_sig)
        z = torch.randn_like(mu)
        return inv_log_transform(mu + sigma * z)

    @torch.no_grad()
    def predict_mean(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        """Return predicted mean (IPD, PW) without sampling. Deterministic."""
        params = self.forward(kmer_ids, meth_probs)
        return inv_log_transform(params[:, :2])

    def get_config(self) -> dict:
        """Return architecture config for model_config.json."""
        return {
            "kmer_size":            self.kmer_size,
            "active_site_index":    self.active_site_index,
            "n_rev_meth":           self.n_rev_meth,
            "base_embed_dim":       self.base_embed_dim,
            "num_meth_types":       self.num_meth_types,
            "meth_proj_dim":        self.meth_proj_dim,
            "conv_dim":             self.conv_dim,
            "n_conv_layers":        self.n_conv_layers,
            "kernel_size":          self.kernel_size,
            "head_dim":             self.head_dim,
            "dropout":              self.dropout_p,
            "kmer_aware_film":      self.kmer_aware_film,
            "biology_mask":         self.biology_mask,
            "log_sigma_clamp_min":  self.log_sigma_clamp_min,
            "log_sigma_clamp_max":  self.log_sigma_clamp_max,
        }


def create_from_config(config: dict) -> nn.Module:
    """Reconstruct a ConvPredictor from a model_config.json dict."""
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
        biology_mask=config.get("biology_mask", True),
        log_sigma_clamp_min=config.get("log_sigma_clamp_min", -6.0),
        log_sigma_clamp_max=config.get("log_sigma_clamp_max", 1.5),
        active_site_index=config.get("active_site_index", KMER_PRED_IDX),
        n_rev_meth=config.get("n_rev_meth", _REV_METH_LEN),
    )


def load_state_dict_from_ckpt(ckpt_path) -> dict:
    """Load a ConvPredictor state_dict from a Lightning .ckpt or legacy .pt.

    - Lightning .ckpt: returns ``ckpt["state_dict"]`` with the ``"model."``
      prefix stripped (the Lightning wrapper exposes the inner model as
      ``self.model``).
    - Legacy .pt (pre-simplification training): returns ``ckpt["model"]``
      directly.

    Raises ``ValueError`` if neither key is present.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        return {
            k[len("model."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    if "model" in ckpt:
        return ckpt["model"]
    raise ValueError(
        f"Unrecognized checkpoint format in {ckpt_path}: "
        f"expected a 'state_dict' (Lightning .ckpt) or 'model' (legacy .pt) key."
    )
