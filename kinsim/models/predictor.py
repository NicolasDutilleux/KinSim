"""Models for kinetic signal prediction.

Two architectures are available:

MLPPredictor (legacy, architecture="mlp")
----------------------------------------------
Flat k-mer embedding table (4^11 = 4.2M entries).  Fast lookup, but treats
each 11-mer as an independent entity — 268M parameters, 99.98% in the
embedding table.  Cannot generalise to unseen k-mers or from unmethylated
to methylated contexts.

ConvPredictor (new, architecture="conv")
-------------------------------------------
Per-base embedding (4 bases x 16-dim) + 1D convolutions learn spatial
patterns across the 11-mer window.  FiLM conditioning from methylation
probabilities modulates the base representations at each position.

~140K parameters (~1900x smaller).  The model is forced to learn
compositional rules rather than memorising each 11-mer:
  - "A at offset -3 from the active site shifts IPD by X"
  - "m6A at the center amplifies signal by Y"

This is critical for the 95/5 unmethylated/methylated class imbalance:
the effect of methylation is learned as a global modulation rule, not
independently per k-mer.

Both models share the same external interface:
    forward(kmer_ids: Long[B], meth_probs: Float[B,K,M]) -> Float[B,4]
    sample(kmer_ids, meth_probs) -> Float[B,2]        (stochastic)
    predict_mean(kmer_ids, meth_probs) -> Float[B,2]  (deterministic)
    get_config() -> dict                               (for model_config.json)

``meth_probs`` is a Float[B, K, M] per-position methylation tensor where K is
the k-mer window size (default 11) and M is num_meth_types (default 4).
Legacy Float[B, M] center-only tensors are accepted for backward compatibility.

Output (all in log1p space):
    [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]

Use ``create_from_config(config_dict)`` to reconstruct a model from a
saved model_config.json.
"""

import torch
import torch.nn as nn

# Log-space transforms for training/inference signal conversion.
from ..data.dataset import inv_log_transform, log_transform  # noqa: F401
from ..utils.encoding import KMER_PRED_IDX
from ..utils.encoding import K as _DEFAULT_K
from ..utils.sample_layout import REV_METH_LEN as _REV_METH_LEN

# =========================================================================
# MLPPredictor (legacy architecture)
# =========================================================================


class MLPPredictor(nn.Module):
    """Flat k-mer embedding + MLP.  Legacy architecture (architecture="mlp").

    The 22-bit encoded 11-mer is looked up in a 4.2M-row embedding table,
    concatenated with a flattened per-position methylation projection, and fed
    through two hidden layers to predict Gaussian (mu, log_sigma) for IPD/PW.

    ``meth_probs`` is Float[B, K, M] (per-position, per-type).  It is flattened
    to Float[B, K*M] before projection.  Legacy Float[B, M] center-only inputs
    are also accepted (zero-padded at all non-center positions).

    Args:
        kmer_embed_dim: Embedding table column count (32 or 64).
        hidden_dim:     Width of the two hidden layers.
        meth_proj_dim:  Methylation linear projection output dim.
        dropout:        Dropout after each LeakyReLU (0.0 = disabled).
        num_meth_types: Number of methylation types M (default 4).
    """

    def __init__(
        self,
        kmer_embed_dim: int = 64,
        hidden_dim: int = 128,
        meth_proj_dim: int = 8,
        dropout: float = 0.0,
        kmer_size: int = _DEFAULT_K,
        num_meth_types: int = 4,
    ):
        super().__init__()

        self.kmer_embed_dim = kmer_embed_dim
        self.hidden_dim = hidden_dim
        self.meth_proj_dim = meth_proj_dim
        self.dropout = dropout
        self.kmer_size = kmer_size
        self.num_meth_types = num_meth_types

        _num_kmers = 4**kmer_size
        self.kmer_embed = nn.Embedding(_num_kmers, kmer_embed_dim)
        # Accepts the (kmer_size + rev_meth) × num_meth_types flat context
        # — forward meth at offsets [-7..+3] plus rev_meth at active-site
        # neighbours [-1, 0, +1] from sample_layout.REV_METH_LEN.
        self._meth_positions = kmer_size + _REV_METH_LEN
        self.meth_proj = nn.Linear(
            self._meth_positions * num_meth_types, meth_proj_dim, bias=False
        )

        input_dim = kmer_embed_dim + meth_proj_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        kmer_ids: torch.Tensor,
        meth_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            kmer_ids:   (B,) Long tensor.
            meth_probs: (B, K, M) per-position methylation, or (B, M) legacy
                        center-only (zero-padded to (B, K, M) internally).
        """
        kmer_emb = self.kmer_embed(kmer_ids)

        if meth_probs.dim() == 2:
            # Legacy (B, M): place at center position; pad rev_meth with 0
            B, M = meth_probs.shape
            full = torch.zeros(
                B, self._meth_positions, M, device=meth_probs.device, dtype=meth_probs.dtype
            )
            full[:, KMER_PRED_IDX, :] = meth_probs
            meth_flat = full.view(B, -1)
        else:
            # (B, total_pos, M) → flatten to (B, total_pos*M)
            meth_flat = meth_probs.reshape(meth_probs.shape[0], -1)

        meth_emb = self.meth_proj(meth_flat)
        x = torch.cat([kmer_emb, meth_emb], dim=1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        params = self.forward(kmer_ids, meth_probs)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
        sigma = torch.exp(log_sig)
        z = torch.randn_like(mu)
        return inv_log_transform(mu + sigma * z)

    @torch.no_grad()
    def predict_mean(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        params = self.forward(kmer_ids, meth_probs)
        return inv_log_transform(params[:, :2])

    def get_config(self) -> dict:
        """Return architecture config for model_config.json."""
        return {
            "architecture": "mlp",
            "kmer_size": self.kmer_size,
            "kmer_embed_dim": self.kmer_embed_dim,
            "hidden_dim": self.hidden_dim,
            "meth_proj_dim": self.meth_proj_dim,
            "dropout": self.dropout,
            "num_meth_types": self.num_meth_types,
        }


# =========================================================================
# ConvPredictor (new architecture)
# =========================================================================


class ConvPredictor(nn.Module):
    """1D-convolutional predictor with FiLM methylation conditioning.

    Replaces the flat 4.2M-row embedding table with per-base embeddings
    (4 x base_embed_dim) processed by a 1D-conv backbone.  This forces the
    model to learn compositional spatial rules — how each base at each
    offset from the active site (center of the 11-mer) influences kinetics.

    Methylation is injected via FiLM (Feature-wise Linear Modulation):
    at each position, the methylation probability vector is projected and
    used to produce scale (gamma) and shift (beta) that modulate the base
    embedding.  When methylation is zero, the modulation is identity —
    the model sees pure sequence context.  This is physically motivated:
    methylation changes *how* the polymerase interacts with a base, it
    doesn't replace the base identity.

    Architecture::

        bases (B, 11) int      -> Embedding(4, 16)   -> (B, 11, 16)
                                   + positional embed     (learnable)
        meth  (B, 11, M) float -> Linear(M, 8)       -> (B, 11, 8)
                                   -> FiLM gamma, beta -> modulate base

        (B, 16, 11)  -- Conv1d x3 (k=3, BN, GELU)  -> (B, conv_dim, 11)

        Readout:  center[:, :, 5]  ||  mean(dim=2)   -> (B, 2*conv_dim)

        Head:  Linear -> LayerNorm -> GELU -> Dropout -> Linear(4)
               [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]

    The external forward() signature is identical to MLPPredictor:
        forward(kmer_ids: Long[B], meth_probs: Float[B, M])
    Internally, kmer_ids are decoded to per-position bases, and center-only
    meth_probs are expanded to (B, 11, M).  When the data pipeline provides
    per-position methylation (B, 11, M), pass it directly via forward_positional().

    Args:
        base_embed_dim: Per-base embedding dimension (default 16).
        num_meth_types: Number of methylation types M (default 4; scalable to 50+).
        meth_proj_dim:  Methylation projection output dimension (default 8).
        conv_dim:       Channel width for conv layers (default 128).
        n_conv_layers:  Number of Conv1d layers (default 3).
        kernel_size:    Conv1d kernel size (default 3).
        head_dim:       Head hidden layer width (default 128).
        dropout:        Dropout probability in the head (default 0.1).
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
    ):
        super().__init__()

        self.base_embed_dim = base_embed_dim
        self.num_meth_types = num_meth_types
        self.meth_proj_dim = meth_proj_dim
        self.conv_dim = conv_dim
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.head_dim = head_dim
        self.dropout_p = dropout
        self.kmer_size = kmer_size

        # --- Per-base embedding: A=0, C=1, G=2, T=3 ---
        self.base_embed = nn.Embedding(4, base_embed_dim)

        # --- Learnable positional embedding (kmer_size positions) ---
        # Captures "distance from active site" effects: the center position
        # is where the polymerase incorporates; flanking bases contribute
        # stiffness, steric effects, unzipping energy.
        self.pos_embed = nn.Parameter(torch.zeros(1, kmer_size, base_embed_dim))

        # --- Methylation projection: GLOBAL embedding from per-position context.
        # Forward meth context [-7, +3] (kmer_size positions) plus rev_meth
        # at active-site neighbours [-1, 0, +1] (REV_METH_LEN positions) —
        # flattened to (B, (kmer_size + REV_METH_LEN) * M) and projected
        # to a single embedding. FiLM conditioning is decoupled from
        # per-position alignment, so the model sees both strands' meth
        # status simultaneously (handles bilateral palindromic sites).
        self._meth_positions = kmer_size + _REV_METH_LEN
        self.meth_proj = nn.Linear(
            self._meth_positions * num_meth_types, meth_proj_dim, bias=False
        )

        # --- FiLM conditioning: meth -> (gamma, beta) -> modulate base emb ---
        # x_modulated = (1 + gamma) * x_base + beta, broadcast over positions.
        # Zero-init ensures identity when methylation context is empty.
        self.film_gamma = nn.Linear(meth_proj_dim, base_embed_dim)
        self.film_beta = nn.Linear(meth_proj_dim, base_embed_dim)

        # --- Conv1d backbone ---
        # Learns local and medium-range spatial patterns.
        # Shared weights across positions (translation equivariance).
        conv_layers: list[nn.Module] = []
        in_ch = base_embed_dim
        for _ in range(n_conv_layers):
            conv_layers.extend(
                [
                    nn.Conv1d(in_ch, conv_dim, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(conv_dim),
                    nn.GELU(),
                ]
            )
            in_ch = conv_dim
        self.conv = nn.Sequential(*conv_layers)

        # --- Readout: center position (active site) + global average pool ---
        # Center captures local context at the incorporation site.
        # Global pool captures long-range effects from the full 11-mer.
        readout_dim = conv_dim * 2

        # --- Output head ---
        self.head = nn.Sequential(
            nn.Linear(readout_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 4),  # [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]
        )

        # --- Bit-shift buffer for decoding kmer_ids -> per-position bases ---
        # Registered as buffer so it moves with .to(device) automatically.
        self.register_buffer(
            "_shifts",
            torch.arange(kmer_size - 1, -1, -1) * 2,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Careful init: Kaiming for conv, Xavier for linear, zero for FiLM."""
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

        # FiLM zero-init: identity when no methylation
        # (1 + 0) * base + 0 = base
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        # Positional embedding: small normal
        nn.init.normal_(self.pos_embed, std=0.02)

    # ------------------------------------------------------------------
    # Input decoding helpers
    # ------------------------------------------------------------------

    def _decode_kmer_ids(self, kmer_ids: torch.Tensor) -> torch.Tensor:
        """Decode packed k-mer IDs to per-position base indices.

        Args:
            kmer_ids: (B,) Long tensor of 2*kmer_size-bit encoded k-mers.

        Returns:
            (B, kmer_size) Long tensor of base indices [0-3].
        """
        return (kmer_ids.unsqueeze(1) >> self._shifts.unsqueeze(0)) & 3

    def _expand_center_meth(self, meth_probs: torch.Tensor) -> torch.Tensor:
        """Expand center-only methylation probs to the full meth tensor.

        Backwards-compat shim for the legacy (B, M) input shape: places
        the centre meth at the prediction position and zeros everywhere
        else (including the rev_meth slots at the tail).
        """
        B, M = meth_probs.shape
        full = torch.zeros(
            B, self._meth_positions, M, device=meth_probs.device, dtype=meth_probs.dtype
        )
        full[:, KMER_PRED_IDX, :] = meth_probs
        return full

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _forward_conv(
        self,
        bases: torch.Tensor,
        meth_full: torch.Tensor,
    ) -> torch.Tensor:
        """Core forward pass with decoded per-position inputs.

        Args:
            bases:     (B, 11) Long tensor of base indices [0-3] — forward kmer.
            meth_full: (B, 11+REV_METH_LEN, M) per-position methylation probs.

        Returns:
            (B, 4) Float tensor: [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw].
        """
        # Per-base embedding + positional encoding
        x = self.base_embed(bases) + self.pos_embed  # (B, 11, base_embed_dim)

        # FiLM conditioning: GLOBAL meth context (forward + rev_meth) modulates
        # the kmer uniformly. Flatten per-position meth (B, 14, M) → (B, 14*M),
        # project to a single embedding, derive (gamma, beta), broadcast.
        # When meth_full is all zeros -> meth_feat=0 -> gamma=0, beta=0 -> identity.
        meth_flat = meth_full.reshape(meth_full.shape[0], -1)  # (B, 11*M)
        meth_feat = self.meth_proj(meth_flat)  # (B, meth_proj_dim)
        gamma = self.film_gamma(meth_feat).unsqueeze(1)  # (B, 1, base_embed_dim)
        beta = self.film_beta(meth_feat).unsqueeze(1)  # (B, 1, base_embed_dim)
        x = (1.0 + gamma) * x + beta

        # Conv1D expects (B, C, L)
        x = x.transpose(1, 2)  # (B, base_embed_dim, 11)
        x = self.conv(x)  # (B, conv_dim, 11)

        # Dual readout: center (active site) + global context
        center = x[:, :, KMER_PRED_IDX]  # (B, conv_dim)
        global_pool = x.mean(dim=2)  # (B, conv_dim)
        readout = torch.cat([center, global_pool], dim=1)  # (B, 2*conv_dim)

        return self.head(readout)  # (B, 4)

    # ------------------------------------------------------------------
    # Public forward (compatible with existing pipeline)
    # ------------------------------------------------------------------

    def forward(
        self,
        kmer_ids: torch.Tensor,
        meth_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Gaussian parameters from packed k-mer IDs and methylation.

        Compatible with the existing pipeline: dataset, training, generation.
        kmer_ids are decoded internally to per-position bases; center-only
        meth_probs are expanded to the full (B, 11, M) tensor.

        Args:
            kmer_ids:   (B,) Long tensor of 22-bit encoded 11-mers.
            meth_probs: (B, M) or (B, 11, M) methylation probability tensor.
                        (B, M) = center-only (current pipeline), expanded internally.
                        (B, 11, M) = per-position (future pipeline), used directly.

        Returns:
            (B, 4): [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw] in log1p space.
        """
        bases = self._decode_kmer_ids(kmer_ids)
        if meth_probs.dim() == 2:
            meth_full = self._expand_center_meth(meth_probs)
        else:
            meth_full = meth_probs  # already (B, 11, M)
        return self._forward_conv(bases, meth_full)

    def forward_positional(
        self,
        bases: torch.Tensor,
        meth_full: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with explicit per-position inputs (future pipeline).

        Args:
            bases:     (B, 11) Long tensor of base indices [0-3].
            meth_full: (B, 11, M) Float tensor of per-position methylation probs.

        Returns:
            (B, 4): [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw] in log1p space.
        """
        return self._forward_conv(bases, meth_full)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        """Sample (IPD, PW) from the predicted Gaussian.  Stochastic."""
        params = self.forward(kmer_ids, meth_probs)
        mu = params[:, :2]
        log_sig = torch.clamp(params[:, 2:], -6.0, 3.0)
        sigma = torch.exp(log_sig)
        z = torch.randn_like(mu)
        return inv_log_transform(mu + sigma * z)

    @torch.no_grad()
    def predict_mean(self, kmer_ids: torch.Tensor, meth_probs: torch.Tensor) -> torch.Tensor:
        """Return predicted mean (IPD, PW) without sampling.  Deterministic."""
        params = self.forward(kmer_ids, meth_probs)
        return inv_log_transform(params[:, :2])

    def get_config(self) -> dict:
        """Return architecture config for model_config.json."""
        return {
            "architecture": "conv",
            "kmer_size": self.kmer_size,
            "base_embed_dim": self.base_embed_dim,
            "num_meth_types": self.num_meth_types,
            "meth_proj_dim": self.meth_proj_dim,
            "conv_dim": self.conv_dim,
            "n_conv_layers": self.n_conv_layers,
            "kernel_size": self.kernel_size,
            "head_dim": self.head_dim,
            "dropout": self.dropout_p,
        }


# =========================================================================
# Factory: reconstruct model from model_config.json
# =========================================================================


def create_from_config(config: dict) -> nn.Module:
    """Reconstruct a model from a model_config.json dict.

    Detects the architecture key and creates the right class.
    Old configs without an "architecture" key default to "mlp".

    Args:
        config: Dict loaded from model_config.json.

    Returns:
        MLPPredictor or ConvPredictor instance (uninitialised weights).
    """
    arch = config.get("architecture", "mlp")

    kmer_size = config.get("kmer_size", _DEFAULT_K)

    if arch == "mlp":
        return MLPPredictor(
            kmer_embed_dim=config.get("kmer_embed_dim", 64),
            hidden_dim=config.get("hidden_dim", 128),
            meth_proj_dim=config.get("meth_proj_dim", 8),
            dropout=config.get("dropout", 0.0),
            kmer_size=kmer_size,
            num_meth_types=config.get("num_meth_types", 4),
        )
    elif arch == "conv":
        return ConvPredictor(
            base_embed_dim=config.get("base_embed_dim", 16),
            num_meth_types=config.get("num_meth_types", 4),
            meth_proj_dim=config.get("meth_proj_dim", 8),
            conv_dim=config.get("conv_dim", 128),
            n_conv_layers=config.get("n_conv_layers", 3),
            kernel_size=config.get("kernel_size", 3),
            head_dim=config.get("head_dim", 128),
            dropout=config.get("dropout", 0.1),
            kmer_size=kmer_size,
        )
    else:
        raise ValueError(
            f"Unknown architecture '{arch}' in model_config.json. Expected 'mlp' or 'conv'."
        )
