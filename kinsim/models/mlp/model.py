"""MLP model for kinetic signal prediction.

Architecture
------------
The predictor takes two integer inputs — an 11-mer encoded as a 22-bit integer
and a methylation state ID — and outputs the parameters of a Gaussian
distribution over (IPD, PW) for that context.

    11-mer (int)      → Embedding(4^11, kmer_embed_dim)  ─┐
    meth_state (int)  → Embedding(4, 8)                   ├─ concat
                                                           ↓
                     Linear(kmer_embed_dim + 8 → hidden)
                     LayerNorm  →  LeakyReLU(0.2)
                     Linear(hidden → hidden)
                     LayerNorm  →  LeakyReLU(0.2)
                     Linear(hidden → 4)
                           ↓
              [μ_ipd, μ_pw, log_σ_ipd, log_σ_pw]  (log1p space)

All values are in log1p space during training. At inference time, the model
samples from N(μ, σ²) then applies expm1 and clamps to [0, 255] to recover
raw uint8 signals compatible with PacBio fi/fp BAM tags.

Methylation state IDs (shared with cGAN mode):
    0 = unmethylated
    1 = m6A
    2 = m4C
    3 = m5C
"""

import torch
import torch.nn as nn

# Reuse log-space transforms from common — shared with cGAN.
from ...common.dataset import log_transform, inv_log_transform  # noqa: F401 (re-exported)

# Total number of 11-mers: 4^11 = 4,194,304
_NUM_KMERS = 4 ** 11

# Number of methylation state categories
_NUM_METH_STATES = 4


class MLPPredictor(nn.Module):
    """Conditional MLP that predicts Gaussian parameters for (IPD, PW).

    The model outputs four values per context:
        μ_ipd, μ_pw       — predicted mean of IPD and PW in log1p space
        log_σ_ipd, log_σ_pw — log standard deviation (learned variance)

    Learned variance is essential: different 11-mer contexts naturally have
    different signal spread (e.g., methylated sites vs. plain sequence).
    Using a fixed σ would produce overconfident or under-dispersed signals.

    Args:
        kmer_embed_dim: Dimension of the 11-mer embedding table.
                        Use 32 to halve memory (~0.5 GB), 64 for full quality.
        hidden_dim:     Width of the two hidden layers.
    """

    def __init__(self, kmer_embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.kmer_embed_dim = kmer_embed_dim
        self.hidden_dim = hidden_dim

        # Embedding tables
        # kmer_embed: maps 4^11 possible 11-mers to dense vectors
        self.kmer_embed = nn.Embedding(_NUM_KMERS, kmer_embed_dim)
        # meth_embed: maps {0,1,2,3} methylation states to dense vectors
        self.meth_embed = nn.Embedding(_NUM_METH_STATES, 8)

        input_dim = kmer_embed_dim + 8

        # Two hidden layers with normalisation and residual-friendly width
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            # Output: [μ_ipd, μ_pw, log_σ_ipd, log_σ_pw]
            nn.Linear(hidden_dim, 4),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers, small normal for embeddings."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        kmer_ids: torch.Tensor,
        meth_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict Gaussian parameters for a batch of contexts.

        Args:
            kmer_ids: Long tensor of shape (batch,) with 22-bit encoded 11-mers.
            meth_ids: Long tensor of shape (batch,) with methylation state IDs.

        Returns:
            Float tensor of shape (batch, 4):
                [:, 0] = μ_ipd      (mean IPD in log1p space)
                [:, 1] = μ_pw       (mean PW in log1p space)
                [:, 2] = log_σ_ipd  (log std-dev of IPD)
                [:, 3] = log_σ_pw   (log std-dev of PW)
        """
        kmer_emb = self.kmer_embed(kmer_ids)   # (batch, kmer_embed_dim)
        meth_emb = self.meth_embed(meth_ids)   # (batch, 8)
        x = torch.cat([kmer_emb, meth_emb], dim=1)
        return self.net(x)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        kmer_ids: torch.Tensor,
        meth_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample (IPD, PW) from the predicted Gaussian distribution.

        This is the standard inference mode: stochastic, matching the natural
        variability of real PacBio signals.

        Args:
            kmer_ids: Long tensor of shape (batch,).
            meth_ids: Long tensor of shape (batch,).

        Returns:
            Float tensor of shape (batch, 2) with raw [IPD, PW] in [0, 255].
        """
        params = self.forward(kmer_ids, meth_ids)   # (batch, 4)
        mu      = params[:, :2]                      # (batch, 2)
        log_sig = params[:, 2:]                      # (batch, 2)

        # Clamp log_σ for numerical stability (prevents σ → 0 or σ → ∞)
        log_sig = torch.clamp(log_sig, -6.0, 3.0)
        sigma = torch.exp(log_sig)

        # Reparameterisation: z ~ N(0,1), sample = μ + σ·z
        z = torch.randn_like(mu)
        sampled_log = mu + sigma * z

        return inv_log_transform(sampled_log)   # (batch, 2) in [0, 255]

    @torch.no_grad()
    def predict_mean(
        self,
        kmer_ids: torch.Tensor,
        meth_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return the predicted mean (IPD, PW) without sampling.

        Use this for deterministic generation (e.g., debugging, ablation).
        Produces the same signal for every read at the same context.

        Args:
            kmer_ids: Long tensor of shape (batch,).
            meth_ids: Long tensor of shape (batch,).

        Returns:
            Float tensor of shape (batch, 2) with raw [IPD, PW] in [0, 255].
        """
        params = self.forward(kmer_ids, meth_ids)
        mu = params[:, :2]
        return inv_log_transform(mu)
