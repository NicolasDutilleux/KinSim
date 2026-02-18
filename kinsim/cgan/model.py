"""Conditional GAN model for kinetic signal generation.

Generator and Discriminator with conditional embeddings for 11-mer context
and methylation state. Uses WGAN-GP loss with gradient penalty.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Transform utilities
# ---------------------------------------------------------------------------

def log_transform(x):
    """Transform IPD/PW values to log1p space for training.

    Args:
        x: Raw signal values (typically in [0, 255])

    Returns:
        Log-transformed values: log(1 + x)
    """
    return torch.log1p(x)


def inv_log_transform(x):
    """Inverse transform from log1p space back to raw signals.

    Args:
        x: Log-transformed values

    Returns:
        Raw signals clamped to [0, 255]
    """
    return torch.clamp(torch.expm1(x), 0, 255)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Conditional generator for (IPD, PW) kinetic signals.

    Architecture:
        [z(noise_dim) + kmer_emb(kmer_embed_dim) + meth_emb(8)]
        → 128 (LayerNorm + LeakyReLU)
        → 128 (LayerNorm + LeakyReLU)
        → 2 (no activation — outputs in log1p space)

    Args:
        noise_dim: Dimension of input noise vector (default: 32)
        kmer_embed_dim: Dimension of 11-mer embedding (default: 64)
        hidden_dim: Hidden layer size (default: 128)
    """

    def __init__(self, noise_dim=32, kmer_embed_dim=64, hidden_dim=128):
        super().__init__()
        self.noise_dim = noise_dim
        self.kmer_embed_dim = kmer_embed_dim

        # Embeddings for conditional generation
        self.kmer_embed = nn.Embedding(4_194_304, kmer_embed_dim)  # 4^11 = 4,194,304
        self.meth_embed = nn.Embedding(4, 8)  # 0=unmethylated, 1=m6A, 2=m4C, 3=m5C

        input_dim = noise_dim + kmer_embed_dim + 8

        # Generator network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim, 2)  # Output: [IPD, PW] in log1p space
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, z, kmer_ids, meth_ids):
        """Generate kinetic signals conditioned on kmer and methylation.

        Args:
            z: Noise tensor of shape (batch_size, noise_dim)
            kmer_ids: Kmer integer IDs of shape (batch_size,)
            meth_ids: Methylation IDs of shape (batch_size,)

        Returns:
            Generated signals of shape (batch_size, 2) in log1p space
        """
        kmer_emb = self.kmer_embed(kmer_ids)
        meth_emb = self.meth_embed(meth_ids)

        # Concatenate noise + conditional embeddings
        x = torch.cat([z, kmer_emb, meth_emb], dim=1)

        return self.net(x)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """Conditional discriminator for WGAN-GP.

    Architecture:
        [signal(2) + kmer_emb(kmer_embed_dim) + meth_emb(8)]
        → 128 (LayerNorm + LeakyReLU + Dropout)
        → 128 (LayerNorm + LeakyReLU + Dropout)
        → 1 (no sigmoid — raw WGAN score)

    Args:
        kmer_embed_dim: Dimension of 11-mer embedding (default: 64)
        hidden_dim: Hidden layer size (default: 128)
        dropout: Dropout probability (default: 0.3)
    """

    def __init__(self, kmer_embed_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.kmer_embed_dim = kmer_embed_dim

        # Embeddings (separate from Generator — no weight sharing)
        self.kmer_embed = nn.Embedding(4_194_304, kmer_embed_dim)
        self.meth_embed = nn.Embedding(4, 8)

        input_dim = 2 + kmer_embed_dim + 8  # signal(2) + kmer + meth

        # Discriminator network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)  # Raw WGAN score (no sigmoid)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, signals, kmer_ids, meth_ids):
        """Discriminate real vs fake signals conditioned on context.

        Args:
            signals: Signal tensor of shape (batch_size, 2) in log1p space
            kmer_ids: Kmer integer IDs of shape (batch_size,)
            meth_ids: Methylation IDs of shape (batch_size,)

        Returns:
            WGAN scores of shape (batch_size, 1) — higher = more real
        """
        kmer_emb = self.kmer_embed(kmer_ids)
        meth_emb = self.meth_embed(meth_ids)

        # Concatenate signals + conditional embeddings
        x = torch.cat([signals, kmer_emb, meth_emb], dim=1)

        return self.net(x)
