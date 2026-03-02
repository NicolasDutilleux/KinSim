"""Shared dataset utilities for all neural KinSim modes (MLP, cGAN).

Every neural mode operates on the same training data format produced by
`kinsim data extract` and `kinsim data merge`:

    dict[(kmer_id: int, meth_id: int)] -> np.ndarray(N, 2)
    columns: [IPD, PW]  (raw uint8 values from fi/fp BAM tags)

This module provides:

  log_transform(x)     — map raw signals [0, 255] → log1p space for training
  inv_log_transform(x) — inverse: log1p → raw uint8 [0, 255]
  KmerSignalDataset    — PyTorch Dataset that loads a merged .pkl and returns
                         (kmer_id, meth_id, log_signal) triples

All models (Generator, MLPPredictor) operate in log1p space during training
and call inv_log_transform at inference time to recover uint8 BAM values.
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Signal-space transforms
# ---------------------------------------------------------------------------

def log_transform(x: torch.Tensor) -> torch.Tensor:
    """Transform raw IPD/PW values into log1p space for stable training.

    log1p compresses the long tail of PacBio signal distributions (values
    can range from 0 to 255) while keeping 0 mapped to 0.

    Args:
        x: Raw signal tensor with values in [0, 255].

    Returns:
        log(1 + x) — same shape as input.
    """
    return torch.log1p(x)


def inv_log_transform(x: torch.Tensor) -> torch.Tensor:
    """Inverse of log_transform: recover raw uint8 signals from log1p space.

    Args:
        x: Log-transformed signal tensor.

    Returns:
        expm1(x) clamped to [0, 255] — values safe to cast to uint8 for BAM tags.
    """
    return torch.clamp(torch.expm1(x), 0, 255)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class KmerSignalDataset(Dataset):
    """Dataset of raw (IPD, PW) samples keyed by (kmer_id, meth_id).

    Loads the output of `kinsim data merge` and flattens it into flat
    tensors for efficient mini-batch sampling.  Signals are log-transformed
    once at load time so the transform is not repeated each epoch.

    Args:
        pkl_path: Path to the merged .pkl file produced by `kinsim data merge`.
                  Structure: dict[(kmer_id, meth_id)] -> np.ndarray(N, 2).
    """

    def __init__(self, pkl_path: str) -> None:
        print(f"Loading training data from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        kmer_ids: list = []
        meth_ids: list = []
        signals:  list = []

        for (kmer_id, meth_id), samples in data_dict.items():
            n = len(samples)
            kmer_ids.extend([kmer_id] * n)
            meth_ids.extend([meth_id] * n)
            signals.append(samples)

        self.kmer_ids = torch.tensor(kmer_ids, dtype=torch.long)
        self.meth_ids = torch.tensor(meth_ids, dtype=torch.long)
        self.signals  = log_transform(
            torch.from_numpy(np.concatenate(signals, axis=0)).float()
        )

        print(f"Dataset loaded: {len(self):,} samples, "
              f"{len(data_dict):,} unique (kmer, meth) contexts")

    def __len__(self) -> int:
        return len(self.kmer_ids)

    def __getitem__(self, idx: int):
        return self.kmer_ids[idx], self.meth_ids[idx], self.signals[idx]
