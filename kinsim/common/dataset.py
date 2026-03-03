"""Shared dataset utilities for all neural KinSim modes (MLP, cGAN).

Every neural mode operates on the same training data format produced by
`kinsim extract` and `kinsim merge`:

    dict[(kmer_id: int, meth_id: int)] -> np.ndarray(N, 2)
    columns: [IPD, PW]  (raw uint8 values from fi/fp BAM tags)

This module provides:

  log_transform(x)     — map raw signals [0, 255] → log1p space for training
  inv_log_transform(x) — inverse: log1p → raw uint8 [0, 255]
  KmerSignalDataset    — cGAN dataset: flattens all samples into flat tensors,
                         returns (kmer_id, meth_id, log_signal) triples
  MLPSignalDataset     — MLP dataset: random-shot sampling with dynamic capping,
                         returns (kmer_id, meth_probs, log_signal) triples

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


# ---------------------------------------------------------------------------
# MLP-specific dataset (random-shot, dynamic capping)
# ---------------------------------------------------------------------------

class MLPSignalDataset(Dataset):
    """Random-shot dataset for MLP training with dynamic per-context capping.

    Unlike KmerSignalDataset (which flattens all samples into a 1-D list),
    this dataset keeps the dict structure: each (kmer_id, meth_id) key maps
    to a pool of observed (IPD, PW) pairs capped at load time.

    __getitem__ picks ONE pair randomly from the pool for the requested key,
    so the model sees the full signal distribution over training epochs rather
    than memorising a fixed ordering.

    Dynamic capping prevents majority class bias:
        meth_id = 0 (unmethylated) → keep at most max_unmeth samples (default 20)
        meth_id ∈ {1, 2, 3}        → keep at most max_meth  samples (default 100)

    The methylation output is a one-hot Float[4] vector — not an integer ID —
    matching the input contract of MLPPredictor v2 (nn.Linear projection).
    This allows soft probabilities at inference time without changing the model.

    Args:
        pkl_path:   Path to a merged .pkl produced by `kinsim merge`.
                    Structure: dict[(kmer_id, meth_id)] -> np.ndarray(N, 2).
        max_unmeth: Maximum samples kept for unmethylated contexts (default 20).
        max_meth:   Maximum samples kept for methylated contexts (default 100).
    """

    def __init__(
        self,
        pkl_path: str,
        max_unmeth: int = 20,
        max_meth:   int = 100,
    ) -> None:
        print(f"Loading training data from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        # ── validate pkl structure ─────────────────────────────────────────────
        # Expected: dict[(kmer_id: int, meth_id: int)] → np.ndarray(N, 2)
        # Produced by: kinsim extract  +  kinsim merge
        if not isinstance(data_dict, dict):
            raise TypeError(
                f"Expected a dict from {pkl_path}, got {type(data_dict).__name__}.\n"
                "The .pkl must be produced by 'kinsim extract' + 'kinsim merge'."
            )
        if len(data_dict) == 0:
            raise ValueError(f"The .pkl file is empty: {pkl_path}")

        # Sample the first key to validate structure
        first_key, first_val = next(iter(data_dict.items()))
        if (not isinstance(first_key, tuple) or len(first_key) != 2
                or not all(isinstance(k, int) for k in first_key)):
            raise TypeError(
                f"Expected dict keys of type (int, int), got {type(first_key)}.\n"
                "Run 'kinsim extract' + 'kinsim merge' to produce the correct format."
            )
        if not isinstance(first_val, np.ndarray) or first_val.ndim != 2 or first_val.shape[1] != 2:
            raise TypeError(
                f"Expected dict values of shape (N, 2), got shape {getattr(first_val, 'shape', '?')}.\n"
                "Each value must be an np.ndarray with columns [IPD, PW]."
            )
        # ──────────────────────────────────────────────────────────────────────

        self._keys:  list = []   # list of (kmer_id, meth_id) tuples
        self._pools: list = []   # list of np.ndarray(N_capped, 2) float32

        n_subsampled = 0
        n_meth_counts = {0: 0, 1: 0, 2: 0, 3: 0}   # for the summary log

        for (kmer_id, meth_id), samples in data_dict.items():
            cap = max_unmeth if meth_id == 0 else max_meth
            if len(samples) > cap:
                # Seeded subsampling: deterministic for the same .pkl across restarts
                rng = np.random.default_rng(seed=kmer_id ^ (meth_id << 22))
                idx = rng.choice(len(samples), size=cap, replace=False)
                samples = samples[idx]
                n_subsampled += 1
            self._keys.append((kmer_id, meth_id))
            self._pools.append(samples.astype(np.float32))
            n_meth_counts[meth_id] = n_meth_counts.get(meth_id, 0) + 1

        n_keys  = len(self._keys)
        n_total = sum(len(p) for p in self._pools)
        meth_labels = {0: "unmeth", 1: "m6A", 2: "m4C", 3: "m5C"}
        meth_summary = ", ".join(
            f"{meth_labels[m]}={n_meth_counts[m]:,}"
            for m in sorted(n_meth_counts)
            if n_meth_counts[m] > 0
        )
        print(
            f"MLPSignalDataset ready: {n_keys:,} unique (kmer, meth) keys "
            f"[{meth_summary}]\n"
            f"  Total capped samples: {n_total:,}  "
            f"({n_subsampled:,} keys subsampled to cap, "
            f"caps: unmeth={max_unmeth}, meth={max_meth})"
        )

    def __len__(self) -> int:
        # One item per unique (kmer_id, meth_id) key.
        # Each epoch, __getitem__ draws a fresh random sample from each key's pool,
        # so the effective training set grows larger than n_keys over multiple epochs.
        return len(self._keys)

    def __getitem__(self, idx: int):
        """Return one random (IPD, PW) observation for the key at position idx.

        Called by DataLoader workers. Uses torch.randint (not np.random) so that
        PyTorch's automatic per-worker seeding produces independent draws across
        workers — avoids correlated samples when num_workers > 0.

        Returns:
            Tuple of:
              kmer_id    — Long scalar tensor (22-bit encoded 11-mer)
              meth_probs — Float tensor of shape (4,): one-hot methylation vector
                           [p_none, p_m6A, p_m4C, p_m5C]
              signal     — Float tensor of shape (2,): [IPD, PW] in log1p space
        """
        kmer_id, meth_id = self._keys[idx]
        pool = self._pools[idx]                       # (N_capped, 2) float32

        # torch.randint is seeded per-worker by PyTorch's DataLoader — safe with
        # num_workers > 0, unlike np.random which shares state across forked workers.
        row_idx = int(torch.randint(len(pool), (1,)).item())
        row = pool[row_idx]                           # (2,) float32

        # One-hot methylation vector [p_none, p_m6A, p_m4C, p_m5C]
        meth_probs = torch.zeros(4, dtype=torch.float32)
        meth_probs[meth_id] = 1.0

        signal = log_transform(torch.from_numpy(row))  # log1p([IPD, PW])

        return (
            torch.tensor(kmer_id, dtype=torch.long),
            meth_probs,
            signal,
        )
