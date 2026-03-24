"""Dataset and signal-space transforms for KinSim MLP training.

Training data format (produced by ``kinsim extract`` / ``kinsim merge``):

    dict[(kmer_id: int, meth_id: int)] -> np.ndarray(N, 3)
    columns: [IPD, PW, fraction]

IPD and PW are raw uint8 values from fi/fp BAM tags (range [0, 255]).
The fraction column is the stoichiometric methylation fraction from the
motif source (e.g., PacBio motifs.csv 'fraction' column).  For motifs
without an explicit fraction, the value is 1.0 (fully methylated); for
unmethylated positions (meth_id = 0) the fraction is 0.0.

Backward compatibility: older .pkl files may have only 2 columns [IPD, PW].
The dataset class detects this and defaults the fraction to 1.0 for
methylated keys (meth_id > 0) and 0.0 for unmethylated keys (meth_id == 0).

This module provides:

  log_transform(x)     — map raw signals [0, 255] → log1p space for training
  inv_log_transform(x) — inverse: log1p → raw uint8 [0, 255]
  MLPSignalDataset     — MLP dataset: random-shot sampling with dynamic capping,
                         returns (kmer_id, meth_probs, log_signal) triples

The model operates in log1p space during training and calls
inv_log_transform at inference time to recover uint8 BAM values.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


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
# PyTorch Dataset — MLP (random-shot, dynamic capping, stoichiometric fractions)
# ---------------------------------------------------------------------------

class MLPSignalDataset(Dataset):
    """Flat-sample dataset for MLP training with stoichiometric soft labels.

    Loads the merged .pkl and pre-flattens all (kmer_id, meth_id, IPD, PW,
    fraction) entries into contiguous arrays so that every sample is seen
    exactly once per epoch.  The DataLoader shuffles the flat index each epoch,
    giving the model full exposure to the training distribution.

    Dynamic capping prevents majority class bias at load time:
        meth_id = 0 (unmethylated) → keep at most max_unmeth samples (default 20)
        meth_id ∈ {1, 2, 3}        → keep at most max_meth  samples (default 100)

    Stoichiometric soft labels
    --------------------------
    The methylation output is a Float[num_meth_types] probability vector built
    from the per-sample stoichiometric fraction stored in column 3 of the data.

    For a sample with meth_id = 1 (m6A) and fraction = 0.75:
        meth_probs = [0, 0.75, 0, 0]

    For an unmethylated sample (meth_id = 0, fraction = 0.0):
        meth_probs = [0, 0, 0, 0]

    Args:
        pkl_path:       Path to a merged .pkl produced by `kinsim merge`.
                        Structure: dict[(kmer_id, meth_id)] -> np.ndarray(N, 2 or 3).
        max_unmeth:     Maximum samples kept for unmethylated contexts (default 20).
        max_meth:       Maximum samples kept for methylated contexts (default 100).
        num_meth_types: Number of methylation states (default 4: none/m6A/m4C/m5C).
    """

    def __init__(
        self,
        pkl_path: str,
        max_unmeth: int = 20,
        max_meth:   int = 100,
        num_meth_types: int = 4,
    ) -> None:
        log.info("Loading training data from %s ...", pkl_path)
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        # ── validate pkl structure ─────────────────────────────────────────────
        if not isinstance(data_dict, dict):
            raise TypeError(
                f"Expected a dict from {pkl_path}, got {type(data_dict).__name__}.\n"
                "The .pkl must be produced by 'kinsim extract' + 'kinsim merge'."
            )
        if len(data_dict) == 0:
            raise ValueError(f"The .pkl file is empty: {pkl_path}")

        first_key, first_val = None, None
        for k, v in data_dict.items():
            if isinstance(k, tuple):
                first_key, first_val = k, v
                break
        if first_key is None:
            raise ValueError(f"No (kmer_id, meth_id) tuple keys found in {pkl_path}")
        if (not isinstance(first_key, tuple) or len(first_key) != 2
                or not all(isinstance(k, int) for k in first_key)):
            raise TypeError(
                f"Expected dict keys of type (int, int), got {type(first_key)}.\n"
                "Run 'kinsim extract' + 'kinsim merge' to produce the correct format."
            )
        if not isinstance(first_val, np.ndarray) or first_val.ndim != 2:
            raise TypeError(
                f"Expected dict values of shape (N, 2 or 3), got shape "
                f"{getattr(first_val, 'shape', '?')}.\n"
                "Each value must be an np.ndarray with columns [IPD, PW] or "
                "[IPD, PW, fraction]."
            )

        n_cols = first_val.shape[1]
        has_fraction = n_cols >= 3
        if not has_fraction:
            log.info("  .pkl has 2-column data (legacy format, no fraction). "
                     "Defaulting fraction=1.0 for methylated, 0.0 for unmethylated.")
        # ──────────────────────────────────────────────────────────────────────

        self._num_meth_types = num_meth_types

        # ── Build flat arrays (all samples, capped per key) ───────────────────
        kmer_ids_list:  list = []
        meth_ids_list:  list = []
        signals_list:   list = []
        fractions_list: list = []

        n_subsampled  = 0
        n_keys        = 0
        n_meth_counts = defaultdict(int)

        for key, samples in data_dict.items():
            if not isinstance(key, tuple):
                continue
            kmer_id, meth_id = key
            cap = max_unmeth if meth_id == 0 else max_meth
            if len(samples) > cap:
                rng = np.random.default_rng(seed=kmer_id ^ (meth_id << 22))
                idx = rng.choice(len(samples), size=cap, replace=False)
                samples = samples[idx]
                n_subsampled += 1

            n = len(samples)
            kmer_ids_list.append(np.full(n, kmer_id, dtype=np.int32))
            meth_ids_list.append(np.full(n, meth_id, dtype=np.int8))
            signals_list.append(samples[:, :2].astype(np.float32))
            if has_fraction:
                fractions_list.append(samples[:, 2].astype(np.float32))
            else:
                frac_val = 1.0 if meth_id > 0 else 0.0
                fractions_list.append(np.full(n, frac_val, dtype=np.float32))

            n_meth_counts[meth_id] += 1
            n_keys += 1

        self._kmer_ids  = np.concatenate(kmer_ids_list)
        self._meth_ids  = np.concatenate(meth_ids_list)
        # Pre-log-transform signals once at load time — avoids per-item overhead
        self._signals   = log_transform(
            torch.from_numpy(np.concatenate(signals_list, axis=0)).float()
        )
        self._fractions = np.concatenate(fractions_list)

        n_total = len(self._kmer_ids)
        meth_labels = {0: "unmeth", 1: "m6A", 2: "m4C", 3: "m5C"}
        meth_summary = ", ".join(
            f"{meth_labels[m]}={n_meth_counts[m]:,}"
            for m in sorted(n_meth_counts)
            if n_meth_counts[m] > 0
        )
        log.info(
            "MLPSignalDataset ready: %s unique (kmer, meth) keys [%s]\n"
            "  Total capped samples: %s  "
            "(%s keys subsampled to cap, "
            "caps: unmeth=%d, meth=%d)\n"
            "  Fraction column: %s",
            f"{n_keys:,}", meth_summary, f"{n_total:,}",
            f"{n_subsampled:,}", max_unmeth, max_meth,
            "from .pkl (stoichiometric)" if has_fraction else "synthesised (legacy compat)",
        )

    def __len__(self) -> int:
        """Total number of individual (IPD, PW) samples across all keys."""
        return len(self._kmer_ids)

    def __getitem__(self, idx: int):
        """Return the (IPD, PW) sample at flat index idx.

        The DataLoader shuffles indices each epoch, so all samples are seen
        exactly once per epoch in random order.

        Returns:
            Tuple of:
              kmer_id    — Long scalar tensor (22-bit encoded 11-mer)
              meth_probs — Float tensor of shape (num_meth_types,): stoichiometric
                           methylation vector built from the stored fraction.
                           e.g. m6A at 75% → [0, 0.75, 0, 0]
              signal     — Float tensor of shape (2,): [IPD, PW] in log1p space
              meth_id    — Long scalar tensor (for per-type metrics)
        """
        kmer_id = int(self._kmer_ids[idx])
        meth_id = int(self._meth_ids[idx])
        signal  = self._signals[idx]                  # already log-transformed
        frac    = float(self._fractions[idx])

        meth_probs = torch.zeros(self._num_meth_types, dtype=torch.float32)
        if meth_id > 0:
            meth_probs[meth_id] = frac

        return (
            torch.tensor(kmer_id, dtype=torch.long),
            meth_probs,
            signal,
            torch.tensor(meth_id, dtype=torch.long),
        )
