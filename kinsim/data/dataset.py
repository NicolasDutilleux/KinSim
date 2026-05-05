"""Dataset and signal-space transforms for KinSim training.

Training data format (produced by ``kinsim extract`` + ``kinsim merge`` + ``refine``):

    dict[kmer_id (int)] -> np.ndarray(N, 36)

with the column layout from ``kinsim.utils.sample_layout``:

    cols 0-1   IPD, PW           (raw uint8 from fi/fp BAM tags, [0, 255])
    col  2     fraction          (stoichiometric methylation fraction)
    cols 3-13  mc_0..mc_10       (per-position meth_id at offsets [-7..+3])
    cols 14-31 profile_IPD/PW    (downstream kinetic profile)
    cols 32-34 rev_meth          (complementary-strand meth at active-site)
    col  35    CATEGORY          (0=baseline, 1=slowed, 2=near_meth)

This module provides:

    log_transform(x)     map raw [0, 255] signals into log1p space for training
    inv_log_transform(x) recover raw uint8 [0, 255] from log1p (inference)
    MLPSignalDataset     flat-sample dataset returning
                         (kmer_id, meth_full, log_signal, meth_id) tuples,
                         with meth_full a Float[K, num_meth_types] tensor
                         encoding the per-position methylation context.

The model operates in log1p space during training and calls
``inv_log_transform`` at inference time to recover uint8 BAM values.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.encoding import K

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
    """Flat-sample dataset for KinSim training with per-position methylation context.

    Loads the merged .pkl, partitions every kmer's rows by ``meth_id`` at the
    prediction position (``mc[KMER_PRED_IDX]``), and pre-flattens all
    ``(kmer_id, meth_id, IPD, PW, fraction, meth_ctx)`` entries into
    contiguous arrays. The ``DataLoader`` shuffles the flat index each
    epoch, so every sample is seen exactly once per epoch in random order.

    Methylation context output
    --------------------------
    ``__getitem__`` returns ``meth_full``: a ``Float[K, num_meth_types]``
    tensor encoding the methylation state at each of the K=11 positions in
    the asymmetric meth-context window ``[-7, +3]`` around the prediction
    position. The prediction position sits at ``KMER_PRED_IDX = 7``.

    Encoding rules:
        - prediction position with meth_id = m: ``meth_full[7, m] = frac``
          (soft label using the stoichiometric fraction)
        - upstream/downstream position with meth_id = m:
          ``meth_full[pos, m] = 1.0`` (hard label)
        - any position with meth_id = 0: row is all zero

    Args:
        pkl_path:       Path to a merged .pkl produced by ``kinsim merge``.
        num_meth_types: Number of methylation states (default 4: none/m6A/m4C/m5C).
        kmer_size:      K-mer window size (default K=11).
    """

    def __init__(
        self,
        pkl_path: str,
        num_meth_types: int = 4,
        kmer_size: int = K,
    ) -> None:
        log.info("Loading training data from %s ...", pkl_path)
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        if not isinstance(data_dict, dict):
            raise TypeError(f"Expected a dict from {pkl_path}, got {type(data_dict).__name__}.")
        if len(data_dict) == 0:
            raise ValueError(f"The .pkl file is empty: {pkl_path}")

        # Probe a representative value to learn column count.
        first_val = None
        for k, v in data_dict.items():
            if k == "__meta__":
                continue
            first_val = v
            break
        if not isinstance(first_val, np.ndarray) or first_val.ndim != 2:
            raise TypeError(
                f"Expected dict values to be 2D ndarray, got shape "
                f"{getattr(first_val, 'shape', '?')}."
            )

        self._num_meth_types = num_meth_types
        self._kmer_size = kmer_size

        from ..utils.encoding import KMER_PRED_IDX

        pred_idx_in_ctx = KMER_PRED_IDX

        kmer_ids_list: list = []
        meth_ids_list: list = []
        signals_list: list = []
        fractions_list: list = []
        meth_ctx_list: list = []
        n_keys = 0
        n_meth_counts = defaultdict(int)

        for key, samples in data_dict.items():
            if key == "__meta__":
                continue
            if not isinstance(key, (int, np.integer)):
                continue
            kmer_id = int(key)
            ctx = samples[:, 3 : 3 + kmer_size].astype(np.uint8)
            meth_at_center = ctx[:, pred_idx_in_ctx].astype(np.int8)
            n = len(samples)
            kmer_ids_list.append(np.full(n, kmer_id, dtype=np.int32))
            meth_ids_list.append(meth_at_center)
            signals_list.append(samples[:, :2].astype(np.float32))
            fractions_list.append(samples[:, 2].astype(np.float32))
            meth_ctx_list.append(ctx)
            for mid in np.unique(meth_at_center):
                n_meth_counts[int(mid)] += int((meth_at_center == mid).sum())
            n_keys += 1

        self._kmer_ids = np.concatenate(kmer_ids_list)
        self._meth_ids = np.concatenate(meth_ids_list)
        # Pre-log-transform signals once at load time — avoids per-item overhead.
        self._signals = log_transform(
            torch.from_numpy(np.concatenate(signals_list, axis=0)).float()
        )
        self._fractions = np.concatenate(fractions_list)
        self._meth_ctx = np.concatenate(meth_ctx_list)

        n_total = len(self._kmer_ids)
        meth_labels = {0: "unmeth", 1: "m6A", 2: "m4C", 3: "m5C"}
        meth_summary = ", ".join(
            f"{meth_labels.get(m, str(m))}={n_meth_counts[m]:,}"
            for m in sorted(n_meth_counts)
            if n_meth_counts[m] > 0
        )
        log.info(
            "MLPSignalDataset ready: %s unique kmers, %s samples [%s]",
            f"{n_keys:,}",
            f"{n_total:,}",
            meth_summary,
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
              kmer_id   — Long scalar tensor (22-bit encoded 11-mer)
              meth_full — Float tensor of shape (K, num_meth_types):
                          per-position methylation encoding.
                          Center (pos K//2): soft label using stored fraction.
                          Flanking (all other positions): hard 0/1 one-hot.
                          Unmethylated positions: all-zero row.
              signal    — Float tensor of shape (2,): [IPD, PW] in log1p space
              meth_id   — Long scalar tensor (for per-type metrics)
        """
        kmer_id = int(self._kmer_ids[idx])
        meth_id = int(self._meth_ids[idx])
        signal = self._signals[idx]  # already log-transformed
        frac = float(self._fractions[idx])
        ctx_ids = self._meth_ctx[idx]  # (L,) uint8 meth IDs ([-7, +3])

        # Prediction position lives at index KMER_PRED_IDX in the asymmetric
        # context window — NOT at kmer_size // 2.
        from ..utils.encoding import KMER_PRED_IDX

        pred_idx = KMER_PRED_IDX
        meth_full = torch.zeros(self._kmer_size, self._num_meth_types, dtype=torch.float32)
        for pos in range(self._kmer_size):
            m = int(ctx_ids[pos])
            if m > 0:
                # Prediction position: soft label via stoichiometric fraction
                # Other positions: hard 1.0 (upstream/downstream modification)
                meth_full[pos, m] = frac if pos == pred_idx else 1.0

        return (
            torch.tensor(kmer_id, dtype=torch.long),
            meth_full,
            signal,
            torch.tensor(meth_id, dtype=torch.long),
        )
