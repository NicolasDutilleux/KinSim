"""Dataset and signal-space transforms for KinSim MLP training.

Training data format (produced by ``kinsim extract`` / ``kinsim merge``):

    dict[(kmer_id: int, meth_id: int)] -> np.ndarray(N, 14)
    columns: [IPD, PW, fraction, mc_0, mc_1, ..., mc_10]

IPD and PW are raw uint8 values from fi/fp BAM tags (range [0, 255]).
Column 2 is the stoichiometric methylation fraction.
Columns 3–13 are the per-position methylation IDs (0=none, 1=m6A, 2=m4C,
3=m5C) for each of the K=11 positions in the k-mer window.  The active site
is at position 5 (K//2).

Backward compatibility:
  - 3-column .pkl  [IPD, PW, fraction]: meth-context columns zero-padded.
  - 2-column .pkl  [IPD, PW]:           fraction synthesised + meth-context zero-padded.

This module provides:

  log_transform(x)     — map raw signals [0, 255] → log1p space for training
  inv_log_transform(x) — inverse: log1p → raw uint8 [0, 255]
  MLPSignalDataset     — MLP dataset: random-shot sampling with dynamic capping,
                         returns (kmer_id, meth_full, log_signal, meth_id) tuples
                         where meth_full is Float[K, num_meth_types] with per-position
                         one-hot methylation encoding.

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
    """Flat-sample dataset for MLP training with per-position methylation context.

    Loads the merged .pkl and pre-flattens all (kmer_id, meth_id, IPD, PW,
    fraction, meth_ctx) entries into contiguous arrays so that every sample is
    seen exactly once per epoch.  The DataLoader shuffles the flat index each
    epoch, giving the model full exposure to the training distribution.

    Dynamic capping prevents majority class bias at load time:
        meth_id = 0 (unmethylated) → keep at most max_unmeth samples (default 20)
        meth_id ∈ {1, 2, 3}        → keep at most max_meth  samples (default 100)

    Methylation context output
    --------------------------
    ``__getitem__`` returns ``meth_full``: a Float[L, num_meth_types] tensor
    encoding the methylation state at each of the L=11 positions in the
    asymmetric meth context window [-7, +3] around the prediction position.

    The PREDICTION position (where IPD/PW is measured) sits at index
    ``KMER_PRED_IDX = 7`` in this tensor — NOT at K//2.

    For a sample whose prediction position carries meth_id = 1 (m6A):
        meth_full[7, :] = [0, frac, 0, 0]   ← soft label at prediction pos

    For an upstream modification (m4C at offset -3 from prediction):
        meth_full[4, :] = [0, 0, 1.0, 0]    ← hard label at offset -3

    For positions with no modification:
        meth_full[pos, :] = [0, 0, 0, 0]    ← all-zero (no contribution)

    Args:
        pkl_path:       Path to a merged .pkl produced by `kinsim merge`.
                        Structure: dict[(kmer_id, meth_id)] -> np.ndarray(N, 2/3/14).
        max_unmeth:     Maximum samples kept for unmethylated contexts (default 20).
        max_meth:       Maximum samples kept for methylated contexts (default 100).
        num_meth_types: Number of methylation states (default 4: none/m6A/m4C/m5C).
        kmer_size:      K-mer window size (default K=11).
    """

    def __init__(
        self,
        pkl_path: str,
        max_unmeth: int = 20,
        max_meth:   int = 100,
        num_meth_types: int = 4,
        kmer_size: int = K,
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

        # Auto-detect v3 (tuple keys) vs v4 (int keys) format.
        fmt = "unknown"
        for k in data_dict:
            if k == "__meta__":
                continue
            if isinstance(k, tuple):
                fmt = "v3"
                break
            if isinstance(k, (int, np.integer)):
                fmt = "v4"
                break
        if fmt == "unknown":
            raise ValueError(f"No data keys (tuple or int) found in {pkl_path}")
        log.info("  Detected format: %s", fmt)

        # Probe a representative value to learn column count.
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
        n_cols = first_val.shape[1]
        has_fraction = n_cols >= 3
        has_meth_ctx = n_cols >= 3 + kmer_size
        has_category = n_cols >= 36   # v4 has explicit CATEGORY column at idx 35

        if not has_fraction:
            log.info("  .pkl has 2-column data (legacy format, no fraction).")
        if not has_meth_ctx:
            log.info("  .pkl has %d-column data (no meth-context columns).", n_cols)
        if has_category:
            log.info("  v4 CATEGORY column present at col 35.")
        # ──────────────────────────────────────────────────────────────────────

        self._num_meth_types = num_meth_types
        self._kmer_size      = kmer_size

        # Prediction position within meth_context — same for both formats.
        from ..utils.encoding import KMER_PRED_IDX
        pred_idx_in_ctx = KMER_PRED_IDX

        # ── Build flat arrays (all samples, capped per key) ───────────────────
        kmer_ids_list:   list = []
        meth_ids_list:   list = []
        signals_list:    list = []
        fractions_list:  list = []
        meth_ctx_list:   list = []

        n_subsampled  = 0
        n_keys        = 0
        n_meth_counts = defaultdict(int)

        for key, samples in data_dict.items():
            if key == "__meta__":
                continue

            # ----- Resolve (kmer_id, meth_id) from key + sample columns -----
            if fmt == "v3":
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                kmer_id, meth_id = int(key[0]), int(key[1])
                # v3 cap: per (kmer, meth) bucket
                cap = max_unmeth if meth_id == 0 else max_meth
                if len(samples) > cap:
                    rng = np.random.default_rng(seed=kmer_id ^ (meth_id << 22))
                    idx = rng.choice(len(samples), size=cap, replace=False)
                    samples = samples[idx]
                    n_subsampled += 1
                n = len(samples)
                meth_ids_list.append(np.full(n, meth_id, dtype=np.int8))
                if has_fraction:
                    fractions_list.append(samples[:, 2].astype(np.float32))
                else:
                    frac_val = 1.0 if meth_id > 0 else 0.0
                    fractions_list.append(np.full(n, frac_val, dtype=np.float32))
                kmer_ids_list.append(np.full(n, kmer_id, dtype=np.int32))
                signals_list.append(samples[:, :2].astype(np.float32))
                if has_meth_ctx:
                    meth_ctx_list.append(samples[:, 3:3 + kmer_size].astype(np.uint8))
                else:
                    meth_ctx_list.append(np.zeros((n, kmer_size), dtype=np.uint8))
                n_meth_counts[meth_id] += 1

            else:    # v4
                if not isinstance(key, (int, np.integer)):
                    continue
                kmer_id = int(key)
                # In v4, meth_id at center is encoded in mc[KMER_PRED_IDX].
                # Cap per kmer in a category-aware way: v4 extract already
                # capped baseline at n_baseline_per_kmer, so we leave its
                # samples alone here (full pass-through). Downstream code
                # can still resample if it wants.
                ctx = samples[:, 3:3 + kmer_size].astype(np.uint8)
                meth_at_center = ctx[:, pred_idx_in_ctx].astype(np.int8)
                n = len(samples)
                kmer_ids_list.append(np.full(n, kmer_id, dtype=np.int32))
                meth_ids_list.append(meth_at_center)
                signals_list.append(samples[:, :2].astype(np.float32))
                if has_fraction:
                    fractions_list.append(samples[:, 2].astype(np.float32))
                else:
                    fractions_list.append(np.zeros(n, dtype=np.float32))
                meth_ctx_list.append(ctx)
                # Bucket counts per inferred meth_id at center
                for mid in np.unique(meth_at_center):
                    n_meth_counts[int(mid)] += int((meth_at_center == mid).sum())

            n_keys += 1

        self._kmer_ids  = np.concatenate(kmer_ids_list)
        self._meth_ids  = np.concatenate(meth_ids_list)
        # Pre-log-transform signals once at load time — avoids per-item overhead
        self._signals   = log_transform(
            torch.from_numpy(np.concatenate(signals_list, axis=0)).float()
        )
        self._fractions = np.concatenate(fractions_list)
        self._meth_ctx  = np.concatenate(meth_ctx_list)   # (N_total, K) uint8

        n_total = len(self._kmer_ids)
        meth_labels = {0: "unmeth", 1: "m6A", 2: "m4C", 3: "m5C"}
        meth_summary = ", ".join(
            f"{meth_labels.get(m, str(m))}={n_meth_counts[m]:,}"
            for m in sorted(n_meth_counts)
            if n_meth_counts[m] > 0
        )
        if fmt == "v4":
            log.info(
                "MLPSignalDataset ready (v4): %s unique kmer keys [%s]\n"
                "  Total samples: %s  (no resampling — v4 extract caps applied "
                "upstream)\n  Fraction column: %s\n"
                "  Meth-context columns (11-pos): %s",
                f"{n_keys:,}", meth_summary, f"{n_total:,}",
                "from .pkl (stoichiometric)" if has_fraction else "default 0.0",
                "from .pkl" if has_meth_ctx else "zero-padded (legacy compat)",
            )
        else:
            log.info(
                "MLPSignalDataset ready (v3): %s unique (kmer, meth) keys [%s]\n"
                "  Total capped samples: %s  "
                "(%s keys subsampled to cap, "
                "caps: unmeth=%d, meth=%d)\n"
                "  Fraction column: %s\n"
                "  Meth-context columns (11-pos): %s",
                f"{n_keys:,}", meth_summary, f"{n_total:,}",
                f"{n_subsampled:,}", max_unmeth, max_meth,
                "from .pkl (stoichiometric)" if has_fraction else "synthesised (legacy compat)",
                "from .pkl" if has_meth_ctx else "zero-padded (legacy compat)",
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
        kmer_id    = int(self._kmer_ids[idx])
        meth_id    = int(self._meth_ids[idx])
        signal     = self._signals[idx]               # already log-transformed
        frac       = float(self._fractions[idx])
        ctx_ids    = self._meth_ctx[idx]              # (L,) uint8 meth IDs ([-7, +3])

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
