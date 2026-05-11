"""Dataset and signal-space transforms for KinSim training.

Training data format (produced by ``kinsim extract`` and optionally
filtered by ``kinsim refine``):

    dict[kmer_id (int)] -> np.ndarray(N, SAMPLE_NCOLS)

with the column layout from ``kinsim.utils.sample_layout`` (currently
20 cols: IPD, PW, frac, mc[11], rev_meth[3], CATEGORY, PARENT_METH,
PARENT_OFFSET).

This module provides:

    log_transform(x)        map raw [0, 255] signals into log1p space for training
    inv_log_transform(x)    recover raw uint8 [0, 255] from log1p (inference)
    MLPSignalDataset        loads a single shard into RAM (small datasets / debugging)
    ShardedSignalDataset    PyTorch ``IterableDataset`` over a list of shard
                            pkls. Memory bounded by one shard regardless of
                            corpus size. Worker-aware (partitions shards across
                            DataLoader workers) and per-epoch shuffling at
                            both the shard level and the row level.

Both datasets emit
``(kmer_id, meth_full, log_signal, meth_id, parent_meth, parent_offset, category)``
tuples. ``meth_full`` has shape
``(kmer_size + REV_METH_LEN, num_meth_types)`` = ``(14, 4)`` — 11
forward-context entries followed by the 3 rev_meth entries at
active-site neighbours. The model's FiLM projection consumes the
flattened tensor, so all 14 positions contribute to ``(γ, β)``.

The last 3 ints (``parent_meth``, ``parent_offset``, ``category``) come
straight from the shard's 20-col layout (``kinsim.utils.sample_layout``)
and are used only in validation/test metric loops to bucket per
``(category, parent_meth, parent_offset)``. The model never sees them.
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

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
# Shared helpers (used by both MLPSignalDataset and ShardedSignalDataset)
# ---------------------------------------------------------------------------


def _flatten_data_dict(
    data_dict: dict, kmer_size: int, num_meth_types: int = 4,
) -> dict:
    """Flatten one ``dict[kmer_id] -> ndarray(N, SAMPLE_NCOLS)`` into per-row arrays.

    **Performance**: this is the hot path for shard loading. We
    vectorise the meth_full construction (forward block + rev_meth
    block) on the full shard at once via fancy-indexing instead of the
    per-row Python loop used in older code — ~50–100× faster on big
    shards.

    Returns a dict with:
      ``kmer_ids``       int32 (N,)
      ``meth_ids``       int8  (N,)        meth at the prediction position
      ``signals_log``    float32 tensor (N, 2)  already log1p
      ``meth_full``      float32 (N, kmer_size + REV_METH_LEN, num_meth_types)
                         pre-built so __iter__ just slices it
      ``parent_meths``   int8  (N,)        PARENT_METH (col 18)
      ``parent_offsets`` int8  (N,)        PARENT_OFFSET (col 19)
      ``categories``     int8  (N,)        CATEGORY (col 17)
      ``n_keys``, ``meth_counts``
    """
    from ..utils.encoding import KMER_PRED_IDX
    from ..utils.sample_layout import (
        COL_CATEGORY,
        COL_METH_CTX_START,
        COL_PARENT_METH,
        COL_PARENT_OFFSET,
        COL_REV_METH,
        REV_METH_LEN,
        SAMPLE_NCOLS,
    )

    pred_idx = KMER_PRED_IDX
    total_pos = kmer_size + REV_METH_LEN

    kmer_ids_list: list = []
    meth_ids_list: list = []
    signals_list: list = []
    fractions_list: list = []
    meth_ctx_list: list = []
    rev_meth_list: list = []
    parent_meth_list: list = []
    parent_offset_list: list = []
    category_list: list = []
    n_keys = 0
    n_meth_counts: dict[int, int] = defaultdict(int)

    for key, samples in data_dict.items():
        if key == "__meta__":
            continue
        if not isinstance(key, (int, np.integer)):
            continue
        if not isinstance(samples, np.ndarray) or samples.ndim != 2:
            continue
        if samples.shape[1] < SAMPLE_NCOLS:
            continue
        kmer_id = int(key)
        ctx = samples[:, COL_METH_CTX_START : COL_METH_CTX_START + kmer_size].astype(np.uint8)
        rev = samples[:, COL_REV_METH : COL_REV_METH + REV_METH_LEN].astype(np.uint8)
        meth_at_center = ctx[:, pred_idx].astype(np.int8)
        n = len(samples)
        kmer_ids_list.append(np.full(n, kmer_id, dtype=np.int32))
        meth_ids_list.append(meth_at_center)
        signals_list.append(samples[:, :2].astype(np.float32))
        fractions_list.append(samples[:, 2].astype(np.float32))
        meth_ctx_list.append(ctx)
        rev_meth_list.append(rev)
        parent_meth_list.append(samples[:, COL_PARENT_METH].astype(np.int8))
        parent_offset_list.append(samples[:, COL_PARENT_OFFSET].astype(np.int8))
        category_list.append(samples[:, COL_CATEGORY].astype(np.int8))
        for mid in np.unique(meth_at_center):
            n_meth_counts[int(mid)] += int((meth_at_center == mid).sum())
        n_keys += 1

    if not kmer_ids_list:
        return {
            "kmer_ids": np.empty(0, dtype=np.int32),
            "meth_ids": np.empty(0, dtype=np.int8),
            "signals_log": torch.empty(0, 2, dtype=torch.float32),
            "meth_full": np.empty((0, total_pos, num_meth_types), dtype=np.float32),
            "parent_meths": np.empty(0, dtype=np.int8),
            "parent_offsets": np.empty(0, dtype=np.int8),
            "categories": np.empty(0, dtype=np.int8),
            "n_keys": 0,
            "meth_counts": dict(n_meth_counts),
        }

    # Concatenate and *immediately* drop the per-key lists so we don't
    # hold both the lists AND the merged arrays at the same time
    # (each was ~half of peak memory before this fix).
    kmer_ids       = np.concatenate(kmer_ids_list);     kmer_ids_list = None
    meth_ids       = np.concatenate(meth_ids_list);     meth_ids_list = None
    fractions      = np.concatenate(fractions_list).astype(np.float32)
    fractions_list = None
    meth_ctx       = np.concatenate(meth_ctx_list);     meth_ctx_list = None
    rev_meth       = np.concatenate(rev_meth_list);     rev_meth_list = None
    parent_meths   = np.concatenate(parent_meth_list);  parent_meth_list = None
    parent_offsets = np.concatenate(parent_offset_list); parent_offset_list = None
    categories     = np.concatenate(category_list);     category_list = None
    signals_log    = log_transform(
        torch.from_numpy(np.concatenate(signals_list, axis=0)).float()
    )
    signals_list   = None

    # ── Vectorised meth_full construction (the big speedup) ────────────────
    # meth_full[i, pos, m] = 1.0 if mc/rev has meth m at pos, else 0.0
    # except: meth_full[i, pred_idx, m] = fractions[i] (stoichiometry).
    n_rows = kmer_ids.shape[0]
    meth_full = np.zeros((n_rows, total_pos, num_meth_types), dtype=np.float32)

    # Forward block: scatter meth_ctx → meth_full[:, 0:kmer_size, :]
    rows_idx, cols_idx = np.where(meth_ctx > 0)
    if rows_idx.size:
        m_ids = meth_ctx[rows_idx, cols_idx].astype(np.int64)
        # clamp to valid meth_id range to avoid IndexError on corrupted shards
        m_ids = np.clip(m_ids, 0, num_meth_types - 1)
        meth_full[rows_idx, cols_idx, m_ids] = 1.0
        # Overwrite pred_idx position with fraction (per-row stoichiometry)
        pred_mask = cols_idx == pred_idx
        if pred_mask.any():
            pred_rows = rows_idx[pred_mask]
            pred_m = m_ids[pred_mask]
            meth_full[pred_rows, pred_idx, pred_m] = fractions[pred_rows]

    # Rev_meth block: positions [kmer_size, kmer_size + REV_METH_LEN)
    rev_rows, rev_cols = np.where(rev_meth > 0)
    if rev_rows.size:
        rev_m_ids = rev_meth[rev_rows, rev_cols].astype(np.int64)
        rev_m_ids = np.clip(rev_m_ids, 0, num_meth_types - 1)
        meth_full[rev_rows, kmer_size + rev_cols, rev_m_ids] = 1.0

    # Free the source arrays now that meth_full holds the same info.
    del meth_ctx, rev_meth, fractions, rows_idx, cols_idx

    return {
        "kmer_ids":       kmer_ids,
        "meth_ids":       meth_ids,
        "signals_log":    signals_log,
        "meth_full":      meth_full,
        "parent_meths":   parent_meths,
        "parent_offsets": parent_offsets,
        "categories":     categories,
        "n_keys":         n_keys,
        "meth_counts":    dict(n_meth_counts),
    }


def _build_meth_full(
    ctx_ids: np.ndarray,
    rev_ids: np.ndarray,
    frac: float,
    pred_idx: int,
    kmer_size: int,
    num_meth_types: int,
) -> torch.Tensor:
    """Build the per-row ``meth_full`` Float[kmer_size + REV_METH_LEN, M] tensor.

    Layout:
      positions [0, kmer_size)         forward meth context (offsets [-7..+3])
      positions [kmer_size, ...)       rev_meth at active-site neighbours [-1, 0, +1]

    Encoding within the forward block:
      - prediction position, meth_id m: ``meth_full[pred_idx, m] = frac``
      - any other position, meth_id m: ``meth_full[pos, m] = 1.0``
    Rev_meth positions are always one-hot (no frac mixing — they describe
    a different strand's methylation status, not this row's stoichiometry).
    """
    from ..utils.sample_layout import REV_METH_LEN

    total_pos = kmer_size + REV_METH_LEN
    meth_full = torch.zeros(total_pos, num_meth_types, dtype=torch.float32)
    for pos in range(kmer_size):
        m = int(ctx_ids[pos])
        if m > 0:
            meth_full[pos, m] = frac if pos == pred_idx else 1.0
    for k in range(REV_METH_LEN):
        m = int(rev_ids[k])
        if m > 0:
            meth_full[kmer_size + k, m] = 1.0
    return meth_full


# ---------------------------------------------------------------------------
# In-memory dataset (small datasets — single .pkl)
# ---------------------------------------------------------------------------


class MLPSignalDataset(Dataset):
    """Flat-sample dataset for KinSim training, in-memory.

    Loads a single merged .pkl entirely into RAM, partitions every kmer's
    rows by ``meth_id`` at the prediction position, and pre-flattens all
    rows into contiguous arrays. The ``DataLoader`` shuffles the flat
    index each epoch — every sample is seen exactly once per epoch.

    Use this for small datasets that fit in RAM. For larger corpora use
    :class:`ShardedSignalDataset`.

    Args:
        pkl_path:       Path to a single shard .pkl from ``kinsim extract`` (or refined).
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

        flat = _flatten_data_dict(data_dict, kmer_size)
        if flat["n_keys"] == 0:
            raise ValueError(f"No int-keyed kmer data found in {pkl_path}")

        self._num_meth_types = num_meth_types
        self._kmer_size = kmer_size
        self._kmer_ids = flat["kmer_ids"]
        self._meth_ids = flat["meth_ids"]
        self._signals = flat["signals_log"]
        self._meth_full = flat["meth_full"]        # pre-built (N, 14, 4) float32
        self._parent_meths   = flat["parent_meths"]
        self._parent_offsets = flat["parent_offsets"]
        self._categories     = flat["categories"]

        n_total = len(self._kmer_ids)
        # Derive id→name from the YAML so adding a new methylation type
        # only requires editing kinsim_config.yaml; "0" stays "unmeth"
        # in this log line because the canonical name "none" reads worse.
        from ..utils.encoding import get_meth_ids

        _name_by_id = {v: k for k, v in get_meth_ids().items()}
        _name_by_id[0] = "unmeth"
        meth_summary = ", ".join(
            f"{_name_by_id.get(m, str(m))}={flat['meth_counts'][m]:,}"
            for m in sorted(flat["meth_counts"])
            if flat["meth_counts"][m] > 0
        )
        log.info(
            "MLPSignalDataset ready: %s unique kmers, %s samples [%s]",
            f"{flat['n_keys']:,}",
            f"{n_total:,}",
            meth_summary,
        )

    def __len__(self) -> int:
        """Total number of individual (IPD, PW) samples across all keys."""
        return len(self._kmer_ids)

    def __getitem__(self, idx: int):
        """Return the 7-tuple ``(kmer_id, meth_full, log_signal, meth_id,
        parent_meth, parent_offset, category)`` at idx.

        Fast path: ``meth_full`` is pre-built in ``_flatten_data_dict``
        on shard load — here we just slice and convert to torch tensors.
        No per-row Python loop.
        """
        return (
            torch.tensor(int(self._kmer_ids[idx]), dtype=torch.long),
            torch.from_numpy(self._meth_full[idx]),
            self._signals[idx],
            torch.tensor(int(self._meth_ids[idx]),       dtype=torch.long),
            torch.tensor(int(self._parent_meths[idx]),   dtype=torch.long),
            torch.tensor(int(self._parent_offsets[idx]), dtype=torch.long),
            torch.tensor(int(self._categories[idx]),     dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Sharded dataset (scales to arbitrary corpus size — never loads all in RAM)
# ---------------------------------------------------------------------------


class ShardedSignalDataset(IterableDataset):
    """Iterable training dataset over a list of shard pkls.

    Each shard is loaded, flattened, shuffled (within the shard) and
    yielded row-by-row. The next shard is loaded only after the previous
    one is fully consumed — peak RAM is bounded by **one shard**, not by
    the corpus size. The shard list itself is shuffled per epoch.

    Worker-aware: when the ``DataLoader`` uses ``num_workers > 0``, the
    shard list is partitioned across workers (each worker gets a disjoint
    subset). This parallelises shard I/O and EM-flatten cost without RAM
    pressure.

    Output rows are identical to :class:`MLPSignalDataset` —
    ``(kmer_id, meth_full, log_signal, meth_id, parent_meth,
    parent_offset, category)`` — so the model code
    is the same regardless of dataset class.

    Args:
        shard_paths:    List of paths to per-strain shard pkls.
        num_meth_types: Number of methylation states (default 4).
        kmer_size:      K-mer window size (default K=11).
        shuffle:        Shuffle shard order per epoch and rows within
                        each shard. Default True (set False for
                        deterministic test set evaluation).
        seed:           Base seed for the shuffler. Combined with
                        worker_id and epoch via ``set_epoch()`` so
                        different workers / epochs see different orders.
    """

    def __init__(
        self,
        shard_paths,
        num_meth_types: int = 4,
        kmer_size: int = K,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._shard_paths = [str(Path(p)) for p in shard_paths]
        if not self._shard_paths:
            raise ValueError("shard_paths is empty")
        self._num_meth_types = num_meth_types
        self._kmer_size = kmer_size
        self._shuffle = shuffle
        self._seed = int(seed)
        self._epoch = 0
        log.info(
            "ShardedSignalDataset ready: %d shards (shuffle=%s)",
            len(self._shard_paths),
            shuffle,
        )

    def set_epoch(self, epoch: int) -> None:
        """Inject the current epoch into the shuffler seed.

        Lightning calls this on each new training epoch so that the
        per-epoch shard order is reproducible-but-different. If you
        manage the loop yourself, call it before each pass.
        """
        self._epoch = int(epoch)

    @property
    def shard_paths(self) -> list[str]:
        """The list of shard pkls this dataset iterates over."""
        return list(self._shard_paths)

    def _worker_shards_and_rng(self) -> tuple[list[str], np.random.Generator]:
        """Partition shards across DataLoader workers, return per-worker RNG."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid, n_workers = 0, 1
        else:
            wid, n_workers = worker_info.id, worker_info.num_workers
        # Disjoint stride partition: worker w gets paths[w :: n_workers]
        my_shards = list(self._shard_paths[wid::n_workers])
        # Seed: base + worker + epoch — different orders per (worker, epoch).
        rng = np.random.default_rng(self._seed + 1000 * wid + self._epoch)
        if self._shuffle:
            rng.shuffle(my_shards)
        return my_shards, rng

    def __iter__(self):
        """Yield one row at a time from each assigned shard.

        Fast path: ``meth_full`` is pre-built in ``_flatten_data_dict``
        and tensors are converted from the numpy arrays *at shard load
        time* (not per row). The per-row overhead is one tensor scalar
        creation × 6 fields + one ``meth_full`` slice — vs. the old
        per-row 14-position Python loop + 5 torch.tensor calls. ~3-5×
        speedup on the data path.

        ``max_rows_per_shard`` (if set on the dataset) caps how many
        rows we yield from a single shard before moving to the next.
        This guarantees each shard is visited at least partially per
        epoch, instead of one worker burning all its limit_train_batches
        budget on a single big shard while other shards (= other
        strains, possibly with rare meth types) are never touched.
        """
        my_shards, rng = self._worker_shards_and_rng()
        cap = getattr(self, "_max_rows_per_shard", None)
        for shard_path in my_shards:
            with open(shard_path, "rb") as f:
                data_dict = pickle.load(f)
            flat = _flatten_data_dict(
                data_dict, self._kmer_size, num_meth_types=self._num_meth_types,
            )
            del data_dict  # release the loaded shard before iterating
            n = len(flat["kmer_ids"])
            if n == 0:
                continue
            order = np.arange(n)
            if self._shuffle:
                rng.shuffle(order)
            if cap is not None and 0 < cap < n:
                order = order[:cap]

            # Convert the numpy arrays to torch once per shard.
            kmer_ids_t       = torch.from_numpy(flat["kmer_ids"].astype(np.int64))
            meth_ids_t       = torch.from_numpy(flat["meth_ids"].astype(np.int64))
            parent_meths_t   = torch.from_numpy(flat["parent_meths"].astype(np.int64))
            parent_offsets_t = torch.from_numpy(flat["parent_offsets"].astype(np.int64))
            categories_t     = torch.from_numpy(flat["categories"].astype(np.int64))
            meth_full_t      = torch.from_numpy(flat["meth_full"])  # already float32
            signals_t        = flat["signals_log"]                  # already torch tensor

            for idx in order:
                i = int(idx)
                yield (
                    kmer_ids_t[i],
                    meth_full_t[i],
                    signals_t[i],
                    meth_ids_t[i],
                    parent_meths_t[i],
                    parent_offsets_t[i],
                    categories_t[i],
                )

            # Explicit cleanup before loading next shard. Tensors built
            # via torch.from_numpy share memory with the numpy arrays in
            # ``flat`` — without explicit del, Python would hold BOTH
            # the old and the new shard's arrays in RAM during the
            # transition (2× shard peak). On a 20 GB shard with 2
            # workers that's an extra ~40 GB peak.
            del kmer_ids_t, meth_ids_t, parent_meths_t, parent_offsets_t
            del categories_t, meth_full_t, signals_t, flat


# ---------------------------------------------------------------------------
# Train/test split helpers
# ---------------------------------------------------------------------------


def list_shards(shards_dir, glob: str = "*_shard*.pkl") -> list[str]:
    """Return a sorted list of shard pkl paths in ``shards_dir``.

    Default glob ``*_shard*.pkl`` matches both raw extract output
    (``<sample>_shard.pkl``) and refined output (``<sample>_shard_clean.pkl``)
    so train can be pointed at either.
    """
    return sorted(str(p) for p in Path(shards_dir).glob(glob))


def shard_sample_id(shard_path: str) -> str:
    """Recover the manifest sample_id from a shard filename.

    Expects the filename ``<sample_id>_shard.pkl`` or
    ``<sample_id>_shard_clean.pkl`` (post-refine). Returns the leading
    sample_id portion.
    """
    name = Path(shard_path).stem
    for suffix in ("_shard_clean", "_shard"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def split_shards(
    shard_paths,
    test_strains: list | None = None,
    test_fraction: float | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split shard paths into (train, test) lists.

    Priority:
      1. ``test_strains`` (explicit sample_id list) — shards whose
         sample_id is in that list go to test. Strict — every name must
         match an existing shard.
      2. ``test_fraction`` (random by-shard split) — round to nearest int.
      3. Otherwise: 90 / 10 random split with ``seed``.
    """
    paths = [str(p) for p in shard_paths]
    if not paths:
        raise ValueError("shard_paths is empty")

    if test_strains:
        wanted = set(test_strains)
        train_paths = [p for p in paths if shard_sample_id(p) not in wanted]
        test_paths = [p for p in paths if shard_sample_id(p) in wanted]
        found = {shard_sample_id(p) for p in test_paths}
        missing = wanted - found
        if missing:
            raise ValueError(
                f"--test-strains contains sample_ids with no matching shard: {sorted(missing)}"
            )
        return train_paths, test_paths

    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    frac = float(test_fraction) if test_fraction is not None else 0.10
    n_test = max(1, round(len(paths) * frac))
    test_idx = set(int(i) for i in idx[:n_test])
    train_paths = [p for i, p in enumerate(paths) if i not in test_idx]
    test_paths = [p for i, p in enumerate(paths) if i in test_idx]
    return train_paths, test_paths
