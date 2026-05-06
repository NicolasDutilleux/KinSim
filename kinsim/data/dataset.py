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

Both datasets emit ``(kmer_id, meth_full, log_signal, meth_id)`` tuples.
``meth_full`` has shape ``(kmer_size + REV_METH_LEN, num_meth_types)`` =
``(14, 4)`` — 11 forward-context entries followed by the 3 rev_meth
entries at active-site neighbours. The model's FiLM projection consumes
the flattened tensor, so all 14 positions contribute to ``(γ, β)``.
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


def _flatten_data_dict(data_dict: dict, kmer_size: int) -> dict:
    """Flatten one ``dict[kmer_id] -> ndarray(N, SAMPLE_NCOLS)`` into per-row ndarrays.

    Returns a dict with: ``kmer_ids`` (int32), ``meth_ids`` (int8 — meth
    at the prediction position), ``signals_log`` (n, 2 log1p tensor),
    ``fractions`` (float32), ``meth_ctx`` (uint8 (n, kmer_size)),
    ``rev_meth`` (uint8 (n, REV_METH_LEN)), ``n_keys``, ``meth_counts``.
    """
    from ..utils.encoding import KMER_PRED_IDX
    from ..utils.sample_layout import (
        COL_METH_CTX_START,
        COL_REV_METH,
        REV_METH_LEN,
        SAMPLE_NCOLS,
    )

    pred_idx = KMER_PRED_IDX
    kmer_ids_list: list = []
    meth_ids_list: list = []
    signals_list: list = []
    fractions_list: list = []
    meth_ctx_list: list = []
    rev_meth_list: list = []
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
        for mid in np.unique(meth_at_center):
            n_meth_counts[int(mid)] += int((meth_at_center == mid).sum())
        n_keys += 1

    if not kmer_ids_list:
        return {
            "kmer_ids": np.empty(0, dtype=np.int32),
            "meth_ids": np.empty(0, dtype=np.int8),
            "signals_log": torch.empty(0, 2, dtype=torch.float32),
            "fractions": np.empty(0, dtype=np.float32),
            "meth_ctx": np.empty((0, kmer_size), dtype=np.uint8),
            "rev_meth": np.empty((0, REV_METH_LEN), dtype=np.uint8),
            "n_keys": 0,
            "meth_counts": dict(n_meth_counts),
        }

    return {
        "kmer_ids": np.concatenate(kmer_ids_list),
        "meth_ids": np.concatenate(meth_ids_list),
        "signals_log": log_transform(
            torch.from_numpy(np.concatenate(signals_list, axis=0)).float()
        ),
        "fractions": np.concatenate(fractions_list),
        "meth_ctx": np.concatenate(meth_ctx_list),
        "rev_meth": np.concatenate(rev_meth_list),
        "n_keys": n_keys,
        "meth_counts": dict(n_meth_counts),
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
        self._fractions = flat["fractions"]
        self._meth_ctx = flat["meth_ctx"]
        self._rev_meth = flat["rev_meth"]

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
        """Return the (kmer_id, meth_full, log_signal, meth_id) tuple at idx."""
        from ..utils.encoding import KMER_PRED_IDX

        kmer_id = int(self._kmer_ids[idx])
        meth_id = int(self._meth_ids[idx])
        signal = self._signals[idx]  # already log1p
        frac = float(self._fractions[idx])
        ctx_ids = self._meth_ctx[idx]
        rev_ids = self._rev_meth[idx]
        meth_full = _build_meth_full(
            ctx_ids, rev_ids, frac, KMER_PRED_IDX,
            self._kmer_size, self._num_meth_types,
        )
        return (
            torch.tensor(kmer_id, dtype=torch.long),
            meth_full,
            signal,
            torch.tensor(meth_id, dtype=torch.long),
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
    ``(kmer_id, meth_full, log_signal, meth_id)`` — so the model code
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
        from ..utils.encoding import KMER_PRED_IDX

        my_shards, rng = self._worker_shards_and_rng()
        for shard_path in my_shards:
            with open(shard_path, "rb") as f:
                data_dict = pickle.load(f)
            flat = _flatten_data_dict(data_dict, self._kmer_size)
            del data_dict  # release the loaded shard before iterating
            n = len(flat["kmer_ids"])
            if n == 0:
                continue
            order = np.arange(n)
            if self._shuffle:
                rng.shuffle(order)
            kmer_ids = flat["kmer_ids"]
            meth_ids = flat["meth_ids"]
            signals = flat["signals_log"]
            fractions = flat["fractions"]
            meth_ctx = flat["meth_ctx"]
            rev_meth = flat["rev_meth"]
            for idx in order:
                meth_full = _build_meth_full(
                    meth_ctx[idx],
                    rev_meth[idx],
                    float(fractions[idx]),
                    KMER_PRED_IDX,
                    self._kmer_size,
                    self._num_meth_types,
                )
                yield (
                    torch.tensor(int(kmer_ids[idx]), dtype=torch.long),
                    meth_full,
                    signals[idx],
                    torch.tensor(int(meth_ids[idx]), dtype=torch.long),
                )


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
