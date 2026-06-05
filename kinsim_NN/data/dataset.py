"""PyTorch datasets for kinsim_NN.

Two flavours:

* :class:`ShardedDataset` — in-memory load of a single shard. Used by
  small-scale tests and the evaluator.
* :class:`MultiShardDataset` — walks a directory of shards lazily,
  yielding one batch index at a time. Used during training to avoid
  loading the full ~32 GB corpus in RAM.

At ``__getitem__`` time we expand the compact uint8 storage into
training tensors:

    base_fwd[K]            uint8       → base_fwd_onehot (K, 4) float32
    base_rev (derived)     uint8       → base_rev_onehot (K, 4) float32
    meth_fwd[K]            uint8       → meth_fwd_onehot (K, M) float32
    meth_rev[K]            uint8       → meth_rev_onehot (K, M) float32
    signal[K, 4]           uint8       → log1p(frames(signal))  (K, 4) float32

The reverse complement of a base is computed via a static table.
"""
from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..utils.encoding import BASE_RC as _BASE_RC
from ..utils.pacbio_codec import FRAMES_TABLE
from .shard import ShardData, read_shard


log = logging.getLogger(__name__)


# log1p(frames) per uint8 byte (uses PacBio codec lookup)
_LOG1P_FRAMES = np.log1p(FRAMES_TABLE.astype(np.float32))


def _onehot(arr: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot encode an int8 array. Last dim is the class dim."""
    out = np.zeros(arr.shape + (n_classes,), dtype=np.float32)
    idx = arr.astype(np.int64)
    np.put_along_axis(out, idx[..., None], 1.0, axis=-1)
    return out


# ---------------------------------------------------------------------------
# Single-shard in-memory Dataset
# ---------------------------------------------------------------------------


class ShardedDataset(Dataset):
    """In-memory dataset wrapping a single :class:`ShardData`.

    Fast random access. Use for small experiments, evaluation, or unit
    tests. For multi-strain training prefer :class:`MultiShardDataset`.
    """

    def __init__(self, shard: ShardData, n_meth_types: int):
        self.shard = shard
        self.n_meth_types = int(n_meth_types)

    def __len__(self) -> int:
        return self.shard.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.shard
        base_fwd = s.base_fwd[idx]                     # (K,) uint8
        base_rev = _BASE_RC[base_fwd]                  # (K,) uint8
        meth_fwd = s.meth_fwd[idx]                     # (K,) uint8
        meth_rev = s.meth_rev[idx]                     # (K,) uint8
        signal_u8 = s.signal[idx]                      # (K, 4) uint8
        signal = _LOG1P_FRAMES[signal_u8]              # (K, 4) float32

        return {
            "base_fwd_onehot": torch.from_numpy(_onehot(base_fwd, 4)),
            "base_rev_onehot": torch.from_numpy(_onehot(base_rev, 4)),
            "meth_fwd_onehot": torch.from_numpy(_onehot(meth_fwd, self.n_meth_types)),
            "meth_rev_onehot": torch.from_numpy(_onehot(meth_rev, self.n_meth_types)),
            "signal": torch.from_numpy(signal),
            "category": int(s.category[idx]),
        }


# ---------------------------------------------------------------------------
# Multi-shard streaming
# ---------------------------------------------------------------------------


def _stable_hash(s: str) -> int:
    """Process-stable 32-bit hash. CPython's builtin ``hash`` is salted
    per process unless ``PYTHONHASHSEED`` is set."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16)


class MultiShardDataset(IterableDataset):
    """Streaming over multiple shards. One shard in RAM at a time.

    The shards are visited in randomised order each epoch; within a
    shard the row indices are shuffled. Worker-aware: each PyTorch
    DataLoader worker gets a disjoint subset of shards.

    Call :meth:`set_epoch` from the training loop before each pass so
    shuffle ordering actually rotates across epochs.

    All shards in ``shard_paths`` must share the same ``K`` (window
    width); a mismatch raises at first read.
    """

    def __init__(
        self,
        shard_paths: list[Path],
        n_meth_types: int,
        shuffle_shards: bool = True,
        shuffle_rows: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        if not self.shard_paths:
            raise ValueError("MultiShardDataset: empty shard_paths")
        self.n_meth_types = int(n_meth_types)
        self.shuffle_shards = shuffle_shards
        self.shuffle_rows = shuffle_rows
        self.seed = int(seed)
        self._epoch = 0
        self._expected_k: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Advance the epoch counter so the per-epoch shuffle changes."""
        self._epoch = int(epoch)

    def _worker_paths(self) -> list[Path]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return list(self.shard_paths)
        n_workers = info.num_workers
        worker_id = info.id
        return [p for i, p in enumerate(self.shard_paths) if i % n_workers == worker_id]

    def __iter__(self) -> Iterable[dict[str, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        epoch_seed = self.seed + self._epoch * 1_000_003 + worker_id * 7919
        rng = random.Random(epoch_seed)

        paths = self._worker_paths()
        if self.shuffle_shards:
            rng.shuffle(paths)

        for p in paths:
            try:
                shard = read_shard(p)
            except (OSError, EOFError, ValueError, KeyError) as e:
                log.warning("Skipping unreadable shard %s: %s", p, e)
                continue
            if self._expected_k is None:
                self._expected_k = shard.k
            elif shard.k != self._expected_k:
                raise ValueError(
                    f"Shard K mismatch: {p.name} has K={shard.k}, "
                    f"expected K={self._expected_k} (first shard read in this worker)"
                )
            n = shard.n
            if n == 0:
                continue
            indices = np.arange(n)
            if self.shuffle_rows:
                row_seed = epoch_seed ^ _stable_hash(str(p))
                np.random.default_rng(row_seed).shuffle(indices)
            ds = ShardedDataset(shard, self.n_meth_types)
            for i in indices:
                yield ds[int(i)]


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------


def list_shards(directory: Path, exclude_strains: set[str] | None = None) -> list[Path]:
    """Return all shard paths under ``directory``, optionally excluding
    test-split strains by sample_id.

    Matching mirrors the eval-side ``glob(f"*{t}_shard.pkl")`` logic:
    a test_strain entry like ``"bc2034"`` excludes both
    ``strepto_bc2034`` and ``vega_bc2034`` shards (the bare barcode
    appears as the trailing underscore-separated component of every
    lineage-prefixed sample_id). A test_strain entry that matches the
    full sample_id (e.g. ``"strepto_bc2034"``) is also honoured.
    Without this trailing-component match a bare barcode in the YAML's
    ``split.test_strains`` list would never match a prefixed shard and
    the held-out evaluation would silently train on its own test set.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"shards dir not found: {directory}")
    exclude_strains = exclude_strains or set()
    paths = sorted(directory.glob("*_shard.pkl"))
    out: list[Path] = []
    for p in paths:
        # Filename like "strepto_bc2033_shard.pkl" → sample_id "strepto_bc2033"
        sid = p.stem.removesuffix("_shard")
        # Match the test_strain either against the full sample_id or against
        # the trailing underscore-separated component (the barcode form).
        sid_tail = sid.rsplit("_", 1)[-1]
        if sid in exclude_strains or sid_tail in exclude_strains:
            continue
        out.append(p)
    return out


def shard_sample_id(path: Path) -> str:
    return Path(path).stem.removesuffix("_shard")


__all__ = [
    "ShardedDataset",
    "MultiShardDataset",
    "list_shards",
    "shard_sample_id",
]
