"""Dataset + signal-space transforms for KinSim training (bilateral v2).

Shard format::

    dict[kmer_id (int)] -> np.ndarray(N, SAMPLE_NCOLS)  # 11 + 2*K cols

Each row emits a 5-tuple from the iterator:

    (kmer_id_fwd, meth_ctx_fwd, meth_ctx_rev, targets_log,
     category_fwd, category_rev)

* ``kmer_id_fwd``: int — forward-strand kmer ID (2K-bit encoded).
* ``meth_ctx_fwd``: float32 (K, M) — + strand meth one-hot per position.
* ``meth_ctx_rev``: float32 (K, M) — - strand meth one-hot per position.
* ``targets_log``: float32 (4,) — log1p(fi, fp, ri, rp).
* ``category_fwd``, ``category_rev``: int8 — for metric bucketing.

``SignalDataset`` loads one shard into RAM; ``ShardedSignalDataset`` is
an ``IterableDataset`` that streams shards (peak RAM bounded by one shard).
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..utils.config import ExtractionParams, get_extraction_params
from ..utils.sample_layout import SampleLayout, get_sample_layout

log = logging.getLogger(__name__)


def log_transform(x: torch.Tensor) -> torch.Tensor:
    """Map raw [0, 255] uint8 signals into log1p space for training."""
    return torch.log1p(x)


def inv_log_transform(x: torch.Tensor) -> torch.Tensor:
    """Inverse of log_transform: clamp expm1 to [0, BAM_TAG_MAX]."""
    from ..utils._defaults import BAM_TAG_MAX
    return torch.clamp(torch.expm1(x), 0, BAM_TAG_MAX)


def _peek_shard_extraction_params(shard_path: str) -> ExtractionParams | None:
    """Read ``ExtractionParams`` from a shard without fully unpickling."""
    path = Path(shard_path)
    if not path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    out = read_shard_extraction_params(data_dict)
    del data_dict
    return out


def read_shard_extraction_params(data_dict: dict) -> ExtractionParams | None:
    """Return the ``ExtractionParams`` from a shard's ``__meta__``."""
    meta = data_dict.get("__meta__")
    if not isinstance(meta, dict):
        return None
    raw = meta.get("extraction_params")
    if not raw:
        return None
    try:
        return ExtractionParams.from_dict(raw)
    except (TypeError, ValueError, KeyError) as exc:
        log.warning("shard __meta__.extraction_params failed to parse: %s", exc)
        return None


def _flatten_data_dict(
    data_dict: dict,
    layout: SampleLayout,
    num_meth_types: int = 4,
) -> dict:
    """Flatten ``dict[kmer_id -> ndarray(N, n_cols)]`` into per-row arrays.

    Outputs (numpy / torch):
        kmer_ids        int64 (N,)        forward-strand kmer IDs
        targets_log     float32 (N, 4)    log1p(fi, fp, ri, rp)
        meth_ctx_fwd    float32 (N, K, M) one-hot + strand meth context
        meth_ctx_rev    float32 (N, K, M) one-hot - strand meth context
        category_fwd    int8   (N,)       per-strand category
        category_rev    int8   (N,)
        parent_meth_fwd int8   (N,)
        parent_offset_fwd int8 (N,)
        parent_meth_rev int8   (N,)
        parent_offset_rev int8 (N,)
        n_keys          int                 # distinct kmer_ids seen
        meth_counts     dict[int, int]      # forward-strand meth_id histogram
    """
    K = layout.kmer_size
    expected_ncols = layout.n_cols

    kmer_ids_list: list = []
    kinetics_list: list = []
    mc_fwd_list: list = []
    mc_rev_list: list = []
    cat_fwd_list: list = []
    cat_rev_list: list = []
    pm_fwd_list: list = []
    po_fwd_list: list = []
    pm_rev_list: list = []
    po_rev_list: list = []
    n_keys = 0
    n_meth_counts: dict[int, int] = defaultdict(int)

    for key, samples in data_dict.items():
        if key == "__meta__":
            continue
        if not isinstance(key, (int, np.integer)):
            continue
        if not isinstance(samples, np.ndarray) or samples.ndim != 2:
            continue
        if samples.shape[1] != expected_ncols:
            raise ValueError(
                f"Shard row width mismatch — shard has {samples.shape[1]} cols "
                f"but layout (K={K}) expects {expected_ncols}. Re-extract under "
                f"the matching geometry."
            )
        n = len(samples)
        kid = int(key)

        kmer_ids_list.append(np.full(n, kid, dtype=np.int64))
        # Kinetics: [ipd_fwd, pw_fwd, ipd_rev, pw_rev]
        kinetics_list.append(samples[:, [
            layout.col_ipd_fwd, layout.col_pw_fwd,
            layout.col_ipd_rev, layout.col_pw_rev,
        ]].astype(np.float32))
        mc_fwd_list.append(samples[:, layout.col_meth_ctx_fwd_start : layout.col_meth_ctx_fwd_end].astype(np.uint8))
        mc_rev_list.append(samples[:, layout.col_meth_ctx_rev_start : layout.col_meth_ctx_rev_end].astype(np.uint8))
        cat_fwd_list.append(samples[:, layout.col_category_fwd].astype(np.int8))
        cat_rev_list.append(samples[:, layout.col_category_rev].astype(np.int8))
        pm_fwd_list.append(samples[:, layout.col_parent_meth_fwd].astype(np.int8))
        po_fwd_list.append(samples[:, layout.col_parent_offset_fwd].astype(np.int8))
        pm_rev_list.append(samples[:, layout.col_parent_meth_rev].astype(np.int8))
        po_rev_list.append(samples[:, layout.col_parent_offset_rev].astype(np.int8))

        for mid in np.unique(samples[:, layout.col_parent_meth_fwd].astype(np.int8)):
            n_meth_counts[int(mid)] += int((samples[:, layout.col_parent_meth_fwd] == mid).sum())
        n_keys += 1

    if not kmer_ids_list:
        return {
            "kmer_ids": np.empty(0, dtype=np.int64),
            "targets_log": torch.empty(0, 4, dtype=torch.float32),
            "meth_ctx_fwd": np.empty((0, K, num_meth_types), dtype=np.float32),
            "meth_ctx_rev": np.empty((0, K, num_meth_types), dtype=np.float32),
            "category_fwd": np.empty(0, dtype=np.int8),
            "category_rev": np.empty(0, dtype=np.int8),
            "parent_meth_fwd": np.empty(0, dtype=np.int8),
            "parent_offset_fwd": np.empty(0, dtype=np.int8),
            "parent_meth_rev": np.empty(0, dtype=np.int8),
            "parent_offset_rev": np.empty(0, dtype=np.int8),
            "n_keys": 0,
            "meth_counts": dict(n_meth_counts),
        }

    kmer_ids = np.concatenate(kmer_ids_list)
    kmer_ids_list = None
    kinetics = np.concatenate(kinetics_list, axis=0)
    kinetics_list = None
    mc_fwd = np.concatenate(mc_fwd_list, axis=0)
    mc_fwd_list = None
    mc_rev = np.concatenate(mc_rev_list, axis=0)
    mc_rev_list = None
    cat_fwd = np.concatenate(cat_fwd_list)
    cat_fwd_list = None
    cat_rev = np.concatenate(cat_rev_list)
    cat_rev_list = None
    pm_fwd = np.concatenate(pm_fwd_list)
    pm_fwd_list = None
    po_fwd = np.concatenate(po_fwd_list)
    po_fwd_list = None
    pm_rev = np.concatenate(pm_rev_list)
    pm_rev_list = None
    po_rev = np.concatenate(po_rev_list)
    po_rev_list = None

    targets_log = log_transform(torch.from_numpy(kinetics).float())

    # Vectorised one-hot build for both meth contexts.
    n_rows = kmer_ids.shape[0]
    mctx_fwd = np.zeros((n_rows, K, num_meth_types), dtype=np.float32)
    rows_idx, cols_idx = np.where(mc_fwd > 0)
    if rows_idx.size:
        m_ids = np.clip(mc_fwd[rows_idx, cols_idx].astype(np.int64), 0, num_meth_types - 1)
        mctx_fwd[rows_idx, cols_idx, m_ids] = 1.0
    mctx_rev = np.zeros((n_rows, K, num_meth_types), dtype=np.float32)
    rows_idx, cols_idx = np.where(mc_rev > 0)
    if rows_idx.size:
        m_ids = np.clip(mc_rev[rows_idx, cols_idx].astype(np.int64), 0, num_meth_types - 1)
        mctx_rev[rows_idx, cols_idx, m_ids] = 1.0

    del mc_fwd, mc_rev, kinetics, rows_idx, cols_idx

    return {
        "kmer_ids": kmer_ids,
        "targets_log": targets_log,
        "meth_ctx_fwd": mctx_fwd,
        "meth_ctx_rev": mctx_rev,
        "category_fwd": cat_fwd,
        "category_rev": cat_rev,
        "parent_meth_fwd": pm_fwd,
        "parent_offset_fwd": po_fwd,
        "parent_meth_rev": pm_rev,
        "parent_offset_rev": po_rev,
        "n_keys": n_keys,
        "meth_counts": dict(n_meth_counts),
    }


class SignalDataset(Dataset):
    """In-memory bilateral dataset for one shard pkl."""

    def __init__(
        self,
        pkl_path: str,
        num_meth_types: int = 4,
        params: ExtractionParams | None = None,
    ) -> None:
        log.info("Loading training data from %s ...", pkl_path)
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)
        if not isinstance(data_dict, dict) or not data_dict:
            raise ValueError(f"Invalid or empty pkl: {pkl_path}")

        shard_params = read_shard_extraction_params(data_dict)
        if params is not None:
            if shard_params is not None:
                params.assert_compatible(shard_params, where=f"shard {pkl_path}")
            resolved = params
        elif shard_params is not None:
            resolved = shard_params
        else:
            resolved = get_extraction_params()

        layout = get_sample_layout(resolved)
        flat = _flatten_data_dict(data_dict, layout, num_meth_types=num_meth_types)
        if flat["n_keys"] == 0:
            raise ValueError(f"No int-keyed kmer data in {pkl_path}")

        self._num_meth_types = num_meth_types
        self._params = resolved
        self._layout = layout
        self._kmer_ids = flat["kmer_ids"]
        self._targets = flat["targets_log"]
        self._meth_ctx_fwd = flat["meth_ctx_fwd"]
        self._meth_ctx_rev = flat["meth_ctx_rev"]
        self._cat_fwd = flat["category_fwd"]
        self._cat_rev = flat["category_rev"]
        self._pm_fwd = flat["parent_meth_fwd"]
        self._po_fwd = flat["parent_offset_fwd"]
        self._pm_rev = flat["parent_meth_rev"]
        self._po_rev = flat["parent_offset_rev"]

        from ..utils.encoding import get_meth_ids
        _name = {v: k for k, v in get_meth_ids().items()}
        _name[0] = "unmeth"
        summary = ", ".join(
            f"{_name.get(m, str(m))}={flat['meth_counts'][m]:,}"
            for m in sorted(flat["meth_counts"])
            if flat["meth_counts"][m] > 0
        )
        log.info(
            "SignalDataset ready: %s kmers, %s samples [%s]",
            f"{flat['n_keys']:,}", f"{len(self._kmer_ids):,}", summary,
        )

    def __len__(self) -> int:
        return len(self._kmer_ids)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(int(self._kmer_ids[idx]), dtype=torch.long),
            torch.from_numpy(self._meth_ctx_fwd[idx]),
            torch.from_numpy(self._meth_ctx_rev[idx]),
            self._targets[idx],
            torch.tensor(int(self._cat_fwd[idx]), dtype=torch.long),
            torch.tensor(int(self._cat_rev[idx]), dtype=torch.long),
        )


class ShardedSignalDataset(IterableDataset):
    """Iterable bilateral dataset over a list of shard pkls.

    Peak RAM bounded by one shard. Worker-aware (each DataLoader worker
    sees a disjoint subset of the shards) and per-epoch shuffling.
    """

    def __init__(
        self,
        shard_paths,
        num_meth_types: int = 4,
        shuffle: bool = True,
        seed: int = 42,
        balance_kmers: bool = False,
        params: ExtractionParams | None = None,
    ) -> None:
        super().__init__()
        self._shard_paths = [str(Path(p)) for p in shard_paths]
        if not self._shard_paths:
            raise ValueError("shard_paths is empty")
        self._num_meth_types = num_meth_types
        if params is None:
            params = _peek_shard_extraction_params(self._shard_paths[0])
            if params is None:
                params = get_extraction_params()
        self._params = params
        self._layout = get_sample_layout(params)
        self._kmer_size = params.kmer_size
        self._shuffle = shuffle
        self._seed = int(seed)
        self._epoch = 0
        self._balance_kmers = bool(balance_kmers)
        log.info(
            "ShardedSignalDataset ready: %d shards  K=%d  n_cols=%d  shuffle=%s  balance=%s",
            len(self._shard_paths), params.kmer_size, params.sample_ncols,
            shuffle, self._balance_kmers,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def shard_paths(self) -> list[str]:
        return list(self._shard_paths)

    def _worker_shards_and_rng(self) -> tuple[list[str], np.random.Generator]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid, n_workers = 0, 1
        else:
            wid, n_workers = worker_info.id, worker_info.num_workers
        my_shards = list(self._shard_paths[wid::n_workers])
        rng = np.random.default_rng(self._seed + 1000 * wid + self._epoch)
        if self._shuffle:
            rng.shuffle(my_shards)
        return my_shards, rng

    def __iter__(self):
        from ..utils.sample_layout import CATEGORY_BASELINE
        my_shards, rng = self._worker_shards_and_rng()
        cap = getattr(self, "_max_rows_per_shard", None)
        balance = self._balance_kmers
        for shard_path in my_shards:
            with open(shard_path, "rb") as f:
                data_dict = pickle.load(f)
            shard_params = read_shard_extraction_params(data_dict)
            if shard_params is not None:
                self._params.assert_compatible(shard_params, where=f"shard {shard_path}")
            flat = _flatten_data_dict(data_dict, self._layout, num_meth_types=self._num_meth_types)
            del data_dict
            n = len(flat["kmer_ids"])
            if n == 0:
                continue

            if balance and n > 1:
                # Inverse-sqrt-frequency draw on (kmer_id, category_fwd) — rare
                # SLOWED rows get more weight than common BASELINE.
                _MULT = 4  # >= max(category)+1, leaves room for future categories.
                composite = (
                    flat["kmer_ids"].astype(np.int64) * _MULT
                    + flat["category_fwd"].astype(np.int64)
                )
                counts = np.bincount(composite)
                row_w = 1.0 / np.sqrt(np.maximum(counts[composite], 1).astype(np.float64))
                row_w /= row_w.sum()
                target = min(cap, n) if (cap and cap > 0) else n
                order = rng.choice(n, size=target, replace=True, p=row_w)
            else:
                order = np.arange(n)
                if self._shuffle:
                    rng.shuffle(order)
                if cap is not None and 0 < cap < n:
                    order = order[:cap]

            kmer_ids_t = torch.from_numpy(flat["kmer_ids"].astype(np.int64))
            cat_fwd_t = torch.from_numpy(flat["category_fwd"].astype(np.int64))
            cat_rev_t = torch.from_numpy(flat["category_rev"].astype(np.int64))
            mc_fwd_t = torch.from_numpy(flat["meth_ctx_fwd"])
            mc_rev_t = torch.from_numpy(flat["meth_ctx_rev"])
            targets_t = flat["targets_log"]

            for idx in order:
                i = int(idx)
                yield (
                    kmer_ids_t[i],
                    mc_fwd_t[i],
                    mc_rev_t[i],
                    targets_t[i],
                    cat_fwd_t[i],
                    cat_rev_t[i],
                )

            del kmer_ids_t, cat_fwd_t, cat_rev_t, mc_fwd_t, mc_rev_t, targets_t, flat


def list_shards(shards_dir, glob: str = "*_shard*.pkl") -> list[str]:
    """List shards in ``shards_dir``. Prefer ``_clean.pkl`` over raw ``_shard.pkl``."""
    paths = sorted(str(p) for p in Path(shards_dir).glob(glob))
    if glob != "*_shard*.pkl":
        return paths
    cleaned_ids = {
        shard_sample_id(p)
        for p in paths
        if Path(p).stem.endswith("_shard_clean")
    }
    return [
        p for p in paths
        if not (Path(p).stem.endswith("_shard") and shard_sample_id(p) in cleaned_ids)
    ]


def shard_sample_id(shard_path: str) -> str:
    """Recover the sample_id from a shard filename."""
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
    """Split shard paths into (train, test)."""
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
