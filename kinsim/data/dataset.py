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
    SignalDataset           loads a single shard into RAM (small datasets / debugging)
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

from ..utils.config import ExtractionParams, get_extraction_params
from ..utils.sample_layout import SampleLayout, get_sample_layout

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
# Shared helpers (used by both SignalDataset and ShardedSignalDataset)
# ---------------------------------------------------------------------------


def _peek_shard_extraction_params(shard_path: str) -> ExtractionParams | None:
    """Load only enough of a shard pkl to recover its ``ExtractionParams``.

    Loading a 20 GB shard just to read its 200-byte meta block is wasteful;
    this helper unpickles the full dict but discards everything except the
    parsed :class:`ExtractionParams`. Used at dataset-construction time to
    resolve the column geometry without paying the full flatten cost.

    Args:
        shard_path: Path to the shard pkl.

    Returns:
        The shard's :class:`ExtractionParams` if recorded in its
        ``__meta__["extraction_params"]`` block, otherwise ``None`` (legacy
        shards built before v0.5).

    Raises:
        FileNotFoundError: If ``shard_path`` does not exist.
    """
    path = Path(shard_path)
    if not path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")
    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    out = read_shard_extraction_params(data_dict)
    del data_dict
    return out


def read_shard_extraction_params(data_dict: dict) -> ExtractionParams | None:
    """Return the :class:`ExtractionParams` recorded in a shard's ``__meta__``.

    New shards (post v0.5) carry ``__meta__["extraction_params"]``; legacy
    shards do not. The legacy fallback returns ``None`` so the caller can
    decide whether to assume the historical K=11 layout (back-compat) or
    refuse to load the shard.
    """
    meta = data_dict.get("__meta__")
    if not isinstance(meta, dict):
        return None
    raw = meta.get("extraction_params")
    if not raw:
        return None
    try:
        return ExtractionParams.from_dict(raw)
    except Exception as exc:
        log.warning("shard __meta__.extraction_params failed to parse: %s", exc)
        return None


def _flatten_data_dict(
    data_dict: dict,
    layout: SampleLayout,
    num_meth_types: int = 4,
) -> dict:
    """Flatten one ``dict[kmer_id -> ndarray(N, layout.n_cols)]`` into per-row arrays.

    **Performance**: this is the hot path for shard loading. We vectorise the
    meth_full construction (forward block + rev_meth block) on the full shard
    at once via fancy-indexing instead of the per-row Python loop used in
    older code — ~50–100× faster on big shards.

    Args:
        data_dict:      The pickled shard dict.
        layout:         The :class:`SampleLayout` matching the shard's
                        geometry (validated by the caller via
                        :func:`read_shard_extraction_params`).
        num_meth_types: Number of methylation states (= max meth_id + 1).

    Returns:
        A dict with:

        - ``kmer_ids``       int64 (N,)  — int64 so it accommodates K up to 31
        - ``meth_ids``       int8  (N,)   meth at the prediction position
        - ``signals_log``    float32 tensor (N, 2) — already log1p
        - ``meth_full``      float32 (N, layout.params.total_meth_positions, num_meth_types)
                              pre-built so ``__iter__`` just slices it
        - ``parent_meths``   int8  (N,)   PARENT_METH
        - ``parent_offsets`` int8  (N,)   PARENT_OFFSET
        - ``categories``     int8  (N,)   CATEGORY
        - ``n_keys``, ``meth_counts``
    """
    kmer_size = layout.kmer_size
    n_rev_meth = layout.n_rev_meth
    pred_idx = layout.active_site_index
    total_pos = layout.params.total_meth_positions
    expected_ncols = layout.n_cols

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
        if samples.shape[1] != expected_ncols:
            raise ValueError(
                f"Shard row width mismatch — shard has {samples.shape[1]} "
                f"columns but the active SampleLayout expects "
                f"{expected_ncols} (kmer_size={kmer_size}, n_rev_meth="
                f"{n_rev_meth}). Re-extract under a matching geometry or "
                f"point training at the right shards."
            )
        kmer_id = int(key)
        ctx = samples[:, layout.col_meth_ctx_start : layout.col_meth_ctx_end].astype(np.uint8)
        rev = samples[:, layout.col_rev_meth : layout.col_category].astype(np.uint8)
        meth_at_center = ctx[:, pred_idx].astype(np.int8)
        n = len(samples)
        kmer_ids_list.append(np.full(n, kmer_id, dtype=np.int64))
        meth_ids_list.append(meth_at_center)
        signals_list.append(samples[:, :2].astype(np.float32))
        fractions_list.append(samples[:, layout.col_fraction].astype(np.float32))
        meth_ctx_list.append(ctx)
        rev_meth_list.append(rev)
        parent_meth_list.append(samples[:, layout.col_parent_meth].astype(np.int8))
        parent_offset_list.append(samples[:, layout.col_parent_offset].astype(np.int8))
        category_list.append(samples[:, layout.col_category].astype(np.int8))
        for mid in np.unique(meth_at_center):
            n_meth_counts[int(mid)] += int((meth_at_center == mid).sum())
        n_keys += 1

    if not kmer_ids_list:
        return {
            "kmer_ids": np.empty(0, dtype=np.int64),
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
    # hold both the lists AND the merged arrays at the same time (each
    # was ~half of peak memory before this fix).
    kmer_ids = np.concatenate(kmer_ids_list)
    kmer_ids_list = None
    meth_ids = np.concatenate(meth_ids_list)
    meth_ids_list = None
    fractions = np.concatenate(fractions_list).astype(np.float32)
    fractions_list = None
    meth_ctx = np.concatenate(meth_ctx_list)
    meth_ctx_list = None
    rev_meth = np.concatenate(rev_meth_list)
    rev_meth_list = None
    parent_meths = np.concatenate(parent_meth_list)
    parent_meth_list = None
    parent_offsets = np.concatenate(parent_offset_list)
    parent_offset_list = None
    categories = np.concatenate(category_list)
    category_list = None
    signals_log = log_transform(torch.from_numpy(np.concatenate(signals_list, axis=0)).float())
    signals_list = None

    # ── Vectorised meth_full construction (the big speedup) ────────────────
    # meth_full[i, pos, m] = 1.0 if mc/rev has meth m at pos, else 0.0.
    # except: at the PARENT meth position the value is replaced by the
    # per-row motif occupancy ``fractions[i]`` so the model can tell a
    # fully-methylated site (frac=1.0) from a partial one (frac=0.5).
    #
    # The parent position inside the kmer is ``pred_idx - parent_offset``:
    #   * SLOWED/m6A@+0 row: parent_offset = 0 → parent_pos = pred_idx
    #     (the methylation IS at the active site)
    #   * SLOWED/m6A@+5 row: parent_offset = 5 → parent_pos = pred_idx - 5
    #     (the methylation is 5 bases upstream of the active site)
    #   * NEAR_METH/m5C@+1 row: parent_offset = 1 → parent_pos = pred_idx - 1
    #
    # Before this fix, frac was applied only at pred_idx, so SLOWED rows
    # at non-zero offsets (m6A@+5, m5C@+2, m5C@+6, m4C@+0 when polymerase
    # is offset, NEAR_METH at any +k) ignored the occupancy entirely —
    # the model couldn't tell a real-methylation context from a half-
    # methylated motif. Fixing this is mostly a win for m5C/m4C which
    # have non-zero signal offsets and are typically partially methylated.
    n_rows = kmer_ids.shape[0]
    meth_full = np.zeros((n_rows, total_pos, num_meth_types), dtype=np.float32)

    # Forward block: scatter meth_ctx → meth_full[:, 0:kmer_size, :] as 1.0.
    rows_idx, cols_idx = np.where(meth_ctx > 0)
    if rows_idx.size:
        m_ids = meth_ctx[rows_idx, cols_idx].astype(np.int64)
        # Clamp to valid meth_id range to avoid IndexError on corrupted shards.
        m_ids = np.clip(m_ids, 0, num_meth_types - 1)
        meth_full[rows_idx, cols_idx, m_ids] = 1.0

    # Overwrite the PARENT meth position with the per-row fraction. Only
    # applies to non-baseline rows (baselines have parent_meths == 0 and
    # carry no occupancy info).
    non_baseline = parent_meths > 0
    if non_baseline.any():
        nb_rows = np.where(non_baseline)[0]
        parent_kmer_pos = pred_idx - parent_offsets[nb_rows].astype(np.int64)
        # Guard against rows where the parent fell outside the kmer window
        # (shouldn't happen with extract's parent_offset ∈ [0, K-1], but
        # cheap insurance against corrupted shards).
        valid = (parent_kmer_pos >= 0) & (parent_kmer_pos < kmer_size)
        if valid.any():
            nb_valid = nb_rows[valid]
            pp_valid = parent_kmer_pos[valid]
            pm_valid = np.clip(parent_meths[nb_valid].astype(np.int64), 0, num_meth_types - 1)
            meth_full[nb_valid, pp_valid, pm_valid] = fractions[nb_valid]

    # Rev_meth block: positions [kmer_size, kmer_size + n_rev_meth).
    rev_rows, rev_cols = np.where(rev_meth > 0)
    if rev_rows.size:
        rev_m_ids = rev_meth[rev_rows, rev_cols].astype(np.int64)
        rev_m_ids = np.clip(rev_m_ids, 0, num_meth_types - 1)
        meth_full[rev_rows, kmer_size + rev_cols, rev_m_ids] = 1.0

    # Free the source arrays now that meth_full holds the same info.
    del meth_ctx, rev_meth, fractions, rows_idx, cols_idx

    return {
        "kmer_ids": kmer_ids,
        "meth_ids": meth_ids,
        "signals_log": signals_log,
        "meth_full": meth_full,
        "parent_meths": parent_meths,
        "parent_offsets": parent_offsets,
        "categories": categories,
        "n_keys": n_keys,
        "meth_counts": dict(n_meth_counts),
    }


# ---------------------------------------------------------------------------
# Paired-positive augmentation (contrastive training — no fake labels)
# ---------------------------------------------------------------------------
#
# For each non-baseline row, we also yield a REAL baseline row of the same
# kmer (signal = real baseline IPD/PW). Pure data augmentation — same kmer,
# real data, real labels. Forces the model to see the contrast (meth /
# no-meth) on the same sequence, eliminating the "meth flag present → boost"
# shortcut without ever introducing mislabelled synthetic data.
#
# Biology-rule constraints (e.g. "m5C cannot occur on A") are enforced at
# the model architecture level by the biology_mask in ConvPredictor — NOT
# by faking baseline-labelled impossible inputs (which would dampen real
# signal). See predictor.py::_forward_conv for the mask.
#
# Inspired by SimCLR-style supervised contrastive pairing (Khosla 2020).


def _expand_with_pairs(flat: dict, augment_seed: int = 42) -> dict:
    """Return a new flat-dict where every non-baseline row is followed by
    one paired baseline row for the same kmer.

    Used by :class:`SignalDataset` (in-memory, expand-once-at-init).
    :class:`ShardedSignalDataset` does the same pairing at iter time so
    the random pair differs per epoch.
    """
    from ..utils.sample_layout import CATEGORY_BASELINE

    rng = np.random.default_rng(int(augment_seed))
    baseline_idx_by_kmer = _build_baseline_index(flat)

    n = len(flat["kmer_ids"])
    cats = flat["categories"]
    non_baseline_mask = cats != CATEGORY_BASELINE
    non_baseline_idx = np.where(non_baseline_mask)[0]

    augmentable = np.array(
        [int(flat["kmer_ids"][i]) in baseline_idx_by_kmer for i in non_baseline_idx],
        dtype=bool,
    )
    aug_src = non_baseline_idx[augmentable]
    if aug_src.size == 0:
        log.warning(
            "augment=True but no augmentable rows found (no kmer has both "
            "baseline AND non-baseline observations). Returning original."
        )
        return flat

    log.info(
        "Paired-positive augmentation: %d original rows + %d paired baselines = %d total",
        n,
        aug_src.size,
        n + aug_src.size,
    )

    # Pre-pick a baseline pair for each augmentable row.
    paired_baseline = np.empty(aug_src.size, dtype=np.int64)
    for j, src in enumerate(aug_src):
        pool = baseline_idx_by_kmer[int(flat["kmer_ids"][src])]
        paired_baseline[j] = int(rng.choice(pool))

    # Concatenate: ORIGINAL rows first, then PAIRS.
    out = {}
    out["n_keys"] = flat["n_keys"]
    out["meth_counts"] = dict(flat["meth_counts"])
    for k, src_dtype in [
        ("kmer_ids", flat["kmer_ids"].dtype),
        ("meth_ids", flat["meth_ids"].dtype),
        ("parent_meths", flat["parent_meths"].dtype),
        ("parent_offsets", flat["parent_offsets"].dtype),
        ("categories", flat["categories"].dtype),
    ]:
        original = flat[k]
        from_pair = original[paired_baseline]
        out[k] = np.concatenate([original, from_pair]).astype(src_dtype)

    sig = flat["signals_log"]
    sig_pair = sig[paired_baseline]
    out["signals_log"] = torch.cat([sig, sig_pair], dim=0)

    pair_mf = flat["meth_full"][paired_baseline]
    out["meth_full"] = np.concatenate([flat["meth_full"], pair_mf], axis=0)
    return out


def _build_baseline_index(flat: dict) -> dict[int, np.ndarray]:
    """Build a ``kmer_id → array(int)`` map of indices into the flat arrays
    pointing at rows whose category == CATEGORY_BASELINE. Only kmers that
    have at least one baseline row are present.
    """
    from ..utils.sample_layout import CATEGORY_BASELINE

    cats = flat["categories"]
    kmer_ids = flat["kmer_ids"]
    baseline_mask = cats == CATEGORY_BASELINE
    if not baseline_mask.any():
        return {}
    baseline_idx = np.where(baseline_mask)[0]
    baseline_kmer_ids = kmer_ids[baseline_mask]
    # Group indices by kmer_id: build a sort-based grouping (fast for large N).
    order = np.argsort(baseline_kmer_ids, kind="stable")
    sorted_kids = baseline_kmer_ids[order]
    sorted_indices = baseline_idx[order]
    unique_kids, starts = np.unique(sorted_kids, return_index=True)
    ends = np.append(starts[1:], len(sorted_kids))
    return {int(uk): sorted_indices[s:e] for uk, s, e in zip(unique_kids, starts, ends)}


# NOTE: the per-row Python builder ``_build_meth_full`` used to live here.
# Removed in v0.5 — it was a relic of the pre-v0.4 in-loop dataset and
# has been replaced by the vectorised ``_flatten_data_dict`` block above
# which builds the entire shard's meth_full tensor in one pass.


# ---------------------------------------------------------------------------
# In-memory dataset (small datasets — single .pkl)
# ---------------------------------------------------------------------------


class SignalDataset(Dataset):
    """Flat-sample dataset for KinSim training, in-memory.

    Loads a single merged .pkl entirely into RAM, partitions every kmer's
    rows by ``meth_id`` at the prediction position, and pre-flattens all
    rows into contiguous arrays. The ``DataLoader`` shuffles the flat
    index each epoch — every sample is seen exactly once per epoch.

    Use this for small datasets that fit in RAM. For larger corpora use
    :class:`ShardedSignalDataset`.

    Args:
        pkl_path:       Single shard pkl to load fully into RAM.
        num_meth_types: Number of methylation states (default 4).
        augment:        Enable offline paired-positive augmentation.
        augment_seed:   PRNG seed for the augmentation pair picker.
        params:         Window-geometry record. If ``None``, read from
                        shard meta then YAML.
    """

    def __init__(
        self,
        pkl_path: str,
        num_meth_types: int = 4,
        augment: bool = False,
        augment_seed: int = 42,
        params: ExtractionParams | None = None,
    ) -> None:
        log.info("Loading training data from %s ...", pkl_path)
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        if not isinstance(data_dict, dict):
            raise TypeError(f"Expected a dict from {pkl_path}, got {type(data_dict).__name__}.")
        if len(data_dict) == 0:
            raise ValueError(f"The .pkl file is empty: {pkl_path}")

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
            raise ValueError(f"No int-keyed kmer data found in {pkl_path}")

        if augment:
            flat = _expand_with_pairs(flat, augment_seed=augment_seed)

        self._num_meth_types = num_meth_types
        self._params = resolved
        self._layout = layout
        self._kmer_size = resolved.kmer_size
        self._augment = augment
        self._kmer_ids = flat["kmer_ids"]
        self._meth_ids = flat["meth_ids"]
        self._signals = flat["signals_log"]
        self._meth_full = flat["meth_full"]
        self._parent_meths = flat["parent_meths"]
        self._parent_offsets = flat["parent_offsets"]
        self._categories = flat["categories"]

        n_total = len(self._kmer_ids)
        from ..utils.encoding import get_meth_ids

        _name_by_id = {v: k for k, v in get_meth_ids().items()}
        _name_by_id[0] = "unmeth"
        meth_summary = ", ".join(
            f"{_name_by_id.get(m, str(m))}={flat['meth_counts'][m]:,}"
            for m in sorted(flat["meth_counts"])
            if flat["meth_counts"][m] > 0
        )
        log.info(
            "SignalDataset ready: %s unique kmers, %s samples [%s]",
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
            torch.tensor(int(self._meth_ids[idx]), dtype=torch.long),
            torch.tensor(int(self._parent_meths[idx]), dtype=torch.long),
            torch.tensor(int(self._parent_offsets[idx]), dtype=torch.long),
            torch.tensor(int(self._categories[idx]), dtype=torch.long),
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

    Output rows are identical to :class:`SignalDataset` —
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
        shuffle: bool = True,
        seed: int = 42,
        augment: bool = False,
        balance_kmers: bool = False,
        params: ExtractionParams | None = None,
    ) -> None:
        super().__init__()
        self._shard_paths = [str(Path(p)) for p in shard_paths]
        if not self._shard_paths:
            raise ValueError("shard_paths is empty")
        self._num_meth_types = num_meth_types
        # Resolve geometry once: explicit `params` > first shard's meta > YAML.
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
        self._augment = bool(augment)
        # When ``balance_kmers`` is on, each shard is sub-sampled by a
        # per-(kmer_id, category) weighted draw before yielding. This
        # equalises the influence of rare kmers / rare categories in the
        # gradient — see He+Garcia 2009, Cui+ 2019. ``shuffle`` must be
        # True for balancing to do anything useful.
        self._balance_kmers = bool(balance_kmers)
        log.info(
            "ShardedSignalDataset ready: %d shards  geometry=(K=%d, "
            "upstream=%d, downstream=%d, rev_meth=%s)  shuffle=%s  "
            "augment=%s  balance_kmers=%s",
            len(self._shard_paths),
            params.kmer_size,
            params.upstream,
            params.downstream,
            list(params.rev_meth_offsets),
            shuffle,
            self._augment,
            self._balance_kmers,
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
        from ..utils.sample_layout import CATEGORY_BASELINE

        my_shards, rng = self._worker_shards_and_rng()
        cap = getattr(self, "_max_rows_per_shard", None)
        augment = bool(getattr(self, "_augment", False))
        balance_kmers = bool(getattr(self, "_balance_kmers", False))
        for shard_path in my_shards:
            with open(shard_path, "rb") as f:
                data_dict = pickle.load(f)
            # Re-verify each shard against the dataset's resolved geometry.
            # Per-shard check (rather than just the first) catches mixed
            # corpora — e.g. an old K=11 shard accidentally left in a K=21
            # refined/ directory.
            shard_params = read_shard_extraction_params(data_dict)
            if shard_params is not None:
                self._params.assert_compatible(shard_params, where=f"shard {shard_path}")
            flat = _flatten_data_dict(
                data_dict,
                self._layout,
                num_meth_types=self._num_meth_types,
            )
            del data_dict  # release the loaded shard before iterating
            n = len(flat["kmer_ids"])
            if n == 0:
                continue

            if balance_kmers and n > 1:
                # Per-(kmer_id, category) inverse-frequency draw — rare
                # (kmer, category) groups (m4C/m5C, rare kmers) get more
                # exposure than common ones (baseline of common kmers).
                #
                # Two non-trivial choices below:
                # * `1/sqrt(count)` (not raw `1/count`) — softens the
                #   weighting so a singleton group doesn't get 1000×
                #   weight and dominate the batch gradient.
                # * `replace=True` — with `replace=False` and
                #   `size = n`, `np.random.choice` ignores `p` and just
                #   returns a permutation. With replacement the weights
                #   actually bias every draw, so rare rows ARE seen more
                #   per epoch. Some common rows are skipped (random) and
                #   re-visited in later epochs — fine for training.
                composite = flat["kmer_ids"].astype(np.int64) * 4 + flat["categories"].astype(
                    np.int64
                )
                counts = np.bincount(composite)
                row_w = 1.0 / np.sqrt(np.maximum(counts[composite], 1).astype(np.float64))
                row_w /= row_w.sum()
                target_size = min(cap, n) if (cap is not None and cap > 0) else n
                order = rng.choice(n, size=target_size, replace=True, p=row_w)
            else:
                order = np.arange(n)
                if self._shuffle:
                    rng.shuffle(order)
                if cap is not None and 0 < cap < n:
                    order = order[:cap]

            # Build per-shard baseline index for augment lookups
            baseline_idx_by_kmer = _build_baseline_index(flat) if augment else {}

            # Convert the numpy arrays to torch once per shard.
            kmer_ids_t = torch.from_numpy(flat["kmer_ids"].astype(np.int64))
            meth_ids_t = torch.from_numpy(flat["meth_ids"].astype(np.int64))
            parent_meths_t = torch.from_numpy(flat["parent_meths"].astype(np.int64))
            parent_offsets_t = torch.from_numpy(flat["parent_offsets"].astype(np.int64))
            categories_t = torch.from_numpy(flat["categories"].astype(np.int64))
            meth_full_t = torch.from_numpy(flat["meth_full"])  # already float32
            signals_t = flat["signals_log"]  # already torch tensor

            for idx in order:
                i = int(idx)
                # 1. Original row.
                yield (
                    kmer_ids_t[i],
                    meth_full_t[i],
                    signals_t[i],
                    meth_ids_t[i],
                    parent_meths_t[i],
                    parent_offsets_t[i],
                    categories_t[i],
                )

                # 2. Paired-positive augmentation: for non-baseline rows,
                # also yield a REAL baseline row of the same kmer (real
                # data + real label). Forces meth/no-meth contrast on the
                # same sequence — no mislabelled synthetic data. Random
                # pair per epoch keeps the signal diverse.
                #
                # Biology constraints (impossible base × meth_id combos)
                # are handled by the biology_mask in ConvPredictor — NOT
                # by faking baseline labels on impossible inputs.
                if augment and int(categories_t[i].item()) != CATEGORY_BASELINE:
                    kmer = int(kmer_ids_t[i].item())
                    pool = baseline_idx_by_kmer.get(kmer)
                    if pool is not None and len(pool) > 0:
                        pair_idx = int(rng.choice(pool))
                        yield (
                            kmer_ids_t[pair_idx],
                            meth_full_t[pair_idx],
                            signals_t[pair_idx],
                            meth_ids_t[pair_idx],
                            parent_meths_t[pair_idx],
                            parent_offsets_t[pair_idx],
                            categories_t[pair_idx],
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
