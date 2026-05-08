"""Build a per-kmer (IPD, PW) empirical table from KinSim extract shards.

Walks every ``*_shard.pkl`` in a directory, filters rows where
``CATEGORY == CATEGORY_BASELINE`` (positions ≥ K bases from any
methylation motif by extract's construction → guaranteed unmodified
kinetics), and populates a per-kmer reservoir of empirical (IPD, PW)
byte pairs.

Why only BASELINE rows?
-----------------------

We want a CLEAN per-kmer null distribution — what the polymerase does
at this kmer when no methylation is involved. Any reasonable per-kmer
sampling model should not be contaminated by reads at methylated motif
positions, because those carry the methylation kinetic signature
mixed in. Extract's BASELINE category is exactly this: positions that
are at least K bases away from any motif occurrence, sampled at the
``baseline_sample_rate`` (default 0.10) and reservoir-capped at
``n_baseline_per_kmer`` (default 50) per strain. With 65 strains that
gives up to ~3,250 truly-unmodified samples per saturated kmer —
plenty to fit an empirical sampling generator.

Shard order is **shuffled deterministically** before walking so the
``first n_per_kmer`` selection per kmer draws from a random subset of
strains rather than always the alphabetically-first ones.

A separate **global pool** of (IPD, PW) pairs (default 1M) is also
collected as a cross-kmer fallback for kmers absent from the table.

Memory: 4194304 × n_per_kmer × 2 bytes ≈ 1.6 GB at n_per_kmer=200.
Walltime: ~5–15 min on a 65-shard directory (5 GB each).

CLI::

    python -m kinsim_baseline build SHARDS_DIR OUTPUT_NPZ \\
        [--n-per-kmer 200] [--n-global-pool 1000000] [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np

from kinsim.utils.sample_layout import (
    CATEGORY_BASELINE,
    COL_CATEGORY,
    COL_IPD,
    COL_PW,
)

from .distribution import NUM_KMERS, KmerDistribution

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vectorised "first-N wins" per-kmer reservoir update
# ---------------------------------------------------------------------------


def _fill_batch(
    ipd_table: np.ndarray,
    pw_table: np.ndarray,
    count: np.ndarray,
    n_per_kmer: int,
    kmer_ids: np.ndarray,
    ipds: np.ndarray,
    pws: np.ndarray,
) -> int:
    """Append new (kmer, ipd, pw) triples to per-kmer banks (first-N wins).

    Within a single batch, multiple triples for the same kmer are
    placed in sequential slots starting at ``count[kmer]``. Triples
    that would overflow ``n_per_kmer`` are dropped silently.

    Returns the number of triples actually written.
    """
    n = kmer_ids.size
    if n == 0:
        return 0

    # Sort by kmer so same-kmer triples are contiguous.
    order = np.argsort(kmer_ids, kind="stable")
    k_sorted = kmer_ids[order]
    i_sorted = ipds[order]
    p_sorted = pws[order]

    # Within-group offset: position in sorted array minus group's start.
    same_as_prev = np.concatenate(([False], k_sorted[1:] == k_sorted[:-1]))
    new_group = ~same_as_prev
    group_id = np.cumsum(new_group) - 1
    group_start = np.where(new_group)[0]
    offset_in_group = np.arange(n, dtype=np.int64) - group_start[group_id]

    # Slot = current count + offset; reject overflows.
    slot = count[k_sorted].astype(np.int64) + offset_in_group
    accept = slot < n_per_kmer
    if not accept.any():
        return 0

    write_k = k_sorted[accept]
    write_s = slot[accept]
    ipd_table[write_k, write_s] = i_sorted[accept]
    pw_table[write_k, write_s] = p_sorted[accept]

    # np.add.at handles duplicate indices correctly (unbuffered scatter).
    np.add.at(count, write_k, 1)
    return int(accept.sum())


def _global_reservoir_update(
    pool_ipd: np.ndarray,
    pool_pw: np.ndarray,
    pool_size: int,
    seen_so_far: int,
    new_ipds: np.ndarray,
    new_pws: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Vectorised reservoir update for the global cross-kmer fallback pool.

    Returns the new value of ``seen_so_far``.
    """
    n = new_ipds.size
    if n == 0 or pool_size == 0:
        return seen_so_far + n

    ks = np.arange(n, dtype=np.int64) + seen_so_far + 1
    idxs = (rng.random(n) * ks).astype(np.int64)
    accept = idxs < pool_size
    if accept.any():
        pool_ipd[idxs[accept]] = new_ipds[accept]
        pool_pw[idxs[accept]] = new_pws[accept]
    return seen_so_far + n


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_table(
    shards_dir: Path,
    n_per_kmer: int = 200,
    n_global_pool: int = 1_000_000,
    seed: int = 42,
) -> KmerDistribution:
    """Build a ``KmerDistribution`` from a directory of extract shards."""
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    shard_paths = sorted(Path(shards_dir).glob("*_shard.pkl"))
    if not shard_paths:
        log.error("No *_shard.pkl in %s", shards_dir)
        sys.exit(1)
    py_rng.shuffle(shard_paths)
    log.info(
        "Building from %d shards (shuffled, seed=%d)  per-kmer cap=%d  "
        "global pool=%d",
        len(shard_paths), seed, n_per_kmer, n_global_pool,
    )

    ipd_table = np.zeros((NUM_KMERS, n_per_kmer), dtype=np.uint8)
    pw_table = np.zeros((NUM_KMERS, n_per_kmer), dtype=np.uint8)
    count = np.zeros(NUM_KMERS, dtype=np.uint16)

    pool_ipd = np.zeros(n_global_pool, dtype=np.uint8)
    pool_pw = np.zeros(n_global_pool, dtype=np.uint8)
    pool_seen = 0

    n_baseline_total = 0
    for i, sp in enumerate(shard_paths, 1):
        with open(sp, "rb") as f:
            data = pickle.load(f)

        n_in_shard = 0
        for kid, arr in data.items():
            if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
                continue
            if arr.shape[1] <= COL_PW:
                continue
            mask = arr[:, COL_CATEGORY].astype(np.int8) == CATEGORY_BASELINE
            if not mask.any():
                continue
            ipds = arr[mask, COL_IPD].astype(np.uint8)
            pws = arr[mask, COL_PW].astype(np.uint8)
            n_baseline_total += ipds.size
            n_in_shard += ipds.size

            kmer_ids = np.full(ipds.size, int(kid), dtype=np.int64)
            _fill_batch(
                ipd_table, pw_table, count, n_per_kmer,
                kmer_ids, ipds, pws,
            )
            pool_seen = _global_reservoir_update(
                pool_ipd, pool_pw, n_global_pool, pool_seen,
                ipds, pws, rng,
            )
        log.info(
            "[%d/%d] %s  +%d baseline rows  (kmer-coverage now %.2f%%, "
            "saturated %d)",
            i, len(shard_paths), sp.name, n_in_shard,
            100.0 * (count > 0).mean(),
            int((count == n_per_kmer).sum()),
        )
        del data

    n_global_actual = min(pool_seen, n_global_pool)
    n_kmers_seen = int((count > 0).sum())
    sat = int((count == n_per_kmer).sum())
    log.info(
        "Built table: %d baseline rows total | %d kmers covered (%.2f%%) | "
        "%d saturated at cap %d | %d global-pool entries",
        n_baseline_total, n_kmers_seen, 100.0 * n_kmers_seen / NUM_KMERS,
        sat, n_per_kmer, n_global_actual,
    )

    return KmerDistribution(
        ipd=ipd_table,
        pw=pw_table,
        count=count,
        global_ipd=pool_ipd[:n_global_actual] if n_global_actual > 0 else None,
        global_pw=pool_pw[:n_global_actual] if n_global_actual > 0 else None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline build",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("shards_dir",
                   help="Directory of *_shard.pkl files (from `kinsim extract`).")
    p.add_argument("output_npz", help="Output .npz table.")
    p.add_argument("--n-per-kmer", type=int, default=200,
                   help="Max empirical samples per kmer (default 200).")
    p.add_argument("--n-global-pool", type=int, default=1_000_000,
                   help="Cross-kmer fallback pool size (default 1M). 0 disables.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    table = build_table(
        Path(args.shards_dir),
        n_per_kmer=args.n_per_kmer,
        n_global_pool=args.n_global_pool,
        seed=args.seed,
    )
    table.save(args.output_npz)
    log.info(
        "Saved: %s  (coverage %.2f%%, %d global-pool entries)",
        args.output_npz,
        100.0 * table.coverage(),
        table.n_global_pool(),
    )


if __name__ == "__main__":
    main()
