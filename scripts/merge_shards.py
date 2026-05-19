"""Merge multiple shard .pkl into a single master.pkl.

Concatenates per-kmer arrays from each input shard. Optionally caps the
total number of samples per kmer (Vitter-style reservoir kept for the
older `kinsim extract --merge-shards` path; here we use a simpler
uniform subsample because all shards have already passed extract's
per-shard reservoir cap).

Examples::

    # Fuse all refined shards into one master
    python scripts/merge_shards.py \
        /data/.../refined/*_clean.pkl \
        --output /data/.../refined/master_clean.pkl

    # Same, with cap of 50 000 samples per kmer
    python scripts/merge_shards.py \
        /data/.../refined/*_clean.pkl \
        --output /data/.../refined/master_clean.pkl \
        --max-per-key 50000

CLI:
    python scripts/merge_shards.py <input.pkl> [<input.pkl> ...] --output <out.pkl> [--max-per-key N] [--seed S]
"""

from __future__ import annotations

import argparse
import glob
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def merge_shards(
    input_paths: list[str],
    output_path: str,
    max_per_key: int | None = None,
    seed: int = 0,
) -> dict:
    """Concatenate kmer arrays across shards. Returns the merged dict."""
    rng = np.random.default_rng(seed)
    merged: dict = {}
    meta_taken = None
    n_shards = 0
    n_total_in = 0

    for raw in input_paths:
        # Expand wildcards so the user can pass shell-quoted globs too.
        for path in sorted(glob.glob(raw)) if any(c in raw for c in "*?[") else [raw]:
            p = Path(path)
            if not p.is_file():
                log.warning("skip %s — not a file", path)
                continue
            log.info("loading %s", p)
            with open(p, "rb") as f:
                data = pickle.load(f)
            if meta_taken is None and isinstance(data.get("__meta__"), dict):
                meta_taken = dict(data["__meta__"])
            for k, v in data.items():
                if not isinstance(k, (int, np.integer)) or not isinstance(v, np.ndarray):
                    continue
                ki = int(k)
                n_total_in += len(v)
                if ki in merged:
                    merged[ki] = np.concatenate([merged[ki], v])
                else:
                    merged[ki] = v
            n_shards += 1

    if n_shards == 0:
        log.error("No shard files matched.")
        sys.exit(1)

    # Optional per-key subsample.
    if max_per_key is not None and max_per_key > 0:
        n_capped = 0
        for k, arr in list(merged.items()):
            if len(arr) > max_per_key:
                idx = rng.choice(len(arr), size=max_per_key, replace=False)
                merged[k] = arr[idx]
                n_capped += 1
        log.info("Subsampled %d kmers to %d rows each", n_capped, max_per_key)

    # Bump __meta__ with provenance.
    meta = meta_taken or {}
    meta["merged_from_n_shards"] = n_shards
    meta["merged_total_input_rows"] = n_total_in
    if max_per_key:
        meta["merged_max_per_key"] = int(max_per_key)
    merged["__meta__"] = meta

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "Writing %s — %d kmers, %d total rows (from %d shards × %d input rows)",
        out,
        len(merged) - (1 if "__meta__" in merged else 0),
        sum(len(v) for v in merged.values() if isinstance(v, np.ndarray)),
        n_shards,
        n_total_in,
    )
    with open(out, "wb") as f:
        pickle.dump(merged, f)
    return merged


def main(argv=None) -> None:
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python scripts/merge_shards.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="Input shard .pkl files (globs OK).")
    p.add_argument("--output", required=True, help="Output master .pkl path.")
    p.add_argument(
        "--max-per-key",
        type=int,
        default=None,
        help="Cap samples per kmer in the output (default: no cap).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)
    merge_shards(args.inputs, args.output, max_per_key=args.max_per_key, seed=args.seed)


if __name__ == "__main__":
    main()
