"""Filter a .pkl training file by coverage, modification type, or key count.

Filtering workflow:
  1. Master .pkl  -- extracted from ALL BAMs for ALL motifs (kinsim extract + merge).
     No filtering. Complete reference.
  2. Training .pkl -- filtered subset with configurable thresholds.

This module provides the filtering step, allowing reproducible and adjustable
filtering without re-extracting from BAMs.

Filtering criteria (all optional, combinable):
  --min-coverage   Minimum samples per (kmer, meth) key
  --min-fraction   Minimum mean methylation fraction for methylated keys (default 0.0)
                   Applied only to keys with meth_id != 0 (unmethylated keys always kept).
                   Requires a 3-column shard (IPD, PW, fraction); 2-col shards are skipped.
  --mod-type       Keep only specific mod types (m6A, m5C, m4C, or comma-sep)
  --max-keys       Keep only top N most data-rich keys

CLI:
    python scripts/filter.py general.pkl training.pkl [--min-coverage 50] [--min-fraction 0.4] [...]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

from kinsim.utils.encoding import METH_IDS

log = logging.getLogger(__name__)

# Inverse mapping: int -> mod type name
_METH_NAMES = {v: k for k, v in METH_IDS.items()}


def filter_pkl(
    input_path: str,
    output_path: str,
    *,
    min_coverage: int = 0,
    min_fraction: float = 0.0,
    mod_types: list[str] | None = None,
    max_keys: int = 0,
) -> dict:
    """Filter a General Dictionary .pkl into a Training Dictionary .pkl.

    Args:
        input_path:   Path to the General Dictionary .pkl file.
        output_path:  Path to write the filtered Training Dictionary .pkl.
        min_coverage: Minimum number of samples per (kmer, meth) key.
                      Keys with fewer samples are dropped.
        min_fraction: Minimum mean methylation fraction for methylated keys
                      (meth_id != 0). Keys whose mean fraction column is below
                      this threshold are dropped. Unmethylated keys (meth_id=0)
                      are never filtered by this criterion. Requires 3-column
                      data; 2-column shards are silently kept regardless.
                      Default: 0.0 (no filtering).
        mod_types:    List of mod type names to keep (e.g. ["m6A", "m5C"]).
                      None or empty means keep all.
        max_keys:     Maximum number of keys to keep (0 = unlimited).
                      Keys are sorted by sample count (descending), so the
                      most data-rich keys are retained.

    Returns:
        dict with filtering statistics: keys_in, keys_out, samples_in, samples_out,
        fraction_dropped (count of keys dropped by min_fraction filter).
    """
    import numpy as np

    input_file = Path(input_path)
    if not input_file.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    log.info("Loading General Dictionary: %s", input_path)
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    # Separate metadata from data keys
    meta = data.pop("__meta__", None)

    # Resolve mod type filter to integer IDs
    allowed_meth_ids: set[int] | None = None
    if mod_types:
        allowed_meth_ids = set()
        for mt in mod_types:
            mt = mt.strip()
            if mt in METH_IDS:
                allowed_meth_ids.add(METH_IDS[mt])
            else:
                log.warning(
                    "Unknown mod type '%s' -- ignored. Valid: %s", mt, list(METH_IDS.keys())
                )

    keys_in = len(data)
    samples_in = sum(len(v) for v in data.values() if isinstance(v, np.ndarray))

    # Apply filters
    filtered = {}
    fraction_dropped = 0
    for key, value in data.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue

        _kmer_id, meth_id = key

        # Mod type filter
        if allowed_meth_ids is not None and meth_id not in allowed_meth_ids:
            continue

        # Coverage filter
        if not isinstance(value, np.ndarray):
            continue
        if len(value) < min_coverage:
            continue

        # Fraction filter: only for methylated keys with a fraction column
        if min_fraction > 0.0 and meth_id != 0 and value.ndim == 2 and value.shape[1] >= 3:
            mean_frac = float(np.mean(value[:, 2]))
            if mean_frac < min_fraction:
                fraction_dropped += 1
                continue

        filtered[key] = value

    # Max keys filter (keep the most data-rich)
    if max_keys > 0 and len(filtered) > max_keys:
        sorted_keys = sorted(filtered.keys(), key=lambda k: len(filtered[k]), reverse=True)
        filtered = {k: filtered[k] for k in sorted_keys[:max_keys]}

    keys_out = len(filtered)
    samples_out = sum(len(v) for v in filtered.values() if isinstance(v, np.ndarray))

    # Re-attach metadata with filter provenance
    if meta is not None:
        if isinstance(meta, dict):
            meta["filtered_from"] = str(input_path)
            meta["filter_params"] = {
                "min_coverage": min_coverage,
                "min_fraction": min_fraction,
                "mod_types": mod_types,
                "max_keys": max_keys,
            }
        filtered["__meta__"] = meta

    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(filtered, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats = {
        "keys_in": keys_in,
        "keys_out": keys_out,
        "samples_in": samples_in,
        "samples_out": samples_out,
        "fraction_dropped": fraction_dropped,
    }

    log.info(
        "Filtered: %d -> %d keys, %d -> %d samples", keys_in, keys_out, samples_in, samples_out
    )
    if fraction_dropped:
        log.info(
            "  Dropped by min_fraction=%.2f: %d methylated keys", min_fraction, fraction_dropped
        )
    log.info("Training Dictionary written to: %s", output_path)

    return stats


def main(argv=None) -> None:
    from kinsim.utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="python scripts/filter.py",
        description=(
            "Filter a General Dictionary .pkl into a Training Dictionary.\n\n"
            "The General Dictionary contains ALL extracted kinetic data.\n"
            "The Training Dictionary is a filtered subset used for model training.\n\n"
            "Filtering is reproducible: re-run with different thresholds without\n"
            "re-extracting from BAMs.\n\n"
            "Examples:\n"
            "  # Keep only well-covered methylated keys:\n"
            "  python scripts/filter.py general.pkl training.pkl --min-coverage 50\n\n"
            "  # Keep methylated keys with >=40% mean fraction:\n"
            "  python scripts/filter.py general.pkl training.pkl --min-fraction 0.4\n\n"
            "  # Keep only m6A data:\n"
            "  python scripts/filter.py general.pkl training.pkl --mod-type m6A\n\n"
            "  # Keep top 100k most data-rich keys, m6A and m5C only:\n"
            "  python scripts/filter.py general.pkl training.pkl \\\n"
            "      --mod-type m6A,m5C --max-keys 100000 --min-coverage 10 --min-fraction 0.4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="General Dictionary .pkl file")
    parser.add_argument("output", help="Output Training Dictionary .pkl file")
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=0,
        help="Minimum samples per (kmer, meth) key (default: 0)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.0,
        help="Minimum mean methylation fraction for methylated keys "
        "(0.0–1.0, default: 0.0). Keys with meth_id!=0 and mean "
        "fraction below this are dropped. Unmethylated keys are unaffected.",
    )
    parser.add_argument(
        "--mod-type",
        type=str,
        default=None,
        help="Comma-separated mod types to keep (e.g. m6A,m5C). Default: keep all.",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=0,
        help="Maximum number of keys to retain (0 = unlimited). Keeps the most data-rich keys.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    mod_types = None
    if args.mod_type:
        mod_types = [mt.strip() for mt in args.mod_type.split(",")]

    stats = filter_pkl(
        args.input,
        args.output,
        min_coverage=args.min_coverage,
        min_fraction=args.min_fraction,
        mod_types=mod_types,
        max_keys=args.max_keys,
    )

    # Summary to stdout
    pct_keys = stats["keys_out"] / stats["keys_in"] * 100 if stats["keys_in"] > 0 else 0
    pct_samples = stats["samples_out"] / stats["samples_in"] * 100 if stats["samples_in"] > 0 else 0
    print(
        f"Keys:    {stats['keys_in']:>10,} -> {stats['keys_out']:>10,}  ({pct_keys:.1f}% retained)"
    )
    if stats.get("fraction_dropped"):
        print(
            f"  Dropped by --min-fraction {args.min_fraction}: "
            f"{stats['fraction_dropped']:,} methylated keys"
        )
    print(
        f"Samples: {stats['samples_in']:>10,} -> {stats['samples_out']:>10,}  "
        f"({pct_samples:.1f}% retained)"
    )
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()
