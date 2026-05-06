"""Balance a merged training .pkl for fair representation of methylation types.

Addresses the natural imbalance in kinetic data where unmethylated contexts
vastly outnumber methylated ones, and where one mod type (e.g. m6A) may dwarf
others (e.g. m4C, m5C).

Balancing strategy
------------------
1. Key-level balance:
   - Target `--meth-fraction` of kept keys to be methylated (default 0.5).
   - Methylated budget split evenly across ALL mod types present in the dict
     (m6A, m4C, m5C), so no type is drowned out.
   - If a mod type has fewer keys than its share of the budget, the remainder
     is redistributed to other types (no waste).
   - Unmethylated keys fill the rest of the budget, chosen by coverage
     (most-sampled first — more data-rich = more reliable signal).

2. Per-key sample diversity:
   - When `--samples-per-key N` is set, N samples are selected from each key
     to maximise spread across the IPD distribution.
   - Implementation: sort samples by IPD, take N evenly-spaced quantiles.
     This covers the full range of the IPD distribution rather than
     over-representing the mode.

CLI:
    python scripts/balance.py merged.pkl balanced.pkl \\
        --meth-fraction 0.5 \\
        --max-keys 200000 \\
        --samples-per-key 200
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from kinsim.utils.encoding import METH_IDS

log = logging.getLogger(__name__)

_METH_NAMES = {v: k for k, v in METH_IDS.items()}
_NONE_ID = METH_IDS["none"]


def _diversity_subsample(arr: np.ndarray, n: int) -> np.ndarray:
    """Return n rows from arr chosen to maximise spread over the IPD axis.

    Sorts by IPD (column 0) and picks n evenly-spaced quantile indices,
    giving coverage of the full IPD distribution rather than clustering
    around the mode.
    """
    if len(arr) <= n:
        return arr
    order = np.argsort(arr[:, 0])
    indices = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
    return arr[order[indices]]


def balance_pkl(
    input_path: str,
    output_path: str,
    *,
    meth_fraction: float = 0.5,
    max_keys: int = 0,
    samples_per_key: int = 0,
) -> dict:
    """Balance a merged .pkl dictionary for fair methylation representation.

    Args:
        input_path:      Path to merged General/Training Dictionary .pkl.
        output_path:     Path to write the balanced output .pkl.
        meth_fraction:   Target fraction of kept keys that are methylated
                         (meth_id != 0). Range [0, 1], default 0.5.
                         Methylated budget is split evenly across all mod
                         types present. If a type has fewer keys than its
                         budget share, excess is redistributed.
        max_keys:        Total key budget (0 = keep all keys, only balance
                         the type distribution).
        samples_per_key: After key selection, subsample each key to this many
                         samples using IPD-quantile diversity selection
                         (0 = keep all samples per key).

    Returns:
        dict with stats: keys_in, keys_out, samples_in, samples_out,
                         keys_by_type_in, keys_by_type_out.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    log.info("Loading dictionary: %s", input_path)
    with open(input_file, "rb") as f:
        data = pickle.load(f)

    meta = data.pop("__meta__", None)

    # Separate data keys from metadata
    keyed = {
        k: v
        for k, v in data.items()
        if isinstance(k, tuple) and len(k) == 2 and isinstance(v, np.ndarray)
    }

    keys_in = len(keyed)
    samples_in = sum(len(v) for v in keyed.values())

    # Group by meth_id
    by_meth: dict[int, list] = defaultdict(list)
    for key in keyed:
        _, meth_id = key
        by_meth[meth_id].append(key)

    unmeth_keys = by_meth.get(_NONE_ID, [])
    meth_by_type: dict[int, list] = {mid: keys for mid, keys in by_meth.items() if mid != _NONE_ID}
    present_types = sorted(meth_by_type.keys())
    total_meth_keys = sum(len(v) for v in meth_by_type.values())

    log.info(
        "Keys in dict: %d total  (%d unmeth, %d meth across %d types)",
        keys_in,
        len(unmeth_keys),
        total_meth_keys,
        len(present_types),
    )
    for mid in present_types:
        log.info("  %s: %d keys", _METH_NAMES.get(mid, mid), len(meth_by_type[mid]))

    # ---- Key-level budget calculation ----
    if max_keys <= 0:
        # Derive total budget from methylated key count so meth_fraction is
        # actually achieved: total = meth_keys / meth_fraction
        # e.g. 6K meth keys at 50% → 12K total → 6K unmeth kept, 2M dropped
        if total_meth_keys > 0 and 0 < meth_fraction < 1:
            total_available = min(keys_in, round(total_meth_keys / meth_fraction))
        else:
            total_available = keys_in
    else:
        total_available = min(max_keys, keys_in)

    n_meth_budget = min(round(total_available * meth_fraction), total_meth_keys)
    n_unmeth_budget = total_available - n_meth_budget

    # Distribute meth budget evenly across mod types; redistribute leftovers
    if present_types:
        per_type_budget = n_meth_budget // len(present_types)
        remainder = n_meth_budget - per_type_budget * len(present_types)

        type_targets: dict[int, int] = {}
        leftover = 0
        for mid in present_types:
            cap = len(meth_by_type[mid])
            allocated = min(per_type_budget, cap)
            type_targets[mid] = allocated
            leftover += per_type_budget - allocated  # unused budget from small types

        # Add remainder + leftover back to types that still have capacity
        extra = remainder + leftover
        for mid in present_types:
            if extra <= 0:
                break
            cap = len(meth_by_type[mid])
            can_add = cap - type_targets[mid]
            add = min(can_add, extra)
            type_targets[mid] += add
            extra -= add
    else:
        type_targets = {}

    log.info("Key budget: %d total  (%.0f%% meth target)", total_available, meth_fraction * 100)
    for mid, n in type_targets.items():
        log.info("  %s: keep %d / %d keys", _METH_NAMES.get(mid, mid), n, len(meth_by_type[mid]))
    log.info("  none: keep %d / %d keys", min(n_unmeth_budget, len(unmeth_keys)), len(unmeth_keys))

    # ---- Select keys ----
    def _select_keys(candidates: list, n: int) -> list:
        """Keep up to n keys, preferring those with more samples (data-rich)."""
        if len(candidates) <= n:
            return candidates
        # Sort by sample count descending — most data-rich keys first
        ranked = sorted(candidates, key=lambda k: len(keyed[k]), reverse=True)
        return ranked[:n]

    selected_keys: list = []
    keys_by_type_out: dict[str, int] = {}

    for mid in present_types:
        chosen = _select_keys(meth_by_type[mid], type_targets.get(mid, 0))
        selected_keys.extend(chosen)
        keys_by_type_out[_METH_NAMES.get(mid, str(mid))] = len(chosen)

    unmeth_chosen = _select_keys(unmeth_keys, n_unmeth_budget)
    selected_keys.extend(unmeth_chosen)
    keys_by_type_out["none"] = len(unmeth_chosen)

    # ---- Per-key sample diversity selection ----
    output: dict = {}
    for key in selected_keys:
        arr = keyed[key]
        if samples_per_key > 0 and arr.ndim == 2 and len(arr) > samples_per_key:
            arr = _diversity_subsample(arr, samples_per_key)
        output[key] = arr

    keys_out = len(output)
    samples_out = sum(len(v) for v in output.values())

    # Re-attach metadata
    if meta is not None:
        if isinstance(meta, dict):
            meta["balanced_from"] = str(input_path)
            meta["balance_params"] = {
                "meth_fraction": meth_fraction,
                "max_keys": max_keys,
                "samples_per_key": samples_per_key,
            }
        output["__meta__"] = meta

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats = {
        "keys_in": keys_in,
        "keys_out": keys_out,
        "samples_in": samples_in,
        "samples_out": samples_out,
        "keys_by_type_in": {_METH_NAMES.get(m, str(m)): len(k) for m, k in by_meth.items()},
        "keys_by_type_out": keys_by_type_out,
    }

    log.info(
        "Balanced: %d -> %d keys,  %d -> %d samples", keys_in, keys_out, samples_in, samples_out
    )
    log.info("Output: %s", output_path)
    return stats


def main(argv=None) -> None:
    from kinsim.utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="python scripts/balance.py",
        description=(
            "Balance a merged training dictionary for fair methylation representation.\n\n"
            "Ensures methylated keys are not drowned out by the far more numerous\n"
            "unmethylated contexts, and that all mod types (m6A, m4C, m5C) each\n"
            "get their share of the training budget.\n\n"
            "Examples:\n"
            "  # 50/50 meth/unmeth split, diversity-subsampled to 200 samples/key:\n"
            "  python scripts/balance.py merged.pkl balanced.pkl \\\n"
            "      --meth-fraction 0.5 --samples-per-key 200\n\n"
            "  # Cap at 200k keys total, at least 25% methylated:\n"
            "  python scripts/balance.py merged.pkl balanced.pkl \\\n"
            "      --meth-fraction 0.25 --max-keys 200000"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Merged dictionary .pkl file")
    parser.add_argument("output", help="Output balanced dictionary .pkl file")
    parser.add_argument(
        "--meth-fraction",
        type=float,
        default=0.5,
        help="Target fraction of kept keys that are methylated (default: 0.5). "
        "Methylated budget is split evenly across all mod types present.",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=0,
        help="Total key budget (default: 0 = keep all keys, only rebalance types).",
    )
    parser.add_argument(
        "--samples-per-key",
        type=int,
        default=0,
        help="Subsample each key to this many samples using IPD-quantile diversity "
        "selection (default: 0 = keep all samples). Recommended: 200.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    stats = balance_pkl(
        args.input,
        args.output,
        meth_fraction=args.meth_fraction,
        max_keys=args.max_keys,
        samples_per_key=args.samples_per_key,
    )

    # Summary
    print(f"Keys:    {stats['keys_in']:>10,} -> {stats['keys_out']:>10,}")
    print(f"Samples: {stats['samples_in']:>10,} -> {stats['samples_out']:>10,}")
    print("Type breakdown (output):")
    for mod, n in sorted(stats["keys_by_type_out"].items()):
        pct = n / stats["keys_out"] * 100 if stats["keys_out"] else 0
        print(f"  {mod:6s}: {n:>8,} keys  ({pct:.1f}%)")
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()
