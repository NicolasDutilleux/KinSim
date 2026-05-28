"""Inspect a PacBio ipdSummary null model (.npz.gz / .h5).

The null model contains per-context expected IPD values for unmodified DNA.
ipdSummary compares observed IPD against these expectations to detect
methylation. Understanding this model is key to understanding the signal
KinSim needs to reproduce.

CLI usage:
    python scripts/inspect_null_model.py /path/to/SP3-C3.npz.gz
    python scripts/inspect_null_model.py /path/to/SP3-C3.npz.gz --output report.txt
    python scripts/inspect_null_model.py /path/to/SP3-C3.npz.gz --dump-csv null_model.csv

What the null model contains (typically):
    - Per-context (kmer) expected mean IPD in native space
    - The context window size (often 12-mer or similar)
    - Strand information (separate models for fwd/rev)
    - Possibly variance or weight information

This tool:
    1. Lists all arrays/keys in the .npz file
    2. Reports shapes, dtypes, value ranges
    3. Analyzes the distribution of expected IPD values
    4. Optionally dumps the full model as CSV for external analysis
"""

from __future__ import annotations

import argparse
import gzip
import logging
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _load_npz(path: str) -> dict:
    """Load .npz or .npz.gz file."""
    p = Path(path)
    if p.suffix == ".gz" or str(p).endswith(".npz.gz"):
        log.info("Loading gzipped npz: %s", path)
        with gzip.open(path, "rb") as gz:
            buf = BytesIO(gz.read())
        npz = np.load(buf, allow_pickle=True)
    else:
        log.info("Loading npz: %s", path)
        npz = np.load(path, allow_pickle=True)
    return dict(npz)


def inspect_model(path: str, dump_csv: str | None = None) -> str:
    """Inspect the null model and return a text report."""
    data = _load_npz(path)

    lines = []
    w = lines.append

    w("=" * 72)
    w(f"  ipdSummary Null Model: {Path(path).name}")
    w("=" * 72)
    w("")

    # --- Keys overview ---
    w("=== Arrays in model ===")
    w("")
    for key in sorted(data.keys()):
        arr = data[key]
        if isinstance(arr, np.ndarray):
            w(f"  {key:<30} shape={arr.shape!s:<20} dtype={arr.dtype}")
        else:
            w(f"  {key:<30} type={type(arr).__name__}  value={arr}")
    w("")

    # --- Detailed stats per array ---
    w("=== Array Statistics ===")
    w("")
    for key in sorted(data.keys()):
        arr = data[key]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.dtype.kind in ("f", "i", "u"):  # numeric
            finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr.ravel()
            if len(finite) == 0:
                w(f"  {key}: all NaN/Inf")
                continue
            w(f"  {key}:")
            w(f"    Min:    {np.min(finite):.6f}")
            w(f"    Max:    {np.max(finite):.6f}")
            w(f"    Mean:   {np.mean(finite):.6f}")
            w(f"    Median: {np.median(finite):.6f}")
            w(f"    Std:    {np.std(finite):.6f}")
            w(f"    Zeros:  {np.sum(finite == 0):,} / {len(finite):,}")
            # Percentiles
            for p in [1, 5, 25, 75, 95, 99]:
                w(f"    P{p:02d}:    {np.percentile(finite, p):.6f}")
            w("")
        elif arr.dtype.kind in ("U", "S", "O"):  # string or object
            w(f"  {key}: {arr.shape} — first 5 entries: {arr.flat[:5].tolist()}")
            w("")

    # --- Interpret structure ---
    w("=== Model Interpretation ===")
    w("")

    # Common PacBio null model structure: a single large 1D or 2D array
    # keyed by context encoding
    for key in sorted(data.keys()):
        arr = data[key]
        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim == 1 and len(arr) > 1000:
            # Likely the main prediction array
            n = len(arr)
            # Check if n is a power of 4 (4^k = context size)
            k = 1
            while 4**k < n:
                k += 1
            if 4**k == n:
                w(f"  {key}: {n:,} entries = 4^{k} → likely {k}-mer context")
            else:
                w(f"  {key}: {n:,} entries (not a power of 4)")

            # Distribution analysis
            if arr.dtype.kind == "f":
                nonzero = arr[arr > 0]
                if len(nonzero) > 0:
                    w(
                        f"    Non-zero entries: {len(nonzero):,} / {n:,} ({100 * len(nonzero) / n:.1f}%)"
                    )
                    w(f"    Non-zero mean:    {np.mean(nonzero):.4f}")
                    w(f"    Non-zero std:     {np.std(nonzero):.4f}")
                    w(f"    This represents expected IPD values for each {k}-mer context")
                    w("    in unmodified DNA. ipdSummary divides observed IPD by these")
                    w("    values to compute the IPD ratio.")
            w("")

        elif arr.ndim == 2:
            w(f"  {key}: {arr.shape[0]:,} rows × {arr.shape[1]} cols")
            if arr.shape[1] == 2:
                w("    Possibly (mean, variance) per context")
            elif arr.shape[1] == 4:
                w("    Possibly per-base or per-strand breakdown")
            w("")

    # --- Dump CSV ---
    if dump_csv:
        main_key = None
        for key in sorted(data.keys()):
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) > 1000:
                main_key = key
                break

        if main_key:
            arr = data[main_key]
            with open(dump_csv, "w") as f:
                f.write("index,value\n")
                for i, v in enumerate(arr):
                    f.write(f"{i},{v}\n")
            w(f"  CSV dump: {dump_csv} ({len(arr):,} rows from '{main_key}')")
        else:
            w("  No suitable array found for CSV dump")

    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/inspect_null_model.py",
        description="Inspect a PacBio ipdSummary null model file.",
    )
    parser.add_argument("model_path", help="Path to .npz.gz or .npz null model file")
    parser.add_argument("--output", "-o", help="Write report to file")
    parser.add_argument("--dump-csv", help="Dump main array to CSV for analysis")

    args = parser.parse_args(argv)

    if not Path(args.model_path).exists():
        log.error("File not found: %s", args.model_path)
        sys.exit(1)

    report = inspect_model(args.model_path, dump_csv=args.dump_csv)
    print(report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        log.info("Report saved: %s", args.output)


if __name__ == "__main__":
    main()
