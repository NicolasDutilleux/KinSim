"""kinsim_baseline CLI router.

Usage::

    python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_TSV [--threshold 1.3]

See ``compute.py`` for the algorithm.
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "usage: python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_DIR [options]\n"
            "\n"
            "  compute MANIFEST_CSV OUTPUT_DIR [--threshold 1.3]\n"
            "      Single-pass walk over manifest BAMs.\n"
            "      Reads modified_base + signal_offsets per meth type from\n"
            "      kinsim_config.yaml. For each read, for each meth type T,\n"
            "      for each position p where read[p] == modified_base[T],\n"
            "      for each k in signal_offsets[T]: record ipd[p+k] / pw[p+k]\n"
            "      into the per-(T, k) 256-bin histogram.\n"
            "      Outputs (in OUTPUT_DIR):\n"
            "        baseline_hist.tsv     long-form histogram\n"
            "        baseline_summary.tsv  per-(T, k) mean / p50 / p95 / p99 + ratio\n"
            "        baseline.json         full histograms\n"
            "        run_info.json         manifest + per-BAM read counts + timing\n"
        )
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "compute":
        from .compute import main as cmd_main
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print("expected: compute", file=sys.stderr)
        sys.exit(1)

    cmd_main(rest)


if __name__ == "__main__":
    main()
