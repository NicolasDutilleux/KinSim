"""kinsim_baseline CLI router.

Usage::

    python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_DIR [--threshold 1.3]
        Build per-(meth_type, offset) IPD/PW 1D histograms + IPD×PW 2D joint
        from the BAMs.

    python -m kinsim_baseline analyze OUTPUT_DIR
        Plot per-(meth_type, offset) IPD distributions from a previous
        compute run. Writes a single interactive HTML (linear + log y).

See ``compute.py`` and ``analyze.py`` for the algorithms.
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "usage: python -m kinsim_baseline <command> [options]\n"
            "\n"
            "  compute MANIFEST_CSV OUTPUT_DIR [--threshold 1.3]\n"
            "      Single-pass walk over manifest BAMs.\n"
            "      Reads modified_base + signal_offsets per meth type from\n"
            "      kinsim_config.yaml. For each read, for each meth type T,\n"
            "      for each position p where read[p] == modified_base[T],\n"
            "      for each k in signal_offsets[T]: record ipd[p+k] / pw[p+k]\n"
            "      into the per-(T, k) 256-bin histogram, plus IPD×PW 2D joint.\n"
            "      Outputs (in OUTPUT_DIR):\n"
            "        baseline_hist.tsv     long-form 1D histogram\n"
            "        baseline_summary.tsv  per-(T, k) mean / p50 / p95 / p99\n"
            "        baseline.json         full 1D + 2D histograms\n"
            "        run_info.json         manifest + per-BAM read counts + timing\n"
            "\n"
            "  analyze OUTPUT_DIR\n"
            "      Read OUTPUT_DIR/baseline.json and write a single interactive\n"
            "      HTML showing all (T, k) IPD distributions overlaid, in two\n"
            "      panels (linear + log y) so the bulk and the right-tail are\n"
            "      both readable.\n"
            "      Output:\n"
            "        ipd_distributions.html\n"
            "      Requires plotly (in the [plot] optional extra).\n"
        )
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "compute":
        from .compute import main as cmd_main
    elif cmd == "analyze":
        from .analyze import main as cmd_main
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print("expected: compute | analyze", file=sys.stderr)
        sys.exit(1)

    cmd_main(rest)


if __name__ == "__main__":
    main()
