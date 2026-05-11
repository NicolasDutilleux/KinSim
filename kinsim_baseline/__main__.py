"""kinsim_baseline CLI router.

Usage::

    python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_DIR [--threshold 1.3]
        Build per-(meth_type, offset) IPD/PW histograms from the BAMs.

    python -m kinsim_baseline analyze OUTPUT_DIR [--no-plot]
        Read OUTPUT_DIR/baseline.json from a previous compute, fit a
        2-component GMM per (T, k), write baseline_gmm.tsv +
        baseline_gmm.json, and (by default) drop distribution plots into
        OUTPUT_DIR/plots/.

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
            "      into the per-(T, k) 256-bin histogram.\n"
            "      Outputs (in OUTPUT_DIR):\n"
            "        baseline_hist.tsv     long-form histogram\n"
            "        baseline_summary.tsv  per-(T, k) mean / p50 / p95 / p99 + ratio\n"
            "        baseline.json         full histograms\n"
            "        run_info.json         manifest + per-BAM read counts + timing\n"
            "\n"
            "  analyze OUTPUT_DIR [--no-plot] [--max-samples N]\n"
            "      Read OUTPUT_DIR/baseline.json (from a previous `compute`),\n"
            "      fit a 2-component Gaussian mixture per (T, k) IPD and PW\n"
            "      histogram, and write:\n"
            "        baseline_gmm.tsv      summary with GMM columns\n"
            "        baseline_gmm.json     full GMM parameters\n"
            "        plots/all_IPD.png     panel of all (T, k) IPD distributions\n"
            "        plots/all_PW.png      same for PW\n"
            "        plots/<T>_off<k>_IPD.png  per-bucket detail plots\n"
            "      Requires matplotlib + scipy (in the [plot] optional extra).\n"
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
