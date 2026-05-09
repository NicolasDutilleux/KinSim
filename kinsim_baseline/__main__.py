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
            "usage: python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_TSV [options]\n"
            "\n"
            "  compute MANIFEST_CSV OUTPUT_TSV [--threshold 1.3] [--output-json out.json]\n"
            "      Two-pass walk over manifest BAMs.\n"
            "      Pass 1: per-(meth_type, offset) baseline mean IPD/PW.\n"
            "              For each base p where read[p] == modified_base[T],\n"
            "              accumulate ipd[p+k]/pw[p+k] for k in signal_offsets[T].\n"
            "      Pass 2: per-(meth_type, offset) modified mean IPD/PW from\n"
            "              positions where observed IPD > threshold × baseline.\n"
            "      Output: per-(T, k) ipd_ratio = modified_mean / baseline_mean.\n"
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
