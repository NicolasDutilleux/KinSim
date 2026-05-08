"""kinsim_baseline CLI router.

Usage::

    python -m kinsim_baseline build SHARDS_DIR OUTPUT_NPZ [...]
    python -m kinsim_baseline generate INPUT_BAM TABLE_NPZ OUTPUT_BAM [...]
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "usage: python -m kinsim_baseline {build|calibrate|generate} [options]\n"
            "\n"
            "  build SHARDS_DIR OUTPUT_NPZ\n"
            "      Build per-kmer (IPD, PW) table from KinSim extract shards.\n"
            "      Filters BASELINE category only — clean unmodified kinetics.\n"
            "\n"
            "  calibrate BASELINE_NPZ MANIFEST_CSV OUTPUT_NPZ\n"
            "      Walk real BAMs; for each kmer, identify positions whose\n"
            "      observed IPD exceeds the kmer's baseline 99th percentile;\n"
            "      accumulate modified-sample bank + per-kmer IPD ratio.\n"
            "      Output extends the baseline .npz with modified data.\n"
            "\n"
            "  generate INPUT_BAM TABLE_NPZ OUTPUT_BAM\n"
            "      Sample fi/fp tags for every read in INPUT_BAM by drawing\n"
            "      from the kmer table. Output BAM matches kinsim generate's\n"
            "      contract (flag=4 unmapped, fi/fp B:C tags).\n"
        )
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "build":
        from .build_table import main as cmd_main
    elif cmd == "calibrate":
        from .calibrate import main as cmd_main
    elif cmd == "generate":
        from .generate import main as cmd_main
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print("expected: build, calibrate, generate", file=sys.stderr)
        sys.exit(1)

    cmd_main(rest)


if __name__ == "__main__":
    main()
