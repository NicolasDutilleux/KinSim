"""CLI dispatcher for the kinsim package.

Subcommands:

    python -m kinsim train     <shards_dir> <ckpt_dir> [--config <yaml>] [--resume]
    python -m kinsim evaluate  <ckpt_or_dir> <shards_dir> [--config <yaml>] [--test-strains ...]
    python -m kinsim generate  <input.bam> <ref.fa> <ckpt> <motifs.csv> <out.bam>
"""
from __future__ import annotations

import sys

from . import evaluate as _evaluate
from . import train as _train


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        _train.main(rest)
    elif cmd == "evaluate":
        _evaluate.main(rest)
    elif cmd == "generate":
        from . import generate as _generate
        _generate.main(rest)
    else:
        print(f"Unknown command: {cmd!r}\n")
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
