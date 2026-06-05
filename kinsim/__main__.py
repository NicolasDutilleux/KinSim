"""CLI dispatcher for the kinsim package.

Usage:

    python -m kinsim train <shards_dir> <ckpt_dir> [--config <yaml>] [--resume]

For now only the ``train`` subcommand is exposed. Generation and
evaluation are unchanged from kinsim_NN and can be invoked through that
package against a kinsim checkpoint (the model_config.json schema is
compatible by construction).
"""
from __future__ import annotations

import sys

from . import train as _train


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        _train.main(rest)
    else:
        print(f"Unknown command: {cmd!r}\n")
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
