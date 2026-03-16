"""KinSim data preparation CLI entry point.

Usage:
    kinsim-prep [--version] <subcommand> [<args>]
    python -m prep [--version] <subcommand> [<args>]
"""

import difflib
import sys

__version__ = "0.3.0"

COMMANDS = ["parse", "rebase", "merge-motifs", "manifest", "prepare", "filter", "balance"]

USAGE = """\
usage: kinsim-prep [--version] <subcommand> [<args>]

Data preparation tools for KinSim.

Subcommands:
  parse          Parse any motif source (PacBio CSV, REBASE, or inline string)
                 into a KinSim motif string
  rebase         Fetch motifs from REBASE by organism number, parse REBASE files,
                 or generate fuzznuc pattern files
  merge-motifs   Merge, filter, and deduplicate motifs from multiple sources
                 (calling CSV + REBASE CSV) into a standard PacBio motifs.csv
  manifest       Inspect and validate manifest CSVs (count / validate / list)
  prepare        Validate BAM/motif pairs (legacy alternating-line format)
  filter         Filter a General Dictionary .pkl into a Training Dictionary
  balance        Balance a merged dictionary: even out mod types (m6A/m4C/m5C)
                 and maximise IPD diversity per key

Use 'kinsim-prep <subcommand> -h' for detailed help.
Use 'kinsim-prep --version' to print the version number.
"""


def _suggest(word, candidates, n=1, cutoff=0.6):
    """Return close matches for typo suggestions."""
    return difflib.get_close_matches(word, candidates, n=n, cutoff=cutoff)


def main(argv=None):
    from kinsim.utils.config import setup_logging
    args = argv if argv is not None else sys.argv[1:]

    setup_logging(verbose=False)

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    if args[0] in ("--version", "-V"):
        print(f"kinsim-prep {__version__}")
        sys.exit(0)

    cmd, rest = args[0], args[1:]

    if cmd == "parse":
        from kinsim.utils.motifs import main as run
        run(rest)

    elif cmd == "rebase":
        from .rebase import main as run
        run(rest)

    elif cmd == "merge-motifs":
        from .motif_merge import main as run
        run(rest)

    elif cmd == "manifest":
        from .manifest import main as run
        run(rest)

    elif cmd == "prepare":
        from .prepare import main as run
        run(rest)

    elif cmd == "filter":
        from .filter import main as run
        run(rest)

    elif cmd == "balance":
        from .balance import main as run
        run(rest)

    else:
        msg = f"Unknown subcommand: '{cmd}'"
        hint = _suggest(cmd, COMMANDS)
        if hint:
            msg += f"\n\nDid you mean:  kinsim-prep {hint[0]}"
        print(msg, file=sys.stderr)
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
