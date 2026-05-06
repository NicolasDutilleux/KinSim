"""KinSim CLI entry point.

Usage:
    kinsim [--version] <command> [<args>]
    python -m kinsim [--version] <command> [<args>]
"""

import difflib
import sys

__version__ = "0.4.0"

COMMANDS = [
    "extract",
    "refine",
    "train",
    "generate",
    "evaluate",
    "verify-generate",
    "analyze",
]

USAGE = """\
usage: kinsim [--version] <command> [<args>]

KinSim — PacBio HiFi kinetic signal simulator.

Pipeline (one verb per stage; each consumes the previous stage's output):

  extract          Aligned BAM + ref + motifs  →  shard.pkl
                   Use ``kinsim extract --refine`` to chain refine in one step.
  refine           Per-(meth, offset) GMM filter on slowed rows.
  train            Train ConvPredictor on shards.
  generate         Trained model + PBSIM3 reads  →  synthetic BAM with kinetics.
  evaluate         Per-(kmer, meth) calibration report on a trained model.
  verify-generate  Compare two shards (real vs simulated) per (kmer, meth).
  analyze          Diagnostic dashboard for any shard or refined directory.

Data preparation:
  Use 'kinsim-prep' for motif parsing, REBASE fetching, manifest tools,
  and .pkl filtering / balancing.

Auxiliary scripts (not pipeline stages, run directly with python):
  scripts/compare.py             Cross-dataset kinetic comparison.
  scripts/inspect_null_model.py  Inspect an ipdSummary .npz.gz null model.
  scripts/sample.py              Subsample a shard pkl.
  scripts/strip_kinetics.py      Strip fi/fp/ri/rp from a BAM.

Use 'kinsim <command> -h' for detailed help on a specific command.
Use 'kinsim --version' to print the version number.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suggest(word, candidates, n=1, cutoff=0.6):
    """Return close matches for typo suggestions."""
    return difflib.get_close_matches(word, candidates, n=n, cutoff=cutoff)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def main(argv=None):
    from .utils.config import setup_logging

    args = argv if argv is not None else sys.argv[1:]

    setup_logging(verbose=False)

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    if args[0] in ("--version", "-V"):
        print(f"kinsim {__version__}")
        sys.exit(0)

    cmd, rest = args[0], args[1:]

    # One verb per pipeline stage. Each command imports its own main()
    # lazily so invoking one verb doesn't pull in unrelated deps.
    DISPATCH = {
        "extract":         "kinsim.extract",
        "refine":          "kinsim.refine",
        "train":           "kinsim.train",
        "generate":        "kinsim.generate",
        "evaluate":        "kinsim.evaluate",
        "verify-generate": "kinsim.verify_generate",
        "analyze":         "kinsim.analyze",
    }

    if cmd in DISPATCH:
        import importlib

        mod = importlib.import_module(DISPATCH[cmd])
        # extract's argparser expects the verb as argv[0]; the others
        # receive only the trailing args.
        mod.main([cmd, *rest] if cmd == "extract" else rest)
        return

    # -- unknown --
    msg = f"Unknown command: '{cmd}'"
    hint = _suggest(cmd, COMMANDS)
    if hint:
        msg += f"\n\nDid you mean:  kinsim {hint[0]}"
    # Data prep commands live in a separate CLI.
    prep_cmds = {"prep", "rebase", "manifest", "filter", "prepare", "parse", "motifs"}
    if cmd in prep_cmds:
        msg += f"\n\nData prep commands live in 'kinsim-prep'.\n  Try:  kinsim-prep {cmd} ..."
    # Auxiliary scripts live in scripts/, not as kinsim subcommands.
    legacy_tool_cmds = {"compare", "inspect-model", "sample", "strip-kinetics", "merge"}
    if cmd in legacy_tool_cmds:
        if cmd == "merge":
            msg += (
                "\n\n'kinsim merge' has been removed. Shards are the canonical training "
                "format now — refine and train both consume the shards directory directly.\n"
                "  kinsim refine shards/  refined/\n  kinsim train  refined/ checkpoints/"
            )
        else:
            msg += (
                f"\n\n'kinsim {cmd}' has been moved out of the main CLI. Run the standalone "
                f"script directly:\n  python scripts/{cmd.replace('-', '_')}.py ..."
            )
    print(msg, file=sys.stderr)
    print(USAGE)
    sys.exit(1)


if __name__ == "__main__":
    main()
