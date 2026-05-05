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
    "merge",
    "refine",
    "train",
    "generate",
    "evaluate",
    "verify-generate",
    "analyze",
    "compare",
    "inspect-model",
    "sample",
    "strip-kinetics",
]

USAGE = """\
usage: kinsim [--version] <command> [<args>]

KinSim — PacBio HiFi kinetic signal simulator.

Commands:
  extract          Extract per-kmer IPD/PW samples from a BAM file (-> .pkl shard)
  merge            Merge per-sample shards into a master training .pkl
  refine           Drop motif-FP slowed samples below the per-kmer baseline p95
  train            Train the ConvPredictor / MLPPredictor model
  generate         Generate synthetic kinetic signals for PBSIM3 reads
  evaluate         Calibration report + per-kmer plots from a trained model
  verify-generate  Compare per-kmer kinetic distributions: reference vs generated
  analyze          Analyse a master / master_clean .pkl (txt + Plotly HTML)
  compare          Compare per-kmer kinetics across datasets (Vega vs Revio, etc.)
  inspect-model    Inspect an ipdSummary null model (.npz.gz)
  sample           Subsample a .pkl for train/test splits
  strip-kinetics   Copy a BAM and remove fi/fp/ri/rp tags

Data preparation:
  Use 'kinsim-prep' for motif parsing, REBASE fetching, manifest tools,
  and .pkl filtering / balancing.

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

    # -- extract --
    if cmd == "extract":
        from .extract import main as run

        run(["extract", *rest])

    # -- merge --
    elif cmd == "merge":
        from .extract import main as run

        run(["merge", *rest])

    # -- refine --
    elif cmd == "refine":
        from .refine import main as run

        run(rest)

    # -- verify-generate --
    elif cmd == "verify-generate":
        from .verify_generate import main as run

        run(rest)

    # -- train --
    elif cmd == "train":
        from .train import main as run

        run(rest)

    # -- generate --
    elif cmd == "generate":
        from .generate import main as run

        run(rest)

    # -- evaluate --
    elif cmd == "evaluate":
        from .evaluate import main as run

        run(rest)

    # -- analyze --
    elif cmd == "analyze":
        from .analyze import main as run

        run(rest)

    # -- compare --
    elif cmd == "compare":
        from .compare import main as run

        run(rest)

    # -- inspect-model --
    elif cmd == "inspect-model":
        from .inspect_null_model import main as run

        run(rest)

    # -- sample --
    elif cmd == "sample":
        from .sample import main as run

        run(rest)

    # -- strip-kinetics --
    elif cmd == "strip-kinetics":
        from .strip_kinetics import main as run

        run(rest)

    # -- unknown --
    else:
        msg = f"Unknown command: '{cmd}'"
        hint = _suggest(cmd, COMMANDS)
        if hint:
            msg += f"\n\nDid you mean:  kinsim {hint[0]}"
        # Hint for users who try prep commands on kinsim
        prep_cmds = {"prep", "rebase", "manifest", "filter", "prepare", "parse", "motifs"}
        if cmd in prep_cmds:
            msg += "\n\nData prep commands live in 'kinsim-prep'.\n"
            msg += f"  Try:  kinsim-prep {cmd} ..."
        # Hint for users who try old model-based syntax
        old_cmds = {"dictionary", "cgan", "mlp"}
        if cmd in old_cmds:
            msg += "\n\nThe --model flag has been removed. KinSim is now MLP-only.\n"
            msg += "  Use:  kinsim train / kinsim generate / kinsim evaluate\n"
            msg += "  Dictionary and cGAN code has been moved to archive/."
        print(msg, file=sys.stderr)
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
