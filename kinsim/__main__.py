"""KinSim CLI entry point.

Usage:
    kinsim [--version] <command> [<args>]
    python -m kinsim [--version] <command> [<args>]
"""

import difflib
import logging
import sys

__version__ = "0.3.0"

COMMANDS = [
    "extract", "merge", "train", "generate", "evaluate", "analyze",
]

USAGE = """\
usage: kinsim [--version] <command> [<args>]

KinSim - PacBio kinetic signal simulator for metagenomic binning.

Core commands:
  extract     Extract raw IPD/PW samples from a BAM file  (-> .pkl shard)
  merge       Merge .pkl shards into a master training set
  train       Train a kinetic model  (--model dictionary | mlp | cgan)
  generate    Generate synthetic kinetic signals for PBSIM3 reads
              (--model dictionary | mlp | cgan)
  evaluate    Evaluate a trained model  (--model mlp)
  analyze     Report coverage and signal statistics for a dictionary .pkl

Data preparation:
  Use 'kinsim-prep' for motif parsing, REBASE fetching, manifest tools,
  and general dictionary filtering.

Use 'kinsim <command> -h' for detailed help on a specific command.
Use 'kinsim --version' to print the version number.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suggest(word, candidates, n=1, cutoff=0.6):
    """Return close matches for typo suggestions."""
    return difflib.get_close_matches(word, candidates, n=n, cutoff=cutoff)


def _pop_model(args):
    """Remove --model <value> from an arg list.

    Returns (model_str, remaining_args).
    model_str is None when the flag is absent.

    Supports both '--model mlp' and '--model=mlp' forms.
    """
    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            return args[i + 1], args[:i] + args[i + 2:]
        if arg.startswith("--model="):
            return arg[len("--model="):], args[:i] + args[i + 1:]
    return None, args


def _require_model(rest, command):
    """Parse --model from rest; exit with a clear message if missing."""
    model, subrest = _pop_model(rest)
    if not model:
        print(
            f"ERROR: 'kinsim {command}' requires --model <dictionary|mlp|cgan>.\n"
            f"  Example: kinsim {command} ... --model mlp",
            file=sys.stderr,
        )
        sys.exit(1)
    if model not in ("dictionary", "mlp", "cgan"):
        print(
            f"ERROR: unknown model '{model}'. "
            "Valid choices: dictionary, mlp, cgan",
            file=sys.stderr,
        )
        sys.exit(1)
    return model, subrest


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main(argv=None):
    from .config import setup_logging
    args = argv if argv is not None else sys.argv[1:]

    # Set up logging early so all submodules emit timestamped output to SLURM logs.
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
        from .common.extract import main as run
        run(["extract"] + rest)

    # -- merge --
    elif cmd == "merge":
        from .common.extract import main as run
        run(["merge"] + rest)

    # -- analyze --
    elif cmd == "analyze":
        from .dictionary.analyze import main as run
        run(rest)

    # -- train --
    elif cmd == "train":
        model, subrest = _require_model(rest, "train")

        if model == "dictionary":
            from .dictionary.train import main as run
            run(["train"] + subrest)

        elif model == "mlp":
            from .models.mlp.train import main as run
            run(subrest)

        elif model == "cgan":
            from .models.cgan.train import main as run
            run(subrest)

    # -- generate --
    elif cmd == "generate":
        model, subrest = _require_model(rest, "generate")

        if model == "dictionary":
            from .dictionary.inject import main as run
            run(subrest)

        elif model == "mlp":
            from .models.mlp.generate import main as run
            run(subrest)

        elif model == "cgan":
            from .models.cgan.generate import main as run
            run(subrest)

    # -- evaluate --
    elif cmd == "evaluate":
        model, subrest = _require_model(rest, "evaluate")

        if model == "mlp":
            from .models.mlp.evaluate import main as run
            run(subrest)
        else:
            print(
                f"ERROR: 'kinsim evaluate' is currently only supported for "
                f"--model mlp (got '{model}').",
                file=sys.stderr,
            )
            sys.exit(1)

    # -- unknown --
    else:
        msg = f"Unknown command: '{cmd}'"
        hint = _suggest(cmd, COMMANDS)
        if hint:
            msg += f"\n\nDid you mean:  kinsim {hint[0]}"
        # Hint for users who try prep commands on kinsim
        prep_cmds = {"prep", "rebase", "manifest", "filter", "prepare", "parse", "motifs"}
        if cmd in prep_cmds:
            msg += f"\n\nData prep commands have moved to 'kinsim-prep'.\n"
            msg += f"  Try:  kinsim-prep {cmd} ..."
        print(msg, file=sys.stderr)
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
