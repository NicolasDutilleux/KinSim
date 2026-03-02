"""KinSim CLI entry point.

Usage:
    kinsim [--version] <command> [<args>]
    python -m kinsim [--version] <command> [<args>]
"""

import difflib
import sys

__version__ = "0.2.0"

COMMANDS = ["prepare", "parse", "extract", "merge", "train", "generate", "analyze"]

USAGE = """\
usage: kinsim [--version] <command> [<args>]

KinSim — PacBio kinetic signal simulator for metagenomic binning.

Commands:
  prepare     Validate BAM/motif pairs into a reusable config file
  parse       Parse any motif source (PacBio CSV, REBASE, or inline string)
  extract     Extract raw IPD/PW samples from a BAM file  (→ .pkl shard)
  merge       Merge .pkl shards into a master training set
  train       Train a kinetic model  (--model dictionary | mlp | cgan)
  generate    Generate synthetic kinetic signals for PBSIM3 reads
              (--model dictionary | mlp | cgan)
  analyze     Report coverage and signal statistics for a dictionary .pkl

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
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    if args[0] in ("--version", "-V"):
        print(f"kinsim {__version__}")
        sys.exit(0)

    cmd, rest = args[0], args[1:]

    # ── prepare ──────────────────────────────────────────────────────────────
    if cmd == "prepare":
        from .prepare import main as run
        run(rest)

    # ── parse ─────────────────────────────────────────────────────────────────
    # Unified entry for CSV, REBASE, and inline motif strings.
    # Replaces the old 'kinsim motifs' and 'kinsim rebase parse/patterns'.
    elif cmd == "parse":
        from .motifs import main as run
        run(rest)

    # ── extract ───────────────────────────────────────────────────────────────
    # Extract raw (IPD, PW) samples from a real PacBio BAM file.
    # Replaces 'kinsim cgan extract'.
    elif cmd == "extract":
        from .common.extract import main as run
        run(["extract"] + rest)

    # ── merge ─────────────────────────────────────────────────────────────────
    # Merge .pkl shards produced by 'kinsim extract' into one master file.
    # Replaces 'kinsim cgan merge'.
    elif cmd == "merge":
        from .common.extract import main as run
        run(["merge"] + rest)

    # ── analyze ───────────────────────────────────────────────────────────────
    # Coverage statistics and signal reports for a dictionary .pkl.
    elif cmd == "analyze":
        from .dictionary.analyze import main as run
        run(rest)

    # ── train ─────────────────────────────────────────────────────────────────
    # Unified training command. --model selects the back-end.
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

    # ── generate ──────────────────────────────────────────────────────────────
    # Unified generation command. --model selects the back-end.
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

    # ── backward-compat: legacy sub-command style ─────────────────────────────
    # Old SLURM scripts and notebooks continue to work unchanged.

    elif cmd == "motifs":
        from .motifs import main as run
        run(rest)

    elif cmd == "rebase":
        from .rebase_parser import main as run
        run(rest)

    elif cmd == "dictionary":
        if not rest or rest[0] in ("-h", "--help"):
            print("Tip: use 'kinsim train --model dictionary' and "
                  "'kinsim generate --model dictionary'")
            sys.exit(0)
        subcmd, subrest = rest[0], rest[1:]
        if subcmd == "train":
            from .dictionary.train import main as run
            run(["train"] + subrest)
        elif subcmd == "merge":
            from .dictionary.train import main as run
            run(["merge"] + subrest)
        elif subcmd == "inject":
            from .dictionary.inject import main as run
            run(subrest)
        elif subcmd == "metagenome":
            from .dictionary.inject import metagenome_main as run
            run(subrest)
        elif subcmd == "analyze":
            from .dictionary.analyze import main as run
            run(subrest)
        else:
            print(f"Unknown dictionary command: {subcmd}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "cgan":
        if not rest:
            sys.exit(0)
        subcmd, subrest = rest[0], rest[1:]
        if subcmd == "extract":
            from .models.cgan.parse_train import main as run
            run(["extract"] + subrest)
        elif subcmd == "merge":
            from .models.cgan.parse_train import main as run
            run(["merge"] + subrest)
        elif subcmd == "train":
            from .models.cgan.train import main as run
            run(subrest)
        elif subcmd == "generate":
            from .models.cgan.generate import main as run
            run(subrest)
        else:
            print(f"Unknown cgan command: {subcmd}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "mlp":
        if not rest:
            sys.exit(0)
        subcmd, subrest = rest[0], rest[1:]
        if subcmd == "train":
            from .models.mlp.train import main as run
            run(subrest)
        elif subcmd == "generate":
            from .models.mlp.generate import main as run
            run(subrest)
        else:
            print(f"Unknown mlp command: {subcmd}", file=sys.stderr)
            sys.exit(1)

    # ── unknown ───────────────────────────────────────────────────────────────
    else:
        msg = f"Unknown command: '{cmd}'"
        hint = _suggest(cmd, COMMANDS)
        if hint:
            msg += f"\n\nDid you mean:  kinsim {hint[0]}"
        print(msg, file=sys.stderr)
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
