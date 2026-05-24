"""``kinsim_nn`` CLI dispatcher.

Subcommands:
    extract     Build shards from BAM + GFF + labelers.
    train       cGAN training loop (WGAN-GP).
    generate    motifs.csv + ref + input BAM → unmapped BAM with fi/fp/ri/rp.
    evaluate    Distribution-level metrics on held-out shards.
    analyze     HTML dashboard of per-category / per-meth / per-offset
                distributions across all shards.
"""
from __future__ import annotations

import sys
from difflib import get_close_matches


COMMANDS = {
    "extract": "kinsim_NN.extract",
    "train": "kinsim_NN.train",
    "generate": "kinsim_NN.generate",
    "evaluate": "kinsim_NN.evaluate",
    "analyze": "kinsim_NN.analyze",
}


def _usage() -> None:
    print("Usage: kinsim_nn <command> [args]\n\nCommands:")
    for cmd, mod in COMMANDS.items():
        print(f"  {cmd:<12} ({mod})")
    print("\nRun 'kinsim_nn <command> --help' for details.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _usage()
        return
    cmd = argv[0]
    rest = argv[1:]
    if cmd not in COMMANDS:
        suggestions = get_close_matches(cmd, COMMANDS.keys(), n=3, cutoff=0.5)
        msg = f"Unknown command: {cmd!r}"
        if suggestions:
            msg += f"  (did you mean: {', '.join(suggestions)}?)"
        print(msg, file=sys.stderr)
        _usage()
        sys.exit(2)
    # Import lazily — keeps startup snappy for --help
    import importlib
    mod = importlib.import_module(COMMANDS[cmd])
    mod.main(rest)


if __name__ == "__main__":
    main()
