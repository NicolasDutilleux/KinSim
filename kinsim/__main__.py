"""KinSim CLI entry point.

Usage:
    kinsim <command> [<args>]
    python -m kinsim <command> [<args>]
"""

import difflib
import sys

COMMANDS = ["prepare", "motifs", "rebase", "dictionary", "cgan"]
DICT_COMMANDS = ["train", "merge", "inject", "analyze"]
CGAN_COMMANDS = ["extract", "merge", "train", "generate"]

REBASE_COMMANDS = ["parse", "patterns"]

REBASE_USAGE = """\
usage: kinsim rebase <command> [<args>]

REBASE file parsing and fuzznuc pattern file generation.

Commands:
  parse       Parse a REBASE file and print the KinSim motif string
  patterns    Convert a motif source into a fuzznuc @pattern file

Use 'kinsim rebase <command> -h' for help on a specific command.
"""

USAGE = """\
usage: kinsim <command> [<args>]

KinSim — PacBio kinetic signal simulator for metagenomic binning.

Shared commands:
  prepare                Parse BAM + motifs.csv pairs into pipeline config
  motifs                 Parse a motif source (CSV, REBASE, or string)
  rebase                 Parse REBASE files / generate fuzznuc pattern files

Dictionary mode:
  dictionary train       Build a kinetic dictionary shard from a BAM file
  dictionary merge       Merge .pkl shards into a master dictionary
  dictionary inject      Inject IPD/PW signals into PBSIM3 reads
  dictionary analyze     Analyze dictionary coverage statistics

cGAN mode:
  cgan extract           Extract raw IPD/PW samples from a BAM file
  cgan merge             Merge cGAN shards into a master training set
  cgan train             Train the conditional GAN model (WGAN-GP)
  cgan generate          Generate kinetic signals for PBSIM3 reads

Use 'kinsim <command> -h' for help on a specific command.
"""

DICT_USAGE = """\
usage: kinsim dictionary <command> [<args>]

Dictionary mode — statistical 11-mer kinetic lookup tables.

Commands:
  train       Build a kinetic dictionary shard from a BAM file
  merge       Merge .pkl shards into a master dictionary
  inject      Inject IPD/PW signals into PBSIM3 reads
  analyze     Analyze dictionary coverage statistics

Use 'kinsim dictionary <command> -h' for help on a specific command.
"""


CGAN_USAGE = """\
usage: kinsim cgan <command> [<args>]

cGAN mode — conditional GAN-based kinetic signal generation.

Commands:
  extract     Extract raw (IPD, PW) samples from a BAM file
  merge       Merge *_cgan.pkl shards into a master training set
  train       Train the conditional GAN model (WGAN-GP)
  generate    Generate kinetic signals for PBSIM3 reads

Use 'kinsim cgan <command> -h' for help on a specific command.
"""


def _suggest(word, candidates, n=1, cutoff=0.6):
    """Return close matches for typo suggestions."""
    return difflib.get_close_matches(word, candidates, n=n, cutoff=cutoff)


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        sys.exit(0)

    cmd, rest = args[0], args[1:]

    if cmd == "prepare":
        from .prepare import main as run
        run(rest)

    elif cmd == "motifs":
        from .motifs import main as run
        run(rest)

    elif cmd == "rebase":
        if not rest or rest[0] in ("-h", "--help"):
            print(REBASE_USAGE)
            sys.exit(0)

        subcmd, subrest = rest[0], rest[1:]

        if subcmd in ("parse", "patterns"):
            from .rebase_parser import main as run
            run(rest)

        else:
            msg = f"Unknown rebase command: {subcmd}"
            hint = _suggest(subcmd, REBASE_COMMANDS)
            if hint:
                msg += f"\n\nDid you mean: kinsim rebase {hint[0]}?"
            print(msg)
            print(REBASE_USAGE)
            sys.exit(1)

    elif cmd == "dictionary":
        if not rest or rest[0] in ("-h", "--help"):
            print(DICT_USAGE)
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

        elif subcmd == "analyze":
            from .dictionary.analyze import main as run
            run(subrest)

        else:
            msg = f"Unknown dictionary command: {subcmd}"
            hint = _suggest(subcmd, DICT_COMMANDS)
            if hint:
                msg += f"\n\nDid you mean: kinsim dictionary {hint[0]}?"
            print(msg)
            print(DICT_USAGE)
            sys.exit(1)

    elif cmd == "cgan":
        if not rest or rest[0] in ("-h", "--help"):
            print(CGAN_USAGE)
            sys.exit(0)

        subcmd, subrest = rest[0], rest[1:]

        if subcmd == "extract":
            from .cgan.parse_train import main as run
            run(["extract"] + subrest)

        elif subcmd == "merge":
            from .cgan.parse_train import main as run
            run(["merge"] + subrest)

        elif subcmd == "train":
            from .cgan.train import main as run
            run(subrest)

        elif subcmd == "generate":
            from .cgan.generate import main as run
            run(subrest)

        else:
            msg = f"Unknown cgan command: {subcmd}"
            hint = _suggest(subcmd, CGAN_COMMANDS)
            if hint:
                msg += f"\n\nDid you mean: kinsim cgan {hint[0]}?"
            print(msg)
            print(CGAN_USAGE)
            sys.exit(1)

    else:
        msg = f"Unknown command: {cmd}"
        hint = _suggest(cmd, COMMANDS)
        if hint:
            msg += f"\n\nDid you mean: kinsim {hint[0]}?"
        print(msg)
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
