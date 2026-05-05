"""CLI for manifest inspection and validation.

Provides three subcommands that operate on a manifest CSV:

    kinsim-prep manifest count    manifest.csv  -- print the number of data rows
    kinsim-prep manifest validate manifest.csv  -- check for errors (files, duplicates)
    kinsim-prep manifest list     manifest.csv  -- tabular display of all entries

The ``count`` subcommand is designed to be used directly in shell scripts, for
example when setting the SLURM array size::

    N=$(kinsim-prep manifest count manifest.csv)
    sbatch --array=1-$N kinsim_extract.slurm manifest.csv shards/ master.pkl

Using ``kinsim-prep manifest count`` instead of ``grep -c .`` or ``wc -l`` is
safer because it reuses the same Python logic as ``load_manifest()`` -- it
correctly skips comment rows (``#``), blank rows, and the header, matching
exactly the row indices that ``kinsim extract --task N`` will use.
"""

import sys


def main(argv=None) -> None:
    import argparse

    from kinsim.utils.config import load_manifest, setup_logging, validate_manifest

    parser = argparse.ArgumentParser(
        prog="kinsim-prep manifest",
        description=(
            "Inspect and validate a KinSim manifest CSV.\n\n"
            "Manifest format:\n"
            "  sample_id,bam_path,motifs\n"
            '  strain1,/data/bams/s1.bam,"m6A,GATC,1"\n'
            "  strain2,/data/bams/s2.bam,/data/motifs/s2.csv\n\n"
            "Comment rows (# ...) and blank rows are skipped, matching the\n"
            "exact row indices used by 'kinsim extract --task N'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # -- count --
    p_count = sub.add_parser(
        "count",
        help="Print the number of active data rows (safe for SLURM --array=1-N)",
        description=(
            "Count data rows in the manifest, skipping comment (#) and blank rows.\n\n"
            "Usage in SLURM scripts:\n"
            "  N=$(kinsim-prep manifest count manifest.csv)\n"
            "  sbatch --array=1-$N kinsim_extract.slurm manifest.csv shards/ master.pkl"
        ),
    )
    p_count.add_argument("manifest", help="Path to the manifest CSV file")

    # -- validate --
    p_validate = sub.add_parser(
        "validate",
        help="Validate the manifest for errors (duplicate IDs, missing files)",
        description=(
            "Check the manifest for:\n"
            "  - Duplicate sample_id values\n"
            "  - BAM files that do not exist on disk\n"
            "  - Motif files that do not exist (when the motifs field is a path)\n\n"
            "Exits 0 if valid, 1 if errors are found."
        ),
    )
    p_validate.add_argument("manifest", help="Path to the manifest CSV file")
    p_validate.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip file-existence checks (validate structure only; "
        "useful when BAMs are on a remote cluster)",
    )

    # -- list --
    p_list = sub.add_parser(
        "list",
        help="Print a tabular summary of all manifest entries",
    )
    p_list.add_argument("manifest", help="Path to the manifest CSV file")
    p_list.add_argument(
        "--no-truncate",
        action="store_true",
        help="Print full paths (default: truncate long paths to 50 characters)",
    )

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # -- count --
    if args.subcommand == "count":
        try:
            entries = load_manifest(args.manifest)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(len(entries))

    # -- validate --
    elif args.subcommand == "validate":
        try:
            entries = load_manifest(args.manifest)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        errors = validate_manifest(entries, check_files=not args.no_check_files)

        if not errors:
            n = len(entries)
            check_label = " (structure only)" if args.no_check_files else ""
            print(f"OK: {n} entr{'y' if n == 1 else 'ies'} valid{check_label}")
        else:
            print(f"ERRORS ({len(errors)} found in {args.manifest}):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)

    # -- list --
    elif args.subcommand == "list":
        try:
            entries = load_manifest(args.manifest)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        max_path = 50

        def _fmt(s: str, width: int) -> str:
            if not args.no_truncate and len(s) > width:
                return "..." + s[-(width - 3) :]
            return s

        # Header
        id_w = max(len(e.sample_id) for e in entries)
        bam_w = min(max_path, max(len(e.bam_path) for e in entries))
        mot_w = min(max_path, max(len(e.motifs) for e in entries))
        id_w = max(id_w, 9)  # at least "sample_id"
        bam_w = max(bam_w, 8)  # at least "bam_path"
        mot_w = max(mot_w, 6)  # at least "motifs"

        header = f"{'#':>4}  {'sample_id':<{id_w}}  {'bam_path':<{bam_w}}  {'motifs':<{mot_w}}"
        print(header)
        print("-" * len(header))

        for idx, entry in enumerate(entries, start=1):
            row = (
                f"{idx:>4}  "
                f"{entry.sample_id:<{id_w}}  "
                f"{_fmt(entry.bam_path, bam_w):<{bam_w}}  "
                f"{_fmt(entry.motifs, mot_w):<{mot_w}}"
            )
            print(row)

        print(f"\n{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} total.")


if __name__ == "__main__":
    main()
