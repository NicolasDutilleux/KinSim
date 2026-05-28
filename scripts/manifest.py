"""CLI for manifest inspection and validation.

Provides three subcommands that operate on a manifest CSV:

    python scripts/manifest.py count    manifest.csv
    python scripts/manifest.py validate manifest.csv
    python scripts/manifest.py list     manifest.csv

The ``count`` subcommand is designed for shell scripts that set a SLURM
array size:

    N=$(python scripts/manifest.py count manifest.csv)
    sbatch --array=0-$((N-1)) kinsim_extract.slurm manifest.csv shards/

The manifest CSV columns are ``sample_id, bam_path, motifs`` with optional
``ref_path``. Comment rows starting with ``#`` and blank rows are skipped.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _load_rows(manifest: Path) -> list[dict]:
    out: list[dict] = []
    with manifest.open() as f:
        reader = csv.DictReader(_strip_comments(f))
        for row in reader:
            sid = (row.get("sample_id") or "").strip()
            if not sid:
                continue
            out.append({k: (v or "").strip() for k, v in row.items()})
    return out


def _strip_comments(lines):
    for line in lines:
        s = line.lstrip()
        if not s or s.startswith("#"):
            continue
        yield line


def _cmd_count(args: argparse.Namespace) -> int:
    rows = _load_rows(Path(args.manifest))
    print(len(rows))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    rows = _load_rows(Path(args.manifest))
    if not rows:
        print("(empty manifest)")
        return 0
    cols = ["sample_id", "bam_path", "motifs", "ref_path"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(r.get(c, "") for c in cols))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    rows = _load_rows(Path(args.manifest))
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, r in enumerate(rows, start=1):
        sid = r.get("sample_id", "")
        if sid in seen_ids:
            errors.append(f"row {i}: duplicate sample_id {sid!r}")
        seen_ids.add(sid)
        bam = r.get("bam_path", "")
        if not bam:
            errors.append(f"row {i} ({sid}): missing bam_path")
        elif not Path(bam).is_file():
            errors.append(f"row {i} ({sid}): bam_path does not exist: {bam}")
        ref = r.get("ref_path", "")
        if ref and not Path(ref).is_file():
            errors.append(f"row {i} ({sid}): ref_path does not exist: {ref}")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print(f"OK: {len(rows)} entries, no errors")
    return 0


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python scripts/manifest.py",
        description="Inspect or validate a KinSim manifest CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    p_count = sub.add_parser("count", help="Print number of active data rows")
    p_count.add_argument("manifest")
    p_count.set_defaults(func=_cmd_count)

    p_validate = sub.add_parser("validate", help="Check files and duplicates")
    p_validate.add_argument("manifest")
    p_validate.set_defaults(func=_cmd_validate)

    p_list = sub.add_parser("list", help="Tab-separated dump of every entry")
    p_list.add_argument("manifest")
    p_list.set_defaults(func=_cmd_list)

    args = p.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
