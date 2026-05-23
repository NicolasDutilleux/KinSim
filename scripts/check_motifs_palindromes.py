"""Pre-flight check: are palindromic motifs in motifs.csv listed for BOTH strands?

Background
----------
``kinsim.utils.motifs.parse_motifs_per_strand`` (called by ``kinsim_NN
generate`` and by older kinsim/kinsim2 pipelines) auto-generates the
reverse-strand entry for each motif row IF the reverse-complement's
``mod_pos`` lands on the right base for that meth type. For a
**palindromic** motif like GATC + m6A at pos 2 (1-based), the auto-rc
formula ``len-1-mod_pos`` lands on position 3 — a T, not an A — so the
auto-rc entry is dropped with a warning. As a result, only the + strand
of every palindromic site is conditioned at generate time, halving the
signal the model is asked to reproduce.

Fix (no code change needed): the source motifs.csv should list BOTH
strand rows explicitly for palindromic motifs. PacBio's motifFinder
output usually does this — but it's worth checking before launching a
full extract / training run.

What this script does
---------------------
For each motifs.csv (or directory of strain dirs containing motifs.csv):
* Parse the rows, group by ``motifString``.
* Identify palindromic motifs (motif == reverse_complement(motif)).
* For each palindromic motif, check whether 1 or 2 rows are listed.
* Report the per-strain count of "single-listed palindromes" — these are
  the ones that will be ignored on one strand at generate time.

Usage::

    python scripts/check_motifs_palindromes.py <motifs.csv | strain_dir | manifest.csv>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from kinsim.utils.motifs import reverse_complement


def _is_palindrome(motif: str) -> bool:
    return motif.upper() == reverse_complement(motif.upper())


def _check_motifs_csv(path: Path) -> dict:
    """Return per-(motif, mod_type) counts + palindrome status."""
    if not path.is_file():
        return {"path": str(path), "error": "missing"}
    rows: list[tuple[str, str]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "motifString" not in reader.fieldnames:
            return {"path": str(path), "error": "no motifString column"}
        for r in reader:
            motif = (r.get("motifString") or "").strip().upper()
            mod = (r.get("modificationType") or "").strip()
            if motif:
                rows.append((motif, mod))
    if not rows:
        return {"path": str(path), "n_rows": 0}

    # Group by motif and tally entries
    from collections import defaultdict
    by_motif: dict[str, int] = defaultdict(int)
    by_motif_type: dict[tuple[str, str], int] = defaultdict(int)
    for m, t in rows:
        by_motif[m] += 1
        by_motif_type[(m, t)] += 1

    palindromes = [m for m in by_motif if _is_palindrome(m)]
    single_listed = [m for m in palindromes if by_motif[m] == 1]
    return {
        "path": str(path),
        "n_rows": len(rows),
        "n_motifs": len(by_motif),
        "n_palindromes": len(palindromes),
        "n_palindromes_single_listed": len(single_listed),
        "single_listed_examples": single_listed[:5],
    }


def _walk_arg(arg: str) -> list[Path]:
    p = Path(arg)
    out: list[Path] = []
    if p.is_file() and p.name.endswith(".csv"):
        if p.name.startswith("manifest"):
            # Treat as manifest CSV: each row → motifs path in column "motifs"
            with open(p) as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "motifs" in reader.fieldnames:
                    for r in reader:
                        mp = (r.get("motifs") or "").strip()
                        if mp:
                            out.append(Path(mp))
                    return out
        out.append(p)
        return out
    if p.is_dir():
        out.extend(sorted(p.rglob("motifs.csv")))
        return out
    return [p]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="motifs.csv | strain dir | manifest.csv")
    args = ap.parse_args(argv)

    paths = _walk_arg(args.input)
    if not paths:
        print(f"No motifs.csv found under {args.input}", file=sys.stderr)
        sys.exit(1)

    total_single = 0
    print(f"{'file':<70} {'rows':>6} {'motifs':>7} {'palin':>6} {'single':>7}  examples")
    print("-" * 130)
    for p in paths:
        r = _check_motifs_csv(p)
        if "error" in r:
            print(f"{str(r['path']):<70} ERROR: {r['error']}")
            continue
        if r.get("n_rows", 0) == 0:
            print(f"{str(r['path']):<70} {'(empty)':>6}")
            continue
        single = r["n_palindromes_single_listed"]
        total_single += single
        examples = ",".join(r["single_listed_examples"]) if single else "-"
        print(
            f"{str(r['path']):<70} {r['n_rows']:>6} {r['n_motifs']:>7} "
            f"{r['n_palindromes']:>6} {single:>7}  {examples}"
        )
    print("-" * 130)
    print(f"TOTAL single-listed palindromes across all files: {total_single}")
    if total_single > 0:
        print(
            "\nWARNING: those palindromes will be conditioned on only ONE strand at\n"
            "generate time. Consider re-running motifFinder or duplicating the rows\n"
            "(swap modificationType-implied position to land on the correct base on rc)\n"
            "before launching the full extract / training run."
        )
    else:
        print("All palindromes are listed on both strands.")


if __name__ == "__main__":
    main()
