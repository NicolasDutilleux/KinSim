"""Motif merging, deduplication, and standardized PacBio CSV output.

Two-step workflow
-----------------
1. Fetch REBASE motifs -> rebase_motifs.csv (standard PacBio format)::

       kinsim-prep rebase fetch <org_num> --output rebase_motifs.csv

2. Merge calling-derived CSV + rebase_motifs.csv -> final_motifs.csv::

       kinsim-prep merge-motifs species_motifs.csv rebase_motifs.csv \\
           --output final_motifs.csv --min-frac 0.8 --min-sites 300

Input formats accepted (auto-detected per file):

    Combined CSV  -- mod_type,motif,offset,frac_mod,n_sites,source
                    (output from modkit + fibertools pipeline)
    PacBio CSV    -- motifString,centerPos,modificationType,fraction,...
                    (output from SMRT Link, or from kinsim-prep rebase)

Filtering (applied before deduplication):
    --min-frac    Minimum frac_mod / fraction  (default: 0.80)
    --min-sites   Minimum n_sites / nGenome    (default: 300)
    Fields that are absent or blank bypass their respective filter
    (e.g. REBASE entries have no n_sites -- they are always retained).

Deduplication
-------------
If motif A (longer) IUPAC-contains motif B (shorter) at the same relative
modified-base position, A is considered a redundant extension of B and is
removed.

Example for E. coli Dam methyltransferase (m6A at GATC):

    6mA, GATC,  offset=1   <- core motif (KEPT)
    6mA, GATCA, offset=1   <- GATC + trailing A  -> REMOVED
    6mA, CGATC, offset=2   <- C + GATC           -> REMOVED
    6mA, TGATC, offset=2   <- T + GATC           -> REMOVED
    ...

Output format
-------------
Standard PacBio motifs.csv (comma-separated, 12 columns):

    motifString, centerPos, modificationType, fraction,
    nDetected, nGenome, groupTag, partnerMotifString,
    meanScore, meanIpdRatio, meanCoverage, objectiveScore

Columns without available data are written as empty strings.
This file is directly parseable by ``kinsim-prep parse`` (PacBioParser),
which converts it to a KinSim motif string for use in the pipeline.
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

# Standard PacBio motifs.csv column order
PACBIO_COLUMNS = [
    'motifString', 'centerPos', 'modificationType', 'fraction',
    'nDetected', 'nGenome', 'groupTag', 'partnerMotifString',
    'meanScore', 'meanIpdRatio', 'meanCoverage', 'objectiveScore',
]

# Normalize mod type names to KinSim/PacBio convention (m6A, m5C, m4C)
_MOD_NORMALIZE: dict[str, str] = {
    '6mA':  'm6A', 'm6A':  'm6A',
    '5mC':  'm5C', 'm5C':  'm5C',
    '4mC':  'm4C', 'm4C':  'm4C',
    '5hmC': 'm5C',   # 5-hydroxymethylcytosine -> treat as m5C
}

# IUPAC base expansion -- used for motif containment checks
_IUPAC_EXPAND: dict[str, frozenset[str]] = {
    'A': frozenset('A'), 'C': frozenset('C'),
    'G': frozenset('G'), 'T': frozenset('T'),
    'R': frozenset('AG'), 'Y': frozenset('CT'),
    'S': frozenset('GC'), 'W': frozenset('AT'),
    'K': frozenset('GT'), 'M': frozenset('AC'),
    'B': frozenset('CGT'),  'D': frozenset('AGT'),
    'H': frozenset('ACT'),  'V': frozenset('ACG'),
    'N': frozenset('ACGT'),
}


# ---------------------------------------------------------------------------
# IUPAC containment utilities
# ---------------------------------------------------------------------------

def _iupac_compatible(a: str, b: str) -> bool:
    """True if two IUPAC bases can represent the same concrete nucleotide."""
    a_set = _IUPAC_EXPAND.get(a.upper(), frozenset())
    b_set = _IUPAC_EXPAND.get(b.upper(), frozenset())
    return bool(a_set & b_set)


def motif_contains(longer: str, offset_longer: int,
                   shorter: str, offset_shorter: int) -> bool:
    """True if *longer* IUPAC-contains *shorter* at the aligned modified position.

    Containment requires two conditions:
      1. There exists a starting position ``p`` within *longer* such that
         every base of *shorter* is IUPAC-compatible with ``longer[p+i]``.
      2. The modified position of *longer* aligns with that of *shorter*:
         ``p + offset_shorter == offset_longer``.

    Args:
        longer:         Longer IUPAC recognition sequence.
        offset_longer:  0-based index of the modified base in *longer*.
        shorter:        Shorter IUPAC recognition sequence.
        offset_shorter: 0-based index of the modified base in *shorter*.

    Returns:
        True if *longer* is a superset context of *shorter* and should be
        considered redundant when *shorter* is already in the motif set.
    """
    len_l, len_s = len(longer), len(shorter)
    if len_l <= len_s:
        return False
    for start in range(len_l - len_s + 1):
        if start + offset_shorter != offset_longer:
            continue
        if all(_iupac_compatible(longer[start + i], shorter[i])
               for i in range(len_s)):
            return True
    return False


# ---------------------------------------------------------------------------
# Internal motif entry structure
# ---------------------------------------------------------------------------

def _make_entry(
    motif_str: str,
    offset: int,
    mod_type: str,
    fraction: float | str = '',
    n_detected: int | str = '',
    n_genome: int | str = '',
    mean_coverage: float | str = '',
    source: str = '',
) -> dict:
    """Create a normalized motif entry dict."""
    norm_mod = _MOD_NORMALIZE.get(mod_type, mod_type)
    return {
        'motif':         motif_str.strip().upper(),
        'offset':        int(offset),
        'mod_type':      norm_mod,
        'fraction':      fraction,
        'n_detected':    n_detected,
        'n_genome':      n_genome,
        'mean_coverage': mean_coverage,
        'source':        source,
    }


# ---------------------------------------------------------------------------
# Input parsers (auto-detect combined CSV vs PacBio CSV per file)
# ---------------------------------------------------------------------------

def _parse_motif_file(filepath: str) -> list[dict]:
    """Parse a motif CSV file; auto-detect combined vs PacBio format.

    Combined CSV format:
        mod_type, motif, offset, frac_mod, n_sites, source

    PacBio CSV format:
        motifString, centerPos, modificationType, fraction,
        nDetected, nGenome, ...
    """
    entries: list[dict] = []

    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            log.warning("Empty or headerless file: %s", filepath)
            return []

        fieldnames = set(reader.fieldnames)
        is_combined = 'mod_type' in fieldnames and 'frac_mod' in fieldnames
        is_pacbio   = 'motifString' in fieldnames and 'centerPos' in fieldnames

        if is_combined:
            fmt_label = "combined CSV"
            entries = _parse_combined_rows(reader, filepath)
        elif is_pacbio:
            fmt_label = "PacBio CSV"
            entries = _parse_pacbio_rows(reader, filepath)
        else:
            log.error(
                "Unrecognized CSV format in '%s'.\n"
                "  Expected combined CSV (mod_type,motif,offset,frac_mod,...)\n"
                "  or PacBio CSV (motifString,centerPos,modificationType,...).\n"
                "  Found columns: %s",
                filepath, sorted(fieldnames),
            )
            sys.exit(1)

    log.info("Parsed %d motifs from %s (format: %s)", len(entries), filepath, fmt_label)
    for e in entries:
        frac_str = f"frac={e['fraction']:.2f}" if isinstance(e['fraction'], float) else "frac=N/A"
        sites_str = f"sites={e['n_genome']}" if isinstance(e['n_genome'], int) else "sites=N/A"
        log.info("  [%s] %s %s offset=%d  %s  %s  (from: %s)",
                 e['mod_type'], e['motif'], e['mod_type'], e['offset'],
                 frac_str, sites_str, filepath)
    return entries


def _parse_combined_rows(reader: csv.DictReader, filepath: str) -> list[dict]:
    """Parse rows from a combined CSV (mod_type,motif,offset,frac_mod,n_sites,source)."""
    entries: list[dict] = []
    for lineno, row in enumerate(reader, 2):
        mod_raw   = row.get('mod_type', '').strip()
        motif_seq = row.get('motif', '').strip().upper()
        off_str   = row.get('offset', '').strip()
        frac_str  = row.get('frac_mod', '').strip()
        ns_str    = row.get('n_sites', '').strip()
        source    = row.get('source', '').strip()

        if not motif_seq or not off_str or not mod_raw:
            continue

        norm_mod = _MOD_NORMALIZE.get(mod_raw)
        if norm_mod is None:
            log.warning("%s line %d: unknown mod_type '%s' -- skipped",
                        filepath, lineno, mod_raw)
            continue

        try:
            offset   = int(off_str)
            fraction = float(frac_str) if frac_str else ''
            n_genome = int(ns_str) if ns_str else ''
        except ValueError:
            log.warning("%s line %d: invalid numeric field -- skipped",
                        filepath, lineno)
            continue

        entries.append(_make_entry(
            motif_str=motif_seq,
            offset=offset,
            mod_type=norm_mod,
            fraction=fraction,
            n_genome=n_genome,
            source=source,
        ))
    return entries


def _parse_pacbio_rows(reader: csv.DictReader, filepath: str) -> list[dict]:
    """Parse rows from a PacBio motifs.csv (motifString,centerPos,...)."""
    entries: list[dict] = []
    fieldnames = set(reader.fieldnames or [])
    has_fraction   = 'fraction'          in fieldnames
    has_ndetected  = 'nDetected'         in fieldnames
    has_ngenome    = 'nGenome'           in fieldnames
    has_mod_type   = 'modificationType'  in fieldnames

    # Base->mod type fallback (for 'modified_base' entries)
    _base_to_meth = {'A': 'm6A', 'C': 'm4C'}

    for lineno, row in enumerate(reader, 2):
        motif_seq  = row.get('motifString', '').strip().upper()
        center_str = row.get('centerPos', '').strip()
        if not motif_seq or not center_str:
            continue

        try:
            offset = int(center_str)
        except ValueError:
            log.warning("%s line %d: invalid centerPos -- skipped",
                        filepath, lineno)
            continue

        # modificationType
        mod_raw = row.get('modificationType', '').strip() if has_mod_type else ''
        norm_mod = _MOD_NORMALIZE.get(mod_raw, '')
        if not norm_mod:
            if mod_raw in ('modified_base', ''):
                if offset >= len(motif_seq):
                    log.warning("%s line %d: centerPos %d OOB for '%s' -- skipped",
                                filepath, lineno, offset, motif_seq)
                    continue
                base = motif_seq[offset].upper()
                norm_mod = _base_to_meth.get(base, '')
                if not norm_mod:
                    log.warning("%s line %d: cannot infer mod type from '%s'[%d] -- skipped",
                                filepath, lineno, motif_seq, offset)
                    continue
            else:
                log.warning("%s line %d: unknown modificationType '%s' -- skipped",
                            filepath, lineno, mod_raw)
                continue

        # fraction
        fraction: float | str = ''
        if has_fraction:
            frac_str = row.get('fraction', '').strip()
            if frac_str:
                try:
                    fraction = float(frac_str)
                except ValueError:
                    pass

        # nDetected
        n_detected: int | str = ''
        if has_ndetected:
            nd_str = row.get('nDetected', '').strip()
            if nd_str:
                try:
                    n_detected = int(nd_str)
                except ValueError:
                    pass

        # nGenome
        n_genome: int | str = ''
        if has_ngenome:
            ng_str = row.get('nGenome', '').strip()
            if ng_str:
                try:
                    n_genome = int(ng_str)
                except ValueError:
                    pass

        entries.append(_make_entry(
            motif_str=motif_seq,
            offset=offset,
            mod_type=norm_mod,
            fraction=fraction,
            n_detected=n_detected,
            n_genome=n_genome,
            source='pacbio',
        ))
    return entries


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _apply_filters(
    entries: list[dict],
    min_frac: float,
    min_sites: int,
) -> list[dict]:
    """Filter motifs by minimum fraction and minimum n_sites.

    Blank fields bypass their filter -- entries without coverage data
    (e.g. REBASE-derived motifs) are always retained.
    """
    kept: list[dict] = []
    for e in entries:
        frac     = e.get('fraction', '')
        n_genome = e.get('n_genome', '')
        label = f"{e['mod_type']} {e['motif']} offset={e['offset']}"

        if isinstance(frac, float) and frac < min_frac:
            log.info("  [FILTERED] %s -- fraction=%.3f < min_frac=%.2f  (source: %s)",
                     label, frac, min_frac, e.get('source', '?'))
            continue
        if isinstance(n_genome, int) and n_genome < min_sites:
            log.info("  [FILTERED] %s -- n_sites=%d < min_sites=%d  (source: %s)",
                     label, n_genome, min_sites, e.get('source', '?'))
            continue

        bypass_parts = []
        if not isinstance(frac, float):
            bypass_parts.append("frac=N/A (bypass)")
        if not isinstance(n_genome, int):
            bypass_parts.append("sites=N/A (bypass)")
        bypass_note = f"  [{', '.join(bypass_parts)}]" if bypass_parts else ""
        log.info("  [KEPT]     %s%s", label, bypass_note)
        kept.append(e)

    n_dropped = len(entries) - len(kept)
    log.info("Filtering summary: %d in -> %d kept, %d dropped (min_frac=%.2f, min_sites=%d)",
             len(entries), len(kept), n_dropped, min_frac, min_sites)
    return kept


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_motifs(entries: list[dict]) -> list[dict]:
    """Remove motifs that are redundant extensions of shorter core motifs.

    For each mod_type group: if motif A (longer) IUPAC-contains motif B
    (shorter) at the aligned modified position, A is redundant and removed.

    The shorter motif (B) captures the same methylation signal as its longer
    extension (A) because the modified base is at the same sequence context.
    Shorter = more fundamental = preferred.

    Args:
        entries: List of motif entry dicts.

    Returns:
        Deduplicated list, sorted by mod_type then motif length.
    """
    by_mod: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_mod[e['mod_type']].append(e)

    result: list[dict] = []
    for mod_type, motifs in sorted(by_mod.items()):
        # Sort ascending by length so shorter motifs come first
        sorted_m = sorted(motifs, key=lambda x: len(x['motif']))
        n = len(sorted_m)
        redundant = [False] * n

        for i in range(n):
            if redundant[i]:
                continue
            for j in range(n):
                if i == j or redundant[j]:
                    continue
                longer  = sorted_m[j]
                shorter = sorted_m[i]
                if len(longer['motif']) <= len(shorter['motif']):
                    continue
                if motif_contains(
                    longer['motif'],  longer['offset'],
                    shorter['motif'], shorter['offset'],
                ):
                    redundant[j] = True
                    log.info("  [DEDUP]    %s %s (offset=%d) removed -- "
                             "contained in shorter core: %s (offset=%d)",
                             mod_type, longer['motif'], longer['offset'],
                             shorter['motif'], shorter['offset'])

        kept = [m for m, r in zip(sorted_m, redundant) if not r]
        for m in kept:
            log.info("  [FINAL]    %s %s offset=%d", mod_type, m['motif'], m['offset'])
        n_removed = n - len(kept)
        log.info("  %s: %d -> %d motifs after deduplication (%d redundant removed)",
                 mod_type, n, len(kept), n_removed)
        result.extend(kept)

    return result


# ---------------------------------------------------------------------------
# Exact-duplicate merge (same mod_type + motif + offset from multiple files)
# ---------------------------------------------------------------------------

def _merge_entries(a: list[dict], b: list[dict]) -> list[dict]:
    """Combine two entry lists, preferring the richer entry for exact duplicates.

    An exact duplicate is: same (mod_type, motif, offset) tuple.
    When both lists contain the same key, the entry with the larger
    n_genome value is kept (more genomic evidence = more reliable).
    """
    seen: dict[tuple, dict] = {}
    # Track which keys came from which list for logging
    from_a: set[tuple] = set()
    from_b: set[tuple] = set()

    for e in a:
        key = (e['mod_type'], e['motif'], e['offset'])
        from_a.add(key)
        seen[key] = e

    for e in b:
        key = (e['mod_type'], e['motif'], e['offset'])
        from_b.add(key)
        if key not in seen:
            seen[key] = e
        else:
            existing_ng = seen[key].get('n_genome') or 0
            new_ng      = e.get('n_genome') or 0
            if new_ng > existing_ng:
                seen[key] = e

    # Log cross-file matches and unique motifs
    matched   = from_a & from_b
    only_in_a = from_a - from_b
    only_in_b = from_b - from_a

    if matched:
        log.info("Cross-file matches (%d motifs found in BOTH sources):", len(matched))
        for key in sorted(matched):
            e = seen[key]
            src = e.get('source', '?')
            log.info("  [MATCH]    %s %s offset=%d  (kept from: %s)",
                     key[0], key[1], key[2], src)

    if only_in_a:
        log.info("Motifs only in previous inputs (%d):", len(only_in_a))
        for key in sorted(only_in_a):
            e = seen[key]
            src = e.get('source', '?')
            log.info("  [UNIQUE]   %s %s offset=%d  (source: %s)",
                     key[0], key[1], key[2], src)

    if only_in_b:
        log.info("Motifs only in new input (%d):", len(only_in_b))
        for key in sorted(only_in_b):
            e = seen[key]
            src = e.get('source', '?')
            log.info("  [UNIQUE]   %s %s offset=%d  (source: %s)",
                     key[0], key[1], key[2], src)

    log.info("Merge: %d + %d -> %d unique motifs (%d matched across files)",
             len(a), len(b), len(seen), len(matched))

    return list(seen.values())


# ---------------------------------------------------------------------------
# Standard PacBio CSV output
# ---------------------------------------------------------------------------

def write_pacbio_motifs_csv(entries: list[dict], filepath: str) -> None:
    """Write motifs to standard PacBio motifs.csv format (comma-separated).

    All 12 standard columns are written; unavailable fields are empty strings.
    The output is directly parseable by PacBioParser::

        kinsim-prep parse final_motifs.csv   -> KinSim motif string

    Args:
        entries:  List of motif entry dicts from :func:`_make_entry`.
        filepath: Output file path (parent directories created if needed).
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(PACBIO_COLUMNS)
        for e in entries:
            writer.writerow([
                e['motif'],                   # motifString
                e['offset'],                  # centerPos
                e['mod_type'],                # modificationType
                e.get('fraction', ''),        # fraction
                e.get('n_detected', ''),      # nDetected
                e.get('n_genome', ''),        # nGenome
                '',                           # groupTag
                e['motif'],                   # partnerMotifString (self-ref)
                '',                           # meanScore
                '',                           # meanIpdRatio
                e.get('mean_coverage', ''),   # meanCoverage
                '',                           # objectiveScore
            ])

    log.info("Wrote %d motifs to %s", len(entries), filepath)


# ---------------------------------------------------------------------------
# Public merge pipeline
# ---------------------------------------------------------------------------

def merge_motifs(
    input_files: list[str],
    output_path: str,
    *,
    min_frac: float = 0.8,
    min_sites: int = 300,
    deduplicate: bool = True,
) -> dict:
    """Merge, filter, and deduplicate motifs from multiple CSV sources.

    Accepts any mix of combined CSV and PacBio CSV files.  Auto-detects the
    format of each input file independently.

    Args:
        input_files: List of input CSV paths (combined or PacBio format).
        output_path: Path to write the merged standard PacBio motifs.csv.
        min_frac:    Minimum frac_mod / fraction to retain (default 0.8).
                     Entries with blank fraction bypass this filter.
        min_sites:   Minimum n_sites / nGenome to retain (default 300).
                     Entries with blank n_sites bypass this filter.
        deduplicate: Apply IUPAC motif containment deduplication (default True).

    Returns:
        Stats dict: {motifs_in, motifs_after_filter, motifs_out}.
    """
    # Parse and merge all input files
    log.info("=" * 60)
    log.info("MERGE-MOTIFS: %d input file(s)", len(input_files))
    for i, path in enumerate(input_files, 1):
        log.info("  Input %d: %s", i, path)
    log.info("  Output:  %s", output_path)
    log.info("  Filters: min_frac=%.2f, min_sites=%d, dedup=%s",
             min_frac, min_sites, deduplicate)
    log.info("=" * 60)

    merged: list[dict] = []
    for i, path in enumerate(input_files, 1):
        log.info("--- Parsing input %d/%d: %s ---", i, len(input_files), path)
        entries = _parse_motif_file(path)
        if merged:
            log.info("--- Merging input %d with previous entries ---", i)
        merged  = _merge_entries(merged, entries)

    motifs_in = len(merged)
    log.info("--- Total unique motifs after merging all inputs: %d ---", motifs_in)

    # Filter
    filtered = _apply_filters(merged, min_frac=min_frac, min_sites=min_sites)
    motifs_after_filter = len(filtered)

    # Deduplicate
    if deduplicate:
        log.info("--- Deduplication (IUPAC containment) ---")
        deduped = deduplicate_motifs(filtered)
    else:
        log.info("--- Deduplication: SKIPPED (--no-dedup) ---")
        deduped = filtered

    motifs_out = len(deduped)

    # Sort: mod_type ASC, motif length ASC, motif string ASC
    deduped.sort(key=lambda e: (e['mod_type'], len(e['motif']), e['motif']))

    # Write standard PacBio CSV
    write_pacbio_motifs_csv(deduped, output_path)

    log.info("=" * 60)
    log.info("MERGE-MOTIFS COMPLETE")
    log.info("  Total from all inputs: %d", motifs_in)
    log.info("  After filtering:       %d  (%d removed)",
             motifs_after_filter, motifs_in - motifs_after_filter)
    log.info("  After deduplication:   %d  (%d redundant removed)",
             motifs_out, motifs_after_filter - motifs_out)
    log.info("  Written to: %s", output_path)
    log.info("=" * 60)

    return {
        'motifs_in':           motifs_in,
        'motifs_after_filter': motifs_after_filter,
        'motifs_out':          motifs_out,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse
    from kinsim.utils.config import setup_logging

    parser = argparse.ArgumentParser(
        prog="kinsim-prep merge-motifs",
        description=(
            "Merge, filter, and deduplicate motifs from multiple sources\n"
            "into a single standard PacBio motifs.csv.\n\n"
            "Accepted input formats (auto-detected per file):\n"
            "  Combined CSV  : mod_type,motif,offset,frac_mod,n_sites,source\n"
            "                  (output from modkit + fibertools pipeline)\n"
            "  PacBio CSV    : motifString,centerPos,modificationType,...\n"
            "                  (output from 'kinsim-prep rebase fetch')\n\n"
            "Deduplication removes motifs that are IUPAC-supersets of shorter cores:\n"
            "  6mA GATC (offset=1) is the core; CGATC, GATCA, TGATC, GGATC\n"
            "  are all extensions and will be removed.\n\n"
            "Output is a standard PacBio motifs.csv, readable by:\n"
            "  kinsim-prep parse final_motifs.csv  -> KinSim motif string\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="+",
        help=(
            "One or more motif CSV files.  Formats accepted:\n"
            "  - Combined CSV  : mod_type,motif,offset,frac_mod,n_sites,source\n"
            "  - PacBio CSV    : motifString,centerPos,modificationType,..."
        ),
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output standard PacBio motifs.csv (e.g. final_motifs.csv)",
    )
    parser.add_argument(
        "--min-frac", type=float, default=0.8,
        help=(
            "Minimum frac_mod / fraction to retain a motif (default: 0.8). "
            "Entries with blank fraction (e.g. from REBASE) bypass this filter."
        ),
    )
    parser.add_argument(
        "--min-sites", type=int, default=300,
        help=(
            "Minimum n_sites / nGenome to retain a motif (default: 300). "
            "Entries with blank n_sites (e.g. from REBASE) bypass this filter."
        ),
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable IUPAC motif containment deduplication.",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG-level logging")

    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    # Validate inputs
    for f in args.inputs:
        if not Path(f).is_file():
            print(f"ERROR: input file not found: {f}", file=sys.stderr)
            sys.exit(1)

    stats = merge_motifs(
        input_files=args.inputs,
        output_path=args.output,
        min_frac=args.min_frac,
        min_sites=args.min_sites,
        deduplicate=not args.no_dedup,
    )

    # Summary to stdout
    n_in   = stats['motifs_in']
    n_filt = stats['motifs_after_filter']
    n_out  = stats['motifs_out']
    pct_f  = n_filt / n_in * 100 if n_in else 0
    pct_d  = n_out  / n_filt * 100 if n_filt else 0

    print(f"Input motifs:        {n_in:>6}")
    print(f"After filtering:     {n_filt:>6}  ({pct_f:.1f}% retained,"
          f" min_frac={args.min_frac}, min_sites={args.min_sites})")
    print(f"After deduplication: {n_out:>6}  ({pct_d:.1f}% retained)")
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()
