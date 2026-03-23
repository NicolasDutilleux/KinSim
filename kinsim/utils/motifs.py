"""IUPAC motif parsing, methylation scanning, and reference pre-scanning.

Supports three motif input sources (auto-detected by load_motif_string):
  1. KinSim motif string  — "m6A,GATC,1;m4C,CCWGG,1"
  2. PacBio motifs.csv    — output of SMRT Link basecall pipeline
  3. REBASE file          — simplified two-column or Format #19 (withrefm)
                            Delegated to kinsim.rebase_parser

Motif scanning backends:
  - EMBOSS fuzznuc (primary): used for reference-level genome pre-scanning.
    A single subprocess call with a named-pattern file (@patterns.txt) covers
    all motifs at once.  Falls back to regex automatically if fuzznuc is not
    installed (no error, just a warning).
  - Python regex (in-memory): retained for per-read scanning during BAM
    training and for unmapped-read fallback paths in inject/generate.
    Running fuzznuc via subprocess inside a per-read BAM loop would be
    prohibitively slow; the regex backend handles these cases efficiently.

Motif string format:
  "m6A,GATC,1;m4C,CCWGG,1;m5C,RGATCY,4"
  Each entry: MOD_TYPE,IUPAC_MOTIF,0-based_MOD_POS[,nDetected[,fraction]]
  Fields 4 (nDetected) and 5 (fraction) are optional metadata from PacBio CSV.
  They are ignored by train/inject/generate logic but preserved for traceability.
"""

from __future__ import annotations

import sys
import csv
import os
import re
import subprocess
import tempfile
import numpy as np
from .encoding import METH_IDS

IUPAC_TO_REGEX = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'N': '.',
    'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
    'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
    'H': '[ACT]', 'V': '[ACG]'
}

COMPLEMENT = {
    'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
    'Y': 'R', 'R': 'Y', 'S': 'S', 'W': 'W', 'K': 'M', 'M': 'K',
    'B': 'V', 'V': 'B', 'D': 'H', 'H': 'D'
}

# PacBio CSV: resolve ambiguous "modified_base" by the base at centerPos
_BASE_TO_METH = {'A': 'm6A', 'C': 'm4C'}

# GFF attribute parser: extracts pattern name from fuzznuc GFF output.
# Matches "Pattern_name=...", "Name=...", or "pattern=..." (case-insensitive).
_GFF_ATTR_NAME_RE = re.compile(r'(?:Pattern_name|Name|pattern)=([^;]+)',
                                re.IGNORECASE)


# ---------------------------------------------------------------------------
# IUPAC helpers
# ---------------------------------------------------------------------------

def iupac_to_re(motif):
    """Convert an IUPAC motif string to a regex pattern string."""
    return "".join(IUPAC_TO_REGEX.get(b, b) for b in motif)


def reverse_complement(seq):
    """Reverse complement supporting IUPAC ambiguity codes."""
    return "".join(COMPLEMENT.get(base, base) for base in reversed(seq))


# ---------------------------------------------------------------------------
# KinSim motif string: parse and scan (in-memory regex backend)
# ---------------------------------------------------------------------------

def parse_motifs(motif_string, revcomp=True):
    """Parse a motif string and compile regex for forward + reverse complement.

    IN-MEMORY REGEX BACKEND — used for per-read scanning during BAM training
    (dictionary/train.py, cgan/parse_train.py) and unmapped-read fallback in
    inject/generate.  This function must remain regex-based because fuzznuc
    subprocess calls per read are prohibitively slow.

    For reference-level scanning (done once per genome), use
    build_reference_meth_map() instead, which uses EMBOSS fuzznuc as the
    primary backend.

    Input format: "m6A,GATC,1;m4C,CCWGG,1;m5C,RGATCY,4"
    Each entry: MOD_TYPE,IUPAC_MOTIF,MOD_POS[,nDetected[,fraction]] — semicolon-delimited.
    Fields beyond index 2 are optional metadata (ignored here, preserved for traceability).

    Args:
        motif_string: Semicolon-delimited motif entries.
        revcomp: If True (default), generate both forward and reverse complement
            patterns.  Set to False when motif_string already contains both
            orientations (e.g., from PacBio CSV with partner motifs).

    Returns list of dicts with keys: 'pattern' (compiled regex with lookahead),
    'id' (methylation type int), 'pos' (modified base offset within match).
    """
    motifs = []
    if not motif_string:
        return motifs
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry:
            continue
        parts = entry.split(',')
        m_type, seq, pos = parts[0], parts[1], parts[2]
        m_id = METH_IDS.get(m_type, 0)
        mod_pos = int(pos) - 1  # 1-based input → 0-based internal

        pairs = [(seq, mod_pos)]
        if revcomp:
            pairs.append((reverse_complement(seq), len(seq) - 1 - mod_pos))

        for s, offset in pairs:
            regex_pattern = re.compile(f'(?=({iupac_to_re(s)}))')
            motifs.append({'pattern': regex_pattern, 'id': m_id, 'pos': offset})
    return motifs


def scan_sequence(seq, motifs):
    """Scan a DNA sequence for methylation motifs (in-memory regex backend).

    IN-MEMORY REGEX BACKEND — called per read during BAM training loops
    (dictionary/train.py, cgan/parse_train.py) and as a fallback for unmapped
    reads during injection (dictionary/inject.py, cgan/generate.py).

    For reference-level scanning (done once per genome), use
    build_reference_meth_map() which delegates to EMBOSS fuzznuc as the
    primary backend.

    Returns an int8 numpy array of length len(seq), where each position
    holds the methylation type ID (0 = unmethylated).
    """
    status = np.zeros(len(seq), dtype=np.int8)
    for motif in motifs:
        for match in motif['pattern'].finditer(seq):
            target_pos = match.start() + motif['pos']
            if target_pos < len(seq):
                status[target_pos] = motif['id']
    return status


# ---------------------------------------------------------------------------
# PacBio motifs.csv parser
# ---------------------------------------------------------------------------

def parse_motifs_csv(csv_path, min_fraction=0.40, min_detected=20):
    """Parse a PacBio motifs.csv and return a KinSim motif string.

    Required columns: ``motifString``, ``centerPos``.
    Optional columns: ``modificationType``, ``fraction``, ``nDetected``.

    Filtering:
      - fraction  < min_fraction  → skipped (blank fraction bypasses filter)
      - nDetected < min_detected  → skipped (blank nDetected bypasses filter)

    modificationType handling:
      - ``m6A`` / ``m4C`` / ``m5C`` → used directly
      - ``modified_base`` / blank   → inferred from base at centerPos (A→m6A, C→m4C)

    Returns an empty string if the file is not a PacBio CSV (missing required
    columns), allowing the caller to fall through to alternative parsers.

    Returns:
        A semicolon-delimited motif string:
        ``"m6A,GCCGATC,5,3551,0.998;m6A,CTGAAG,5,2891,1.0"``
        Fields: MOD_TYPE, MOTIF, POS, nDetected, fraction
    """
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])

        # Required columns — return "" if absent (not a PacBio CSV)
        if 'motifString' not in fieldnames or 'centerPos' not in fieldnames:
            return ""

        has_fraction   = 'fraction'         in fieldnames
        has_ndetected  = 'nDetected'        in fieldnames
        has_mod_type   = 'modificationType' in fieldnames

        for lineno, row in enumerate(reader, 2):
            motif_seq  = row.get('motifString', '').strip()
            center_str = row.get('centerPos', '').strip()
            if not motif_seq or not center_str:
                continue

            try:
                center_pos = int(center_str)
            except ValueError:
                log.warning("motifs.csv line %d: invalid centerPos — skipped", lineno)
                continue

            # fraction — blank → bypass filter
            fraction: float | None = None
            if has_fraction:
                frac_str = row.get('fraction', '').strip()
                if frac_str:
                    try:
                        fraction = float(frac_str)
                    except ValueError:
                        pass

            if fraction is not None and fraction < min_fraction:
                continue

            # nDetected — blank → bypass filter
            n_detected: int | None = None
            if has_ndetected:
                nd_str = row.get('nDetected', '').strip()
                if nd_str:
                    try:
                        n_detected = int(nd_str)
                    except ValueError:
                        pass

            if n_detected is not None and n_detected < min_detected:
                continue

            # modificationType
            mod_type = row.get('modificationType', '').strip() if has_mod_type else ''

            if mod_type in ('modified_base', ''):
                idx = center_pos - 1   # centerPos is 1-based in CSV
                if idx < 0 or idx >= len(motif_seq):
                    log.warning("motifs.csv line %d: centerPos %d OOB for '%s' — skipped",
                                lineno, center_pos, motif_seq)
                    continue
                base = motif_seq[idx].upper()
                resolved = _BASE_TO_METH.get(base)
                if resolved is None:
                    log.warning("motifs.csv line %d: cannot infer mod type at "
                                "%s[%d]='%s' — skipped",
                                lineno, motif_seq, center_pos, base)
                    continue
                mod_type = resolved

            if mod_type not in METH_IDS:
                log.warning("motifs.csv line %d: unknown mod type '%s' for "
                            "%s — skipped", lineno, mod_type, motif_seq)
                continue

            nd_out  = n_detected if n_detected is not None else 0
            fr_out  = fraction   if fraction   is not None else 1.0
            entries.append(
                f"{mod_type},{motif_seq},{center_pos},{nd_out},{fr_out:.6g}"
            )

    return ";".join(entries)


# ---------------------------------------------------------------------------
# Unified motif-string loader (auto-detect source)
# ---------------------------------------------------------------------------

def load_motif_string(motifs_arg, min_fraction=0.40, min_detected=20,
                      parser_name=None):
    """Load a KinSim motif string from a file path or return the argument as-is.

    Auto-detection (when parser_name is None):
        1. If motifs_arg is an existing file path ending in '.csv'
           -> parse as PacBio motifs.csv (applies min_fraction / min_detected)
        2. Try auto_detect_parser() from the callers registry
        3. Fall through to REBASE file parser
        4. Otherwise -> treat as a literal KinSim motif string

    Args:
        motifs_arg:    File path or motif string.
        min_fraction:  Minimum fraction threshold (PacBio CSV only).
        min_detected:  Minimum nDetected threshold (PacBio CSV only).
        parser_name:   Explicit parser name ("pacbio", "modkit", "ipd_summary").
                       When provided, bypasses auto-detection.

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    # Explicit parser requested
    if parser_name is not None:
        from prep.callers import create_parser
        parser = create_parser(parser_name)
        return parser.parse(motifs_arg,
                            min_fraction=min_fraction,
                            min_detected=min_detected)

    if os.path.isfile(motifs_arg):
        # Try the callers registry first (covers combined CSV, PacBio CSV,
        # modkit, ipd_summary) — auto-detection is more precise than the
        # legacy parse_motifs_csv fallback.
        from prep.callers import auto_detect_parser
        parser = auto_detect_parser(motifs_arg)
        if parser is not None:
            return parser.parse(motifs_arg,
                                min_fraction=min_fraction,
                                min_detected=min_detected)

        # Legacy PacBio CSV parser (returns "" when columns are missing,
        # so it is safe to try on any .csv file)
        if motifs_arg.lower().endswith('.csv'):
            result = parse_motifs_csv(motifs_arg,
                                      min_fraction=min_fraction,
                                      min_detected=min_detected)
            if result:
                return result

        # Fall through to REBASE
        from prep.rebase import parse_rebase_file
        return parse_rebase_file(motifs_arg)

    return motifs_arg


# ---------------------------------------------------------------------------
# Reference-level methylation map (pre-scan entire genome once)
# ---------------------------------------------------------------------------

def build_reference_meth_map(ref_seqs, motif_string, revcomp=True,
                              no_fuzznuc=False):
    """Pre-scan a reference genome for methylation sites.

    PRIMARY BACKEND: EMBOSS fuzznuc — tried first unless no_fuzznuc=True.
    Uses a single subprocess call with a named-pattern file (@patterns.txt),
    covering all motifs at once for efficiency and scientific reproducibility.

    FALLBACK: Python regex — used automatically if fuzznuc is not installed
    (prints a warning) or if no_fuzznuc=True.

    Scanning the reference once and caching results in a per-position array
    enables O(1) methylation lookup during read injection, regardless of
    whether fuzznuc or regex is used.

    Args:
        ref_seqs:     dict[name] -> sequence string (from load_reference).
        motif_string: KinSim motif string ("m6A,GATC,1;m4C,CCWGG,1").
        revcomp:      Also scan the reverse complement strand (default True).
        no_fuzznuc:   Force Python regex mode; skip fuzznuc entirely.

    Returns:
        dict[ref_name] -> np.int8 array of shape (ref_len,)
        Each position holds the methylation type ID (0 = unmethylated).
        For circular-genome lookups, index with pos % ref_len.
    """
    if not no_fuzznuc:
        try:
            return _build_meth_map_fuzznuc(ref_seqs, motif_string, revcomp)
        except FileNotFoundError:
            print("  WARN: fuzznuc not found on PATH - falling back to "
                  "Python regex scanner", file=sys.stderr)
    return _build_meth_map_regex(ref_seqs, motif_string, revcomp)


def _build_meth_map_regex(ref_seqs, motif_string, revcomp=True):
    """Build reference methylation map using Python regex (fallback backend)."""
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    return {name: scan_sequence(seq, motifs) for name, seq in ref_seqs.items()}


def build_reference_frac_map(ref_seqs, motif_string, revcomp=True):
    """Build per-position stoichiometric fraction map for the reference genome.

    Each methylated position gets the fraction from the specific motif that
    matched it.  Unmethylated positions remain 0.0.  This avoids the problem
    of collapsing different fractions for the same meth_id (e.g. two m6A
    motifs with 99% and 10% fractions).

    Uses Python regex scanning (fast enough for single-genome validation).

    Args:
        ref_seqs:     dict[name] -> sequence string (from load_reference).
        motif_string: KinSim motif string with optional fraction field.
        revcomp:      Also scan the reverse complement strand (default True).

    Returns:
        dict[ref_name] -> np.float32 array of shape (ref_len,)
        Each position holds the stoichiometric fraction (0.0 = unmethylated).
    """
    # Parse motifs with their fractions
    motif_entries = []
    if motif_string:
        for entry in motif_string.split(';'):
            if not entry or ',' not in entry:
                continue
            parts = entry.split(',')
            if len(parts) < 3:
                continue
            seq = parts[1]
            mod_pos = int(parts[2]) - 1  # 1-based → 0-based
            frac = float(parts[4]) if len(parts) >= 5 else 1.0

            pairs = [(seq, mod_pos)]
            if revcomp:
                pairs.append((reverse_complement(seq), len(seq) - 1 - mod_pos))

            for s, offset in pairs:
                regex_pattern = re.compile(f'(?=({iupac_to_re(s)}))')
                motif_entries.append((regex_pattern, offset, frac))

    frac_map = {}
    for name, seq in ref_seqs.items():
        fmap = np.zeros(len(seq), dtype=np.float32)
        for pattern, offset, frac in motif_entries:
            for match in pattern.finditer(seq):
                target_pos = match.start() + offset
                if target_pos < len(seq):
                    fmap[target_pos] = frac
        frac_map[name] = fmap
    return frac_map


def _build_meth_map_fuzznuc(ref_seqs, motif_string, revcomp=True):
    """Build reference methylation map using EMBOSS fuzznuc (primary backend).

    A single fuzznuc subprocess call scans all motifs at once using a named
    pattern file.  GFF output is parsed, and each hit's pattern name (from
    the attributes column) is decoded to retrieve meth_id and mod_pos.

    Strand-position arithmetic:
        + strand match at [Start, End] (1-based), modified pos p (0-based):
            meth_pos = (Start - 1) + p
        - strand match at [Start, End] (1-based), modified pos p (0-based):
            meth_pos = (End - 1) - p
        (End is 1-based inclusive; the - strand 5' corresponds to End on +)
    """
    from prep.rebase import write_fuzznuc_pattern_file

    if not motif_string:
        return {name: np.zeros(len(seq), dtype=np.int8)
                for name, seq in ref_seqs.items()}

    meth_map = {name: np.zeros(len(seq), dtype=np.int8)
                for name, seq in ref_seqs.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write reference FASTA
        ref_fa = os.path.join(tmpdir, 'ref.fa')
        with open(ref_fa, 'w') as fh:
            for name, seq in ref_seqs.items():
                fh.write(f'>{name}\n{seq}\n')

        # Write named-pattern file and get lookup dict
        pattern_file = os.path.join(tmpdir, 'patterns.txt')
        pattern_lookup = write_fuzznuc_pattern_file(motif_string, pattern_file)

        if not pattern_lookup:
            return meth_map

        out_gff = os.path.join(tmpdir, 'hits.gff')
        cmd = [
            'fuzznuc',
            '-sequence', ref_fa,
            '-pattern', f'@{pattern_file}',
            '-pmismatch', '0',
            '-complement', 'Y' if revcomp else 'N',
            '-rformat', 'gff',
            '-outfile', out_gff,
            '-auto',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARN: fuzznuc failed: {result.stderr.strip()}\n"
                  f"  Falling back to Python regex scanner.", file=sys.stderr)
            return _build_meth_map_regex(ref_seqs, motif_string, revcomp)

        if not os.path.exists(out_gff):
            print("  WARN: fuzznuc produced no output file - falling back to "
                  "Python regex scanner", file=sys.stderr)
            return _build_meth_map_regex(ref_seqs, motif_string, revcomp)

        # Parse GFF output: extract pattern name from attributes to identify motif
        with open(out_gff) as gff:
            for line in gff:
                if line.startswith('#') or not line.strip():
                    continue
                cols = line.split('\t')
                if len(cols) < 7:
                    continue
                ref_name   = cols[0]
                start_1b   = int(cols[3])
                end_1b     = int(cols[4])
                strand     = cols[6]
                attrs      = cols[8].strip() if len(cols) > 8 else ''

                if ref_name not in meth_map:
                    continue

                # Decode which motif this hit corresponds to
                meth_id, mod_pos = 0, 0
                attr_match = _GFF_ATTR_NAME_RE.search(attrs)
                if attr_match:
                    pname = attr_match.group(1).strip()
                    if pname in pattern_lookup:
                        meth_id, mod_pos = pattern_lookup[pname]
                    else:
                        # Try decode from name convention directly
                        from prep.rebase import decode_fuzznuc_pattern_name
                        meth_id, mod_pos = decode_fuzznuc_pattern_name(pname)

                if strand == '+':
                    meth_pos = (start_1b - 1) + mod_pos
                else:
                    meth_pos = (end_1b - 1) - mod_pos

                ref_len = len(ref_seqs[ref_name])
                if 0 <= meth_pos < ref_len:
                    meth_map[ref_name][meth_pos] = meth_id

    return meth_map


# ---------------------------------------------------------------------------
# CLI: kinsim motifs
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim motifs",
        description=(
            "Parse a motif source and print the KinSim motif string.\n\n"
            "Accepted inputs:\n"
            "  PacBio motifs.csv  — filtered by --min-fraction / --min-detected\n"
            "  REBASE file        — simplified two-column or Format #19 (withrefm)\n"
            "  Motif string       — pass directly as the 'input' argument\n\n"
            "Auto-detection: if the argument is a file ending in '.csv' it is\n"
            "treated as PacBio CSV; any other existing file is treated as REBASE;\n"
            "otherwise it is printed as-is after basic validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",
                        help="PacBio motifs.csv, REBASE file, or KinSim motif string")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold for PacBio CSV (default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold for PacBio CSV (default: 20)")
    args = parser.parse_args(argv)

    result = load_motif_string(args.input,
                               min_fraction=args.min_fraction,
                               min_detected=args.min_detected)
    if result:
        print(result)
    else:
        print("No motifs found / passed the filter.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
