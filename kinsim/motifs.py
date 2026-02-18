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
  Each entry: MOD_TYPE,IUPAC_MOTIF,0-based_MOD_POS[,nDetected]
"""

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
    Each entry: MOD_TYPE,IUPAC_MOTIF,MOD_POS[,extra] — semicolon-delimited.

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
        mod_pos = int(pos)

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

    Logic:
      1. Skip rows where fraction < min_fraction OR nDetected < min_detected.
      2. If modificationType is already m6A / m4C / m5C, use it directly.
      3. If modificationType is "modified_base", resolve by looking at the
         base at motifString[centerPos]: A -> m6A, C -> m4C.
      4. Rows that can't be resolved are skipped with a warning.

    Returns:
        A semicolon-delimited motif string with 4 fields per entry:
        "m6A,GCCGATC,5,3551;m6A,CTGAAG,5,2891"
        Fields: MOD_TYPE,MOTIF,POS,nDetected
    """
    entries = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fraction = float(row['fraction'])
            n_detected = int(row['nDetected'])
            if fraction < min_fraction or n_detected < min_detected:
                continue

            motif_seq = row['motifString'].strip()
            center_pos = int(row['centerPos'])
            mod_type = row['modificationType'].strip()

            if mod_type == 'modified_base':
                if center_pos >= len(motif_seq):
                    print(f"  WARN: centerPos {center_pos} out of bounds "
                          f"for {motif_seq} — skipped", file=sys.stderr)
                    continue
                base = motif_seq[center_pos].upper()
                resolved = _BASE_TO_METH.get(base)
                if resolved is None:
                    print(f"  WARN: cannot resolve modified_base at "
                          f"{motif_seq}[{center_pos}]='{base}' — skipped",
                          file=sys.stderr)
                    continue
                mod_type = resolved

            if mod_type not in METH_IDS:
                print(f"  WARN: unknown mod type '{mod_type}' for "
                      f"{motif_seq} — skipped", file=sys.stderr)
                continue

            entries.append(f"{mod_type},{motif_seq},{center_pos},{n_detected}")

    return ";".join(entries)


# ---------------------------------------------------------------------------
# Unified motif-string loader (auto-detect source)
# ---------------------------------------------------------------------------

def load_motif_string(motifs_arg, min_fraction=0.40, min_detected=20):
    """Load a KinSim motif string from a file path or return the argument as-is.

    Auto-detection:
        - If motifs_arg is an existing file path ending in '.csv'
          -> parse as PacBio motifs.csv (applies min_fraction / min_detected)
        - If motifs_arg is any other existing file path
          -> parse as REBASE file (auto-detects simplified or Format #19)
        - Otherwise -> treat as a literal KinSim motif string

    Args:
        motifs_arg:    File path or motif string.
        min_fraction:  Minimum fraction threshold (PacBio CSV only).
        min_detected:  Minimum nDetected threshold (PacBio CSV only).

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    if os.path.isfile(motifs_arg):
        if motifs_arg.lower().endswith('.csv'):
            return parse_motifs_csv(motifs_arg,
                                    min_fraction=min_fraction,
                                    min_detected=min_detected)
        else:
            from .rebase_parser import parse_rebase_file
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
            print("  WARN: fuzznuc not found on PATH — falling back to "
                  "Python regex scanner", file=sys.stderr)
    return _build_meth_map_regex(ref_seqs, motif_string, revcomp)


def _build_meth_map_regex(ref_seqs, motif_string, revcomp=True):
    """Build reference methylation map using Python regex (fallback backend)."""
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    return {name: scan_sequence(seq, motifs) for name, seq in ref_seqs.items()}


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
    from .rebase_parser import write_fuzznuc_pattern_file

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
            print("  WARN: fuzznuc produced no output file — falling back to "
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
                        from .rebase_parser import decode_fuzznuc_pattern_name
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
