"""IUPAC motif parsing, regex compilation, and methylation scanning."""

import sys
import csv
import re
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


def iupac_to_re(motif):
    """Convert an IUPAC motif string to a regex pattern string.
    Example: GYCAGCYC → G[CT]CAG[CT]C
    """
    return "".join(IUPAC_TO_REGEX.get(b, b) for b in motif)


def reverse_complement(seq):
    """Reverse complement supporting IUPAC ambiguity codes."""
    return "".join(COMPLEMENT.get(base, base) for base in reversed(seq))


def parse_motifs(motif_string):
    """Parse a motif string and compile regex for forward + reverse complement.

    Input format: "m6A,GATC,2;m4C,CCWGG,1;m5C,RGATCY,4"
    Each entry: MOD_TYPE,IUPAC_MOTIF,MOD_POS[,extra] separated by semicolons.
    A 4th comma-separated field (e.g. nDetected) is accepted and ignored.

    Returns list of dicts with keys: 'pattern' (compiled regex with lookahead),
    'id' (methylation type int), 'pos' (modified base offset within match).
    Both forward and reverse complement patterns are generated.
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
        p = int(pos)

        for s, offset in [(seq, p), (reverse_complement(seq), len(seq) - 1 - p)]:
            regex_pattern = re.compile(f'(?=({iupac_to_re(s)}))')
            motifs.append({'pattern': regex_pattern, 'id': m_id, 'pos': offset})
    return motifs


def scan_sequence(seq, motifs):
    """Scan a DNA sequence for methylation motifs.

    Returns an int8 numpy array of length len(seq), where each position
    holds the methylation type ID (0 = unmethylated).
    """
    status = np.zeros(len(seq), dtype=np.int8)
    for m in motifs:
        for match in m['pattern'].finditer(seq):
            target_pos = match.start() + m['pos']
            if target_pos < len(seq):
                status[target_pos] = m['id']
    return status


# ---------------------------------------------------------------------------
# PacBio motifs.csv parser
# ---------------------------------------------------------------------------

# Resolve "modified_base" to a concrete type based on the base at centerPos
_BASE_TO_METH = {'A': 'm6A', 'C': 'm4C'}


def parse_motifs_csv(csv_path, min_fraction=0.40, min_detected=20):
    """Parse a PacBio motifs.csv and return a KinSim motif string.

    Logic:
      1. Skip rows where fraction < min_fraction OR nDetected < min_detected.
      2. If modificationType is already m6A / m4C / m5C, use it directly.
      3. If modificationType is "modified_base", resolve by looking at the
         base at motifString[centerPos]: A → m6A, C → m4C.
         Rows that can't be resolved are skipped with a warning.
      4. Clean: strip whitespace from motifString, ensure centerPos is int.

    Returns:
        A semicolon-delimited motif string with 4 fields per entry:
        "m6A,GCCGATC,5,3551;m6A,CTGAAG,5,2891"
        Fields: MOD_TYPE,MOTIF,POS,nDetected
        The 4th field (nDetected) is used by cGAN mode and ignored by dictionary mode.
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

            # Resolve modification type
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

            # Validate mod_type is known
            if mod_type not in METH_IDS:
                print(f"  WARN: unknown mod type '{mod_type}' for "
                      f"{motif_seq} — skipped", file=sys.stderr)
                continue

            entries.append(f"{mod_type},{motif_seq},{center_pos},{n_detected}")

    return ";".join(entries)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim motifs",
        description="Parse a single PacBio motifs.csv and print the KinSim motif string. "
                    "Filters by fraction and nDetected thresholds, resolves ambiguous "
                    "'modified_base' types (A->m6A, C->m4C).",
    )
    parser.add_argument("csv", help="Path to PacBio motifs.csv file")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold (default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (default: 20)")
    args = parser.parse_args(argv)
    result = parse_motifs_csv(args.csv, min_fraction=args.min_fraction,
                              min_detected=args.min_detected)
    if result:
        print(result)
    else:
        print("No motifs passed the filter.", file=sys.stderr)


if __name__ == "__main__":
    main()
