"""REBASE methylation file parsing and fuzznuc pattern file generation.

Supports two REBASE input formats:
  1. Simplified two-column format:
         GATC    2(6)
         CCWGG   2(5),-1(5)
  2. REBASE Format #19 (withrefm / allenz-style tagged records):
         ID   M.TaqI
         RS   TCGA, ?;
         MS   3(6mA);
         //

Also provides write_fuzznuc_pattern_file() which converts a KinSim motif
string into a named-pattern file for fuzznuc's '@file' syntax, enabling a
single subprocess call across all motifs.

REBASE X(Y) position notation:
    X = 1-based position (positive = forward strand,
                          negative = complementary strand, from its 5' end)
    Y = 6 (m6A), 5 (m5C), 4 (m4C)

REBASE Format #19 MS field uses position(type) where type is 6mA, N4mC, or 5mC.
"""

import os
import re
import sys

from .encoding import METH_IDS

# REBASE Y-code → KinSim mod type (used in simple X(Y) notation)
_REBASE_CODE_TO_METH = {'6': 'm6A', '5': 'm5C', '4': 'm4C'}

# REBASE Format #19 MS type strings → KinSim mod type
_REBASE_TYPE_TO_METH = {
    '6mA': 'm6A',
    '5mC': 'm5C',
    'N4mC': 'm4C',
}

# Regex for simplified two-column REBASE annotations: "2(6)" or "-1(4)"
_SITE_RE = re.compile(r'(-?\d+)\((\d)\)')

# Regex for Format #19 MS field annotations: "3(6mA)" or "-1(N4mC)"
_MS_SITE_RE = re.compile(r'(-?\d+)\((\w+)\)')


# ---------------------------------------------------------------------------
# X(Y) notation parser (shared between simple and Format #19)
# ---------------------------------------------------------------------------

def parse_rebase_annotation(recognition_seq, meth_annotation):
    """Parse a REBASE X(Y) methylation annotation into KinSim motif entries.

    REBASE methylation site notation:
        X(Y)  or  X1(Y1),X2(Y2)
    Where:
        X  = 1-based position within the recognition sequence (positive =
             forward strand; negative = complementary strand counted from
             its 5' end, i.e. from the 3' end of the top strand)
        Y  = modification type: 6 = m6A, 5 = m5C, 4 = m4C

    Conversion to KinSim 0-based position:
        positive X  ->  pos = X - 1
        negative X  ->  pos = len(recognition_seq) - abs(X)  (0-based, top strand)

    Returns:
        List of "MOD_TYPE,RECOGNITION_SEQ,POS" strings (no nDetected field).
    """
    recognition_seq = recognition_seq.strip().upper()
    seq_len = len(recognition_seq)
    entries = []

    for site_match in _SITE_RE.finditer(meth_annotation):
        x = int(site_match.group(1))
        y = site_match.group(2)

        meth_type = _REBASE_CODE_TO_METH.get(y)
        if meth_type is None:
            print(f"  WARN: REBASE: unknown methylation code ({y}) — skipped",
                  file=sys.stderr)
            continue

        if x > 0:
            pos_0 = x - 1            # 1-based -> 0-based
        else:
            pos_0 = seq_len - abs(x)  # from 5' of complementary strand

        if not (0 <= pos_0 < seq_len):
            print(f"  WARN: REBASE: position {x} out of range for "
                  f"'{recognition_seq}' (len={seq_len}) — skipped", file=sys.stderr)
            continue

        entries.append(f"{meth_type},{recognition_seq},{pos_0}")
    return entries


# ---------------------------------------------------------------------------
# Simplified two-column REBASE format
# ---------------------------------------------------------------------------

def parse_rebase_simple(filepath):
    """Parse a simplified two-column REBASE tab-delimited file.

    Expects lines of the form:
        RECOGNITION_SEQUENCE    METHYLATION_SITES

    Example:
        GATC    2(6)
        CCWGG   2(5)
        GCWGC   2(6),-1(6)

    Lines beginning with '#' are comments and are skipped.
    Blank lines are skipped.

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    all_entries = []
    with open(filepath) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"  WARN: REBASE line {lineno}: expected 2 columns, "
                      f"got {len(parts)} — skipped", file=sys.stderr)
                continue
            rec_seq = parts[0]
            meth_ann = parts[1]
            entries = parse_rebase_annotation(rec_seq, meth_ann)
            all_entries.extend(entries)
    return ';'.join(all_entries)


# ---------------------------------------------------------------------------
# REBASE Format #19 (withrefm / allenz-style tagged records)
# ---------------------------------------------------------------------------

def parse_rebase_withrefm(filepath):
    """Parse a REBASE Format #19 file (withrefm or allenz-style).

    Record format:
        ID   enzyme_name
        ET   enzyme_type
        OS   source_organism
        RS   recognition_sequence, cut_site;
        MS   position(type)[,position(type)];
        //   end of record

    MS field position notation:
        position  = 1-based (positive = forward strand, negative = complementary)
        type      = 6mA | N4mC | 5mC

    Records are skipped when:
        - RS is '?' (unknown recognition sequence)
        - MS is absent or '?' (no methylation info)
        - RS contains characters other than IUPAC codes

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    all_entries = []

    with open(filepath) as f:
        current = {}
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('//'):
                # End of record: process if we have RS and MS
                rec_seq = current.get('RS', '').strip()
                ms_raw  = current.get('MS', '').strip()

                if rec_seq and rec_seq != '?' and ms_raw and ms_raw != '?':
                    # Clean recognition sequence: remove cleavage site indicators
                    # RS can be "GATC, 2;" or "G^AATTC" or "GATC"
                    rec_clean = re.sub(r'[^ACGTRYMKSWHBVDN]', '', rec_seq.upper())
                    if rec_clean:
                        for site_match in _MS_SITE_RE.finditer(ms_raw):
                            x   = int(site_match.group(1))
                            typ = site_match.group(2)
                            meth_type = _REBASE_TYPE_TO_METH.get(typ)
                            if meth_type is None:
                                continue

                            seq_len = len(rec_clean)
                            if x > 0:
                                pos_0 = x - 1
                            else:
                                pos_0 = seq_len - abs(x)

                            if 0 <= pos_0 < seq_len:
                                all_entries.append(
                                    f"{meth_type},{rec_clean},{pos_0}")

                current = {}
                continue

            if '   ' in line or '\t' in line:
                # Tagged-field line: "ID   name" or "RS   GATC, 2;"
                sep = line.find('\t') if '\t' in line else line.find('   ')
                tag = line[:sep].strip()
                val = line[sep:].strip()
                # Only keep the last RS/MS per record (some entries repeat)
                if tag in ('ID', 'RS', 'MS', 'ET'):
                    current[tag] = val

    return ';'.join(all_entries)


# ---------------------------------------------------------------------------
# Auto-detecting REBASE file parser
# ---------------------------------------------------------------------------

def parse_rebase_file(filepath):
    """Auto-detect REBASE format and parse accordingly.

    Detection heuristic:
        - If the file contains lines starting with 'ID   ' or 'RS   '
          (three spaces after the two-char tag) -> Format #19 (withrefm)
        - Otherwise -> simplified two-column format

    Returns:
        A semicolon-delimited KinSim motif string.
    """
    with open(filepath) as f:
        for line in f:
            if line.startswith(('ID   ', 'RS   ', 'MS   ')):
                return parse_rebase_withrefm(filepath)
    return parse_rebase_simple(filepath)


# ---------------------------------------------------------------------------
# Fuzznuc pattern file generator
# ---------------------------------------------------------------------------

def write_fuzznuc_pattern_file(motif_string, filepath):
    """Convert a KinSim motif string to a fuzznuc named-pattern file.

    Writes a file in PROSITE/fuzznuc format with named patterns:
        >m6A_GATC_1
        GATC
        >m4C_CCWGG_1
        CCWGG

    Pattern names encode the motif type, sequence, and 0-based modified
    position, allowing GFF output to be linked back to meth_id and mod_pos
    without re-parsing the pattern file.

    Args:
        motif_string: KinSim semicolon-delimited motif string.
        filepath:     Path to write the pattern file.

    Returns:
        dict mapping pattern_name -> (meth_id, mod_pos)
        so the caller can look up each fuzznuc GFF hit.
    """
    pattern_lookup = {}  # name -> (meth_id, mod_pos)

    lines = []
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry:
            continue
        parts = entry.split(',')
        m_type, seq, pos_str = parts[0], parts[1], parts[2]
        meth_id  = METH_IDS.get(m_type, 0)
        mod_pos  = int(pos_str)

        # Unique name encodes all fields needed for GFF lookup
        name = f"{m_type}_{seq}_{mod_pos}"

        # Deduplicate: if two entries produce the same name, keep first
        if name not in pattern_lookup:
            pattern_lookup[name] = (meth_id, mod_pos)
            lines.append(f">{name}\n{seq}")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    return pattern_lookup


def decode_fuzznuc_pattern_name(name):
    """Decode a pattern name generated by write_fuzznuc_pattern_file.

    Inverse of the naming scheme: "{m_type}_{seq}_{mod_pos}"

    Returns (meth_id, mod_pos) or (0, 0) if decoding fails.
    """
    parts = name.split('_')
    if len(parts) < 3:
        return 0, 0
    try:
        mod_pos = int(parts[-1])
        m_type  = parts[0]
        meth_id = METH_IDS.get(m_type, 0)
        return meth_id, mod_pos
    except (ValueError, IndexError):
        return 0, 0


# ---------------------------------------------------------------------------
# CLI: kinsim rebase
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim rebase",
        description=(
            "Parse REBASE files and generate fuzznuc pattern files.\n\n"
            "Accepted REBASE formats:\n"
            "  Simplified two-column   — RECOGNITION_SEQ  X(Y)[,X2(Y2)]\n"
            "  Format #19 (withrefm)   — tagged records (ID/RS/MS fields)\n"
            "                           Auto-detected from file content.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- parse subcommand --
    p_parse = sub.add_parser(
        "parse",
        help="Parse a REBASE file and print the KinSim motif string",
        description=(
            "Parse a REBASE file (auto-detects simplified or Format #19) and\n"
            "print the resulting KinSim motif string to stdout.\n\n"
            "The motif string can then be passed directly to any kinsim command\n"
            "that accepts a motif string argument."
        ),
    )
    p_parse.add_argument("input",
                         help="REBASE file (simplified two-column or Format #19)")

    # -- patterns subcommand --
    p_patt = sub.add_parser(
        "patterns",
        help="Convert a motif source to a fuzznuc pattern file",
        description=(
            "Convert a motif source (KinSim string, PacBio CSV, or REBASE file)\n"
            "to a fuzznuc-compatible named pattern file.\n\n"
            "The output can be used with: fuzznuc -pattern @<output> ..."
        ),
    )
    p_patt.add_argument("motifs",
                        help="Motif source: KinSim string, REBASE file, or PacBio CSV")
    p_patt.add_argument("output",
                        help="Output fuzznuc pattern file")
    p_patt.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold (PacBio CSV only, default: 0.40)")
    p_patt.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (PacBio CSV only, default: 20)")

    args = parser.parse_args(argv)

    if args.command == "parse":
        result = parse_rebase_file(args.input)
        if result:
            print(result)
        else:
            print("No motifs found in the REBASE file.", file=sys.stderr)
            sys.exit(1)

    elif args.command == "patterns":
        from .motifs import load_motif_string
        motif_string = load_motif_string(args.motifs,
                                         min_fraction=args.min_fraction,
                                         min_detected=args.min_detected)
        if not motif_string:
            print("ERROR: no motifs found from the provided source.", file=sys.stderr)
            sys.exit(1)
        lookup = write_fuzznuc_pattern_file(motif_string, args.output)
        print(f"Pattern file written to {args.output} ({len(lookup)} patterns)")
        for name, (mid, pos) in lookup.items():
            print(f"  {name}  ->  meth_id={mid} mod_pos={pos}")


if __name__ == "__main__":
    main()
