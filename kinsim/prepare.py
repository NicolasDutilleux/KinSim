"""Prepare config files for KinSim pipeline (dictionary and cGAN modes).

Reads a text file with alternating lines:
  - Odd lines:  absolute path to a BAM file
  - Even lines: absolute path to a motif source file

Supported motif source formats (auto-detected):
  - PacBio motifs.csv  (ends in '.csv')
  - REBASE tab-delimited file  (any other extension)
  - KinSim motif string  (not a file path — used as-is)

Outputs a new text file with the same BAM-path lines, but the motif source
lines are replaced by compact KinSim motif strings:
  "m6A,GCCGATC,5,3551;m6A,CTGAAG,5,2891"

The output format is compatible with both:
  - kinsim.dictionary.train (uses first 3 fields: type,motif,pos)
  - kinsim.cgan (uses all 4 fields: type,motif,pos,nDetected)

Note: nDetected is only present in PacBio CSV output.  REBASE-derived entries
do not include a 4th field; cGAN mode will treat missing nDetected as 0.
"""

import sys
import os

from .motifs import load_motif_string


def prepare_config(input_file, output_file, min_fraction=0.40, min_detected=20):
    """Read BAM + motif-source pairs and write BAM + motif-string pairs.

    Args:
        input_file:    Path to text file with alternating BAM / motif-source lines.
                       Motif sources may be PacBio CSV, REBASE files, or inline
                       KinSim motif strings.
        output_file:   Path to output config (alternating BAM / motif-string lines).
        min_fraction:  Minimum fraction threshold for PacBio CSV filtering.
        min_detected:  Minimum nDetected threshold for PacBio CSV filtering.
    """
    with open(input_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) % 2 != 0:
        print(f"ERROR: input file must have an even number of non-empty lines "
              f"(got {len(lines)})", file=sys.stderr)
        sys.exit(1)

    n_pairs = len(lines) // 2
    output_lines = []
    skipped = 0

    for i in range(n_pairs):
        bam_path  = lines[i * 2]
        motif_src = lines[i * 2 + 1]
        label     = os.path.basename(bam_path)

        # If motif_src looks like a file path, check it exists
        if os.sep in motif_src or motif_src.endswith('.csv') or motif_src.endswith('.txt'):
            if not os.path.isfile(motif_src):
                print(f"  WARN: motif file not found: {motif_src} — skipping pair",
                      file=sys.stderr)
                skipped += 1
                continue

        motif_string = load_motif_string(motif_src,
                                         min_fraction=min_fraction,
                                         min_detected=min_detected)

        if not motif_string:
            print(f"  WARN: no motifs found for {label} — skipping pair",
                  file=sys.stderr)
            skipped += 1
            continue

        output_lines.append(bam_path)
        output_lines.append(motif_string)

    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')

    kept = n_pairs - skipped
    print(f"Prepared {kept}/{n_pairs} strain pairs -> {output_file}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim prepare",
        description=(
            "Parse BAM + motif-source pairs into a config file for the KinSim pipeline.\n\n"
            "Accepted motif sources (auto-detected per line):\n"
            "  PacBio motifs.csv  — filtered by --min-fraction / --min-detected\n"
            "  REBASE file        — tab-delimited: RECOGNITION_SEQ  X(Y)[,X2(Y2)]\n"
            "  KinSim string      — used as-is: 'm6A,GATC,1;m4C,CCWGG,1'\n\n"
            "Input format (alternating lines):\n"
            "  /path/to/strain1.bam\n"
            "  /path/to/strain1/motifs.csv          # or a REBASE file\n"
            "  /path/to/strain2.bam\n"
            "  /path/to/strain2/rebase_motifs.txt   # or an inline motif string\n\n"
            "Output format (alternating lines):\n"
            "  /path/to/strain1.bam\n"
            "  m6A,GCCGATC,5,3551;m6A,CTGAAG,5,2891"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",
                        help="Text file with alternating BAM / motif-source lines")
    parser.add_argument("output",
                        help="Output config file (BAM / motif-string lines)")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold for PacBio CSV (default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold for PacBio CSV (default: 20)")
    args = parser.parse_args(argv)
    prepare_config(args.input, args.output,
                   min_fraction=args.min_fraction,
                   min_detected=args.min_detected)


if __name__ == "__main__":
    main()
