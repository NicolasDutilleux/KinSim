"""Prepare config files for KinSim pipeline (dictionary and cGAN modes).

Reads a text file with alternating lines:
  - Odd lines:  absolute path to a BAM file
  - Even lines: absolute path to the corresponding motifs.csv

Outputs a new text file with the same structure, but the CSV paths are
replaced by compact one-liner motif strings parsed from the CSV.

The output format is compatible with both:
  - kinsim.dictionary.train (uses first 3 fields: type,motif,pos)
  - kinsim.cgan (uses all 4 fields: type,motif,pos,nDetected)
"""

import sys
import os

from .motifs import parse_motifs_csv


def prepare_config(input_file, output_file, min_fraction=0.40, min_detected=20):
    """Read BAM+CSV pairs and write BAM+motif-string pairs.

    Args:
        input_file:  path to text file with alternating BAM / motifs.csv lines
        output_file: path to output config (alternating BAM / motif-string lines)
        min_fraction: minimum fraction threshold for motif filtering
        min_detected: minimum nDetected threshold for motif filtering
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
        bam_path = lines[i * 2]
        csv_path = lines[i * 2 + 1]
        label = os.path.basename(bam_path)

        if not os.path.isfile(csv_path):
            print(f"  WARN: CSV not found: {csv_path} — skipping pair", file=sys.stderr)
            skipped += 1
            continue

        motif_string = parse_motifs_csv(csv_path, min_fraction=min_fraction,
                                        min_detected=min_detected)

        if not motif_string:
            print(f"  WARN: no motifs passed filter for {label} — skipping pair",
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
        description="Parse BAM + motifs.csv pairs into a config file for KinSim pipeline. "
                    "Filters motifs by fraction and nDetected, resolves modification types.",
        epilog="Input format (alternating lines):\n"
               "  /path/to/strain1.bam\n"
               "  /path/to/strain1/motifs.csv\n"
               "  /path/to/strain2.bam\n"
               "  /path/to/strain2/motifs.csv\n\n"
               "Output format (alternating lines):\n"
               "  /path/to/strain1.bam\n"
               "  m6A,GCCGATC,5,3551;m6A,CTGAAG,5,2891",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Text file with alternating BAM / motifs.csv lines")
    parser.add_argument("output", help="Output config file (BAM / motif-string lines)")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold (default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (default: 20)")
    args = parser.parse_args(argv)
    prepare_config(args.input, args.output,
                   min_fraction=args.min_fraction, min_detected=args.min_detected)


if __name__ == "__main__":
    main()
