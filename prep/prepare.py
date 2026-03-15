"""Prepare config files for KinSim pipeline (dictionary and cGAN modes).

Reads a text file with alternating lines:
  - Odd lines:  absolute path to a BAM file
  - Even lines: absolute path to a motif source file

Supported motif source formats (auto-detected):
  - PacBio motifs.csv  (ends in '.csv')
  - REBASE tab-delimited file  (any other extension)
  - KinSim motif string  (not a file path -- used as-is)

Outputs a new text file with the same BAM-path lines, but the motif source
lines are replaced by compact KinSim motif strings:
  "m6A,GCCGATC,5,3551,0.998;m6A,CTGAAG,5,2891,1.0"

Field layout per semicolon-delimited entry:
  1. MOD_TYPE     -- m6A, m4C, or m5C
  2. IUPAC_MOTIF  -- recognition sequence
  3. POS          -- 0-based position of modified base
  4. nDetected    -- number of detected occurrences (PacBio CSV only)
  5. fraction     -- methylation fraction 0.0-1.0  (PacBio CSV only)

Fields 4 and 5 are absent for REBASE-derived entries. Downstream tools
(train, inject, generate) only read the first 3 fields. nDetected is
additionally used by cGAN mode for optional per-motif weighting.
"""

import logging
import os
import sys

from kinsim.motifs import load_motif_string

log = logging.getLogger(__name__)


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
        log.error(
            "Input file must have an even number of non-empty lines (got %d): %s",
            len(lines), input_file,
        )
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
                log.warning("Motif file not found: %s -- skipping pair", motif_src)
                skipped += 1
                continue

        motif_string = load_motif_string(motif_src,
                                         min_fraction=min_fraction,
                                         min_detected=min_detected)

        if not motif_string:
            log.warning("No motifs found for %s -- skipping pair", label)
            skipped += 1
            continue

        log.info("  %s -> %s", label, motif_string[:60] + ("..." if len(motif_string) > 60 else ""))
        output_lines.append(bam_path)
        output_lines.append(motif_string)

    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines) + '\n')

    kept = n_pairs - skipped
    log.info("Prepared %d/%d strain pairs -> %s", kept, n_pairs, output_file)


def main(argv=None):
    import argparse
    from kinsim.config import setup_logging
    parser = argparse.ArgumentParser(
        prog="kinsim-prep prepare",
        description=(
            "Parse BAM + motif-source pairs into a config file for the KinSim pipeline.\n\n"
            "Accepted motif sources (auto-detected per line):\n"
            "  PacBio motifs.csv  -- filtered by --min-fraction / --min-detected\n"
            "  REBASE file        -- tab-delimited: RECOGNITION_SEQ  X(Y)[,X2(Y2)]\n"
            "  KinSim string      -- used as-is: 'm6A,GATC,1;m4C,CCWGG,1'\n\n"
            "Input format (alternating lines):\n"
            "  /path/to/strain1.bam\n"
            "  /path/to/strain1/motifs.csv          # or a REBASE file\n"
            "  /path/to/strain2.bam\n"
            "  /path/to/strain2/rebase_motifs.txt   # or an inline motif string\n\n"
            "Output format (alternating lines):\n"
            "  /path/to/strain1.bam\n"
            "  m6A,GCCGATC,5,3551,0.998;m6A,CTGAAG,5,2891,1.0\n"
            "  (fields: MOD_TYPE,MOTIF,POS,nDetected,fraction)"
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
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG-level logging")
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)
    prepare_config(args.input, args.output,
                   min_fraction=args.min_fraction,
                   min_detected=args.min_detected)


if __name__ == "__main__":
    main()
