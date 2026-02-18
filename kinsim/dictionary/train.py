"""Build 11-mer kinetic dictionary from BAM files, or merge .pkl shards."""

import sys
import os
import pickle
import glob
import numpy as np
import pysam
from collections import defaultdict

from ..encoding import BASE_MAP, KMER_MASK, K, KMER_BITS
from ..motifs import parse_motifs, scan_sequence, load_motif_string


def train_single_bam(bam_path, motif_string, revcomp=True):
    """Process a single BAM file and return the lookup dictionary.

    For each read: extract sequence + fi/fp tags, scan methylation motifs,
    then slide an 11-mer window accumulating (n, sum_ipd, sum_ipd², sum_pw, sum_pw²).

    Args:
        bam_path: Path to BAM file with fi/fp kinetic tags.
        motif_string: Semicolon-delimited motif string.
        revcomp: Generate reverse complement motif patterns (default True).

    Returns dict[(int_kmer, meth_id)] -> np.array([n, sum_ipd, sum_ipd2, sum_pw, sum_pw2])
    """
    mid = K // 2  # 5
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    lookup = defaultdict(lambda: np.zeros(5, dtype=np.float64))

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            seq = read.query_sequence
            if not (seq and len(seq) >= K and read.has_tag('fi')):
                continue

            ipds = read.get_tag('fi')
            pws = read.get_tag('fp')
            min_len = min(len(seq), len(ipds), len(pws))

            # Scan methylation positions using in-memory regex.
            # (fuzznuc is used only for reference-level pre-scanning in inject.py;
            # per-read subprocess calls would be prohibitively slow here)
            meth_status = scan_sequence(seq[:min_len], motifs)

            # Sliding window bit-packing
            current_kmer = 0
            for i in range(min_len):
                base_val = BASE_MAP.get(seq[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK
                if i >= K - 1:
                    center = i - mid
                    key = (current_kmer, int(meth_status[center]))
                    acc = lookup[key]
                    ipd_val = float(ipds[center])
                    pw_val = float(pws[center])
                    acc[0] += 1
                    acc[1] += ipd_val
                    acc[2] += ipd_val ** 2
                    acc[3] += pw_val
                    acc[4] += pw_val ** 2

    return dict(lookup)


def merge_shards(input_dir, output_file):
    """Merge multiple *_binary.pkl shard files into one master dictionary."""
    master = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    pattern = os.path.join(input_dir, "*_binary.pkl")
    files = glob.glob(pattern)

    if not files:
        print(f"Error: no '*_binary.pkl' files found in {input_dir}")
        return

    print(f"Merging {len(files)} shards from {input_dir}...")
    for f_path in files:
        with open(f_path, 'rb') as f:
            shard = pickle.load(f)
            for key, data in shard.items():
                master[key] += data

    with open(output_file, 'wb') as f:
        pickle.dump(dict(master), f)
    print(f"Master dictionary saved to {output_file}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim dictionary",
        description="Build an 11-mer kinetic dictionary from a BAM file, or merge "
                    "multiple .pkl shards into a master dictionary.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train subcommand --
    p_train = sub.add_parser(
        "train",
        help="Train a dictionary shard from a single BAM file",
        description="Extract IPD/PW accumulators from a BAM file for each 11-mer + "
                    "methylation state. Outputs a .pkl shard.",
    )
    p_train.add_argument("bam", help="Input BAM file with fi/fp kinetic tags")
    p_train.add_argument("motifs",
                         help="Motif source: KinSim string ('m6A,GATC,1'), "
                              "PacBio motifs.csv, or REBASE file (auto-detected)")
    p_train.add_argument("output", help="Output .pkl file for the dictionary shard")
    p_train.add_argument("--no-revcomp", action="store_true",
                         help="Do not scan reverse complement strand for motifs "
                              "(use when motif source already includes both orientations)")
    p_train.add_argument("--min-fraction", type=float, default=0.40,
                         help="Minimum fraction threshold for PacBio CSV (default: 0.40)")
    p_train.add_argument("--min-detected", type=int, default=20,
                         help="Minimum nDetected threshold for PacBio CSV (default: 20)")

    # -- merge subcommand --
    p_merge = sub.add_parser(
        "merge",
        help="Merge multiple *_binary.pkl shards into one master dictionary",
        description="Combine .pkl shard files by summing their accumulators. "
                    "Looks for *_binary.pkl files in the input directory.",
    )
    p_merge.add_argument("input_dir", help="Directory containing *_binary.pkl shard files")
    p_merge.add_argument("output", help="Output master dictionary .pkl file")

    args = parser.parse_args(argv)

    if args.command == "merge":
        merge_shards(args.input_dir, args.output)
    else:
        motif_string = load_motif_string(args.motifs,
                                         min_fraction=args.min_fraction,
                                         min_detected=args.min_detected)
        if not motif_string:
            print("ERROR: no motifs found from the provided source.", file=sys.stderr)
            sys.exit(1)
        print(f"Training on {os.path.basename(args.bam)}...")
        lookup = train_single_bam(args.bam, motif_string, revcomp=not args.no_revcomp)
        with open(args.output, 'wb') as f:
            pickle.dump(lookup, f)
        print(f"Dictionary saved to {args.output} ({len(lookup)} entries)")


if __name__ == "__main__":
    main()
