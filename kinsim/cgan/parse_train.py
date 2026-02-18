"""Extract raw IPD/PW training samples from BAM files for cGAN training.

Unlike dictionary mode (which stores running accumulators), cGAN mode
collects individual (IPD, PW) observations per (11-mer, methylation_state).
These raw samples are needed to train the conditional generator, which must
learn the full distribution shape — not just mean and variance.

Each shard is a dict[(int_kmer, meth_id)] -> np.ndarray of shape (N, 2),
where columns are [IPD, PW]. Shards from multiple BAMs are concatenated
during the merge step.

The nDetected field from the motif string (4th field) is preserved in the
output metadata for optional per-motif weighting during GAN training.
"""

import os
import sys
import pickle
import numpy as np
import pysam
from collections import defaultdict

from ..encoding import BASE_MAP, KMER_MASK, K
from ..motifs import parse_motifs, scan_sequence, load_motif_string


# ---------------------------------------------------------------------------
# Training: extract raw samples from a single BAM
# ---------------------------------------------------------------------------

def extract_samples_from_bam(bam_path, motif_string, max_samples_per_key=10_000, revcomp=True):
    """Extract raw (IPD, PW) pairs from a BAM file for each 11-mer context.

    For each read: extract sequence + fi/fp kinetic tags, scan methylation
    motifs, then slide an 11-mer window collecting raw signal values.

    Args:
        bam_path:   Path to BAM file with fi/fp kinetic tags.
        motif_string: Semicolon-delimited motif string (e.g. "m6A,GATC,2,3551").
        max_samples_per_key: Cap per (kmer, meth_id) to limit memory usage.
            Once a key reaches this count, new samples are randomly replaced
            (reservoir sampling) to maintain an unbiased sample.
        revcomp: Generate reverse complement motif patterns (default True).

    Returns:
        dict[(int_kmer, meth_id)] -> list of [IPD, PW] pairs
    """
    mid = K // 2  # 5
    motifs = parse_motifs(motif_string, revcomp=revcomp)
    samples = defaultdict(list)
    counts = defaultdict(int)  # total observations seen (for reservoir sampling)

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            seq = read.query_sequence
            if not (seq and len(seq) >= K and read.has_tag('fi')):
                continue

            ipds = read.get_tag('fi')
            pws = read.get_tag('fp')
            min_len = min(len(seq), len(ipds), len(pws))

            # Scan methylation positions using in-memory regex.
            # (fuzznuc is used only for reference-level pre-scanning in generate.py;
            # per-read subprocess calls would be prohibitively slow here)
            meth_status = scan_sequence(seq[:min_len], motifs)

            # Sliding window: encode 11-mers and collect raw signals
            current_kmer = 0
            for i in range(min_len):
                base_val = BASE_MAP.get(seq[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                if i >= K - 1:
                    center = i - mid
                    key = (current_kmer, int(meth_status[center]))
                    ipd_val = float(ipds[center])
                    pw_val = float(pws[center])

                    counts[key] += 1
                    n = counts[key]

                    if n <= max_samples_per_key:
                        # Still under the cap: append directly
                        samples[key].append([ipd_val, pw_val])
                    else:
                        # Reservoir sampling: replace a random existing sample
                        # with probability max_samples_per_key / n
                        j = np.random.randint(0, n)
                        if j < max_samples_per_key:
                            samples[key][j] = [ipd_val, pw_val]

    # Convert lists to numpy arrays for compact storage
    result = {}
    for key, vals in samples.items():
        result[key] = np.array(vals, dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# Merging: concatenate shards from multiple BAMs
# ---------------------------------------------------------------------------

def merge_shards(input_dir, output_file, max_samples_per_key=50_000):
    """Merge multiple *_cgan.pkl shard files into one master training set.

    Concatenates raw sample arrays per key. If a key exceeds
    max_samples_per_key after concatenation, a random subsample is taken.

    Args:
        input_dir:  Directory containing *_cgan.pkl shard files.
        output_file: Path for the merged output .pkl file.
        max_samples_per_key: Maximum samples to keep per (kmer, meth_id).
    """
    import glob

    master = defaultdict(list)  # key -> list of np.arrays to concatenate
    pattern = os.path.join(input_dir, "*_cgan.pkl")
    files = glob.glob(pattern)

    if not files:
        print(f"Error: no '*_cgan.pkl' files found in {input_dir}")
        return

    print(f"Merging {len(files)} cGAN shards from {input_dir}...")
    for f_path in files:
        with open(f_path, 'rb') as f:
            shard = pickle.load(f)
            for key, arr in shard.items():
                master[key].append(arr)

    # Concatenate and cap
    result = {}
    for key, arrays in master.items():
        combined = np.concatenate(arrays, axis=0)
        if len(combined) > max_samples_per_key:
            indices = np.random.choice(len(combined), max_samples_per_key, replace=False)
            combined = combined[indices]
        result[key] = combined

    with open(output_file, 'wb') as f:
        pickle.dump(result, f)

    total_keys = len(result)
    total_samples = sum(len(v) for v in result.values())
    print(f"Master cGAN dataset saved to {output_file}")
    print(f"  {total_keys:,} unique contexts, {total_samples:,} total samples")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim cgan",
        description="Extract raw IPD/PW training samples from BAM files for cGAN "
                    "training, or merge multiple shards into a master training set.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- extract subcommand --
    p_train = sub.add_parser(
        "extract",
        help="Extract raw (IPD, PW) samples from a single BAM file",
        description="Collect individual IPD/PW observations per 11-mer + methylation "
                    "state. Outputs a *_cgan.pkl shard with raw sample arrays.",
    )
    p_train.add_argument("bam", help="Input BAM file with fi/fp kinetic tags")
    p_train.add_argument("motifs",
                         help="Motif source: KinSim string ('m6A,GATC,1,3551'), "
                              "PacBio motifs.csv, or REBASE file (auto-detected)")
    p_train.add_argument("output", help="Output .pkl file for the cGAN shard")
    p_train.add_argument("--max-samples", type=int, default=10_000,
                         help="Max samples per (kmer, meth_id) via reservoir sampling "
                              "(default: 10000)")
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
        help="Merge multiple *_cgan.pkl shards into one master training set",
        description="Concatenate raw sample arrays from multiple shards. "
                    "Looks for *_cgan.pkl files in the input directory.",
    )
    p_merge.add_argument("input_dir", help="Directory containing *_cgan.pkl shard files")
    p_merge.add_argument("output", help="Output master training set .pkl file")
    p_merge.add_argument("--max-samples", type=int, default=50_000,
                         help="Max samples per (kmer, meth_id) after merging "
                              "(default: 50000)")

    args = parser.parse_args(argv)

    if args.command == "merge":
        merge_shards(args.input_dir, args.output,
                     max_samples_per_key=args.max_samples)
    else:
        motif_string = load_motif_string(args.motifs,
                                         min_fraction=args.min_fraction,
                                         min_detected=args.min_detected)
        if not motif_string:
            print("ERROR: no motifs found from the provided source.", file=sys.stderr)
            sys.exit(1)
        print(f"Extracting cGAN samples from {os.path.basename(args.bam)}...")
        result = extract_samples_from_bam(args.bam, motif_string,
                                         max_samples_per_key=args.max_samples,
                                         revcomp=not args.no_revcomp)
        with open(args.output, 'wb') as f:
            pickle.dump(result, f)
        total_samples = sum(len(v) for v in result.values())
        print(f"cGAN shard saved to {args.output} "
              f"({len(result)} contexts, {total_samples:,} samples)")


if __name__ == "__main__":
    main()
