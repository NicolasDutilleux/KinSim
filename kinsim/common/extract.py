"""Extract raw IPD/PW training samples from BAM files.

This is the shared data-preparation pipeline used by ALL neural KinSim modes
(MLP, cGAN, and future models).  It has no dependency on any specific model.

Data format
-----------
Each shard is a pickle file containing:

    dict[(kmer_id: int, meth_id: int)] -> np.ndarray(N, 2)

where columns are [IPD, PW] as raw float32 values read from the fi/fp BAM
tags.  Shards from multiple BAMs are combined in the merge step.

Why raw (not log-transformed)?
    The extract/merge pipeline stores raw values so that:
      - Shards can be inspected and plotted without model knowledge
      - Different models can apply their own transforms at load time
      - KmerSignalDataset (common/dataset.py) applies log_transform once

CLI
---
    kinsim data extract reads.bam "m6A,GATC,1" shard.pkl
    kinsim data merge   shards/   master_data.pkl
"""

import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pysam

from ..encoding import BASE_MAP, K, KMER_MASK
from ..motifs import load_motif_string, parse_motifs, scan_sequence


# ---------------------------------------------------------------------------
# Extract: raw samples from one BAM file
# ---------------------------------------------------------------------------

def extract_samples_from_bam(
    bam_path: str,
    motif_string: str,
    max_samples_per_key: int = 10_000,
    revcomp: bool = True,
) -> dict:
    """Extract raw (IPD, PW) pairs from a BAM file for each 11-mer context.

    For each read: extract sequence + fi/fp kinetic tags, scan methylation
    motifs, then slide an 11-mer window collecting raw signal values.

    Reservoir sampling keeps memory bounded: once a (kmer, meth_id) key
    reaches max_samples_per_key, new samples randomly replace existing ones
    with probability max_samples_per_key / n_seen, giving an unbiased sample.

    Args:
        bam_path:            Path to BAM file with fi/fp kinetic tags.
        motif_string:        Semicolon-delimited motif string (e.g. "m6A,GATC,2,3551").
        max_samples_per_key: Maximum samples stored per (kmer, meth_id) key.
        revcomp:             Include reverse complement motif patterns (default True).

    Returns:
        dict[(kmer_id, meth_id)] -> np.ndarray(N, 2)  columns: [IPD, PW]
    """
    mid    = K // 2   # Centre position of the 11-mer window (= 5)
    motifs = parse_motifs(motif_string, revcomp=revcomp)

    samples: dict = defaultdict(list)
    counts:  dict = defaultdict(int)   # total observations seen per key

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            seq = read.query_sequence
            if not (seq and len(seq) >= K and read.has_tag("fi")):
                continue

            ipds    = read.get_tag("fi")
            pws     = read.get_tag("fp")
            min_len = min(len(seq), len(ipds), len(pws))

            # Per-read regex scan for methylation positions.
            # (fuzznuc is only used for reference-level pre-scanning in
            # generate.py; subprocess calls per-read would be too slow.)
            meth_status = scan_sequence(seq[:min_len], motifs)

            # Slide 11-mer window across the read
            current_kmer = 0
            for i in range(min_len):
                base_val     = BASE_MAP.get(seq[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                if i >= K - 1:
                    center   = i - mid
                    key      = (current_kmer, int(meth_status[center]))
                    ipd_val  = float(ipds[center])
                    pw_val   = float(pws[center])

                    counts[key] += 1
                    n = counts[key]

                    if n <= max_samples_per_key:
                        samples[key].append([ipd_val, pw_val])
                    else:
                        # Reservoir sampling: replace a random existing entry
                        j = np.random.randint(0, n)
                        if j < max_samples_per_key:
                            samples[key][j] = [ipd_val, pw_val]

    return {key: np.array(vals, dtype=np.float32) for key, vals in samples.items()}


# ---------------------------------------------------------------------------
# Merge: combine shards from multiple BAMs
# ---------------------------------------------------------------------------

def merge_shards(
    input_dir: str,
    output_file: str,
    max_samples_per_key: int = 50_000,
) -> None:
    """Merge multiple shard pickle files into one master training set.

    Looks for files matching ``*_cgan.pkl`` in input_dir (the naming
    convention is historical; the format is model-agnostic).

    After concatenation, keys exceeding max_samples_per_key are randomly
    subsampled to keep the master file manageable.

    Args:
        input_dir:           Directory containing ``*_cgan.pkl`` shard files.
        output_file:         Path for the merged output .pkl file.
        max_samples_per_key: Maximum samples to keep per (kmer, meth_id).
    """
    import glob

    pattern = os.path.join(input_dir, "*_cgan.pkl")
    files   = glob.glob(pattern)

    if not files:
        print(f"ERROR: no '*_cgan.pkl' files found in {input_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(files)} shards from {input_dir}...")
    master: dict = defaultdict(list)

    for f_path in files:
        with open(f_path, "rb") as f:
            shard = pickle.load(f)
        for key, arr in shard.items():
            master[key].append(arr)

    result = {}
    for key, arrays in master.items():
        combined = np.concatenate(arrays, axis=0)
        if len(combined) > max_samples_per_key:
            idx      = np.random.choice(len(combined), max_samples_per_key,
                                        replace=False)
            combined = combined[idx]
        result[key] = combined

    with open(output_file, "wb") as f:
        pickle.dump(result, f)

    total_keys    = len(result)
    total_samples = sum(len(v) for v in result.values())
    print(f"Master dataset saved to {output_file}")
    print(f"  {total_keys:,} unique contexts, {total_samples:,} total samples")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim data",
        description=(
            "Extract raw (IPD, PW) training samples from BAM files, or merge\n"
            "multiple shards into a master training set.\n\n"
            "The output is consumed by BOTH:\n"
            "  kinsim mlp  train   master_data.pkl  checkpoints_mlp/\n"
            "  kinsim cgan train   master_data.pkl  checkpoints_cgan/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- extract subcommand --
    p_extract = sub.add_parser(
        "extract",
        help="Extract raw (IPD, PW) samples from a single BAM file",
        description=(
            "Collect individual IPD/PW observations per (11-mer, methylation_state).\n"
            "Outputs a *_cgan.pkl shard with raw sample arrays.\n"
            "Run one job per BAM file, then merge all shards."
        ),
    )
    p_extract.add_argument("bam",
                           help="Input BAM file with fi/fp kinetic tags")
    p_extract.add_argument("motifs",
                           help="Motif source: KinSim string ('m6A,GATC,1,3551'), "
                                "PacBio motifs.csv, or REBASE file (auto-detected)")
    p_extract.add_argument("output",
                           help="Output .pkl shard file")
    p_extract.add_argument("--max-samples", type=int, default=10_000,
                           help="Max samples per (kmer, meth_id) via reservoir "
                                "sampling (default: 10000)")
    p_extract.add_argument("--no-revcomp", action="store_true",
                           help="Do not scan reverse complement strand for motifs")
    p_extract.add_argument("--min-fraction", type=float, default=0.40,
                           help="Minimum fraction threshold for PacBio CSV (default: 0.40)")
    p_extract.add_argument("--min-detected", type=int, default=20,
                           help="Minimum nDetected threshold for PacBio CSV (default: 20)")

    # -- merge subcommand --
    p_merge = sub.add_parser(
        "merge",
        help="Merge multiple *_cgan.pkl shards into one master training set",
        description=(
            "Concatenate raw sample arrays from all shards in a directory.\n"
            "Looks for *_cgan.pkl files.  Subsamples per key if needed."
        ),
    )
    p_merge.add_argument("input_dir",
                         help="Directory containing *_cgan.pkl shard files")
    p_merge.add_argument("output",
                         help="Output master training set .pkl file")
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

        print(f"Extracting samples from {os.path.basename(args.bam)}...")
        result = extract_samples_from_bam(
            args.bam, motif_string,
            max_samples_per_key=args.max_samples,
            revcomp=not args.no_revcomp,
        )

        with open(args.output, "wb") as f:
            pickle.dump(result, f)

        total_samples = sum(len(v) for v in result.values())
        print(f"Shard saved to {args.output} "
              f"({len(result):,} contexts, {total_samples:,} samples)")


if __name__ == "__main__":
    main()
