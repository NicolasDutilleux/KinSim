"""Inject IPD/PW kinetic signals into PBSIM3 simulated reads.

Uses a trained 11-mer dictionary + the PBSIM3 .maf alignment to resolve
reference context for edge bases (first/last 5 positions of each read).
Outputs an unaligned BAM with fi (IPD) and fp (PW) tags.
"""

import sys
import os
import gzip
import pickle
import array
import numpy as np
import pysam

from ..encoding import BASE_MAP, KMER_MASK, K, get_ipd_stats, get_pw_stats
from ..motifs import parse_motifs, scan_sequence

MID = K // 2  # 5


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------

def load_reference(ref_path):
    """Load a FASTA file (.ref or .fna) into {name: sequence} dict.

    Handles both plain and gzip-compressed files.
    """
    open_func = gzip.open if ref_path.endswith('.gz') else open
    seqs = {}
    current_name = None
    parts = []
    with open_func(ref_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name:
                    seqs[current_name] = ''.join(parts)
                current_name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.upper())
    if current_name:
        seqs[current_name] = ''.join(parts)
    return seqs


# ---------------------------------------------------------------------------
# MAF parser
# ---------------------------------------------------------------------------

def parse_maf(maf_path):
    """Parse PBSIM3 .maf.gz to extract read-to-reference mapping.

    Returns dict[read_name] -> (ref_name, ref_start, ref_strand, ref_src_size)

    MAF 's' line format:
      s name start size strand srcSize alignment
    Each alignment block has a reference line (1st 's') and read line (2nd 's').
    """
    mapping = {}
    open_func = gzip.open if maf_path.endswith('.gz') else open
    with open_func(maf_path, 'rt') as f:
        lines_in_block = []
        for line in f:
            line = line.strip()
            if line.startswith('a'):
                lines_in_block = []
            elif line.startswith('s'):
                lines_in_block.append(line)
                if len(lines_in_block) == 2:
                    # Parse reference line
                    ref_parts = lines_in_block[0].split()
                    ref_name = ref_parts[1]
                    ref_start = int(ref_parts[2])
                    ref_strand = ref_parts[4]
                    ref_src_size = int(ref_parts[5])

                    # Parse read line
                    read_parts = lines_in_block[1].split()
                    read_name = read_parts[1]

                    mapping[read_name] = (ref_name, ref_start, ref_strand, ref_src_size)
    return mapping


# ---------------------------------------------------------------------------
# Reference context extraction
# ---------------------------------------------------------------------------

def get_extended_context(ref_seq, ref_start, read_len, circular=True):
    """Get the reference sequence context for a read, extended by MID on each side.

    For edge bases of a read, we need reference context beyond the read boundaries.
    For circular genomes (bacteria), wraps around. Otherwise pads with 'N'.

    Returns a string of length (read_len + 2*MID) representing the reference context
    from (ref_start - MID) to (ref_start + read_len + MID).
    """
    ref_len = len(ref_seq)
    start = ref_start - MID
    end = ref_start + read_len + MID

    if circular and ref_len > 0:
        # Circular extraction: wrap around
        context = []
        for i in range(start, end):
            context.append(ref_seq[i % ref_len])
        return ''.join(context)
    else:
        # Linear: pad with N
        context = []
        for i in range(start, end):
            if 0 <= i < ref_len:
                context.append(ref_seq[i])
            else:
                context.append('N')
        return ''.join(context)


# ---------------------------------------------------------------------------
# Signal sampling
# ---------------------------------------------------------------------------

def sample_signal(mu, sigma):
    """Sample a non-negative kinetic value from N(mu, sigma), clamped to [0, 255]."""
    val = max(0, np.random.normal(mu, sigma))
    return min(int(round(val)), 255)


# ---------------------------------------------------------------------------
# Main injection
# ---------------------------------------------------------------------------

def inject_signals(fastq_path, maf_path, ref_path, pkl_path,
                   motif_string, output_bam, circular=True):
    """Inject IPD/PW signals into PBSIM3 reads.

    Pipeline:
      1. Load reference genome
      2. Load trained dictionary
      3. Parse .maf alignment mapping
      4. For each read in .fq.gz:
         a. Get reference context (extended by 5bp each side) via .maf
         b. Scan motifs on the extended reference context
         c. Encode 11-mers and sample IPD/PW from dictionary
      5. Write unaligned BAM with fi/fp tags
    """
    print(f"Loading reference: {ref_path}")
    ref_seqs = load_reference(ref_path)

    print(f"Loading dictionary: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        lookup = pickle.load(f)

    print(f"Parsing MAF: {maf_path}")
    maf_mapping = parse_maf(maf_path)

    motifs = parse_motifs(motif_string)

    # Fallback stats for unknown kmers (global mean of unmethylated entries)
    default_acc = np.zeros(5, dtype=np.float64)

    print(f"Injecting signals into reads from {fastq_path}...")
    n_reads = 0
    n_mapped = 0
    n_unmapped = 0

    # Write unaligned BAM
    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'unknown'}
    })

    open_func = gzip.open if fastq_path.endswith('.gz') else open
    with pysam.AlignmentFile(output_bam, "wb", header=header) as bam_out, \
         open_func(fastq_path, 'rt') as fq:

        while True:
            hdr_line = fq.readline()
            if not hdr_line:
                break
            seq_line = fq.readline()
            fq.readline()  # +
            qual_line = fq.readline()

            read_name = hdr_line.strip()[1:].split()[0]
            seq = seq_line.strip()
            qual_str = qual_line.strip()
            read_len = len(seq)
            n_reads += 1

            # Look up reference mapping from MAF
            maf_info = maf_mapping.get(read_name)

            if maf_info and maf_info[0] in ref_seqs:
                ref_name, ref_start, ref_strand, ref_src_size = maf_info
                ref_seq = ref_seqs[ref_name]

                # Get extended context: MID extra bases on each side
                ext_context = get_extended_context(ref_seq, ref_start, read_len, circular)

                # Scan motifs on the extended context
                meth_status = scan_sequence(ext_context, motifs)

                # Encode 11-mers from the extended context and sample signals
                ipd_vals = []
                pw_vals = []
                current_kmer = 0

                for i in range(len(ext_context)):
                    base_val = BASE_MAP.get(ext_context[i], 0)
                    current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                    if i >= K - 1:
                        # Position in read = i - MID - (K-1-MID) = i - (K-1)
                        read_pos = i - (K - 1)
                        if 0 <= read_pos < read_len:
                            center = i - MID
                            key = (current_kmer, int(meth_status[center]))
                            acc = lookup.get(key, default_acc)

                            mu_ipd, sig_ipd = get_ipd_stats(acc)
                            mu_pw, sig_pw = get_pw_stats(acc)

                            ipd_vals.append(sample_signal(mu_ipd, sig_ipd))
                            pw_vals.append(sample_signal(mu_pw, sig_pw))

                n_mapped += 1
            else:
                # No MAF info: fall back to read-only context (edge bases get defaults)
                meth_status = scan_sequence(seq, motifs)
                ipd_vals = []
                pw_vals = []
                current_kmer = 0

                for i in range(read_len):
                    base_val = BASE_MAP.get(seq[i], 0)
                    current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                    if i < K - 1:
                        # Not enough context yet — use defaults
                        ipd_vals.append(sample_signal(1.0, 0.1))
                        pw_vals.append(sample_signal(1.0, 0.1))
                    else:
                        center = i - MID
                        key = (current_kmer, int(meth_status[center]))
                        acc = lookup.get(key, default_acc)

                        mu_ipd, sig_ipd = get_ipd_stats(acc)
                        mu_pw, sig_pw = get_pw_stats(acc)

                        ipd_vals.append(sample_signal(mu_ipd, sig_ipd))
                        pw_vals.append(sample_signal(mu_pw, sig_pw))

                n_unmapped += 1

            # Build pysam AlignedSegment
            seg = pysam.AlignedSegment(header)
            seg.query_name = read_name
            seg.flag = 4  # unmapped
            seg.query_sequence = seq
            seg.query_qualities = pysam.qualitystring_to_array(qual_str)
            seg.set_tag('fi', array.array('B', ipd_vals), 'B')
            seg.set_tag('fp', array.array('B', pw_vals), 'B')
            bam_out.write(seg)

    print(f"Done. {n_reads} reads processed ({n_mapped} with ref context, {n_unmapped} without).")
    print(f"Output: {output_bam}")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim dictionary inject",
        description="Inject IPD/PW kinetic signals into PBSIM3 simulated reads. "
                    "Uses a trained 11-mer dictionary and the .maf alignment to resolve "
                    "reference context for edge bases. Outputs an unaligned BAM with "
                    "fi (IPD) and fp (PW) tags.",
    )
    parser.add_argument("fastq", help="PBSIM3 simulated reads (.fq or .fq.gz)")
    parser.add_argument("maf", help="PBSIM3 alignment file (.maf or .maf.gz)")
    parser.add_argument("ref", help="Reference genome FASTA (.fna, .fa, or .gz)")
    parser.add_argument("pkl", help="Trained kinetic dictionary (.pkl)")
    parser.add_argument("motifs", help="Motif string: 'm6A,GATC,2;m4C,CCWGG,1'")
    parser.add_argument("output", help="Output unaligned BAM file")
    parser.add_argument("--linear", action="store_true",
                        help="Treat genome as linear (default: circular wrapping for bacteria)")
    args = parser.parse_args(argv)
    inject_signals(
        fastq_path=args.fastq,
        maf_path=args.maf,
        ref_path=args.ref,
        pkl_path=args.pkl,
        motif_string=args.motifs,
        output_bam=args.output,
        circular=not args.linear,
    )


if __name__ == "__main__":
    main()
