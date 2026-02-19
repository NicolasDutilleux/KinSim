"""Inject IPD/PW kinetic signals into PBSIM3 simulated reads.

Uses a trained 11-mer dictionary + the PBSIM3 .maf alignment to resolve
reference context for edge bases (first/last 5 positions of each read).
Outputs an unaligned BAM with fi (IPD) and fp (PW) tags.

Two calling modes (auto-detected):
  Directory mode:   kinsim dictionary inject <pbsim3_dir> <dict.pkl> <motifs> <output_dir>
  Per-genome mode:  kinsim dictionary inject <fq.gz> <maf.gz> <ref.fna> <dict.pkl> <motifs> <out.bam>

In directory mode, all .fq.gz files in pbsim3_dir are processed. Matching
.maf.gz and .fna files must share the same basename (e.g. genome1.fq.gz,
genome1.maf.gz, genome1.fna). Also accepts .fq/.maf/.fa/.fasta extensions.

Motif input (auto-detected in both modes):
  - KinSim motif string  — "m6A,GATC,1;m4C,CCWGG,1"
  - PacBio motifs.csv    — file path ending in .csv
  - REBASE file          — any other file path

Signal context design note:
  For edge bases (first/last 5 positions of each read), the polymerase
  experienced the full genomic 11-mer context even though those flanking
  bases are not part of the synthetic read.  We therefore use the .maf
  alignment to extend each read 5 bp into the reference on both sides,
  ensuring correct kinetic signal sampling for all positions.  Use
  --no-context to disable this and fall back to in-read context only
  (edge bases get default signals).
"""

import sys
import os
import glob
import gzip
import pickle
import array
import numpy as np
import pysam

from ..encoding import BASE_MAP, KMER_MASK, K, get_ipd_stats, get_pw_stats
from ..motifs import load_motif_string, parse_motifs, scan_sequence, build_reference_meth_map

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
                    ref_parts  = lines_in_block[0].split()
                    ref_name   = ref_parts[1]
                    ref_start  = int(ref_parts[2])
                    ref_strand = ref_parts[4]
                    ref_src_size = int(ref_parts[5])

                    read_parts = lines_in_block[1].split()
                    read_name  = read_parts[1]

                    mapping[read_name] = (ref_name, ref_start, ref_strand, ref_src_size)
    return mapping


# ---------------------------------------------------------------------------
# Reference context extraction (for 11-mer kmer building at edge bases)
# ---------------------------------------------------------------------------

def get_extended_context(ref_seq, ref_start, read_len, circular=True):
    """Get the reference sequence context for a read, extended by MID on each side.

    Returns a string of length (read_len + 2*MID) representing the reference
    context from (ref_start - MID) to (ref_start + read_len + MID).

    This extended context is used only for 11-mer kmer encoding at edge bases.
    Methylation status is looked up from the pre-computed reference meth_map.
    """
    ref_len = len(ref_seq)
    start = ref_start - MID
    end   = ref_start + read_len + MID

    if circular and ref_len > 0:
        return ''.join(ref_seq[i % ref_len] for i in range(start, end))
    else:
        return ''.join(ref_seq[i] if 0 <= i < ref_len else 'N'
                       for i in range(start, end))


# ---------------------------------------------------------------------------
# Signal sampling
# ---------------------------------------------------------------------------

def sample_signal(mu, sigma):
    """Sample a non-negative kinetic value from N(mu, sigma), clamped to [0, 255]."""
    val = max(0, np.random.normal(mu, sigma))
    return min(int(round(val)), 255)


# ---------------------------------------------------------------------------
# Main injection (single genome)
# ---------------------------------------------------------------------------

def inject_signals(fastq_path, maf_path, ref_path, pkl_path,
                   motif_string, output_bam,
                   circular=True, revcomp=True, no_fuzznuc=False):
    """Inject IPD/PW signals into PBSIM3 reads for a single genome.

    Pipeline:
      1. Load reference genome
      2. Pre-scan reference for methylation sites (fuzznuc primary, regex fallback)
      3. Load trained dictionary
      4. Parse .maf alignment mapping
      5. For each read in .fq.gz:
         a. Get extended reference context via .maf (for kmer building at edges)
         b. Look up methylation status from pre-computed reference map
         c. Encode 11-mers and sample IPD/PW from dictionary
      6. Write unaligned BAM with fi/fp tags

    Args:
        circular:   Treat genome as circular (default True for bacteria).
        revcomp:    Scan reverse complement strand for motifs (default True).
        no_fuzznuc: Force Python regex for reference scanning; skip fuzznuc.
                    By default fuzznuc is tried first, falling back to regex
                    automatically if EMBOSS is not installed.
    """
    print(f"Loading reference: {ref_path}")
    ref_seqs = load_reference(ref_path)

    backend = "regex (forced)" if no_fuzznuc else "fuzznuc (primary, regex fallback)"
    print(f"Pre-scanning reference for methylation sites ({backend})...")
    meth_map = build_reference_meth_map(ref_seqs, motif_string,
                                        revcomp=revcomp,
                                        no_fuzznuc=no_fuzznuc)

    # Keep regex-parsed motifs for fallback (unmapped reads)
    fallback_motifs = parse_motifs(motif_string, revcomp=revcomp)

    print(f"Loading dictionary: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        lookup = pickle.load(f)

    print(f"Parsing MAF: {maf_path}")
    maf_mapping = parse_maf(maf_path)

    default_acc = np.zeros(5, dtype=np.float64)

    print(f"Injecting signals into reads from {fastq_path}...")
    n_reads    = 0
    n_mapped   = 0
    n_unmapped = 0

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
            seq_line  = fq.readline()
            fq.readline()  # +
            qual_line = fq.readline()

            read_name = hdr_line.strip()[1:].split()[0]
            seq       = seq_line.strip()
            qual_str  = qual_line.strip()
            read_len  = len(seq)
            n_reads  += 1

            maf_info = maf_mapping.get(read_name)

            if maf_info and maf_info[0] in ref_seqs:
                # --- Mapped read: use extended reference context ---
                ref_name, ref_start, ref_strand, ref_src_size = maf_info
                ref_seq  = ref_seqs[ref_name]
                ref_len  = len(ref_seq)
                ref_meth = meth_map[ref_name]

                ext_context = get_extended_context(ref_seq, ref_start,
                                                   read_len, circular)

                ipd_vals = []
                pw_vals  = []
                current_kmer = 0

                for i in range(len(ext_context)):
                    base_val = BASE_MAP.get(ext_context[i], 0)
                    current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                    if i >= K - 1:
                        read_pos = i - (K - 1)
                        if 0 <= read_pos < read_len:
                            # Check for N in the 11-mer window
                            context_window = ext_context[i - (K - 1): i + 1]
                            if 'N' in context_window:
                                ipd_vals.append(sample_signal(1.0, 0.1))
                                pw_vals.append(sample_signal(1.0, 0.1))
                            else:
                                # Methylation status from pre-computed map
                                # center in ext_context = ref_start + read_pos (on reference)
                                ref_pos = ref_start + read_pos
                                if circular:
                                    meth_id = int(ref_meth[ref_pos % ref_len])
                                elif 0 <= ref_pos < ref_len:
                                    meth_id = int(ref_meth[ref_pos])
                                else:
                                    meth_id = 0

                                key = (current_kmer, meth_id)
                                acc = lookup.get(key, default_acc)

                                mu_ipd, sig_ipd = get_ipd_stats(acc)
                                mu_pw,  sig_pw  = get_pw_stats(acc)

                                ipd_vals.append(sample_signal(mu_ipd, sig_ipd))
                                pw_vals.append(sample_signal(mu_pw,  sig_pw))

                n_mapped += 1

            else:
                # --- Unmapped read: fall back to read-only context ---
                # Per-read regex scanning (fuzznuc is only used for the reference
                # pre-scan above; subprocess calls per read would be prohibitively slow)
                meth_status = scan_sequence(seq, fallback_motifs)
                ipd_vals = []
                pw_vals  = []
                current_kmer = 0

                for i in range(read_len):
                    base_val = BASE_MAP.get(seq[i], 0)
                    current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                    if i < K - 1:
                        ipd_vals.append(sample_signal(1.0, 0.1))
                        pw_vals.append(sample_signal(1.0, 0.1))
                    else:
                        center = i - MID
                        key = (current_kmer, int(meth_status[center]))
                        acc = lookup.get(key, default_acc)

                        mu_ipd, sig_ipd = get_ipd_stats(acc)
                        mu_pw,  sig_pw  = get_pw_stats(acc)

                        ipd_vals.append(sample_signal(mu_ipd, sig_ipd))
                        pw_vals.append(sample_signal(mu_pw,  sig_pw))

                n_unmapped += 1

            seg = pysam.AlignedSegment(header)
            seg.query_name     = read_name
            seg.flag           = 4  # unmapped
            seg.query_sequence = seq
            seg.query_qualities = pysam.qualitystring_to_array(qual_str)
            seg.set_tag('fi', array.array('B', ipd_vals), 'B')
            seg.set_tag('fp', array.array('B', pw_vals),  'B')
            bam_out.write(seg)

    print(f"Done. {n_reads} reads processed "
          f"({n_mapped} with ref context, {n_unmapped} without).")
    print(f"Output: {output_bam}")


# ---------------------------------------------------------------------------
# Directory mode helpers
# ---------------------------------------------------------------------------

def _find_pbsim3_files(pbsim3_dir):
    """Discover all PBSIM3 genome sets in a directory.

    Each genome set consists of three files sharing the same basename:
      <basename>.fq.gz   (or .fq)    — simulated reads
      <basename>.maf.gz  (or .maf)   — read-to-reference alignment
      <basename>.fna     (or .fa, .fasta) — reference genome

    Returns a sorted list of (fq_path, maf_path, ref_path, basename) tuples.
    Skips any genome where .maf or .fna cannot be found (prints a warning).
    """
    fq_files = sorted(glob.glob(os.path.join(pbsim3_dir, '*.fq.gz')))
    if not fq_files:
        fq_files = sorted(glob.glob(os.path.join(pbsim3_dir, '*.fq')))

    results = []
    for fq_path in fq_files:
        basename = os.path.basename(fq_path)
        stem = basename[:-len('.fq.gz')] if basename.endswith('.fq.gz') else basename[:-len('.fq')]

        # Find .maf.gz or .maf
        maf_path = os.path.join(pbsim3_dir, stem + '.maf.gz')
        if not os.path.isfile(maf_path):
            maf_path = os.path.join(pbsim3_dir, stem + '.maf')
        if not os.path.isfile(maf_path):
            print(f"  WARN: no .maf.gz/.maf for '{stem}' — skipping", file=sys.stderr)
            continue

        # Find .fna, .fa, or .fasta
        ref_path = None
        for ext in ('.fna', '.fa', '.fasta'):
            candidate = os.path.join(pbsim3_dir, stem + ext)
            if os.path.isfile(candidate):
                ref_path = candidate
                break
        if ref_path is None:
            print(f"  WARN: no .fna/.fa/.fasta for '{stem}' — skipping", file=sys.stderr)
            continue

        results.append((fq_path, maf_path, ref_path, stem))

    return results


def inject_directory(pbsim3_dir, pkl_path, motif_source, output_dir,
                     circular=True, revcomp=True, no_fuzznuc=False,
                     min_fraction=0.40, min_detected=20):
    """Inject signals into all genomes found in pbsim3_dir.

    Discovers all .fq.gz files and matches them with .maf.gz and .fna by
    basename. Applies the same motif source to every genome. Output BAMs
    are written to output_dir as <basename>_kinsim.bam.
    """
    genomes = _find_pbsim3_files(pbsim3_dir)
    if not genomes:
        print(f"ERROR: No .fq.gz files with matching .maf.gz/.fna found in {pbsim3_dir}",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    motif_string = load_motif_string(motif_source,
                                     min_fraction=min_fraction,
                                     min_detected=min_detected)
    if not motif_string:
        print("ERROR: no motifs found from the provided source.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(genomes)} genome(s) in {pbsim3_dir}")

    for fq_path, maf_path, ref_path, stem in genomes:
        out_bam = os.path.join(output_dir, stem + '_kinsim.bam')
        print(f"\n--- {stem} ---")
        inject_signals(fq_path, maf_path, ref_path, pkl_path, motif_string, out_bam,
                       circular=circular, revcomp=revcomp, no_fuzznuc=no_fuzznuc)

    print(f"\nAll done. {len(genomes)} BAM(s) written to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    if argv is None:
        argv = sys.argv[1:]

    # Auto-detect mode: directory (4 positional args) vs per-genome (6 positional args)
    if argv and os.path.isdir(argv[0]):
        _main_directory(argv)
    else:
        _main_per_genome(argv)


def _main_directory(argv):
    """CLI for directory mode: processes all genomes in pbsim3_dir."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim dictionary inject",
        description=(
            "Inject IPD/PW kinetic signals into all PBSIM3 genomes in a directory.\n\n"
            "Discovers all .fq.gz files in pbsim3_dir and matches them with\n"
            ".maf.gz and .fna by basename. Outputs <basename>_kinsim.bam files.\n\n"
            "Per-genome mode (single genome):\n"
            "  kinsim dictionary inject <fq.gz> <maf.gz> <ref.fna> <dict.pkl> <motifs> <out.bam>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pbsim3_dir",
                        help="Directory containing .fq.gz, .maf.gz, and .fna files "
                             "(same basename per genome)")
    parser.add_argument("dict",
                        help="Trained kinetic dictionary (.pkl)")
    parser.add_argument("motifs",
                        help="Motif source: KinSim string ('m6A,GATC,1'), "
                             "PacBio motifs.csv, or REBASE file (applied to all genomes)")
    parser.add_argument("output_dir",
                        help="Output directory for injected BAM files")
    parser.add_argument("--linear", action="store_true",
                        help="Treat genomes as linear (default: circular for bacteria)")
    parser.add_argument("--no-revcomp", action="store_true",
                        help="Do not scan reverse complement strand for motifs")
    parser.add_argument("--no-fuzznuc", action="store_true",
                        help="Force Python regex for reference methylation scanning "
                             "(by default fuzznuc is tried first, regex as fallback)")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold (PacBio CSV only, default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (PacBio CSV only, default: 20)")
    args = parser.parse_args(argv)

    inject_directory(
        pbsim3_dir=args.pbsim3_dir,
        pkl_path=args.dict,
        motif_source=args.motifs,
        output_dir=args.output_dir,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        no_fuzznuc=args.no_fuzznuc,
        min_fraction=args.min_fraction,
        min_detected=args.min_detected,
    )


def _main_per_genome(argv):
    """CLI for per-genome mode: processes a single .fq.gz file."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim dictionary inject",
        description=(
            "Inject IPD/PW kinetic signals into a single PBSIM3 genome.\n\n"
            "Uses a trained 11-mer dictionary and the .maf alignment to resolve\n"
            "reference context for edge bases.  The reference is pre-scanned once\n"
            "for methylation sites; subsequent per-read lookups are O(1).\n\n"
            "Directory mode (all genomes at once):\n"
            "  kinsim dictionary inject <pbsim3_dir> <dict.pkl> <motifs> <output_dir>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fastq",  help="PBSIM3 simulated reads (.fq or .fq.gz)")
    parser.add_argument("maf",    help="PBSIM3 alignment file (.maf or .maf.gz)")
    parser.add_argument("ref",    help="Reference genome FASTA (.fna, .fa, or .gz)")
    parser.add_argument("pkl",    help="Trained kinetic dictionary (.pkl)")
    parser.add_argument("motifs",
                        help="Motif source: KinSim string ('m6A,GATC,1'), "
                             "PacBio motifs.csv, or REBASE file (auto-detected)")
    parser.add_argument("output", help="Output unaligned BAM file")
    parser.add_argument("--linear", action="store_true",
                        help="Treat genome as linear (default: circular for bacteria)")
    parser.add_argument("--no-revcomp", action="store_true",
                        help="Do not scan reverse complement strand for motifs "
                             "(use when motif source already includes both orientations)")
    parser.add_argument("--no-fuzznuc", action="store_true",
                        help="Force Python regex for reference methylation scanning. "
                             "By default, EMBOSS fuzznuc is tried first as the primary "
                             "backend and falls back to regex automatically if fuzznuc "
                             "is not installed.")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold (PacBio CSV only, default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (PacBio CSV only, default: 20)")
    args = parser.parse_args(argv)

    motif_string = load_motif_string(args.motifs,
                                     min_fraction=args.min_fraction,
                                     min_detected=args.min_detected)
    if not motif_string:
        print("ERROR: no motifs found from the provided source.", file=sys.stderr)
        sys.exit(1)

    inject_signals(
        fastq_path=args.fastq,
        maf_path=args.maf,
        ref_path=args.ref,
        pkl_path=args.pkl,
        motif_string=motif_string,
        output_bam=args.output,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        no_fuzznuc=args.no_fuzznuc,
    )


if __name__ == "__main__":
    main()
