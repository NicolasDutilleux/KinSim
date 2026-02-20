"""Generate kinetic signals for PBSIM3 reads using trained conditional GAN.

Mirrors dictionary/inject.py but uses the Generator for signal synthesis.
Batches generation across multiple reads for GPU efficiency.

Two calling modes (auto-detected):
  Directory mode:   kinsim cgan generate <pbsim3_dir> <checkpoint.pt> <motifs> <output_dir>
  Per-genome mode:  kinsim cgan generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>

Directory mode supports the same two layouts as dictionary inject (auto-detected):
  - Species subdirectories: pbsim3_dir/Ecoli/, pbsim3_dir/Salmonella/, ...
  - Flat layout: all files directly in pbsim3_dir, matched by basename.

Motif input (auto-detected):
  - KinSim motif string       — "m6A,GATC,1;m4C,CCWGG,1"  (applied to all species)
  - Per-species mapping file  — text file with "species_name|motif_string" per line
  - PacBio motifs.csv         — file path ending in .csv
  - REBASE file               — any other file path

The reference genome is pre-scanned once for methylation sites using EMBOSS
fuzznuc as the primary backend (falls back to Python regex automatically if
fuzznuc is not installed).  Results are cached in O(1)-lookup arrays,
avoiding repeated per-read scanning.
"""

import os
import sys
import json
import gzip
import array
import torch
import pysam
import numpy as np

from .model import Generator, inv_log_transform
from ..encoding import BASE_MAP, KMER_MASK, K
from ..motifs import (load_motif_string, parse_motifs, scan_sequence,
                      build_reference_meth_map)

# Reuse from dictionary.inject (no code duplication)
from ..dictionary.inject import (load_reference, parse_maf, get_extended_context, MID,
                                  _find_pbsim3_files, _resolve_motifs_for_species)


# ---------------------------------------------------------------------------
# Batched GAN inference
# ---------------------------------------------------------------------------

def generate_signals_batch(G, kmer_ids, meth_ids, device, noise_dim=32):
    """Generate IPD/PW signals for a batch of contexts using the Generator.

    Args:
        G:         Trained Generator model
        kmer_ids:  List of kmer integer IDs
        meth_ids:  List of methylation IDs
        device:    torch device
        noise_dim: Noise vector dimension (must match training)

    Returns:
        np.ndarray of shape (N, 2) with [IPD, PW] in log1p space
    """
    n = len(kmer_ids)
    kmer_tensor = torch.tensor(kmer_ids, dtype=torch.long,  device=device)
    meth_tensor = torch.tensor(meth_ids, dtype=torch.long,  device=device)
    z = torch.randn(n, noise_dim, device=device)

    with torch.no_grad():
        fake_log = G(z, kmer_tensor, meth_tensor)
        fake     = inv_log_transform(fake_log)

    return fake.cpu().numpy()


# ---------------------------------------------------------------------------
# Main injection
# ---------------------------------------------------------------------------

def generate_signals(
    fastq_path, maf_path, ref_path, checkpoint_path,
    motif_string, output_bam,
    circular=True, revcomp=True,
    device='cuda', batch_reads=1000,
    no_fuzznuc=False,
):
    """Inject GAN-generated IPD/PW signals into PBSIM3 reads.

    Pipeline:
      1. Load reference genome
      2. Pre-scan reference for methylation sites (fuzznuc primary, regex fallback)
      3. Load trained Generator from checkpoint
      4. Parse .maf alignment mapping
      5. For batches of reads in .fq.gz:
         a. Collect all (kmer_id, meth_id) contexts for the batch
            using pre-computed reference methylation map
         b. Generate signals in one batched forward pass
         c. Write unaligned BAM with fi/fp tags

    Args:
        circular:   Treat genome as circular (default True for bacteria).
        revcomp:    Scan reverse complement strand for motifs (default True).
        no_fuzznuc: Force Python regex for reference scanning; skip fuzznuc.
                    By default fuzznuc is tried first, falling back to regex
                    automatically if EMBOSS is not installed.
        batch_reads: Number of reads to batch for GPU inference (default 1000).
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading reference: {ref_path}")
    ref_seqs = load_reference(ref_path)

    backend = "regex (forced)" if no_fuzznuc else "fuzznuc (primary, regex fallback)"
    print(f"Pre-scanning reference for methylation sites ({backend})...")
    meth_map = build_reference_meth_map(ref_seqs, motif_string,
                                        revcomp=revcomp,
                                        no_fuzznuc=no_fuzznuc)

    # Keep regex motifs for fallback (unmapped reads)
    fallback_motifs = parse_motifs(motif_string, revcomp=revcomp)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        noise_dim      = config.get('noise_dim', 32)
        kmer_embed_dim = config.get('kmer_embed_dim', 64)
    else:
        print("  Warning: model_config.json not found, using defaults "
              "(noise_dim=32, kmer_embed_dim=64)")
        noise_dim, kmer_embed_dim = 32, 64

    G = Generator(noise_dim=noise_dim, kmer_embed_dim=kmer_embed_dim).to(device)
    G.load_state_dict(ckpt['generator'])
    G.eval()
    print(f"  Generator loaded (noise_dim={noise_dim}, kmer_embed_dim={kmer_embed_dim})")

    print(f"Parsing MAF: {maf_path}")
    maf_mapping = parse_maf(maf_path)

    print(f"Generating signals for reads from {fastq_path}...")
    n_reads   = 0
    n_mapped  = 0
    n_unmapped = 0

    header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'unknown'}
    })

    open_func = gzip.open if fastq_path.endswith('.gz') else open

    with pysam.AlignmentFile(output_bam, "wb", header=header) as bam_out, \
         open_func(fastq_path, 'rt') as fq:

        batch = []

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

            batch.append({'name': read_name, 'seq': seq,
                          'qual': qual_str, 'len': read_len})

            if len(batch) >= batch_reads:
                n_mapped_batch, n_unmapped_batch = _process_batch(
                    batch, ref_seqs, maf_mapping, meth_map, fallback_motifs,
                    G, device, noise_dim, circular, bam_out, header)
                n_mapped   += n_mapped_batch
                n_unmapped += n_unmapped_batch
                batch = []

        if batch:
            n_mapped_batch, n_unmapped_batch = _process_batch(
                batch, ref_seqs, maf_mapping, meth_map, fallback_motifs,
                G, device, noise_dim, circular, bam_out, header)
            n_mapped   += n_mapped_batch
            n_unmapped += n_unmapped_batch

    print(f"Done. {n_reads} reads processed "
          f"({n_mapped} with ref context, {n_unmapped} without).")
    print(f"Output: {output_bam}")


def _process_batch(batch, ref_seqs, maf_mapping, meth_map, fallback_motifs,
                   G, device, noise_dim, circular, bam_out, header):
    """Process a batch of reads with batched GAN inference.

    Returns (n_mapped, n_unmapped) counts for the batch.
    """
    all_kmer_ids = []
    all_meth_ids = []
    is_n_context = []   # per-position flag for N-context fallback
    read_offsets = [0]

    n_mapped = n_unmapped = 0

    for read_data in batch:
        read_name = read_data['name']
        seq       = read_data['seq']
        read_len  = read_data['len']

        maf_info = maf_mapping.get(read_name)

        if maf_info and maf_info[0] in ref_seqs:
            ref_name, ref_start, ref_strand, ref_src_size = maf_info
            ref_seq  = ref_seqs[ref_name]
            ref_len  = len(ref_seq)
            ref_meth = meth_map[ref_name]

            ext_context = get_extended_context(ref_seq, ref_start, read_len, circular)
            current_kmer = 0

            for i in range(len(ext_context)):
                base_val = BASE_MAP.get(ext_context[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                if i >= K - 1:
                    read_pos = i - (K - 1)
                    if 0 <= read_pos < read_len:
                        context_window = ext_context[i - (K - 1): i + 1]
                        has_n = 'N' in context_window
                        is_n_context.append(has_n)

                        if has_n:
                            all_kmer_ids.append(0)
                            all_meth_ids.append(0)
                        else:
                            ref_pos = ref_start + read_pos
                            if circular:
                                meth_id = int(ref_meth[ref_pos % ref_len])
                            elif 0 <= ref_pos < ref_len:
                                meth_id = int(ref_meth[ref_pos])
                            else:
                                meth_id = 0
                            all_kmer_ids.append(current_kmer)
                            all_meth_ids.append(meth_id)

            n_mapped += 1

        else:
            # Fallback: read-only context.
            # Per-read regex scanning (fuzznuc is only used for the reference
            # pre-scan above; subprocess calls per read would be prohibitively slow)
            meth_status  = scan_sequence(seq, fallback_motifs)
            current_kmer = 0

            for i in range(read_len):
                base_val = BASE_MAP.get(seq[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                if i < K - 1:
                    is_n_context.append(True)  # treat as N → use default
                    all_kmer_ids.append(0)
                    all_meth_ids.append(0)
                else:
                    is_n_context.append(False)
                    center = i - MID
                    all_kmer_ids.append(current_kmer)
                    all_meth_ids.append(int(meth_status[center]))

            n_unmapped += 1

        read_offsets.append(len(all_kmer_ids))

    # Batched GAN inference
    if len(all_kmer_ids) > 0:
        all_signals = generate_signals_batch(G, all_kmer_ids, all_meth_ids,
                                             device, noise_dim)
    else:
        all_signals = np.zeros((0, 2), dtype=np.float32)

    # Split signals back to reads and write to BAM
    for idx, read_data in enumerate(batch):
        start   = read_offsets[idx]
        end     = read_offsets[idx + 1]
        signals = all_signals[start:end]
        is_n    = is_n_context[start:end]

        ipd_vals = np.clip(signals[:, 0], 0, 255).astype(np.uint8)
        pw_vals  = np.clip(signals[:, 1], 0, 255).astype(np.uint8)

        # Replace N-context positions with default signals
        for pos_idx, n_flag in enumerate(is_n):
            if n_flag:
                ipd_vals[pos_idx] = 1
                pw_vals[pos_idx]  = 1

        seg = pysam.AlignedSegment(header)
        seg.query_name      = read_data['name']
        seg.flag            = 4  # unmapped
        seg.query_sequence  = read_data['seq']
        seg.query_qualities = pysam.qualitystring_to_array(read_data['qual'])
        seg.set_tag('fi', array.array('B', ipd_vals.tolist()), 'B')
        seg.set_tag('fp', array.array('B', pw_vals.tolist()),  'B')
        bam_out.write(seg)

    return n_mapped, n_unmapped


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------

def generate_directory(pbsim3_dir, checkpoint_path, motif_source, output_dir,
                       circular=True, revcomp=True,
                       device='cuda', batch_reads=1000,
                       no_fuzznuc=False,
                       min_fraction=0.40, min_detected=20):
    """Generate signals for all species found under pbsim3_dir.

    Supports the same two directory layouts as dictionary inject (auto-detected):
      - Species subdirectories: pbsim3_dir/Ecoli/, pbsim3_dir/Salmonella/, ...
      - Flat layout: all files directly in pbsim3_dir, matched by basename.

    Output BAMs are written to output_dir as <species_name>_cgan.bam.
    """
    genomes = _find_pbsim3_files(pbsim3_dir)
    if not genomes:
        print(f"ERROR: No genome sets found in {pbsim3_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(genomes)} species in {pbsim3_dir}")

    for fq_path, maf_path, ref_path, species in genomes:
        motif_string = _resolve_motifs_for_species(motif_source, species,
                                                   min_fraction, min_detected)
        if not motif_string:
            print(f"ERROR: no motifs found for species '{species}'.", file=sys.stderr)
            sys.exit(1)

        out_bam = os.path.join(output_dir, species + '_cgan.bam')
        print(f"\n--- {species} ---")
        generate_signals(fq_path, maf_path, ref_path, checkpoint_path, motif_string, out_bam,
                         circular=circular, revcomp=revcomp,
                         device=device, batch_reads=batch_reads,
                         no_fuzznuc=no_fuzznuc)

    print(f"\nAll done. {len(genomes)} BAM(s) written to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if argv and os.path.isdir(argv[0]):
        _main_directory(argv)
    else:
        _main_per_genome(argv)


def _main_directory(argv):
    """CLI for directory mode: processes all species in pbsim3_dir."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim cgan generate",
        description=(
            "Generate GAN kinetic signals for all PBSIM3 species in a directory.\n\n"
            "Supports two directory layouts (auto-detected):\n\n"
            "  Species subdirectories (recommended):\n"
            "    pbsim3_dir/\n"
            "      Ecoli/          <- species name = subdir name\n"
            "        reads.fq.gz\n"
            "        reads.maf.gz\n"
            "        Ecoli.fna\n"
            "      Salmonella/\n"
            "        ...\n\n"
            "  Flat layout (files matched by basename):\n"
            "    pbsim3_dir/\n"
            "      Ecoli.fq.gz   Ecoli.maf.gz   Ecoli.fna\n"
            "      Salmonella.fq.gz ...\n\n"
            "Per-genome mode (single genome):\n"
            "  kinsim cgan generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pbsim3_dir",
                        help="Directory containing species subdirs or flat .fq.gz files")
    parser.add_argument("checkpoint", help="Trained GAN checkpoint (.pt)")
    parser.add_argument("motifs",
                        help="Motifs: KinSim string (applied to all), PacBio .csv, "
                             "REBASE file, or per-species file ('species|motif_string' per line)")
    parser.add_argument("output_dir",
                        help="Output directory for generated BAM files")
    parser.add_argument("--linear", action="store_true",
                        help="Treat genomes as linear (default: circular for bacteria)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--batch-reads", type=int, default=1000,
                        help="Number of reads to batch for GPU inference (default: 1000)")
    parser.add_argument("--no-revcomp", action="store_true",
                        help="Do not scan reverse complement strand for motifs")
    parser.add_argument("--no-fuzznuc", action="store_true",
                        help="Force Python regex for reference methylation scanning")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum fraction threshold (PacBio CSV only, default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (PacBio CSV only, default: 20)")
    args = parser.parse_args(argv)

    generate_directory(
        pbsim3_dir=args.pbsim3_dir,
        checkpoint_path=args.checkpoint,
        motif_source=args.motifs,
        output_dir=args.output_dir,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        device=args.device,
        batch_reads=args.batch_reads,
        no_fuzznuc=args.no_fuzznuc,
        min_fraction=args.min_fraction,
        min_detected=args.min_detected,
    )


def _main_per_genome(argv):
    """CLI for per-genome mode: processes a single .fq.gz file."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim cgan generate",
        description=(
            "Generate kinetic signals for PBSIM3 reads using a trained conditional GAN.\n\n"
            "Uses the .maf alignment to resolve reference context for edge bases.\n"
            "The reference is pre-scanned once for methylation sites; per-read\n"
            "lookups are O(1).  Outputs an unaligned BAM with fi (IPD) and fp (PW) tags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fastq",      help="PBSIM3 simulated reads (.fq or .fq.gz)")
    parser.add_argument("maf",        help="PBSIM3 alignment file (.maf or .maf.gz)")
    parser.add_argument("ref",        help="Reference genome FASTA (.fna, .fa, or .gz)")
    parser.add_argument("checkpoint", help="Trained GAN checkpoint (.pt)")
    parser.add_argument("motifs",
                        help="Motif source: KinSim string ('m6A,GATC,1'), "
                             "PacBio motifs.csv, or REBASE file (auto-detected)")
    parser.add_argument("output",     help="Output unaligned BAM file")
    parser.add_argument("--linear", action="store_true",
                        help="Treat genome as linear (default: circular for bacteria)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device to use (default: cuda)")
    parser.add_argument("--batch-reads", type=int, default=1000,
                        help="Number of reads to batch for GPU inference (default: 1000)")
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

    generate_signals(
        fastq_path=args.fastq,
        maf_path=args.maf,
        ref_path=args.ref,
        checkpoint_path=args.checkpoint,
        motif_string=motif_string,
        output_bam=args.output,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        device=args.device,
        batch_reads=args.batch_reads,
        no_fuzznuc=args.no_fuzznuc,
    )


if __name__ == "__main__":
    main()
