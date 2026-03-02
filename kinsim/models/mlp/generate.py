"""Generate kinetic signals for PBSIM3 reads using a trained MLP predictor.

Mirrors cgan/generate.py but replaces the GAN Generator with MLPPredictor.
The key difference: no noise vector — the model predicts (μ, σ) per context,
then either samples from N(μ, σ²) (stochastic, default) or returns μ directly
(deterministic, via --deterministic flag).

Two calling modes (auto-detected):
  Directory mode:   kinsim mlp generate <pbsim3_dir> <checkpoint.pt> <motifs> <output_dir>
  Per-genome mode:  kinsim mlp generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>

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

Output BAMs use the suffix _mlp.bam and are structurally identical to cGAN
output: unaligned BAM (flag=4) with fi:B:C (IPD) and fp:B:C (PW) tags.
"""

import array
import gzip
import json
import os
import sys

import numpy as np
import pysam
import torch

from .model import MLPPredictor
from ...encoding import BASE_MAP, K, KMER_MASK
from ...motifs import (build_reference_meth_map, load_motif_string,
                       parse_motifs, scan_sequence)

# Reuse from dictionary.inject — no code duplication
from ...dictionary.inject import (MID, _find_pbsim3_files,
                                   _resolve_motifs_for_species,
                                   get_extended_context, load_reference,
                                   parse_maf)


# ---------------------------------------------------------------------------
# Batched MLP inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_signals_batch(
    model: MLPPredictor,
    kmer_ids: list,
    meth_ids: list,
    device: torch.device,
    deterministic: bool = False,
) -> np.ndarray:
    """Generate IPD/PW signals for a batch of contexts using MLPPredictor.

    Args:
        model:         Trained MLPPredictor in eval mode.
        kmer_ids:      List of kmer integer IDs (22-bit encoded 11-mers).
        meth_ids:      List of methylation IDs (0–3).
        device:        Torch device.
        deterministic: If True, return the predicted mean μ (no sampling).
                       If False, sample from N(μ, σ²) for biological realism.

    Returns:
        np.ndarray of shape (N, 2) with raw [IPD, PW] values in [0, 255].
    """
    kmer_tensor = torch.tensor(kmer_ids, dtype=torch.long, device=device)
    meth_tensor = torch.tensor(meth_ids, dtype=torch.long, device=device)

    if deterministic:
        signals = model.predict_mean(kmer_tensor, meth_tensor)
    else:
        signals = model.sample(kmer_tensor, meth_tensor)

    return signals.cpu().numpy()


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device) -> MLPPredictor:
    """Load MLPPredictor from a checkpoint file.

    Reads model_config.json from the same directory as the checkpoint to
    reconstruct the exact architecture used during training.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:          Torch device to load the model onto.

    Returns:
        MLPPredictor in eval mode, ready for inference.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # model_config.json is always written at the start of training (before the
    # first epoch), so it must exist alongside the checkpoint.  Using wrong
    # defaults here would silently produce a mismatched architecture.
    config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
    if not os.path.exists(config_path):
        print(
            f"ERROR: model_config.json not found in {os.path.dirname(checkpoint_path)}\n"
            "       This file is written by 'kinsim mlp train' at the start of training.\n"
            "       Ensure the checkpoint directory contains model_config.json.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)
    kmer_embed_dim = config["kmer_embed_dim"]
    hidden_dim     = config["hidden_dim"]

    model = MLPPredictor(kmer_embed_dim=kmer_embed_dim,
                         hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"  MLPPredictor loaded (kmer_embed_dim={kmer_embed_dim}, "
          f"hidden_dim={hidden_dim})")
    return model


# ---------------------------------------------------------------------------
# Main injection
# ---------------------------------------------------------------------------

def generate_signals(
    fastq_path: str,
    maf_path: str,
    ref_path: str,
    checkpoint_path: str,
    motif_string: str,
    output_bam: str,
    circular: bool = True,
    revcomp: bool = True,
    device: str = "cuda",
    batch_reads: int = 1000,
    no_fuzznuc: bool = False,
    deterministic: bool = False,
) -> None:
    """Inject MLP-predicted IPD/PW signals into PBSIM3 reads.

    Pipeline:
      1. Load reference genome
      2. Pre-scan reference for methylation sites (fuzznuc primary, regex fallback)
      3. Load trained MLPPredictor from checkpoint
      4. Parse .maf alignment mapping
      5. For batches of reads in .fq.gz:
         a. Collect all (kmer_id, meth_id) contexts using the pre-computed map
         b. Generate signals in one batched forward pass
         c. Write unaligned BAM with fi/fp tags

    Args:
        circular:      Treat genome as circular (default True for bacteria).
        revcomp:       Scan reverse complement strand for motifs (default True).
        no_fuzznuc:    Force Python regex for reference scanning; skip fuzznuc.
                       By default fuzznuc is tried first, falling back to regex
                       automatically if EMBOSS is not installed.
        batch_reads:   Number of reads to batch for GPU inference (default 1000).
        deterministic: If True, use predicted μ only (no stochastic sampling).
                       Default False matches natural PacBio signal variability.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading reference: {ref_path}")
    ref_seqs = load_reference(ref_path)

    backend = "regex (forced)" if no_fuzznuc else "fuzznuc (primary, regex fallback)"
    print(f"Pre-scanning reference for methylation sites ({backend})...")
    meth_map = build_reference_meth_map(ref_seqs, motif_string,
                                        revcomp=revcomp,
                                        no_fuzznuc=no_fuzznuc)

    # Keep regex motifs for the fallback path (unmapped reads)
    fallback_motifs = parse_motifs(motif_string, revcomp=revcomp)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = _load_model(checkpoint_path, device)
    mode_label = "deterministic (mean)" if deterministic else "stochastic (sample)"
    print(f"  Inference mode: {mode_label}")

    print(f"Parsing MAF: {maf_path}")
    maf_mapping = parse_maf(maf_path)

    print(f"Generating signals for reads from {fastq_path}...")
    n_reads    = 0
    n_mapped   = 0
    n_unmapped = 0

    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "unknown"}
    })

    open_func = gzip.open if fastq_path.endswith(".gz") else open

    with pysam.AlignmentFile(output_bam, "wb", header=header) as bam_out, \
         open_func(fastq_path, "rt") as fq:

        batch = []

        while True:
            hdr_line = fq.readline()
            if not hdr_line:
                break
            seq_line  = fq.readline()
            fq.readline()   # '+'
            qual_line = fq.readline()

            read_name = hdr_line.strip()[1:].split()[0]
            seq       = seq_line.strip()
            qual_str  = qual_line.strip()
            read_len  = len(seq)
            n_reads  += 1

            batch.append({"name": read_name, "seq": seq,
                          "qual": qual_str,  "len": read_len})

            if len(batch) >= batch_reads:
                n_m, n_u = _process_batch(
                    batch, ref_seqs, maf_mapping, meth_map, fallback_motifs,
                    model, device, deterministic, circular, bam_out, header)
                n_mapped   += n_m
                n_unmapped += n_u
                batch = []

        if batch:
            n_m, n_u = _process_batch(
                batch, ref_seqs, maf_mapping, meth_map, fallback_motifs,
                model, device, deterministic, circular, bam_out, header)
            n_mapped   += n_m
            n_unmapped += n_u

    print(f"Done. {n_reads} reads processed "
          f"({n_mapped} with ref context, {n_unmapped} without).")
    print(f"Output: {output_bam}")


def _process_batch(
    batch, ref_seqs, maf_mapping, meth_map, fallback_motifs,
    model, device, deterministic, circular, bam_out, header,
):
    """Process a batch of reads with batched MLP inference.

    Builds a flat list of (kmer_id, meth_id) pairs for all positions across
    all reads in the batch, runs a single forward pass, then writes each read
    to the BAM with its slice of the generated signals.

    Returns:
        Tuple (n_mapped, n_unmapped) — read counts for the batch.
    """
    all_kmer_ids = []
    all_meth_ids = []
    is_n_context = []   # Per-position flag: True = N-context, use default signal
    read_offsets = [0]

    n_mapped = n_unmapped = 0

    for read_data in batch:
        read_name = read_data["name"]
        seq       = read_data["seq"]
        read_len  = read_data["len"]

        maf_info = maf_mapping.get(read_name)

        if maf_info and maf_info[0] in ref_seqs:
            # ---- Mapped path: use reference context for edge bases ----
            ref_name, ref_start, _ref_strand, _ref_src_size = maf_info
            ref_seq  = ref_seqs[ref_name]
            ref_len  = len(ref_seq)
            ref_meth = meth_map[ref_name]

            # Extended context pads K//2 bases on each side from the reference,
            # ensuring accurate 11-mer encoding at the read edges.
            ext_context  = get_extended_context(ref_seq, ref_start, read_len, circular)
            current_kmer = 0

            for i in range(len(ext_context)):
                base_val     = BASE_MAP.get(ext_context[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                if i >= K - 1:
                    read_pos = i - (K - 1)
                    if 0 <= read_pos < read_len:
                        context_window = ext_context[i - (K - 1): i + 1]
                        has_n = "N" in context_window
                        is_n_context.append(has_n)

                        if has_n:
                            # N positions get a placeholder; replaced with default (1,1) later
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
            # ---- Unmapped path: read-only context ----
            # Per-read regex scanning (fuzznuc is only used for the reference
            # pre-scan above; subprocess calls per read would be too slow).
            meth_status  = scan_sequence(seq, fallback_motifs)
            current_kmer = 0

            for i in range(read_len):
                base_val     = BASE_MAP.get(seq[i], 0)
                current_kmer = ((current_kmer << 2) | base_val) & KMER_MASK

                if i < K - 1:
                    # First K-1 positions lack a full 11-mer window
                    is_n_context.append(True)
                    all_kmer_ids.append(0)
                    all_meth_ids.append(0)
                else:
                    is_n_context.append(False)
                    center = i - MID
                    all_kmer_ids.append(current_kmer)
                    all_meth_ids.append(int(meth_status[center]))

            n_unmapped += 1

        read_offsets.append(len(all_kmer_ids))

    # Batched MLP inference: one forward pass for all positions in the batch
    if len(all_kmer_ids) > 0:
        all_signals = generate_signals_batch(model, all_kmer_ids, all_meth_ids,
                                             device, deterministic)
    else:
        all_signals = np.zeros((0, 2), dtype=np.float32)

    # Split signals back to individual reads and write BAM records
    for idx, read_data in enumerate(batch):
        start   = read_offsets[idx]
        end     = read_offsets[idx + 1]
        signals = all_signals[start:end]
        is_n    = is_n_context[start:end]

        ipd_vals = np.clip(signals[:, 0], 0, 255).astype(np.uint8)
        pw_vals  = np.clip(signals[:, 1], 0, 255).astype(np.uint8)

        # N-context positions: replace with a safe default of 1 (not 0, which
        # could be mis-interpreted as a missing tag by downstream tools)
        for pos_idx, n_flag in enumerate(is_n):
            if n_flag:
                ipd_vals[pos_idx] = 1
                pw_vals[pos_idx]  = 1

        seg = pysam.AlignedSegment(header)
        seg.query_name      = read_data["name"]
        seg.flag            = 4   # unmapped
        seg.query_sequence  = read_data["seq"]
        seg.query_qualities = pysam.qualitystring_to_array(read_data["qual"])
        seg.set_tag("fi", array.array("B", ipd_vals.tolist()), "B")
        seg.set_tag("fp", array.array("B", pw_vals.tolist()),  "B")
        bam_out.write(seg)

    return n_mapped, n_unmapped


# ---------------------------------------------------------------------------
# Directory mode
# ---------------------------------------------------------------------------

def generate_directory(
    pbsim3_dir: str,
    checkpoint_path: str,
    motif_source: str,
    output_dir: str,
    circular: bool = True,
    revcomp: bool = True,
    device: str = "cuda",
    batch_reads: int = 1000,
    no_fuzznuc: bool = False,
    deterministic: bool = False,
    min_fraction: float = 0.40,
    min_detected: int = 20,
) -> None:
    """Generate signals for all species found under pbsim3_dir.

    Supports the same two directory layouts as dictionary inject (auto-detected):
      - Species subdirectories: pbsim3_dir/Ecoli/, pbsim3_dir/Salmonella/, ...
      - Flat layout: all files directly in pbsim3_dir, matched by basename.

    Output BAMs are written to output_dir as <species_name>_mlp.bam.
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
            print(f"ERROR: no motifs found for species '{species}'.",
                  file=sys.stderr)
            sys.exit(1)

        out_bam = os.path.join(output_dir, species + "_mlp.bam")
        print(f"\n--- {species} ---")
        generate_signals(
            fq_path, maf_path, ref_path, checkpoint_path, motif_string, out_bam,
            circular=circular, revcomp=revcomp,
            device=device, batch_reads=batch_reads,
            no_fuzznuc=no_fuzznuc, deterministic=deterministic,
        )

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
        prog="kinsim mlp generate",
        description=(
            "Generate MLP kinetic signals for all PBSIM3 species in a directory.\n\n"
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
            "  kinsim mlp generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pbsim3_dir",
                        help="Directory containing species subdirs or flat .fq.gz files")
    parser.add_argument("checkpoint", help="Trained MLP checkpoint (.pt)")
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
    parser.add_argument("--deterministic", action="store_true",
                        help="Use predicted mean μ only — no stochastic sampling. "
                             "Produces identical signals for every read at the same "
                             "context (useful for ablations). Default: stochastic.")
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
        deterministic=args.deterministic,
        min_fraction=args.min_fraction,
        min_detected=args.min_detected,
    )


def _main_per_genome(argv):
    """CLI for per-genome mode: processes a single .fq.gz file."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim mlp generate",
        description=(
            "Generate kinetic signals for PBSIM3 reads using a trained MLP predictor.\n\n"
            "Uses the .maf alignment to resolve reference context for edge bases.\n"
            "The reference is pre-scanned once for methylation sites; per-read\n"
            "lookups are O(1). Outputs an unaligned BAM with fi (IPD) and fp (PW) tags.\n\n"
            "Data preparation (same pipeline as cGAN):\n"
            "  kinsim cgan extract reads.bam motifs shard.pkl\n"
            "  kinsim cgan merge   shards/    master_data.pkl\n"
            "  kinsim mlp  train   master_data.pkl checkpoints_mlp/"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fastq",      help="PBSIM3 simulated reads (.fq or .fq.gz)")
    parser.add_argument("maf",        help="PBSIM3 alignment file (.maf or .maf.gz)")
    parser.add_argument("ref",        help="Reference genome FASTA (.fna, .fa, or .gz)")
    parser.add_argument("checkpoint", help="Trained MLP checkpoint (.pt)")
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
    parser.add_argument("--deterministic", action="store_true",
                        help="Use predicted mean μ only — no stochastic sampling. "
                             "Produces identical signals for every read at the same "
                             "context (useful for ablations). Default: stochastic.")
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
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
