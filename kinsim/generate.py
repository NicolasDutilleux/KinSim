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
output: unaligned BAM (flag=4) with all four PacBio kinetic tags:
  fi:B:C  — forward strand IPD (polymerase on template strand)
  fp:B:C  — forward strand PW
  ri:B:C  — reverse strand IPD (polymerase on complementary strand)
  rp:B:C  — reverse strand PW

For ri/rp, the kmer context is RC(forward_kmer_at_position_i): the polymerase
reading the reverse strand encounters the reverse-complement sequence, so the
11-mer it "sees" at each position is the RC of the forward 11-mer.
The methylation state (meth_id) is shared: the meth_map (built with revcomp=True)
already encodes both-strand methylation at each reference position.
"""

import array
import gzip
import json
import logging
import os
import sys

import numpy as np
import pysam
import torch
import torch.nn as nn

from .models.predictor import MLPPredictor, create_from_config
from .utils.encoding import BASE_MAP, K, KMER_MASK
from .utils.motifs import (build_reference_meth_map, load_motif_string,
                     parse_motifs, scan_sequence)
from .utils.io import (MID, find_pbsim3_files, resolve_motifs_for_species,
                        get_extended_context, load_reference, parse_maf)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RC kmer helper
# ---------------------------------------------------------------------------

def _rc_kmer(kmer_id: int) -> int:
    """Reverse complement a 22-bit encoded 11-mer.

    Encoding convention (big-endian): seq[0] occupies bits 21-20, seq[10]
    occupies bits 1-0.  Complement maps A↔T (0↔3) and C↔G (1↔2), i.e.
    XOR with 3 on each 2-bit unit.  Reversing bit-pair order yields the RC.

    Example: encode_kmer("ACGT...") → _rc_kmer() → encode_kmer(reverse_complement("ACGT..."))
    """
    rc = 0
    for _ in range(K):   # K = 11 iterations, one per base
        base = kmer_id & 3
        rc   = (rc << 2) | (base ^ 3)   # complement: 0↔3 (A↔T), 1↔2 (C↔G)
        kmer_id >>= 2
    return rc


# ---------------------------------------------------------------------------
# Fraction lookup from motif string
# ---------------------------------------------------------------------------

def _build_fraction_lookup(motif_string: str) -> dict[int, float]:
    """Parse motif string to build a meth_id → fraction lookup.

    The fraction is the 5th field in PacBio-derived motif strings
    (e.g. "m6A,GATC,1,3551,0.998").  Defaults to 1.0 when absent.
    Unmethylated (meth_id=0) always maps to 0.0.
    """
    from .utils.encoding import METH_IDS

    fracs: dict[int, float] = {0: 0.0}
    if not motif_string:
        return fracs
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry:
            continue
        parts = entry.split(',')
        if len(parts) < 3:
            continue
        m_id = METH_IDS.get(parts[0], 0)
        frac = float(parts[4]) if len(parts) >= 5 else 1.0
        fracs[m_id] = frac
    return fracs


# ---------------------------------------------------------------------------
# Batched MLP inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_signals_batch(
    model: MLPPredictor,
    kmer_ids: list,
    meth_ids: list,
    fractions: list,
    device: torch.device,
    deterministic: bool = False,
) -> np.ndarray:
    """Generate IPD/PW signals for a batch of contexts using MLPPredictor.

    Args:
        model:         Trained MLPPredictor in eval mode.
        kmer_ids:      List of kmer integer IDs (22-bit encoded 11-mers).
        meth_ids:      List of methylation IDs (0–3).
        fractions:     List of stoichiometric fractions (0.0–1.0) per position.
                       For unmethylated positions (meth_id=0), fraction is 0.0.
        device:        Torch device.
        deterministic: If True, return the predicted mean μ (no sampling).
                       If False, sample from N(μ, σ²) for biological realism.

    Returns:
        np.ndarray of shape (N, 2) with raw [IPD, PW] values in [0, 255].
    """
    N = len(kmer_ids)
    CHUNK = 50_000  # positions per GPU forward pass — prevents OOM on long reads

    if N <= CHUNK:
        kmer_tensor = torch.tensor(kmer_ids, dtype=torch.long, device=device)
        meth_ids_t  = torch.tensor(meth_ids,   dtype=torch.long,  device=device)
        fractions_t = torch.tensor(fractions,   dtype=torch.float, device=device)
        meth_probs  = torch.zeros(N, 4, device=device)
        meth_probs.scatter_(1, meth_ids_t.unsqueeze(1), fractions_t.unsqueeze(1))
        if deterministic:
            return model.predict_mean(kmer_tensor, meth_probs).cpu().numpy()
        else:
            return model.sample(kmer_tensor, meth_probs).cpu().numpy()

    # Large batch: process in position-level chunks to avoid OOM.
    chunks = []
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        k_t = torch.tensor(kmer_ids[start:end], dtype=torch.long, device=device)
        m_t = torch.tensor(meth_ids[start:end], dtype=torch.long, device=device)
        f_t = torch.tensor(fractions[start:end], dtype=torch.float, device=device)
        mp  = torch.zeros(end - start, 4, device=device)
        mp.scatter_(1, m_t.unsqueeze(1), f_t.unsqueeze(1))
        if deterministic:
            out = model.predict_mean(k_t, mp)
        else:
            out = model.sample(k_t, mp)
        chunks.append(out.cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a trained model from a checkpoint file.

    Reads model_config.json from the same directory as the checkpoint to
    reconstruct the exact architecture used during training.  Supports both
    ConvPredictor (architecture="conv") and MLPPredictor (architecture="mlp").

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device:          Torch device to load the model onto.

    Returns:
        Model in eval mode, ready for inference.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    config_path = os.path.join(os.path.dirname(checkpoint_path), "model_config.json")
    if not os.path.exists(config_path):
        log.error(
            "model_config.json not found in %s. "
            "This file is written by 'kinsim mlp train' at the start of training. "
            "Ensure the checkpoint directory contains model_config.json.",
            os.path.dirname(checkpoint_path),
        )
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    model = create_from_config(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    arch = config.get("architecture", "mlp")
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model loaded: architecture=%s  params=%s  checkpoint=%s",
             arch, f"{n_params:,}", os.path.basename(checkpoint_path))
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
         c. Write unaligned BAM with fi/fp/ri/rp tags

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
    log.info("Using device: %s", device)

    log.info("Loading reference: %s", ref_path)
    ref_seqs = load_reference(ref_path)

    backend = "regex (forced)" if no_fuzznuc else "fuzznuc (primary, regex fallback)"
    log.info("Pre-scanning reference for methylation sites (%s)...", backend)
    meth_map = build_reference_meth_map(ref_seqs, motif_string,
                                        revcomp=revcomp,
                                        no_fuzznuc=no_fuzznuc)

    # Fraction lookup: meth_id → stoichiometric fraction from the motif string
    frac_lookup = _build_fraction_lookup(motif_string)

    # Keep regex motifs for the fallback path (unmapped reads)
    fallback_motifs = parse_motifs(motif_string, revcomp=revcomp)

    log.info("Loading checkpoint: %s", checkpoint_path)
    model = _load_model(checkpoint_path, device)
    mode_label = "deterministic (mean)" if deterministic else "stochastic (sample)"
    log.info("Inference mode: %s", mode_label)

    log.info("Parsing MAF: %s", maf_path)
    maf_mapping = parse_maf(maf_path)

    log.info("Generating signals for reads from %s...", fastq_path)
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
                    batch, ref_seqs, maf_mapping, meth_map, frac_lookup,
                    fallback_motifs,
                    model, device, deterministic, circular, bam_out, header)
                n_mapped   += n_m
                n_unmapped += n_u
                batch = []

        if batch:
            n_m, n_u = _process_batch(
                batch, ref_seqs, maf_mapping, meth_map, frac_lookup,
                fallback_motifs,
                model, device, deterministic, circular, bam_out, header)
            n_mapped   += n_m
            n_unmapped += n_u

    log.info("Done. %d reads processed (%d with ref context, %d without).",
             n_reads, n_mapped, n_unmapped)
    log.info("Output: %s", output_bam)


def _process_batch(
    batch, ref_seqs, maf_mapping, meth_map, frac_lookup, fallback_motifs,
    model, device, deterministic, circular, bam_out, header,
):
    """Process a batch of reads with batched MLP inference.

    Builds a flat list of (kmer_id, meth_id, fraction) triples for all
    positions across all reads in the batch, runs a single forward pass,
    then writes each read to the BAM with its slice of the generated signals.

    Args:
        frac_lookup: dict mapping meth_id → stoichiometric fraction.

    Returns:
        Tuple (n_mapped, n_unmapped) — read counts for the batch.
    """
    all_kmer_ids    = []
    all_meth_ids    = []
    all_fractions   = []   # stoichiometric fraction per position
    all_rc_kmer_ids = []   # RC kmer IDs for ri/rp inference
    is_n_context    = []   # Per-position flag: True = N-context, use default signal
    read_offsets    = [0]

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
                            all_rc_kmer_ids.append(0)
                            all_meth_ids.append(0)
                            all_fractions.append(0.0)
                        else:
                            ref_pos = ref_start + read_pos
                            if circular:
                                meth_id = int(ref_meth[ref_pos % ref_len])
                            elif 0 <= ref_pos < ref_len:
                                meth_id = int(ref_meth[ref_pos])
                            else:
                                meth_id = 0
                            all_kmer_ids.append(current_kmer)
                            all_rc_kmer_ids.append(_rc_kmer(current_kmer))
                            all_meth_ids.append(meth_id)
                            all_fractions.append(frac_lookup.get(meth_id, 0.0))

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
                    all_rc_kmer_ids.append(0)
                    all_meth_ids.append(0)
                    all_fractions.append(0.0)
                else:
                    is_n_context.append(False)
                    center  = i - MID
                    meth_id = int(meth_status[center])
                    all_kmer_ids.append(current_kmer)
                    all_rc_kmer_ids.append(_rc_kmer(current_kmer))
                    all_meth_ids.append(meth_id)
                    all_fractions.append(frac_lookup.get(meth_id, 0.0))

            n_unmapped += 1

        read_offsets.append(len(all_kmer_ids))

    # Batched MLP inference:
    #   Pass 1 — forward kmers  → fi (IPD) and fp (PW)
    #   Pass 2 — RC kmers       → ri (IPD) and rp (PW)
    # Same meth_ids/fractions for both: the meth_map (built with revcomp=True)
    # encodes both-strand methylation at each reference position.
    if len(all_kmer_ids) > 0:
        all_signals    = generate_signals_batch(model, all_kmer_ids, all_meth_ids,
                                                all_fractions, device, deterministic)
        all_rc_signals = generate_signals_batch(model, all_rc_kmer_ids, all_meth_ids,
                                                all_fractions, device, deterministic)
    else:
        all_signals    = np.zeros((0, 2), dtype=np.float32)
        all_rc_signals = np.zeros((0, 2), dtype=np.float32)

    # Split signals back to individual reads and write BAM records
    for idx, read_data in enumerate(batch):
        start      = read_offsets[idx]
        end        = read_offsets[idx + 1]
        signals    = all_signals[start:end]
        rc_signals = all_rc_signals[start:end]
        is_n       = is_n_context[start:end]

        ipd_vals = np.clip(signals[:, 0],    0, 255).astype(np.uint8)
        pw_vals  = np.clip(signals[:, 1],    0, 255).astype(np.uint8)
        ri_vals  = np.clip(rc_signals[:, 0], 0, 255).astype(np.uint8)
        rp_vals  = np.clip(rc_signals[:, 1], 0, 255).astype(np.uint8)

        # N-context positions: replace with a safe default of 1 (not 0, which
        # could be mis-interpreted as a missing tag by downstream tools)
        for pos_idx, n_flag in enumerate(is_n):
            if n_flag:
                ipd_vals[pos_idx] = 1
                pw_vals[pos_idx]  = 1
                ri_vals[pos_idx]  = 1
                rp_vals[pos_idx]  = 1

        seg = pysam.AlignedSegment(header)
        seg.query_name      = read_data["name"]
        seg.flag            = 4   # unmapped
        seg.query_sequence  = read_data["seq"]
        seg.query_qualities = pysam.qualitystring_to_array(read_data["qual"])
        seg.set_tag("fi", array.array("B", ipd_vals.tolist()))
        seg.set_tag("fp", array.array("B", pw_vals.tolist()))
        seg.set_tag("ri", array.array("B", ri_vals.tolist()))
        seg.set_tag("rp", array.array("B", rp_vals.tolist()))
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
    genomes = find_pbsim3_files(pbsim3_dir)
    if not genomes:
        log.error("No genome sets found in %s", pbsim3_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    log.info("Found %d species in %s", len(genomes), pbsim3_dir)

    for fq_path, maf_path, ref_path, species in genomes:
        motif_string = resolve_motifs_for_species(motif_source, species,
                                                   min_fraction, min_detected)
        if not motif_string:
            log.error("No motifs found for species '%s'.", species)
            sys.exit(1)

        out_bam = os.path.join(output_dir, species + "_mlp.bam")
        log.info("--- %s ---", species)
        generate_signals(
            fq_path, maf_path, ref_path, checkpoint_path, motif_string, out_bam,
            circular=circular, revcomp=revcomp,
            device=device, batch_reads=batch_reads,
            no_fuzznuc=no_fuzznuc, deterministic=deterministic,
        )

    log.info("All done. %d BAM(s) written to: %s", len(genomes), output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# BAM input mode
# ---------------------------------------------------------------------------

def generate_from_bam(
    input_bam: str,
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
    """Generate kinetic signals for reads from an existing (aligned) BAM file.

    Reads sequences and alignment coordinates directly from the BAM — no FASTQ
    or MAF file needed.  Alignment info (reference_name, reference_start) is
    used exactly as the MAF would be, giving accurate reference-context padding
    at read edges.  Unaligned reads fall back to read-only context.

    Use this with a BAM that has had fi/fp/ri/rp tags stripped.
    Output BAM has the same reads + sequences as input, with fresh fi/fp/ri/rp.
    """
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device_obj)

    log.info("Loading reference: %s", ref_path)
    ref_seqs = load_reference(ref_path)

    backend = "regex (forced)" if no_fuzznuc else "fuzznuc (primary, regex fallback)"
    log.info("Pre-scanning reference for methylation sites (%s)...", backend)
    meth_map = build_reference_meth_map(ref_seqs, motif_string,
                                        revcomp=revcomp, no_fuzznuc=no_fuzznuc)
    frac_lookup    = _build_fraction_lookup(motif_string)
    fallback_motifs = parse_motifs(motif_string, revcomp=revcomp)

    log.info("Loading checkpoint: %s", checkpoint_path)
    model = _load_model(checkpoint_path, device_obj)
    mode_label = "deterministic (mean)" if deterministic else "stochastic (sample)"
    log.info("Inference mode: %s", mode_label)

    header_out = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "unknown"}})

    n_reads = n_mapped = n_unmapped = 0
    batch: list = []
    batch_maf: dict = {}

    log.info("Reading reads from: %s", input_bam)

    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam_in, \
         pysam.AlignmentFile(output_bam, "wb", header=header_out) as bam_out:

        for read in bam_in:
            if read.query_sequence is None:
                continue

            seq      = read.query_sequence
            qual     = read.query_qualities
            qual_str = pysam.array_to_qualitystring(qual) if qual is not None \
                       else "I" * len(seq)

            batch.append({
                "name": read.query_name,
                "seq":  seq,
                "qual": qual_str,
                "len":  len(seq),
            })

            # Build maf_mapping entry from BAM alignment — same fields parse_maf returns
            if (not read.is_unmapped
                    and read.reference_name is not None
                    and read.reference_name in ref_seqs):
                ref_len = len(ref_seqs[read.reference_name])
                batch_maf[read.query_name] = (
                    read.reference_name, read.reference_start, "+", ref_len
                )

            n_reads += 1

            if len(batch) >= batch_reads:
                n_m, n_u = _process_batch(
                    batch, ref_seqs, batch_maf, meth_map, frac_lookup,
                    fallback_motifs, model, device_obj, deterministic,
                    circular, bam_out, header_out,
                )
                n_mapped   += n_m
                n_unmapped += n_u
                batch = []
                batch_maf = {}
                if n_reads % 1000 == 0:
                    log.info("Progress: %d reads processed...", n_reads)

        if batch:
            n_m, n_u = _process_batch(
                batch, ref_seqs, batch_maf, meth_map, frac_lookup,
                fallback_motifs, model, device_obj, deterministic,
                circular, bam_out, header_out,
            )
            n_mapped   += n_m
            n_unmapped += n_u

    log.info("Done. %d reads processed (%d with ref context, %d without).",
             n_reads, n_mapped, n_unmapped)
    log.info("Output: %s", output_bam)


def _main_from_bam(argv):
    """CLI for BAM input mode: stripped real BAM → BAM with synthetic fi/fp/ri/rp."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="kinsim generate",
        description=(
            "Generate synthetic kinetic signals for reads in an existing BAM.\n\n"
            "Input BAM must have fi/fp/ri/rp tags removed (use strip-kinetics first).\n"
            "Alignment coordinates are read from the BAM — no MAF file needed.\n\n"
            "Usage:\n"
            "  kinsim generate <input.bam> <ref.fna> <checkpoint.pt> <motifs> <output.bam>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_bam",   help="Aligned BAM with fi/fp/ri/rp stripped")
    parser.add_argument("ref",         help="Reference genome FASTA (.fna / .fa / .gz)")
    parser.add_argument("checkpoint",  help="Trained checkpoint (.pt)")
    parser.add_argument("motifs",      help="Motif string, PacBio motifs.csv, or REBASE file")
    parser.add_argument("output",      help="Output BAM with synthetic fi/fp/ri/rp tags")
    parser.add_argument("--linear",       action="store_true",
                        help="Treat genome as linear (default: circular for bacteria)")
    parser.add_argument("--device",       default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-reads",  type=int, default=1000)
    parser.add_argument("--no-revcomp",   action="store_true")
    parser.add_argument("--no-fuzznuc",   action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--min-fraction", type=float, default=0.40)
    parser.add_argument("--min-detected", type=int, default=20)
    args = parser.parse_args(argv)

    motif_string = load_motif_string(args.motifs,
                                     min_fraction=args.min_fraction,
                                     min_detected=args.min_detected)
    if not motif_string:
        log.error("No motifs found from the provided source.")
        sys.exit(1)

    generate_from_bam(
        input_bam=args.input_bam,
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


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if argv and os.path.isdir(argv[0]):
        _main_directory(argv)
    elif argv and argv[0].endswith(".bam"):
        _main_from_bam(argv)
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
                        help="Use predicted mean (mu) only - no stochastic sampling. "
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
                        help="Use predicted mean (mu) only - no stochastic sampling. "
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
        log.error("No motifs found from the provided source.")
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
