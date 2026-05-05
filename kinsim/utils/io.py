"""File I/O utilities for FASTA references, MAF alignments, and PBSIM3 discovery.

Functions for loading FASTA references, parsing PBSIM3 MAF alignments,
extracting extended reference context for 11-mer encoding at read edges,
and discovering PBSIM3 output directory layouts.
"""

from __future__ import annotations

import glob
import gzip
import logging
import os

from .encoding import KMER_LEFT_PAD, KMER_RIGHT_PAD
from .motifs import load_motif_string

log = logging.getLogger(__name__)

# Asymmetric padding around each prediction position. With kmer covering
# [-KMER_LEFT_PAD, +KMER_RIGHT_PAD] from the prediction position, the
# extended context needs KMER_LEFT_PAD bases on the left side and
# KMER_RIGHT_PAD on the right of the read.
#
# `MID` is no longer a single value — kept temporarily as KMER_LEFT_PAD for
# any external import; new code should explicitly use KMER_LEFT_PAD or
# KMER_RIGHT_PAD depending on intent.
MID = KMER_LEFT_PAD


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------


def load_reference(ref_path):
    """Load a FASTA file (.ref or .fna) into {name: sequence} dict."""
    open_func = gzip.open if ref_path.endswith(".gz") else open
    seqs = {}
    current_name = None
    parts = []
    with open_func(ref_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    seqs[current_name] = "".join(parts)
                current_name = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.upper())
    if current_name:
        seqs[current_name] = "".join(parts)
    return seqs


# ---------------------------------------------------------------------------
# MAF parser
# ---------------------------------------------------------------------------


def parse_maf(maf_path):
    """Parse PBSIM3 .maf.gz to extract read-to-reference mapping.

    Returns dict[read_name] -> (ref_name, ref_start, ref_strand, ref_src_size)
    """
    mapping = {}
    open_func = gzip.open if maf_path.endswith(".gz") else open
    with open_func(maf_path, "rt") as f:
        lines_in_block = []
        for line in f:
            line = line.strip()
            if line.startswith("a"):
                lines_in_block = []
            elif line.startswith("s"):
                lines_in_block.append(line)
                if len(lines_in_block) == 2:
                    ref_parts = lines_in_block[0].split()
                    ref_name = ref_parts[1]
                    ref_start = int(ref_parts[2])
                    ref_strand = ref_parts[4]
                    ref_src_size = int(ref_parts[5])

                    read_parts = lines_in_block[1].split()
                    read_name = read_parts[1]

                    mapping[read_name] = (ref_name, ref_start, ref_strand, ref_src_size)
    return mapping


# ---------------------------------------------------------------------------
# Reference context extraction
# ---------------------------------------------------------------------------


def get_extended_context(ref_seq, ref_start, read_len, circular=True):
    """Get reference context extended ASYMMETRICALLY for edge-base encoding.

    For the asymmetric kmer [-KMER_LEFT_PAD, +KMER_RIGHT_PAD] window we need
    KMER_LEFT_PAD bases on the upstream side and KMER_RIGHT_PAD bases on the
    downstream side of the read so every read position has a complete kmer.
    """
    ref_len = len(ref_seq)
    start = ref_start - KMER_LEFT_PAD
    end = ref_start + read_len + KMER_RIGHT_PAD

    if circular and ref_len > 0:
        return "".join(ref_seq[i % ref_len] for i in range(start, end))
    else:
        return "".join(ref_seq[i] if 0 <= i < ref_len else "N" for i in range(start, end))


# ---------------------------------------------------------------------------
# PBSIM3 file discovery
# ---------------------------------------------------------------------------


def _find_file_by_extensions(directory, extensions):
    """Return the first file in directory matching any of the given extensions."""
    for ext in extensions:
        matches = sorted(glob.glob(os.path.join(directory, "*" + ext)))
        if matches:
            return matches[0]
    return None


def find_pbsim3_files(pbsim3_dir):
    """Discover all PBSIM3 genome sets under pbsim3_dir.

    Supports two layouts (auto-detected):
      - Species subdirectories: pbsim3_dir/Ecoli/reads.fq.gz ...
      - Flat layout: pbsim3_dir/Ecoli.fq.gz ...

    Returns list of (fq_path, maf_path, ref_path, species_name) tuples.
    """
    FQ_EXTS = (".fq.gz", ".fq")
    MAF_EXTS = (".maf.gz", ".maf")
    REF_EXTS = (".fna", ".fa", ".fasta")

    has_flat_fq = any(glob.glob(os.path.join(pbsim3_dir, "*" + ext)) for ext in FQ_EXTS)

    if has_flat_fq:
        fq_files = sorted(
            glob.glob(os.path.join(pbsim3_dir, "*.fq.gz"))
            + glob.glob(os.path.join(pbsim3_dir, "*.fq"))
        )
        search_dirs = [(pbsim3_dir, f) for f in fq_files]
    else:
        subdirs = sorted(d for d in glob.glob(os.path.join(pbsim3_dir, "*/")) if os.path.isdir(d))
        if not subdirs:
            return []
        search_dirs = []
        for subdir in subdirs:
            fq = _find_file_by_extensions(subdir, FQ_EXTS)
            if fq:
                search_dirs.append((subdir, fq))

    results = []
    for search_dir, fq_path in search_dirs:
        if has_flat_fq:
            fname = os.path.basename(fq_path)
            species = fname[: -len(".fq.gz")] if fname.endswith(".fq.gz") else fname[: -len(".fq")]
        else:
            species = os.path.basename(os.path.dirname(fq_path))

        maf_path = _find_file_by_extensions(search_dir, MAF_EXTS)
        if maf_path is None:
            log.warning("No .maf.gz/.maf for '%s' -- skipping", species)
            continue

        ref_path = _find_file_by_extensions(search_dir, REF_EXTS)
        if ref_path is None:
            log.warning("No .fna/.fa/.fasta for '%s' -- skipping", species)
            continue

        results.append((fq_path, maf_path, ref_path, species))

    return results


def resolve_motifs_for_species(motif_source, species_name, min_fraction=0.40, min_detected=20):
    """Return a motif string for one species.

    motif_source can be:
      - A KinSim motif string, PacBio .csv, or REBASE file -> applied to all.
      - A per-species mapping file with lines: "species_name|motif_string"
    """
    if os.path.isfile(motif_source) and not motif_source.endswith(".csv"):
        with open(motif_source) as f:
            for line in f:
                line = line.strip()
                if line.startswith(species_name + "|"):
                    return line.split("|", 1)[1]
    return load_motif_string(motif_source, min_fraction=min_fraction, min_detected=min_detected)
