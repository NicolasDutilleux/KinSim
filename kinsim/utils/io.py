"""File I/O utilities for FASTA references, MAF alignments, GFF annotations,
and PBSIM3 discovery.

Functions for loading FASTA references, parsing PBSIM3 MAF alignments,
loading ipdSummary GFF annotations for read-level methylation labelling,
extracting extended reference context for 11-mer encoding at read edges,
and discovering PBSIM3 output directory layouts.
"""

from __future__ import annotations

import glob
import gzip
import logging
import os
import re
import sys

import numpy as np

from .encoding import K, METH_IDS
from .motifs import load_motif_string

log = logging.getLogger(__name__)

MID = K // 2  # 5 — flanking bases on each side of the center position


# ---------------------------------------------------------------------------
# Reference loader
# ---------------------------------------------------------------------------

def load_reference(ref_path):
    """Load a FASTA file (.ref or .fna) into {name: sequence} dict."""
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
# GFF annotation loader (ipdSummary / kineticsTools)
# ---------------------------------------------------------------------------

_GFF_ATTR_RE = re.compile(r'(\w+)=([^;]+)')

# Map ipdSummary "identificationQv" base context to KinSim meth type.
# ipdSummary infers modification type from the modified base:
#   A → m6A,  C → m4C (or m5C if indicated).
# GFF records may also carry an explicit "modificationType" attribute.
_BASE_TO_METH = {'A': 'm6A', 'C': 'm4C'}


def load_gff_annotations(
    gff_path: str,
    min_score: float = 20.0,
    min_ipd_ratio: float = 0.0,
    allowed_mods: set[str] | None = None,
) -> dict[tuple[str, int, str], int]:
    """Load ipdSummary GFF3 into a position → meth_id lookup.

    Each qualifying GFF record is mapped to a (contig, 0-based position, strand)
    key.  The value is the integer meth_id from METH_IDS.

    New modification types appearing in the GFF that are not in METH_IDS are
    registered dynamically so the system generalizes to future types.

    Args:
        gff_path:       Path to ipdSummary .gff or .gff3 file.
        min_score:      Minimum kinetic score (-10*log10(pvalue)). Default 20
                        corresponds roughly to p < 0.01.
        min_ipd_ratio:  Optional minimum IPD ratio filter (0 = no filter).
        allowed_mods:   If provided, only keep records whose modification type
                        is in this set (e.g. {"m6A", "m4C"}).  Positions with
                        excluded types are SKIPPED entirely (not relabelled as
                        unmethylated), so they never appear in the training
                        data and cannot contaminate the unmeth class.
                        None (default) = accept all recognised types.

    Returns:
        dict mapping (contig, pos_0based, strand) → meth_id.
    """
    annotations: dict[tuple[str, int, str], int] = {}
    n_total = 0
    n_kept = 0
    type_counts: dict[str, int] = {}

    open_func = gzip.open if gff_path.endswith('.gz') else open

    with open_func(gff_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            fields = line.split('\t')
            if len(fields) < 9:
                continue

            feature = fields[2]
            if feature not in ('modified_base', 'kinetic', 'm6A', 'm4C', 'm5C'):
                continue

            n_total += 1

            # Score filter
            try:
                score = float(fields[5])
            except ValueError:
                continue
            if score < min_score:
                continue

            contig = fields[0]
            pos_1based = int(fields[3])
            strand = fields[6]
            attributes = fields[8]

            attrs = dict(_GFF_ATTR_RE.findall(attributes))

            # IPD ratio filter
            if min_ipd_ratio > 0:
                ipd_ratio_str = attrs.get('IPDRatio', '0')
                try:
                    if float(ipd_ratio_str) < min_ipd_ratio:
                        continue
                except ValueError:
                    continue

            # Determine modification type — only accept explicitly typed records.
            # 'modified_base' is a generic catch-all from ipdSummary meaning
            # "kinetically unusual" — NOT a confirmed modification type.
            # Only trust: (a) explicit feature type m6A/m4C/m5C, or
            #             (b) modificationType attribute in the GFF record.
            mod_type = attrs.get('modificationType', '')

            # If feature itself is a known mod type, use it
            if not mod_type and feature in METH_IDS:
                mod_type = feature

            # Do NOT infer from context base — that was producing false labels
            # for generic 'modified_base' records.

            if not mod_type:
                continue

            # Type-filter: skip records whose mod type is not in allowed_mods.
            # Skipping (rather than relabelling as unmeth) is intentional — it
            # prevents contaminating the unmeth class with truly-modified sites
            # the user has chosen to exclude.
            if allowed_mods is not None and mod_type not in allowed_mods:
                continue

            # Resolve to meth_id (register new types dynamically)
            if mod_type in METH_IDS:
                meth_id = METH_IDS[mod_type]
            else:
                # Dynamic registration for unknown modification types
                new_id = max(METH_IDS.values()) + 1
                METH_IDS[mod_type] = new_id
                log.info("Registered new modification type: %s -> id %d", mod_type, new_id)
                meth_id = new_id

            # GFF is 1-based, convert to 0-based
            annotations[(contig, pos_1based - 1, strand)] = meth_id
            n_kept += 1
            type_counts[mod_type] = type_counts.get(mod_type, 0) + 1

    log.info("GFF loaded: %s", gff_path)
    log.info("  %d records total, %d kept (score >= %.0f)",
             n_total, n_kept, min_score)
    if allowed_mods is not None:
        log.info("  allowed_mods filter: %s", sorted(allowed_mods))
    for mod_type, count in sorted(type_counts.items()):
        log.info("  %s: %d positions", mod_type, count)

    return annotations


def build_read_meth_array(
    annotations: dict[tuple[str, int, str], int],
    contig: str,
    ref_start: int,
    read_len: int,
    strand: str = '+',
) -> np.ndarray:
    """Build a per-base methylation array for one aligned read.

    Maps each read position to the reference coordinate, looks up the
    GFF annotation, and returns an int8 array of meth_ids (0 = unmethylated).

    Args:
        annotations: Output of load_gff_annotations().
        contig:      Reference contig name.
        ref_start:   0-based reference start of the alignment.
        read_len:    Length of the read (aligned portion).
        strand:      '+' or '-'.

    Returns:
        np.ndarray of shape (read_len,) with int8 meth_ids.
    """
    meth_array = np.zeros(read_len, dtype=np.int8)

    if strand == '+':
        for i in range(read_len):
            ref_pos = ref_start + i
            key = (contig, ref_pos, '+')
            if key in annotations:
                meth_array[i] = annotations[key]
    else:
        # Reverse strand: positions map in reverse order
        for i in range(read_len):
            ref_pos = ref_start + (read_len - 1 - i)
            key = (contig, ref_pos, '-')
            if key in annotations:
                meth_array[i] = annotations[key]

    return meth_array


# ---------------------------------------------------------------------------
# MAF parser
# ---------------------------------------------------------------------------

def parse_maf(maf_path):
    """Parse PBSIM3 .maf.gz to extract read-to-reference mapping.

    Returns dict[read_name] -> (ref_name, ref_start, ref_strand, ref_src_size)
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
# Reference context extraction
# ---------------------------------------------------------------------------

def get_extended_context(ref_seq, ref_start, read_len, circular=True):
    """Get reference context extended by MID on each side for edge-base encoding."""
    ref_len = len(ref_seq)
    start = ref_start - MID
    end   = ref_start + read_len + MID

    if circular and ref_len > 0:
        return ''.join(ref_seq[i % ref_len] for i in range(start, end))
    else:
        return ''.join(ref_seq[i] if 0 <= i < ref_len else 'N'
                       for i in range(start, end))


# ---------------------------------------------------------------------------
# PBSIM3 file discovery
# ---------------------------------------------------------------------------

def _find_file_by_extensions(directory, extensions):
    """Return the first file in directory matching any of the given extensions."""
    for ext in extensions:
        matches = sorted(glob.glob(os.path.join(directory, '*' + ext)))
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
    FQ_EXTS  = ('.fq.gz', '.fq')
    MAF_EXTS = ('.maf.gz', '.maf')
    REF_EXTS = ('.fna', '.fa', '.fasta')

    has_flat_fq = any(
        glob.glob(os.path.join(pbsim3_dir, '*' + ext)) for ext in FQ_EXTS
    )

    if has_flat_fq:
        fq_files = sorted(
            glob.glob(os.path.join(pbsim3_dir, '*.fq.gz')) +
            glob.glob(os.path.join(pbsim3_dir, '*.fq'))
        )
        search_dirs = [(pbsim3_dir, f) for f in fq_files]
    else:
        subdirs = sorted(
            d for d in glob.glob(os.path.join(pbsim3_dir, '*/'))
            if os.path.isdir(d)
        )
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
            species = fname[:-len('.fq.gz')] if fname.endswith('.fq.gz') else fname[:-len('.fq')]
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


def resolve_motifs_for_species(motif_source, species_name,
                               min_fraction=0.40, min_detected=20):
    """Return a motif string for one species.

    motif_source can be:
      - A KinSim motif string, PacBio .csv, or REBASE file -> applied to all.
      - A per-species mapping file with lines: "species_name|motif_string"
    """
    if os.path.isfile(motif_source) and not motif_source.endswith('.csv'):
        with open(motif_source) as f:
            for line in f:
                line = line.strip()
                if line.startswith(species_name + '|'):
                    return line.split('|', 1)[1]
    return load_motif_string(motif_source,
                             min_fraction=min_fraction,
                             min_detected=min_detected)
