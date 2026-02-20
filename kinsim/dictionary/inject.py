"""Inject IPD/PW kinetic signals into PBSIM3 simulated reads.

Uses a trained 11-mer dictionary + the PBSIM3 .maf alignment to resolve
reference context for edge bases (first/last 5 positions of each read).
Outputs an unaligned BAM with fi (IPD) and fp (PW) tags.

Two calling modes (auto-detected):
  Directory mode:   kinsim dictionary inject <pbsim3_dir> <dict.pkl> <motifs> <output_dir>
  Per-genome mode:  kinsim dictionary inject <fq.gz> <maf.gz> <ref.fna> <dict.pkl> <motifs> <out.bam>

Directory mode supports two layouts (auto-detected):

  Species subdirectories (recommended — one subdir per species):
    pbsim3_dir/
      Ecoli/
        reads.fq.gz
        reads.maf.gz
        Ecoli.fna
      Salmonella/
        reads.fq.gz
        reads.maf.gz
        Salmonella.fna

  Flat layout (all species files in one directory, matched by basename):
    pbsim3_dir/
      Ecoli.fq.gz   Ecoli.maf.gz   Ecoli.fna
      Salmonella.fq.gz ...

Motif input (auto-detected in both modes):
  - KinSim motif string       — "m6A,GATC,1;m4C,CCWGG,1"  (applied to all genomes)
  - Per-species mapping file  — text file with "species_name|motif_string" per line
  - PacBio motifs.csv         — file path ending in .csv
  - REBASE file               — any other file path

Signal context design note:
  For edge bases (first/last 5 positions of each read), the polymerase
  experienced the full genomic 11-mer context even though those flanking
  bases are not part of the synthetic read.  We therefore use the .maf
  alignment to extend each read 5 bp into the reference on both sides,
  ensuring correct kinetic signal sampling for all positions.
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
                   circular=True, revcomp=True, no_fuzznuc=False,
                   lookup=None, read_group=None):
    """Inject IPD/PW signals into PBSIM3 reads for a single genome.

    Pipeline:
      1. Load reference genome
      2. Pre-scan reference for methylation sites (fuzznuc primary, regex fallback)
      3. Load trained dictionary (skipped if lookup is pre-loaded)
      4. Parse .maf alignment mapping
      5. For each read in .fq.gz:
         a. Get extended reference context via .maf (for kmer building at edges)
         b. Look up methylation status from pre-computed reference map
         c. Encode 11-mers and sample IPD/PW from dictionary
      6. Write unaligned BAM with fi/fp tags

    Args:
        circular:    Treat genome as circular (default True for bacteria).
        revcomp:     Scan reverse complement strand for motifs (default True).
        no_fuzznuc:  Force Python regex for reference scanning; skip fuzznuc.
                     By default fuzznuc is tried first, falling back to regex
                     automatically if EMBOSS is not installed.
        lookup:      Pre-loaded dictionary (dict). If None, loaded from pkl_path.
                     Pass a pre-loaded dict to avoid repeated I/O when processing
                     many species (e.g. in metagenome mode).
        read_group:  Species name for @RG header entry and RG tag on each read.
                     If None, no read group information is added to the BAM.
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

    if lookup is None:
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

    header_dict = {'HD': {'VN': '1.6', 'SO': 'unknown'}}
    if read_group:
        header_dict['RG'] = [{'ID': read_group, 'SM': read_group}]
    header = pysam.AlignmentHeader.from_dict(header_dict)

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
                ref_name, ref_start, _, _ = maf_info
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
            if read_group:
                seg.set_tag('RG', read_group, 'Z')
            bam_out.write(seg)

    print(f"Done. {n_reads} reads processed "
          f"({n_mapped} with ref context, {n_unmapped} without).")
    print(f"Output: {output_bam}")


# ---------------------------------------------------------------------------
# Directory mode helpers
# ---------------------------------------------------------------------------

def _find_file_by_extensions(directory, extensions):
    """Return the first file in directory matching any of the given extensions."""
    for ext in extensions:
        matches = sorted(glob.glob(os.path.join(directory, '*' + ext)))
        if matches:
            return matches[0]
    return None


def _find_pbsim3_files(pbsim3_dir):
    """Discover all PBSIM3 genome sets under pbsim3_dir.

    Supports two layouts (auto-detected):

    Species subdirectories (recommended):
      pbsim3_dir/
        Ecoli/          <- species name = subdir name
          reads.fq.gz
          reads.maf.gz
          Ecoli.fna
        Salmonella/
          ...

    Flat layout (all files in one directory, matched by basename):
      pbsim3_dir/
        Ecoli.fq.gz   Ecoli.maf.gz   Ecoli.fna
        Salmonella.fq.gz ...

    Auto-detection: if any .fq.gz files exist directly in pbsim3_dir → flat mode,
    otherwise looks for subdirectories containing .fq.gz files → subdir mode.

    Returns a sorted list of (fq_path, maf_path, ref_path, species_name) tuples.
    Skips entries where .maf or reference cannot be found (prints a warning).
    """
    FQ_EXTS  = ('.fq.gz', '.fq')
    MAF_EXTS = ('.maf.gz', '.maf')
    REF_EXTS = ('.fna', '.fa', '.fasta')

    # --- Detect layout ---
    has_flat_fq = any(
        glob.glob(os.path.join(pbsim3_dir, '*' + ext)) for ext in FQ_EXTS
    )

    if has_flat_fq:
        # ---- Flat layout: files matched by basename ----
        fq_files = sorted(
            glob.glob(os.path.join(pbsim3_dir, '*.fq.gz')) +
            glob.glob(os.path.join(pbsim3_dir, '*.fq'))
        )
        search_dirs = [(pbsim3_dir, f) for f in fq_files]
    else:
        # ---- Subdir layout: one subdirectory per species ----
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
        # Species name: subdir basename (subdir mode) or file stem (flat mode)
        if has_flat_fq:
            fname = os.path.basename(fq_path)
            species = fname[:-len('.fq.gz')] if fname.endswith('.fq.gz') else fname[:-len('.fq')]
        else:
            species = os.path.basename(os.path.dirname(fq_path))

        maf_path = _find_file_by_extensions(search_dir, MAF_EXTS)
        if maf_path is None:
            print(f"  WARN: no .maf.gz/.maf for '{species}' — skipping", file=sys.stderr)
            continue

        ref_path = _find_file_by_extensions(search_dir, REF_EXTS)
        if ref_path is None:
            print(f"  WARN: no .fna/.fa/.fasta for '{species}' — skipping", file=sys.stderr)
            continue

        results.append((fq_path, maf_path, ref_path, species))

    return results


def _resolve_motifs_for_species(motif_source, species_name,
                                min_fraction=0.40, min_detected=20):
    """Return a motif string for one species.

    motif_source can be:
      - A KinSim motif string, PacBio .csv, or REBASE file → applied to all species.
      - A per-species mapping file with lines: "species_name|motif_string"
        → looked up by species_name.
    """
    # Check for per-species mapping file: lines like "Ecoli|m6A,GATC,2"
    if os.path.isfile(motif_source) and not motif_source.endswith('.csv'):
        with open(motif_source) as f:
            for line in f:
                line = line.strip()
                if line.startswith(species_name + '|'):
                    return line.split('|', 1)[1]
        # Not a mapping file hit → fall through to load_motif_string
    return load_motif_string(motif_source,
                             min_fraction=min_fraction,
                             min_detected=min_detected)


def inject_directory(pbsim3_dir, pkl_path, motif_source, output_dir,
                     circular=True, revcomp=True, no_fuzznuc=False,
                     min_fraction=0.40, min_detected=20):
    """Inject signals into all species found under pbsim3_dir.

    Supports two directory layouts (auto-detected):
      - Species subdirectories: pbsim3_dir/Ecoli/, pbsim3_dir/Salmonella/, ...
      - Flat: all files directly in pbsim3_dir, matched by basename.

    motif_source can be:
      - A single KinSim motif string → applied to all species.
      - A per-species mapping file (lines: "species_name|motif_string").
      - A PacBio .csv or REBASE file → applied to all species.

    Output BAMs are written to output_dir as <species_name>_kinsim.bam.
    """
    genomes = _find_pbsim3_files(pbsim3_dir)
    if not genomes:
        print(f"ERROR: No genome sets found in {pbsim3_dir}", file=sys.stderr)
        print("  Expected either species subdirectories (each with .fq.gz + .maf.gz + .fna)",
              file=sys.stderr)
        print("  or flat layout (.fq.gz files matched with .maf.gz/.fna by basename).",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Found {len(genomes)} species in {pbsim3_dir}")

    # Load dictionary once for all species
    print(f"Loading dictionary: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        lookup = pickle.load(f)

    for fq_path, maf_path, ref_path, species in genomes:
        motif_string = _resolve_motifs_for_species(motif_source, species,
                                                   min_fraction, min_detected)
        if not motif_string:
            print(f"ERROR: no motifs found for species '{species}'.", file=sys.stderr)
            sys.exit(1)

        out_bam = os.path.join(output_dir, species + '_kinsim.bam')
        print(f"\n--- {species} ---")
        inject_signals(fq_path, maf_path, ref_path, pkl_path, motif_string, out_bam,
                       circular=circular, revcomp=revcomp, no_fuzznuc=no_fuzznuc,
                       lookup=lookup)

    print(f"\nAll done. {len(genomes)} BAM(s) written to: {output_dir}")


# ---------------------------------------------------------------------------
# Metagenomic mode — multi-species pooling
# ---------------------------------------------------------------------------

def inject_metagenome(root_dir, pkl_path, output_dir,
                      circular=True, revcomp=True, no_fuzznuc=False,
                      min_fraction=0.40, min_detected=20,
                      keep_species=False):
    """Inject signals into all species in root_dir and pool into one BAM.

    Each species subdirectory must contain:
      - genome.fna (or .fa / .fasta)        — reference genome
      - reads.fq.gz (or .fq / .fastq / .fastq.gz)  — PBSIM3 reads
      - reads.maf.gz (or .maf)              — PBSIM3 alignment
      - motifs.csv                           — PacBio motif summary

    Pipeline:
      1. Load dictionary ONCE (not per-species)
      2. For each species: inject signals with @RG tag → per-species BAM
      3. Merge all per-species BAMs into one unified BAM with all @RG entries
      4. Sort merged BAM by query name (standard for unaligned reads)
      5. Optionally clean up per-species intermediate BAMs

    Output:
      output_dir/meta_community.bam   — merged, queryname-sorted BAM

    Note: the final BAM is unaligned (flag=4, no reference coordinates).
    To use with MetaBAT2/SemiBin/ipdSummary, align to a reference catalog
    (e.g. minimap2 or pbmm2), then sort by coordinate and index.
    """
    # ---- Discover species subdirectories ----
    species_dirs = sorted(
        d for d in glob.glob(os.path.join(root_dir, '*/'))
        if os.path.isdir(d)
    )
    if not species_dirs:
        print(f"ERROR: No species subdirectories found in {root_dir}", file=sys.stderr)
        print("  Each species must have its own subdirectory containing:", file=sys.stderr)
        print("    genome.fna, reads.fq.gz, reads.maf.gz, motifs.csv", file=sys.stderr)
        sys.exit(1)

    FQ_EXTS  = ('.fq.gz', '.fastq.gz', '.fq', '.fastq')
    MAF_EXTS = ('.maf.gz', '.maf')
    REF_EXTS = ('.fna', '.fa', '.fasta')

    # ---- Validate all species before starting ----
    species_info = []  # (species_name, fq, maf, ref, motifs_csv)
    for species_dir in species_dirs:
        species = os.path.basename(os.path.normpath(species_dir))

        fq_path  = _find_file_by_extensions(species_dir, FQ_EXTS)
        maf_path = _find_file_by_extensions(species_dir, MAF_EXTS)
        ref_path = _find_file_by_extensions(species_dir, REF_EXTS)
        csv_path = _find_file_by_extensions(species_dir, ('.csv',))

        missing = []
        if not fq_path:  missing.append('.fq.gz / .fastq.gz')
        if not maf_path: missing.append('.maf.gz / .maf')
        if not ref_path: missing.append('.fna / .fa')
        if not csv_path: missing.append('motifs.csv')

        if missing:
            print(f"ERROR: Species '{species}' is missing: {', '.join(missing)}",
                  file=sys.stderr)
            sys.exit(1)

        species_info.append((species, fq_path, maf_path, ref_path, csv_path))

    print(f"Found {len(species_info)} species in {root_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load dictionary ONCE ----
    print(f"Loading dictionary: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        lookup = pickle.load(f)

    # ---- Inject per species ----
    species_bams = []
    for species, fq_path, maf_path, ref_path, csv_path in species_info:
        motif_string = load_motif_string(csv_path,
                                         min_fraction=min_fraction,
                                         min_detected=min_detected)
        if not motif_string:
            print(f"ERROR: No motifs found in {csv_path} for species '{species}'.",
                  file=sys.stderr)
            sys.exit(1)

        out_bam = os.path.join(output_dir, f"{species}_kinsim.bam")
        print(f"\n--- {species} ---")
        inject_signals(fq_path, maf_path, ref_path, pkl_path, motif_string, out_bam,
                       circular=circular, revcomp=revcomp, no_fuzznuc=no_fuzznuc,
                       lookup=lookup, read_group=species)
        species_bams.append((species, out_bam))

    # ---- Merge into one BAM with unified @RG header ----
    print(f"\nMerging {len(species_bams)} species BAMs...")
    merged_unsorted = os.path.join(output_dir, "_meta_unsorted.bam")

    rg_entries = [{'ID': sp, 'SM': sp} for sp, _ in species_bams]
    merged_header = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'unsorted'},
        'RG': rg_entries,
    })

    with pysam.AlignmentFile(merged_unsorted, "wb", header=merged_header) as out_bam:
        for species, bam_path in species_bams:
            with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as in_bam:
                for seg in in_bam:
                    out_bam.write(seg)

    # ---- Sort by query name (appropriate for unaligned reads) ----
    print("Sorting by query name...")
    final_bam = os.path.join(output_dir, "meta_community.bam")
    pysam.sort("-n", "-o", final_bam, merged_unsorted)
    os.remove(merged_unsorted)

    # ---- Clean up per-species BAMs ----
    if not keep_species:
        for _, bam_path in species_bams:
            if os.path.isfile(bam_path):
                os.remove(bam_path)

    print(f"\nDone.")
    print(f"  Output:  {final_bam}")
    print(f"  Species: {len(species_bams)} (@RG tags: {', '.join(s for s, _ in species_bams)})")
    print()
    print("  Next steps for metagenomic binning:")
    print("    1. Align to a concatenated reference catalog:")
    print("         minimap2 -a -x map-hifi catalog.fna meta_community.bam | samtools sort -o aligned.bam")
    print("    2. Index the aligned BAM:")
    print("         samtools index aligned.bam")
    print("    3. Run binners (MetaBAT2, SemiBin) on aligned.bam")


def metagenome_main(argv=None):
    """CLI for kinsim dictionary metagenome."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="kinsim dictionary metagenome",
        description=(
            "Inject kinetic signals into all species in a directory and pool into\n"
            "a single BAM file ready for metagenomic binning.\n\n"
            "Each species subdirectory must contain:\n"
            "  genome.fna     — reference genome\n"
            "  reads.fq.gz    — PBSIM3 simulated reads\n"
            "  reads.maf.gz   — PBSIM3 alignment file\n"
            "  motifs.csv     — PacBio motif summary\n\n"
            "Example directory structure:\n"
            "  metagenome_root/\n"
            "    Ecoli/\n"
            "      genome.fna  reads.fq.gz  reads.maf.gz  motifs.csv\n"
            "    Salmonella/\n"
            "      genome.fna  reads.fq.gz  reads.maf.gz  motifs.csv\n\n"
            "Output: output_dir/meta_community.bam\n"
            "  — Merged, queryname-sorted BAM with @RG tags per species\n"
            "  — fi (IPD) and fp (PW) tags on every read\n"
            "  — Dictionary loaded ONCE for all species"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root_dir",
                        help="Root directory containing one species subdirectory per species")
    parser.add_argument("dict",
                        help="Trained kinetic dictionary (.pkl)")
    parser.add_argument("output_dir",
                        help="Output directory (receives meta_community.bam)")
    parser.add_argument("--linear", action="store_true",
                        help="Treat genomes as linear (default: circular for bacteria)")
    parser.add_argument("--no-revcomp", action="store_true",
                        help="Do not scan reverse complement strand for motifs")
    parser.add_argument("--no-fuzznuc", action="store_true",
                        help="Force Python regex for reference methylation scanning "
                             "(by default fuzznuc is tried first, regex as fallback)")
    parser.add_argument("--keep-species", action="store_true",
                        help="Keep per-species BAMs after merging (default: delete them)")
    parser.add_argument("--min-fraction", type=float, default=0.40,
                        help="Minimum motif fraction threshold (motifs.csv, default: 0.40)")
    parser.add_argument("--min-detected", type=int, default=20,
                        help="Minimum nDetected threshold (motifs.csv, default: 20)")
    args = parser.parse_args(argv)

    inject_metagenome(
        root_dir=args.root_dir,
        pkl_path=args.dict,
        output_dir=args.output_dir,
        circular=not args.linear,
        revcomp=not args.no_revcomp,
        no_fuzznuc=args.no_fuzznuc,
        keep_species=args.keep_species,
        min_fraction=args.min_fraction,
        min_detected=args.min_detected,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
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
            "Inject IPD/PW kinetic signals into all PBSIM3 species in a directory.\n\n"
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
            "  kinsim dictionary inject <fq.gz> <maf.gz> <ref.fna> <dict.pkl> <motifs> <out.bam>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pbsim3_dir",
                        help="Directory containing species subdirs or flat .fq.gz files")
    parser.add_argument("dict",
                        help="Trained kinetic dictionary (.pkl)")
    parser.add_argument("motifs",
                        help="Motifs: KinSim string (applied to all), PacBio .csv, "
                             "REBASE file, or per-species file ('species|motif_string' per line)")
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
