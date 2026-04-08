"""Merge multiple BAM files into a single metagenomic BAM.

Concatenates unaligned BAM files produced by ``kinsim generate`` into one
BAM file that mimics a multiplexed PacBio sequencing run.  This is the
input expected by metagenomic analysis pipelines (nf-core/mag, etc.).

All input BAMs must be unaligned (flag=4) with PacBio kinetic tags
(fi/fp and optionally ri/rp).  Headers are merged: read groups from
each input are collected into the output header so downstream tools
can identify the source of each read.

CLI:
    kinsim merge-bam <input_dir_or_bams...> <output.bam>
    kinsim merge-bam species1.bam species2.bam species3.bam merged.bam
    kinsim merge-bam /path/to/bam_dir/ merged.bam
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pysam

log = logging.getLogger(__name__)


def merge_bams(
    input_paths: list[str],
    output_path: str,
) -> None:
    """Concatenate multiple BAM files into one.

    Args:
        input_paths: List of BAM file paths to merge.
        output_path: Path for the merged output BAM.
    """
    if not input_paths:
        log.error("No input BAM files provided.")
        sys.exit(1)

    # Collect read groups from all inputs, deduplicating by ID
    all_rgs: dict[str, dict] = {}
    for bam_path in input_paths:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            hd = bam.header.to_dict()
            for rg in hd.get("RG", []):
                rg_id = rg.get("ID", "unknown")
                if rg_id not in all_rgs:
                    all_rgs[rg_id] = rg

    # Build merged header
    out_dict = {
        "HD": {"VN": "1.6", "SO": "unknown"},
        "RG": list(all_rgs.values()) if all_rgs else [
            {"ID": "00000001", "PL": "PACBIO", "DS": "READTYPE=CCS"}
        ],
    }
    header_out = pysam.AlignmentHeader.from_dict(out_dict)

    n_reads = 0
    n_files = 0

    with pysam.AlignmentFile(output_path, "wb", header=header_out) as bam_out:
        for bam_path in input_paths:
            n_file_reads = 0
            with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam_in:
                for read in bam_in:
                    # Re-create segment under the new header
                    seg = pysam.AlignedSegment(header_out)
                    seg.query_name = read.query_name
                    seg.flag = read.flag
                    seg.query_sequence = read.query_sequence
                    seg.query_qualities = read.query_qualities

                    # Copy all tags (fi, fp, ri, rp, RG, etc.)
                    for tag, val in read.get_tags(with_value_type=True):
                        seg.set_tag(tag[0], tag[1], tag[2])

                    bam_out.write(seg)
                    n_file_reads += 1

            n_reads += n_file_reads
            n_files += 1
            log.info("  %s: %d reads", Path(bam_path).name, n_file_reads)

    log.info("Merged %d files, %d total reads → %s", n_files, n_reads, output_path)


def main(argv=None):
    """CLI entry point for merge-bam."""
    parser = argparse.ArgumentParser(
        prog="kinsim merge-bam",
        description="Merge multiple BAM files into a single metagenomic BAM.",
    )
    parser.add_argument(
        "inputs", nargs="+",
        help=(
            "Input BAM files or a single directory containing BAM files. "
            "The last argument is always the output BAM path."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args(argv)

    if len(args.inputs) < 2:
        parser.error("Need at least one input and one output path.")

    output_path = args.inputs[-1]
    input_args = args.inputs[:-1]

    # If single input is a directory, glob for BAMs
    if len(input_args) == 1 and os.path.isdir(input_args[0]):
        bam_dir = input_args[0]
        input_paths = sorted(
            str(p) for p in Path(bam_dir).glob("*.bam")
            if p.name != Path(output_path).name
        )
        if not input_paths:
            log.error("No .bam files found in: %s", bam_dir)
            sys.exit(1)
        log.info("Found %d BAM files in %s", len(input_paths), bam_dir)
    else:
        input_paths = input_args
        for p in input_paths:
            if not os.path.exists(p):
                log.error("Input BAM not found: %s", p)
                sys.exit(1)

    log.info("Merging %d BAM files → %s", len(input_paths), output_path)
    merge_bams(input_paths, output_path)
