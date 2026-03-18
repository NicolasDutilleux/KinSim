#!/usr/bin/env python3
"""Strip kinetic tags (fi, fp, ri, rp) from a PacBio BAM file.

Copies the input BAM to a new file, then removes the four PacBio kinetic tags
from every read in the copy.  The original BAM is never modified.

Usage:
    python strip_kinetics.py <input.bam> <output.bam>

Example:
    python strip_kinetics.py bc2036_real.bam bc2036_stripped.bam

The output BAM retains all other tags (RG, np, rq, zm, ...) and the full
alignment (or unmapped flag).  Only fi, fp, ri, rp are removed so that
kinsim generate can inject fresh synthetic kinetics.
"""

import argparse
import sys

import pysam

KINETIC_TAGS = ("fi", "fp", "ri", "rp")


def strip_kinetics(input_bam: str, output_bam: str) -> None:
    """Copy input_bam to output_bam, removing fi/fp/ri/rp tags from every read."""
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam_in:
        header = bam_in.header.to_dict()
        with pysam.AlignmentFile(output_bam, "wb", header=header) as bam_out:
            n_reads  = 0
            n_stripped = 0
            for read in bam_in:
                removed_any = False
                for tag in KINETIC_TAGS:
                    if read.has_tag(tag):
                        read.set_tag(tag, None)
                        removed_any = True
                bam_out.write(read)
                n_reads += 1
                if removed_any:
                    n_stripped += 1

    print(f"Done. {n_reads:,} reads written to {output_bam}")
    print(f"      {n_stripped:,} reads had kinetic tags stripped")
    print(f"      {n_reads - n_stripped:,} reads had no kinetic tags (passed through)")


def main():
    parser = argparse.ArgumentParser(
        prog="strip_kinetics.py",
        description=(
            "Copy a PacBio BAM and remove fi/fp/ri/rp kinetic tags from the copy.\n"
            "The original BAM is never touched."
        ),
    )
    parser.add_argument("input_bam",  help="Source BAM (read-only, not modified)")
    parser.add_argument("output_bam", help="Destination BAM without kinetic tags")
    args = parser.parse_args()

    if args.input_bam == args.output_bam:
        print("ERROR: input and output paths are the same — refusing to overwrite.", file=sys.stderr)
        sys.exit(1)

    print(f"Stripping kinetic tags from: {args.input_bam}")
    print(f"Writing stripped copy to:    {args.output_bam}")
    strip_kinetics(args.input_bam, args.output_bam)


if __name__ == "__main__":
    main()
