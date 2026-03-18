"""Strip kinetic tags (fi, fp, ri, rp) from a PacBio BAM file.

Copies the input BAM to a new file then removes the four PacBio kinetic tags
from every read in the copy.  The original BAM is never modified.

CLI:
    kinsim strip-kinetics <input.bam> <output.bam>
"""

import argparse
import logging
import sys

import pysam

log = logging.getLogger(__name__)

KINETIC_TAGS = ("fi", "fp", "ri", "rp")


def strip_kinetics(input_bam: str, output_bam: str) -> None:
    """Copy input_bam to output_bam, removing fi/fp/ri/rp tags from every read.

    All other tags (RG, np, rq, zm, ...) and alignment records are preserved
    unchanged.  Only the four PacBio kinetic tags are stripped so that
    ``kinsim generate`` can inject fresh synthetic kinetics.

    Args:
        input_bam:  Source BAM (never modified).
        output_bam: Destination BAM without fi/fp/ri/rp tags.
    """
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam_in:
        header = bam_in.header.to_dict()
        with pysam.AlignmentFile(output_bam, "wb", header=header) as bam_out:
            n_reads    = 0
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

    log.info("Done. %d reads written to %s", n_reads, output_bam)
    log.info("      %d reads had kinetic tags stripped", n_stripped)
    log.info("      %d reads had no kinetic tags (passed through)",
             n_reads - n_stripped)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="kinsim strip-kinetics",
        description=(
            "Copy a PacBio BAM and remove fi/fp/ri/rp kinetic tags from the copy.\n"
            "The original BAM is never touched.\n\n"
            "Use this before 'kinsim generate' when you want to replace real kinetics\n"
            "with synthetic ones while keeping the read sequences and alignments."
        ),
    )
    parser.add_argument("input_bam",  help="Source BAM (read-only, not modified)")
    parser.add_argument("output_bam", help="Destination BAM without kinetic tags")
    args = parser.parse_args(argv)

    if args.input_bam == args.output_bam:
        log.error("input and output paths are the same — refusing to overwrite.")
        sys.exit(1)

    log.info("Stripping kinetic tags from: %s", args.input_bam)
    log.info("Writing stripped copy to:    %s", args.output_bam)
    strip_kinetics(args.input_bam, args.output_bam)
