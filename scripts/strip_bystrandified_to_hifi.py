"""Reduce a bystrandified+aligned BAM to a HiFi-like input for ``kinsim_nn generate``.

Pipeline:
  - keep only ``/fwd`` reads (one record per ZMW carrying the original
    HiFi-forward sequence)
  - keep the alignment (flag, reference, position, CIGAR) — the trained
    generator was conditioned on the *reference* methylation context, so the
    multiprocess+precompute path needs each query position to be mappable
    to a ref position via the read's CIGAR
  - drop the kinetics tags (``ip``, ``pw``, ``fi``, ``fp``, ``ri``, ``rp``)
    so the generator can inject fresh synthetic kinetics
  - drop the ``/fwd`` suffix from the read name so downstream tools see a
    HiFi-shaped read name

The output BAM stays aligned with the same ``@SQ`` header; ``kinsim_nn
generate`` will compute fresh kinetics on the aligned reads and emit an
unaligned output (its ``--emit-unaligned`` default), which feeds into the
bystrandify → pbmm2 → ipdSummary chain unchanged.

Implementation note: the read is mutated in place (rather than rebuilt via
``set_tags``) because pysam refuses to round-trip ``B`` array tags through
``set_tags`` with explicit ``value_type='B'``.

CLI:
    python scripts/strip_bystrandified_to_hifi.py <input.bam> <output.bam>
"""
from __future__ import annotations

import argparse
import logging
import sys

import pysam


log = logging.getLogger(__name__)

KINETICS_TAGS = ("ip", "pw", "fi", "fp", "ri", "rp")


def strip(input_bam: str, output_bam: str) -> tuple[int, int]:
    """Return (kept_fwd, skipped_other)."""
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam_in:
        # Keep the original header verbatim: alignment + @SQ entries stay.
        with pysam.AlignmentFile(output_bam, "wb", header=bam_in.header) as bam_out:
            kept = 0
            skipped = 0
            for read in bam_in.fetch(until_eof=True):
                if not read.query_name.endswith("/fwd"):
                    skipped += 1
                    continue
                # Drop the bystrandified "/fwd" suffix → HiFi-shaped name.
                read.query_name = read.query_name[:-4]
                # Drop kinetics tags so the generator writes fresh ones.
                for t in KINETICS_TAGS:
                    if read.has_tag(t):
                        read.set_tag(t, None)
                bam_out.write(read)
                kept += 1
                if kept % 50000 == 0:
                    log.info("  %d /fwd reads written", kept)
    return kept, skipped


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    ap = argparse.ArgumentParser(
        prog="python scripts/strip_bystrandified_to_hifi.py",
        description=(
            "Reduce a bystrandified+aligned BAM to /fwd reads only, drop kinetics\n"
            "tags, KEEP alignment. Output is the input expected by `kinsim_nn\n"
            "generate` running in the multiprocess+precompute (mapped-reads) path."
        ),
    )
    ap.add_argument("input_bam")
    ap.add_argument("output_bam")
    args = ap.parse_args(argv)
    if args.input_bam == args.output_bam:
        log.error("input and output paths are the same — refusing to overwrite.")
        return 1
    log.info("Stripping bystrandified BAM: %s", args.input_bam)
    log.info("Output (aligned, /fwd-only, kinetics-stripped): %s", args.output_bam)
    kept, skipped = strip(args.input_bam, args.output_bam)
    log.info("Done. %d /fwd reads kept, %d other reads skipped", kept, skipped)
    if kept == 0:
        log.error("No /fwd reads found — was this BAM produced by bystrandify?")
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
