"""Strip a bystrandified+aligned BAM back to a HiFi-like unaligned BAM.

Keeps only ``/fwd`` reads (the strand carrying the original HiFi sequence),
drops the alignment, drops all kinetics tags (``ip``, ``pw``, ``fi``, ``fp``,
``ri``, ``rp``) plus alignment-derived tags (``MD``, ``NM``, ``AS``, ``XS``,
``SA``, ``cs``, ``ms``), and strips the ``/fwd`` suffix from the read name.

The output is a one-record-per-ZMW unmapped BAM that ``kinsim_nn generate``
can consume as if it were a fresh HiFi BAM with kinetics removed. We can then
re-inject synthetic kinetics with the trained generator and compare the
downstream ipdSummary chain against the real one.

CLI:
    python scripts/strip_bystrandified_to_hifi.py <input.bam> <output.bam>
"""
from __future__ import annotations

import argparse
import logging
import sys

import pysam


log = logging.getLogger(__name__)

TAGS_TO_STRIP = frozenset({
    "ip", "pw",                 # bystrandified kinetics
    "fi", "fp", "ri", "rp",     # raw HiFi kinetics (shouldn't be here, defensive)
    "MD", "NM", "AS", "XS",     # alignment-derived
    "SA", "cs", "ms",
})


def strip(input_bam: str, output_bam: str) -> tuple[int, int]:
    """Return (kept_fwd, skipped_other)."""
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam_in:
        header = bam_in.header.to_dict()
        header.pop("SQ", None)
        header_hd = header.setdefault("HD", {"VN": "1.6"})
        header_hd["SO"] = "unknown"
        out_header = pysam.AlignmentHeader.from_dict(header)

        with pysam.AlignmentFile(output_bam, "wb", header=out_header) as bam_out:
            kept = 0
            skipped = 0
            for read in bam_in.fetch(until_eof=True):
                if not read.query_name.endswith("/fwd"):
                    skipped += 1
                    continue
                new = pysam.AlignedSegment(out_header)
                new.query_name = read.query_name[:-4]
                new.query_sequence = read.get_forward_sequence()
                new.query_qualities = read.get_forward_qualities()
                new.flag = 4
                new.reference_id = -1
                new.reference_start = -1
                new.mapping_quality = 0
                new.cigar = []
                new.next_reference_id = -1
                new.next_reference_start = -1
                new.template_length = 0
                tags = [
                    (t, v, vt)
                    for (t, v, vt) in read.get_tags(with_value_type=True)
                    if t not in TAGS_TO_STRIP
                ]
                new.set_tags(tags)
                bam_out.write(new)
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
            "Convert a bystrandified+aligned BAM into a HiFi-like unaligned BAM\n"
            "for input to `kinsim_nn generate`. Keeps /fwd reads only, drops\n"
            "alignment and all kinetics tags."
        ),
    )
    ap.add_argument("input_bam")
    ap.add_argument("output_bam")
    args = ap.parse_args(argv)
    if args.input_bam == args.output_bam:
        log.error("input and output paths are the same — refusing to overwrite.")
        return 1
    log.info("Stripping bystrandified BAM: %s", args.input_bam)
    log.info("Output (unaligned HiFi-like): %s", args.output_bam)
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
