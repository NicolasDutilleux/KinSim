"""Reduce a bystrandified+aligned BAM to a HiFi-like input for ``kinsim_nn generate``.

Pipeline:
  - keep only ``/fwd`` reads (one record per ZMW carrying the original
    HiFi-forward sequence)
  - keep the alignment (flag, reference, position, CIGAR) — the trained
    generator was conditioned on the *reference* methylation context, so the
    multiprocess+precompute path needs each query position to be mappable
    to a ref position via the read's CIGAR
  - drop kinetics tags (``ip``, ``pw``, ``fi``, ``fp``, ``ri``, ``rp``) so
    the generator can inject fresh synthetic kinetics
  - drop alignment-derived tags (``MD``, ``NM``, ``AS``, ``XS``, ``SA``,
    ``cs``, ``ms``) so the downstream ``ccs-kinetics-bystrandify`` (which
    sees the unaligned output of ``kinsim_nn generate``) doesn't get
    confused by stale alignment metadata on unmapped reads
  - clean the ``@RG DS`` field: strip the ``Ipd:CodecV1=ip`` and
    ``PulseWidth:CodecV1=pw`` codec declarations that were added by the
    upstream bystrandify run. Our kinsim BAM carries fresh fi/fp/ri/rp
    (the raw HiFi codec); leaving the ip/pw codec declarations in @RG
    makes bystrandify silently discard every record with the misleading
    "has 0 PulseWidths" warning
  - drop the ``/fwd`` suffix from the read name so downstream tools see a
    HiFi-shaped read name

The output BAM stays aligned with the same ``@SQ`` header but a cleaned
``@RG``; ``kinsim_nn generate`` will compute fresh kinetics on the aligned
reads and emit an unaligned output (its ``--emit-unaligned`` default),
which feeds into the bystrandify → pbmm2 → ipdSummary chain.

Implementation note: the read is mutated in place (rather than rebuilt via
``set_tags``) because pysam refuses to round-trip ``B`` array tags through
``set_tags`` with explicit ``value_type='B'``.

CLI:
    python scripts/strip_bystrandified_to_hifi.py <input.bam> <output.bam>
"""
from __future__ import annotations

import argparse
import logging
import re
import sys

import pysam


log = logging.getLogger(__name__)

KINETICS_TAGS = ("ip", "pw", "fi", "fp", "ri", "rp")
ALIGNMENT_TAGS = ("MD", "NM", "AS", "XS", "SA", "cs", "ms")
TAGS_TO_DROP = KINETICS_TAGS + ALIGNMENT_TAGS


# Match ``Ipd:CodecV1=ip;`` and ``PulseWidth:CodecV1=pw;`` (trailing ``;``
# is optional — the field is semicolon-separated and the codec entry may
# be in the middle or at the end).
_DS_CODEC_RE = re.compile(r"(?:Ipd|PulseWidth):CodecV1=[a-zA-Z0-9]+;?")


def _clean_rg_ds(ds_value: str) -> str:
    """Remove the ip/pw codec declarations from an @RG DS field."""
    cleaned = _DS_CODEC_RE.sub("", ds_value)
    # Collapse accidental ";;" produced by the substitution.
    cleaned = re.sub(r";+", ";", cleaned)
    # Strip a leading/trailing semicolon if any.
    return cleaned.strip(";")


def _clean_header(header_dict: dict) -> dict:
    """Return a copy of the header dict with @RG DS codec declarations stripped."""
    out = dict(header_dict)
    if "RG" not in out:
        return out
    new_rgs = []
    for rg in out["RG"]:
        rg = dict(rg)
        if "DS" in rg:
            rg["DS"] = _clean_rg_ds(rg["DS"])
        new_rgs.append(rg)
    out["RG"] = new_rgs
    return out


def strip(input_bam: str, output_bam: str) -> tuple[int, int]:
    """Return (kept_fwd, skipped_other)."""
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as bam_in:
        header_dict = _clean_header(bam_in.header.to_dict())
        out_header = pysam.AlignmentHeader.from_dict(header_dict)
        with pysam.AlignmentFile(output_bam, "wb", header=out_header) as bam_out:
            kept = 0
            skipped = 0
            for read in bam_in.fetch(until_eof=True):
                if not read.query_name.endswith("/fwd"):
                    skipped += 1
                    continue
                # Drop the bystrandified "/fwd" suffix → HiFi-shaped name.
                read.query_name = read.query_name[:-4]
                # Drop kinetics + alignment-derived tags so the downstream
                # bystrandify sees a clean raw-HiFi-shaped read.
                for t in TAGS_TO_DROP:
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
            "and alignment-derived tags, clean the @RG DS codec declarations.\n"
            "KEEPS the alignment so `kinsim_nn generate` can run its\n"
            "multiprocess+precompute (mapped-reads) path."
        ),
    )
    ap.add_argument("input_bam")
    ap.add_argument("output_bam")
    args = ap.parse_args(argv)
    if args.input_bam == args.output_bam:
        log.error("input and output paths are the same — refusing to overwrite.")
        return 1
    log.info("Stripping bystrandified BAM: %s", args.input_bam)
    log.info("Output (aligned, /fwd-only, kinetics+alignment tags stripped, @RG DS cleaned): %s", args.output_bam)
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
