"""Count BAM-level base occurrences INSIDE vs OUTSIDE methylation motif sites.

Quantifies the class-imbalance behind why naive position-aggregated baselines
drown the methylation signal in unmethylated noise.

Algorithm
---------
1. Load reference FASTA + motif source.
2. Scan reference once → per-contig int8 array of ``meth_id`` per position
   (``METH_IDS['m6A'] = 1`` etc.).  Stored in a dict ``ref_meth_maps``.
3. Walk the BAM.  For each aligned read, vectorise the count:
   - map each aligned (query_pos, ref_pos) pair
   - check the base at query_pos and the meth_id at ref_pos
   - bump the corresponding counter
4. Print summary per meth type:
     in_motif    : observed bases at a candidate-methylation site
     unmodified  : observed bases of the same type NOT at a motif site
     ratio       : in_motif / (in_motif + unmodified)

Usage::

    python scripts/count_modified_positions.py BAM REF MOTIFS [--limit N]

``MOTIFS`` is anything that :func:`load_motif_string` accepts (KinSim motif
string, PacBio motifs.csv, combined CSV, REBASE file).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pysam

from kinsim.utils.config import setup_logging
from kinsim.utils.encoding import METH_IDS
from kinsim.utils.io import load_reference
from kinsim.utils.motifs import load_motif_string, parse_motifs, scan_sequence

log = logging.getLogger(__name__)


def build_ref_meth_maps(ref_seqs: dict, motif_string: str) -> dict[str, np.ndarray]:
    """Return ``{contig_name: int8 array of meth_id per ref position}``."""
    motifs = parse_motifs(motif_string, revcomp=True)
    out = {}
    for name, seq in ref_seqs.items():
        status = scan_sequence(seq, motifs)
        out[name] = np.asarray(status, dtype=np.int8)
        n_meth = int((status > 0).sum())
        log.info("  %-30s  %s bp  %s methylated positions (%.3f%%)",
                 name, f"{len(seq):,}", f"{n_meth:,}",
                 100 * n_meth / max(len(seq), 1))
    return out


def count_bam(bam_path: Path, ref_meth_maps: dict[str, np.ndarray],
              limit: int | None = None) -> tuple[dict, int]:
    """Walk BAM, return per-meth-type counts of ``in_motif`` vs ``unmodified``."""
    counts: dict[str, dict[str, int]] = {
        T: {"in_motif": 0, "unmodified": 0}
        for T in ("m6A", "m4C", "m5C")
    }

    m6A_id = METH_IDS["m6A"]
    m4C_id = METH_IDS["m4C"]
    m5C_id = METH_IDS["m5C"]

    A_ord, C_ord = ord("A"), ord("C")

    n_reads = 0
    n_used = 0
    t0 = time.time()
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False, threads=4) as bam:
        for read in bam:
            n_reads += 1
            if limit and n_used >= limit:
                break
            if read.is_unmapped or read.reference_name not in ref_meth_maps:
                continue
            seq = read.query_sequence
            if seq is None:
                continue

            ref_meth = ref_meth_maps[read.reference_name]

            # Aligned pairs as a numpy array — drops Nones (matches_only=True).
            pairs = read.get_aligned_pairs(matches_only=True)
            if not pairs:
                continue
            pairs_arr = np.asarray(pairs, dtype=np.int64)
            qpos = pairs_arr[:, 0]
            rpos = pairs_arr[:, 1]

            # Clip out-of-range ref positions (rare, defensive).
            in_range = rpos < ref_meth.size
            qpos = qpos[in_range]
            rpos = rpos[in_range]
            if qpos.size == 0:
                continue

            seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            base_at_q = seq_arr[qpos]
            meth_at_r = ref_meth[rpos]

            a_mask = base_at_q == A_ord
            c_mask = base_at_q == C_ord

            m6A_hit = a_mask & (meth_at_r == m6A_id)
            m4C_hit = c_mask & (meth_at_r == m4C_id)
            m5C_hit = c_mask & (meth_at_r == m5C_id)

            counts["m6A"]["in_motif"]   += int(m6A_hit.sum())
            counts["m6A"]["unmodified"] += int(a_mask.sum() - m6A_hit.sum())
            counts["m4C"]["in_motif"]   += int(m4C_hit.sum())
            counts["m5C"]["in_motif"]   += int(m5C_hit.sum())
            # Unmodified C = C and not (m4C or m5C site)
            unmodified_c = c_mask & ~(m4C_hit | m5C_hit)
            counts["m4C"]["unmodified"] += int(unmodified_c.sum())
            counts["m5C"]["unmodified"] += int(unmodified_c.sum())

            n_used += 1
            if n_used % 5000 == 0:
                log.info("    ... %s reads (%.1f s)", f"{n_used:,}", time.time() - t0)

    return counts, n_used


def print_summary(counts: dict, n_used: int) -> None:
    log.info("=" * 70)
    log.info("BAM modified vs unmodified position count summary")
    log.info("=" * 70)
    log.info("Reads processed: %s", f"{n_used:,}")
    log.info("")
    log.info("%-6s %15s %15s %15s %8s",
             "type", "in_motif", "unmodified", "total", "ratio_%")
    for T, c in counts.items():
        in_m = c["in_motif"]
        un   = c["unmodified"]
        tot  = in_m + un
        if tot == 0:
            log.info("%-6s %15s %15s %15s %8s",
                     T, "0", "0", "0", "—")
            continue
        ratio = 100 * in_m / tot
        log.info("%-6s %15s %15s %15s %8.4f",
                 T, f"{in_m:,}", f"{un:,}", f"{tot:,}", ratio)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python scripts/count_modified_positions.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("bam", help="Aligned BAM file (read positions used).")
    p.add_argument("ref", help="Reference FASTA matching the BAM.")
    p.add_argument("motifs", help="Motif source (KinSim string, motifs.csv, REBASE file).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N reads (default: all). Useful for quick test.")
    p.add_argument("--min-fraction", type=float, default=0.40,
                   help="Min fraction threshold passed to motif loader (default 0.40).")
    p.add_argument("--min-detected", type=int, default=20,
                   help="Min nDetected threshold passed to motif loader (default 20).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    log.info("Loading reference: %s", args.ref)
    ref_seqs = load_reference(args.ref)
    log.info("  %d contigs, %s bp total", len(ref_seqs),
             f"{sum(len(s) for s in ref_seqs.values()):,}")

    log.info("Loading motifs: %s", args.motifs)
    motif_string = load_motif_string(
        args.motifs, min_fraction=args.min_fraction, min_detected=args.min_detected,
    )
    if not motif_string:
        log.error("No motifs parsed — aborting.")
        return 1
    log.info("Motif string: %s", motif_string)

    log.info("Building reference meth maps (motif scan)...")
    ref_meth_maps = build_ref_meth_maps(ref_seqs, motif_string)

    log.info("Walking BAM: %s", args.bam)
    counts, n_used = count_bam(Path(args.bam), ref_meth_maps, limit=args.limit)

    print_summary(counts, n_used)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
