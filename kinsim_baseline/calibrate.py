"""calibrate: identify candidate-modification positions in real BAMs and
   compute per-kmer modified IPD distribution + IPD ratio.

Pipeline (matches user's spec):

    shards/         ──► build  ──► baseline_table.npz   (per-kmer baseline samples)
                                       │
    real BAMs ──┬─► calibrate ◄────────┘
                │
                ▼
        calibrated_table.npz
        (baseline + modified samples + per-kmer ratio)

Algorithm
---------

1. Load the baseline table (per-kmer (IPD, PW) samples from extract's
   CATEGORY_BASELINE rows — guaranteed unmodified).
2. Compute the per-kmer ``threshold_percentile`` (default 99th
   percentile) of the baseline IPD distribution. Vectorised
   ``np.nanquantile`` with zero-padding masked out.
3. Walk each BAM in the manifest. For every base position, compute the
   centred 11-mer's index and look up its threshold. If the observed IPD
   exceeds the threshold, the position is a **candidate modification**:
   add (observed_IPD, observed_PW) to that kmer's modified-sample bank
   (first-N wins, capped at ``--n-modified-per-kmer``).
4. Save an extended .npz table that contains the original baseline
   bank PLUS the modified bank PLUS a per-kmer IPD ratio
   (modified_mean / baseline_mean).

The threshold is a per-kmer property of the BASELINE distribution — not
a global threshold and not the 99th percentile of the BAM data itself.
This is the correct null-model framing: "anything beyond what unmodified
data does at this kmer is a candidate modification."

Output table can be consumed by :mod:`kinsim_baseline.generate` (future
work: at methylated motif positions, sample from the modified bank
instead of the baseline bank).

CLI::

    python -m kinsim_baseline calibrate BASELINE_TABLE_NPZ MANIFEST_CSV \\
        OUTPUT_NPZ [--threshold-percentile 0.99] \\
        [--n-modified-per-kmer 50]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from kinsim.utils.config import load_manifest
from kinsim.utils.encoding import K

from .build_table import _fill_batch
from .distribution import NUM_KMERS, KmerDistribution
from .generate import encode_sequence_to_kmers

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-kmer threshold computation
# ---------------------------------------------------------------------------


def compute_kmer_thresholds(
    baseline: KmerDistribution,
    percentile: float,
) -> np.ndarray:
    """Per-kmer ``percentile``-th IPD threshold, computed on the baseline bank.

    Zero-padded slots (where ``count[k] < n_per_kmer``) are masked out
    via NaN so they don't bias the quantile.

    Returns:
        ``(NUM_KMERS,)`` uint8. ``255`` for kmers with no baseline data
        (effectively "no threshold can be exceeded" → no positions
        flagged for those kmers).
    """
    log.info("Computing per-kmer baseline %g-percentile thresholds ...", percentile)
    # Mask zero padding (PacBio IPDs are >=1; 0 means "no data" / padding).
    ipd_float = baseline.ipd.astype(np.float32)
    mask = baseline.ipd > 0
    # Build an array where padded slots are NaN.
    masked = np.where(mask, ipd_float, np.nan)
    # nanquantile along axis=1 — vectorised across all 4M kmers.
    thresholds = np.nanquantile(masked, percentile, axis=1)
    # Where every slot was NaN (count == 0), nanquantile returns NaN → 255.
    thresholds = np.nan_to_num(thresholds, nan=255.0)
    return np.clip(thresholds, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# BAM walk: identify candidate-modification positions
# ---------------------------------------------------------------------------


def calibrate_from_bams(
    bam_paths: list[Path],
    baseline: KmerDistribution,
    threshold_percentile: float = 0.99,
    n_modified_per_kmer: int = 50,
    seed: int = 42,
) -> KmerDistribution:
    """Walk BAMs, accumulate per-kmer modified samples + ratio."""
    import pysam

    rng = np.random.default_rng(seed)

    bam_paths = list(bam_paths)
    if not bam_paths:
        log.error("no BAM paths supplied")
        sys.exit(1)
    log.info("Calibrating against %d BAMs", len(bam_paths))

    thresholds = compute_kmer_thresholds(baseline, threshold_percentile)
    log.info(
        "Thresholds populated: %d kmers have a finite threshold (median=%.1f)",
        int((thresholds < 255).sum()),
        float(np.median(thresholds[thresholds < 255])) if (thresholds < 255).any() else float("nan"),
    )

    # Modified bank — same layout as baseline but typically smaller per-kmer cap.
    modified_ipd = np.zeros((NUM_KMERS, n_modified_per_kmer), dtype=np.uint8)
    modified_pw = np.zeros((NUM_KMERS, n_modified_per_kmer), dtype=np.uint8)
    modified_count = np.zeros(NUM_KMERS, dtype=np.uint16)

    centre_offset = K // 2  # 5 for K=11
    n_reads_total = 0
    n_candidates_total = 0

    for bi, bam_path in enumerate(bam_paths, 1):
        if not Path(bam_path).is_file():
            log.warning("[%d/%d] missing BAM: %s — skip", bi, len(bam_paths), bam_path)
            continue
        log.info("[%d/%d] %s", bi, len(bam_paths), bam_path)
        n_reads = 0
        n_candidates = 0
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            for read in bam:
                if read.has_tag("ip"):
                    ipd_arr = np.asarray(read.get_tag("ip"), dtype=np.uint8)
                    pw_arr = np.asarray(read.get_tag("pw"), dtype=np.uint8)
                elif read.has_tag("fi"):
                    ipd_arr = np.asarray(read.get_tag("fi"), dtype=np.uint8)
                    pw_arr = np.asarray(read.get_tag("fp"), dtype=np.uint8)
                else:
                    continue
                seq = read.query_sequence
                if seq is None or len(seq) != ipd_arr.size or len(seq) < K:
                    continue

                kmer_ids = encode_sequence_to_kmers(seq)
                if kmer_ids.size == 0:
                    continue

                centre_pos = np.arange(kmer_ids.size, dtype=np.int64) + centre_offset
                centre_ipd = ipd_arr[centre_pos]
                centre_pw = pw_arr[centre_pos]

                # Per-position: is observed IPD > kmer's baseline threshold?
                kmer_thresholds = thresholds[kmer_ids]
                exceed = centre_ipd > kmer_thresholds
                if not exceed.any():
                    n_reads += 1
                    continue

                _fill_batch(
                    modified_ipd, modified_pw, modified_count, n_modified_per_kmer,
                    kmer_ids[exceed], centre_ipd[exceed], centre_pw[exceed],
                )
                n_candidates += int(exceed.sum())
                n_reads += 1
        log.info(
            "  → %d reads, %d candidate-modification positions, "
            "modified-coverage %.2f%% (saturated %d kmers)",
            n_reads, n_candidates,
            100.0 * (modified_count > 0).mean(),
            int((modified_count == n_modified_per_kmer).sum()),
        )
        n_reads_total += n_reads
        n_candidates_total += n_candidates

    log.info(
        "Calibration done: %d reads, %d candidates total | "
        "%d kmers have modified samples (%.2f%%)",
        n_reads_total, n_candidates_total,
        int((modified_count > 0).sum()),
        100.0 * (modified_count > 0).mean(),
    )

    return KmerDistribution(
        ipd=baseline.ipd,
        pw=baseline.pw,
        count=baseline.count,
        global_ipd=baseline._global_ipd,
        global_pw=baseline._global_pw,
        modified_ipd=modified_ipd,
        modified_pw=modified_pw,
        modified_count=modified_count,
        threshold_percentile=threshold_percentile,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline calibrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("baseline_npz", help=".npz from `kinsim_baseline build`")
    p.add_argument("manifest_csv", help="KinSim manifest CSV (BAMs to scan)")
    p.add_argument("output_npz", help="Output .npz with baseline + modified samples + ratio")
    p.add_argument("--threshold-percentile", type=float, default=0.99,
                   help="Per-kmer baseline percentile above which a BAM "
                        "position is flagged as candidate-modification "
                        "(default 0.99 = upper 1%%).")
    p.add_argument("--n-modified-per-kmer", type=int, default=50,
                   help="Max modified samples per kmer (default 50).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    log.info("Loading baseline: %s", args.baseline_npz)
    baseline = KmerDistribution.load(args.baseline_npz)
    log.info("Baseline coverage: %.2f%%", 100.0 * baseline.coverage())

    entries = load_manifest(args.manifest_csv)
    bam_paths = [Path(e.bam_path) for e in entries]
    log.info("Loaded %d BAM paths from manifest", len(bam_paths))

    calibrated = calibrate_from_bams(
        bam_paths,
        baseline,
        threshold_percentile=args.threshold_percentile,
        n_modified_per_kmer=args.n_modified_per_kmer,
        seed=args.seed,
    )
    calibrated.save(args.output_npz)

    # Compute and report headline ratio statistics.
    ratio = calibrated.ipd_ratio()
    valid = np.isfinite(ratio)
    log.info(
        "Saved: %s  |  per-kmer IPD ratio: %d kmers with both baseline "
        "and modified samples",
        args.output_npz, int(valid.sum()),
    )
    if valid.any():
        r = ratio[valid]
        log.info(
            "  ratio quantiles: p10=%.2f  median=%.2f  p90=%.2f  max=%.2f",
            float(np.quantile(r, 0.10)),
            float(np.quantile(r, 0.50)),
            float(np.quantile(r, 0.90)),
            float(r.max()),
        )


if __name__ == "__main__":
    main()
