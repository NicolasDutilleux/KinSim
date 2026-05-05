"""Refine a KinSim master .pkl by filtering false-positive slowed samples.

Input format: ``dict[kmer_id (int)] -> ndarray(N, 36)`` produced by
``kinsim extract`` + ``kinsim merge``. Col 35 carries the CATEGORY enum
(0=baseline, 1=slowed, 2=near_meth — see kinsim.utils.sample_layout).

Algorithm: pool the IPD of all CATEGORY_BASELINE samples, compute the
``secondary_percentile`` percentile of the per-kmer baseline mean
distribution, drop CATEGORY_SLOWED samples whose IPD falls below it.
CATEGORY_BASELINE and CATEGORY_NEAR_METH pass through unchanged.

Why per-kmer baseline mean (not per-sample): PacBio raw IPD has a heavy
right tail per sample (natural polymerase pauses on unmodified DNA).
Per-sample p95 catches the noise floor, not the real outliers, and
wipes weak signals like m4C / m5C. Averaging baseline samples per kmer
first (CLT) gives a tight distribution whose p95 reflects truly
anomalous kmers.

Why filter at all: a motif that the regex flagged at extract time but
that the polymerase did not actually slow down produces SLOWED samples
sitting at baseline IPD. They survive the categorical assignment but
fail this quantile filter, so we drop them as motif false positives.

Usage:
    kinsim refine in.pkl out.pkl
    kinsim refine in.pkl out.pkl --secondary-percentile 90  # more permissive
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# Min baseline samples per kmer required for that kmer to contribute a
# mean to the threshold computation. Below this the per-kmer mean is too
# noisy to trust.
MIN_BASELINE_PER_KMER_FOR_THRESHOLD = 5


def slowed_split(
    data: dict,
    secondary_pct: float,
) -> tuple[dict, dict]:
    """Drop CATEGORY_SLOWED samples whose IPD falls below the
    ``secondary_pct`` percentile of the per-kmer baseline mean
    distribution.

    Args:
        data: dict[kmer_id -> ndarray(N, 36)]. Col 35 carries the
            category enum (0=baseline, 1=slowed, 2=near_meth).
        secondary_pct: percentile of the per-kmer baseline mean used as
            the lower threshold for slowed samples (typically 95).

    Returns:
        (new_data, stats) where new_data is a fresh dict with the same
        kmer keys and surviving rows, and stats is a small dict with
        in/out counts per category, the threshold used, and the
        baseline-distribution summary statistics.
    """
    from .utils.sample_layout import (
        CATEGORY_BASELINE,
        CATEGORY_NEAR_METH,
        CATEGORY_SLOWED,
        COL_CATEGORY,
        COL_IPD,
    )

    # 1. Threshold = percentile of per-kmer baseline means.
    kmer_baseline_means: list = []
    pooled_for_stats: list = []  # for diagnostics only
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        m = cats == CATEGORY_BASELINE
        n_b = int(m.sum())
        if n_b == 0:
            continue
        ipds = arr[m, COL_IPD]
        pooled_for_stats.append(ipds)
        if n_b >= MIN_BASELINE_PER_KMER_FOR_THRESHOLD:
            kmer_baseline_means.append(float(ipds.mean()))

    if pooled_for_stats and kmer_baseline_means:
        pooled = np.concatenate(pooled_for_stats)
        kmer_means = np.array(kmer_baseline_means, dtype=np.float32)
        threshold = float(np.percentile(kmer_means, secondary_pct))

        log.info(
            "[refine] baseline IPD per-sample:  n=%d  mean=%.2f  std=%.2f",
            len(pooled),
            float(pooled.mean()),
            float(pooled.std()),
        )
        log.info(
            "[refine]   per-sample quantiles:    "
            "p5=%.0f  p50=%.0f  p75=%.0f  p90=%.0f  p95=%.0f  p99=%.0f  max=%.0f",
            float(np.percentile(pooled, 5)),
            float(np.percentile(pooled, 50)),
            float(np.percentile(pooled, 75)),
            float(np.percentile(pooled, 90)),
            float(np.percentile(pooled, 95)),
            float(np.percentile(pooled, 99)),
            float(pooled.max()),
        )
        log.info(
            "[refine] baseline PER-KMER MEAN: n_kmers=%d (>= %d samples each)  mean=%.2f  std=%.2f",
            len(kmer_means),
            MIN_BASELINE_PER_KMER_FOR_THRESHOLD,
            float(kmer_means.mean()),
            float(kmer_means.std()),
        )
        log.info(
            "[refine]   per-kmer-mean quantiles: "
            "p5=%.1f  p50=%.1f  p75=%.1f  p90=%.1f  p95=%.1f  p99=%.1f  max=%.1f",
            float(np.percentile(kmer_means, 5)),
            float(np.percentile(kmer_means, 50)),
            float(np.percentile(kmer_means, 75)),
            float(np.percentile(kmer_means, 90)),
            float(np.percentile(kmer_means, 95)),
            float(np.percentile(kmer_means, 99)),
            float(kmer_means.max()),
        )
        bins = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256]
        h, _ = np.histogram(kmer_means, bins=bins)
        total = max(len(kmer_means), 1)
        log.info("[refine] per-kmer-mean histogram (bin -> count, %%):")
        for i in range(len(h)):
            marker = ""
            if bins[i] <= threshold < bins[i + 1]:
                marker = "   <-- threshold here"
            log.info(
                "    [%3d-%3d): %12d  (%5.2f%%)%s",
                bins[i],
                bins[i + 1],
                int(h[i]),
                100.0 * h[i] / total,
                marker,
            )
        log.info(
            "[refine] threshold = p%g(baseline PER-KMER MEAN) = %.2f  "
            "(slowed samples below this IPD are dropped)",
            secondary_pct,
            threshold,
        )
    else:
        threshold = 0.0
        log.warning("[refine] no baseline samples — threshold=0 (no FP filter)")

    # 2. Filter slowed by IPD >= threshold; baseline + near_meth pass through.
    new_data: dict = {}
    n_baseline_in = n_baseline_out = 0
    n_near_in = n_near_out = 0
    n_slowed_in = n_slowed_kept = n_slowed_dropped = 0
    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] <= COL_CATEGORY:
            continue
        cats = arr[:, COL_CATEGORY].astype(np.int8)
        base_m = cats == CATEGORY_BASELINE
        slow_m = cats == CATEGORY_SLOWED
        near_m = cats == CATEGORY_NEAR_METH
        n_baseline_in += int(base_m.sum())
        n_near_in += int(near_m.sum())
        n_slowed_in += int(slow_m.sum())
        slow_keep_mask = slow_m & (arr[:, COL_IPD] >= threshold)
        n_slowed_kept += int(slow_keep_mask.sum())
        n_slowed_dropped += int(slow_m.sum() - slow_keep_mask.sum())
        keep_rows = base_m | near_m | slow_keep_mask
        if keep_rows.any():
            new_data[int(kid)] = arr[keep_rows].copy()
            n_baseline_out += int(base_m.sum())
            n_near_out += int(near_m.sum())

    log.info("[refine] baseline:  %d in -> %d kept (pass-through)", n_baseline_in, n_baseline_out)
    log.info("[refine] near_meth: %d in -> %d kept (pass-through)", n_near_in, n_near_out)
    log.info(
        "[refine] slowed:    %d in -> %d kept, %d dropped (IPD < %.2f)  survival = %.2f%%",
        n_slowed_in,
        n_slowed_kept,
        n_slowed_dropped,
        threshold,
        100.0 * n_slowed_kept / max(n_slowed_in, 1),
    )

    # Diagnostic distributions of the kept slowed and the near_meth pool.
    slowed_kept_ipds: list = []
    near_ipds: list = []
    for _kid, arr in new_data.items():
        if not isinstance(arr, np.ndarray) or arr.shape[1] <= COL_CATEGORY:
            continue
        cats_n = arr[:, COL_CATEGORY].astype(np.int8)
        slowed_kept_ipds.append(arr[cats_n == CATEGORY_SLOWED, COL_IPD])
        near_ipds.append(arr[cats_n == CATEGORY_NEAR_METH, COL_IPD])
    if slowed_kept_ipds:
        sp = np.concatenate(slowed_kept_ipds)
        if len(sp) > 0:
            log.info(
                "[refine] slowed kept quantiles:  p5=%.0f  p50=%.0f  p95=%.0f  max=%.0f  mean=%.2f",
                float(np.percentile(sp, 5)),
                float(np.percentile(sp, 50)),
                float(np.percentile(sp, 95)),
                float(sp.max()),
                float(sp.mean()),
            )
    if near_ipds:
        npa = np.concatenate(near_ipds)
        if len(npa) > 0:
            log.info(
                "[refine] near_meth quantiles:    "
                "p5=%.0f  p50=%.0f  p95=%.0f  max=%.0f  mean=%.2f  "
                "(should look baseline-like)",
                float(np.percentile(npa, 5)),
                float(np.percentile(npa, 50)),
                float(np.percentile(npa, 95)),
                float(npa.max()),
                float(npa.mean()),
            )

    stats = {
        "secondary_percentile": secondary_pct,
        "threshold": threshold,
        "n_baseline_in": n_baseline_in,
        "n_baseline_out": n_baseline_out,
        "n_near_in": n_near_in,
        "n_near_out": n_near_out,
        "n_slowed_in": n_slowed_in,
        "n_slowed_kept": n_slowed_kept,
        "n_slowed_dropped": n_slowed_dropped,
    }
    return new_data, stats


def refine_pkl(
    in_path: Path,
    out_path: Path,
    secondary_percentile: float | None = None,
) -> dict:
    """Load a master .pkl, run ``slowed_split``, write the refined pkl.

    ``secondary_percentile`` defaults to ``refine.slowed_split.secondary_percentile``
    in ``kinsim_config.yaml`` (typically 95).
    """
    log.info("Loading: %s  (%.2f GB)", in_path, in_path.stat().st_size / 1e9)
    with open(in_path, "rb") as f:
        data = pickle.load(f)
    orig_meta = data.pop("__meta__", None)

    if secondary_percentile is None:
        from .utils.config import load_kinsim_config

        cfg = load_kinsim_config()
        secondary_percentile = float(
            ((cfg.get("refine") or {}).get("slowed_split") or {}).get("secondary_percentile", 95.0)
        )
    log.info("Refine: secondary_percentile = %g", secondary_percentile)

    int_keyed = {k: v for k, v in data.items() if isinstance(k, (int, np.integer))}
    if not int_keyed:
        log.error("No int-keyed data found — input is not a kinsim extract pkl.")
        sys.exit(1)

    # Fail fast on the old 36-col layout (no parent-meth columns). Re-extract
    # is the right answer; analyze + per-meth plots depend on cols 36/37.
    from .utils.sample_layout import SAMPLE_NCOLS

    sample_arr = next(iter(int_keyed.values()))
    if sample_arr.shape[1] < SAMPLE_NCOLS:
        log.error(
            "Input pkl uses an obsolete %d-col layout; current layout is %d cols "
            "(adds PARENT_METH at col 36, PARENT_OFFSET at col 37). "
            "Re-run `kinsim extract` to regenerate.",
            sample_arr.shape[1],
            SAMPLE_NCOLS,
        )
        sys.exit(1)

    new_data, stats = slowed_split(int_keyed, secondary_percentile)

    new_data["__meta__"] = {
        "refined_from": str(in_path),
        "method": "p95_per_kmer_baseline_mean",
        "stats": stats,
        "original_meta": orig_meta,
    }
    log.info("Writing: %s", out_path)
    with open(out_path, "wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return stats


def main(argv=None):
    from .utils.config import setup_logging

    ap = argparse.ArgumentParser(
        prog="kinsim refine",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input_pkl", help="Input master .pkl from `kinsim merge`")
    ap.add_argument("output_pkl", help="Output refined .pkl")
    ap.add_argument(
        "--secondary-percentile",
        type=float,
        default=None,
        help="Percentile of per-kmer baseline mean used as the "
        "lower threshold for slowed samples. Overrides "
        "kinsim_config.yaml refine.slowed_split.secondary_percentile.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)

    in_p = Path(args.input_pkl)
    out_p = Path(args.output_pkl)
    if not in_p.exists():
        print(f"ERROR: {in_p} not found", file=sys.stderr)
        sys.exit(1)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    refine_pkl(in_p, out_p, secondary_percentile=args.secondary_percentile)


if __name__ == "__main__":
    main()
