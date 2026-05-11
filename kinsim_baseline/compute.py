"""Per-(meth_type, offset) IPD/PW distribution computation.

Algorithm
---------
Single-pass walk over the manifest's BAMs. For each read:

  1. Sequence array ``seq``, IPD ``ipd``, PW ``pw`` (uint8 in [0, 255]).
  2. For each meth type T declared in ``kinsim_config.yaml``
     (``modified_base[T]`` and ``signal_offsets[T]``):
        - Find every position ``p`` where ``seq[p] == modified_base[T]``.
        - For every offset ``k`` in ``signal_offsets[T]``:
            * Look at ``ipd[p + k]`` and ``pw[p + k]``.
            * Add into the running 256-bin histogram for ``(T, k)``.

So for each ``(T, k)`` bucket we accumulate a **population** of IPD and
PW values at the affected positions across the whole corpus. Because
most A's (or C's) in real DNA are not methylated, the bulk of the
histogram is unmodified kinetics — that's the baseline distribution.
The high tail is the modified subset.

Outputs (written into ``out_dir/``)
------------------------------------
    baseline_hist.tsv      long-form, columns:
                           meth_type, offset, modified_base, metric, bin, count
                           (metric ∈ {IPD, PW}; bin ∈ [0..255])
    baseline_summary.tsv   per-(T, k) summary stats:
                           n, ipd_mean, ipd_p50/p95/p99,
                           pw_mean,  pw_p50/p95/p99,
                           n_above, ipd_mean_above, ipd_ratio,
                           pw_mean_above,  pw_ratio,
                           threshold (× mean for the "modified" cut)
    baseline.json          full histograms (per-(T, k) × {IPD,PW} → 256-bin list)
    run_info.json          inputs, timestamps, n_reads per BAM
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from kinsim.utils.config import load_kinsim_config, load_manifest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signature loading from kinsim_config.yaml
# ---------------------------------------------------------------------------


def load_signatures() -> dict:
    """Read ``kinetic_signatures`` from ``kinsim_config.yaml``.

    Returns a dict keyed by meth_type, each entry containing
    ``modified_base`` (single uppercase char) and ``signal_offsets``
    (list of int). Skips entries missing either field with a WARNING.
    """
    cfg = load_kinsim_config()
    raw = cfg.get("kinetic_signatures", {}) or {}
    out: dict[str, dict] = {}
    for mtype, info in raw.items():
        mb = info.get("modified_base")
        offsets = info.get("signal_offsets")
        if not mb or not offsets:
            log.warning(
                "kinetic_signatures.%s missing modified_base/signal_offsets — skipping",
                mtype,
            )
            continue
        out[mtype] = {
            "modified_base": str(mb).upper()[:1],
            "signal_offsets": [int(k) for k in offsets],
        }
    if not out:
        log.error("No usable entries in kinetic_signatures — aborting.")
        sys.exit(1)
    log.info("Loaded %d meth types from kinsim_config.yaml:", len(out))
    for T, info in out.items():
        log.info("  %s: modified_base=%s  signal_offsets=%s",
                 T, info["modified_base"], info["signal_offsets"])
    return out


# ---------------------------------------------------------------------------
# Per-read accumulation
# ---------------------------------------------------------------------------


def _read_kinetics(read) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract (seq_bytes, ipd, pw) from a pysam read, or ``None`` if invalid.

    Bystrandified BAMs use ``ip``/``pw``; raw HiFi BAMs use ``fi``/``fp``.
    """
    if read.has_tag("ip"):
        ipd = np.asarray(read.get_tag("ip"), dtype=np.uint8)
        pw = np.asarray(read.get_tag("pw"), dtype=np.uint8)
    elif read.has_tag("fi"):
        ipd = np.asarray(read.get_tag("fi"), dtype=np.uint8)
        pw = np.asarray(read.get_tag("fp"), dtype=np.uint8)
    else:
        return None
    seq = read.query_sequence
    if seq is None or len(seq) != ipd.size:
        return None
    seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return seq_arr, ipd, pw


def _accumulate_read(
    seq_arr: np.ndarray,
    ipd: np.ndarray,
    pw: np.ndarray,
    signatures: dict,
    hist_ipd: dict,
    hist_pw: dict,
) -> None:
    """For each (T, k), bincount IPD/PW at positions p+k where seq[p]==target."""
    L = seq_arr.size
    for T, info in signatures.items():
        target_byte = ord(info["modified_base"])
        positions = np.where(seq_arr == target_byte)[0]
        if positions.size == 0:
            continue
        for k in info["signal_offsets"]:
            tgt = positions + k
            valid = (tgt >= 0) & (tgt < L)
            tgt = tgt[valid]
            if tgt.size == 0:
                continue
            hist_ipd[(T, k)] += np.bincount(ipd[tgt], minlength=256).astype(np.int64)
            hist_pw[(T, k)] += np.bincount(pw[tgt], minlength=256).astype(np.int64)


# ---------------------------------------------------------------------------
# Single-pass driver
# ---------------------------------------------------------------------------


def compute_histograms(
    bam_paths: list[Path],
    progress_every: int = 50_000,
) -> tuple[dict, dict, dict, dict]:
    """Single pass: build per-(T, k) IPD and PW histograms.

    Returns
    -------
    signatures : dict from kinsim_config.yaml (T -> {modified_base, signal_offsets})
    hist_ipd   : dict (T, k) -> int64[256]
    hist_pw    : dict (T, k) -> int64[256]
    per_bam    : dict bam_path -> {"n_reads", "elapsed_s"}
    """
    import pysam  # heavy import here so the package import doesn't pay it

    signatures = load_signatures()

    hist_ipd: dict = defaultdict(lambda: np.zeros(256, dtype=np.int64))
    hist_pw: dict = defaultdict(lambda: np.zeros(256, dtype=np.int64))
    per_bam: dict = {}

    log.info("Single-pass histogram walk across %d BAMs", len(bam_paths))
    for bi, bam_path in enumerate(bam_paths, 1):
        bam_path = Path(bam_path)
        if not bam_path.is_file():
            log.warning("[%d/%d] missing BAM: %s — skip", bi, len(bam_paths), bam_path)
            per_bam[str(bam_path)] = {"n_reads": 0, "elapsed_s": 0.0, "skipped": True}
            continue
        log.info("[%d/%d] %s", bi, len(bam_paths), bam_path)
        t0 = time.time()
        n_reads = 0
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            for read in bam:
                kin = _read_kinetics(read)
                if kin is None:
                    continue
                seq_arr, ipd, pw = kin
                _accumulate_read(seq_arr, ipd, pw, signatures, hist_ipd, hist_pw)
                n_reads += 1
                if n_reads % progress_every == 0:
                    log.info("    ... %d reads", n_reads)
        dt = time.time() - t0
        log.info("    → %d reads in %.1f s", n_reads, dt)
        per_bam[str(bam_path)] = {"n_reads": n_reads, "elapsed_s": round(dt, 2)}

    return signatures, dict(hist_ipd), dict(hist_pw), per_bam


# ---------------------------------------------------------------------------
# Histogram → summary stats
# ---------------------------------------------------------------------------


def _hist_stats(h: np.ndarray) -> dict:
    """n, mean, p50, p95, p99 from a 256-bin histogram. ``None`` if empty."""
    n = int(h.sum())
    if n == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    bins = np.arange(256, dtype=np.float64)
    mean = float((bins * h).sum() / n)
    cum = np.cumsum(h)
    p50 = int(np.searchsorted(cum, n * 0.50))
    p95 = int(np.searchsorted(cum, n * 0.95))
    p99 = int(np.searchsorted(cum, n * 0.99))
    return {"n": n, "mean": mean, "p50": p50, "p95": p95, "p99": p99}


def _hist_above(h: np.ndarray, cutoff: float) -> dict:
    """n and mean for the part of the histogram with bin > cutoff."""
    bins = np.arange(256, dtype=np.float64)
    mask = bins > cutoff
    n = int(h[mask].sum())
    if n == 0:
        return {"n": 0, "mean": None}
    mean = float((bins[mask] * h[mask]).sum() / n)
    return {"n": n, "mean": mean}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _fmt(x, fmt="%.4f"):
    return fmt % x if x is not None else "NA"


def write_hist_tsv(hist_ipd: dict, hist_pw: dict, signatures: dict, path: Path) -> None:
    """Long-form histogram TSV: one row per (T, k, metric, bin)."""
    cols = ["meth_type", "offset", "modified_base", "metric", "bin", "count"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for T, info in signatures.items():
            mb = info["modified_base"]
            for k in info["signal_offsets"]:
                for metric, hist in (("IPD", hist_ipd), ("PW", hist_pw)):
                    h = hist.get((T, k))
                    if h is None:
                        continue
                    for b in range(256):
                        if h[b] == 0:
                            continue
                        f.write(f"{T}\t{k:+d}\t{mb}\t{metric}\t{b}\t{int(h[b])}\n")


def write_summary_tsv(
    hist_ipd: dict, hist_pw: dict, signatures: dict, path: Path, threshold: float = 1.3,
) -> None:
    """Per-(T, k) summary: baseline (full histogram) + modified (above cutoff) + ratio."""
    cols = [
        "meth_type", "offset", "modified_base",
        "n",
        "ipd_mean", "ipd_p50", "ipd_p95", "ipd_p99",
        "pw_mean",  "pw_p50",  "pw_p95",  "pw_p99",
        "threshold", "n_above",
        "ipd_mean_above", "ipd_ratio",
        "pw_mean_above",  "pw_ratio",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for T, info in signatures.items():
            for k in info["signal_offsets"]:
                hi = hist_ipd.get((T, k), np.zeros(256, dtype=np.int64))
                hp = hist_pw.get((T, k), np.zeros(256, dtype=np.int64))
                s_i, s_p = _hist_stats(hi), _hist_stats(hp)
                if s_i["mean"] is None:
                    cutoff = float("inf")
                    above_i = {"n": 0, "mean": None}
                    above_p = {"n": 0, "mean": None}
                    ipd_ratio = None
                    pw_ratio = None
                else:
                    cutoff = threshold * s_i["mean"]
                    above_i = _hist_above(hi, cutoff)
                    above_p = _hist_above(hp, cutoff)
                    ipd_ratio = (above_i["mean"] / s_i["mean"]) if above_i["mean"] else None
                    pw_ratio = (above_p["mean"] / s_p["mean"]) if (above_p["mean"] and s_p["mean"]) else None
                row = [
                    T, f"{k:+d}", info["modified_base"],
                    str(s_i["n"]),
                    _fmt(s_i["mean"], "%.3f"), _fmt(s_i["p50"], "%d"),
                    _fmt(s_i["p95"], "%d"),   _fmt(s_i["p99"], "%d"),
                    _fmt(s_p["mean"], "%.3f"), _fmt(s_p["p50"], "%d"),
                    _fmt(s_p["p95"], "%d"),   _fmt(s_p["p99"], "%d"),
                    _fmt(threshold, "%.3f"), str(above_i["n"]),
                    _fmt(above_i["mean"], "%.3f"), _fmt(ipd_ratio, "%.3f"),
                    _fmt(above_p["mean"], "%.3f"), _fmt(pw_ratio,  "%.3f"),
                ]
                f.write("\t".join(row) + "\n")


def write_json(
    hist_ipd: dict, hist_pw: dict, signatures: dict, path: Path,
) -> None:
    """Full histograms in JSON, keyed ``"T@+k"`` for IPD and PW."""
    out = {"signatures": signatures, "ipd": {}, "pw": {}}
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            hi = hist_ipd.get((T, k))
            hp = hist_pw.get((T, k))
            if hi is not None:
                out["ipd"][key] = hi.tolist()
            if hp is not None:
                out["pw"][key] = hp.tolist()
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


def log_summary(hist_ipd: dict, signatures: dict, threshold: float) -> None:
    log.info("=" * 72)
    log.info("PER-(meth_type, offset) IPD DISTRIBUTION SUMMARY  (threshold=%.2f)", threshold)
    log.info("=" * 72)
    log.info("%-6s %5s  %-12s %7s %5s %5s %5s  %-10s %7s  %5s",
             "meth", "off", "n", "ipd_mean", "p50", "p95", "p99",
             "n_above", "mean_a", "ratio")
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            h = hist_ipd.get((T, k), np.zeros(256, dtype=np.int64))
            s = _hist_stats(h)
            if s["mean"] is None:
                log.info("  %-6s %+5d  NO DATA", T, k)
                continue
            cutoff = threshold * s["mean"]
            ab = _hist_above(h, cutoff)
            ratio = (ab["mean"] / s["mean"]) if ab["mean"] else None
            log.info(
                "%-6s %+5d  %-12d %7.3f %5d %5d %5d  %-10d %7s  %5s",
                T, k, s["n"], s["mean"], s["p50"], s["p95"], s["p99"],
                ab["n"], _fmt(ab["mean"], "%.3f"), _fmt(ratio, "%.3f"),
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline compute",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("manifest_csv", help="KinSim manifest CSV (sample_id,bam_path,...)")
    p.add_argument("output_dir",
                   help="Output directory; baseline_hist.tsv / baseline_summary.tsv / "
                        "baseline.json / run_info.json land inside.")
    p.add_argument("--threshold", type=float, default=1.3,
                   help="Multiplier × baseline mean IPD used in the summary's "
                        "'above-cutoff' / IPD-ratio columns (default 1.3).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(args.manifest_csv)
    bam_paths = [Path(e.bam_path) for e in entries]
    log.info("Loaded %d BAM paths from manifest %s", len(bam_paths), args.manifest_csv)

    t0 = time.time()
    signatures, hist_ipd, hist_pw, per_bam = compute_histograms(bam_paths)
    elapsed = time.time() - t0

    hist_path    = out_dir / "baseline_hist.tsv"
    summary_path = out_dir / "baseline_summary.tsv"
    json_path    = out_dir / "baseline.json"
    info_path    = out_dir / "run_info.json"

    write_hist_tsv(hist_ipd, hist_pw, signatures, hist_path)
    write_summary_tsv(hist_ipd, hist_pw, signatures, summary_path, threshold=args.threshold)
    write_json(hist_ipd, hist_pw, signatures, json_path)
    with open(info_path, "w") as f:
        json.dump({
            "manifest_csv": str(args.manifest_csv),
            "threshold":    args.threshold,
            "elapsed_s":    round(elapsed, 2),
            "per_bam":      per_bam,
            "signatures":   signatures,
        }, f, indent=2)

    log.info("Saved: %s", hist_path)
    log.info("Saved: %s", summary_path)
    log.info("Saved: %s", json_path)
    log.info("Saved: %s", info_path)
    log_summary(hist_ipd, signatures, threshold=args.threshold)


if __name__ == "__main__":
    main()
