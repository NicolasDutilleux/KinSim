"""Per-(meth_type, offset) IPD/PW baseline + modified-ratio computation.

See module docstring (``kinsim_baseline.__init__``) for the algorithm.
This file implements the two-pass walk over a manifest's BAMs.

Output schema (TSV columns)
---------------------------
    meth_type        e.g. m6A
    offset           signed int, e.g. +0, +5
    modified_base    target base from YAML (A / C / ...)
    baseline_n       count of unmodified samples (positions p+k where
                     read[p] == modified_base, regardless of methylation)
    baseline_ipd     mean IPD over baseline_n
    baseline_pw      mean PW over baseline_n
    modified_n       count of "above-threshold" samples (candidate methylated)
    modified_ipd     mean IPD over modified_n
    modified_pw      mean PW over modified_n
    ipd_ratio        modified_ipd / baseline_ipd (NaN if either is missing)
    pw_ratio         modified_pw / baseline_pw
    threshold        threshold multiplier used in pass 2 (e.g. 1.3)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
            "signal_offsets": list(int(k) for k in offsets),
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
# Per-read accumulation primitives (vectorised)
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


def _per_read_accumulate_baseline(
    seq_arr: np.ndarray,
    ipd: np.ndarray,
    pw: np.ndarray,
    signatures: dict,
    sums_ipd: dict,
    sums_pw: dict,
    counts: dict,
) -> None:
    """Pass 1: for each (T, k) bucket, sum ipd[p+k]+pw[p+k] over positions
    p where seq[p] == modified_base[T].
    """
    L = seq_arr.size
    for T, info in signatures.items():
        target_byte = ord(info["modified_base"])
        positions = np.where(seq_arr == target_byte)[0]
        if positions.size == 0:
            continue
        for k in info["signal_offsets"]:
            target_pos = positions + k
            valid = (target_pos >= 0) & (target_pos < L)
            target_pos = target_pos[valid]
            if target_pos.size == 0:
                continue
            sums_ipd[(T, k)] += ipd[target_pos].astype(np.float64).sum()
            sums_pw[(T, k)] += pw[target_pos].astype(np.float64).sum()
            counts[(T, k)] += int(target_pos.size)


def _per_read_accumulate_modified(
    seq_arr: np.ndarray,
    ipd: np.ndarray,
    pw: np.ndarray,
    signatures: dict,
    baseline_ipd_means: dict,
    threshold: float,
    sums_ipd: dict,
    sums_pw: dict,
    counts: dict,
) -> None:
    """Pass 2: same walk, but only accumulate positions where observed
    IPD exceeds ``threshold × baseline_ipd_mean[T, k]``.
    """
    L = seq_arr.size
    for T, info in signatures.items():
        target_byte = ord(info["modified_base"])
        positions = np.where(seq_arr == target_byte)[0]
        if positions.size == 0:
            continue
        for k in info["signal_offsets"]:
            base_ipd = baseline_ipd_means.get((T, k))
            if base_ipd is None or base_ipd <= 0:
                continue
            cutoff = threshold * base_ipd
            target_pos = positions + k
            valid = (target_pos >= 0) & (target_pos < L)
            target_pos = target_pos[valid]
            if target_pos.size == 0:
                continue
            ipd_at = ipd[target_pos].astype(np.float64)
            above = ipd_at > cutoff
            if not above.any():
                continue
            sums_ipd[(T, k)] += ipd_at[above].sum()
            sums_pw[(T, k)] += pw[target_pos][above].astype(np.float64).sum()
            counts[(T, k)] += int(above.sum())


# ---------------------------------------------------------------------------
# Two-pass driver
# ---------------------------------------------------------------------------


def compute_ratios(
    bam_paths: list[Path],
    threshold: float = 1.3,
    progress_every: int = 50_000,
) -> dict:
    """Two-pass per-(meth_type, offset) baseline + modified + ratio.

    Returns a dict keyed by ``(T, k)`` containing baseline / modified
    counts, IPD means, PW means, IPD ratio, PW ratio.
    """
    import pysam  # heavy import here so the package import doesn't pay it

    signatures = load_signatures()

    # ── Pass 1: baseline ────────────────────────────────────────────────
    base_sum_ipd: dict = defaultdict(float)
    base_sum_pw: dict = defaultdict(float)
    base_count: dict = defaultdict(int)

    log.info("PASS 1/2 — baseline accumulation across %d BAMs", len(bam_paths))
    for bi, bam_path in enumerate(bam_paths, 1):
        if not Path(bam_path).is_file():
            log.warning("[%d/%d] missing BAM: %s — skip", bi, len(bam_paths), bam_path)
            continue
        log.info("[%d/%d pass1] %s", bi, len(bam_paths), bam_path)
        n_reads = 0
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            for read in bam:
                kin = _read_kinetics(read)
                if kin is None:
                    continue
                seq_arr, ipd, pw = kin
                _per_read_accumulate_baseline(
                    seq_arr, ipd, pw, signatures,
                    base_sum_ipd, base_sum_pw, base_count,
                )
                n_reads += 1
                if n_reads % progress_every == 0:
                    log.info("    ... %d reads processed", n_reads)
        log.info("    → %d reads", n_reads)

    baseline_means_ipd: dict = {}
    baseline_means_pw: dict = {}
    for key, n in base_count.items():
        if n > 0:
            baseline_means_ipd[key] = base_sum_ipd[key] / n
            baseline_means_pw[key] = base_sum_pw[key] / n

    log.info("Baseline means (n / IPD / PW):")
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            n = base_count.get((T, k), 0)
            if n == 0:
                log.info("  %s@%+d: NO DATA", T, k)
                continue
            log.info("  %s@%+d: n=%-12d IPD=%6.2f PW=%6.2f",
                     T, k, n, baseline_means_ipd[(T, k)], baseline_means_pw[(T, k)])

    # ── Pass 2: modified pool ───────────────────────────────────────────
    mod_sum_ipd: dict = defaultdict(float)
    mod_sum_pw: dict = defaultdict(float)
    mod_count: dict = defaultdict(int)

    log.info("PASS 2/2 — modified pool (threshold = %.2f × baseline)", threshold)
    for bi, bam_path in enumerate(bam_paths, 1):
        if not Path(bam_path).is_file():
            continue
        log.info("[%d/%d pass2] %s", bi, len(bam_paths), bam_path)
        n_reads = 0
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            for read in bam:
                kin = _read_kinetics(read)
                if kin is None:
                    continue
                seq_arr, ipd, pw = kin
                _per_read_accumulate_modified(
                    seq_arr, ipd, pw, signatures,
                    baseline_means_ipd, threshold,
                    mod_sum_ipd, mod_sum_pw, mod_count,
                )
                n_reads += 1
                if n_reads % progress_every == 0:
                    log.info("    ... %d reads processed", n_reads)
        log.info("    → %d reads", n_reads)

    # ── Assemble results ────────────────────────────────────────────────
    results: dict = {}
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            base_n = base_count.get((T, k), 0)
            mod_n = mod_count.get((T, k), 0)
            base_ipd = baseline_means_ipd.get((T, k))
            base_pw = baseline_means_pw.get((T, k))
            mod_ipd = (mod_sum_ipd[(T, k)] / mod_n) if mod_n > 0 else None
            mod_pw = (mod_sum_pw[(T, k)] / mod_n) if mod_n > 0 else None
            ipd_ratio = (mod_ipd / base_ipd) if (mod_ipd and base_ipd and base_ipd > 0) else None
            pw_ratio = (mod_pw / base_pw) if (mod_pw and base_pw and base_pw > 0) else None
            results[(T, k)] = {
                "meth_type": T,
                "offset": k,
                "modified_base": info["modified_base"],
                "baseline_n": base_n,
                "baseline_ipd": base_ipd,
                "baseline_pw": base_pw,
                "modified_n": mod_n,
                "modified_ipd": mod_ipd,
                "modified_pw": mod_pw,
                "ipd_ratio": ipd_ratio,
                "pw_ratio": pw_ratio,
                "threshold": threshold,
            }
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(x, fmt="%.4f"):
    return fmt % x if x is not None else "NA"


def write_tsv(results: dict, path: Path) -> None:
    cols = [
        "meth_type", "offset", "modified_base",
        "baseline_n", "baseline_ipd", "baseline_pw",
        "modified_n", "modified_ipd", "modified_pw",
        "ipd_ratio", "pw_ratio", "threshold",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for (T, k), r in sorted(results.items()):
            row = [
                r["meth_type"], f"{r['offset']:+d}", r["modified_base"],
                str(r["baseline_n"]), _fmt(r["baseline_ipd"]), _fmt(r["baseline_pw"]),
                str(r["modified_n"]), _fmt(r["modified_ipd"]), _fmt(r["modified_pw"]),
                _fmt(r["ipd_ratio"]), _fmt(r["pw_ratio"]), _fmt(r["threshold"], "%.3f"),
            ]
            f.write("\t".join(row) + "\n")


def write_json(results: dict, path: Path) -> None:
    serialisable = {f"{T}@{k:+d}": v for (T, k), v in results.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


def log_summary(results: dict) -> None:
    log.info("=" * 64)
    log.info("PER-(meth_type, offset) IPD RATIO SUMMARY")
    log.info("=" * 64)
    log.info("%-6s %+5s  base_n     base_IPD  mod_n     mod_IPD   IPD_ratio  PW_ratio",
             "meth", "off")
    for (T, k), r in sorted(results.items()):
        log.info(
            "%-6s %+5d  %-10d %7s   %-9d %7s   %7s   %7s",
            T, k, r["baseline_n"], _fmt(r["baseline_ipd"], "%.2f"),
            r["modified_n"], _fmt(r["modified_ipd"], "%.2f"),
            _fmt(r["ipd_ratio"], "%.3f"), _fmt(r["pw_ratio"], "%.3f"),
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
    p.add_argument("output_tsv", help="Output TSV with per-(meth_type, offset) stats")
    p.add_argument("--threshold", type=float, default=1.3,
                   help="Multiplier above baseline mean IPD to flag a position "
                        "as candidate-modification in pass 2 (default 1.3).")
    p.add_argument("--output-json", default=None,
                   help="Optional path for the same results in JSON format.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    entries = load_manifest(args.manifest_csv)
    bam_paths = [Path(e.bam_path) for e in entries]
    log.info("Loaded %d BAM paths from manifest %s", len(bam_paths), args.manifest_csv)

    results = compute_ratios(bam_paths, threshold=args.threshold)

    write_tsv(results, Path(args.output_tsv))
    log.info("Saved TSV: %s", args.output_tsv)
    if args.output_json:
        write_json(results, Path(args.output_json))
        log.info("Saved JSON: %s", args.output_json)

    log_summary(results)


if __name__ == "__main__":
    main()
