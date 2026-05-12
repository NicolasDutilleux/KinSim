"""Per-kmer outlier detection using the AI model as baseline.

The trained model predicts (μ_pred, σ_pred) per 11-mer under the
``meth_id = none`` scenario — that's the AI's null hypothesis: "if this
kmer carried no methylation, what should its IPD/PW look like?".

Given that null model and a manifest of BAMs, we walk every read,
encode every length-K window, look up the AI's null prediction for
that kmer and ask:

    is_outlier(pos) = (observed_IPD[pos] > μ_pred[kmer] + N · σ_pred[kmer])

For every kmer we accumulate:

    n_total           total observations
    n_above           observations above the per-kmer threshold
    sum_obs           Σ of observed IPDs (all)
    sum2_obs          Σ of observed IPD² (for σ_obs)
    sum_above         Σ of IPDs above threshold
    sum2_above        Σ of IPD² above threshold
    (same for PW)

The point: if the AI baseline is right, the **above-rate**
``n_above / n_total`` per kmer is the empirical false-positive rate
where the kmer never carries modification (~2.5 % for N=2σ on a
Gaussian baseline). Kmers that genuinely carry methylation events
in the corpus (e.g. ``...GATC...`` for Dam, ``...CCWGG...`` for Dcm)
will produce an above-rate FAR higher than that, and the
above-threshold population's mean / sigma will sit as a real second
mode — the signal we want to confirm exists.

Usage::

    python -m kinsim_baseline per-kmer PREDICT_NPZ MANIFEST_CSV OUTPUT_DIR
        [--threshold 2.0]

``PREDICT_NPZ`` is the file written by ``kinsim predict-kmers``.
``MANIFEST_CSV`` is a standard KinSim manifest (``sample_id, bam_path, motifs``).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from kinsim.utils.config import load_manifest
from kinsim.utils.encoding import K, KMER_PRED_IDX

log = logging.getLogger(__name__)

N_KMERS = 4 ** K  # 4 194 304


# ---------------------------------------------------------------------------
# Baseline loading (AI predictions for the 'none' scenario)
# ---------------------------------------------------------------------------


def load_baseline(predict_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (μ_ipd, σ_ipd, μ_pw, σ_pw) per kmer from ``predict-kmers`` output.

    The AI's ``none`` scenario is the unmethylated null hypothesis. All
    arrays returned have length ``N_KMERS`` and ``float32`` dtype.
    """
    data = np.load(predict_npz)
    for k in ("none__mu_ipd", "none__sigma_ipd", "none__mu_pw", "none__sigma_pw"):
        if k not in data.files:
            raise KeyError(
                f"{predict_npz} is missing '{k}'. Found: {list(data.files)}. "
                f"Re-run `kinsim predict-kmers`."
            )
    mu_ipd = data["none__mu_ipd"].astype(np.float32)
    sigma_ipd = data["none__sigma_ipd"].astype(np.float32)
    mu_pw = data["none__mu_pw"].astype(np.float32)
    sigma_pw = data["none__sigma_pw"].astype(np.float32)
    if mu_ipd.shape != (N_KMERS,):
        raise ValueError(
            f"Expected length {N_KMERS}, got {mu_ipd.shape}. "
            f"Re-run `kinsim predict-kmers` with K={K}."
        )
    log.info(
        "AI baseline loaded from %s:", predict_npz,
    )
    log.info("  μ_IPD: min=%.2f mean=%.2f max=%.2f",
             float(mu_ipd.min()), float(mu_ipd.mean()), float(mu_ipd.max()))
    log.info("  σ_IPD: min=%.2f mean=%.2f max=%.2f",
             float(sigma_ipd.min()), float(sigma_ipd.mean()), float(sigma_ipd.max()))
    return mu_ipd, sigma_ipd, mu_pw, sigma_pw


# ---------------------------------------------------------------------------
# Vectorised kmer encoding from a read sequence
# ---------------------------------------------------------------------------


_BASE_LUT = np.full(256, -1, dtype=np.int8)
for ch, v in zip(b"ACGT", (0, 1, 2, 3)):
    _BASE_LUT[ch] = v
for ch, v in zip(b"acgt", (0, 1, 2, 3)):
    _BASE_LUT[ch] = v


def _kmers_from_seq(seq_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For an ASCII-byte sequence, return ``(kmer_ids, valid_mask)``.

    A window is valid only if all K bases are ACGT (no N). The encoding
    matches ``encoding.encode_kmer`` (MSB-first, position 0 is most-significant).
    """
    base_ids = _BASE_LUT[seq_arr]
    n = base_ids.shape[0] - K + 1
    if n <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=bool)
    out = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for offset in range(K):
        slc = base_ids[offset : offset + n]
        valid &= slc >= 0
        out = (out << 2) | np.where(slc >= 0, slc, 0).astype(np.int64)
    return out, valid


def _read_kinetics(read):
    """Return ``(seq_arr, ipd, pw)`` or ``None``. Supports ip/pw and fi/fp."""
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


# ---------------------------------------------------------------------------
# Per-read accumulation
# ---------------------------------------------------------------------------


def _accumulate_read(
    seq_arr: np.ndarray, ipd: np.ndarray, pw: np.ndarray,
    thr_ipd: np.ndarray, thr_pw: np.ndarray,
    n_total: np.ndarray, n_above_ipd: np.ndarray, n_above_pw: np.ndarray,
    sum_obs_ipd: np.ndarray, sum2_obs_ipd: np.ndarray,
    sum_above_ipd: np.ndarray, sum2_above_ipd: np.ndarray,
    sum_obs_pw: np.ndarray, sum2_obs_pw: np.ndarray,
    sum_above_pw: np.ndarray, sum2_above_pw: np.ndarray,
    hist_ipd: np.ndarray | None = None, hist_n_bins: int = 64,
) -> int:
    """For each valid 11-mer window, compare its centre IPD/PW against the
    AI's per-kmer threshold and update the running accumulators.

    The kmer covering read positions ``[i, i+K)`` predicts IPD/PW at
    ``i + KMER_PRED_IDX``.
    """
    L = seq_arr.shape[0]
    n_windows = L - K + 1
    if n_windows <= 0:
        return 0

    kmer_ids, valid = _kmers_from_seq(seq_arr)
    if not valid.any():
        return 0

    centre = KMER_PRED_IDX
    # The kmer at window-start i has its prediction position at i + centre.
    # All valid pred positions live inside ipd / pw because centre < K <= L.
    obs_ipd = ipd[centre : centre + n_windows].astype(np.float32)
    obs_pw  = pw[centre  : centre + n_windows].astype(np.float32)

    kmer_ids = kmer_ids[valid]
    obs_ipd = obs_ipd[valid]
    obs_pw = obs_pw[valid]
    if kmer_ids.size == 0:
        return 0

    ones = np.ones(kmer_ids.size, dtype=np.int64)
    np.add.at(n_total, kmer_ids, ones)

    # Observed totals (across all observations) — used to compare μ_obs vs μ_pred
    np.add.at(sum_obs_ipd,  kmer_ids, obs_ipd.astype(np.float64))
    np.add.at(sum2_obs_ipd, kmer_ids, (obs_ipd.astype(np.float64)) ** 2)
    np.add.at(sum_obs_pw,   kmer_ids, obs_pw.astype(np.float64))
    np.add.at(sum2_obs_pw,  kmer_ids, (obs_pw.astype(np.float64)) ** 2)

    # Above-threshold subset (per-kmer threshold from the AI baseline)
    thr_i = thr_ipd[kmer_ids]
    above_i = obs_ipd > thr_i
    if above_i.any():
        ki = kmer_ids[above_i]
        oi = obs_ipd[above_i].astype(np.float64)
        np.add.at(n_above_ipd,    ki, np.ones(ki.size, dtype=np.int64))
        np.add.at(sum_above_ipd,  ki, oi)
        np.add.at(sum2_above_ipd, ki, oi * oi)

    thr_p = thr_pw[kmer_ids]
    above_p = obs_pw > thr_p
    if above_p.any():
        kp = kmer_ids[above_p]
        op = obs_pw[above_p].astype(np.float64)
        np.add.at(n_above_pw,    kp, np.ones(kp.size, dtype=np.int64))
        np.add.at(sum_above_pw,  kp, op)
        np.add.at(sum2_above_pw, kp, op * op)

    # Per-kmer observed IPD histogram — 64-bin (each bin = 4 IPD units).
    # Stored as uint16 (max 65535 per bin); rare overflow gets clamped on
    # save. ``hist_ipd`` shape: (N_KMERS, hist_n_bins).
    if hist_ipd is not None:
        bin_idx = (obs_ipd.astype(np.int32) // (256 // hist_n_bins)).clip(
            0, hist_n_bins - 1,
        )
        # Flatten (kmer_id, bin_idx) → linear idx for one np.add.at
        flat = kmer_ids * hist_n_bins + bin_idx
        # We add into a flat view of hist_ipd to keep one scatter
        np.add.at(hist_ipd.reshape(-1), flat, 1)

    return int(kmer_ids.size)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def accumulate_per_kmer(
    bam_paths: list[Path],
    mu_ipd: np.ndarray, sigma_ipd: np.ndarray,
    mu_pw: np.ndarray,  sigma_pw: np.ndarray,
    threshold_factor: float,
    max_reads_per_bam: int | None = None,
    progress_every: int = 50_000,
    collect_hist: bool = True,
    hist_n_bins: int = 64,
) -> dict:
    """Walk all BAMs, return per-kmer outlier accumulators (size N_KMERS each)."""
    import pysam

    thr_ipd = (mu_ipd + threshold_factor * sigma_ipd).astype(np.float32)
    thr_pw  = (mu_pw  + threshold_factor * sigma_pw ).astype(np.float32)

    n_total       = np.zeros(N_KMERS, dtype=np.int64)
    n_above_ipd   = np.zeros(N_KMERS, dtype=np.int64)
    n_above_pw    = np.zeros(N_KMERS, dtype=np.int64)
    sum_obs_ipd   = np.zeros(N_KMERS, dtype=np.float64)
    sum2_obs_ipd  = np.zeros(N_KMERS, dtype=np.float64)
    sum_above_ipd = np.zeros(N_KMERS, dtype=np.float64)
    sum2_above_ipd= np.zeros(N_KMERS, dtype=np.float64)
    sum_obs_pw    = np.zeros(N_KMERS, dtype=np.float64)
    sum2_obs_pw   = np.zeros(N_KMERS, dtype=np.float64)
    sum_above_pw  = np.zeros(N_KMERS, dtype=np.float64)
    sum2_above_pw = np.zeros(N_KMERS, dtype=np.float64)
    # Per-kmer observed IPD histogram (~1 GB for 4.2M × 64 × uint32). uint32
    # to avoid silent overflow on np.add.at; we can compress when saving.
    hist_ipd = (
        np.zeros((N_KMERS, hist_n_bins), dtype=np.uint32) if collect_hist else None
    )
    per_bam: dict = {}

    log.info(
        "Walking %d BAMs with threshold = μ_pred + %.2f · σ_pred (per kmer)",
        len(bam_paths), threshold_factor,
    )
    if collect_hist:
        log.info("  Collecting per-kmer IPD histogram (%d bins × %d kmers ≈ %.0f MB)",
                 hist_n_bins, N_KMERS, hist_ipd.nbytes / 1e6)
    log.info("  IPD threshold: min=%.2f mean=%.2f max=%.2f",
             float(thr_ipd.min()), float(thr_ipd.mean()), float(thr_ipd.max()))

    for bi, bam_path in enumerate(bam_paths, 1):
        bam_path = Path(bam_path)
        if not bam_path.is_file():
            log.warning("[%d/%d] missing BAM: %s — skip", bi, len(bam_paths), bam_path)
            per_bam[str(bam_path)] = {"n_reads": 0, "elapsed_s": 0.0, "skipped": True}
            continue
        log.info("[%d/%d] %s", bi, len(bam_paths), bam_path)
        t0 = time.time()
        n_reads = 0
        n_windows = 0
        with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
            for read in bam:
                if max_reads_per_bam and n_reads >= max_reads_per_bam:
                    break
                kin = _read_kinetics(read)
                if kin is None:
                    continue
                seq_arr, ipd, pw = kin
                n_windows += _accumulate_read(
                    seq_arr, ipd, pw, thr_ipd, thr_pw,
                    n_total, n_above_ipd, n_above_pw,
                    sum_obs_ipd, sum2_obs_ipd, sum_above_ipd, sum2_above_ipd,
                    sum_obs_pw,  sum2_obs_pw,  sum_above_pw,  sum2_above_pw,
                    hist_ipd=hist_ipd, hist_n_bins=hist_n_bins,
                )
                n_reads += 1
                if n_reads % progress_every == 0:
                    log.info("    ... %d reads (%.1f M windows)", n_reads, n_windows / 1e6)
        dt = time.time() - t0
        log.info("    → %d reads, %d windows, %.1f s", n_reads, n_windows, dt)
        per_bam[str(bam_path)] = {
            "n_reads": n_reads, "n_windows": n_windows, "elapsed_s": round(dt, 2),
        }

    return {
        "n_total": n_total,
        "n_above_ipd": n_above_ipd, "n_above_pw": n_above_pw,
        "sum_obs_ipd": sum_obs_ipd, "sum2_obs_ipd": sum2_obs_ipd,
        "sum_above_ipd": sum_above_ipd, "sum2_above_ipd": sum2_above_ipd,
        "sum_obs_pw": sum_obs_pw, "sum2_obs_pw": sum2_obs_pw,
        "sum_above_pw": sum_above_pw, "sum2_above_pw": sum2_above_pw,
        "hist_ipd": hist_ipd,        # (N_KMERS, hist_n_bins) uint32 or None
        "hist_n_bins": hist_n_bins,
        "per_bam": per_bam,
    }


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline per-kmer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("predict_npz",
                   help="Output of `kinsim predict-kmers` (.npz with the per-kmer "
                        "μ, σ predictions under the 'none' scenario).")
    p.add_argument("manifest_csv",
                   help="KinSim manifest CSV (sample_id, bam_path, motifs).")
    p.add_argument("output_dir",
                   help="Output directory; per_kmer_observed.npz lands here.")
    p.add_argument("--threshold", type=float, default=2.0,
                   help="σ-multiplier for the outlier threshold (default 2.0).")
    p.add_argument("--max-reads-per-bam", type=int, default=None,
                   help="Cap reads per BAM (default: all). Useful for a quick test.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_ipd, sigma_ipd, mu_pw, sigma_pw = load_baseline(Path(args.predict_npz))

    entries = load_manifest(args.manifest_csv)
    bam_paths = [Path(e.bam_path) for e in entries]
    log.info("Loaded %d BAM paths from %s", len(bam_paths), args.manifest_csv)

    t0 = time.time()
    out = accumulate_per_kmer(
        bam_paths, mu_ipd, sigma_ipd, mu_pw, sigma_pw,
        threshold_factor=args.threshold,
        max_reads_per_bam=args.max_reads_per_bam,
    )
    elapsed = time.time() - t0
    log.info("Walk complete in %.1f s (%.1f min)", elapsed, elapsed / 60)

    obs_total = int(out["n_total"].sum())
    above_total_ipd = int(out["n_above_ipd"].sum())
    above_total_pw  = int(out["n_above_pw"].sum())
    n_covered = int((out["n_total"] > 0).sum())
    log.info("Coverage: %d / %d kmers (%.2f%%)",
             n_covered, N_KMERS, 100 * n_covered / N_KMERS)
    log.info("Observations: %d total", obs_total)
    log.info("  above-threshold (IPD): %d (%.3f%%)",
             above_total_ipd, 100 * above_total_ipd / max(obs_total, 1))
    log.info("  above-threshold (PW):  %d (%.3f%%)",
             above_total_pw,  100 * above_total_pw  / max(obs_total, 1))

    out_path = out_dir / "per_kmer_observed.npz"
    save_kwargs = {
        "n_total": out["n_total"],
        "n_above_ipd": out["n_above_ipd"], "n_above_pw": out["n_above_pw"],
        "sum_obs_ipd": out["sum_obs_ipd"], "sum2_obs_ipd": out["sum2_obs_ipd"],
        "sum_above_ipd": out["sum_above_ipd"], "sum2_above_ipd": out["sum2_above_ipd"],
        "sum_obs_pw": out["sum_obs_pw"],   "sum2_obs_pw": out["sum2_obs_pw"],
        "sum_above_pw": out["sum_above_pw"], "sum2_above_pw": out["sum2_above_pw"],
        "mu_pred_ipd": mu_ipd, "sigma_pred_ipd": sigma_ipd,
        "mu_pred_pw": mu_pw,   "sigma_pred_pw": sigma_pw,
        "threshold_factor": np.float32(args.threshold),
    }
    if out["hist_ipd"] is not None:
        # Clip to uint16 max (65535) to halve the disk size at the cost of
        # losing exact counts for the ~0.01% of bins that overflow.
        clipped = np.clip(out["hist_ipd"], 0, 65535).astype(np.uint16)
        save_kwargs["hist_ipd"] = clipped
        save_kwargs["hist_n_bins"] = np.int32(out["hist_n_bins"])
    np.savez_compressed(out_path, **save_kwargs)
    log.info("Saved: %s", out_path)

    info_path = out_dir / "per_kmer_observed_info.json"
    info_path.write_text(json.dumps({
        "predict_npz":    str(args.predict_npz),
        "manifest_csv":   str(args.manifest_csv),
        "threshold":      args.threshold,
        "n_kmers":        N_KMERS,
        "n_covered":      n_covered,
        "obs_total":      obs_total,
        "above_total_ipd": above_total_ipd,
        "above_total_pw":  above_total_pw,
        "elapsed_s":      round(elapsed, 2),
        "per_bam":        out["per_bam"],
    }, indent=2))
    log.info("Saved: %s", info_path)


if __name__ == "__main__":
    main()
