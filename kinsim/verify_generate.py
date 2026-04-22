"""Verify a generated BAM reproduces the per-kmer kinetics of a reference BAM.

Extracts raw (IPD, PW) samples from both the reference (real) BAM and the
newly generated BAM using the same motif labelling, then compares
per-(kmer, meth_id) mean and standard deviation.

Output TSV columns:
    kmer_id  kmer  meth     n_ref  mu_ipd_ref  sd_ipd_ref  mu_pw_ref  sd_pw_ref
                             n_gen  mu_ipd_gen  sd_ipd_gen  mu_pw_gen  sd_pw_gen
                             d_mu_ipd  d_mu_pw

Summary lines at the end include Pearson correlation of (mu_ref, mu_gen)
across all keys — the headline sanity number.

Usage:
    kinsim verify-generate <ref.bam> <gen.bam> <motifs> <output.tsv>
    kinsim verify-generate ref.bam gen.bam motifs.csv report.tsv --min-samples 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .utils.encoding import decode_kmer
from .utils.config import setup_logging
from .utils.motifs import load_motif_string
from .refine import MOD_NAMES

log = logging.getLogger(__name__)


def _summarize(d: dict) -> dict:
    """Map (kmer_id, meth_id) -> (n, mu_ipd, sd_ipd, mu_pw, sd_pw)."""
    out: dict = {}
    for key, arr in d.items():
        if not isinstance(key, tuple) or not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            continue
        ipd = arr[:, 0].astype(np.float32)
        pw  = arr[:, 1].astype(np.float32)
        out[key] = (
            int(arr.shape[0]),
            float(ipd.mean()), float(ipd.std(ddof=1) if len(ipd) > 1 else 0.0),
            float(pw.mean()),  float(pw.std(ddof=1)  if len(pw)  > 1 else 0.0),
        )
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def verify(
    ref_bam: str,
    gen_bam: str,
    motif_arg: str,
    output_tsv: Path,
    max_samples_per_key: int = 50_000,
    min_samples: int = 5,
) -> dict:
    from .extract import extract_samples_from_bam

    motif_string = load_motif_string(motif_arg)
    log.info("Motif string: %s", motif_string)

    log.info("=== Extracting from REFERENCE BAM: %s ===", ref_bam)
    ref = extract_samples_from_bam(
        ref_bam, motif_string,
        max_samples_per_key=max_samples_per_key,
        use_reverse_strand=True,
        binarize=False,
    )
    ref.pop("__meta__", None)

    log.info("=== Extracting from GENERATED BAM: %s ===", gen_bam)
    gen = extract_samples_from_bam(
        gen_bam, motif_string,
        max_samples_per_key=max_samples_per_key,
        use_reverse_strand=True,
        binarize=False,
    )
    gen.pop("__meta__", None)

    ref_stats = _summarize(ref)
    gen_stats = _summarize(gen)
    all_keys = sorted(set(ref_stats) | set(gen_stats))
    log.info("Keys: ref=%d  gen=%d  union=%d",
             len(ref_stats), len(gen_stats), len(all_keys))

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing TSV: %s", output_tsv)

    mu_ipd_ref_arr, mu_ipd_gen_arr = [], []
    mu_pw_ref_arr,  mu_pw_gen_arr  = [], []
    n_written = 0

    with open(output_tsv, "w") as f:
        f.write(
            "kmer_id\tkmer\tmeth\t"
            "n_ref\tmu_ipd_ref\tsd_ipd_ref\tmu_pw_ref\tsd_pw_ref\t"
            "n_gen\tmu_ipd_gen\tsd_ipd_gen\tmu_pw_gen\tsd_pw_gen\t"
            "d_mu_ipd\td_mu_pw\n"
        )
        for key in all_keys:
            kmer_id, meth_id = key
            r = ref_stats.get(key, (0, 0.0, 0.0, 0.0, 0.0))
            g = gen_stats.get(key, (0, 0.0, 0.0, 0.0, 0.0))
            if r[0] < min_samples and g[0] < min_samples:
                continue
            kmer_str = decode_kmer(kmer_id) if kmer_id >= 0 else "?"
            meth_str = MOD_NAMES.get(meth_id, f"mod{meth_id}")
            d_mu_ipd = g[1] - r[1]
            d_mu_pw  = g[3] - r[3]
            f.write(
                f"{kmer_id}\t{kmer_str}\t{meth_str}\t"
                f"{r[0]}\t{r[1]:.4f}\t{r[2]:.4f}\t{r[3]:.4f}\t{r[4]:.4f}\t"
                f"{g[0]}\t{g[1]:.4f}\t{g[2]:.4f}\t{g[3]:.4f}\t{g[4]:.4f}\t"
                f"{d_mu_ipd:+.4f}\t{d_mu_pw:+.4f}\n"
            )
            n_written += 1
            if r[0] >= min_samples and g[0] >= min_samples:
                mu_ipd_ref_arr.append(r[1]); mu_ipd_gen_arr.append(g[1])
                mu_pw_ref_arr.append(r[3]);  mu_pw_gen_arr.append(g[3])

    mu_ipd_ref_arr = np.asarray(mu_ipd_ref_arr, dtype=np.float64)
    mu_ipd_gen_arr = np.asarray(mu_ipd_gen_arr, dtype=np.float64)
    mu_pw_ref_arr  = np.asarray(mu_pw_ref_arr,  dtype=np.float64)
    mu_pw_gen_arr  = np.asarray(mu_pw_gen_arr,  dtype=np.float64)

    r_ipd = _pearson(mu_ipd_ref_arr, mu_ipd_gen_arr)
    r_pw  = _pearson(mu_pw_ref_arr,  mu_pw_gen_arr)
    mae_ipd = float(np.mean(np.abs(mu_ipd_gen_arr - mu_ipd_ref_arr))) if len(mu_ipd_ref_arr) else float("nan")
    mae_pw  = float(np.mean(np.abs(mu_pw_gen_arr  - mu_pw_ref_arr )))  if len(mu_pw_ref_arr ) else float("nan")

    log.info("=" * 56)
    log.info("  VERIFY-GENERATE SUMMARY")
    log.info("=" * 56)
    log.info("  Rows written:         %d", n_written)
    log.info("  Paired keys (n >= %d): %d", min_samples, len(mu_ipd_ref_arr))
    log.info("  Pearson r (mu_ipd):   %.4f", r_ipd)
    log.info("  Pearson r (mu_pw):    %.4f", r_pw)
    log.info("  MAE (mu_ipd):         %.4f", mae_ipd)
    log.info("  MAE (mu_pw):          %.4f", mae_pw)
    log.info("=" * 56)

    return {
        "n_rows":           n_written,
        "n_paired":         int(len(mu_ipd_ref_arr)),
        "pearson_mu_ipd":   r_ipd,
        "pearson_mu_pw":    r_pw,
        "mae_mu_ipd":       mae_ipd,
        "mae_mu_pw":        mae_pw,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="kinsim verify-generate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("ref_bam", help="Reference (real) BAM with fi/fp kinetic tags")
    ap.add_argument("gen_bam", help="Generated BAM with fi/fp synthetic tags")
    ap.add_argument("motifs",  help="KinSim motif string OR path to motifs.csv / REBASE file")
    ap.add_argument("output_tsv", help="Output TSV with per-(kmer, meth) comparison")
    ap.add_argument("--max-samples", type=int, default=50_000,
                    help="Reservoir cap per (kmer, meth_id) key during extraction (default 50000)")
    ap.add_argument("--min-samples", type=int, default=5,
                    help="Drop rows where both BAMs have fewer than this many samples (default 5)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)

    if not Path(args.ref_bam).exists():
        print(f"ERROR: ref_bam not found: {args.ref_bam}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.gen_bam).exists():
        print(f"ERROR: gen_bam not found: {args.gen_bam}", file=sys.stderr)
        sys.exit(1)

    verify(
        args.ref_bam, args.gen_bam, args.motifs,
        Path(args.output_tsv),
        max_samples_per_key=args.max_samples,
        min_samples=args.min_samples,
    )


if __name__ == "__main__":
    main()
