"""Print the AI's per-kmer kinetic predictions across all scenarios.

Looks up a single 11-mer in the ``predict_kmers.npz`` produced by
``kinsim predict-kmers`` and prints, for every scenario in the YAML:

  - μ_IPD ± σ_IPD  (physical, uint8-comparable units)
  - μ_PW  ± σ_PW
  - log-space μ/σ (model-native, used for sampling)
  - a few stochastic samples drawn from the predicted distribution
  - ratio vs ``none`` (quick sanity check: m6A@+0 should be > 1.0)

Usage::

    python scripts/predict_kmer.py KMER --predict-npz predict_kmers.npz [--n N]

Examples (Dam GATC sits at kmer positions 6-9 → centre A is m6A target):

    # Unmethylated AT-rich kmer
    python scripts/predict_kmer.py AAAAAAAAAAA --predict-npz ...

    # Centre A is the m6A of a Dam GATC (G at pos 6, A at pos 7=centre, T at 8, C at 9)
    python scripts/predict_kmer.py AAAAAAGATCA --predict-npz ...

    # Centre C is the m5C of Dcm CCWGG  (C at pos 5, C at 6, W at 7=centre, G at 8, G at 9)
    python scripts/predict_kmer.py AAAAACCAGGA --predict-npz ...
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from kinsim.utils.encoding import K, encode_kmer


def _scenario_key(label: str) -> str:
    return label.replace("@", "_at_").replace("+", "p").replace("-", "m")


def main():
    p = argparse.ArgumentParser(
        prog="python scripts/predict_kmer.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("kmer", help=f"DNA sequence of length {K} (ACGT only)")
    p.add_argument("--predict-npz", required=True,
                   help="Path to predict_kmers.npz (kinsim predict-kmers output)")
    p.add_argument("--n", type=int, default=5,
                   help="Number of stochastic IPD samples per scenario (default 5)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    kmer = args.kmer.upper().strip()
    if len(kmer) != K:
        print(f"ERROR: kmer must be exactly {K} bases, got {len(kmer)}", file=sys.stderr)
        sys.exit(1)
    if any(c not in "ACGT" for c in kmer):
        print("ERROR: kmer must contain only ACGT", file=sys.stderr)
        sys.exit(1)

    kmer_id = encode_kmer(kmer)
    print(f"Kmer: {kmer}   (kmer_id = {kmer_id})")
    print()

    data = np.load(args.predict_npz, allow_pickle=False)
    if "scenarios_label" not in data.files:
        print("ERROR: npz lacks 'scenarios_label' — re-run kinsim predict-kmers "
              "(this script needs the log-space outputs added recently).", file=sys.stderr)
        sys.exit(1)

    labels = [str(s) for s in data["scenarios_label"].tolist()]
    rng = np.random.default_rng(args.seed)

    print(f"{'Scenario':<12}  "
          f"{'μ_IPD':>7}  {'σ_IPD':>7}  "
          f"{'μ_PW':>7}  {'σ_PW':>7}  "
          f"{'σ_log':>6}  "
          f"sampled IPD")
    print("-" * 90)
    none_mu_ipd = None

    for label in labels:
        sk = _scenario_key(label)
        mu_ipd  = float(data[f"{sk}__mu_ipd"][kmer_id])
        sig_ipd = float(data[f"{sk}__sigma_ipd"][kmer_id])
        mu_pw   = float(data[f"{sk}__mu_pw"][kmer_id])
        sig_pw  = float(data[f"{sk}__sigma_pw"][kmer_id])
        mu_log  = float(data[f"{sk}__mu_ipd_log"][kmer_id])
        sig_log = float(data[f"{sk}__sigma_ipd_log"][kmer_id])

        # Sample N IPD values in log1p space then inv-transform → physical uint8
        samples_log = mu_log + sig_log * rng.standard_normal(args.n)
        samples = np.clip(np.expm1(samples_log), 0, 255).round().astype(int)
        if label == "none":
            none_mu_ipd = mu_ipd

        print(f"{label:<12}  "
              f"{mu_ipd:>7.2f}  {sig_ipd:>7.2f}  "
              f"{mu_pw:>7.2f}  {sig_pw:>7.2f}  "
              f"{sig_log:>6.3f}  "
              f"{list(samples)}")

    if none_mu_ipd and none_mu_ipd > 0:
        print()
        print("Ratios vs 'none' baseline (IPD):")
        for label in labels:
            if label == "none":
                continue
            sk = _scenario_key(label)
            mu_ipd = float(data[f"{sk}__mu_ipd"][kmer_id])
            ratio  = mu_ipd / none_mu_ipd
            indicator = " ← signal!" if ratio > 1.5 else ""
            print(f"  {label:<12}  μ_IPD / μ_IPD_none = {ratio:>5.2f}{indicator}")


if __name__ == "__main__":
    main()
