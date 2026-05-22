"""Static summary of a ``predict_kmers`` .npz: ratio distributions + IPD factor table.

For presentations: the canonical ``predict-kmers`` HTML output can be hundreds of
MB at K=11 (4M kmers × N scenarios). This script dumps a lightweight static PNG
and a one-row-per-scenario TSV from the same .npz, so the numbers are
copy-pasteable into slides.

Usage::

    python scripts/plot_predict_kmers_summary.py <npz_path> <output_prefix>

Writes::

    <prefix>_stats.tsv     scenario | n_valid | ratio_median | IQR | sigma_median ...
    <prefix>_ratios.png    histogram of μ_meth / μ_baseline per scenario
    <prefix>_sigmas.png    σ distributions per scenario (shows the variance bug)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WONG = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#F0E442"}


def _sk(label: str) -> str:
    return label.replace("@", "_at_").replace("+", "p").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", help="predict_kmers .npz")
    ap.add_argument("output_prefix", help="Output prefix (without extension)")
    args = ap.parse_args()

    z = np.load(args.npz_path)
    labels = [str(s) for s in z["scenarios_label"]]
    m_ids = z["scenarios_meth_id"]
    offsets = z["scenarios_offset"]

    none_mu = z["none__mu_ipd"]
    none_sigma = z["none__sigma_ipd"]
    eps = 1e-6
    none_safe = np.maximum(none_mu, eps)

    meth_scn = [
        (lbl, int(mid), int(off))
        for lbl, mid, off in zip(labels, m_ids, offsets)
        if lbl != "none"
    ]

    print(f"\nBaseline σ_ipd: median = {np.median(none_sigma):.3f}, "
          f"mean = {none_sigma.mean():.3f}, max = {none_sigma.max():.3f}")
    print(f"Baseline μ_ipd: median = {np.median(none_mu):.3f}\n")

    rows: list[dict] = []
    for lbl, mid, off in meth_scn:
        sk = _sk(lbl)
        mu = z[f"{sk}__mu_ipd"]
        sig = z[f"{sk}__sigma_ipd"]
        ratio = mu / none_safe
        valid = ~np.isnan(ratio)
        r_v = ratio[valid]
        s_v = sig[valid]
        mu_v = mu[valid]
        q05, q25, q50, q75, q95 = np.percentile(r_v, [5, 25, 50, 75, 95])
        sig_q50, sig_mean = float(np.median(s_v)), float(s_v.mean())
        rows.append({
            "scenario": lbl,
            "n_valid": int(valid.sum()),
            "n_total": int(ratio.size),
            "ratio_median": float(q50),
            "ratio_iqr_low": float(q25),
            "ratio_iqr_high": float(q75),
            "ratio_p05": float(q05),
            "ratio_p95": float(q95),
            "ratio_mean": float(r_v.mean()),
            "mu_meth_median": float(np.median(mu_v)),
            "sigma_median": sig_q50,
            "sigma_mean": sig_mean,
        })
        print(f"  {lbl:>10}  IPD factor median ×{q50:.3f}  "
              f"IQR=[{q25:.3f}, {q75:.3f}]  "
              f"σ_med={sig_q50:.3f}  σ_mean={sig_mean:.3f}")

    cols = ["scenario", "n_valid", "n_total", "ratio_median", "ratio_iqr_low",
            "ratio_iqr_high", "ratio_p05", "ratio_p95", "ratio_mean",
            "mu_meth_median", "sigma_median", "sigma_mean"]
    tsv_path = Path(args.output_prefix + "_stats.tsv")
    with open(tsv_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                for c in cols
            ) + "\n")
    print(f"\nWrote {tsv_path}")

    n = len(meth_scn)
    cols_p = min(3, n)
    rows_p = (n + cols_p - 1) // cols_p
    fig, axes = plt.subplots(rows_p, cols_p, figsize=(5 * cols_p, 3.6 * rows_p), squeeze=False)
    for i, (lbl, mid, _off) in enumerate(meth_scn):
        sk = _sk(lbl)
        mu = z[f"{sk}__mu_ipd"]
        ratio = mu / none_safe
        r_v = ratio[~np.isnan(ratio)]
        ax = axes[i // cols_p, i % cols_p]
        ax.hist(
            r_v, bins=np.linspace(0, 4, 161),
            color=WONG.get(int(mid), "#888"), edgecolor="black", linewidth=0.2,
        )
        med = float(np.median(r_v))
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="no shift (=1)")
        ax.axvline(med, color="#D55E00", linestyle="-", linewidth=2, label=f"median = {med:.2f}")
        ax.set_title(lbl, fontsize=11)
        ax.set_xlabel("μ_meth / μ_baseline")
        ax.set_ylabel("count")
        ax.set_xlim(0, 4)
        ax.legend(loc="upper right", fontsize=8)
    for j in range(n, rows_p * cols_p):
        axes[j // cols_p, j % cols_p].set_visible(False)
    fig.suptitle(
        "Per-kmer IPD factor (μ_meth / μ_baseline) across all biology-valid kmers\n"
        "Methylation should produce a median > 1 (m6A: expected ~3-5)",
        fontsize=12,
    )
    plt.tight_layout()
    png_ratios = Path(args.output_prefix + "_ratios.png")
    fig.savefig(png_ratios, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_ratios}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    ns_v = none_sigma[~np.isnan(none_sigma)]
    sig_p99 = float(np.percentile(ns_v, 99))
    ax2.hist(
        ns_v, bins=80, alpha=0.5, label="none (baseline)",
        histtype="step", linewidth=2.0, color="#888", linestyle="--",
    )
    for lbl, mid, _off in meth_scn:
        sk = _sk(lbl)
        sig = z[f"{sk}__sigma_ipd"]
        sig_v = sig[~np.isnan(sig)]
        ax2.hist(
            sig_v, bins=80, alpha=0.6, label=lbl,
            histtype="step", linewidth=1.7, color=WONG.get(int(mid), "#888"),
        )
        sig_p99 = max(sig_p99, float(np.percentile(sig_v, 99)))
    ax2.set_xlim(0, sig_p99 * 1.1)
    ax2.set_xlabel("σ_ipd (uint8 space)")
    ax2.set_ylabel("count")
    ax2.set_title(
        "Per-kmer σ_ipd distribution\n"
        "Pre-audit clamp_max = 3.0 allows σ up to e³ ≈ 20 in log-space — "
        "look for the long right tail",
        fontsize=11,
    )
    ax2.legend(fontsize=9)
    plt.tight_layout()
    png_sig = Path(args.output_prefix + "_sigmas.png")
    fig2.savefig(png_sig, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote {png_sig}")


if __name__ == "__main__":
    main()
