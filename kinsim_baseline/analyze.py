"""Post-hoc GMM fit + distribution plots from an existing ``compute`` output.

Reads ``<baseline_dir>/baseline.json`` (per-(T, k) × {IPD, PW} histograms),
fits a 2-component Gaussian mixture on each IPD distribution to split the
unmodified bulk from the modified tail, and writes:

    <baseline_dir>/baseline_gmm.tsv      summary with GMM columns
    <baseline_dir>/plots/all_IPD.png     panel of all (T, k) IPD distributions
    <baseline_dir>/plots/all_PW.png      same for PW
    <baseline_dir>/plots/<T>_off<k>_IPD.png  per-bucket detailed plot

Why GMM? The ``ipd_ratio`` column in ``baseline_summary.tsv`` is a fixed
multiplier × mean cut — same percentile in every right-skewed distribution,
so the ratio is ~constant across (T, k) and tells you nothing about
methylation strength. A 2-component GMM separates the unmodified bulk from
the modified tail, giving per-(T, k) parameters that actually differ:

    μ_unmod, σ_unmod, weight_unmod    baseline (most A's / C's)
    μ_mod,   σ_mod,   weight_mod      methylated subset
    ratio = μ_mod / μ_unmod           per-(T, k) signal strength
    weight_mod                        estimate of methylation rate (~1–5%)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2-component GMM fit from a 256-bin histogram
# ---------------------------------------------------------------------------


def fit_gmm_from_hist(
    hist_256: list,
    max_samples: int = 200_000,
    seed: int = 0,
) -> Optional[dict]:
    """Fit a 2-component Gaussian mixture from a 256-bin uint8 histogram.

    Reconstructs samples by drawing from the bin-frequency distribution
    (multinomial), capped at ``max_samples`` to keep the fit fast. Adds
    light dequantising noise (σ=0.5) so the GMM doesn't get stuck on the
    integer grid. Components are sorted by mean (low = unmod, high = mod).

    Returns ``None`` if fewer than 1000 samples.
    """
    from sklearn.mixture import GaussianMixture

    hist = np.asarray(hist_256, dtype=np.int64)
    n = int(hist.sum())
    if n < 1000:
        return None

    bins = np.arange(256, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if n > max_samples:
        probs = hist / n
        idx = rng.choice(256, size=max_samples, p=probs)
        sample = idx.astype(np.float64)
    else:
        sample = np.repeat(bins, hist)

    # Dequantising noise — IPD/PW are uint8, GMM converges better on
    # continuous-looking data.
    sample = sample + rng.normal(0.0, 0.5, size=sample.shape)
    sample = sample.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=seed, max_iter=300).fit(sample)

    order = np.argsort(gmm.means_.flatten())
    mu_lo, mu_hi = gmm.means_.flatten()[order]
    sig_lo, sig_hi = np.sqrt(np.abs(gmm.covariances_.flatten()[order]))
    w_lo, w_hi = gmm.weights_[order]

    return {
        "n_samples": n,
        "n_used":    int(sample.size),
        "mu_unmod":    float(mu_lo),
        "sigma_unmod": float(sig_lo),
        "weight_unmod": float(w_lo),
        "mu_mod":      float(mu_hi),
        "sigma_mod":   float(sig_hi),
        "weight_mod":  float(w_hi),
        "ratio":     (float(mu_hi / mu_lo) if mu_lo > 0 else None),
        "converged": bool(gmm.converged_),
        "bic":       float(gmm.bic(sample)),
    }


# ---------------------------------------------------------------------------
# Summary TSV with GMM columns
# ---------------------------------------------------------------------------


def _fmt(x, fmt="%.4f"):
    return fmt % x if x is not None else "NA"


def write_gmm_summary_tsv(
    signatures: dict,
    hist_ipd_json: dict,
    hist_pw_json: dict,
    gmm_ipd: dict,
    gmm_pw: dict,
    path: Path,
) -> None:
    """Per-(T, k) summary with GMM columns for both IPD and PW."""
    cols = [
        "meth_type", "offset", "modified_base",
        "n_samples",
        # IPD
        "ipd_mu_unmod", "ipd_sigma_unmod", "ipd_w_unmod",
        "ipd_mu_mod",   "ipd_sigma_mod",   "ipd_w_mod",
        "ipd_ratio_gmm",
        # PW
        "pw_mu_unmod", "pw_sigma_unmod", "pw_w_unmod",
        "pw_mu_mod",   "pw_sigma_mod",   "pw_w_mod",
        "pw_ratio_gmm",
        # diagnostics
        "ipd_converged", "ipd_bic",
    ]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for T, info in signatures.items():
            for k in info["signal_offsets"]:
                key = f"{T}@{k:+d}"
                gi = gmm_ipd.get(key)
                gp = gmm_pw.get(key)
                if gi is None:
                    continue
                row = [
                    T, f"{k:+d}", info["modified_base"],
                    str(gi["n_samples"]),
                    _fmt(gi["mu_unmod"],    "%.3f"),
                    _fmt(gi["sigma_unmod"], "%.3f"),
                    _fmt(gi["weight_unmod"],"%.4f"),
                    _fmt(gi["mu_mod"],      "%.3f"),
                    _fmt(gi["sigma_mod"],   "%.3f"),
                    _fmt(gi["weight_mod"],  "%.4f"),
                    _fmt(gi["ratio"],       "%.3f"),
                ]
                if gp is None:
                    row += ["NA"] * 7
                else:
                    row += [
                        _fmt(gp["mu_unmod"],    "%.3f"),
                        _fmt(gp["sigma_unmod"], "%.3f"),
                        _fmt(gp["weight_unmod"],"%.4f"),
                        _fmt(gp["mu_mod"],      "%.3f"),
                        _fmt(gp["sigma_mod"],   "%.3f"),
                        _fmt(gp["weight_mod"],  "%.4f"),
                        _fmt(gp["ratio"],       "%.3f"),
                    ]
                row += [str(gi["converged"]), _fmt(gi["bic"], "%.1f")]
                f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _gmm_density(x: np.ndarray, g: dict, component: str) -> np.ndarray:
    """Evaluate the unmod / mod component density on ``x``."""
    from scipy.stats import norm
    if component == "unmod":
        return norm.pdf(x, g["mu_unmod"], max(g["sigma_unmod"], 0.5)) * g["weight_unmod"]
    elif component == "mod":
        return norm.pdf(x, g["mu_mod"], max(g["sigma_mod"], 0.5)) * g["weight_mod"]
    else:
        raise ValueError(component)


def plot_bucket(
    key: str,
    hist: np.ndarray,
    gmm: dict | None,
    metric: str,
    path: Path,
) -> None:
    """Single-bucket plot: histogram + GMM components on log y."""
    import matplotlib.pyplot as plt

    n = float(hist.sum())
    if n <= 0:
        return
    density = hist / n
    bins = np.arange(256)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bins, density, width=1.0, color="steelblue", alpha=0.6,
           label=f"empirical (n={int(n):,})")

    if gmm:
        x = np.linspace(0, 255, 512)
        d_lo = _gmm_density(x, gmm, "unmod")
        d_hi = _gmm_density(x, gmm, "mod")
        ax.plot(x, d_lo, color="green", lw=2,
                label=(f"unmod  μ={gmm['mu_unmod']:.1f}  σ={gmm['sigma_unmod']:.1f}  "
                       f"w={gmm['weight_unmod']:.1%}"))
        ax.plot(x, d_hi, color="red", lw=2,
                label=(f"mod    μ={gmm['mu_mod']:.1f}  σ={gmm['sigma_mod']:.1f}  "
                       f"w={gmm['weight_mod']:.1%}"))
        ax.plot(x, d_lo + d_hi, color="black", ls="--", lw=1.2, label="GMM total")
        ratio = gmm["ratio"]
        ax.set_title(f"{key}  —  {metric}   ratio μ_mod/μ_unmod = "
                     f"{ratio:.2f}" if ratio else f"{key} — {metric}")
    else:
        ax.set_title(f"{key} — {metric}")

    ax.set_xlabel(f"{metric} bin (uint8)")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.set_ylim(max(density[density > 0].min() * 0.5, 1e-7), density.max() * 2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_panel(
    signatures: dict,
    hist_json: dict,
    gmm_results: dict,
    metric: str,
    path: Path,
) -> None:
    """Combined panel: one subplot per (T, k) for a given metric."""
    import matplotlib.pyplot as plt

    buckets = [(T, k) for T, info in signatures.items() for k in info["signal_offsets"]]
    n = len(buckets)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (T, k) in zip(axes, buckets):
        key = f"{T}@{k:+d}"
        hist = np.asarray(hist_json.get(key, []), dtype=np.float64)
        if hist.size == 0:
            ax.text(0.5, 0.5, f"{key}: NO DATA", ha="center", va="center",
                    transform=ax.transAxes)
            continue
        nn = float(hist.sum())
        density = hist / nn
        bins = np.arange(256)
        ax.bar(bins, density, width=1.0, color="steelblue", alpha=0.6)

        g = gmm_results.get(key)
        if g:
            x = np.linspace(0, 255, 512)
            d_lo = _gmm_density(x, g, "unmod")
            d_hi = _gmm_density(x, g, "mod")
            ax.plot(x, d_lo, color="green", lw=1.5)
            ax.plot(x, d_hi, color="red", lw=1.5)
            ax.plot(x, d_lo + d_hi, color="black", ls="--", lw=1)
            ratio_txt = (f"  ratio={g['ratio']:.2f}" if g["ratio"] else "")
            ax.set_title(f"{key}   μ_u={g['mu_unmod']:.1f}  μ_m={g['mu_mod']:.1f}  "
                         f"w_m={g['weight_mod']:.1%}{ratio_txt}",
                         fontsize=10)
        else:
            ax.set_title(f"{key}  (no GMM)", fontsize=10)

        ax.set_yscale("log")
        ax.set_ylabel("density")

    axes[-1].set_xlabel(f"{metric} bin (uint8)")
    fig.suptitle(f"Per-(meth_type, offset) {metric} distributions", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze(baseline_dir: Path, plot: bool = True, max_samples: int = 200_000) -> None:
    """Read ``baseline.json``, fit per-(T, k) 2-GMM, write summary + plots."""
    baseline_dir = Path(baseline_dir)
    json_path = baseline_dir / "baseline.json"
    if not json_path.is_file():
        log.error("baseline.json not found in %s — did you run `compute` first?",
                  baseline_dir)
        raise SystemExit(1)

    log.info("Loading %s", json_path)
    data = json.loads(json_path.read_text())
    signatures = data["signatures"]
    hist_ipd = data["ipd"]
    hist_pw  = data["pw"]

    gmm_ipd: dict[str, dict] = {}
    gmm_pw:  dict[str, dict] = {}

    log.info("Fitting 2-GMM per (T, k) on IPD + PW histograms ...")
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            g_i = fit_gmm_from_hist(hist_ipd.get(key, []), max_samples=max_samples)
            g_p = fit_gmm_from_hist(hist_pw.get(key, []),  max_samples=max_samples)
            gmm_ipd[key] = g_i
            gmm_pw[key]  = g_p
            if g_i:
                log.info("  %-8s  IPD  μ_u=%.2f  μ_m=%.2f  w_m=%.2f%%  ratio=%.2f",
                         key, g_i["mu_unmod"], g_i["mu_mod"],
                         g_i["weight_mod"] * 100, g_i["ratio"] or float("nan"))
            else:
                log.warning("  %s  IPD  fit failed (too few samples)", key)

    summary_path = baseline_dir / "baseline_gmm.tsv"
    write_gmm_summary_tsv(signatures, hist_ipd, hist_pw, gmm_ipd, gmm_pw, summary_path)
    log.info("Saved %s", summary_path)

    # Also dump the GMM params as JSON for downstream re-use
    gmm_json = {"signatures": signatures, "ipd": gmm_ipd, "pw": gmm_pw}
    (baseline_dir / "baseline_gmm.json").write_text(json.dumps(gmm_json, indent=2))
    log.info("Saved %s", baseline_dir / "baseline_gmm.json")

    if plot:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            log.warning("matplotlib not installed — skipping plots. `pip install matplotlib scipy`")
            return
        try:
            import scipy  # noqa: F401
        except ImportError:
            log.warning("scipy not installed — GMM curves won't be drawn. `pip install scipy`")

        plot_dir = baseline_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        log.info("Plotting combined panels ...")
        plot_panel(signatures, hist_ipd, gmm_ipd, "IPD", plot_dir / "all_IPD.png")
        plot_panel(signatures, hist_pw,  gmm_pw,  "PW",  plot_dir / "all_PW.png")

        log.info("Plotting per-bucket detail figures ...")
        for T, info in signatures.items():
            for k in info["signal_offsets"]:
                key = f"{T}@{k:+d}"
                tag = f"{T}_off{k:+d}".replace("+", "p").replace("-", "m")
                hi = np.asarray(hist_ipd.get(key, []), dtype=np.float64)
                hp = np.asarray(hist_pw.get(key, []),  dtype=np.float64)
                if hi.size:
                    plot_bucket(key, hi, gmm_ipd.get(key), "IPD",
                                plot_dir / f"{tag}_IPD.png")
                if hp.size:
                    plot_bucket(key, hp, gmm_pw.get(key), "PW",
                                plot_dir / f"{tag}_PW.png")
        log.info("Plots in %s/", plot_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline analyze",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("baseline_dir",
                   help="Directory containing baseline.json from `compute`.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation (TSV / JSON only).")
    p.add_argument("--max-samples", type=int, default=200_000,
                   help="Cap for GMM sample size per bucket (default 200 000).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze(Path(args.baseline_dir),
            plot=not args.no_plot,
            max_samples=args.max_samples)


if __name__ == "__main__":
    main()
