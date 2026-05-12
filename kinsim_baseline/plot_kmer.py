"""Visualise per-kmer outlier-rate results from ``per_kmer_observed.npz``.

Three panels in one interactive HTML:

  1. **Scatter μ_pred vs μ_obs (IPD)** — does the AI's null baseline
     match the per-kmer empirical mean (over ALL observations)? Points
     on the diagonal = perfect calibration. Kmers that carry methylation
     in the corpus drift ABOVE the diagonal because their pool mixes
     unmethylated + methylated observations.

  2. **Above-rate distribution per kmer** — fraction
     ``n_above / n_total`` per kmer. For a perfectly calibrated null
     this should peak around the Gaussian false-positive rate
     (~2.3 % for N = 2 σ, ~0.1 % for N = 3 σ). Kmers with a long
     above-rate tail are candidate methylation-carrying kmers.

  3. **Top-K kmer detail** — for the K kmers with highest above-rate
     (and enough observations), show side by side:
        - μ_pred ± σ_pred  (AI baseline, blue)
        - μ_above ± σ_above  (above-threshold subset mean ± sd, red)
        - threshold = μ_pred + N · σ_pred  (black ×)
     A real methylation signal looks like ``μ_above >> threshold``
     with a narrow σ_above — the outliers form a coherent second mode.

Use ``--pattern GATC`` to restrict the top-K panel to kmers that
contain a given substring (e.g. all kmers with the Dam motif).

Usage::

    python -m kinsim_baseline plot-per-kmer OUTPUT_DIR
        [--top-k 30] [--min-obs 100] [--pattern GATC]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from kinsim.utils.encoding import K, decode_kmer

log = logging.getLogger(__name__)

N_KMERS = 4 ** K


def _load(out_dir: Path) -> dict:
    path = out_dir / "per_kmer_observed.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — run `per-kmer` first.")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _build_per_kmer_hist_figure(
    d: dict, derived: dict, top_idx: np.ndarray,
):
    """Multi-subplot figure: for each top kmer, plot the empirical IPD
    histogram (from the observed BAM data) overlaid with the AI's predicted
    Gaussian. This is the "smoking gun" view — bimodal distributions visible
    here imply the kmer carries methylation events in the corpus.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if "hist_ipd" not in d or top_idx.size == 0:
        return None

    hist_ipd = d["hist_ipd"]                                  # (N_KMERS, n_bins)
    n_bins = int(d.get("hist_n_bins", hist_ipd.shape[1]))
    bin_width = 256 / n_bins
    bin_centers = (np.arange(n_bins) + 0.5) * bin_width       # IPD-value at bin centre

    mu_pred = d["mu_pred_ipd"]
    sigma_pred = d["sigma_pred_ipd"]
    mu_obs = derived["mu_obs"]
    sigma_obs = derived["sigma_obs"]
    n_total = derived["n_total"]
    above_rate = derived["above_rate"]

    # Pretty-print up to 12 kmers in a 3×4 grid
    n_panels = min(top_idx.size, 12)
    idx = top_idx[:n_panels]
    rows = int(np.ceil(n_panels / 3))
    titles = [
        f"{decode_kmer(int(k))}  n={int(n_total[k])}  above={above_rate[k]*100:.2f}%"
        for k in idx
    ]

    fig = make_subplots(rows=rows, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    x_smooth = np.linspace(0, 255, 512)
    for j, k in enumerate(idx):
        r = j // 3 + 1
        c = j % 3 + 1
        h = hist_ipd[k].astype(np.float64)
        if h.sum() == 0:
            continue
        # Empirical density (so it overlays cleanly with the Gaussian PDF)
        h_density = h / (h.sum() * bin_width)
        fig.add_trace(go.Bar(
            x=bin_centers, y=h_density, width=bin_width * 0.95,
            marker={"color": "#7f8c8d"}, name="observed",
            showlegend=(j == 0),
        ), row=r, col=c)

        # Predicted Gaussian density
        mu_p = float(mu_pred[k])
        sg_p = max(float(sigma_pred[k]), 1e-3)
        y_pred = (
            1.0 / (sg_p * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x_smooth - mu_p) / sg_p) ** 2)
        )
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_pred, mode="lines",
            line={"color": "#3498db", "width": 2}, name="AI predicted",
            showlegend=(j == 0),
        ), row=r, col=c)

        # Empirical Gaussian (μ_obs, σ_obs) — to make the spread comparison obvious
        mu_o = float(mu_obs[k])
        sg_o = max(float(sigma_obs[k]), 1e-3)
        y_obs = (
            1.0 / (sg_o * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x_smooth - mu_o) / sg_o) ** 2)
        )
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_obs, mode="lines",
            line={"color": "#e74c3c", "width": 2, "dash": "dash"},
            name="empirical Gaussian", showlegend=(j == 0),
        ), row=r, col=c)

        fig.update_xaxes(title_text="IPD" if r == rows else "",
                         range=[0, max(60, mu_p + sg_p * 5)], row=r, col=c)
        fig.update_yaxes(title_text="density" if c == 1 else "", row=r, col=c)

    fig.update_layout(
        height=320 * rows, template="plotly_white",
        title={"text": "Per-kmer empirical IPD vs AI prediction",
               "x": 0.5, "xanchor": "center"},
        bargap=0.0, margin={"t": 80, "b": 40, "l": 60, "r": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.08,
                "xanchor": "center", "x": 0.5},
    )
    return fig


def _derive(d: dict) -> dict:
    """Compute μ/σ (all obs) and μ/σ (above-threshold) per kmer."""
    n = d["n_total"].astype(np.float64)
    safe_n = np.where(n > 0, n, 1.0)

    mu_obs = d["sum_obs_ipd"] / safe_n
    var_obs = np.clip(d["sum2_obs_ipd"] / safe_n - mu_obs ** 2, 0.0, None)
    sigma_obs = np.sqrt(var_obs)

    n_a = d["n_above_ipd"].astype(np.float64)
    safe_a = np.where(n_a > 0, n_a, 1.0)
    mu_above = d["sum_above_ipd"] / safe_a
    var_above = np.clip(d["sum2_above_ipd"] / safe_a - mu_above ** 2, 0.0, None)
    sigma_above = np.sqrt(var_above)

    above_rate = np.where(n > 0, n_a / safe_n, 0.0)
    return {
        "mu_obs": mu_obs.astype(np.float32),
        "sigma_obs": sigma_obs.astype(np.float32),
        "mu_above": mu_above.astype(np.float32),
        "sigma_above": sigma_above.astype(np.float32),
        "above_rate": above_rate.astype(np.float32),
        "n_total": n.astype(np.int64),
        "n_above": n_a.astype(np.int64),
    }


def _filter_pattern(kmer_ids: np.ndarray, pattern: str | None) -> np.ndarray:
    """Return the subset of ``kmer_ids`` whose decoded kmer contains ``pattern``."""
    if not pattern:
        return kmer_ids
    pat = pattern.upper()
    keep = np.array(
        [pat in decode_kmer(int(k)) for k in kmer_ids],
        dtype=bool,
    )
    return kmer_ids[keep]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_figure(d, derived, top_k: int, min_obs: int, pattern: str | None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_total = derived["n_total"]
    above_rate = derived["above_rate"]
    mu_pred = d["mu_pred_ipd"]
    sigma_pred = d["sigma_pred_ipd"]
    mu_obs = derived["mu_obs"]
    mu_above = derived["mu_above"]
    sigma_above = derived["sigma_above"]
    threshold_factor = float(d["threshold_factor"])

    has_obs = n_total >= min_obs
    n_covered = int(has_obs.sum())

    # Subsample for the scatter panel
    rng = np.random.default_rng(0)
    idx_pool = np.where(has_obs)[0]
    if idx_pool.size > 100_000:
        idx_s = rng.choice(idx_pool, size=100_000, replace=False)
    else:
        idx_s = idx_pool

    # Top-K candidates (with pattern filter applied)
    cand = idx_pool
    if pattern:
        cand = _filter_pattern(cand, pattern)
        log.info("Pattern '%s' matches %d kmers (of %d with n≥%d)",
                 pattern, cand.size, idx_pool.size, min_obs)
    if cand.size:
        order = np.argsort(-above_rate[cand])
        top_idx = cand[order[:top_k]]
    else:
        top_idx = np.empty(0, dtype=np.int64)
        log.warning("No kmers passed the filter — top-K panel will be empty.")

    title_pattern = f" — pattern '{pattern}'" if pattern else ""

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            (f"μ_pred (AI baseline) vs μ_obs (BAM, all observations) — "
             f"{idx_s.size:,} kmers shown (of {n_covered:,} with n≥{min_obs})"),
            (f"Above-rate per kmer — fraction of observations > μ_pred + "
             f"{threshold_factor:.1f} · σ_pred"),
            (f"Top {top_k} kmers by above-rate{title_pattern} — "
             "AI baseline (blue) vs above-threshold population (red)"),
        ],
        specs=[[{"type": "scatter"}], [{"type": "histogram"}], [{"type": "scatter"}]],
        vertical_spacing=0.10, row_heights=[0.32, 0.22, 0.46],
    )

    # 1. Scatter μ_pred vs μ_obs
    fig.add_trace(go.Scattergl(
        x=mu_pred[idx_s], y=mu_obs[idx_s], mode="markers",
        marker={
            "size": 2,
            "color": np.log10(np.maximum(n_total[idx_s], 1)),
            "colorscale": "Viridis", "showscale": True,
            "colorbar": {"title": "log10(n obs)", "len": 0.28, "y": 0.86},
        },
        text=[f"n={int(c):,}" for c in n_total[idx_s]],
        hovertemplate="μ_pred=%{x:.1f}<br>μ_obs=%{y:.1f}<br>%{text}<extra></extra>",
        name="kmers", showlegend=False,
    ), row=1, col=1)
    lim_max = float(max(np.nanmax(mu_pred[idx_s]), np.nanmax(mu_obs[idx_s])))
    fig.add_trace(go.Scatter(
        x=[0, lim_max], y=[0, lim_max], mode="lines",
        line={"color": "red", "dash": "dash"}, name="y = x",
    ), row=1, col=1)
    fig.update_xaxes(title_text="μ_pred (AI baseline, uint8 frames)", row=1, col=1)
    fig.update_yaxes(title_text="μ_obs (empirical, uint8 frames)", row=1, col=1)

    # 2. Above-rate histogram (log y to see the tail)
    fig.add_trace(go.Histogram(
        x=above_rate[has_obs], nbinsx=120,
        marker={"color": "#e74c3c"}, showlegend=False,
    ), row=2, col=1)
    fig.update_xaxes(title_text="fraction of obs above threshold", row=2, col=1)
    fig.update_yaxes(title_text="# kmers (log)", type="log", row=2, col=1)

    # 3. Top-K kmer detail panel
    if top_idx.size > 0:
        labels = [
            f"{decode_kmer(int(k))}  n={int(n_total[k])}  rate={above_rate[k]*100:.2f}%"
            for k in top_idx
        ]
        x_lab = np.arange(top_idx.size)

        fig.add_trace(go.Scatter(
            x=x_lab, y=mu_pred[top_idx],
            error_y={"type": "data", "array": sigma_pred[top_idx],
                     "color": "#3498db", "thickness": 1.5},
            mode="markers",
            marker={"color": "#3498db", "size": 9, "symbol": "circle"},
            name="μ_pred ± σ_pred  (AI baseline)",
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=x_lab, y=mu_above[top_idx],
            error_y={"type": "data", "array": sigma_above[top_idx],
                     "color": "#e74c3c", "thickness": 1.5},
            mode="markers",
            marker={"color": "#e74c3c", "size": 9, "symbol": "diamond"},
            name="μ_above ± σ_above  (outlier population)",
        ), row=3, col=1)

        thr_top = mu_pred[top_idx] + threshold_factor * sigma_pred[top_idx]
        fig.add_trace(go.Scatter(
            x=x_lab, y=thr_top,
            mode="markers",
            marker={"color": "black", "size": 6, "symbol": "x"},
            name=f"threshold = μ_pred + {threshold_factor:.1f} σ_pred",
        ), row=3, col=1)

        fig.update_xaxes(
            title_text="kmer",
            tickmode="array", tickvals=x_lab.tolist(), ticktext=labels,
            tickangle=-60, row=3, col=1,
        )
        fig.update_yaxes(title_text="IPD (uint8 frames)", row=3, col=1)

    fig.update_layout(
        height=1400, template="plotly_white",
        title={
            "text": (f"Per-kmer outliers vs AI baseline — "
                     f"threshold = μ_pred + {threshold_factor:.1f} σ_pred"),
            "x": 0.5, "xanchor": "center",
        },
        margin={"t": 80, "b": 120, "l": 70, "r": 30},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.04,
                "xanchor": "center", "x": 0.5},
    )
    return fig


# ---------------------------------------------------------------------------
# Summary TSV (top kmers by above-rate)
# ---------------------------------------------------------------------------


def write_top_tsv(
    d: dict, derived: dict, out_dir: Path,
    n_rows: int = 500, min_obs: int = 100, pattern: str | None = None,
) -> None:
    """Top-N kmers by above-rate, with predicted vs above-threshold stats."""
    n_total = derived["n_total"]
    above_rate = derived["above_rate"]
    mu_pred = d["mu_pred_ipd"]
    sigma_pred = d["sigma_pred_ipd"]
    mu_above = derived["mu_above"]
    sigma_above = derived["sigma_above"]
    threshold_factor = float(d["threshold_factor"])

    has_obs = n_total >= min_obs
    cand = np.where(has_obs)[0]
    if pattern:
        cand = _filter_pattern(cand, pattern)
    if cand.size == 0:
        log.warning("No kmers passed filter — skipping top-N TSV")
        return

    order = np.argsort(-above_rate[cand])
    top_idx = cand[order[:n_rows]]

    suffix = f"_{pattern}" if pattern else ""
    path = out_dir / f"per_kmer_top{suffix}.tsv"

    lines = [
        "\t".join([
            "kmer", "kmer_id", "n_total", "n_above", "above_rate",
            "mu_pred", "sigma_pred", "threshold",
            "mu_above", "sigma_above", "delta_above",
        ])
    ]
    for k in top_idx:
        thr = float(mu_pred[k] + threshold_factor * sigma_pred[k])
        lines.append("\t".join([
            decode_kmer(int(k)),
            str(int(k)),
            str(int(n_total[k])),
            str(int(derived["n_above"][k])),
            f"{float(above_rate[k]):.5f}",
            f"{float(mu_pred[k]):.3f}",
            f"{float(sigma_pred[k]):.3f}",
            f"{thr:.3f}",
            f"{float(mu_above[k]):.3f}",
            f"{float(sigma_above[k]):.3f}",
            f"{float(mu_above[k]) - thr:.3f}",
        ]))
    path.write_text("\n".join(lines) + "\n")
    log.info("Saved: %s", path)


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline plot-per-kmer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_dir",
                   help="Directory containing per_kmer_observed.npz "
                        "(from the per-kmer step).")
    p.add_argument("--top-k", type=int, default=30,
                   help="Number of top kmers to show in the detail panel "
                        "(default 30).")
    p.add_argument("--min-obs", type=int, default=100,
                   help="Drop kmers with fewer total observations (default 100).")
    p.add_argument("--pattern", type=str, default=None,
                   help="Filter the top-K panel to kmers containing this substring "
                        "(e.g. 'GATC' for Dam, 'CCWGG' is not valid as a substring — "
                        "use a literal pattern like 'CCAGG' / 'CCTGG').")
    p.add_argument("--top-n-tsv", type=int, default=500,
                   help="How many rows to dump in the top-kmers TSV (default 500).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    out_dir = Path(args.output_dir)
    d = _load(out_dir)
    derived = _derive(d)

    log.info("Loaded per_kmer_observed.npz")
    log.info("  n_kmers covered (>=%d obs): %d",
             args.min_obs, int((derived["n_total"] >= args.min_obs).sum()))

    try:
        import plotly  # noqa: F401
    except ImportError:
        log.error("plotly not installed — `pip install plotly`")
        raise SystemExit(1)

    fig = build_figure(
        d, derived,
        top_k=args.top_k, min_obs=args.min_obs, pattern=args.pattern,
    )

    import plotly.io as pio
    out_html = out_dir / "per_kmer_outliers.html"
    if args.pattern:
        out_html = out_dir / f"per_kmer_outliers_{args.pattern}.html"
    pio.write_html(
        fig, str(out_html),
        include_plotlyjs="cdn", full_html=True,
        config={"responsive": True, "displayModeBar": True},
    )
    log.info("Saved: %s", out_html)

    # Additional per-kmer histogram view (top-12 kmers) — only when the
    # observed histogram was collected. Visual "smoking gun" view.
    if "hist_ipd" in d:
        n_total = derived["n_total"]
        above_rate = derived["above_rate"]
        cand = np.where(n_total >= args.min_obs)[0]
        if args.pattern:
            cand = _filter_pattern(cand, args.pattern)
        if cand.size:
            order = np.argsort(-above_rate[cand])
            top_idx_hist = cand[order[:12]]
            fig_hist = _build_per_kmer_hist_figure(d, derived, top_idx_hist)
            if fig_hist is not None:
                hist_suffix = f"_{args.pattern}" if args.pattern else ""
                hist_html = out_dir / f"per_kmer_histograms{hist_suffix}.html"
                pio.write_html(
                    fig_hist, str(hist_html),
                    include_plotlyjs="cdn", full_html=True,
                    config={"responsive": True, "displayModeBar": True},
                )
                log.info("Saved: %s", hist_html)

    write_top_tsv(
        d, derived, out_dir,
        n_rows=args.top_n_tsv, min_obs=args.min_obs, pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
