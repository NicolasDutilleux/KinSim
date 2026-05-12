"""Build a presentation-ready HTML dashboard from a ``compute`` output.

Reads ``<baseline_dir>/baseline.json`` (plus ``baseline_summary.tsv`` and
``run_info.json`` if present) and produces a single self-contained HTML
report that tells the story:

  1. We tried a naïve statistical baseline (no ML, no motifs, no kmer
     context): per-(meth_type, offset) IPD/PW distribution from the
     corpus's BAMs.
  2. The distributions across buckets look nearly identical — even the
     "1.3 × mean" threshold-ratio is ~constant at ~2.2 everywhere.
  3. Math of the dilution: real methylation is 1–5% of base positions,
     so the methylated subpopulation contributes ~1–5% to the histogram
     — invisible at the global aggregation level.
  4. A simple Gaussian fit fails (σ > μ → would go negative). The
     distribution is log-normal-ish, dominated by the unmodified bulk.
  5. Conclusion: this justifies kinsim's kmer-aware design. To detect
     methylation in real data you need either kmer context (kinsim
     train) or motif-filtered analysis (kinsim extract).

Output:
    <baseline_dir>/baseline_report.html

The report embeds plotly figures via CDN (single-file HTML, opens
anywhere). All values are in PacBio frames after LUT decoding.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PacBio codeToFramesV1 LUT
# ---------------------------------------------------------------------------


def _code_to_frames_lut() -> np.ndarray:
    frames = np.zeros(256, dtype=np.int64)
    codes = np.arange(256)
    frames[:64]     = codes[:64]
    frames[64:128]  = 64  + (codes[64:128]  - 64)  * 2
    frames[128:192] = 192 + (codes[128:192] - 128) * 4
    frames[192:255] = 448 + (codes[192:255] - 192) * 8
    frames[255]     = 952
    return frames


def _bin_widths_frames() -> np.ndarray:
    w = np.ones(256, dtype=np.float64)
    w[64:128]  = 2.0
    w[128:192] = 4.0
    w[192:256] = 8.0
    return w


FRAMES_X      = _code_to_frames_lut()
BIN_WIDTHS    = _bin_widths_frames()
FRAME_CENTRES = FRAMES_X + BIN_WIDTHS / 2.0


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return y
    k = 2 * window + 1
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(y, kernel, mode="same")


def _to_frame_density(hist_256: np.ndarray, smooth: int = 0) -> np.ndarray:
    hist = hist_256.astype(np.float64)
    n = hist.sum()
    if n <= 0:
        return np.zeros(256, dtype=np.float64)
    if smooth > 0:
        hist = _smooth(hist, smooth)
        s = hist.sum()
        if s > 0:
            hist = hist * (n / s)
    return (hist / n) / BIN_WIDTHS


def _stats_in_frames(hist_256: np.ndarray) -> dict | None:
    """μ, σ, p25/p50/p75/p95/p99 in frame space."""
    hist = hist_256.astype(np.float64)
    n = int(hist.sum())
    if n == 0:
        return None
    frames = FRAME_CENTRES
    mean = float((frames * hist).sum() / n)
    var  = float(((frames - mean) ** 2 * hist).sum() / n)
    sigma = float(np.sqrt(max(var, 0.0)))
    cum = np.cumsum(hist)

    def at_q(q: float) -> float:
        idx = int(np.searchsorted(cum, q * n))
        if idx >= 256:
            idx = 255
        return float(FRAME_CENTRES[idx])

    return {
        "n":     n, "mean":  mean, "sigma": sigma,
        "p25":   at_q(0.25), "p50": at_q(0.50), "p75": at_q(0.75),
        "p95":   at_q(0.95), "p99": at_q(0.99),
    }


def _ratio_above_threshold(hist_256: np.ndarray, threshold_factor: float) -> tuple[float, float, int]:
    """For the histogram, compute ratio = mean_above_threshold / overall_mean.

    Same math as kinsim_baseline.compute's summary — included here to
    show that the ~constant 2.2 ratio is a function of the distribution
    shape (right-skew), NOT of methylation.
    """
    hist = hist_256.astype(np.float64)
    n = int(hist.sum())
    if n == 0:
        return float("nan"), float("nan"), 0
    frames = FRAME_CENTRES
    mean_all = float((frames * hist).sum() / n)
    cutoff = threshold_factor * mean_all
    mask = frames > cutoff
    n_above = int(hist[mask].sum())
    if n_above == 0:
        return mean_all, float("nan"), 0
    mean_above = float((frames[mask] * hist[mask]).sum() / n_above)
    return mean_all, mean_above, n_above


# ---------------------------------------------------------------------------
# Plot helpers (plotly)
# ---------------------------------------------------------------------------


_METH_BASE = {
    "m6A": ["#e74c3c", "#c0392b"],
    "m4C": ["#3498db", "#1f618d"],
    "m5C": ["#27ae60", "#196f3d"],
}
_FALLBACK = ["#7f8c8d", "#34495e", "#9b59b6", "#f39c12"]


def _color_for(meth_type: str, offset_index: int) -> str:
    shades = _METH_BASE.get(meth_type, _FALLBACK)
    return shades[offset_index % len(shades)]


def _distribution_panel(
    signatures: dict, hist_ipd: dict, stats_by_key: dict, smooth: int,
):
    """Two-row figure: bulk (cut at max(p75)) + wide (cut at max(p99))."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    p75s = [s["p75"] for s in stats_by_key.values() if s]
    p99s = [s["p99"] for s in stats_by_key.values() if s]
    cut_p75 = float(np.ceil(max(p75s, default=100.0) / 5.0)) * 5.0
    cut_p99 = float(np.ceil(max(p99s, default=250.0) / 25.0)) * 25.0

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Vue bulk — x ≤ p75 ({cut_p75:.0f} frames). C'est ici qu'on devrait voir la séparation entre buckets.",
            f"Vue large — x ≤ p99 ({cut_p99:.0f} frames). La queue droite contient théoriquement le signal méthylé.",
        ],
        vertical_spacing=0.18,
    )

    for T, info in signatures.items():
        for i, k in enumerate(info["signal_offsets"]):
            key = f"{T}@{k:+d}"
            h = np.asarray(hist_ipd.get(key, []), dtype=np.float64)
            if h.size == 0:
                continue
            s = stats_by_key.get(key)
            if s is None:
                continue
            density = _to_frame_density(h, smooth=smooth)
            color = _color_for(T, i)
            label = (f"{key}   n={int(s['n']/1e9):.1f}B   "
                     f"μ={s['mean']:.1f}fr  p50={s['p50']:.0f}  p99={s['p99']:.0f}")
            fig.add_trace(go.Scatter(
                x=FRAMES_X, y=density, mode="lines", name=label,
                line={"color": color, "width": 2},
                legendgroup=key, showlegend=True,
                hovertemplate=f"{key}<br>IPD=%{{x}} fr<br>density=%{{y:.6f}}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=FRAMES_X, y=density, mode="lines", name=label,
                line={"color": color, "width": 2},
                legendgroup=key, showlegend=False,
                hovertemplate=f"{key}<br>IPD=%{{x}} fr<br>density=%{{y:.6f}}<extra></extra>",
            ), row=2, col=1)

    fig.update_xaxes(range=[0, cut_p75], title_text="IPD (frames)", row=1, col=1)
    fig.update_xaxes(range=[0, cut_p99], title_text="IPD (frames)", row=2, col=1)
    fig.update_yaxes(title_text="density per frame", row=1, col=1)
    fig.update_yaxes(title_text="density per frame", row=2, col=1)
    fig.update_layout(
        height=820,
        template="plotly_white",
        legend={
            "orientation": "v",
            "yanchor": "top", "y": 1.0,
            "xanchor": "left", "x": 1.01,
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "#ccc", "borderwidth": 1,
            "font": {"size": 11},
        },
        margin={"t": 60, "b": 60, "l": 70, "r": 380},
        hovermode="x unified",
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = {"size": 12, "color": "#444"}
        ann["xanchor"] = "left"
        ann["x"] = 0.0
    return fig


def _ratio_bar_figure(signatures: dict, hist_ipd: dict, threshold_factor: float = 1.3):
    """Bar chart showing the ipd_ratio = mean_above / mean_all per (T, k).

    Drives home the punch line: all buckets give ~the same ratio, which
    means the ratio is a function of the **shape** of the distribution,
    not of methylation.
    """
    import plotly.graph_objects as go
    keys, ratios = [], []
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            h = np.asarray(hist_ipd.get(key, []), dtype=np.float64)
            if h.size == 0:
                continue
            mean_all, mean_above, _ = _ratio_above_threshold(h, threshold_factor)
            if mean_all <= 0 or np.isnan(mean_above):
                continue
            keys.append(key)
            ratios.append(mean_above / mean_all)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=keys, y=ratios,
        marker={"color": "#e67e22"},
        text=[f"{r:.2f}" for r in ratios],
        textposition="outside",
    ))
    fig.update_layout(
        title=(f"Ratio mean_above / mean_all à seuil {threshold_factor}× "
               "— quasi-constant car artefact de forme right-skewed"),
        xaxis_title="meth_type @ offset", yaxis_title="ratio",
        template="plotly_white", height=380,
        margin={"t": 60, "b": 80}, yaxis={"range": [0, max(ratios) * 1.2 if ratios else 3]},
    )
    return fig


def _gaussian_vs_empirical_figure(
    signatures: dict, hist_ipd: dict, stats_by_key: dict, smooth: int,
):
    """One subplot per (T, k) showing empirical density + 3 fitted curves.

    Three overlays:

      - **Gaussian (moments)** (red dashed): μ = mean, σ = std. The
        canonical moment-based fit. For right-skewed data this gives
        σ > μ, so the Gaussian is too wide and flat — bad fit.
      - **Gaussian (robust)** (orange dashed): μ = median (p50),
        σ = IQR / 1.349 (the constant 1.349 makes IQR-based σ equal
        the true σ for a real Gaussian). Insensitive to the heavy tail,
        so the curve actually fits the peak.
      - **Log-normal** (green dashed): the correct family for this
        right-skewed shape. Fit on log(IPD + 1): μ_log, σ_log are the
        sample moments in log space.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    buckets = [
        (T, k) for T, info in signatures.items() for k in info["signal_offsets"]
        if stats_by_key.get(f"{T}@{k:+d}") is not None
    ]
    n = len(buckets)
    if n == 0:
        return go.Figure()

    # Pre-compute log-normal params per bucket from the histogram.
    lognormal_params: dict[str, tuple[float, float]] = {}
    log_centres = np.log(FRAME_CENTRES + 1.0)
    for T, k in buckets:
        key = f"{T}@{k:+d}"
        h = np.asarray(hist_ipd[key], dtype=np.float64)
        n_h = h.sum()
        if n_h <= 0:
            continue
        mu_log = float((log_centres * h).sum() / n_h)
        var_log = float(((log_centres - mu_log) ** 2 * h).sum() / n_h)
        sigma_log = float(np.sqrt(max(var_log, 1e-12)))
        lognormal_params[key] = (mu_log, sigma_log)

    titles = []
    for T, k in buckets:
        s = stats_by_key[f"{T}@{k:+d}"]
        iqr = max(s["p75"] - s["p25"], 1e-6)
        sigma_robust = iqr / 1.349
        mu_log, sigma_log = lognormal_params.get(f"{T}@{k:+d}", (0.0, 0.0))
        titles.append(
            f"{T}@{k:+d}    "
            f"N(μ={s['mean']:.1f}, σ²={s['sigma']:.1f}²)   moments=large    "
            f"N(p50={s['p50']:.1f}, σ_iqr²={sigma_robust:.1f}²)   robust    "
            f"LogN(μ_log={mu_log:.2f}, σ_log={sigma_log:.2f})"
        )

    fig = make_subplots(
        rows=(n + 1) // 2, cols=2,
        subplot_titles=titles, vertical_spacing=0.18, horizontal_spacing=0.08,
    )

    x_dense = np.linspace(0.001, 300, 600)
    for i, (T, k) in enumerate(buckets):
        row = i // 2 + 1
        col = i % 2 + 1
        key = f"{T}@{k:+d}"
        h = np.asarray(hist_ipd[key], dtype=np.float64)
        s = stats_by_key[key]
        emp = _to_frame_density(h, smooth=smooth)

        # Empirical
        fig.add_trace(go.Scatter(
            x=FRAMES_X, y=emp, mode="lines",
            name="empirique" if i == 0 else None,
            line={"color": "#34495e", "width": 2}, showlegend=(i == 0),
        ), row=row, col=col)

        # Gaussian (moments) — bad fit, included for reference
        sigma_m = max(s["sigma"], 1e-6)
        gauss_m = np.exp(-0.5 * ((x_dense - s["mean"]) / sigma_m) ** 2) / (
            sigma_m * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(
            x=x_dense, y=gauss_m, mode="lines",
            name="N(moments) — large car σ>μ" if i == 0 else None,
            line={"color": "#e74c3c", "width": 1.2, "dash": "dash"},
            showlegend=(i == 0),
        ), row=row, col=col)

        # Gaussian (robust, IQR-based σ)
        iqr = max(s["p75"] - s["p25"], 1e-6)
        sigma_r = max(iqr / 1.349, 1e-6)
        gauss_r = np.exp(-0.5 * ((x_dense - s["p50"]) / sigma_r) ** 2) / (
            sigma_r * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(
            x=x_dense, y=gauss_r, mode="lines",
            name="N(robust IQR) — fit le peak" if i == 0 else None,
            line={"color": "#f39c12", "width": 1.5, "dash": "dash"},
            showlegend=(i == 0),
        ), row=row, col=col)

        # Log-normal
        mu_log, sigma_log = lognormal_params.get(key, (0.0, 0.0))
        sigma_log = max(sigma_log, 1e-6)
        log_x = np.log(x_dense + 1.0)
        logn = np.exp(-0.5 * ((log_x - mu_log) / sigma_log) ** 2) / (
            (x_dense + 1.0) * sigma_log * np.sqrt(2 * np.pi))
        fig.add_trace(go.Scatter(
            x=x_dense, y=logn, mode="lines",
            name="Log-normale (vrai bon fit)" if i == 0 else None,
            line={"color": "#27ae60", "width": 1.8, "dash": "dot"},
            showlegend=(i == 0),
        ), row=row, col=col)

        fig.update_xaxes(
            range=[0, 200],
            title_text="IPD (frames)" if row == (n + 1) // 2 else None,
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text="density per frame" if col == 1 else None,
            row=row, col=col,
        )

    fig.update_layout(
        height=340 * ((n + 1) // 2),
        template="plotly_white",
        margin={"t": 80, "b": 60, "l": 70, "r": 30},
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = {"size": 10, "color": "#444"}
    return fig


def _dilution_illustration_figure(stats_by_key: dict, signatures: dict):
    """A toy figure showing what we'd EXPECT if 1% of positions were methylated.

    Builds a synthetic histogram = 0.99 × empirical_unmod_shape + 0.01
    × Gaussian_at_3×_mean. Plots:
      - the empirical baseline (in grey)
      - the synthetic "expected if 1% methylated" (red) — should be
        visually indistinguishable from the empirical
    Demonstrates the dilution problem geometrically.
    """
    import plotly.graph_objects as go

    # Use m6A@+0 stats if available (or any first available bucket)
    key = None
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            cand = f"{T}@{k:+d}"
            if stats_by_key.get(cand):
                key = cand
                break
        if key:
            break
    if key is None:
        return go.Figure()
    s = stats_by_key[key]
    mu_unmod = s["mean"]
    sigma_unmod = max(s["sigma"], 1.0)
    mu_mod = 3.0 * mu_unmod  # ×3 = vrai signal m6A attendu
    sigma_mod = sigma_unmod * 0.5

    x = np.linspace(0, 300, 600)
    # log-normal-ish baseline: gamma proxy on x with mode at ~p50, mean at mu_unmod
    # For simplicity, use a normal at (μ_unmod, σ_unmod) — same as the Gaussian fit
    base = np.exp(-0.5 * ((x - mu_unmod) / sigma_unmod) ** 2) / (sigma_unmod * np.sqrt(2 * np.pi))
    mod  = np.exp(-0.5 * ((x - mu_mod) / sigma_mod) ** 2) / (sigma_mod * np.sqrt(2 * np.pi))

    fig = go.Figure()
    for frac, color, name in [
        (0.0,  "#34495e", "0% méthylé  (baseline)"),
        (0.01, "#e74c3c", "1% méthylé  (signal réel attendu)"),
        (0.05, "#f39c12", "5% méthylé  (max biologique)"),
        (0.20, "#9b59b6", "20% méthylé  (impossible — pour comparaison)"),
    ]:
        mix = (1 - frac) * base + frac * mod
        fig.add_trace(go.Scatter(
            x=x, y=mix, mode="lines", name=name,
            line={"color": color, "width": 2 if frac in (0.0, 0.01) else 1.5,
                  "dash": "dot" if frac in (0.05, 0.20) else "solid"},
        ))
    fig.add_vline(x=mu_unmod, line={"color": "gray", "dash": "dot"},
                  annotation_text=f"μ_unmod = {mu_unmod:.0f}", annotation_position="top")
    fig.add_vline(x=mu_mod, line={"color": "gray", "dash": "dot"},
                  annotation_text=f"μ_mod (×3) = {mu_mod:.0f}", annotation_position="top")
    fig.update_layout(
        title=(f"Simulation: si X% des bases étaient méthylées, à quoi ressemblerait la distribution? "
               f"(modèle Normal autour de μ={mu_unmod:.0f} ± σ={sigma_unmod:.0f})"),
        xaxis_title="IPD (frames)", yaxis_title="density",
        template="plotly_white", height=480,
        legend={"orientation": "h", "y": -0.18},
        yaxis_type="log",
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f0f2f5; color: #2c3e50; line-height: 1.55; }
.page { max-width: 1700px; margin: 0 auto; padding: 28px; }
.header { background: linear-gradient(135deg, #1a2a4a, #2c5282); color: #fff;
          padding: 36px 36px; border-radius: 14px; margin-bottom: 28px; }
.header h1 { font-size: 2.1em; margin-bottom: 8px; letter-spacing: -0.5px; }
.header .sub { opacity: .85; font-size: 1.05em; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
             gap: 16px; margin-bottom: 28px; }
.stat-card { background: #fff; padding: 20px; border-radius: 10px;
             box-shadow: 0 2px 10px rgba(0,0,0,.07); text-align: center; }
.stat-card .val { font-size: 1.65em; font-weight: 700; color: #2c5282; }
.stat-card .lbl { color: #718096; margin-top: 6px; font-size: .82em;
                  text-transform: uppercase; letter-spacing: 0.5px; }
.section { background: #fff; padding: 30px; border-radius: 12px;
           box-shadow: 0 2px 10px rgba(0,0,0,.07); margin-bottom: 26px; }
.section h2 { color: #2c5282; border-bottom: 3px solid #3182ce;
              padding-bottom: 12px; margin-bottom: 20px; font-size: 1.45em; }
.section h3 { color: #2c5282; margin-top: 18px; margin-bottom: 12px; font-size: 1.15em; }
.section p { margin-bottom: 14px; max-width: 1100px; color: #2d3748; }
.section ul { margin-left: 28px; margin-bottom: 14px; }
.section li { margin-bottom: 6px; }
.callout { background: #fef5e7; border-left: 5px solid #f39c12;
           padding: 16px 22px; margin: 16px 0 22px 0; border-radius: 6px; }
.callout.bad { background: #fdedec; border-left-color: #c0392b; }
.callout.good { background: #eafaf1; border-left-color: #27ae60; }
.callout p { margin-bottom: 6px; }
.kpi-table { width: 100%; border-collapse: collapse; margin: 12px 0 20px 0;
             font-size: .92em; background: #fff; }
.kpi-table th { background: #2c5282; color: #fff; padding: 10px;
                text-align: left; font-weight: 600; }
.kpi-table td { padding: 10px; border-bottom: 1px solid #e2e8f0; }
.kpi-table tr:nth-child(even) { background: #f7fafc; }
.kpi-table .num { text-align: right; font-family: 'SF Mono', Menlo, monospace; }
code { background: #edf2f7; padding: 2px 6px; border-radius: 4px;
       font-family: 'SF Mono', Menlo, monospace; font-size: 0.92em; color: #c53030; }
.plot-box { width: 100%; min-height: 480px; margin-top: 8px; }
.footer { text-align: center; color: #a0aec0; font-size: .85em; padding: 26px 0; }
.figure-caption { color: #718096; font-size: .85em; margin-top: 6px;
                  font-style: italic; text-align: center; }
"""


def _figure_div(fig, fig_id: str) -> str:
    import plotly.io as pio
    spec = pio.to_json(fig)
    return (
        f"<div id='{fig_id}' class='plot-box'></div>"
        f"<script>(function(){{var s={spec};s.layout=s.layout||{{}};"
        f"s.layout.autosize=true;Plotly.newPlot('{fig_id}',s.data,s.layout,"
        f"{{responsive:true,displayModeBar:true}});}})();</script>"
    )


def _stat_card(val: str, lbl: str) -> str:
    return f'<div class="stat-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def _kpi_table_html(signatures: dict, stats_by_key: dict, hist_ipd: dict,
                    threshold_factor: float = 1.3) -> str:
    rows = []
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            s = stats_by_key.get(key)
            if s is None:
                continue
            h = np.asarray(hist_ipd[key], dtype=np.float64)
            mean_all, mean_above, n_above = _ratio_above_threshold(h, threshold_factor)
            ratio = (mean_above / mean_all) if (mean_all and not np.isnan(mean_above)) else None
            sigma_msg = "σ > μ ⚠" if s["sigma"] > s["mean"] else "σ ≤ μ"
            rows.append(
                f"<tr><td><b>{T}@{k:+d}</b></td>"
                f"<td class='num'>{s['n']:,}</td>"
                f"<td class='num'>{s['mean']:.2f}</td>"
                f"<td class='num'>{s['sigma']:.2f}</td>"
                f"<td class='num'>{sigma_msg}</td>"
                f"<td class='num'>{s['p50']:.0f}</td>"
                f"<td class='num'>{s['p75']:.0f}</td>"
                f"<td class='num'>{s['p95']:.0f}</td>"
                f"<td class='num'>{s['p99']:.0f}</td>"
                f"<td class='num'><b>{ratio:.3f}</b></td>"
                f"</tr>"
            )
    body = "\n".join(rows)
    return (
        "<table class='kpi-table'>"
        "<thead><tr>"
        "<th>bucket</th><th>n</th><th>μ (fr)</th><th>σ (fr)</th><th>σ vs μ</th>"
        "<th>p50</th><th>p75</th><th>p95</th><th>p99</th>"
        f"<th>ratio @ {threshold_factor}×μ</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _build_html(
    baseline_dir: Path, out_path: Path,
    signatures: dict, hist_ipd: dict, stats_by_key: dict,
    run_info: dict | None,
    smooth: int,
) -> None:
    # Header stats
    n_total = sum(int(s["n"]) for s in stats_by_key.values() if s)
    n_buckets = len(stats_by_key)
    n_bams = len((run_info or {}).get("per_bam", {}))
    n_skipped = sum(1 for v in (run_info or {}).get("per_bam", {}).values() if v.get("skipped"))
    elapsed = (run_info or {}).get("elapsed_s")
    elapsed_str = f"{elapsed/60:.0f} min" if elapsed else "—"
    threshold = (run_info or {}).get("threshold", 1.3)

    cards = [
        ("Meth types (YAML)", str(len(signatures))),
        ("Buckets (T, k)", str(n_buckets)),
        ("Total samples", f"{n_total:,}"),
        ("BAMs walked", f"{n_bams - n_skipped} / {n_bams}"),
        ("Walltime compute", elapsed_str),
        ("Threshold ratio", f"{threshold:.2f} × μ"),
    ]
    card_html = "\n".join(_stat_card(v, k) for k, v in cards)

    # Figures
    fig_dist = _distribution_panel(signatures, hist_ipd, stats_by_key, smooth)
    fig_ratio = _ratio_bar_figure(signatures, hist_ipd, threshold)
    fig_gauss = _gaussian_vs_empirical_figure(signatures, hist_ipd, stats_by_key, smooth)
    fig_dilu = _dilution_illustration_figure(stats_by_key, signatures)

    kpi = _kpi_table_html(signatures, stats_by_key, hist_ipd, threshold)

    # First-pick bucket ref (for dilution math reference)
    first_bucket = next(iter(stats_by_key.keys()), "m6A@+0")
    first_stats = stats_by_key.get(first_bucket, {})

    html = f"""<!DOCTYPE html>
<html lang="fr"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kinsim_baseline — Rapport technique</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{_CSS}</style>
</head><body>
<div class="page">

  <div class="header">
    <h1>kinsim_baseline — Distribution IPD par (meth_type, offset)</h1>
    <p class="sub">Stats + visualisations + équations pour le baseline statistique
       (no ML, no motif filter, no kmer context).</p>
  </div>

  <div class="stat-grid">{card_html}</div>

  <div class="section">
    <h2>Statistiques par bucket (frame-space, LUT-decoded)</h2>
    {kpi}
    <p style="color:#718096;font-size:0.9em">σ &gt; μ ⇒ distribution
       right-skewed (log-normale-ish). Une vraie Gaussienne avec σ &gt; μ
       aurait une fraction de masse à IPD &lt; 0 ⇒ pas un bon fit.</p>
  </div>

  <div class="section">
    <h2>Distributions IPD (density per frame)</h2>
    {_figure_div(fig_dist, "fig_dist")}
    <p class="figure-caption">Haut: zoom bulk (x ≤ p75). Bas: vue large
       jusqu'à p99 incluant la queue droite.</p>
  </div>

  <div class="section">
    <h2>Ratio @ seuil {threshold:.2f}×μ par bucket</h2>
    {_figure_div(fig_ratio, "fig_ratio")}
    <p class="figure-caption">Le ratio <code>mean_above_threshold /
       mean_overall</code> est ~constant à 2.2 car il dépend de la forme
       de la distribution (right-skew), pas du signal méthylé.</p>
  </div>

  <div class="section">
    <h2>Empirique vs 3 fits (Gaussienne moments / Gaussienne robuste / Log-normale)</h2>
    {_figure_div(fig_gauss, "fig_gauss")}
    <p class="figure-caption">Trois overlays sur la density empirique (gris):
       <span style="color:#e74c3c">Rouge dashed</span> = Gaussienne
       <code>N(μ=moyenne, σ=stddev)</code> — trop large car la queue gonfle σ.
       <span style="color:#f39c12">Orange dashed</span> = Gaussienne
       <code>N(μ=p50, σ=IQR/1.349)</code> — σ robuste insensible à la queue,
       fit le peak.
       <span style="color:#27ae60">Vert pointillé</span> =
       <b>Log-normale</b> sur <code>log(IPD+1)</code> — fit complet incluant
       la queue. C'est le vrai bon modèle pour ces distributions.</p>
  </div>

  <div class="section">
    <h2>Modèle de dilution: si X% des bases étaient méthylées</h2>
    {_figure_div(fig_dilu, "fig_dilu")}
    <p class="figure-caption">Simulation: <code>density(x) = (1-p) ·
       N(μ_unmod, σ_unmod²) + p · N(μ_mod=3μ_unmod, σ_mod²)</code>.
       À p=1% (biologique), la 2e bosse représente 1% de la masse —
       visible en log y, invisible en linéaire.</p>
  </div>

  <div class="section">
    <h2>Équations utilisées</h2>

    <h3>1. PacBio codeToFramesV1 (LUT décoding)</h3>
    <p>L'IPD est stocké comme uint8 via une LUT non-uniforme:</p>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px;font-family:'SF Mono',Menlo,monospace;font-size:0.88em">
codeToFrames(c) = c                             si  0 ≤ c &lt; 64    (bin width = 1 frame)
                = 64 + (c-64)·2                 si 64 ≤ c &lt; 128   (bin width = 2 frames)
                = 192 + (c-128)·4               si 128 ≤ c &lt; 192  (bin width = 4 frames)
                = 448 + (c-192)·8               si 192 ≤ c ≤ 254  (bin width = 8 frames)
                = 952                            si c = 255         (overflow)</pre>

    <h3>2. Density per frame</h3>
    <p>Si <code>hist[c]</code> = count au code c et N = Σ hist:</p>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
density_per_frame[c] = (hist[c] / N) / bin_width_frames(c)</pre>
    <p>Sans la division par la largeur de bin, on aurait des "pics"
       artificiels aux frontières 64/128/192 dus au doublement de
       largeur du LUT.</p>

    <h3>3. Statistiques par bucket</h3>
    <p>Avec <code>f[c] = code-to-frame centre</code> et <code>w = bin width</code>:</p>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
μ  = Σ f[c]·hist[c] / N
σ² = Σ (f[c]-μ)²·hist[c] / N
p_q (quantile q) = f[c]  tel que  cumsum(hist)[c] ≥ q·N</pre>

    <h3>4. Ratio à seuil (utilisé par compute / summary)</h3>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
cutoff = threshold_factor · μ
n_above = Σ_(c: f[c] &gt; cutoff)   hist[c]
mean_above = Σ_(c: f[c] &gt; cutoff)  f[c]·hist[c] / n_above
ratio = mean_above / μ</pre>

    <h3>5. Gaussian (N(μ, σ²))</h3>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
pdf(x) = (1 / (σ·√(2π))) · exp(-(x-μ)² / (2σ²))
log-likelihood per sample: -log(σ·√(2π)) - (x-μ)² / (2σ²)</pre>

    <h3>6. Gaussian NLL loss (utilisée par kinsim train sur log1p IPD)</h3>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
NLL(θ; x) = 0.5 · (2·log σ + (x - μ)² / σ²)
          = log σ + 0.5·(x - μ)² / σ²        (constante log(2π) omise)</pre>
    <p>Le modèle prédit (μ, log σ) directement; cette forme est numériquement
       stable.</p>

    <h3>7. Log-normale (le vrai bon fit pour ces données)</h3>
    <p>Si <code>X+1 = exp(Y)</code> avec <code>Y ~ N(μ_log, σ_log²)</code>:</p>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
Y = ln(X + 1)
μ_log = Σ Y[bin] · count[bin] / N
σ_log = √( Σ (Y[bin] - μ_log)² · count[bin] / N )

pdf(x) = (1 / ((x+1)·σ_log·√(2π))) · exp(-(ln(x+1) - μ_log)² / (2σ_log²))

mean(X)   ≈ exp(μ_log + σ_log²/2) - 1
median(X) ≈ exp(μ_log) - 1
var(X)    = (exp(σ_log²) - 1) · exp(2μ_log + σ_log²)</pre>
    <p>C'est aussi ce que <code>kinsim train</code> fait (cf.
       <code>log_transform</code>) — entraînement directement en
       <code>ln(IPD + 1)</code> où la distribution est approximativement
       Gaussienne et la loss GNLL est bien comportée.</p>

    <h3>7b. Gaussienne robuste (sigma depuis IQR)</h3>
    <p>Sur des distributions right-skewed, σ-moments est dominée par la queue
       et donne un fit Gaussien trop large. Alternative robuste:</p>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
μ_robust = p50 (médiane)
σ_robust = IQR / 1.349    où IQR = p75 - p25

(le facteur 1.349 est choisi pour que σ_robust = σ_vrai pour une vraie Gaussienne;
 c'est l'écart interquartile divisé par l'écart-type d'une N(0,1) standard.)</pre>

    <h3>8. Modèle de mélange (dilution)</h3>
    <pre style="background:#edf2f7;padding:14px;border-radius:6px">
density(x) = (1 - p) · density_unmod(x) + p · density_mod(x)
avec p ∈ [0.01, 0.05]  (fraction biologique de méthylation)
et density_mod centrée à ~3·μ_unmod (signature m6A typique)</pre>
  </div>

  <div class="footer">
    Généré par <code>python -m kinsim_baseline analyze</code> ·
    Source: <code>{baseline_dir}/baseline.json</code>
  </div>
</div></body></html>
"""

    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze(baseline_dir: Path, smooth: int = 2) -> None:
    baseline_dir = Path(baseline_dir)
    json_path = baseline_dir / "baseline.json"
    if not json_path.is_file():
        log.error("baseline.json not found in %s — did you run `compute` first?",
                  baseline_dir)
        raise SystemExit(1)

    log.info("Loading %s", json_path)
    data = json.loads(json_path.read_text())
    signatures = data["signatures"]
    hist_ipd_raw = data.get("ipd", {})

    hist_ipd: dict[str, np.ndarray] = {
        key: np.asarray(v, dtype=np.int64) for key, v in hist_ipd_raw.items()
    }

    stats_by_key: dict[str, dict] = {}
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            h = hist_ipd.get(key)
            if h is None or h.size == 0:
                continue
            stats_by_key[key] = _stats_in_frames(h)

    try:
        import plotly  # noqa: F401
    except ImportError:
        log.error("plotly not installed — `pip install plotly`")
        raise SystemExit(1)

    run_info = None
    info_path = baseline_dir / "run_info.json"
    if info_path.is_file():
        try:
            run_info = json.loads(info_path.read_text())
        except Exception:
            pass

    # Print equations + percentiles to stdout for the record
    log.info("=" * 88)
    log.info("Per-(meth_type, offset) — frame-space stats")
    log.info("=" * 88)
    log.info("%-10s %5s %5s %5s %5s %5s %5s   %s",
             "bucket", "p25", "p50", "p75", "p95", "p99", "μ→σ", "n")
    for key, s in stats_by_key.items():
        sigma_flag = "σ>μ ⚠" if s["sigma"] > s["mean"] else "σ≤μ"
        log.info("  %-8s %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f/%4.1f %s  n=%d",
                 key, s["p25"], s["p50"], s["p75"], s["p95"], s["p99"],
                 s["mean"], s["sigma"], sigma_flag, s["n"])
    log.info("=" * 88)

    out_path = baseline_dir / "baseline_report.html"
    log.info("Building dashboard at %s ...", out_path)
    _build_html(baseline_dir, out_path, signatures, hist_ipd, stats_by_key,
                run_info, smooth)
    log.info("Done. Open %s in a browser.", out_path)


def main(argv=None):
    from kinsim.utils.config import setup_logging

    p = argparse.ArgumentParser(
        prog="python -m kinsim_baseline analyze",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("baseline_dir",
                   help="Directory containing baseline.json from `compute`.")
    p.add_argument("--smooth", type=int, default=2,
                   help="Moving-average half-window in code space (default 2 → "
                        "5-point smoothing). 0 to disable.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze(Path(args.baseline_dir), smooth=args.smooth)


if __name__ == "__main__":
    main()
