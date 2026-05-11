"""Post-hoc dashboard + GMM analysis from an existing ``compute`` output.

Reads ``<baseline_dir>/baseline.json`` (per-(T, k) × {IPD 1D, PW 1D, joint
IPD×PW 2D} histograms) plus ``run_info.json`` and ``baseline_summary.tsv``,
fits a 2-component Gaussian mixture per (T, k) on IPD and PW, and writes:

    <baseline_dir>/baseline_gmm.tsv      summary with GMM columns
    <baseline_dir>/baseline_gmm.json     full GMM parameters
    <baseline_dir>/dashboard.html        interactive single-page report
    <baseline_dir>/plots/<T>_off<k>.html per-bucket detail page

The dashboard packages every relevant view into one HTML file (plotly via
CDN — single-file output, opens anywhere). Sections:

  - Run info / corpus stats / kinsim_config.yaml signatures
  - Per-(T, k) cards: stats + GMM params + IPD 1D + PW 1D + IPD×PW 2D heatmap
  - Cross-bucket comparison: ratios bar chart, (μ_IPD, μ_PW) scatter,
    methylation-rate (weight_mod) bar

Why GMM? The ``ipd_ratio`` column in ``baseline_summary.tsv`` is a fixed
multiplier × mean cut — same percentile in every right-skewed distribution,
so the ratio is ~constant across (T, k) and tells you nothing about
methylation strength. A 2-component GMM separates the unmodified bulk from
the modified tail:

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
# Summary stats from 1D hist (mean / p50 / p95 / p99)
# ---------------------------------------------------------------------------


def hist_stats(h: np.ndarray) -> dict:
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


# ---------------------------------------------------------------------------
# Summary TSV with GMM columns
# ---------------------------------------------------------------------------


def _fmt(x, fmt="%.4f"):
    return fmt % x if x is not None else "NA"


def write_gmm_summary_tsv(
    signatures: dict, gmm_ipd: dict, gmm_pw: dict, path: Path,
) -> None:
    cols = [
        "meth_type", "offset", "modified_base",
        "n_samples",
        "ipd_mu_unmod", "ipd_sigma_unmod", "ipd_w_unmod",
        "ipd_mu_mod",   "ipd_sigma_mod",   "ipd_w_mod",
        "ipd_ratio_gmm",
        "pw_mu_unmod", "pw_sigma_unmod", "pw_w_unmod",
        "pw_mu_mod",   "pw_sigma_mod",   "pw_w_mod",
        "pw_ratio_gmm",
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
# Sparse joint → dense 256×256
# ---------------------------------------------------------------------------


def _joint_to_dense(sparse: list) -> Optional[np.ndarray]:
    if sparse is None:
        return None
    arr = np.zeros((256, 256), dtype=np.int64)
    if not sparse:
        return arr
    a = np.asarray(sparse, dtype=np.int64)
    arr[a[:, 0], a[:, 1]] = a[:, 2]
    return arr


# ---------------------------------------------------------------------------
# Plot helpers (plotly)
# ---------------------------------------------------------------------------


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 0.5)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def _gmm_components(x: np.ndarray, g: dict) -> tuple[np.ndarray, np.ndarray]:
    d_lo = _gaussian_pdf(x, g["mu_unmod"], g["sigma_unmod"]) * g["weight_unmod"]
    d_hi = _gaussian_pdf(x, g["mu_mod"],   g["sigma_mod"])   * g["weight_mod"]
    return d_lo, d_hi


def _hist1d_figure(hist: np.ndarray, gmm: dict | None, metric: str, key: str):
    import plotly.graph_objects as go
    n = float(hist.sum())
    density = hist / n if n > 0 else hist
    bins = np.arange(256)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bins, y=density, name=f"empirical (n={int(n):,})",
        marker={"color": "steelblue"}, opacity=0.65,
        hovertemplate=f"{metric}=%{{x}}<br>density=%{{y:.5f}}<extra></extra>",
    ))
    if gmm:
        x = np.linspace(0, 255, 512)
        d_lo, d_hi = _gmm_components(x, gmm)
        fig.add_trace(go.Scatter(
            x=x, y=d_lo, mode="lines",
            name=f"unmod  μ={gmm['mu_unmod']:.1f}  σ={gmm['sigma_unmod']:.1f}  w={gmm['weight_unmod']:.1%}",
            line={"color": "green", "width": 2},
        ))
        fig.add_trace(go.Scatter(
            x=x, y=d_hi, mode="lines",
            name=f"mod    μ={gmm['mu_mod']:.1f}  σ={gmm['sigma_mod']:.1f}  w={gmm['weight_mod']:.1%}",
            line={"color": "crimson", "width": 2},
        ))
        fig.add_trace(go.Scatter(
            x=x, y=d_lo + d_hi, mode="lines", name="GMM total",
            line={"color": "black", "width": 1.2, "dash": "dash"},
        ))
    fig.update_layout(
        title=f"{key} — {metric} 1D distribution",
        xaxis_title=f"{metric} bin (uint8)", yaxis_title="density",
        yaxis_type="log", bargap=0, template="plotly_white",
        legend={"orientation": "v", "x": 1.02, "y": 1.0},
        margin={"t": 50, "b": 50, "l": 60, "r": 220},
        height=420,
    )
    return fig


def _heatmap2d_figure(joint: np.ndarray | None, key: str,
                      gmm_ipd: dict | None, gmm_pw: dict | None):
    import plotly.graph_objects as go

    if joint is None or joint.sum() == 0:
        fig = go.Figure()
        fig.update_layout(
            title=f"{key} — IPD × PW (no 2D data in baseline.json — re-run "
                  f"`compute` to capture joint)",
            template="plotly_white", height=420,
        )
        return fig

    # log1p for display so the modified tail is visible
    z = np.log1p(joint.astype(np.float64))
    # Crop to the populated region to avoid 80% empty plot
    nz_rows = np.where(joint.sum(axis=1) > 0)[0]
    nz_cols = np.where(joint.sum(axis=0) > 0)[0]
    if nz_rows.size and nz_cols.size:
        r_lo, r_hi = int(nz_rows.min()), int(nz_rows.max()) + 1
        c_lo, c_hi = int(nz_cols.min()), int(nz_cols.max()) + 1
    else:
        r_lo, r_hi, c_lo, c_hi = 0, 256, 0, 256

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z[r_lo:r_hi, c_lo:c_hi],
        x=np.arange(c_lo, c_hi), y=np.arange(r_lo, r_hi),
        colorscale="Viridis",
        colorbar={"title": "log(1+count)"},
        hovertemplate="IPD=%{y}<br>PW=%{x}<br>log(1+n)=%{z:.2f}<extra></extra>",
    ))

    # Overlay (μ_unmod, μ_unmod_PW) and (μ_mod, μ_mod_PW) as markers
    if gmm_ipd and gmm_pw:
        fig.add_trace(go.Scatter(
            x=[gmm_pw["mu_unmod"]], y=[gmm_ipd["mu_unmod"]],
            mode="markers+text", text=["unmod"], textposition="bottom right",
            marker={"color": "lime", "size": 14, "symbol": "x", "line": {"color": "black", "width": 1.5}},
            name="μ_unmod",
            hovertemplate=f"μ_IPD={gmm_ipd['mu_unmod']:.1f}<br>μ_PW={gmm_pw['mu_unmod']:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[gmm_pw["mu_mod"]], y=[gmm_ipd["mu_mod"]],
            mode="markers+text", text=["mod"], textposition="top right",
            marker={"color": "red", "size": 14, "symbol": "x", "line": {"color": "black", "width": 1.5}},
            name="μ_mod",
            hovertemplate=f"μ_IPD={gmm_ipd['mu_mod']:.1f}<br>μ_PW={gmm_pw['mu_mod']:.1f}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{key} — IPD × PW joint distribution (log color)",
        xaxis_title="PW bin (uint8)", yaxis_title="IPD bin (uint8)",
        template="plotly_white", height=520,
        margin={"t": 50, "b": 50, "l": 60, "r": 60},
    )
    return fig


def _ratio_bar_figure(signatures: dict, gmm_ipd: dict, gmm_pw: dict):
    import plotly.graph_objects as go
    keys, ipd_r, pw_r = [], [], []
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            gi = gmm_ipd.get(key)
            gp = gmm_pw.get(key)
            if gi is None:
                continue
            keys.append(key)
            ipd_r.append(gi["ratio"] if gi.get("ratio") else None)
            pw_r.append(gp["ratio"] if gp and gp.get("ratio") else None)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=keys, y=ipd_r, name="IPD ratio (GMM)",
                        marker={"color": "crimson"},
                        text=[f"{v:.2f}" if v else "" for v in ipd_r],
                        textposition="outside"))
    fig.add_trace(go.Bar(x=keys, y=pw_r,  name="PW ratio (GMM)",
                        marker={"color": "steelblue"},
                        text=[f"{v:.2f}" if v else "" for v in pw_r],
                        textposition="outside"))
    fig.add_hline(y=1.0, line={"color": "gray", "dash": "dot"},
                  annotation_text="ratio = 1 (no signal)")
    fig.update_layout(
        title="μ_mod / μ_unmod ratio per (T, k) — true signal strength",
        xaxis_title="meth_type @ offset", yaxis_title="μ_mod / μ_unmod",
        template="plotly_white", barmode="group", height=420,
        margin={"t": 50, "b": 80},
    )
    return fig


def _weight_mod_bar_figure(signatures: dict, gmm_ipd: dict):
    import plotly.graph_objects as go
    keys, w_mod = [], []
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            gi = gmm_ipd.get(key)
            if gi is None:
                continue
            keys.append(key)
            w_mod.append(gi["weight_mod"] * 100)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=keys, y=w_mod, marker={"color": "darkorange"},
                        text=[f"{v:.1f}%" for v in w_mod], textposition="outside"))
    fig.update_layout(
        title="GMM weight of the 'modified' component (≈ methylation rate)",
        xaxis_title="meth_type @ offset", yaxis_title="weight_mod (%)",
        template="plotly_white", height=420, margin={"t": 50, "b": 80},
    )
    return fig


def _mean_scatter_figure(signatures: dict, gmm_ipd: dict, gmm_pw: dict):
    import plotly.graph_objects as go
    labels, mu_ipd_u, mu_ipd_m, mu_pw_u, mu_pw_m = [], [], [], [], []
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            gi = gmm_ipd.get(key)
            gp = gmm_pw.get(key)
            if not (gi and gp):
                continue
            labels.append(key)
            mu_ipd_u.append(gi["mu_unmod"])
            mu_ipd_m.append(gi["mu_mod"])
            mu_pw_u.append(gp["mu_unmod"])
            mu_pw_m.append(gp["mu_mod"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mu_pw_u, y=mu_ipd_u, mode="markers+text", text=labels, textposition="top right",
        marker={"color": "green", "size": 14}, name="unmod",
    ))
    fig.add_trace(go.Scatter(
        x=mu_pw_m, y=mu_ipd_m, mode="markers+text", text=labels, textposition="bottom right",
        marker={"color": "crimson", "size": 14}, name="mod",
    ))
    # Connect each pair with a faint arrow line
    for i, lab in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[mu_pw_u[i], mu_pw_m[i]], y=[mu_ipd_u[i], mu_ipd_m[i]],
            mode="lines", line={"color": "lightgray", "width": 1, "dash": "dot"},
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        title="(μ_PW, μ_IPD) per bucket — unmod (green) → mod (red) trajectory",
        xaxis_title="μ_PW", yaxis_title="μ_IPD",
        template="plotly_white", height=520, margin={"t": 50},
    )
    return fig


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------


_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f0f2f5; color: #333; }
.page { max-width: 1600px; margin: 0 auto; padding: 24px; }
.header { background: linear-gradient(135deg, #1a2a4a, #2c5282); color: #fff;
          padding: 32px; border-radius: 12px; margin-bottom: 24px; }
.header h1 { font-size: 1.9em; margin-bottom: 6px; }
.header p  { opacity: .85; font-size: .9em; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr));
             gap: 16px; margin-bottom: 24px; }
.stat-card { background: #fff; padding: 18px; border-radius: 10px;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); text-align: center; }
.stat-card .val { font-size: 1.5em; font-weight: 700; color: #2c5282; }
.stat-card .lbl { color: #718096; margin-top: 6px; font-size: .82em; }
.nav { background: #fff; padding: 14px 20px; border-radius: 10px;
       box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 24px;
       display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.nav span { font-weight: 600; color: #4a5568; }
.nav a { color: #3182ce; text-decoration: none; font-size: .88em;
         padding: 3px 10px; border: 1px solid #bee3f8; border-radius: 20px; }
.nav a:hover { background: #ebf8ff; }
.section { background: #fff; padding: 24px; border-radius: 10px;
           box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 24px; }
.section h2 { color: #2c5282; border-bottom: 3px solid #3182ce;
              padding-bottom: 10px; margin-bottom: 18px; }
.bucket-card { background: #fafbfc; padding: 18px; border-radius: 8px;
               margin-bottom: 22px; border-left: 5px solid #3182ce; }
.bucket-card h3 { color: #2c5282; margin-bottom: 12px; }
.bucket-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px,1fr));
               gap: 14px; }
.gmm-table { width: 100%; border-collapse: collapse; margin: 6px 0 14px 0;
             font-size: .9em; background: #fff; }
.gmm-table th { background: #2c5282; color: #fff; padding: 8px;
                text-align: left; font-weight: 600; }
.gmm-table td { padding: 8px; border-bottom: 1px solid #e2e8f0; }
.gmm-table tr:nth-child(even) { background: #f7fafc; }
.gmm-table tr.unmod-row td:first-child { color: #2f855a; font-weight: 600; }
.gmm-table tr.mod-row   td:first-child { color: #c53030; font-weight: 600; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px;
      margin-bottom: 12px; font-size: .9em; }
.kv .k { color: #718096; }
.kv .v { color: #2d3748; font-weight: 500; }
.plot-box { width: 100%; min-height: 420px; }
.footer { text-align: center; color: #a0aec0; font-size: .8em; padding: 20px 0; }
"""


def _stat_card(val: str, lbl: str) -> str:
    return f'<div class="stat-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def _gmm_table_html(g: dict, metric: str) -> str:
    if g is None:
        return f"<p><em>{metric}: GMM fit unavailable</em></p>"
    rows = [
        ("unmod-row", "unmod", g["mu_unmod"], g["sigma_unmod"], g["weight_unmod"] * 100),
        ("mod-row",   "mod",   g["mu_mod"],   g["sigma_mod"],   g["weight_mod"]   * 100),
    ]
    body = "\n".join(
        f"<tr class='{cls}'><td>{name}</td><td>{mu:.2f}</td><td>{sg:.2f}</td><td>{w:.2f}%</td></tr>"
        for cls, name, mu, sg, w in rows
    )
    ratio = g.get("ratio")
    foot = f"<tr><td><strong>ratio</strong></td><td colspan='3'><strong>μ_mod / μ_unmod = {ratio:.3f}</strong></td></tr>" if ratio else ""
    return (
        f"<table class='gmm-table'><thead><tr><th>{metric} component</th>"
        f"<th>μ</th><th>σ</th><th>weight</th></tr></thead>"
        f"<tbody>{body}{foot}</tbody></table>"
    )


def _bucket_kv_html(T: str, k: int, mb: str, stats_ipd: dict, stats_pw: dict) -> str:
    return (
        f"<div class='kv'>"
        f"<div class='k'>meth_type</div><div class='v'>{T}</div>"
        f"<div class='k'>offset</div><div class='v'>{k:+d}</div>"
        f"<div class='k'>modified_base</div><div class='v'>{mb}</div>"
        f"<div class='k'>n samples</div><div class='v'>{stats_ipd['n']:,}</div>"
        f"<div class='k'>IPD mean / p50 / p95 / p99</div>"
        f"<div class='v'>{stats_ipd['mean']:.2f} / {stats_ipd['p50']} / {stats_ipd['p95']} / {stats_ipd['p99']}</div>"
        f"<div class='k'>PW mean / p50 / p95 / p99</div>"
        f"<div class='v'>{stats_pw['mean']:.2f} / {stats_pw['p50']} / {stats_pw['p95']} / {stats_pw['p99']}</div>"
        f"</div>"
    )


def _figure_div(fig, fig_id: str) -> str:
    import plotly.io as pio
    spec = pio.to_json(fig)
    return (
        f"<div id='{fig_id}' class='plot-box'></div>"
        f"<script>(function(){{var s={spec};s.layout=s.layout||{{}};"
        f"s.layout.autosize=true;Plotly.newPlot('{fig_id}',s.data,s.layout,"
        f"{{responsive:true,displayModeBar:true}});}})();</script>"
    )


def _build_dashboard(
    out_path: Path,
    signatures: dict,
    hist_ipd: dict, hist_pw: dict, hist_joint: dict,
    gmm_ipd: dict, gmm_pw: dict,
    run_info: dict | None,
) -> None:
    import plotly  # noqa: F401  (needed before any go.Figure)
    # Stats cards
    n_total = sum(np.sum(hist_ipd[k]) for k in hist_ipd if hist_ipd[k] is not None)
    n_bams = (len(run_info.get("per_bam", {})) if run_info else 0)
    n_skipped = sum(1 for v in (run_info or {}).get("per_bam", {}).values()
                    if v.get("skipped"))
    elapsed = (run_info or {}).get("elapsed_s")
    elapsed_str = f"{elapsed/60:.1f} min" if elapsed else "?"
    cards = [
        ("Meth types", str(len(signatures))),
        ("(T, k) buckets", str(sum(len(v["signal_offsets"]) for v in signatures.values()))),
        ("Total samples", f"{int(n_total):,}"),
        ("BAMs walked", f"{n_bams - n_skipped} / {n_bams}"),
        ("compute walltime", elapsed_str),
        ("Joint 2D", "yes" if any(hist_joint.values()) else "no"),
    ]
    card_html = "\n".join(_stat_card(v, k) for k, v in cards)

    # YAML signatures
    sig_html = "<table class='gmm-table'><thead><tr><th>meth_type</th>"\
               "<th>modified_base</th><th>signal_offsets</th></tr></thead><tbody>"
    for T, info in signatures.items():
        sig_html += f"<tr><td>{T}</td><td>{info['modified_base']}</td>"\
                    f"<td>{info['signal_offsets']}</td></tr>"
    sig_html += "</tbody></table>"

    # Build all bucket cards
    bucket_html_parts = []
    buckets = [(T, k) for T, info in signatures.items() for k in info["signal_offsets"]]
    for idx, (T, k) in enumerate(buckets):
        key = f"{T}@{k:+d}"
        hi = hist_ipd.get(key)
        hp = hist_pw.get(key)
        hj = hist_joint.get(key)
        if hi is None or hp is None:
            continue
        s_i = hist_stats(hi)
        s_p = hist_stats(hp)
        gi = gmm_ipd.get(key)
        gp = gmm_pw.get(key)

        kv = _bucket_kv_html(T, k, signatures[T]["modified_base"], s_i, s_p)
        tab_ipd = _gmm_table_html(gi, "IPD")
        tab_pw  = _gmm_table_html(gp, "PW")
        fig_ipd = _hist1d_figure(hi, gi, "IPD", key)
        fig_pw  = _hist1d_figure(hp, gp, "PW",  key)
        fig_2d  = _heatmap2d_figure(hj, key, gi, gp)

        bucket_html_parts.append(f"""
<div class="bucket-card" id="bucket-{idx}">
  <h3>{key}  —  modified_base = {signatures[T]['modified_base']}</h3>
  {kv}
  <div class="bucket-grid">
    <div>{tab_ipd}{_figure_div(fig_ipd, f"fig_ipd_{idx}")}</div>
    <div>{tab_pw}{_figure_div(fig_pw,  f"fig_pw_{idx}")}</div>
  </div>
  <div style="margin-top:14px">{_figure_div(fig_2d, f"fig_2d_{idx}")}</div>
</div>
""")

    # Cross-bucket comparison plots
    fig_ratio = _ratio_bar_figure(signatures, gmm_ipd, gmm_pw)
    fig_wmod  = _weight_mod_bar_figure(signatures, gmm_ipd)
    fig_means = _mean_scatter_figure(signatures, gmm_ipd, gmm_pw)

    nav_links = " ".join(
        f"<a href='#bucket-{i}'>{f'{T}@{k:+d}'}</a>"
        for i, (T, k) in enumerate(buckets)
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kinsim_baseline — distribution dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{_HTML_CSS}</style>
</head><body>
<div class="page">
  <div class="header">
    <h1>kinsim_baseline distribution dashboard</h1>
    <p>Per-(meth_type, offset) IPD/PW distributions + 2-component GMM fit + 2D joint</p>
  </div>

  <div class="stat-grid">{card_html}</div>

  <div class="nav"><span>Jump to bucket:</span>{nav_links}
    <a href="#section-overview">overview</a>
    <a href="#section-compare">comparison</a>
  </div>

  <div class="section" id="section-overview">
    <h2>kinsim_config.yaml signatures</h2>
    {sig_html}
    <p style="margin-top:12px;color:#4a5568">
      The base each meth type sits on and the downstream offsets that carry the
      kinetic signature are read from <code>kinsim_config.yaml</code>. The
      analysis here walks every BAM, finds every position p where the read base
      matches <code>modified_base</code>, and accumulates IPD/PW at
      <code>p + k</code> for each <code>k</code> in <code>signal_offsets</code>.
    </p>
  </div>

  <div class="section">
    <h2>Per-(T, k) bucket — distribution + GMM + 2D heatmap</h2>
    {''.join(bucket_html_parts)}
  </div>

  <div class="section" id="section-compare">
    <h2>Cross-bucket comparison</h2>
    {_figure_div(fig_ratio, "fig_ratio")}
    {_figure_div(fig_wmod,  "fig_wmod")}
    {_figure_div(fig_means, "fig_means")}
  </div>

  <div class="footer">Generated by <code>python -m kinsim_baseline analyze</code></div>
</div></body></html>"""

    out_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze(baseline_dir: Path, max_samples: int = 200_000) -> None:
    """Read ``baseline.json``, fit per-(T, k) 2-GMM, write summary + dashboard."""
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
    hist_pw_raw  = data.get("pw",  {})
    hist_joint_raw = data.get("joint", {})

    hist_ipd:   dict[str, np.ndarray] = {}
    hist_pw:    dict[str, np.ndarray] = {}
    hist_joint: dict[str, np.ndarray] = {}
    for key in hist_ipd_raw:
        hist_ipd[key]   = np.asarray(hist_ipd_raw[key],   dtype=np.int64)
    for key in hist_pw_raw:
        hist_pw[key]    = np.asarray(hist_pw_raw[key],    dtype=np.int64)
    for key, sparse in hist_joint_raw.items():
        hist_joint[key] = _joint_to_dense(sparse)
    if not hist_joint:
        log.warning("No 'joint' key in baseline.json — 2D heatmaps will say "
                    "'no 2D data'. Re-run `compute` to capture joint histograms.")

    gmm_ipd: dict[str, dict] = {}
    gmm_pw:  dict[str, dict] = {}

    log.info("Fitting 2-GMM per (T, k) on IPD + PW histograms ...")
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            g_i = fit_gmm_from_hist(hist_ipd_raw.get(key, []), max_samples=max_samples)
            g_p = fit_gmm_from_hist(hist_pw_raw.get(key, []),  max_samples=max_samples)
            gmm_ipd[key] = g_i
            gmm_pw[key]  = g_p
            if g_i:
                log.info("  %-8s  IPD  μ_u=%.2f  μ_m=%.2f  w_m=%.2f%%  ratio=%.2f",
                         key, g_i["mu_unmod"], g_i["mu_mod"],
                         g_i["weight_mod"] * 100, g_i["ratio"] or float("nan"))
            else:
                log.warning("  %s  IPD  fit failed (too few samples)", key)

    summary_path = baseline_dir / "baseline_gmm.tsv"
    write_gmm_summary_tsv(signatures, gmm_ipd, gmm_pw, summary_path)
    log.info("Saved %s", summary_path)

    gmm_json = {"signatures": signatures, "ipd": gmm_ipd, "pw": gmm_pw}
    (baseline_dir / "baseline_gmm.json").write_text(json.dumps(gmm_json, indent=2))
    log.info("Saved %s", baseline_dir / "baseline_gmm.json")

    try:
        import plotly  # noqa: F401
    except ImportError:
        log.warning("plotly not installed — skipping dashboard. `pip install plotly`")
        return

    run_info_path = baseline_dir / "run_info.json"
    run_info = None
    if run_info_path.is_file():
        try:
            run_info = json.loads(run_info_path.read_text())
        except Exception as e:
            log.warning("Could not read run_info.json: %s", e)

    dashboard_path = baseline_dir / "dashboard.html"
    log.info("Building dashboard at %s", dashboard_path)
    _build_dashboard(
        dashboard_path, signatures,
        hist_ipd, hist_pw, hist_joint,
        gmm_ipd, gmm_pw,
        run_info,
    )
    log.info("Done. Open %s in a browser.", dashboard_path)


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
    p.add_argument("--max-samples", type=int, default=200_000,
                   help="Cap for GMM sample size per bucket (default 200 000).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze(Path(args.baseline_dir), max_samples=args.max_samples)


if __name__ == "__main__":
    main()
