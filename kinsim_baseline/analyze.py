"""Plot per-(meth_type, offset) IPD distributions + Gaussian fits.

Reads ``<baseline_dir>/baseline.json`` and produces an HTML with two
stacked panels (both linear y, both LUT-corrected to frames):

  - Top  ▸ **bulk view**, x cut at ``max(p75)`` across buckets. The
    methylation tail is excluded so the y-scale belongs to the
    unmodified bulk → differences between buckets become readable.
  - Bottom ▸ **wide view**, x cut at ``max(p99)`` across buckets. Shows
    the full distribution including the tail where the methylated
    candidates live.

A single Gaussian is also fit per (T, k) and printed to stdout +
written to ``baseline_gaussian.tsv``. Pass ``--gaussian`` to overlay
the Gaussian curves on the plot (off by default — they're visually
distracting because the data is strongly right-skewed, σ > μ).

PacBio IPD encoding
-------------------
PacBio stores IPDs as uint8 codes via a non-uniform LUT (codes 0–63 ↦
1 frame, 64–127 ↦ 2 frames, 128–191 ↦ 4 frames, 192–254 ↦ 8 frames).
Plotting density-per-CODE produces artefactual peaks at the boundaries.
We convert to frames and divide per-bin density by the bin width so
everything is density-per-frame in real physical units.

Output:
    <baseline_dir>/ipd_distributions.html
    <baseline_dir>/baseline_gaussian.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PacBio codeToFramesV1 LUT (precomputed once, vectorised)
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


# ---------------------------------------------------------------------------
# Smoothing + frame-density transform
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stats in frame units
# ---------------------------------------------------------------------------


def _stats_in_frames(hist_256: np.ndarray) -> dict | None:
    """Gaussian moments + p25/p50/p75/p95/p99 in frame space."""
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
        target = q * n
        idx = int(np.searchsorted(cum, target))
        if idx >= 256:
            idx = 255
        return float(FRAME_CENTRES[idx])

    return {
        "n":     n,
        "mean":  mean,
        "sigma": sigma,
        "p25":   at_q(0.25),
        "p50":   at_q(0.50),
        "p75":   at_q(0.75),
        "p95":   at_q(0.95),
        "p99":   at_q(0.99),
    }


# ---------------------------------------------------------------------------
# Plot helpers
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


def _gaussian_curve(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def build_figure(
    signatures: dict, hist_ipd: dict, stats_by_key: dict,
    smooth: int = 2, show_gaussian: bool = False,
):
    """Two-row figure: bulk (cut at max p75) + wide (cut at max p99)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Determine x-axis cuts
    p75_values = [s["p75"] for s in stats_by_key.values() if s]
    p99_values = [s["p99"] for s in stats_by_key.values() if s]
    cut_p75 = max(p75_values) if p75_values else 100.0
    cut_p99 = max(p99_values) if p99_values else 250.0
    # Round up to a clean tick
    cut_p75 = float(np.ceil(cut_p75 / 5.0)) * 5.0
    cut_p99 = float(np.ceil(cut_p99 / 25.0)) * 25.0

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"Bulk view — x ≤ p75 (= {cut_p75:.0f} frames). Differences between buckets visible here.",
            f"Wide view — x ≤ p99 (= {cut_p99:.0f} frames). Includes the methylated-candidate tail.",
        ],
        vertical_spacing=0.18,
    )

    x_lin = np.linspace(0, cut_p99, int(cut_p99) + 1)
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
            label = (f"{key}  n={int(s['n']/1e9):.1f}B  μ={s['mean']:.1f}  "
                     f"p50={s['p50']:.0f}  p99={s['p99']:.0f}")

            # Empirical traces on both rows
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

            if show_gaussian:
                g = _gaussian_curve(x_lin, s["mean"], s["sigma"])
                fig.add_trace(go.Scatter(
                    x=x_lin, y=g, mode="lines",
                    line={"color": color, "width": 1, "dash": "dash"},
                    legendgroup=key, showlegend=False,
                    hovertemplate=(f"{key}  Gaussian<br>IPD=%{{x:.1f}} fr<br>"
                                   f"density=%{{y:.6f}}<extra></extra>"),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=x_lin, y=g, mode="lines",
                    line={"color": color, "width": 1, "dash": "dash"},
                    legendgroup=key, showlegend=False,
                    hovertemplate=(f"{key}  Gaussian<br>IPD=%{{x:.1f}} fr<br>"
                                   f"density=%{{y:.6f}}<extra></extra>"),
                ), row=2, col=1)

    fig.update_xaxes(range=[0, cut_p75], title_text="IPD (frames)", row=1, col=1)
    fig.update_xaxes(range=[0, cut_p99], title_text="IPD (frames)", row=2, col=1)
    fig.update_yaxes(title_text="density per frame", row=1, col=1)
    fig.update_yaxes(title_text="density per frame", row=2, col=1)
    fig.update_layout(
        height=900,
        template="plotly_white",
        title={
            "text": "Per-(meth_type, offset) IPD distributions — LUT-corrected, density per frame",
            "x": 0.02, "xanchor": "left",
            "font": {"size": 16},
        },
        legend={
            "orientation": "v",
            "yanchor": "top",   "y": 1.0,
            "xanchor": "left",  "x": 1.01,
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "#ccc",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        margin={"t": 80, "b": 60, "l": 70, "r": 360},
        hovermode="x unified",
    )
    # Adjust subplot title font + position so they don't fight the legend
    for ann in fig["layout"]["annotations"]:
        ann["font"] = {"size": 12, "color": "#444"}
        ann["xanchor"] = "left"
        ann["x"] = 0.0
    return fig


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def write_gaussian_tsv(signatures: dict, stats_by_key: dict, path: Path) -> None:
    cols = ["meth_type", "offset", "modified_base",
            "n", "mean_frames", "sigma_frames",
            "p25", "p50", "p75", "p95", "p99"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for T, info in signatures.items():
            for k in info["signal_offsets"]:
                key = f"{T}@{k:+d}"
                s = stats_by_key.get(key)
                if s is None:
                    continue
                f.write("\t".join([
                    T, f"{k:+d}", info["modified_base"],
                    str(s["n"]),
                    f"{s['mean']:.3f}",
                    f"{s['sigma']:.3f}",
                    f"{s['p25']:.1f}",
                    f"{s['p50']:.1f}",
                    f"{s['p75']:.1f}",
                    f"{s['p95']:.1f}",
                    f"{s['p99']:.1f}",
                ]) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze(
    baseline_dir: Path, smooth: int = 2, show_gaussian: bool = False,
) -> None:
    baseline_dir = Path(baseline_dir)
    json_path = baseline_dir / "baseline.json"
    if not json_path.is_file():
        log.error("baseline.json not found in %s — did you run `compute` first?",
                  baseline_dir)
        raise SystemExit(1)

    log.info("Loading %s", json_path)
    data = json.loads(json_path.read_text())
    signatures = data["signatures"]
    hist_ipd = data.get("ipd", {})

    stats_by_key: dict[str, dict] = {}
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            h = np.asarray(hist_ipd.get(key, []), dtype=np.float64)
            if h.size == 0:
                continue
            stats_by_key[key] = _stats_in_frames(h)

    # Print equations + percentiles
    log.info("=" * 92)
    log.info("Per-(meth_type, offset) IPD baseline — Gaussian fit + percentiles "
             "(frames, after LUT decoding)")
    log.info("=" * 92)
    log.info("%-10s %-28s  %5s  %5s  %5s  %5s  %5s   %s",
             "bucket", "equation N(μ, σ²)",
             "p25", "p50", "p75", "p95", "p99", "n")
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            s = stats_by_key.get(key)
            if s is None:
                log.info("  %-8s  NO DATA", key)
                continue
            eq = f"N(μ={s['mean']:.2f}, σ={s['sigma']:.2f}²)"
            log.info("  %-8s %-28s  %5.1f  %5.1f  %5.1f  %5.1f  %5.1f   %d",
                     key, eq, s["p25"], s["p50"], s["p75"], s["p95"], s["p99"], s["n"])
    log.info("=" * 92)

    # TSV
    tsv_path = baseline_dir / "baseline_gaussian.tsv"
    write_gaussian_tsv(signatures, stats_by_key, tsv_path)
    log.info("Saved %s", tsv_path)

    # Plot
    try:
        import plotly  # noqa: F401
    except ImportError:
        log.error("plotly not installed — `pip install plotly`")
        raise SystemExit(1)

    fig = build_figure(signatures, hist_ipd, stats_by_key,
                       smooth=smooth, show_gaussian=show_gaussian)

    import plotly.io as pio
    out_path = baseline_dir / "ipd_distributions.html"
    pio.write_html(
        fig, str(out_path),
        include_plotlyjs="cdn", full_html=True,
        config={"responsive": True, "displayModeBar": True},
    )
    log.info("Saved %s", out_path)


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
    p.add_argument("--gaussian", action="store_true",
                   help="Overlay the fitted Gaussian on the plot (off by "
                        "default — bad fit for right-skewed data).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze(Path(args.baseline_dir),
            smooth=args.smooth, show_gaussian=args.gaussian)


if __name__ == "__main__":
    main()
