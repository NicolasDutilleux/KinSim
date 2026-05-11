"""Plot per-(meth_type, offset) IPD distributions + Gaussian fits.

Reads ``<baseline_dir>/baseline.json`` and:

  1. Builds a single interactive HTML with two stacked panels (linear y +
     log y) showing every (T, k) IPD distribution in **frame units**,
     LUT-corrected (density per frame) so the apparent "peaks" at PacBio
     code boundaries 64 / 128 / 192 disappear.

  2. Fits ONE Gaussian per (T, k) on the full frame-space distribution
     (the corpus is dominated by unmethylated positions, so the moments
     give a baseline ``N(μ, σ²)`` model in frames). The Gaussian curve is
     overlaid on the plot, and the formula plus empirical percentiles
     (mean, p50, p95, p99) are written to the log.

PacBio IPD encoding
-------------------
PacBio stores IPDs as uint8 codes via a **non-uniform LUT**
(``codeToFramesV1``): codes 0–63 are 1 frame wide, 64–127 are 2 frames,
128–191 are 4 frames, 192–254 are 8 frames. We convert each code to
frames and divide per-bin density by the bin width before plotting and
before computing the Gaussian moments — so everything is in physical
units.

Output:
    <baseline_dir>/ipd_distributions.html
    <baseline_dir>/baseline_gaussian.tsv   per-(T, k) μ, σ, percentiles
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PacBio codeToFramesV1 LUT  (precomputed once, vectorised)
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


FRAMES_X    = _code_to_frames_lut()          # frame value at each code
BIN_WIDTHS  = _bin_widths_frames()           # bin width in frames per code
FRAME_CENTRES = FRAMES_X + BIN_WIDTHS / 2.0  # for stats (centre of bin)


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
# Stats: Gaussian moments + percentiles, all in frame units
# ---------------------------------------------------------------------------


def _stats_in_frames(hist_256: np.ndarray) -> dict | None:
    """Per-(T, k) stats computed in **frame space**.

    Returns dict with: n, mean (μ in frames), sigma (σ), p50, p95, p99.
    """
    hist = hist_256.astype(np.float64)
    n = int(hist.sum())
    if n == 0:
        return None
    frames = FRAME_CENTRES
    mean = float((frames * hist).sum() / n)
    var = float(((frames - mean) ** 2 * hist).sum() / n)
    sigma = float(np.sqrt(max(var, 0.0)))
    cum = np.cumsum(hist)
    def frame_at_quantile(q: float) -> float:
        # Use the per-code cumulative; interpolate the frame at the requested
        # quantile so we get a real physical value, not just a bin index.
        target = q * n
        idx = int(np.searchsorted(cum, target))
        if idx >= 256:
            idx = 255
        return float(FRAME_CENTRES[idx])
    return {
        "n":    n,
        "mean": mean,
        "sigma": sigma,
        "p50":  frame_at_quantile(0.50),
        "p95":  frame_at_quantile(0.95),
        "p99":  frame_at_quantile(0.99),
    }


# ---------------------------------------------------------------------------
# Plot
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
    smooth: int = 0, max_frame: int = 200,
):
    """Two-row plotly figure in frame units with Gaussian overlay."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"IPD distributions — linear y  (smooth={smooth})",
            f"IPD distributions — log y, tail visible  (smooth={smooth})",
        ],
        vertical_spacing=0.12,
    )

    x_dense_linear = np.linspace(0, max_frame, 400)
    x_dense_log    = np.linspace(0, 952,        952)

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
            label = (f"{key}  n={int(s['n']):,}  "
                     f"μ={s['mean']:.1f}  σ={s['sigma']:.1f} fr")

            # Empirical
            mask = FRAMES_X <= max_frame
            fig.add_trace(go.Scatter(
                x=FRAMES_X[mask], y=density[mask], mode="lines", name=label,
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

            # Gaussian overlay (dashed, same colour)
            g_lin = _gaussian_curve(x_dense_linear, s["mean"], s["sigma"])
            g_log = _gaussian_curve(x_dense_log,    s["mean"], s["sigma"])
            fig.add_trace(go.Scatter(
                x=x_dense_linear, y=g_lin, mode="lines",
                line={"color": color, "width": 1.2, "dash": "dash"},
                name=f"{key}  N({s['mean']:.1f}, {s['sigma']:.1f}²)",
                legendgroup=key, showlegend=False,
                hovertemplate=f"{key}  Gaussian<br>IPD=%{{x:.1f}} fr<br>density=%{{y:.6f}}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x_dense_log, y=g_log, mode="lines",
                line={"color": color, "width": 1.2, "dash": "dash"},
                legendgroup=key, showlegend=False,
                hovertemplate=f"{key}  Gaussian<br>IPD=%{{x:.1f}} fr<br>density=%{{y:.6f}}<extra></extra>",
            ), row=2, col=1)

    fig.update_xaxes(title_text="IPD (frames)", row=2, col=1)
    fig.update_xaxes(range=[0, max_frame], row=1, col=1)
    fig.update_yaxes(title_text="density per frame", row=1, col=1)
    fig.update_yaxes(title_text="density per frame (log)", type="log", row=2, col=1)
    fig.update_layout(
        height=900,
        template="plotly_white",
        title=("Per-(meth_type, offset) IPD distributions "
               "— LUT-corrected, Gaussian fit overlaid (dashed)"),
        legend={
            "orientation": "h", "y": 1.08, "x": 0.5, "xanchor": "center",
            "bgcolor": "rgba(255,255,255,0.85)",
        },
        margin={"t": 110, "b": 60, "l": 70, "r": 30},
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# TSV writer
# ---------------------------------------------------------------------------


def write_gaussian_tsv(signatures: dict, stats_by_key: dict, path: Path) -> None:
    cols = ["meth_type", "offset", "modified_base",
            "n", "mean_frames", "sigma_frames", "p50", "p95", "p99"]
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
                    f"{s['p50']:.1f}",
                    f"{s['p95']:.1f}",
                    f"{s['p99']:.1f}",
                ]) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze(baseline_dir: Path, smooth: int = 2, max_frame: int = 200) -> None:
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

    # Per-(T, k) stats in frame units
    stats_by_key: dict[str, dict] = {}
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            h = np.asarray(hist_ipd.get(key, []), dtype=np.float64)
            if h.size == 0:
                continue
            stats_by_key[key] = _stats_in_frames(h)

    # Print equations + percentiles
    log.info("=" * 78)
    log.info("Per-(meth_type, offset) IPD baseline — Gaussian fit + percentiles")
    log.info("(All values in PacBio frames after LUT decoding.)")
    log.info("=" * 78)
    log.info("%-10s  %-30s  %5s  %5s  %5s  %5s",
             "bucket", "equation N(μ, σ²)", "p50", "p95", "p99", "n")
    for T, info in signatures.items():
        for k in info["signal_offsets"]:
            key = f"{T}@{k:+d}"
            s = stats_by_key.get(key)
            if s is None:
                log.info("  %-8s  NO DATA", key)
                continue
            eq = f"N(μ={s['mean']:.2f}, σ={s['sigma']:.2f}²)"
            log.info("  %-8s  %-30s  %5.1f  %5.1f  %5.1f  %d",
                     key, eq, s["p50"], s["p95"], s["p99"], s["n"])
    log.info("=" * 78)

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
                       smooth=smooth, max_frame=max_frame)

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
                        "5-point smoothing). Set to 0 to disable.")
    p.add_argument("--max-frame", type=int, default=200,
                   help="X-axis cap (in frames) for the linear panel (default 200).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze(Path(args.baseline_dir),
            smooth=args.smooth, max_frame=args.max_frame)


if __name__ == "__main__":
    main()
