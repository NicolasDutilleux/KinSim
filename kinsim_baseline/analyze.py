"""Plot per-(meth_type, offset) IPD distributions from a ``compute`` output.

Reads ``<baseline_dir>/baseline.json`` and produces a single interactive
HTML file with two stacked panels:

    Top:    linear y — shows the bulk shape clearly.
    Bottom: log y    — shows the right-tail (modified subset) clearly.

All (T, k) buckets are overlaid as smooth line traces, color-coded by
meth_type. Toggle individual traces with the legend.

Output:
    <baseline_dir>/ipd_distributions.html
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# Stable per-meth_type colour palette (extra offsets within a type get
# darker / lighter shades so they remain visually distinct).
_METH_BASE = {
    "m6A": ["#e74c3c", "#c0392b"],
    "m4C": ["#3498db", "#1f618d"],
    "m5C": ["#27ae60", "#196f3d"],
}
_FALLBACK = ["#7f8c8d", "#34495e", "#9b59b6", "#f39c12"]


def _color_for(meth_type: str, offset_index: int) -> str:
    shades = _METH_BASE.get(meth_type, _FALLBACK)
    return shades[offset_index % len(shades)]


def build_figure(signatures: dict, hist_ipd: dict):
    """Two-row plotly figure: linear y on top, log y on bottom."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "IPD distributions — linear y",
            "IPD distributions — log y (tail visible)",
        ],
        vertical_spacing=0.12,
    )

    bins = np.arange(256)
    for T, info in signatures.items():
        for i, k in enumerate(info["signal_offsets"]):
            key = f"{T}@{k:+d}"
            h = np.asarray(hist_ipd.get(key, []), dtype=np.float64)
            if h.size == 0:
                continue
            n = h.sum()
            density = h / n if n > 0 else h
            color = _color_for(T, i)
            label = f"{key}  (n={int(n):,})"

            fig.add_trace(go.Scatter(
                x=bins, y=density, mode="lines", name=label,
                line={"color": color, "width": 2},
                legendgroup=key, showlegend=True,
                hovertemplate=f"{key}<br>IPD=%{{x}}<br>density=%{{y:.5f}}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bins, y=density, mode="lines", name=label,
                line={"color": color, "width": 2},
                legendgroup=key, showlegend=False,
                hovertemplate=f"{key}<br>IPD=%{{x}}<br>density=%{{y:.5f}}<extra></extra>",
            ), row=2, col=1)

    fig.update_xaxes(title_text="IPD bin (uint8, 0–255)", row=2, col=1)
    fig.update_yaxes(title_text="density",     row=1, col=1)
    fig.update_yaxes(title_text="density (log)", type="log", row=2, col=1)
    fig.update_layout(
        height=900,
        template="plotly_white",
        title="Per-(meth_type, offset) IPD distributions",
        legend={
            "orientation": "h", "y": 1.08, "x": 0.5, "xanchor": "center",
            "bgcolor": "rgba(255,255,255,0.85)",
        },
        margin={"t": 110, "b": 60, "l": 70, "r": 30},
        hovermode="x unified",
    )
    return fig


def analyze(baseline_dir: Path) -> None:
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

    try:
        import plotly  # noqa: F401
    except ImportError:
        log.error("plotly not installed — `pip install plotly`")
        raise SystemExit(1)

    fig = build_figure(signatures, hist_ipd)

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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)

    analyze(Path(args.baseline_dir))


if __name__ == "__main__":
    main()
