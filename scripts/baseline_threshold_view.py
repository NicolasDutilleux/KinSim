"""Visualize methylation signal emergence at different threshold cutoffs.

Loads a ``baseline.json`` (output of ``kinsim_baseline compute``) and renders
a single HTML with multiple panels:

  Panel 0:    Full distribution per bucket (the "raw" view — signal drowned
              in unmethylated baseline)
  Panel 1..N: Same distributions but with each bucket clipped to bins
              ``> threshold × baseline_mean``. The unmethylated bulk is
              filtered out, leaving only the right-tail. If a real signal
              exists, it stays visible; if it's all noise, the panel
              collapses to near-zero.

So you see DIRECTLY whether the methylation signal emerges from the bulk as
the threshold is raised.

Usage::

    python scripts/baseline_threshold_view.py BASELINE_DIR
        [--thresholds 1.0,1.3,1.5,2.0]
        [--metric ipd|pw]

The baseline reference mean is computed from the ``baseline`` bucket
(the unmethylated control) if present in the JSON; otherwise it's the
weighted mean across all buckets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _hist_mean(h: np.ndarray, bins: np.ndarray) -> float:
    n = h.sum()
    return float((bins * h).sum() / max(n, 1))


def main():
    p = argparse.ArgumentParser(
        prog="python scripts/baseline_threshold_view.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("baseline_dir",
                   help="Directory containing baseline.json (kinsim_baseline compute output)")
    p.add_argument("--thresholds", default="1.0,1.3,1.5,2.0",
                   help="Comma-separated threshold multipliers vs baseline mean")
    p.add_argument("--metric", choices=["ipd", "pw"], default="ipd")
    p.add_argument("--out", default=None, help="Output HTML path (default: in baseline_dir)")
    args = p.parse_args()

    baseline_dir = Path(args.baseline_dir)
    bjson_path = baseline_dir / "baseline.json"
    if not bjson_path.is_file():
        print(f"ERROR: {bjson_path} not found", file=sys.stderr)
        sys.exit(1)

    bj = json.loads(bjson_path.read_text())
    if args.metric not in bj:
        print(f"ERROR: metric '{args.metric}' not in baseline.json. "
              f"Available: {list(bj.keys())}", file=sys.stderr)
        sys.exit(1)

    thresholds = [float(t) for t in args.thresholds.split(",")]
    bins = np.arange(256, dtype=np.float64)
    metric_data = bj[args.metric]
    buckets = list(metric_data.keys())
    print(f"Loaded {len(buckets)} buckets from {bjson_path}: {buckets}")

    # Compute reference baseline mean.
    if "baseline" in buckets:
        ref_hist = np.asarray(metric_data["baseline"], dtype=np.float64)
        ref_mean = _hist_mean(ref_hist, bins)
        ref_label = "baseline bucket"
    else:
        # Aggregate all buckets as reference
        agg = np.zeros(256, dtype=np.float64)
        for b in buckets:
            agg += np.asarray(metric_data[b], dtype=np.float64)
        ref_mean = _hist_mean(agg, bins)
        ref_label = "all-bucket aggregate"
    print(f"Reference mean ({ref_label}): {ref_mean:.2f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed", file=sys.stderr)
        sys.exit(1)

    n_panels = 1 + len(thresholds)
    titles = [f"Full ({args.metric.upper()})"] + [
        f"> {t:.2f} × ref_mean (= {t*ref_mean:.1f})"
        for t in thresholds
    ]
    fig = make_subplots(
        rows=1, cols=n_panels, subplot_titles=titles,
        horizontal_spacing=0.04, shared_yaxes=False,
    )

    # Plot palette — deterministic, distinguishable, colorblind-friendly-ish
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a",
    ]

    for bi, bucket in enumerate(buckets):
        h = np.asarray(metric_data[bucket], dtype=np.float64)
        n_total = h.sum()
        if n_total == 0:
            continue
        density_full = h / n_total
        color = palette[bi % len(palette)]

        # Panel 0: full distribution
        fig.add_trace(go.Scatter(
            x=bins, y=density_full, mode="lines",
            line={"color": color, "width": 1.6},
            name=f"{bucket} (n={int(n_total):,})",
            showlegend=True,
            legendgroup=bucket,
        ), row=1, col=1)

        # Panels 1..N: thresholded views.
        # Show density of the above-threshold subset, renormalised so each
        # bucket's filtered density sums to 1 (the "shape of the right tail").
        for ti, t in enumerate(thresholds, start=2):
            cutoff = t * ref_mean
            mask = bins > cutoff
            n_above = h[mask].sum()
            if n_above == 0:
                continue
            density_above = np.zeros_like(density_full)
            density_above[mask] = h[mask] / n_above
            fig.add_trace(go.Scatter(
                x=bins, y=density_above, mode="lines",
                line={"color": color, "width": 1.6},
                name=bucket, showlegend=False,
                legendgroup=bucket,
                hovertemplate=(
                    f"<b>{bucket}</b><br>"
                    f"threshold={cutoff:.1f}<br>"
                    f"n_above={int(n_above):,} ({100*n_above/n_total:.2f}%)<br>"
                    "%{{x}}: density=%{{y:.4f}}<extra></extra>"
                ),
            ), row=1, col=ti)

    fig.update_layout(
        height=560, template="plotly_white",
        title={
            "text": (f"Methylation signal emergence — {args.metric.upper()} distributions "
                     f"at different threshold cutoffs (ref={ref_label}, μ={ref_mean:.2f})"),
            "x": 0.5, "xanchor": "center",
        },
        legend={"orientation": "v", "yanchor": "top", "y": 1.0,
                "xanchor": "left", "x": 1.02},
        margin={"t": 90, "b": 60, "l": 60, "r": 220},
    )
    for col in range(1, n_panels + 1):
        fig.update_xaxes(title_text=args.metric.upper(), row=1, col=col)
    fig.update_yaxes(title_text="density (renorm. above-cutoff in panels 2+)",
                     row=1, col=1)

    out_path = Path(args.out) if args.out else baseline_dir / "threshold_emergence.html"
    fig.write_html(
        str(out_path), include_plotlyjs="cdn", full_html=True,
        config={"responsive": True, "displayModeBar": True},
    )
    print(f"Saved: {out_path}")

    # Also print a quick stats table per (bucket, threshold)
    print()
    print(f"Above-threshold counts per bucket (ref_mean = {ref_mean:.2f}):")
    print(f"{'bucket':<35} {'n_total':>12} " +
          " ".join(f"{'n>' + f'{t:.1f}×ref':>14}" for t in thresholds))
    print("-" * (47 + 16 * len(thresholds)))
    for bucket in buckets:
        h = np.asarray(metric_data[bucket], dtype=np.float64)
        n_total = int(h.sum())
        if n_total == 0:
            continue
        cols = []
        for t in thresholds:
            cutoff = t * ref_mean
            mask = bins > cutoff
            n_above = int(h[mask].sum())
            pct = 100 * n_above / max(n_total, 1)
            cols.append(f"{n_above:>9,} ({pct:>4.1f}%)")
        print(f"{bucket:<35} {n_total:>12,} " + " ".join(cols))


if __name__ == "__main__":
    main()
