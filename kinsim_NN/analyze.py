"""``kinsim_nn analyze`` — distribution dashboard for extracted shards.

Walks a directory of ``*_shard.pkl`` files (or a single shard) and produces
an HTML report visualising:

  * Per-strain sample counts (BASELINE / SLOWED / NEAR_METH)
  * Per-category IPD distribution (active-strand pooling)
  * Per-(category, parent_meth) IPD distribution — does m6A SLOWED look
    different from m5C SLOWED?
  * Per-parent_offset IPD trajectory for each meth type — confirms the
    expected signature shape (m6A peak at 0 and +5, m5C peak at +2 and +6)
  * Per-strain breakdown table

Designed to be run after the extract array completes (~5 min per shard
walk on a typical 7-8 GB shard) to QC the corpus before training.

CLI::

    python -m kinsim_NN analyze <shards_dir> --output-dir reports/
    python -m kinsim_NN analyze <strain_shard.pkl> --output-dir reports/
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data.shard import (
    CATEGORY_BASELINE,
    CATEGORY_NEAR_METH,
    CATEGORY_SLOWED,
    ShardData,
    read_shard,
)
from .utils.config import load_config, setup_logging
from .utils.pacbio_codec import FRAMES_TABLE


log = logging.getLogger(__name__)


CATEGORY_NAMES = {
    CATEGORY_BASELINE: "BASELINE",
    CATEGORY_SLOWED: "SLOWED",
    CATEGORY_NEAR_METH: "NEAR_METH",
}

# Wong colorblind-safe palette
WONG_PALETTE = {
    "BASELINE": "#000000",     # black
    "SLOWED": "#D55E00",       # vermilion
    "NEAR_METH": "#0072B2",    # blue
}

# Histogram bins for IPD in frames (the natural unit). 50 log-spaced bins
# from 1 to 952 (= FRAMES_TABLE[-1]).
IPD_BIN_EDGES = np.logspace(0, np.log10(952), 51)


# ---------------------------------------------------------------------------
# Per-shard accumulator
# ---------------------------------------------------------------------------


@dataclass
class StrainStats:
    """Per-strain accumulated statistics."""

    strain_id: str
    # category → count
    counts: dict[int, int] = field(default_factory=lambda: dict.fromkeys([0, 1, 2], 0))
    # category → IPD histogram (50 bins)
    ipd_hist: dict[int, np.ndarray] = field(
        default_factory=lambda: {c: np.zeros(50, dtype=np.int64) for c in [0, 1, 2]}
    )
    # (category, parent_meth) → IPD histogram
    ipd_hist_by_meth: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    # (category, parent_meth, parent_offset) → list of IPD values (capped at 5000 per bucket)
    ipd_samples_by_offset: dict[tuple[int, int, int], list[float]] = field(default_factory=dict)
    # (category, parent_meth, parent_offset) → total count
    counts_by_offset: dict[tuple[int, int, int], int] = field(default_factory=dict)


def _active_strand_ipd(shard: ShardData) -> np.ndarray:
    """Pick the centre-position IPD on the strand matching the parent meth.

    For samples on the ``+`` parent strand we use ``IPD_fwd`` (channel 0);
    for ``-`` parent strand we use ``IPD_rev`` (channel 2). For BASELINE
    samples we pick ``IPD_fwd`` deterministically, regardless of the
    random ``strand`` value extract may have stored, because the strand
    has no biological meaning when no methylation is present on either
    channel; using channel 0 keeps the per-strain baseline histograms
    reproducible across re-extractions.

    Returns ``(N,)`` uint8 bytes (PacBio codec); use :data:`FRAMES_TABLE`
    for frames.
    """
    half = shard.k // 2
    sig_center = shard.signal[:, half]      # (N, 4)
    is_baseline = shard.category == CATEGORY_BASELINE
    strand = shard.strand                   # (N,) int8
    out = np.where(strand >= 0, sig_center[:, 0], sig_center[:, 2])
    # Baselines always read channel 0, deterministically.
    out = np.where(is_baseline, sig_center[:, 0], out)
    return out.astype(np.uint8)


def accumulate_shard(shard: ShardData, sample_cap_per_offset_bucket: int = 5000) -> StrainStats:
    """Walk one shard and accumulate per-category, per-meth, per-offset stats."""
    strain_id = shard.meta.get("strain_id", "unknown")
    stats = StrainStats(strain_id=strain_id)

    cats = shard.category
    pmeth = shard.parent_meth
    poff = shard.parent_offset
    ipd_u8 = _active_strand_ipd(shard)
    ipd_frames = FRAMES_TABLE[ipd_u8].astype(np.float32)

    # Offset range comes from the shard metadata, not a hard-coded 0..10.
    # If a future config bumps ``extract.near_meth_max_dist`` past 10, the
    # full range is preserved here. Falls back to 10 for older shards that
    # did not persist the field.
    near_max = int(shard.meta.get("near_meth_max_dist", 10))
    offset_range = list(range(0, near_max + 1))

    for cat in (0, 1, 2):
        mask = cats == cat
        n = int(mask.sum())
        stats.counts[cat] = n
        if n == 0:
            continue
        hist, _ = np.histogram(ipd_frames[mask], bins=IPD_BIN_EDGES)
        stats.ipd_hist[cat] = hist

    # (category, parent_meth) breakdown
    unique_meths = np.unique(pmeth[(cats == CATEGORY_SLOWED) | (cats == CATEGORY_NEAR_METH)])
    for cat in (CATEGORY_SLOWED, CATEGORY_NEAR_METH):
        for m in unique_meths:
            mask = (cats == cat) & (pmeth == m)
            if not mask.any():
                continue
            hist, _ = np.histogram(ipd_frames[mask], bins=IPD_BIN_EDGES)
            stats.ipd_hist_by_meth[(int(cat), int(m))] = hist

    # (category, parent_meth, parent_offset) breakdown — keep a capped
    # subsample of raw values for boxplots
    rng = np.random.default_rng(42)
    for cat in (CATEGORY_SLOWED, CATEGORY_NEAR_METH):
        for m in unique_meths:
            for off in offset_range:
                mask = (cats == cat) & (pmeth == m) & (poff == off)
                n = int(mask.sum())
                if n == 0:
                    continue
                key = (int(cat), int(m), int(off))
                stats.counts_by_offset[key] = n
                values = ipd_frames[mask]
                if values.size > sample_cap_per_offset_bucket:
                    sel = rng.choice(values.size, sample_cap_per_offset_bucket, replace=False)
                    values = values[sel]
                stats.ipd_samples_by_offset[key] = values.tolist()

    return stats


# ---------------------------------------------------------------------------
# Cross-shard combine + report writers
# ---------------------------------------------------------------------------


def write_summary_csv(per_strain: list[StrainStats], out_path: Path) -> None:
    """Write a per-strain category-count CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strain_id", "baseline", "slowed", "near_meth", "total"])
        for s in per_strain:
            b = s.counts.get(CATEGORY_BASELINE, 0)
            sl = s.counts.get(CATEGORY_SLOWED, 0)
            nm = s.counts.get(CATEGORY_NEAR_METH, 0)
            w.writerow([s.strain_id, b, sl, nm, b + sl + nm])
        # Totals row
        totals = {c: sum(s.counts.get(c, 0) for s in per_strain) for c in [0, 1, 2]}
        w.writerow(["TOTAL", totals[0], totals[1], totals[2], sum(totals.values())])
    log.info("Wrote summary CSV: %s", out_path)


def build_html_report(
    per_strain: list[StrainStats],
    meth_name_by_id: dict[int, str],
    out_path: Path,
    near_meth_max_dist: int = 10,
) -> None:
    """Generate an HTML dashboard from accumulated stats.

    ``near_meth_max_dist`` controls the offset range plotted on the
    per-meth-trajectory and counts-heatmap figures. Defaults to 10 so
    older shards that did not persist the field still render the
    historical 0..10 range.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.error("plotly is not installed. pip install -e .[plot]")
        sys.exit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    offsets_range = list(range(0, near_meth_max_dist + 1))
    n_offsets = len(offsets_range)

    # ---- Aggregate across strains ----
    total_counts = dict.fromkeys([0, 1, 2], 0)
    total_ipd_hist = {c: np.zeros(50, dtype=np.int64) for c in [0, 1, 2]}
    total_ipd_hist_by_meth: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(50, dtype=np.int64)
    )
    total_samples_by_offset: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    total_counts_by_offset: dict[tuple[int, int, int], int] = defaultdict(int)

    for s in per_strain:
        for c, v in s.counts.items():
            total_counts[c] += v
        for c, h in s.ipd_hist.items():
            total_ipd_hist[c] = total_ipd_hist[c] + h
        for k, h in s.ipd_hist_by_meth.items():
            total_ipd_hist_by_meth[k] = total_ipd_hist_by_meth[k] + h
        for k, vs in s.ipd_samples_by_offset.items():
            total_samples_by_offset[k].extend(vs)
        for k, n in s.counts_by_offset.items():
            total_counts_by_offset[k] += n

    bin_centers = (IPD_BIN_EDGES[:-1] + IPD_BIN_EDGES[1:]) / 2

    figs_html: list[str] = []

    # ---- Fig 1: per-strain counts stacked bar ----
    fig1 = go.Figure()
    strain_ids = [s.strain_id for s in per_strain]
    for cat_id, cat_name in CATEGORY_NAMES.items():
        fig1.add_trace(go.Bar(
            name=cat_name,
            x=strain_ids,
            y=[s.counts.get(cat_id, 0) for s in per_strain],
            marker_color=WONG_PALETTE[cat_name],
        ))
    fig1.update_layout(
        title="Sample counts per strain (stacked by category)",
        barmode="stack",
        xaxis_title="Strain",
        yaxis_title="Samples",
        height=500,
    )
    figs_html.append(fig1.to_html(full_html=False, include_plotlyjs="cdn"))

    # ---- Fig 2: per-category IPD distribution (overall) ----
    fig2 = go.Figure()
    for cat_id, cat_name in CATEGORY_NAMES.items():
        h = total_ipd_hist[cat_id]
        if h.sum() == 0:
            continue
        fig2.add_trace(go.Scatter(
            x=bin_centers,
            y=h / h.sum(),
            mode="lines",
            name=cat_name,
            line=dict(color=WONG_PALETTE[cat_name], width=2),
        ))
    fig2.update_layout(
        title="IPD distribution per category (corpus-wide, density-normalised)",
        xaxis_title="IPD (frames, log scale)",
        xaxis_type="log",
        yaxis_title="Density",
        height=500,
    )
    figs_html.append(fig2.to_html(full_html=False, include_plotlyjs=False))

    # ---- Fig 3: per-(category, parent_meth) IPD distribution ----
    meth_ids_in_data = sorted({m for (c, m) in total_ipd_hist_by_meth.keys()})
    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("SLOWED", "NEAR_METH"),
    )
    meth_colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7"]
    for col, cat_id in enumerate([CATEGORY_SLOWED, CATEGORY_NEAR_METH], start=1):
        for i, m in enumerate(meth_ids_in_data):
            h = total_ipd_hist_by_meth.get((cat_id, m))
            if h is None or h.sum() == 0:
                continue
            meth_name = meth_name_by_id.get(m, f"meth_{m}")
            fig3.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=h / h.sum(),
                    mode="lines",
                    name=meth_name,
                    line=dict(color=meth_colors[i % len(meth_colors)], width=2),
                    legendgroup=meth_name,
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )
    fig3.update_xaxes(title_text="IPD (frames, log)", type="log")
    fig3.update_yaxes(title_text="Density")
    fig3.update_layout(
        title="IPD distribution per parent methylation type (SLOWED vs NEAR_METH)",
        height=500,
    )
    figs_html.append(fig3.to_html(full_html=False, include_plotlyjs=False))

    # ---- Fig 4: per-parent-offset IPD median (separate panel per meth type) ----
    # Quick sanity-check that signature offsets actually peak in the data
    fig4 = make_subplots(
        rows=1, cols=len(meth_ids_in_data) or 1,
        subplot_titles=[meth_name_by_id.get(m, f"meth_{m}") for m in meth_ids_in_data],
        shared_yaxes=True,
    )
    for col, m in enumerate(meth_ids_in_data, start=1):
        offsets = offsets_range
        slowed_medians, near_meth_medians = [], []
        slowed_ns, near_meth_ns = [], []
        for off in offsets:
            sl_vals = total_samples_by_offset.get((CATEGORY_SLOWED, m, off), [])
            nm_vals = total_samples_by_offset.get((CATEGORY_NEAR_METH, m, off), [])
            slowed_medians.append(float(np.median(sl_vals)) if sl_vals else None)
            near_meth_medians.append(float(np.median(nm_vals)) if nm_vals else None)
            slowed_ns.append(total_counts_by_offset.get((CATEGORY_SLOWED, m, off), 0))
            near_meth_ns.append(total_counts_by_offset.get((CATEGORY_NEAR_METH, m, off), 0))
        fig4.add_trace(
            go.Scatter(
                x=offsets,
                y=slowed_medians,
                mode="markers+lines",
                marker=dict(symbol="circle", size=10, color=WONG_PALETTE["SLOWED"]),
                line=dict(color=WONG_PALETTE["SLOWED"], width=2),
                name="SLOWED",
                legendgroup="SLOWED",
                showlegend=(col == 1),
                hovertext=[f"offset={o} n={n}" for o, n in zip(offsets, slowed_ns)],
            ),
            row=1, col=col,
        )
        fig4.add_trace(
            go.Scatter(
                x=offsets,
                y=near_meth_medians,
                mode="markers+lines",
                marker=dict(symbol="x", size=10, color=WONG_PALETTE["NEAR_METH"]),
                line=dict(color=WONG_PALETTE["NEAR_METH"], width=2, dash="dash"),
                name="NEAR_METH",
                legendgroup="NEAR_METH",
                showlegend=(col == 1),
                hovertext=[f"offset={o} n={n}" for o, n in zip(offsets, near_meth_ns)],
            ),
            row=1, col=col,
        )
    fig4.update_xaxes(title_text="Offset from parent meth (bp)")
    fig4.update_yaxes(title_text="Median IPD (frames)")
    fig4.update_layout(
        title="Median IPD vs offset, per parent methylation type — "
              "v3 signature offsets (m6A: 0, 5 | m4C: 0 | m5C: 2, 6) should show peaks",
        height=500,
    )
    figs_html.append(fig4.to_html(full_html=False, include_plotlyjs=False))

    # ---- Fig 5: counts heatmap per (parent_meth, parent_offset) ----
    fig5_data_slowed = np.zeros((len(meth_ids_in_data), n_offsets), dtype=np.int64)
    fig5_data_nm = np.zeros((len(meth_ids_in_data), n_offsets), dtype=np.int64)
    for i, m in enumerate(meth_ids_in_data):
        for off in offsets_range:
            fig5_data_slowed[i, off] = total_counts_by_offset.get((CATEGORY_SLOWED, m, off), 0)
            fig5_data_nm[i, off] = total_counts_by_offset.get((CATEGORY_NEAR_METH, m, off), 0)
    fig5 = make_subplots(rows=1, cols=2, subplot_titles=("SLOWED counts", "NEAR_METH counts"))
    meth_labels = [meth_name_by_id.get(m, f"meth_{m}") for m in meth_ids_in_data]
    fig5.add_trace(
        go.Heatmap(
            z=fig5_data_slowed,
            x=offsets_range,
            y=meth_labels,
            colorscale="Oranges",
            text=fig5_data_slowed,
            texttemplate="%{text:,}",
            showscale=False,
        ),
        row=1, col=1,
    )
    fig5.add_trace(
        go.Heatmap(
            z=fig5_data_nm,
            x=offsets_range,
            y=meth_labels,
            colorscale="Blues",
            text=fig5_data_nm,
            texttemplate="%{text:,}",
            showscale=False,
        ),
        row=1, col=2,
    )
    fig5.update_xaxes(title_text="Offset (bp from parent)")
    fig5.update_layout(
        title="Sample counts per (parent_meth, parent_offset) — verifies expansion correctness",
        height=400,
    )
    figs_html.append(fig5.to_html(full_html=False, include_plotlyjs=False))

    # ---- Assemble HTML ----
    total = sum(total_counts.values())
    summary_block = f"""
    <h2>Corpus summary</h2>
    <table border=1 cellpadding=6>
      <tr><th>Category</th><th>Samples</th><th>%</th></tr>
      <tr><td>BASELINE</td><td>{total_counts[CATEGORY_BASELINE]:,}</td>
          <td>{100*total_counts[CATEGORY_BASELINE]/max(total,1):.1f}%</td></tr>
      <tr><td>SLOWED</td><td>{total_counts[CATEGORY_SLOWED]:,}</td>
          <td>{100*total_counts[CATEGORY_SLOWED]/max(total,1):.1f}%</td></tr>
      <tr><td>NEAR_METH</td><td>{total_counts[CATEGORY_NEAR_METH]:,}</td>
          <td>{100*total_counts[CATEGORY_NEAR_METH]/max(total,1):.1f}%</td></tr>
      <tr><th>Total</th><th>{total:,}</th><th>100%</th></tr>
    </table>
    <p>{len(per_strain)} strain(s) analysed.</p>
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>kinsim_NN analyze — {len(per_strain)} strains</title>
  <style>
    body {{ font-family: sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ border-bottom: 2px solid #333; }}
    table {{ border-collapse: collapse; margin: 12px 0; }}
    th {{ background: #eee; }}
  </style>
</head>
<body>
  <h1>kinsim_NN analyze report</h1>
  {summary_block}
  {"<hr/>".join(figs_html)}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    log.info("Wrote HTML report: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_shards(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*_shard.pkl"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kinsim_nn analyze", description=__doc__)
    ap.add_argument("input", help="Shard pkl or directory of shards")
    ap.add_argument("--output-dir", default="reports/kinsim_nn_analyze",
                    help="Output directory for HTML + CSV (default: %(default)s)")
    ap.add_argument("--config", default=None, help="kinsim_nn_config.yaml path")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    setup_logging(verbose=args.verbose)

    cfg = load_config(args.config)
    meth_name_by_id = cfg.meth_name_by_id

    input_path = Path(args.input)
    shards = _list_shards(input_path)
    if not shards:
        sys.exit(f"No shards found under {input_path}")
    log.info("Found %d shards under %s", len(shards), input_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_strain: list[StrainStats] = []
    near_meth_max_dist: int = 10
    for i, p in enumerate(shards, 1):
        log.info("[%d/%d] Reading %s", i, len(shards), p.name)
        try:
            shard = read_shard(p)
        except (OSError, EOFError, ValueError) as e:
            log.warning("Skipping %s: %s", p.name, e)
            continue
        if shard.n == 0:
            log.warning("Empty shard: %s", p.name)
            continue
        # All shards in a corpus share the same extraction geometry; take
        # the first non-empty shard's value to drive the plot range.
        if not per_strain:
            near_meth_max_dist = int(shard.meta.get("near_meth_max_dist", 10))
        stats = accumulate_shard(shard)
        log.info("  %s: B=%d  S=%d  N=%d",
                 stats.strain_id,
                 stats.counts.get(CATEGORY_BASELINE, 0),
                 stats.counts.get(CATEGORY_SLOWED, 0),
                 stats.counts.get(CATEGORY_NEAR_METH, 0))
        per_strain.append(stats)
        # Release shard memory before the next one (these are 7-8 GB each)
        del shard

    if not per_strain:
        sys.exit("No usable shards.")

    write_summary_csv(per_strain, out_dir / "per_strain_counts.csv")
    build_html_report(
        per_strain, meth_name_by_id, out_dir / "report.html",
        near_meth_max_dist=near_meth_max_dist,
    )

    log.info("Done. Open %s in a browser.", out_dir / "report.html")


if __name__ == "__main__":
    main()
