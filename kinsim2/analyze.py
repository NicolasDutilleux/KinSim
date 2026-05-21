"""``kinsim analyze`` — diagnostic dashboard for bilateral v2 shards.

For a shard pkl (or directory of shards), emit:
  - A summary text report: per-meth counts (baseline / slowed / near_meth)
    per strand, per-kmer top-N kmers by SLOWED count.
  - An HTML page with histograms of fwd-strand IPD vs baseline for the
    most-populated kmers (one panel per meth type).

Usage::

    kinsim analyze <pkl-or-dir> [--output-dir reports/] [--no-html] [--top-n 12]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data.dataset import list_shards, read_shard_extraction_params
from .utils.config import setup_logging
from .utils.encoding import decode_kmer, get_meth_ids
from .utils.sample_layout import (
    CATEGORY_BASELINE,
    CATEGORY_NEAR_METH,
    CATEGORY_SLOWED,
    SampleLayout,
    get_sample_layout,
)

log = logging.getLogger(__name__)


def _iter_shards(path: Path):
    if path.is_dir():
        for shard in list_shards(str(path)):
            with open(shard, "rb") as f:
                yield Path(shard), pickle.load(f)
    else:
        with open(path, "rb") as f:
            yield path, pickle.load(f)


def _accumulate_stats(
    shard_path: Path, data: dict, layout: SampleLayout, stats: dict,
) -> None:
    n_kmers = 0
    cat_counts_fwd = defaultdict(int)
    cat_counts_rev = defaultdict(int)
    meth_counts_fwd = defaultdict(int)
    meth_counts_rev = defaultdict(int)
    slowed_by_kmer = defaultdict(int)
    slowed_ipd_fwd_by_meth: dict[int, list[np.ndarray]] = defaultdict(list)
    baseline_ipd_fwd = []

    for kid, arr in data.items():
        if not isinstance(kid, (int, np.integer)) or not isinstance(arr, np.ndarray):
            continue
        if arr.shape[1] != layout.n_cols:
            continue
        n_kmers += 1
        cat_fwd = arr[:, layout.col_category_fwd].astype(np.int8)
        cat_rev = arr[:, layout.col_category_rev].astype(np.int8)
        pm_fwd = arr[:, layout.col_parent_meth_fwd].astype(np.int8)
        pm_rev = arr[:, layout.col_parent_meth_rev].astype(np.int8)
        ipd_fwd = arr[:, layout.col_ipd_fwd]

        for cid in (CATEGORY_BASELINE, CATEGORY_SLOWED, CATEGORY_NEAR_METH):
            cat_counts_fwd[cid] += int((cat_fwd == cid).sum())
            cat_counts_rev[cid] += int((cat_rev == cid).sum())

        slowed_mask = cat_fwd == CATEGORY_SLOWED
        if slowed_mask.any():
            slowed_by_kmer[int(kid)] += int(slowed_mask.sum())
            for mid in np.unique(pm_fwd[slowed_mask]):
                m_int = int(mid)
                if m_int <= 0:
                    continue
                mask_m = slowed_mask & (pm_fwd == m_int)
                meth_counts_fwd[m_int] += int(mask_m.sum())
                slowed_ipd_fwd_by_meth[m_int].append(ipd_fwd[mask_m].astype(np.float32))

        slowed_mask_rev = cat_rev == CATEGORY_SLOWED
        if slowed_mask_rev.any():
            for mid in np.unique(pm_rev[slowed_mask_rev]):
                m_int = int(mid)
                if m_int > 0:
                    meth_counts_rev[m_int] += int(
                        (slowed_mask_rev & (pm_rev == m_int)).sum()
                    )

        baseline_mask = cat_fwd == CATEGORY_BASELINE
        if baseline_mask.any():
            baseline_ipd_fwd.append(ipd_fwd[baseline_mask].astype(np.float32))

    stats.setdefault("shards", []).append({
        "name": shard_path.stem,
        "n_kmers": n_kmers,
        "cat_counts_fwd": dict(cat_counts_fwd),
        "cat_counts_rev": dict(cat_counts_rev),
        "meth_counts_fwd": dict(meth_counts_fwd),
        "meth_counts_rev": dict(meth_counts_rev),
    })

    for cid, n in cat_counts_fwd.items():
        stats["cat_counts_fwd"][cid] = stats["cat_counts_fwd"].get(cid, 0) + n
    for cid, n in cat_counts_rev.items():
        stats["cat_counts_rev"][cid] = stats["cat_counts_rev"].get(cid, 0) + n
    for mid, n in meth_counts_fwd.items():
        stats["meth_counts_fwd"][mid] = stats["meth_counts_fwd"].get(mid, 0) + n
    for mid, n in meth_counts_rev.items():
        stats["meth_counts_rev"][mid] = stats["meth_counts_rev"].get(mid, 0) + n
    for kid, n in slowed_by_kmer.items():
        stats["slowed_by_kmer"][kid] = stats["slowed_by_kmer"].get(kid, 0) + n
    for mid, chunks in slowed_ipd_fwd_by_meth.items():
        stats["slowed_ipd_fwd_by_meth"].setdefault(mid, []).extend(chunks)
    if baseline_ipd_fwd:
        stats["baseline_ipd_fwd"].extend(baseline_ipd_fwd)


def _write_text_report(stats: dict, output_path: Path) -> None:
    meth_names = {v: k for k, v in get_meth_ids().items()}
    lines = ["KinSim2 bilateral analyze report", "=" * 60]
    lines.append(f"Total shards: {len(stats['shards'])}")
    for sh in stats["shards"]:
        lines.append(f"  - {sh['name']}: {sh['n_kmers']} kmers")
    lines.append("")
    lines.append("Per-strand category totals:")
    for strand in ("fwd", "rev"):
        c = stats[f"cat_counts_{strand}"]
        lines.append(
            f"  {strand}:  baseline={c.get(CATEGORY_BASELINE, 0):,}  "
            f"slowed={c.get(CATEGORY_SLOWED, 0):,}  "
            f"near_meth={c.get(CATEGORY_NEAR_METH, 0):,}"
        )
    lines.append("")
    lines.append("Per-strand slowed counts by meth type:")
    for strand in ("fwd", "rev"):
        mc = stats[f"meth_counts_{strand}"]
        for mid, n in sorted(mc.items()):
            name = meth_names.get(mid, f"meth{mid}")
            lines.append(f"  {strand}  {name}: {n:,}")
    lines.append("")
    top_kmers = sorted(stats["slowed_by_kmer"].items(), key=lambda kv: -kv[1])[:20]
    lines.append("Top-20 kmers by fwd-strand SLOWED count:")
    for kid, n in top_kmers:
        lines.append(f"  {decode_kmer(kid)}  ({kid})  n={n:,}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Text report: %s", output_path)


def _write_html(stats: dict, output_path: Path, top_n: int) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.warning("plotly not installed — skipping HTML")
        return

    meth_names = {v: k for k, v in get_meth_ids().items()}
    figs = []

    baseline = (np.concatenate(stats["baseline_ipd_fwd"]) if stats["baseline_ipd_fwd"] else None)
    for mid in sorted(stats["slowed_ipd_fwd_by_meth"].keys()):
        chunks = stats["slowed_ipd_fwd_by_meth"][mid]
        if not chunks:
            continue
        slowed = np.concatenate(chunks)
        name = meth_names.get(mid, f"meth{mid}")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=slowed, nbinsx=80, name=f"slowed {name}",
            marker_color="#E69F00", opacity=0.7,
        ))
        if baseline is not None and len(baseline) > 0:
            fig.add_trace(go.Histogram(
                x=baseline, nbinsx=80, name="baseline",
                marker_color="#56B4E9", opacity=0.5,
            ))
        fig.update_layout(
            title=f"IPD_fwd distribution: {name} (slowed) vs baseline",
            barmode="overlay",
            xaxis_title="IPD (uint8)",
            yaxis_title="count",
        )
        figs.append(fig)

    if not figs:
        log.warning("No slowed data — nothing to plot")
        return

    html_parts = ["<html><head><title>kinsim2 analyze</title></head><body>"]
    html_parts.append("<h1>KinSim2 bilateral analysis</h1>")
    for fig in figs:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    log.info("HTML report: %s", output_path)


def analyze_path(input_path: Path, output_dir: Path, top_n: int, write_html: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = None
    stats = {
        "shards": [],
        "cat_counts_fwd": {},
        "cat_counts_rev": {},
        "meth_counts_fwd": {},
        "meth_counts_rev": {},
        "slowed_by_kmer": {},
        "slowed_ipd_fwd_by_meth": {},
        "baseline_ipd_fwd": [],
    }
    for shard_path, data in _iter_shards(input_path):
        params = read_shard_extraction_params(data)
        cur_layout = get_sample_layout(params)
        if layout is None:
            layout = cur_layout
        elif layout.n_cols != cur_layout.n_cols:
            log.warning(
                "shard %s has different geometry (n_cols=%d, was %d) — skipping",
                shard_path.name, cur_layout.n_cols, layout.n_cols,
            )
            continue
        _accumulate_stats(shard_path, data, layout, stats)

    if layout is None:
        log.error("No shards found at %s", input_path)
        return

    _write_text_report(stats, output_dir / "analyze.txt")
    summary = {
        "cat_counts_fwd": stats["cat_counts_fwd"],
        "cat_counts_rev": stats["cat_counts_rev"],
        "meth_counts_fwd": stats["meth_counts_fwd"],
        "meth_counts_rev": stats["meth_counts_rev"],
        "n_shards": len(stats["shards"]),
        "layout_n_cols": layout.n_cols,
        "kmer_size": layout.kmer_size,
    }
    (output_dir / "analyze.json").write_text(json.dumps(summary, indent=2))
    log.info("JSON summary: %s", output_dir / "analyze.json")

    if write_html:
        _write_html(stats, output_dir / "analyze.html", top_n)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="kinsim analyze", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_path", help="A shard .pkl or a directory of shards")
    p.add_argument("--output-dir", default="reports/", help="Output directory")
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--no-html", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(verbose=args.verbose)
    analyze_path(Path(args.input_path), Path(args.output_dir), args.top_n, not args.no_html)


if __name__ == "__main__":
    main()
