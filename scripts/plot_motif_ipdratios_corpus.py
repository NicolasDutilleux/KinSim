"""Plot meanIpdRatio distribution across the training motifs corpus.

Aggregates ``meanIpdRatio`` values across all per-strain motif CSVs in a
lineage's training tree, broken down by source (the merged file used as kinsim
input, the per-caller files ``*_motifs_ipdsummary.csv`` and
``*_motifs_jasmine.csv``) and modification type.

Outputs a 1×3 panel PNG (one panel per source) with box+jitter plots per
modification type, plus a stats summary printed to stdout.

Usage::

    python scripts/plot_motif_ipdratios_corpus.py \\
        --lineage-dir /data/...training/Strepto \\
        --manifest manifest_strepto.csv \\
        --output <out.png>
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WONG = {
    "m6A": "#E69F00",
    "m4C": "#56B4E9",
    "m5C": "#009E73",
    "5mC": "#009E73",
    "modified_base": "#999999",
}


def safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_motifs(path: Path) -> list[tuple[str, float]]:
    """Return [(mod_type, meanIpdRatio), ...] from a motif CSV."""
    out: list[tuple[str, float]] = []
    if not path.is_file():
        return out
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                mt = (r.get("modificationType") or r.get("type") or "?").strip()
                ipd = safe_float(r.get("meanIpdRatio") or r.get("ipd_ratio"))
                if ipd is not None and mt:
                    out.append((mt, ipd))
    except (OSError, csv.Error):
        pass
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lineage-dir", required=True,
                    help="e.g. /data/.../training/Strepto")
    ap.add_argument("--manifest", default="manifest_strepto.csv",
                    help="manifest CSV name relative to --lineage-dir")
    ap.add_argument("--output", required=True, help="output PNG")
    args = ap.parse_args(argv)

    root = Path(args.lineage_dir)
    pipeline = root / "pipeline"
    manifest = root / args.manifest
    if not manifest.is_file():
        sys.exit(f"manifest not found: {manifest}")

    strains = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            sid = row.get("sample_id")
            if sid:
                strains.append(sid)
    print(f"manifest = {manifest}: {len(strains)} strains")

    # Source label → list of (strain, csv_path)
    sources: dict[str, list[tuple[str, Path]]] = {
        "merged (manifest source)": [],
        "ipdSummary alone": [],
        "jasmine alone": [],
    }
    for sid in strains:
        merged = root / sid / "motifs.csv"
        if merged.is_file():
            sources["merged (manifest source)"].append((sid, merged))
        ipdsum = pipeline / sid / f"{sid}_motifs_ipdsummary.csv"
        if ipdsum.is_file():
            sources["ipdSummary alone"].append((sid, ipdsum))
        jasm = pipeline / sid / f"{sid}_motifs_jasmine.csv"
        if jasm.is_file():
            sources["jasmine alone"].append((sid, jasm))

    print("\nFiles found per source:")
    for src, items in sources.items():
        print(f"  {src:<28} {len(items):>3} CSVs")

    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for src, items in sources.items():
        for _sid, p in items:
            for mt, ipd in read_motifs(p):
                data[src][mt].append(ipd)

    print("\n=== meanIpdRatio stats by source × modification type ===")
    print(f"  {'source':<28} {'type':>14} {'n':>6} {'median':>8} {'q25':>8} {'q75':>8} {'max':>8}")
    print("  " + "-" * 80)
    for src in sources:
        for mt in sorted(data[src].keys()):
            vals = np.asarray(data[src][mt], dtype=np.float64)
            if vals.size == 0:
                continue
            print(f"  {src:<28} {mt:>14} {len(vals):>6} {np.median(vals):>8.2f} "
                  f"{np.percentile(vals, 25):>8.2f} {np.percentile(vals, 75):>8.2f} "
                  f"{vals.max():>8.2f}")

    type_order = ["m6A", "m4C", "m5C", "5mC", "modified_base"]
    src_order = list(sources.keys())
    fig, axes = plt.subplots(1, len(src_order), figsize=(5.5 * len(src_order), 6),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(42)

    for ax, src in zip(axes[0], src_order):
        d = data[src]
        types_present = [t for t in type_order if t in d and len(d[t]) > 0]
        # Append any unknown types not in the canonical order
        for t in sorted(d.keys()):
            if t not in types_present and len(d[t]) > 0:
                types_present.append(t)
        if not types_present:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(src)
            continue
        box_data = [d[t] for t in types_present]
        colors = [WONG.get(t, "#888") for t in types_present]
        labels = [f"{t}\n(n={len(d[t])})" for t in types_present]
        bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                        showfliers=False, widths=0.55)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for j, (vals, color) in enumerate(zip(box_data, colors), start=1):
            x = j + rng.normal(0.0, 0.06, size=len(vals))
            ax.scatter(x, vals, alpha=0.35, s=10, color=color, edgecolors="none")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("meanIpdRatio")
        ax.set_title(f"{src}\n{sum(len(v) for v in box_data)} motifs",
                     fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Real training corpus meanIpdRatio — {root.name} "
        f"({len(strains)} strains)",
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
