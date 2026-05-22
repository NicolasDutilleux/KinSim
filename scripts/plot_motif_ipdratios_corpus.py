"""Plot meanIpdRatio distribution across the training motifs corpus.

Reads one or more KinSim-style manifests (CSV with at least
``sample_id`` and ``motifs`` columns) and aggregates the ``meanIpdRatio``
column from every motif CSV they point to, broken down by modification type.

Outputs a per-source PNG (1 panel per manifest given) with box + jitter
plots per modification type, plus a stats summary printed to stdout.

Usage::

    python scripts/plot_motif_ipdratios_corpus.py \\
        --manifests "kinsim input:$PREFIX/manifest.csv" \\
        --output <out.png>

    # Multiple sources side-by-side
    python scripts/plot_motif_ipdratios_corpus.py \\
        --manifests "kinsim:$PREFIX/manifest.csv,source:/training/Strepto/manifest_strepto.csv" \\
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
                mt = (r.get("modificationType") or r.get("type") or "").strip()
                ipd = safe_float(r.get("meanIpdRatio") or r.get("ipd_ratio"))
                if ipd is not None and mt:
                    out.append((mt, ipd))
    except (OSError, csv.Error):
        pass
    return out


def collect_from_manifest(manifest_path: Path, motifs_col: str) -> tuple[int, int, dict[str, list[float]]]:
    """Returns (n_rows, n_csvs_found, {mod_type: [ipd_ratios]})."""
    by_type: dict[str, list[float]] = defaultdict(list)
    n_rows = 0
    n_found = 0
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            n_rows += 1
            p = (row.get(motifs_col) or "").strip()
            if not p:
                continue
            csv_path = Path(p)
            if not csv_path.is_file():
                continue
            n_found += 1
            for mt, ipd in read_motifs(csv_path):
                by_type[mt].append(ipd)
    return n_rows, n_found, by_type


def parse_manifests_arg(spec: str) -> list[tuple[str, Path]]:
    """Parse 'label1:path1,label2:path2' into [(label, Path), ...]."""
    out: list[tuple[str, Path]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            out.append((Path(item).name, Path(item)))
        else:
            label, path = item.split(":", 1)
            out.append((label.strip(), Path(path.strip())))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--manifests", required=True,
        help="Comma-separated 'label:path' entries (e.g. 'kinsim input:$PREFIX/manifest.csv').",
    )
    ap.add_argument("--motifs-column", default="motifs",
                    help="Name of the motifs path column in the manifest CSV. Default 'motifs'.")
    ap.add_argument("--output", required=True, help="Output PNG path.")
    args = ap.parse_args(argv)

    manifests = parse_manifests_arg(args.manifests)
    if not manifests:
        sys.exit("No manifests parsed")
    for label, p in manifests:
        if not p.is_file():
            sys.exit(f"Manifest not found: {p}  (label={label!r})")

    data: dict[str, dict[str, list[float]]] = {}
    print("Reading manifests:")
    for label, p in manifests:
        n_rows, n_found, by_type = collect_from_manifest(p, args.motifs_column)
        data[label] = by_type
        print(f"  [{label}] {p}: {n_rows} manifest rows, {n_found} motif CSVs read")

    print("\n=== meanIpdRatio stats by source × modification type ===")
    print(f"  {'source':<26} {'type':>14} {'n':>6} {'median':>8} {'q25':>8} {'q75':>8} {'max':>8}")
    print("  " + "-" * 80)
    for label, _ in manifests:
        for mt in sorted(data[label].keys()):
            vals = np.asarray(data[label][mt], dtype=np.float64)
            if vals.size == 0:
                continue
            print(f"  {label:<26} {mt:>14} {len(vals):>6} {np.median(vals):>8.2f} "
                  f"{np.percentile(vals, 25):>8.2f} {np.percentile(vals, 75):>8.2f} "
                  f"{vals.max():>8.2f}")

    type_order = ["m6A", "m4C", "m5C", "5mC", "modified_base"]
    n_panels = len(manifests)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.5 * max(1, n_panels), 6),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(42)

    for ax, (label, _) in zip(axes[0], manifests):
        d = data[label]
        types_present = [t for t in type_order if t in d and len(d[t]) > 0]
        for t in sorted(d.keys()):
            if t not in types_present and len(d[t]) > 0:
                types_present.append(t)
        if not types_present:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(label)
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
        ax.set_title(f"{label}\n{sum(len(v) for v in box_data)} motifs",
                     fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Training corpus meanIpdRatio — per-strain motif CSVs from manifest(s)",
                 fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
