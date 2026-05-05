"""Cross-dataset kinetic comparison tool.

Builds per-kmer dictionaries from multiple .pkl files and compares
IPD/PW distributions across datasets (e.g. Vega vs Revio vs Sequel).

CLI usage:
    kinsim compare \\
        --label Revio  revio_master.pkl \\
        --label Vega   vega_master.pkl \\
        --output-dir   compare_output/

    # Quick text-only report:
    kinsim compare revio.pkl vega.pkl --no-html

Output:
    compare_output/
        comparison_report.txt   — per-meth-type stats + IPD ratio comparison
        comparison_report.html  — interactive Plotly visualisations
        kmer_dictionary.csv     — full per-kmer stats table for all datasets

Metrics:
    IPD ratio       mean_ipd(meth) / mean_ipd(unmeth) for matching kmers
    ΔPW             PW shift between meth and unmeth contexts
    Matched kmers   Fraction of kmers present in all datasets (coverage overlap)
    Distribution    Per-kmer IPD/PW mean and sigma comparison across datasets
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from .utils.encoding import METH_IDS, decode_kmer

try:
    from sklearn.mixture import GaussianMixture

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

log = logging.getLogger(__name__)

_ID_TO_NAME = {v: k for k, v in METH_IDS.items()}


def _meth_name(meth_id: int) -> str:
    return _ID_TO_NAME.get(meth_id, f"meth_id={meth_id}")


# ---------------------------------------------------------------------------
# Per-kmer stats extraction
# ---------------------------------------------------------------------------


def _build_kmer_dict(data: dict) -> dict:
    """Build per-kmer stats from a .pkl data dict.

    Returns:
        {(kmer_id, meth_id): {
            'n': int,
            'ipd_mean': float, 'ipd_std': float,
            'pw_mean': float,  'pw_std': float,
            'frac_mean': float,
        }}
    """
    stats = {}
    for k, v in data.items():
        if not (isinstance(k, tuple) and len(k) == 2 and isinstance(v, np.ndarray)):
            continue
        kmer_id, meth_id = k
        arr = v
        n = len(arr)
        if n == 0:
            continue
        stats[(kmer_id, meth_id)] = {
            "n": n,
            "ipd_mean": float(np.mean(arr[:, 0])),
            "ipd_std": float(np.std(arr[:, 0])),
            "pw_mean": float(np.mean(arr[:, 1])),
            "pw_std": float(np.std(arr[:, 1])),
            "frac_mean": float(np.mean(arr[:, 2])) if arr.shape[1] >= 3 else float("nan"),
        }
    return stats


def load_pkl(path: str) -> dict:
    """Load a .pkl file and return the raw data dict."""
    log.info("Loading %s ...", path)
    t0 = time.time()
    with open(path, "rb") as f:
        data = pickle.load(f)
    elapsed = time.time() - t0
    n_keys = sum(1 for k in data if isinstance(k, tuple))
    log.info("  %d keys in %.1fs", n_keys, elapsed)
    return data


# ---------------------------------------------------------------------------
# IPD ratio computation
# ---------------------------------------------------------------------------


def compute_ipd_ratios(stats: dict, meth_id: int, min_samples: int = 10):
    """Compute per-kmer IPD ratio: meth_mean / unmeth_mean.

    Returns:
        kmer_ids:   np.ndarray of kmer IDs
        ratios:     np.ndarray of IPD ratios
        pw_deltas:  np.ndarray of PW differences (meth - unmeth)
    """
    kmer_ids = []
    ratios = []
    pw_deltas = []

    for (kid, mid), s in stats.items():
        if mid != meth_id or s["n"] < min_samples:
            continue
        unmeth = stats.get((kid, 0))
        if unmeth is None or unmeth["n"] < min_samples:
            continue
        if unmeth["ipd_mean"] < 1.0:
            continue
        ratio = s["ipd_mean"] / unmeth["ipd_mean"]
        pw_delta = s["pw_mean"] - unmeth["pw_mean"]
        kmer_ids.append(kid)
        ratios.append(ratio)
        pw_deltas.append(pw_delta)

    return np.array(kmer_ids), np.array(ratios), np.array(pw_deltas)


# ---------------------------------------------------------------------------
# Bimodality check on the unmeth class
# ---------------------------------------------------------------------------


def _pool_unmeth_signals(
    data: dict,
    max_samples: int = 500_000,
    seed: int = 42,
) -> np.ndarray | None:
    """Pool raw IPD values across all unmeth (meth_id=0) kmers.

    Signals are log1p-transformed — PacBio IPDs have a heavy tail on the raw
    scale, and log1p matches the training-space the model sees.

    Returns:
        Flat np.ndarray of log1p(IPD) values, or None if no unmeth data.
    """
    parts = []
    for k, v in data.items():
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        if k[1] != 0 or not isinstance(v, np.ndarray) or v.ndim != 2:
            continue
        parts.append(v[:, 0].astype(np.float32))
    if not parts:
        return None
    pooled = np.concatenate(parts)
    if len(pooled) > max_samples:
        rng = np.random.default_rng(seed)
        pooled = rng.choice(pooled, max_samples, replace=False)
    return np.log1p(pooled)


def check_unmeth_bimodality(
    data: dict,
    max_samples: int = 500_000,
) -> dict | None:
    """Fit 1- and 2-component GMMs to the unmeth IPD distribution.

    A substantially-better 2-component fit (large positive ΔBIC) is a red
    flag that the unmeth class is contaminated — most likely by modification
    types that the GFF/motif input did not label.  This is the detector the
    user asked for to spot "hidden" distributions in the unmeth data.

    Returns a dict with the fit summary, or None if sklearn is unavailable
    or there are no unmeth samples.
    """
    if not _HAS_SKLEARN:
        log.warning(
            "sklearn not installed — skipping bimodality check. "
            "Install with: pip install scikit-learn"
        )
        return None

    x = _pool_unmeth_signals(data, max_samples=max_samples)
    if x is None or len(x) < 50:
        return None
    x_col = x.reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(x_col)
    gmm2 = GaussianMixture(n_components=2, random_state=42, covariance_type="full", n_init=3).fit(
        x_col
    )

    # Sort components by mean so the report is deterministic.
    means = gmm2.means_.ravel()
    sigmas = np.sqrt(gmm2.covariances_.ravel())
    weights = gmm2.weights_.ravel()
    order = np.argsort(means)

    return {
        "n_samples": len(x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "bic_1": float(gmm1.bic(x_col)),
        "bic_2": float(gmm2.bic(x_col)),
        "delta_bic": float(gmm1.bic(x_col) - gmm2.bic(x_col)),
        "comp_weights": weights[order].tolist(),
        "comp_means": means[order].tolist(),
        "comp_sigmas": sigmas[order].tolist(),
    }


def _bimodality_verdict(delta_bic: float) -> str:
    """Translate ΔBIC into a human-readable verdict."""
    # Kass & Raftery thresholds (reversed sign since ΔBIC here = BIC_1 - BIC_2):
    #   >10 = very strong, 6-10 = strong, 2-6 = positive, <2 = weak.
    if delta_bic > 10:
        return "VERY STRONG bimodality (possible contamination)"
    if delta_bic > 6:
        return "STRONG bimodality"
    if delta_bic > 2:
        return "weak bimodality"
    return "unimodal (as expected)"


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------


def generate_report(
    datasets: list[tuple[str, dict]],
    output_dir: str,
    bimodality: list[tuple[str, dict | None]] | None = None,
) -> str:
    """Generate comparison report.

    Args:
        datasets: list of (label, kmer_stats_dict) pairs
        output_dir: directory for output files
        bimodality: optional list of (label, bimodality_result) pairs, as
                    returned by check_unmeth_bimodality.  Appended as a
                    section at the end of the report.

    Returns:
        Report text
    """
    lines = []
    w = lines.append

    w("=" * 72)
    w("  KinSim Cross-Dataset Kinetic Comparison")
    w("=" * 72)
    w("")

    # --- Overview ---
    w("=== Overview ===")
    w("")
    for label, stats in datasets:
        n_keys = len(stats)
        meth_types = sorted(set(mid for _, mid in stats.keys()))
        total_samples = sum(s["n"] for s in stats.values())
        w(f"  {label}:")
        w(f"    Keys:       {n_keys:,}")
        w(f"    Samples:    {total_samples:,}")
        w(f"    Meth types: {[_meth_name(m) for m in meth_types]}")
        w("")

    # --- Per-meth-type stats ---
    all_meth_ids = sorted(set(mid for _, stats in datasets for _, mid in stats.keys()))

    w("=== Per-Type Signal Statistics ===")
    w("")
    header = f"{'Type':<10}"
    for label, _ in datasets:
        header += f"  {label + ' IPD':>18}  {label + ' PW':>18}"
    w(header)
    w("-" * len(header))

    for mid in all_meth_ids:
        row = f"{_meth_name(mid):<10}"
        for _label, stats in datasets:
            ipd_vals = [s["ipd_mean"] for (k, m), s in stats.items() if m == mid]
            pw_vals = [s["pw_mean"] for (k, m), s in stats.items() if m == mid]
            if ipd_vals:
                ipd_m, ipd_s = np.mean(ipd_vals), np.std(ipd_vals)
                pw_m, pw_s = np.mean(pw_vals), np.std(pw_vals)
                row += f"  {ipd_m:7.2f} ± {ipd_s:<7.2f}  {pw_m:7.2f} ± {pw_s:<7.2f}"
            else:
                row += f"  {'n/a':>18}  {'n/a':>18}"
        w(row)
    w("")

    # --- IPD ratios ---
    w("=== IPD Ratios (meth / unmeth, per-kmer) ===")
    w("")
    for mid in all_meth_ids:
        if mid == 0:
            continue
        w(f"--- {_meth_name(mid)} ---")
        header2 = f"  {'Dataset':<20} {'N_kmers':>8} {'Median':>8} {'Mean':>8} {'P10':>8} {'P25':>8} {'P75':>8} {'P90':>8}"
        w(header2)

        for label, stats in datasets:
            _, ratios, _ = compute_ipd_ratios(stats, mid)
            if len(ratios) == 0:
                w(f"  {label:<20} {'n/a':>8}")
                continue
            w(
                f"  {label:<20} {len(ratios):>8d} {np.median(ratios):>8.3f} {np.mean(ratios):>8.3f}"
                f" {np.percentile(ratios, 10):>8.3f} {np.percentile(ratios, 25):>8.3f}"
                f" {np.percentile(ratios, 75):>8.3f} {np.percentile(ratios, 90):>8.3f}"
            )
        w("")

    # --- PW deltas ---
    w("=== PW Shift (meth - unmeth, per-kmer) ===")
    w("")
    for mid in all_meth_ids:
        if mid == 0:
            continue
        w(f"--- {_meth_name(mid)} ---")
        header3 = f"  {'Dataset':<20} {'N_kmers':>8} {'Median':>8} {'Mean':>8} {'Std':>8}"
        w(header3)

        for label, stats in datasets:
            _, _, pw_deltas = compute_ipd_ratios(stats, mid)
            if len(pw_deltas) == 0:
                w(f"  {label:<20} {'n/a':>8}")
                continue
            w(
                f"  {label:<20} {len(pw_deltas):>8d} {np.median(pw_deltas):>8.3f}"
                f" {np.mean(pw_deltas):>8.3f} {np.std(pw_deltas):>8.3f}"
            )
        w("")

    # --- Kmer overlap ---
    w("=== Kmer Coverage Overlap ===")
    w("")
    if len(datasets) >= 2:
        for mid in all_meth_ids:
            kmer_sets = []
            for label, stats in datasets:
                kmers = set(kid for (kid, m) in stats.keys() if m == mid)
                kmer_sets.append((label, kmers))

            w(f"--- {_meth_name(mid)} ---")
            for i, (l1, s1) in enumerate(kmer_sets):
                for j, (l2, s2) in enumerate(kmer_sets):
                    if j <= i:
                        continue
                    overlap = len(s1 & s2)
                    union = len(s1 | s2)
                    jaccard = overlap / union if union else 0
                    w(f"  {l1} ∩ {l2}: {overlap:,} / {union:,} ({jaccard:.1%} Jaccard)")
            w("")

    # --- Unmeth bimodality (GMM 1 vs 2 components) ---
    if bimodality:
        w("=== Unmethylated-Class Bimodality (GMM on log1p(IPD)) ===")
        w("")
        w("  ΔBIC = BIC_1 - BIC_2  (positive = 2-component fit is better).")
        w("  A large ΔBIC on the unmeth class suggests hidden distributions,")
        w("  typically unlabelled modifications leaking into unmeth.")
        w("")
        header_b = (
            f"  {'Dataset':<16} {'N':>10} {'mean':>7} {'std':>7} {'ΔBIC':>10}  {'verdict':<46}"
        )
        w(header_b)
        w("-" * len(header_b))
        for label, res in bimodality:
            if res is None:
                w(f"  {label:<16} {'n/a':>10}  (no unmeth data or sklearn missing)")
                continue
            w(
                f"  {label:<16} {res['n_samples']:>10,} {res['mean']:>7.3f} "
                f"{res['std']:>7.3f} {res['delta_bic']:>10.1f}  "
                f"{_bimodality_verdict(res['delta_bic']):<46}"
            )
        w("")
        w("  2-component fit details:")
        for label, res in bimodality:
            if res is None:
                continue
            w(f"  {label}:")
            for i, (wi, mi, si) in enumerate(
                zip(res["comp_weights"], res["comp_means"], res["comp_sigmas"], strict=False)
            ):
                w(f"    comp {i}: weight={wi:.3f}  μ={mi:.3f}  σ={si:.3f}")
        w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_kmer_csv(datasets: list[tuple[str, dict]], output_path: str) -> None:
    """Export full per-kmer stats table for all datasets."""
    # Collect all unique (kmer_id, meth_id) keys across datasets
    all_keys = set()
    for _, stats in datasets:
        all_keys.update(stats.keys())

    all_keys = sorted(all_keys)
    labels = [label for label, _ in datasets]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["kmer_seq", "kmer_id", "meth_type", "meth_id"]
        for label in labels:
            header.extend(
                [
                    f"{label}_n",
                    f"{label}_ipd_mean",
                    f"{label}_ipd_std",
                    f"{label}_pw_mean",
                    f"{label}_pw_std",
                ]
            )
        writer.writerow(header)

        for kmer_id, meth_id in all_keys:
            try:
                kmer_seq = decode_kmer(kmer_id)
            except (ValueError, IndexError):
                kmer_seq = f"kmer_{kmer_id}"

            row = [kmer_seq, kmer_id, _meth_name(meth_id), meth_id]
            for _, stats in datasets:
                s = stats.get((kmer_id, meth_id))
                if s:
                    row.extend(
                        [
                            s["n"],
                            f"{s['ipd_mean']:.2f}",
                            f"{s['ipd_std']:.2f}",
                            f"{s['pw_mean']:.2f}",
                            f"{s['pw_std']:.2f}",
                        ]
                    )
                else:
                    row.extend(["", "", "", "", ""])
            writer.writerow(row)

    log.info("Exported %d kmer rows to %s", len(all_keys), output_path)


# ---------------------------------------------------------------------------
# HTML report with Plotly
# ---------------------------------------------------------------------------


def generate_html(datasets: list[tuple[str, dict]], output_path: str) -> None:
    """Generate interactive HTML comparison plots."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        log.warning("plotly not installed — skipping HTML report")
        return

    all_meth_ids = sorted(set(mid for _, stats in datasets for _, mid in stats.keys()))
    meth_types_no_unmeth = [m for m in all_meth_ids if m != 0]

    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]

    figs = []

    # --- Figure 1: IPD ratio distributions ---
    for mid in meth_types_no_unmeth:
        fig = go.Figure()
        for i, (label, stats) in enumerate(datasets):
            _, ratios, _ = compute_ipd_ratios(stats, mid)
            if len(ratios) == 0:
                continue
            fig.add_trace(
                go.Histogram(
                    x=ratios,
                    name=label,
                    opacity=0.6,
                    nbinsx=100,
                    marker_color=colors[i % len(colors)],
                )
            )
        fig.update_layout(
            title=f"IPD Ratio Distribution — {_meth_name(mid)}",
            xaxis_title="IPD ratio (meth / unmeth)",
            yaxis_title="Count (kmers)",
            barmode="overlay",
            width=900,
            height=500,
        )
        figs.append(fig)

    # --- Figure 2: IPD mean scatter (dataset A vs B) ---
    if len(datasets) >= 2:
        for mid in all_meth_ids:
            label_a, stats_a = datasets[0]
            label_b, stats_b = datasets[1]

            common_kmers = set(kid for kid, m in stats_a.keys() if m == mid) & set(
                kid for kid, m in stats_b.keys() if m == mid
            )

            if len(common_kmers) < 10:
                continue

            # Subsample for plotting
            common = sorted(common_kmers)
            if len(common) > 50_000:
                rng = np.random.default_rng(42)
                common = list(rng.choice(common, 50_000, replace=False))

            x = [stats_a[(k, mid)]["ipd_mean"] for k in common]
            y = [stats_b[(k, mid)]["ipd_mean"] for k in common]

            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=2, opacity=0.3),
                    name=f"{_meth_name(mid)} ({len(common):,} kmers)",
                )
            )
            # Add y=x line
            mx = max(max(x), max(y))
            fig.add_trace(
                go.Scatter(
                    x=[0, mx],
                    y=[0, mx],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    showlegend=False,
                )
            )
            fig.update_layout(
                title=f"IPD Mean: {label_a} vs {label_b} — {_meth_name(mid)}",
                xaxis_title=f"{label_a} IPD mean",
                yaxis_title=f"{label_b} IPD mean",
                width=700,
                height=700,
            )
            figs.append(fig)

    # --- Figure 3: PW mean scatter ---
    if len(datasets) >= 2:
        for mid in all_meth_ids:
            label_a, stats_a = datasets[0]
            label_b, stats_b = datasets[1]

            common_kmers = set(kid for kid, m in stats_a.keys() if m == mid) & set(
                kid for kid, m in stats_b.keys() if m == mid
            )

            if len(common_kmers) < 10:
                continue

            common = sorted(common_kmers)
            if len(common) > 50_000:
                rng = np.random.default_rng(42)
                common = list(rng.choice(common, 50_000, replace=False))

            x = [stats_a[(k, mid)]["pw_mean"] for k in common]
            y = [stats_b[(k, mid)]["pw_mean"] for k in common]

            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=2, opacity=0.3),
                    name=f"{_meth_name(mid)} ({len(common):,} kmers)",
                )
            )
            mx = max(max(x), max(y))
            fig.add_trace(
                go.Scatter(
                    x=[0, mx],
                    y=[0, mx],
                    mode="lines",
                    line=dict(dash="dash", color="gray"),
                    showlegend=False,
                )
            )
            fig.update_layout(
                title=f"PW Mean: {label_a} vs {label_b} — {_meth_name(mid)}",
                xaxis_title=f"{label_a} PW mean",
                yaxis_title=f"{label_b} PW mean",
                width=700,
                height=700,
            )
            figs.append(fig)

    # --- Write HTML ---
    if not figs:
        log.warning("No plots generated")
        return

    html_parts = [
        "<!DOCTYPE html><html><head>",
        '<meta charset="utf-8">',
        "<title>KinSim Cross-Dataset Comparison</title>",
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
        "</head><body>",
        "<h1>KinSim Cross-Dataset Kinetic Comparison</h1>",
    ]
    for i, fig in enumerate(figs):
        html_parts.append(f'<div id="fig{i}"></div>')
        html_parts.append("<script>")
        html_parts.append(f"Plotly.newPlot('fig{i}', {fig.to_json()});")
        html_parts.append("</script>")

    html_parts.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(html_parts))

    log.info("HTML report: %s (%d figures)", output_path, len(figs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="kinsim compare",
        description="Compare per-kmer kinetic distributions across datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  kinsim compare --label Revio revio.pkl --label Vega vega.pkl
  kinsim compare revio.pkl vega.pkl sequel.pkl --output-dir compare/
  kinsim compare revio.pkl vega.pkl --no-html --min-samples 20
""",
    )
    parser.add_argument(
        "pkl_files",
        nargs="+",
        help="Input .pkl files to compare (2+ recommended).",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=[],
        help="Label for the NEXT .pkl file. Use before each .pkl path. "
        "If omitted, filenames are used as labels.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory for output files (default: cwd).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples per kmer for ratio computation (default: 10).",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report (text + CSV only).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV kmer dictionary export.",
    )
    parser.add_argument(
        "--bimodality",
        action="store_true",
        help="Fit 1- and 2-component GMMs to the unmeth IPD distribution of "
        "each dataset, to detect hidden sub-populations (e.g. an unlabelled "
        "modification contaminating the unmeth class).",
    )
    parser.add_argument(
        "--bimodality-max-samples",
        type=int,
        default=500_000,
        help="Max unmeth samples pooled per dataset for the GMM fit (default: 500,000).",
    )

    args = parser.parse_args(argv)

    if len(args.pkl_files) < 1:
        parser.error("At least one .pkl file is required.")

    # Build (label, path) pairs
    # Labels can be interleaved: --label A file1.pkl --label B file2.pkl
    # or omitted entirely (filenames used)
    labels = args.labels or []
    paths = args.pkl_files

    if len(labels) > len(paths):
        parser.error("More --label flags than .pkl files.")

    # Pad missing labels with filenames
    while len(labels) < len(paths):
        name = Path(paths[len(labels)]).stem
        # Shorten common prefixes
        for prefix in ("master_", "shards_", "training_"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        labels.append(name)

    # Load and build stats
    datasets = []
    bimodality_results: list[tuple[str, dict | None]] = []
    for label, path in zip(labels, paths, strict=False):
        if not os.path.isfile(path):
            log.error("File not found: %s", path)
            sys.exit(1)
        data = load_pkl(path)
        stats = _build_kmer_dict(data)
        datasets.append((label, stats))
        if args.bimodality:
            log.info("Fitting GMM on unmeth IPD for %s ...", label)
            bimodality_results.append(
                (label, check_unmeth_bimodality(data, max_samples=args.bimodality_max_samples))
            )
        del data  # free memory

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Text report
    report = generate_report(
        datasets,
        str(out_dir),
        bimodality=bimodality_results if args.bimodality else None,
    )
    print(report)

    report_path = out_dir / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    log.info("Text report: %s", report_path)

    # CSV export
    if not args.no_csv:
        csv_path = out_dir / "kmer_dictionary.csv"
        export_kmer_csv(datasets, str(csv_path))

    # HTML report
    if not args.no_html:
        html_path = out_dir / "comparison_report.html"
        generate_html(datasets, str(html_path))

    log.info("Done.")


if __name__ == "__main__":
    main()
