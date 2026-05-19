"""kinsim_baseline CLI router.

Usage::

    python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_DIR [--threshold 1.3]
        Build per-(meth_type, offset) IPD/PW 1D histograms + IPD×PW 2D joint
        from the BAMs.

    python -m kinsim_baseline analyze OUTPUT_DIR
        Plot per-(meth_type, offset) IPD distributions from a previous
        compute run. Writes a single interactive HTML (linear + log y).

    python -m kinsim_baseline per-kmer PREDICT_NPZ MANIFEST_CSV OUTPUT_DIR
        Use the AI model's per-kmer predictions (from `kinsim predict-kmers`,
        'none' scenario) as the null baseline. Walks the BAMs and flags
        every observed IPD/PW exceeding μ_pred + N·σ_pred per kmer.

    python -m kinsim_baseline plot-per-kmer OUTPUT_DIR
        3-panel HTML: scatter μ_pred vs μ_obs, above-rate distribution,
        and top-K kmers detail (AI baseline vs above-threshold population).

    python -m kinsim_baseline generate --format bam|pkl --baseline BASELINE_JSON --output OUT [...]
        Naive-Gaussian generation. --format bam writes a kinsim-generate
        lookup NPZ (consume via `kinsim generate --use-lookup`). --format
        pkl rewrites a shard's IPD/PW with samples from the per-(T, k)
        Gaussian (requires --input SHARD.pkl). Both modes ignore kmer
        context — the baseline against which to benchmark the ML model.

See ``compute.py``, ``analyze.py``, ``per_kmer.py``, ``plot_kmer.py``, ``generate.py``.
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "usage: python -m kinsim_baseline <command> [options]\n"
            "\n"
            "  compute MANIFEST_CSV OUTPUT_DIR [--threshold 1.3]\n"
            "      Single-pass walk over manifest BAMs.\n"
            "      Reads modified_base + signal_offsets per meth type from\n"
            "      kinsim_config.yaml. For each read, for each meth type T,\n"
            "      for each position p where read[p] == modified_base[T],\n"
            "      for each k in signal_offsets[T]: record ipd[p+k] / pw[p+k]\n"
            "      into the per-(T, k) 256-bin histogram, plus IPD×PW 2D joint.\n"
            "      Outputs (in OUTPUT_DIR):\n"
            "        baseline_hist.tsv     long-form 1D histogram\n"
            "        baseline_summary.tsv  per-(T, k) mean / p50 / p95 / p99\n"
            "        baseline.json         full 1D + 2D histograms\n"
            "        run_info.json         manifest + per-BAM read counts + timing\n"
            "\n"
            "  analyze OUTPUT_DIR\n"
            "      Read OUTPUT_DIR/baseline.json and write a single interactive\n"
            "      HTML showing all (T, k) IPD distributions overlaid, in two\n"
            "      panels (linear + log y) so the bulk and the right-tail are\n"
            "      both readable.\n"
            "      Output:\n"
            "        ipd_distributions.html\n"
            "      Requires plotly (in the [plot] optional extra).\n"
            "\n"
            "  per-kmer MANIFEST_CSV OUTPUT_DIR\n"
            "      Walk manifest BAMs and accumulate per-11-mer (μ_IPD, σ_IPD,\n"
            "      μ_PW, σ_PW, n) using vectorised np.add.at. Skips any window\n"
            "      with a non-ACGT base. KMER_PRED_IDX=7 centres the window\n"
            "      asymmetrically (offsets -7..+3).\n"
            "      Outputs (in OUTPUT_DIR):\n"
            "        per_kmer.npz         raw moments (sum, sum2, count) ×4M kmers\n"
            "        per_kmer_stats.npz   derived μ, σ\n"
            "\n"
            "  plot-per-kmer OUTPUT_DIR [--threshold 2.0]\n"
            "      4-panel plotly HTML from per_kmer_stats.npz:\n"
            "        - scatter μ vs σ (subsampled to 100k, log-coloured by n)\n"
            "        - μ histogram across kmers\n"
            "        - σ histogram across kmers (informs outlier threshold)\n"
            "        - coverage histogram (log y)\n"
            "      Output: per_kmer_dashboard.html + per_kmer_summary.tsv\n"
            "\n"
            "  generate --format bam|pkl --baseline BASELINE_JSON --output OUT [opts]\n"
            "      Naive-Gaussian generation. Per-(T, k) μ/σ in log1p space\n"
            "      from baseline.json are applied without kmer context.\n"
            "        --format bam  Build a kinsim-generate-compatible lookup\n"
            "                      NPZ where every kmer has the SAME (μ, σ)\n"
            "                      per scenario. Then run\n"
            "                        kinsim generate <bam> <ref> <DUMMY> <motifs>\n"
            "                          <out.bam> --use-lookup OUT.npz\n"
            "                      to produce the naive BAM.\n"
            "        --format pkl  Rewrite IPD/PW columns of an existing shard\n"
            "                      (requires --input SHARD.pkl). The kmer keys,\n"
            "                      meth_context, parent_meth/offset all stay;\n"
            "                      only IPD/PW are replaced with naive samples.\n"
            "      Output: <out>.npz  (bam mode)  or  <out>.pkl  (pkl mode)\n"
        )
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "compute":
        from .compute import main as cmd_main
    elif cmd == "analyze":
        from .analyze import main as cmd_main
    elif cmd == "per-kmer":
        from .per_kmer import main as cmd_main
    elif cmd == "plot-per-kmer":
        from .plot_kmer import main as cmd_main
    elif cmd == "generate":
        from .generate import main as cmd_main
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print(
            "expected: compute | analyze | per-kmer | plot-per-kmer | generate",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd_main(rest)


if __name__ == "__main__":
    main()
