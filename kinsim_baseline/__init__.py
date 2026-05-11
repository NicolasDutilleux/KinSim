"""kinsim_baseline — per-(meth_type, offset) IPD/PW distribution model.

Biology-aware statistical "Plan B" comparison model. No neural network,
no kmer specificity, no motif input. Reads ``kinsim_config.yaml`` to learn:

  - which base each methylation type sits on (``modified_base``: A for m6A,
    C for m5C/m4C, ...) — generalisable to any new meth type added to YAML.
  - which downstream offsets carry the kinetic signature
    (``signal_offsets``: e.g. m6A → [0, 5]).

Single-pass walk through the manifest's BAMs:

    For every read, for every meth type T:
      - find positions p where ``read[p] == modified_base[T]``;
      - for each k in ``signal_offsets[T]``, record ``ipd[p+k]`` and
        ``pw[p+k]`` in the per-``(T, k)`` 256-bin histogram.

That gives one IPD distribution and one PW distribution per ``(T, k)``
bucket. Most A's (or C's) in real DNA are unmodified, so the bulk of
the histogram IS the baseline; the high tail captures the modified
subset. Everything downstream (mean, percentiles, modified-mean,
IPD ratio at any threshold) is derived from the histogram.

Outputs (written into ``out_dir/``):

    baseline_hist.tsv     long-form per (T, k, metric, bin) histogram
    baseline_summary.tsv  per-(T, k) mean / p50 / p95 / p99 + IPD ratio
    baseline.json         full histograms in JSON
    run_info.json         manifest + per-BAM read counts + timing

Usage::

    python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_DIR \\
        [--threshold 1.3]
"""
