"""kinsim_baseline — per-(meth_type, offset) IPD/PW baseline + modified-ratio model.

Biology-aware statistical "Plan B" comparison model. No neural network, no
kmer specificity, no motif input. Reads ``kinsim_config.yaml`` to learn:

  - which base each methylation type sits on (``modified_base``: A for m6A,
    C for m5C/m4C, ...) — generalisable to any new meth type added to YAML.
  - which downstream offsets carry the kinetic signature
    (``signal_offsets``: e.g. m6A → [0, 5]).

Pipeline (two passes through the manifest's BAMs):

    Pass 1 (baseline) — for every base ``p`` in every read where
        ``read[p] == modified_base[T]``, accumulate ``ipd[p+k]`` and
        ``pw[p+k]`` into a per-``(T, k)`` running sum + count. Most A's
        (or C's) in real DNA are unmethylated, so the per-``(T, k)``
        mean is dominated by unmodified kinetics.

    Pass 2 (modified) — same walk; now record only positions where the
        observed IPD exceeds ``threshold × baseline_mean[T, k]`` (default
        threshold=1.3). Aggregate into a per-``(T, k)`` modified pool.

    Output — per-``(T, k)`` baseline_mean, modified_mean, IPD ratio
        (= modified_mean / baseline_mean) and PW ratio. Total ~30 numbers
        for the typical m6A / m4C / m5C config.

Usage::

    python -m kinsim_baseline compute MANIFEST_CSV OUTPUT_TSV \\
        [--threshold 1.3] [--output-json out.json]
"""
