# `slurm/eval/` — held-out evaluation chain

Two helper scripts that submit and inspect a pair of `kinsim_nn evaluate`
jobs on the same training run, one for `best_G.pt` (the in-training
best-held-out-W1 snapshot) and one for the latest `G.pt`. The output
TSVs are bucketed identically, so the per-meth and overall W1 columns
can be compared side-by-side.

## Submission

```bash
bash slurm/eval/eval_dual.sh [<run_dir>]
```

`run_dir` defaults to
`/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega`. It must
contain:

```
<run_dir>/
├── ckpts_v2_big/
│   ├── G.pt
│   ├── best_G.pt
│   └── model_config.json
└── shards_nn/
    └── *_shard.pkl
```

The script submits two GPU jobs to `pgpu`, each ~1–2 minutes once it
starts. It prints the two `JOBID`s and a fresh `squeue`.

## Reading the results

Once both jobs disappear from `squeue -u $USER`:

```bash
bash slurm/eval/eval_show.sh [<run_dir>]
```

Prints:

1. `best_G_stats.tsv` — column-aligned.
2. `current_G_stats.tsv` — column-aligned.
3. **Side-by-side W1 delta** — per `meth_id`: `W1(best)`, `W1(current)`,
   `Δ`, and a `verdict` flag (`best_G better`, `current_G better`,
   `comparable` within 5 %).
4. Tail of the most recent log for each job.

## Interpreting the verdict

`best_G better` on the per-meth rows means the in-training W1 was correct
to flag `best_G.pt` as the local optimum, and `current_G.pt` has drifted
since. `current_G better` means the training continued to improve after
the previously-saved best and should be allowed to run further (or that
the in-training W1 metric was noisier than its drift suggested).
`comparable` means the two are within ±5 % on this bucket — typically
sampling noise rather than a real model difference.

The `meth_id = 0` row (baseline / no methylation) and the `meth_id ≥ 1`
rows are now directly comparable across buckets thanks to the fix that
removed the baseline double-count (`BUGS_FOUND.md`, Bug 7).
