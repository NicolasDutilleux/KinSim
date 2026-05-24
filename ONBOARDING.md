# KinSim — session snapshot (2026-05-21)

Pick up where the CLI session left off. Repo state at commit `e9da0ec`, branch `Develop`.

## What this project is

KinSim simulates PacBio HiFi kinetic signals (IPD, PW) for metagenomic
binning research. Two packages share one repo:

- **`kinsim`** — legacy single-strand pipeline (20-col shards, ConvPredictor,
  pre-FiLM, bystrandify-dependent).
- **`kinsim2`** — bilateral architecture (33-col shards, joint
  `(ipd_fwd, pw_fwd, ipd_rev, pw_rev)` prediction, post-conv FiLM,
  no bystrandify dependency at train time).

See `CLAUDE.md` for the full developer reference and
`memory/MEMORY.md` for cross-session preferences.

## Recent work in this session

1. **Finished kinsim2 bilateral refactor**: extract, refine, train,
   generate, evaluate, predict_kmers, analyze, verify_generate all on the
   33-col layout (`n_cols = 11 + 2*K`, `KINSIM_LAYOUT_VERSION=2`).
2. **Bystrandify bug fixed** (commit `6ce6cec` and predecessors):
   YAML preset stamped `np:i:3` on every read; bystrandify silently
   rejected them. Restored per-read PacBio tag propagation
   (`np / rq / sn / zm / MM / ML`).
3. **FiLM moved AFTER conv backbone** in `kinsim2/models/predictor.py`
   (Perez+ 2018 placement). Old `conv_bilateral_v2` checkpoints
   incompatible — new arch string `"conv_bilateral_v2_postfilm"`.
4. **Configurable `--p-fire`** (default 0.5) on signature offsets in
   `kinsim2/generate.py`. Vectorised the mapped-path `aligned_pairs` loop
   (~50× faster on long reads).
5. **Determinism**: `L.seed_everything(workers=True)` in train, explicit
   `torch / np / random` seeding in generate. CLI `--seed` (default 42)
   or `KINSIM_SEED` env var.
6. **Provenance**: `model_config.json` now records `git_sha`,
   `kinsim_version`, `timestamp_utc`.
7. **Audit fixes** also applied to `kinsim` where applicable.
8. **Verified consistency** at end: 29 kinsim2 files AST-clean,
   layout sane at K=11 (33) and K=21 (53), p_fire helpers wired,
   bilateral row schema matches across extract/refine/train/generate.

## What is NOT done (not requested, listed only for memory)

- LUT fast path in `kinsim2/generate.py` (mapped path only today).
- Directory mode (PBSIM3 fastq dir) in `kinsim2/generate.py`.
- End-to-end smoke tests in `kinsim2/`.
- Strand-routing real-data verification on bc2045 (cluster job pending
  `git pull` + FORCE_GEN=1 rerun on the validate chain).
- K=21 actual training run.

## Cluster status (last known)

Validate chain on 4 strains (bc2034, bc2045, bc2048, bc2082) was waiting
on a `git pull` on the IBU cluster to pick up the bystrandify fix.
SKIP-first means existing-but-stale shards must be `rm -rf`'d before a
`FORCE_GEN=1` relaunch.

## How to resume at home

```powershell
git pull
git log --oneline -5     # confirm e9da0ec is HEAD
# pick up MEMORY.md cross-session preferences automatically
```

Memory directory `C:\Users\nicod\.claude\projects\c--Dev-KinSim\memory\`
holds the durable user/feedback/project/reference notes — open the repo
in Claude Code at home and it will reload them.
