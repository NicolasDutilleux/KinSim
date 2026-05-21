# Changelog

All notable changes to KinSim are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `kinsim2/` — bilateral architecture (v2). Single forward pass jointly
  predicts (ipd_fwd, pw_fwd, ipd_rev, pw_rev) from a raw HiFi aligned
  BAM, no bystrandify dependency. 33-column shard layout
  (`11 + 2*K`), cross-meth FiLM, per-strand
  `category / parent_meth / parent_offset`, 4-channel Beta-NLL.
  Installed as a separate CLI `kinsim2` alongside `kinsim`.
- `model_config.json` now records `git_sha`, `kinsim_version` and
  `timestamp_utc` for reproducibility — both packages.
- Global determinism: `L.seed_everything(workers=True)` in `kinsim
  train` / `kinsim2 train`; `torch.manual_seed` + `np.random.seed` +
  `random.seed` in `kinsim generate` / `kinsim2 generate` (seed via
  `--seed` CLI flag or `KINSIM_SEED` env var).

### Fixed
- `kinsim generate`: per-read propagation of PacBio auxiliary tags
  (`np`, `rq`, `sn`, `zm`, `MM`, `ML`, …) from the input record, so
  ccs-kinetics-bystrandify finds the metadata it needs. The previous
  YAML-driven preset stamped a single `np=3` on every read, which
  bystrandify silently rejected.
- `kinsim2/analyze.py`: scans every row instead of skipping bad ones
  silently; raises with a precise diagnostic if any shard row diverges
  from the bilateral v2 layout.
- `kinsim2/utils/config.py`: NameError on import (forward-reference to
  `_FALLBACK_*` constants).

### Changed
- `kinsim/refine.py` / `kinsim2/refine.py`: baseline-anchored 1D-IPD
  GMM (per `(meth_type, parent_offset)`) is now the only refine method.
  K∈{1,2,3} picked by BIC with a strict biological veto rejecting
  sub-baseline components for K>2.
- `kinsim/train.py` / `kinsim2/train.py`: default loss is Beta-NLL
  (β=0.5, Seitzer 2022) to prevent the model from inflating σ in place
  of fitting μ.
- Sharded mode is the default training path; `ShardedSignalDataset`
  bounds peak RAM at one shard regardless of corpus size.
- Train / test split: `--test-strains a,b,c` (by sample_id) or
  `--test-fraction 0.10 --split-seed N` (random per-shard).

### Removed
- v3/v4 dispatch concepts; the codebase is one bilateral path
  (`kinsim2`) plus the legacy single-strand path (`kinsim`).
- `MLPPredictor` references in docs (only `ConvPredictor` survives).
- `requirements.txt` (dependencies live in `pyproject.toml`).

## [0.4.0] - 2026-04 → 2026-05

- 20-column single-strand shard layout
  (`IPD, PW, fraction, mc_ctx[K], rev_meth[3], category, parent_meth,
  parent_offset`).
- ConvPredictor (per-base embed + positional + FiLM(meth) + 3 conv
  layers + dual readout) as the default architecture.
- Modular methylation types via `kinsim_config.yaml`
  (`kinetic_signatures.<T>.{modified_base, signal_offsets, aliases}`).
- Per-package READMEs (`kinsim/`, `slurm_kinsim/`, `kinsim_baseline/`).
- Validate chain end-to-end: SIM.bam → bystrandify → align →
  ipdSummary + jasmine → merge_motifs → compare to real motifs.

[Unreleased]: https://github.com/NicolasDutilleux/KinSim/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/NicolasDutilleux/KinSim/releases/tag/v0.4.0
