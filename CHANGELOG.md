# Changelog

All notable changes to KinSim are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Single v4 codebase: dropped all v3 dispatch paths from
  `analyze`, `refine`, `extract`, `dataset`, and `sample_layout`.
- `kinsim refine` is now a single pass (`slowed_split`); the threshold
  is the `secondary_percentile`-th percentile (default 95) of the
  per-kmer baseline-mean distribution.
- `kinsim_config.yaml`: removed the dead `gmm_signature` block; the
  `extract` and `refine` sections now describe only the active knobs.
- HTML report (`kinsim analyze`): focused 4-figure verification
  dashboard (IPD-by-category, per-kmer baseline mean, signature
  profiles, sample counts).
- `MLPSignalDataset`: dropped the unused `max_unmeth` / `max_meth`
  caps; partitioning is by `meth_id` at `mc[KMER_PRED_IDX]`.
- Project hygiene: PEP 621 metadata in `pyproject.toml`,
  ruff/pytest/coverage configuration, GitHub Actions CI matrix
  (3.10, 3.11, 3.12), pre-commit, dependabot, issue / PR templates,
  CITATION.cff, py.typed markers.

### Removed
- `--v4` flag on `kinsim extract` (single-pass extract is the only mode).
- `--refined-pkl` opt-in oracle.
- `--method gmm_signature` / `--method em` / `--method clustered` /
  `--method mahalanobis` from `kinsim refine`.
- `--report` flag from `kinsim refine` (refine stats now live in the
  output `__meta__["stats"]`).
- `requirements.txt` (dependencies live in `pyproject.toml`).
- `compute_neighbor_sensitivity` and the v3 plot generators in
  `analyze.py` (~1000 lines of dead code).

## [0.4.0] - 2026-04 - 2026-05

- Single-pass v4 extract emitting the 36-column layout with the
  three-category enum (BASELINE / SLOWED / NEAR_METH).
- Per-kmer baseline-mean p95 refine.
- Modular methylation types via `kinsim_config.yaml`.
- ConvPredictor as the default model architecture.
- Per-package READMEs (kinsim/, prep/, slurm_kinsim/).

[Unreleased]: https://github.com/NicolasDutilleux/KinSim/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/NicolasDutilleux/KinSim/releases/tag/v0.4.0
