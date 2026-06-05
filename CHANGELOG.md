# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-05

### Added
- `kinsim_NN/` — conditional WGAN-GP with a transformer generator
  (~12 M parameters, 8 layers, 8 heads, `d_model = 256`, `z_dim = 128`)
  and a transformer critic (~3 M parameters, 6 layers, 6 heads,
  `d_model = 192`) on a 21 bp window with 4 kinetic channels per
  position (`IPD_fwd`, `PW_fwd`, `IPD_rev`, `PW_rev`).
- AdaLN-Zero conditioning ([Peebles & Xie, 2023](https://doi.org/10.48550/arXiv.2212.09748))
  on every generator block, with `(shift, scale, gate)` derived from
  `cond_emb = MLP(z) + Linear(tokens.mean)`.
- Three-category extraction (`BASELINE` / `SLOWED` / `NEAR_METH`) on
  labeled methylation positions, with per-strand methylation context on
  all 21 window positions.
- Modular labelers: `GFFLabeler` (motifs.gff via pbmotifmaker, with
  optional `require_motif` filter) and `JasmineMMMLLabeler` (5mC via
  MM/ML tags on a jasmine BAM).
- Held-out per-test-strain Wasserstein-1 evaluation during training,
  with `best_G.pt` selection on the global metric.
- `kinsim_nn evaluate` and `kinsim_nn analyze` CLIs for post-training
  diagnostics on shards.
- Smoke tests under `tests/test_kinsim_nn_smoke.py`.

### Removed
- The legacy `kinsim/` package (ConvPredictor with FiLM, single-strand
  per-position Gaussian-NLL prediction).
- The exploratory `kinsim2/` (duplicate of `kinsim/`).
- `kinsim_baseline/` (naive Gaussian baseline).
- The SLURM orchestrators for the legacy training pipeline
  (`slurm/ml/`, per-lineage `slurm/strepto/`, `slurm/vega/`,
  `slurm/sequel/`, the legacy `validate/` chain).
- Legacy `kinsim_config.yaml`. Single configuration source is now
  `kinsim_nn_config.yaml`.

### Fixed
- Five format-boundary bugs in the BAM emission path
  (`flag = 4`, `@HD SO:unknown`, stale `ip` / `pw` stripping,
  `template_length` preservation, kinetic-array clamping to `≥ 1`).
  See [`BUGS_FOUND.md`](BUGS_FOUND.md). Aggregate effect: end-to-end
  yield on bc2034 went from ~250 retained per-position kinetic calls
  to 16.7 million.
- Strand-aware kinetic-tag reversal in
  `kinsim_NN/utils/bam_io.py` for bystrandified inputs. Verified by
  byte-level cross-check against the raw BAM at high-confidence m6A
  positions.
- Bug 10 — `kinsim_NN/generate.py` now writes `fn:i:1` and `rn:i:1`
  alongside `fi`/`fp`/`ri`/`rp`. Without these scalar per-strand
  subread-count tags, `ccs-kinetics-bystrandify` silently dropped
  every record (missing `fn` → no `/fwd`; missing `rn` → no `/rev`;
  missing both → every ZMW gone). Diagnosed 2026-06-02 by tag
  ablation on a known-good Sequel raw HiFi.
- Bug 13 — `kinsim_NN/data/dataset.py` `list_shards()` now matches
  the YAML `test_strains` entries against both the full sample_id
  (`strepto_bc2034`) and the trailing barcode component (`bc2034`).
  Prior behaviour: bare-barcode entries never matched lineage-prefixed
  shards, so the held-out test strains silently leaked into the
  training set. Every v6 W1 number prior to this fix is a
  training-set fidelity metric, not a held-out generalisation result.
  See [`BUGS_FOUND.md`](BUGS_FOUND.md) Bug 13.
- `kinsim_NN/labelers/gff.py` now rejects GFF rows whose strand
  column is anything other than `"+"` or `"-"`. Prior behaviour:
  malformed strands (`"?"`, `"."`, leading whitespace) were inserted
  into the labels dict with keys that never matched the downstream
  `"+"` / `"-"` lookup in `extract.py`, silently discarding the
  methylation. The `n_skipped_strand` count is now logged at parse
  time so the discard is visible.
- `kinsim_nn_config.yaml` — m5C `label_sources` no longer lists
  `jasmine_mm_ml` while the labeler is commented out in the
  `labelers:` list. The YAML now reflects what actually runs.

### Added
- `PACBIO_COMPATIBILITY.md` — forward-looking compatibility rules
  between `pbmm2`, `ccs-kinetics-bystrandify`, `ipdSummary`,
  `pbmotifmaker`, `jasmine`, `modkit`. Consolidates every silent-drop
  rule (tag-presence, flag, SO, codec, header) and the SIF vs conda
  version split.
- `slurm/validate/` — v6 validation chain split into three SLURM
  jobs (`v6_strip.slurm` on pibu_el8, `v6_generate.slurm` on pgpu,
  `v6_downstream.slurm` on pibu_el8) chained via `--dependency=afterok`.

## [0.4.0] — 2026-04 / 2026-05 (legacy)

Single-strand ConvPredictor architecture (~140 K parameters,
FiLM conditioning on methylation context) with per-position Gaussian
output trained under Beta-NLL. Removed in the current release in
favour of `kinsim_NN/`.

[Unreleased]: https://github.com/NicolasDutilleux/KinSim/tree/main
[0.4.0]: https://github.com/NicolasDutilleux/KinSim/releases/tag/v0.4.0
