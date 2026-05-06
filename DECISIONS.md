# KinSim Decisions Log

Engineering and architectural decisions, with rationale. New entries
go on top. Each entry: **what** changed, **why**, **affected files**.

---

## 2026-05-06 — Statistical-firing decomposition + 20-col layout + rev_meth FiLM + extract speed

### What
- Dropped the `profile_IPD` and `profile_PW` columns from the row layout.
  `SAMPLE_NCOLS` shrunk from 38 to 20.
- Reordered remaining columns (CATEGORY 35→17, PARENT_METH 36→18,
  PARENT_OFFSET 37→19, REV_METH 32–34→14–16).
- Refine now records `mean_occupancy` per `(meth, offset)` bucket
  alongside `p_fire`. Both propagate through train into
  `model_config.json`.
- Generate now applies a per-site Bernoulli at rate
  `target_frac × p_efficiency` where
  `p_efficiency = p_fire / mean_occupancy`. Per-position frac comes from
  `build_reference_frac_map` against the target genome's motifs.csv.
- Dataset emits a `(B, 14, M)` meth tensor (11 forward + 3 rev_meth).
  Both `MLPPredictor` and `ConvPredictor` extend `meth_proj` accordingly.
- Extract: per-contig precompute of strand-routed lookup arrays
  (`fwd_meth_arr`, `slowed_T`, `slowed_off`, `slowed_frac`,
  `excluded`, etc.), replacing per-position dict lookups in the BAM
  inner loop with O(1) array indexing. Sliding-window 11-mer encoding
  via numpy bit shifts.
- Extract: row 2 (`COL_FRACTION`) now carries the canonical site's
  motif frac (was hardcoded to 1.0).
- Deleted `scripts/diagnose_offset_split.py` (vestigial — analyze.py
  covers the same diagnostic via per-(T, off) bucket means now).

### Why
- **Profile drop**: the `[0..+8]` downstream profile was a legacy
  validation artefact from before `PARENT_OFFSET` was written at
  extract time. Refine doesn't use it (uses `(T, offset)` buckets
  directly), train doesn't use it (only col 0/1), generate doesn't use
  it. Only analyze read it, and the per-bucket interpretation broke
  once rows were emitted at every signature offset rather than only at
  the canonical site. Storage shrinks ~50% per row; analyze switches to
  per-bucket scalar mean IPD/PW which is the more honest diagnostic.
- **p_fire decomposition**: refine's `p_fire = n_kept / n_in` mixes
  motif occupancy (how many sites are actually methylated) with kinetic
  efficiency (given a methylated site, how often does the polymerase
  pause). Applying training-corpus `p_fire` uniformly at generate time
  underfires when the target strain has higher occupancy than the
  training average, and overfires when it has lower. Decomposing into
  `target_frac × p_efficiency` lets each target site contribute its own
  occupancy. Linear approximation; sufficient first cut. Per-occupancy
  curve is documented as future work in CLAUDE.md.
- **rev_meth into FiLM**: `rev_meth` at offsets [-1, 0, +1] was already
  stored in every row but unused by the model. For palindromic Type II
  R-M sites both strands carry the modification; the polymerase
  contacts both strands of the duplex over the active-site footprint,
  so the opposite-strand methylation also shifts kinetics. Feeding
  rev_meth into FiLM lets the model distinguish hemi- from
  fully-methylated sites. ~30 lines, requires retrain.
- **Extract speed**: the previous orientation-aware path was 2× slower
  than the legacy raw-HiFi path because of per-row Python dict lookups
  for slowed/near categorisation, per-row 22-lookup distance scan for
  baseline filtering, and per-row `_kmer_at_ref` calls. Pre-building
  per-contig arrays (memory cost ~14 bytes per ref base — trivial for
  bacterial genomes) replaces all of that with array indexing. Baseline
  exclusion uses a single boolean OR-roll loop instead of 22 dict
  lookups per position. Sliding 11-mer via numpy bit shifts replaces
  a Python loop over each window's bases.
- **frac in COL_FRACTION**: refine needs per-row `frac` to compute
  `mean_occupancy` per bucket. Hardcoding to 1.0 made the
  decomposition collapse (`p_efficiency = p_fire / 1 = p_fire`,
  identity).
- **Deleted diagnostic script**: it operated on `[0..+8]` profile
  columns; with the columns gone its only purpose was to detect "flat
  profile pathology" which analyze's per-bucket mean IPD already shows.

### Affected files
- `kinsim/utils/sample_layout.py` (rewritten)
- `kinsim/extract.py` (vectorised, frac-aware, profile-free)
- `kinsim/refine.py` (mean_occupancy in per_bucket stats)
- `kinsim/train.py` (extract + plumb mean_occupancy to model_config.json)
- `kinsim/generate.py` (target_frac × p_efficiency Bernoulli, frac_map back)
- `kinsim/data/dataset.py` (emit (B, 14, M) meth tensor with rev_meth tail)
- `kinsim/models/predictor.py` (meth_proj input dim 11*M → 14*M, both architectures)
- `kinsim/analyze.py` (compute_signature_profiles emits scalar mean_ipd/pw)
- `tests/test_v4_pipeline.py` (SAMPLE_NCOLS=20 + mean_ipd assertions)
- `CLAUDE.md`, `kinsim/README.md` (column reference updates)
- Deleted: `scripts/diagnose_offset_split.py`

### Migration note
**Every existing shard pkl is invalid** under the new layout. After v6
verify confirms orientation works:
1. Re-extract all strains with the new code → 20-col shards
2. Refine → produces `mean_occupancy` per bucket alongside `p_fire`
3. Train → embeds both in `model_config.json`
4. Generate → applies `target_frac × p_efficiency` decomposition

---

## 2026-05-06 — Repository cleanup: removed nextflow/, cluster/, archive artefacts

### What
- Deleted `nextflow/` directory (alternative PREPARE pipeline,
  duplicated `slurm_kinsim/{strepto,vega,sequel}/`).
- Deleted `cluster/` (stale `rebase_motifs.csv`, `strains_stats.csv`).
- Deleted local artefact directories: `dictionary_test/`,
  `reports_strepto/`, `kinsim.egg-info/`, `.ruff_cache/`.
- Removed `archive/*` excludes from `pyproject.toml` and
  `.pre-commit-config.yaml`.
- Updated `slurm_kinsim/ml/run.sh` to drop the `merge` step (sharded
  end-to-end now: extract → refine → train → evaluate).
- Updated all docs (CLAUDE.md, README.md, slurm_kinsim/README.md,
  CONTRIBUTING.md, scripts/sample.py header, 00_extract.slurm header,
  05_baselines.slurm header) to remove references to removed files.
- Renamed `extract_aligned_to_shard` → `extract_to_shard` (single
  extract path, not "aligned vs raw").

### Why
- `nextflow/` was an alternative pipeline never adopted as the
  canonical path; `slurm_kinsim/` is what actually runs on IBU.
- `cluster/` data files were committed during early exploration and
  no longer have consumers.
- `archive/` was already deleted in a previous commit; the excludes in
  config files were dead references.
- The "merge" step (`kinsim merge` shards/ → master.pkl) was removed
  when the pipeline went sharded-only; the orchestrator still chained
  it, so submitting `bash run.sh all` would have failed at the missing
  step.

### Affected files
- Deleted: `nextflow/`, `cluster/`, `dictionary_test/`,
  `reports_strepto/`, `kinsim.egg-info/`, `.ruff_cache/`
- Modified: `slurm_kinsim/ml/run.sh`, `slurm_kinsim/README.md`,
  `slurm_kinsim/ml/00_extract.slurm`, `slurm_kinsim/05_baselines.slurm`,
  `kinsim/extract.py` (renames), `kinsim/utils/config.py` (docstrings),
  `kinsim/README.md`, `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`,
  `pyproject.toml`, `.pre-commit-config.yaml`, `scripts/sample.py`

---

## 2026-05-06 — p_fire from GMM survival + phantom-footprint fix

### What
- Refine records `p_fire = n_kept / n_in` per `(meth, offset)` bucket
  in `__meta__["stats"]["per_bucket"]`. Skipped/validation-failed
  buckets get `p_fire = 1.0` (always-fire fallback).
- Train reads `p_fire` from any refined shard's meta and embeds it in
  `model_config.json`.
- Generate replaces the centre-only motif-frac Bernoulli with a
  per-row mc-walk: for each non-zero `mc[i]` whose offset
  `k = pred_idx − i` is a signature offset of meth type T, roll
  `Bernoulli(p_fire[T, k])`; on no-fire, zero `mc[i]` so the model
  emits baseline-like signal at this row.
- Helpers added: `_build_p_fire_lookup`,
  `_build_sig_offsets_by_meth_id`, `_apply_p_fire_to_mc`,
  `_load_p_fire`. Removed: `_build_fraction_lookup`,
  per-position frac_map plumbing through `_process_batch`.
- `train._read_pkl_meta` now also accepts a shards directory (reads
  meta from the first shard found — refine writes the same global
  per_bucket dict into every shard).

### Why
- The GMM in refine drops rows that don't show the expected joint
  (IPD, PW) shift — that's the "non-firing" population. The dropped
  fraction is the noise the simulator needs to put back at generate
  time so output reads have realistic bimodal distributions (only a
  fraction of motif sites show kinetic slowing, matching what
  ipdSummary expects to find).
- The previous centre-only Bernoulli (using motif `frac`) had a
  phantom-footprint bug: when a canonical m6A "didn't fire" for a
  read, the centre's meth_id was zeroed but rows downstream at +5 still
  saw the m6A in their mc context window (built from unchanged
  `ref_meth`) and emitted the slowed +5 signal regardless. The mc-walk
  approach fixes this in one sweep — zeroing `mc[i]` at the +5 row
  makes the model see baseline context there too.

### Affected files
- `kinsim/refine.py` (p_fire in per_bucket stats)
- `kinsim/train.py` (extract p_fire, embed in model_config.json)
- `kinsim/generate.py` (mc-walking Bernoulli, replaces centre-frac one)

### Superseded by
The 2026-05-06 statistical-firing decomposition entry above splits
this into `mean_occupancy` + `p_efficiency` so target-genome
occupancy is honoured per-site rather than averaged. The plumbing
this entry put in place is the foundation.

---

## 2026-04 onwards — KinSim v0.4.0 baseline

These predate this decision log; documented in CHANGELOG.md and
captured in the 0.4.0 release. Summary of the architectural choices
that the more recent work builds on:

### Asymmetric kmer + meth context window [-7, +3]
**Why**: polymerase has read more bases upstream than downstream at
incorporation; all kinetic signatures sit downstream of the
modification. To predict IPD/PW at row position Y, the model needs
upstream context that *contains* methylations whose downstream effect
reaches Y. Inspired by Feng et al. 2013 (`kineticsTools`/`ipdSummary`,
`[-7, +2]` for unmodified DNA), extended one base on the right to round
to K=11. The prediction position lives at index 7 of the kmer.

### Three-category emission (BASELINE / SLOWED / NEAR_METH)
**Why**: training only on methylation centres would deprive the model
of the slowing signature at downstream offsets (m5C at +2/+6 in
particular has no signal at the centre itself). Emitting rows at every
signature offset and tagging them with `(PARENT_METH, PARENT_OFFSET)`
lets refine bucket per `(T, off)` cleanly and lets the model learn the
position-dependent kinetic effect.

### Per-(meth, offset) GMM in refine
**Why**: a single per-meth-type GMM pools all offsets together. A
noisy offset (e.g. m6A@+5 from a Type I R-M motif that doesn't
actually carry a +5 signature) then contaminates the GMM and
over-rejects the clean +0 samples. Per-(T, offset) buckets isolate
noise: each offset gets its own fit, its own validation, its own cut.

### ConvPredictor + FiLM as default
**Why**: a flat 4.2M-row kmer embedding (MLPPredictor, 268M params)
can't generalise across methylation states or to unseen kmers — it
memorises. ConvPredictor (140K params) is forced to learn
compositional rules: per-base embedding + 1D conv backbone + FiLM
modulation from methylation context. The 1900× parameter reduction is
explicit pressure toward generalisation, justified by the 95/5
class imbalance between unmethylated and methylated training rows.

### Orientation-aware aligned-only extract
**Why**: raw HiFi BAM `query_sequence` orientation is arbitrary
per-read (CCS+lima choose it from barcode), so for ~50% of reads
`fi`/`ri` are swapped relative to the reference strand of the
methylation. Per-row signal averages away. Bystrandify+pbmm2 alignment
gives one read per polymerase-pass strand with a single `ip`/`pw`
array; `read.is_reverse` disambiguates which reference strand the
polymerase templated. The extract pre-builds two per-strand methylation
maps and routes each aligned read's `ip[read_pos]` lookups to the right
map. Empirical validation: 3.25× tail enrichment in the high-IPD
distribution after orientation correction.

### YAML-driven methylation alphabet
**Why**: hardcoding `m6A`/`m4C`/`m5C` in source means adding a new
type (e.g. `m4mC`) is a multi-file edit. With
`kinsim_config.yaml:kinetic_signatures.<type>` the entire pipeline
(extract, refine, train, analyze) auto-picks up new types — single
config change, stable IDs across re-runs (pinned by name for the
canonical four).

### Sharded mode end-to-end
**Why**: collapsing N strain shards into a master.pkl at v3 needed
~750 GB RAM for full Streptomyces corpora. Sharded refine (pool
harvest across shards → fit GMMs once globally → apply per-shard) and
sharded train (`ShardedSignalDataset` worker-aware iterable) bound
peak memory to one shard regardless of corpus size.

---

## How to use this log

- **One entry per architectural change** worth defending later (thesis,
  retrospective, code review).
- **What** describes the surface change; **why** records the
  motivation that's not obvious from the code itself.
- New work goes on top. Use today's date in `YYYY-MM-DD` format.
- When a later decision supersedes an earlier one, link them with
  "Superseded by ..." rather than deleting the older entry — the
  reasoning chain matters.
