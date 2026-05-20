# KinSim — Developer Reference for Claude

## Project Summary

KinSim simulates PacBio HiFi kinetic signals (IPD and PW) for metagenomic
binning research. Given reference genomes + methylation motif annotations
+ aligned bystrandified BAMs, KinSim trains a per-context ConvPredictor
(~140K params, FiLM-conditioned on methylation context) and injects
biologically realistic per-base IPD/PW values into unaligned BAM files
using the trained model.

Output BAMs carry standard PacBio raw-HiFi tags `fi:B:C` (forward IPD),
`fp:B:C` (forward PW), `ri:B:C` (reverse IPD), `rp:B:C` (reverse PW).
The validate chain runs `ccs-kinetics-bystrandify` on the generated BAM
to convert to per-strand ip/pw before feeding ipdSummary — exactly
matching the real-data preprocessing pipeline.

A single CLI is installed:
- **`kinsim`** — ML pipeline: extract, refine, train, generate, evaluate,
  verify-generate, analyze, predict-kmers

Offline tooling lives in `scripts/` and is run via `python scripts/<name>.py`:
manifest count/validate/list, sample, strip-kinetics, merge_shards,
compare, inspect_null_model.

The `kinsim_baseline` module (separate CLI: `python -m kinsim_baseline`)
provides a context-free naive Gaussian baseline as a benchmark for the
ML model.

---

## YAML — single source of biology truth

`kinsim_config.yaml` at the repo root is read by **every** stage of the
pipeline (`extract`, `refine`, `train`, `generate`, `predict-kmers`,
`evaluate`, `analyze`). Edits propagate without code changes — but they
also propagate **between stages**, so changing the YAML mid-pipeline is
a known footgun:

- **`kinetic_signatures.<T>.signal_offsets`** declares where each meth
  type causes pause: m6A=[0,5], m4C=[0], m5C=[2,6] (default). Both
  extract (which positions get CATEGORY_SLOWED) and generate (which
  bucket fires the Bernoulli) read this. Editing between train and
  generate produces a model that disagrees with the generator.
- **`extraction.kmer_size`, `upstream`, `downstream`** define the K-mer
  geometry. The shard `__meta__["extraction_params"]` freezes them so
  refine and train can cross-check. The model's config also records
  them so generate / predict-kmers / evaluate read them from the
  checkpoint (K-aware).
- **`extraction.rev_meth_offsets`** feeds FiLM. Must match training
  values structurally.
- **`generation.use_motif_fraction`** is a generate-only toggle. Default
  `false` because p_fire absorbs occupancy already.

The model checkpoint `model_config.json` carries:
- Architecture: kmer_size, active_site_index, num_meth_types, biology_mask,
  log_sigma_clamp_min/max
- `meth_id_map`: frozen at training time so the generate-side mapping
  matches even if the YAML was edited afterwards
- `p_fire[(meth, offset)]` and `mean_occupancy[(meth, offset)]` from
  refine, used by generate's Bernoulli firing

---

## Sharded mode (preferred for ≥ 10 strains)

`scripts/merge_shards.py` collapses N shards into a master pkl whose RAM
footprint scales linearly with the corpus. For larger runs the entire
pipeline supports a **sharded mode** that never holds the corpus in RAM:

```
extract  ──► <shards>/<sample>_shard.pkl       (one per strain, parallel via SLURM array)
refine   ──► <refined>/<sample>_clean.pkl      (per-bucket GMM filter, baseline-anchored)
train    ──► reads <refined>/ via SignalDataset (PyTorch IterableDataset).
             Worker-aware shard partition; per-epoch shard + row shuffling.
analyze  ──► concatenates shards in-memory before stats.
```

Train/test splitting (sharded mode):

- `--test-strains bc2080,bc2081,bc2082` — explicit by-sample-id holdout.
  Those shards never enter training. Real generalisation metric.
- `--test-fraction 0.10` — random per-shard split, reproducible via
  `--split-seed`.

`scripts/merge_shards.py` is kept for small datasets / debug / baseline.
Auto-detection on input path: directory → sharded path; file → in-memory
path. No CLI changes for the small-data case.

---

## Training set construction

`extract` is a single-pass pipeline that captures **positions of
expected kinetic signature** in addition to methylation positions
themselves. m6A/m4C/m5C signatures are observed at configured downstream
offsets — training only on methylation centers would deprive the model
of the slowing signal at those offsets.

```
  aligned bystrandified BAM + ref + motifs.csv
                                ↓
                          kinsim extract
                                ↓
                       shards/<sample>_shard.pkl
                                ↓
                          kinsim refine
                                ↓
                     refined/<sample>_clean.pkl
                                ↓
                          kinsim train
                                ↓
              checkpoints/  (model_config.json carries p_fire)
```

**Three categories** (col 17 of the 20-col layout):
- `0` BASELINE  — far from any methylation; meth_context window is empty.
- `1` SLOWED    — at a signature offset of a methylation. Includes the
  methylation itself when `0 ∈ signature_offsets[T]` (m6A, m4C). For m5C
  the methylation centre is NEAR_METH, since 0 is not in `[2, 6]`.
- `2` NEAR_METH — close to a methylation (within `[+1, near_meth_max_dist]`)
  but NOT at a signature offset of it. Negative control: meth in mc but
  IPD should look baseline.

**Parent meth attribution** (col 18 PARENT_METH + col 19 PARENT_OFFSET):
extract records which methylation produced each SLOWED/NEAR_METH row.
This lets analyze bucket rows per `(category, parent meth, parent offset)`
cleanly with a vectorised mask — no need to re-infer from the mc[] window.

Precedence on overlapping motif claims (rare):
- SLOWED beats NEAR_METH.
- Within SLOWED: last writer wins (deterministic given YAML iteration order).
- Within NEAR_METH: first writer wins.

**Per-position emission rules:**
For each motif-match position `p` of type `T`:
- For each `k ∈ signature_offsets[T]` (including 0): position `p+k` →
  CATEGORY_SLOWED.
- For each `k ∈ [0, near_meth_max_dist]` NOT in `signature_offsets[T]`:
  position `p+k` → CATEGORY_NEAR_METH (slowed wins on conflict).
- Positions far from any methylation (distance ≥ `baseline_min_dist_to_meth`,
  default = K) → CATEGORY_BASELINE candidate, capped per kmer at
  `n_baseline_per_kmer` via streaming reservoir sampling.

**Refine** uses a baseline-anchored 2-3 component Gaussian Mixture per
(meth_type, parent_offset) bucket, fit on raw IPD/PW (not log1p — that
empirically gives cleaner cluster separation given the uint8 quantisation).
The first component is initialised at the global baseline pool's
(mean, cov), so EM keeps it pinned at baseline kinetics. Free components
fit the meth signal. K∈{1,2,3} is BIC-picked with a strict biological
veto on K>2 (any non-anchor component placed at or below the anchor's
IPD is rejected — methylation never produces sub-baseline kinetics).

Refine keeps only slowed rows whose argmax-posterior component has mean
IPD strictly above the global baseline pool mean. Drops the rest
(including the initialised anchor if it stayed near baseline). The
per-bucket `p_fire = n_kept / n_in` is stored in `__meta__` and
propagates to the trained checkpoint as `model_config.p_fire`.

---

## Repository Layout

```
KinSim/
├── pyproject.toml                  single entry point: kinsim
│
├── kinsim/                         ML pipeline package
│   ├── __init__.py                 __version__ (single source)
│   ├── __main__.py                 CLI router
│   │
│   ├── extract.py                  Aligned bystrandified BAM → shard.pkl
│   ├── refine.py                   Baseline-anchored GMM per-(meth, offset)
│   ├── train.py                    Supervised training loop (ConvPredictor)
│   ├── generate.py                 BAM generation with trained model
│   ├── evaluate.py                 Calibration report + per-kmer plots
│   ├── verify_generate.py          Per-(kmer, meth) ref vs gen BAM
│   ├── analyze.py                  Training-data analysis dashboard
│   ├── predict_kmers.py            Dump (μ, σ) for every kmer × YAML scenario
│   │
│   ├── data/                       dataset classes
│   │   ├── __init__.py
│   │   └── dataset.py              log_transform, inv_log_transform,
│   │                               SignalDataset, KineticDataModule,
│   │                               list_shards, shard_sample_id
│   │
│   ├── models/                     neural model implementations
│   │   ├── __init__.py
│   │   └── predictor.py            ConvPredictor + create_from_config()
│   │
│   └── utils/                      shared utilities
│       ├── __init__.py
│       ├── encoding.py             K-mer bit-packing + get_meth_ids() (YAML-driven)
│       ├── sample_layout.py        20-col row layout + dynamic SampleLayout
│       ├── motifs.py               IUPAC motif parsing, sequence scanning, meth maps
│       ├── config.py               manifest CSV loader, YAML config, logging setup
│       ├── io.py                   FASTA loading, MAF parsing, PBSIM3 discovery
│       └── parsers/                methylation caller output parsers (plugin registry)
│           ├── __init__.py
│           ├── base.py             BaseOutputParser ABC
│           ├── registry.py         @register decorator, factory
│           ├── pacbio.py           PacBio motifs.csv parser
│           ├── modkit.py           modkit pileup bedMethyl (per-site, NOT IUPAC)
│           ├── combined.py         combined-format CSV
│           ├── rebase.py           REBASE web fetch + parse + fuzznuc patterns
│           └── motif_merge.py      merge / filter / dedup motifs
│
├── kinsim_baseline/                naive-Gaussian benchmark
│   ├── __main__.py                 CLI router
│   ├── compute.py                  Walks BAMs → per-(meth_type, offset) histograms
│   ├── analyze.py                  HTML plots of those histograms
│   ├── per_kmer.py                 AI vs observed per-kmer null comparison
│   ├── plot_kmer.py                4-panel dashboard for per_kmer output
│   └── generate.py                 Naive-Gaussian generation
│                                   (--format bam → lookup NPZ ; --format pkl → rewrite shard IPD/PW)
│
├── scripts/                        offline utilities (run with python scripts/X.py)
│   ├── manifest.py                 manifest CSV CLI (count / validate / list)
│   ├── sample.py                   subsample a shard pkl
│   ├── balance.py                  (legacy: errors out on int-keyed shards)
│   ├── filter.py                   (legacy: same)
│   ├── strip_kinetics.py           remove fi/fp/ri/rp from a BAM
│   ├── merge_shards.py             fuse multiple shard pkl into a master pkl
│   ├── compare.py                  cross-dataset kinetic comparison
│   └── inspect_null_model.py       inspect an ipdSummary .npz.gz null model
│
└── slurm_kinsim/                   HPC SLURM job scripts
    ├── pbsim3_simulate.slurm       PBSIM3 read simulation
    ├── ccs_subreads.slurm          ccs → HiFi BAM
    ├── validate.sh                 per-strain validate chain orchestrator
    │
    ├── prep/                       shared prep modules
    │   ├── bystrandify.slurm       ccs-kinetics-bystrandify
    │   ├── align_pbmm2.slurm       pbmm2 align (SKIP-first, filter unmapped, pbindex)
    │   ├── index_bam.slurm         samtools index + pbindex
    │   ├── assembly_hifiasm.slurm  hifiasm draft assembly
    │   └── README.md
    │
    ├── callers/                    methylation callers (any aligned BAM)
    │   ├── ipdsummary.slurm        ipdSummary SP3-C3 (m6A + m4C)
    │   ├── pbmotifmaker.slurm      motif discovery from ipdSummary GFF
    │   ├── jasmine_modkit.slurm    jasmine + modkit (5mC via CpG model)
    │   ├── merge_motifs.slurm      union of caller CSVs with threshold + dedup
    │   └── README.md
    │
    ├── validate/                   per-task SLURM for validate chain
    │   ├── prep.slurm              strip_kinetics + regions.txt
    │   ├── generate.slurm          kinsim generate (SKIP-first, FORCE_GEN=1 override)
    │   ├── merge.slurm             samtools merge + pbindex of shards
    │   └── write_regions.py
    │
    ├── ml/                         ML pipeline orchestrator
    │   ├── 00_extract.slurm        kinsim extract (array)
    │   ├── 02_refine.slurm         kinsim refine
    │   ├── 03_train.slurm          kinsim train (1 GPU)
    │   ├── 04_generate.slurm       kinsim generate on PBSIM3 reads (array)
    │   ├── 05_evaluate.slurm       kinsim evaluate
    │   ├── 06_verify_generate.slurm
    │   └── run.sh
    │
    ├── vega/                       per-dataset orchestrator
    ├── sequel/                     idem
    └── strepto/                    idem
```

---

## Validate chain (per-strain)

```
real_aligned.bam (bystrandified + aligned, ip/pw)
    │
    ▼  scripts/strip_kinetics.py
stripped.bam  (same, fi/fp/ri/rp removed)
    │
    ▼  kinsim generate (array, per-region)
shards/shard_NNN.bam  (flag=4 unmapped HiFi, fi/fp/ri/rp)
    │
    ▼  samtools merge + pbindex
SIM.bam  (one unmapped HiFi BAM, fi/fp/ri/rp)
    │
    ▼  ccs-kinetics-bystrandify
SIM_bystrandified.bam  (2 records per ZMW, ip/pw — matches real-data pipeline)
    │
    ├─► pbmm2 align ──► SIM_aligned.bam ──► ipdSummary ──► motifmaker ──► motifs_ipdsummary.csv
    │                                                                                      │
    └─► jasmine (re-aligns internally) + modkit ──► motifs_jasmine.csv                     │
                                                                                           ▼
                                                       merge_motifs.slurm (threshold=0.7, dedup)
                                                                                           │
                                                                                           ▼
                                                                       SIM_motifs_merged.csv
```

Final comparison: per-strain `SIM_motifs_merged.csv` ↔
`real_motifs_merged.csv` (recall, precision, motif-by-motif diff).

---

## Key Files — What Each Does

### `kinsim/utils/encoding.py`

```python
K = 11                          # default kmer size (legacy global)
KMER_LEFT_PAD = 7               # upstream of active site
KMER_RIGHT_PAD = 3              # downstream
KMER_PRED_IDX = KMER_LEFT_PAD   # = 7
KMER_MASK = (1 << (2 * K)) - 1
BASE_MAP  = {'A':0,'C':1,'G':2,'T':3}
METH_IDS  = {'none':0,'m6A':1,'m4C':2,'m5C':3}  # legacy; runtime → get_meth_ids()

encode_kmer(seq: str) -> int    # K-char string -> 2K-bit integer
decode_kmer(val: int) -> str    # 2K-bit integer -> K-char string
get_meth_ids() -> dict          # YAML-driven; extends METH_IDS with user types
```

Most consumers call `get_meth_ids()` rather than `METH_IDS` directly,
so YAML-declared meth types propagate.

### `kinsim/utils/motifs.py`

```python
iupac_to_re(motif)                                   # "RGATCY" -> "[AG]GAT[CT][CT]"
reverse_complement(seq)                              # IUPAC-aware
parse_motifs(motif_string, revcomp=True)             # "m6A,GATC,2;..." -> list of dicts
parse_motifs_per_strand(motif_string)                # → (fwd_motifs, rev_motifs)
scan_sequence(seq, motifs) -> np.int8[]              # per-base methylation ID array
parse_motifs_csv(path, ...)                          # thin wrapper around PacBioParser
load_motif_string(arg, ..., parser_name=None)        # auto-detect or explicit parser
build_reference_meth_map(ref_seqs, motif_string)     # genome-wide O(1) lookup array
build_reference_meth_map_per_strand(ref_seqs, ...)   # → (fwd_map, rev_map)
```

**Motif string format**: `"mod_type,pattern,1-based_position[,nDetected[,fraction]]"`
semicolon-delimited. Position is 1-based (matches PacBio `centerPos`).
`parse_motifs` subtracts 1 internally.

### `kinsim/utils/config.py`

```python
@dataclass
class SampleEntry:
    sample_id: str
    bam_path:  str
    motifs:    str   # KinSim string or path

@dataclass
class ExtractionParams:
    kmer_size: int
    upstream: int
    downstream: int
    rev_meth_offsets: tuple[int, ...]
    near_meth_max_dist: int
    n_baseline_per_kmer: int
    baseline_min_dist_to_meth: int
    baseline_sample_rate: float

load_manifest(manifest_path) -> list[SampleEntry]
validate_manifest(entries, check_files=True) -> list[str]
load_yaml_config(path) -> dict
load_kinsim_config(explicit_path=None) -> dict   # cached
get_extraction_params() -> ExtractionParams       # cached
get_signature_offsets(meth_name) -> list[int]
setup_logging(verbose=False)
```

### `kinsim/utils/sample_layout.py`

```python
SAMPLE_NCOLS = 20                # default K=11 layout
COL_IPD = 0
COL_PW = 1
COL_FRACTION = 2
COL_METH_CTX_START = 3
COL_METH_CTX_END = 14
COL_REV_METH = 14
COL_CATEGORY = 17
COL_PARENT_METH = 18
COL_PARENT_OFFSET = 19

CATEGORY_BASELINE = 0
CATEGORY_SLOWED = 1
CATEGORY_NEAR_METH = 2

# Dynamic for non-default K:
class SampleLayout:
    @classmethod
    def from_params(cls, params: ExtractionParams) -> SampleLayout: ...
get_sample_layout(params=None) -> SampleLayout
get_categories(arr) -> int8 ndarray
```

### `kinsim/data/dataset.py`

```python
log_transform(x) / inv_log_transform(x)
class SignalDataset(IterableDataset): ...        # walks one shard at a time
class KineticDataModule(L.LightningDataModule): ...
list_shards(dir) -> list[str]                   # prefers _clean.pkl when both exist
shard_sample_id(path) -> str
```

### `kinsim/extract.py`

```python
_check_bystrandified(bam_path)                  # refuses non-bystrandified input
extract_to_shard(bam_path, motif_string, output_path, params, ...)
extract_from_manifest_task(manifest_path, task_index, output_dir, ...)
```

**Input requirements**: aligned bystrandified BAM (2 records per ZMW with
`ip/pw`). KinSim refuses raw HiFi (fi/fp/ri/rp) — bystrandify it first.

### `kinsim/models/predictor.py`

```python
class ConvPredictor(nn.Module):
    # Per-base embedding + positional embedding + FiLM(meth) + Conv1D backbone
    # Dual readout: center position + global average pool
    # → [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]
    #
    # Geometry kwargs (read from YAML / model_config.json):
    #   kmer_size, active_site_index, n_rev_meth
    # log_sigma clamp: defaults (-6, 1.5) post-v0.5.0 bug fix
    # biology_mask: default False (extract already enforces base/meth chemistry)

def create_from_config(config: dict) -> nn.Module: ...
```

### `kinsim/train.py`
Default loss = **Beta-NLL (β=0.5)** — scale-corrected Gaussian NLL that
prevents the model from gaming σ in place of fitting μ.

CLI: `kinsim train <shards_dir> <output_dir> [--loss betanll|gnll|mse|huber]
[--biology-mask] [--log-sigma-clamp-max 1.5] [--test-strains a,b,c]
[--test-fraction 0.10] [--config kinsim_config.yaml]`

### `kinsim/generate.py`
Three calling modes (auto-detected):
- Directory mode: `kinsim generate <pbsim3_dir> <ckpt> <motifs> <output_dir>`
- BAM mode: `kinsim generate <input.bam> <ref.fna> <ckpt> <motifs> <output.bam>`
- Per-genome mode: `kinsim generate <fq.gz> <maf.gz> <ref.fna> <ckpt> <motifs> <out.bam>`

The mapped path is K-aware via the checkpoint's `kmer_size` /
`active_site_index`. Output is unaligned HiFi (`flag=4`,
`fi/fp/ri/rp`) — feed to `ccs-kinetics-bystrandify` then `pbmm2 align`
for downstream PacBio tools.

YAML reuse (see module docstring): per-site `p_fire` from refine drives
a Bernoulli at each motif site; `signal_offsets` from YAML decides
which positions in the kmer can fire.

### `kinsim/evaluate.py`
Calibration report + per-kmer distribution plots. Reads
`log_sigma_clamp_(min,max)` from the model so reported σ match what
generate emits.

CLI: `kinsim evaluate <ckpt_dir> <pkl>`
     `kinsim evaluate <ckpt_dir> <pkl> --kmer ACGT...ACGT --meth m6A --plot out.png`

### `kinsim/analyze.py`
Diagnostic dashboard for any shard or refined directory.

CLI: `kinsim analyze <pkl-or-dir> [--output-dir reports/] [--no-html]`

Per-meth-type kmer figures:
- Top-12 by slowed-meth count (Gaussian fit overlay, μ_b/σ_b/μ_s/σ_s annotated)
- 12 random kmers (≥ 50 slowed rows; same overlay)
- Per-kmer baseline-mean distribution
- 3D (IPD, PW) density per bucket

### `kinsim/predict_kmers.py`
Dump (μ, σ) for every kmer × every YAML methylation scenario. Outputs:
- `<prefix>.tsv` — wide human-readable table
- `<prefix>.npz` — compact binary, consumed by `generate --use-lookup`
  and by `kinsim_baseline make-lookup`
- `<prefix>.html` — per-scenario μ_ipd/μ_baseline distribution across
  all 4^K kmers

K-aware via `model_config.json`. Errors out gracefully if 4^K > 1e8.

### `kinsim/verify_generate.py`
Per-(kmer, meth) reference vs generated BAM comparison.

CLI: `kinsim verify-generate <ref.bam> <gen.bam> <motifs> <report.tsv>`

### `kinsim/utils/parsers/`
Plugin registry with `@register` decorator.

```python
from kinsim.utils.parsers import create_parser, list_parsers, auto_detect_parser
parser = create_parser("pacbio")     # or "modkit", "combined"
motif_string = parser.parse("motifs.csv", min_fraction=0.40, min_detected=20)
parser = auto_detect_parser("output.csv")
list_parsers()                       # ['combined', 'modkit', 'pacbio']
```

**Note**: `modkit` emits per-site pseudo-motifs (`chrom:pos:strand`)
that are NOT valid IUPAC. Use it for upstream motif-discovery
(`modkit find-motifs`), not as direct input to `kinsim extract`.

---

## Data Flow (short)

```
Aligned bystrandified BAMs (sorted, indexed, ip/pw tags)
      ↓  motif discovery (jasmine + modkit + ipdSummary + pbmotifmaker)
      ↓  manifest.csv  [sample_id, bam_path, motifs]
      ↓  kinsim extract → shards/
strain1_shard.pkl   strain2_shard.pkl   ...
      ↓   (20 cols, key=kmer_id, CATEGORY at col 17, PARENT_METH 18, PARENT_OFFSET 19)
      ↓  kinsim refine shards/ refined/
refined/<sample>_clean.pkl
      ↓  kinsim train refined/ checkpoints/
checkpoint_epochNN.pt + model_config.json (p_fire, meth_id_map)
      ↓  kinsim generate <stripped.bam> <ref> <ckpt> <motifs> <output.bam>
SIM.bam  (unmapped HiFi, fi/fp/ri/rp)
      ↓  ccs-kinetics-bystrandify + pbmm2 align
      ↓  ipdSummary + motifmaker, jasmine + modkit
      ↓  merge_motifs (threshold 0.7)
SIM_motifs_merged.csv  — compare to real_motifs_merged.csv
```

---

## BAM Output Contract

| Field | Value |
|---|---|
| `flag` | `4` (unmapped, raw HiFi convention) |
| `fi:B:C` | IPD forward, uint8, length == read length |
| `fp:B:C` | PW forward, uint8 |
| `ri:B:C` | IPD reverse, uint8 |
| `rp:B:C` | PW reverse, uint8 |
| N-context positions | signal = `1` (not `0`, which means "no data") |
| Header `@PG` | KinSim entry with version + training corpus + K + arch, chained to upstream tools |
| Header `@RG` | inherited from input (suffix-cleaned), or synthetic `00000000/0--0` with full PacBio metadata |

---

## CLI Command Map

```
# kinsim — ML pipeline ----------------------------------------------------
kinsim extract                -> kinsim/extract.py            (aligned bystr. BAM → shard.pkl)
kinsim refine                 -> kinsim/refine.py             (baseline-anchored GMM)
kinsim train                  -> kinsim/train.py
kinsim generate               -> kinsim/generate.py
kinsim evaluate               -> kinsim/evaluate.py
kinsim verify-generate        -> kinsim/verify_generate.py
kinsim analyze                -> kinsim/analyze.py
kinsim predict-kmers          -> kinsim/predict_kmers.py

# kinsim_baseline — naive Gaussian benchmark -----------------------------
python -m kinsim_baseline compute        — walks BAMs → per-(T, k) histograms
python -m kinsim_baseline analyze        — plots those histograms
python -m kinsim_baseline per-kmer       — AI-baseline vs observed per-kmer
python -m kinsim_baseline plot-per-kmer  — 4-panel dashboard
python -m kinsim_baseline generate       — naive BAM (lookup NPZ) or naive shard

# Offline tooling --------------------------------------------------------
python scripts/manifest.py        — count / validate / list manifest CSV
python scripts/sample.py          — subsample a shard pkl
python scripts/strip_kinetics.py  — strip fi/fp/ri/rp from a BAM
python scripts/merge_shards.py    — fuse shards into a master pkl
python scripts/compare.py         — cross-dataset kinetic comparison

python -m kinsim.utils.parsers.rebase       — REBASE fetch / parse
python -m kinsim.utils.parsers.motif_merge  — merge + filter + dedup motifs
```

Typo suggestions via `difflib.get_close_matches` in `kinsim/__main__.py`.

---

## Coding Conventions

### General
- Python 3.10+. Type hints on public function signatures in `models/`.
- No global mutable state. All functions take explicit arguments.
- `Path` objects for file I/O, except in SLURM scripts.
- `sys.exit(1)` on fatal errors with a stderr message.
- Never catch `Exception` broadly. Catch specific exceptions.

### Neural Models
- Xavier uniform init for `nn.Linear`, small normal (`std=0.02`) for `nn.Embedding`.
- Always `model.eval()` before inference; `model.train()` at start of epoch.
- `@torch.no_grad()` on all inference functions.
- `ReduceLROnPlateau` — never `verbose=True` (PyTorch ≥ 2.1).
- Wrap training loops in `try/finally` to ensure CSV/TensorBoard closed on crash.
- Always save `model_config.json` **before** the first epoch.
- Include `"scheduler"` in checkpoint dict for resume support.
- Use `create_from_config()` to load — never hardcode architecture.

### Logging
- Every module: `log = logging.getLogger(__name__)`.
- Never `print()` for operational output.
- `setup_logging()` from `kinsim.utils.config` called once in each CLI `main()`.

### Signal Space
- **Training**: log1p (via `log_transform`).
- **Inference**: log1p → `inv_log_transform` → uint8 [0, 255].
- **Storage (.pkl)**: raw values. 20-col layout.
- **Metadata**: every .pkl has `"__meta__"` (provenance, ExtractionParams).

### SLURM
- Array jobs for per-task / per-region work.
- SKIP-first by default. `FORCE_X=1` env override.
- `set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail`
  preamble (IBU cluster — bashrc trips unbound vars).
- Diagnostics: date, hostname, GPU info, timing, exit codes.

### Manifest CSV
- Columns: `sample_id`, `bam_path`, `motifs`. Optional: `ref_path`.
- Row count: `python scripts/manifest.py count manifest.csv`.
- Output shard: `shards/<sample_id>_shard.pkl`.

---

## What NOT To Do

- **Do not** store log-transformed data in `.pkl` — raw values only.
- **Do not** use `verbose=True` in `ReduceLROnPlateau`.
- **Do not** hardcode architecture in `generate.py` — read `model_config.json`.
- **Do not** add caller parsers outside `kinsim/utils/parsers/`.
- **Do not** add pipeline logic to `kinsim/__main__.py` — convenience CLIs
  live in `scripts/`.
- **Do not** import parsers eagerly — use lazy imports.
- **Do not** edit `kinsim_config.yaml` between training and generation
  unless you understand the implications.

---

## Adding a New Motif Parser

1. Create `kinsim/utils/parsers/<name>.py` with `@register` class
   inheriting `BaseOutputParser`.
2. Define `name`, `supported_mods`, `parse()`, `is_file_for_this_parser()`.
3. Import the new module in `kinsim/utils/parsers/__init__.py`.

---

## Key Numbers

| Constant | Value | Where |
|---|---|---|
| K (default kmer size) | 11 | `kinsim_config.yaml` (overridable) |
| Total kmers at K=11 | 4,194,304 | `4 ** K` |
| Default meth states | 4 (none/m6A/m4C/m5C) | YAML kinetic_signatures |
| Default upstream | 7 | YAML |
| Default downstream | 3 | YAML |
| Reservoir cap (per-kmer baseline) | 50 | YAML `n_baseline_per_kmer` |
| BAM signal range | [0, 255] uint8 | `kinsim/generate.py` |
| N-context default signal | 1 | `kinsim/generate.py` |
| log_sigma clamp | [-6, 1.5] | YAML `model.log_sigma_clamp_max` |
| Train/val split | 90% / 10% | `kinsim/train.py` |
| ConvPredictor params | ~140K (default cd=128, ncl=3, hd=128) | `kinsim/models/predictor.py` |
| meth_proj_dim | 8 | YAML `model.meth_proj_dim` |

---

## Future Work

- **K=21 unmapped path in generate**: mapped path is K-aware, but
  `_process_read_unmapped_vec` still uses module-level `K` /
  `KMER_MASK`. Predict-kmers refuses 4^K > 1e8 to avoid OOM — for K=21
  a sample-and-extrapolate strategy is needed.
- **Wider rev_meth window**: today captures only `[-1, 0, +1]`. 8+ bp
  palindromic motifs (some Type II R-M) place the partner meth at
  ±3 to ±5. Layout change + retrain.
- **Per-occupancy p_fire curve**: today's decomposition is linear in
  occupancy. Bin per occupancy could capture non-linearities.
- **Vectorise the mapped path inner loop**: ~50× slower than the
  unmapped path. Blocks K=21 experiments at scale.
- **Pure naive baseline integration in validate.sh**: kinsim_baseline
  generate emits a lookup NPZ; wire it as an opt-in `--baseline-only`
  mode for side-by-side ML vs naive recall.
