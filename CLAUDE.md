# KinSim — Developer Reference for Claude

## Project Summary

KinSim simulates PacBio HiFi kinetic signals (IPD and PW) for metagenomic binning research.
Given PBSIM3-simulated reads and a reference genome, KinSim injects biologically realistic
per-base IPD/PW values into unaligned BAM files using a supervised MLP/Conv model that
predicts N(mu, sigma^2) per 11-mer context with methylation conditioning.

Output BAMs carry standard PacBio tags: `fi:B:C` (IPD) and `fp:B:C` (PW).

Two CLI tools are installed from the same repository:
- **`kinsim`**      — ML pipeline: extract, merge, train, generate, evaluate, analyze
- **`kinsim-prep`** — Data preparation: rebase, merge-motifs, manifest, filter, balance, parse

## v4 Training Set (current — single-pass extract)

The v4 architecture (April-May 2026) extends extraction to capture
**positions of expected kinetic signature** in addition to methylation
positions themselves. Rationale: m6A/m4C/m5C signatures are observed at
configured downstream offsets (m6A at 0 and +5, m5C at +2 and +6, etc.,
declared in `kinsim_config.yaml`). Training only on methylation centers
deprives the model of the slowing signal at those offsets.

**Single-pass pipeline (no bootstrap):**

```
  BAM + motifs --> kinsim extract --v4    --> shard_v4.pkl
                                                 (36 cols, key=kmer_id, CATEGORY at col 35)
               --> kinsim merge            --> master_v4.pkl
               --> kinsim refine           --> master_v4_clean.pkl
                       (auto-detects v4)
               --> kinsim train            --> checkpoint.pt
```

**Three categories** (col 35 of the 36-col layout):
- `0` BASELINE  — far from any methylation, meth_context window is empty.
- `1` SLOWED    — at a signature offset of a methylation. Includes the
  methylation itself when `0 ∈ signature_offsets[T]` (m6A, m4C). For m5C
  the methylation center is NEAR_METH, since 0 is not in `[2, 6]`.
- `2` NEAR_METH — close to a methylation (within `[+1, near_meth_max_dist]`)
  but NOT at a signature offset of it. Negative control: meth in mc but
  IPD should look baseline.

**Per-position emission rules in v4 extract (`--v4` flag):**
For each motif-match position `p` of type `T`:
- For each `k ∈ signature_offsets[T]` (including 0): position `p+k` →
  CATEGORY_SLOWED.
- For each `k ∈ [0, near_meth_max_dist]` NOT in `signature_offsets[T]`:
  position `p+k` → CATEGORY_NEAR_METH (slowed wins on conflict).
Positions far from any methylation (distance ≥ `baseline_min_dist_to_meth`,
default = K = 11) → CATEGORY_BASELINE candidate, capped per kmer at
`n_baseline_per_kmer` via end-of-stream uniform subsample.

**Refine pipeline:** ONE pass — `slowed_split_v4`. Pools the IPD of all
CATEGORY_BASELINE samples, computes the `secondary_percentile`-th
percentile (default 95) as the lower threshold, drops CATEGORY_SLOWED
samples below the threshold (FP motifs whose expected slowing did not
occur). CATEGORY_BASELINE and CATEGORY_NEAR_METH pass through
unchanged. NO GMM step in v4.

**The v3 35-col layout is preserved for backward compatibility** (calling
`kinsim extract` without `--v4` emits the legacy format). It is no longer
required for any v4 step. The `--refined-pkl PATH` flag still exists as
an opt-in oracle (uses a trusted master_clean from a previous run as
confirmation source instead of motif-match); almost no one needs it.

---

## Repository Layout

```
KinSim/
├── pyproject.toml                  entry points: kinsim + kinsim-prep
├── requirements.txt
│
├── kinsim/                         ML pipeline package (v0.4.0, MLP-only)
│   ├── __init__.py
│   ├── __main__.py                 CLI router (v0.4.0)
│   │
│   ├── extract.py                  BAM extraction + shard merging (motif-based, manifest-driven)
│   ├── refine.py                   v3: GMM only. v4: p95 filter on slowed only
│   ├── train.py                    supervised training loop (ConvPredictor/MLPPredictor)
│   ├── generate.py                 BAM generation with trained model
│   ├── evaluate.py                 calibration report + per-kmer distribution plots
│   ├── verify_generate.py          per-(kmer, meth) reference vs generated BAM comparison
│   ├── analyze.py                  training data analysis (coverage, signals, sensitivity)
│   ├── sample.py                   random subsampling of .pkl files
│   ├── strip_kinetics.py           remove fi/fp/ri/rp tags from BAM copy
│   │
│   ├── data/                       dataset classes
│   │   ├── __init__.py
│   │   └── dataset.py              log_transform, inv_log_transform, MLPSignalDataset
│   │
│   ├── models/                     neural model implementations
│   │   ├── __init__.py
│   │   └── predictor.py            ConvPredictor + MLPPredictor + create_from_config()
│   │
│   └── utils/                      shared utilities
│       ├── __init__.py
│       ├── encoding.py             11-mer bit-packing (no dependencies)
│       ├── motifs.py               IUPAC motif parsing, sequence scanning, meth maps
│       ├── config.py               manifest CSV loader, YAML config, logging setup
│       └── io.py                   FASTA loading, MAF parsing, PBSIM3 discovery
│
├── prep/                           data preparation package (kinsim-prep CLI)
│   ├── __init__.py
│   ├── __main__.py                 CLI router for kinsim-prep
│   │
│   ├── rebase.py                   REBASE web fetch + file parsing + fuzznuc patterns
│   ├── motif_merge.py              merge/filter/dedup motifs -> standard PacBio CSV
│   ├── manifest.py                 manifest CSV CLI (count / validate / list)
│   ├── balance.py                  balance .pkl by methylation type
│   ├── filter.py                   filter .pkl by coverage, mod type, max keys
│   └── callers/                    methylation caller output parsers (plugin registry)
│       ├── __init__.py             exports: BaseOutputParser, create_parser, list_parsers, auto_detect_parser
│       ├── base.py                 BaseOutputParser ABC
│       ├── registry.py             @register decorator, factory functions
│       ├── pacbio.py               PacBioParser -- motifs.csv with variable columns
│       ├── modkit.py               ModkitParser -- modkit pileup --bedMethyl TSV
│       ├── ipd_summary.py          IpdSummaryParser -- ipdSummary CSV/GFF3
│       └── combined.py             CombinedParser -- mod_type,motif,offset,frac_mod,n_sites,source
│
├── archive/                        archived code (dictionary, cGAN) — not active
│
├── baseline/                       baseline models for comparison
│   ├── __init__.py
│   ├── global_gaussian.py          4 Gaussians (one per meth type, no kmer context)
│   ├── kmer_gaussian.py            per-kmer Gaussian + global IPD ratio shift
│   └── conv_no_film.py             ConvPredictor without FiLM (post-hoc ratio shift)
│
└── slurm_kinsim/                   HPC SLURM job scripts
    ├── pbsim3_simulate.slurm       PBSIM3 read simulation
    ├── jasmine_5mc.slurm           jasmine + modkit 5mC motif discovery (array)
    │
    ├── vega/                       PREP pipeline 1 — Vega HiFi → assembly → motifmaker
    │   ├── 00_assembly.slurm       hifiasm draft assembly
    │   ├── 01_bystrandify.slurm    ccs-kinetics-bystrandify
    │   ├── 02_align.slurm          pbmm2 align
    │   ├── 03_index.slurm          samtools index + pbindex
    │   ├── 04_ipdsummary.slurm     ipdSummary SP3-C3
    │   ├── 05_motifmaker.slurm     pbmotifmaker find
    │   ├── 06_build_manifest.sh    emit manifest_vega_gff.csv
    │   └── run.sh                  orchestrator (chains all with afterok)
    │
    ├── sequel/                     PREP pipeline 2 — Sequel subreads → CCS → motifmaker
    │   ├── 00_ccs.slurm            subreads → HiFi
    │   ├── 01_bystrandify.slurm
    │   ├── 02_align.slurm
    │   ├── 03_index.slurm
    │   ├── 04_ipdsummary.slurm
    │   ├── 05_motifmaker.slurm
    │   ├── 06_build_manifest.sh
    │   └── run.sh
    │
    ├── strepto/                    PREP pipeline 3 — Strepto HiFi (manifest-driven)
    │   ├── 00_bystrandify.slurm
    │   ├── 01_align.slurm
    │   ├── 02_index.slurm
    │   ├── 03_ipdsummary.slurm
    │   ├── 04_motifmaker.slurm
    │   ├── 05_build_manifest.sh
    │   └── run.sh
    │
    ├── ml/                         ML pipeline — generic across Vega/Sequel/Strepto
    │   ├── 00_extract.slurm        kinsim extract — legacy v3 (35-col)
    │   ├── 00b_extract_v4.slurm    kinsim extract --v4 — recommended (36-col CATEGORY)
    │   ├── 01_merge.slurm          kinsim merge shards → master.pkl
    │   ├── 02_refine.slurm         kinsim refine — v3 (GMM only) / v4 (p95 on slowed)
    │   ├── 03_train.slurm          kinsim train (1 GPU)
    │   ├── 04_generate.slurm       kinsim generate on PBSIM3 reads (array)
    │   ├── 05_evaluate.slurm       kinsim evaluate
    │   ├── 06_verify_generate.slurm   kinsim verify-generate (array)
    │   └── run.sh                  orchestrator — extract/merge/refine/train/evaluate chain
    │
    ├── config/                     example configuration files
    │   ├── config_example.yaml     training config example
    │   └── manifest_example.csv    manifest CSV example
    │
    └── msa1003/                    MSA1003 data extraction pipeline (legacy reference)
        ├── prep_rebase.sh          fetch REBASE motifs for each species
        ├── prep_merge.sh           merge calling + REBASE motifs, build manifest
        ├── 00_align_split.slurm    align bc2036 BAM + split by species
        ├── 00b_add_ippw.slurm      convert fi/fp/ri/rp → ip/pw per species
        ├── 01_ipdsummary.slurm     run ipdSummary per species
        ├── 01b_modkit.slurm        run modkit as alternative caller
        └── 02_pbmotifmaker.slurm   detect motifs from ipdSummary GFF
```

---

## Key Files — What Each Does

### `kinsim/utils/encoding.py`
Pure functions, no external dependencies. The foundation of all k-mer logic.

```python
K = 11                          # window size (fixed everywhere)
KMER_MASK = (1 << 22) - 1      # 22-bit mask for sliding window
BASE_MAP  = {'A':0,'C':1,'G':2,'T':3}
METH_IDS  = {'none':0,'m6A':1,'m4C':2,'m5C':3}

encode_kmer(seq: str) -> int    # 11-char string -> 22-bit integer
decode_kmer(val: int) -> str    # 22-bit integer -> 11-char string
```

### `kinsim/utils/motifs.py`
Everything methylation-motif related.

```python
iupac_to_re(motif)                        # "RGATCY" -> "[AG]GAT[CT][CT]"
reverse_complement(seq)                    # IUPAC-aware
parse_motifs(motif_string, revcomp=True)   # "m6A,GATC,1;..." -> list of dicts
scan_sequence(seq, motifs) -> np.int8[]   # per-base methylation ID array
parse_motifs_csv(path, min_fraction=0.40)  # PacBio motifs.csv -> KinSim string
load_motif_string(arg, ..., parser_name=None)  # auto-detect or explicit parser
build_reference_meth_map(ref_seqs, motif_string)  # genome-wide O(1) lookup array
```

**Motif string format**: `"mod_type,pattern,position"` semicolon-delimited.
Example: `"m6A,GATC,1;m4C,CCWGG,2"`.
Position is 0-based index of the modified base within the pattern.

`load_motif_string()` uses lazy imports from `prep.callers` and `prep.rebase`
for file-based motif loading. Both packages are installed together so these
imports always succeed.

`build_reference_meth_map()` uses EMBOSS fuzznuc as primary backend for genome-wide
motif scanning, with automatic fallback to Python regex if fuzznuc is not installed
or returns empty results (fuzznuc can silently fail on some IUPAC patterns).

### `kinsim/utils/config.py`
Manifest CSV parsing, YAML config loading, and logging setup.
Used by both `kinsim` and `prep` packages.

```python
@dataclass
class SampleEntry:
    sample_id: str
    bam_path:  str
    motifs:    str    # KinSim string or path (resolved by load_motif_string)

load_manifest(manifest_path) -> list[SampleEntry]
validate_manifest(entries, check_files=True) -> list[str]
load_yaml_config(path) -> dict
load_kinsim_config(explicit_path=None) -> dict   # parses kinsim_config.yaml
get_signature_offsets(meth_name) -> list[int]    # signature offsets per meth type
setup_logging(verbose=False)
```

#### Project-wide config — `kinsim_config.yaml`

A single YAML at the repo root holds the biology- and refine-related parameters
that the user must keep up-to-date. Loaded once and cached by
`load_kinsim_config()`.

```yaml
kinetic_signatures:
  m6A: { signal_offsets: [0, 5] }   # at modified A AND +5 downstream
  m4C: { signal_offsets: [0] }      # at modified C only
  m5C: { signal_offsets: [2, 6] }   # +2 and +6, NOT at the C itself
  # User MUST add an entry per methylation type their data carries.
  # If a type is missing, KinSim falls back to [0] and logs a warning —
  # this is correct for m4C, but WRONG for m5C and incomplete for m6A.

meth_context:    { left: 7, right: 3 }    # asymmetric kmer / FiLM window
kinetic_profile: { start: 0, end: 8 }     # downstream profile stored per sample

refine:
  default_strategy: "gmm_signature"
  gmm_signature:
    k_max: 3
    chi2_threshold: 9.21
    min_signature_ratio: 1.3
    min_pi: 0.05
    min_samples_for_gmm: 5
  slowed_split:                    # v4 pass-2 (operates on v4 36-col input)
    n_baseline_per_kmer: 50
    secondary_percentile: 95

extract:                           # v4 extract knobs
  n_baseline_per_kmer:        50   # per-kmer baseline cap
  baseline_min_dist_to_meth:  11   # bases (>= K so meth_context stays clean)
```

Strain-specific signatures (e.g. m6A at +8 instead of +5 for some
methyltransferases) are handled by editing the YAML — no code change.

### `kinsim/utils/io.py`
File I/O for FASTA references, MAF alignments, and PBSIM3 directory discovery.

```python
load_reference(fasta_path) -> dict[str, str]       # contig_name -> sequence
parse_maf(maf_path) -> iterator                     # MAF alignment records
get_extended_context(ref, pos, k) -> str            # extract 11-mer from reference
discover_pbsim3_layout(directory) -> list[dict]     # auto-detect flat vs subdirectory
```

### `kinsim/utils/sample_layout.py`
Per-sample column layout, format detection, category inference. Pure
Python (no pysam) so refine/dataset/tests can import it without the
extract/BAM dependency.

```python
SAMPLE_NCOLS_V3 = 35    # legacy v3 layout
SAMPLE_NCOLS    = 36    # v4 layout (= V3 + 1 CATEGORY column)
COL_CATEGORY    = 35    # last col on v4
CATEGORY_BASELINE  = 0
CATEGORY_SLOWED    = 1   # at a signature offset (incl. meth itself if 0 ∈ sig)
CATEGORY_NEAR_METH = 2   # close to meth but not at signature offset

is_v4_format(arr)    -> bool   # True iff arr.shape[1] >= 36
get_categories(arr, signature_offsets_by_meth=None,
               near_meth_max_dist=7) -> int8 ndarray
    # v4: arr[:, COL_CATEGORY]
    # v3: inferred from meth_context using sig offsets + proximity window

slice_meth_context(meth_status, center) -> list[11]
slice_rev_meth(meth_status_complement, center) -> list[3]
slice_kinetic_profile(ipds, pws, center) -> list[18]
```

### `kinsim/data/dataset.py`
Dataset class and signal transforms. Never import transforms from model files.

```python
log_transform(x: Tensor) -> Tensor      # log1p -- raw [0,255] -> training space
inv_log_transform(x: Tensor) -> Tensor  # expm1 clamped to [0,255] -- inference

class MLPSignalDataset(Dataset):
    # Auto-detects v3 (tuple keys, per-bucket cap) vs v4 (int kmer keys,
    # no resampling — extract has already capped baseline).
    # Returns (kmer_id, meth_probs, log_signal, meth_id) tuples.
    # For v4, meth_id at center is read from mc[KMER_PRED_IDX] (col 10).
```

### `kinsim/extract.py`
The data preparation pipeline. **Motif-based extraction only** (GFF mode was
removed in v3; the model now learns kinetic signatures from the per-position
methylation context fed to FiLM, so a separate aligned-BAM path is unnecessary).

```python
validate_bam_kinetics(bam_path, n_check=10)

# v3 legacy path (no --v4 flag): emits dict[(kmer, meth_id)] -> 35-col
extract_samples_from_bam(bam_path, motif_string, ...)
extract_from_manifest_task(manifest_path, task_index, output_dir, ...)

# v4 default standalone path (--v4 flag): emits dict[kmer_id] -> 36-col with CATEGORY
# refined_pkl_path is OPTIONAL; without it, every motif-match is a candidate
# methylation and `kinsim refine` GMM filters FP downstream.
extract_samples_v4_from_bam(bam, motif_string, refined_pkl_path=None,
                             n_baseline_per_kmer=50,
                             baseline_min_dist_to_meth=K, ...)
extract_v4_from_manifest_task(..., refined_pkl_path=None)

# Auto-detects v3 vs v4 shards, rejects mixing both
merge_shards(input_dir, output_file, max_samples_per_key=50000)
```

**Supported BAM formats:**

| Format | Tags | Reverse-strand support | Recommended |
|---|---|---|---|
| Raw HiFi (unaligned) | `fi/fp` + `ri/rp` | ✅ Both strands per read | ✅ Yes |
| Bystrandified | `ip/pw` (×2 reads) | ✅ Each strand = own read | ✅ Yes (modern) |
| Aligned post-pbmm2 | `ip/pw` only | ❌ `ri/rp` dropped on alignment | ❌ Not supported — pass an unaligned BAM |

`validate_bam_kinetics()` returns `"fi"` or `"ip"` to auto-route the forward
extraction; the reverse-strand pass is gated by `read.has_tag("ri")`. Users
should pass either a **raw HiFi BAM** or a **bystrandified BAM** to get full
two-strand training data. Aligned BAMs lose `ri/rp` and only the forward path
is captured (half the data).

### `kinsim/models/predictor.py`
Dual architecture with factory pattern:

```python
class ConvPredictor(nn.Module):    # DEFAULT (~140K params)
    # Per-base embedding (4x16) + positional embedding (11x16)
    # FiLM conditioning: methylation modulates base embeddings
    # Conv1D backbone (3 layers, k=3, BatchNorm, GELU)
    # Dual readout: center position + global average pool
    # -> [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]

class MLPPredictor(nn.Module):     # LEGACY (~268M params)
    # Flat 4.2M-row kmer embedding + 2-layer MLP

def create_from_config(config: dict) -> nn.Module:
    # Reads "architecture" key ("conv" or "mlp"), defaults to "conv"
```

Both share interface: `forward(kmer_ids, meth_probs)`, `sample()`, `predict_mean()`, `get_config()`.

### `kinsim/train.py`
Supervised training with Gaussian NLL loss.

```python
Loss = 0.5 * (2*log_sigma + (target - mu)^2 / sigma^2)
```

CLI: `kinsim train <pkl> <output_dir> [--architecture conv|mlp] [--config config.yaml]`

### `kinsim/generate.py`
BAM generation with trained model. Uses `create_from_config()` for loading.

Three auto-detected calling modes:
- Directory mode: `kinsim generate <pbsim3_dir> <checkpoint.pt> <motifs> <output_dir>`
- BAM mode: `kinsim generate <input.bam> <ref.fna> <checkpoint.pt> <motifs> <output.bam>`
- Per-genome mode: `kinsim generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>`

### `kinsim/evaluate.py`
Calibration report + per-kmer distribution plots.
CLI: `kinsim evaluate <ckpt_dir> <pkl>`

### `kinsim/analyze.py`
Training data analysis report (text + optional HTML).
CLI: `kinsim analyze <pkl> [--output-dir reports/] [--no-html]`

### `prep/` — Data Preparation Package

#### `prep/callers/`
Read-only parsing library for methylation caller output files. Plugin registry
with `@register` decorator — adding a new format = one file + `@register` class.

```python
from prep.callers import create_parser, list_parsers, auto_detect_parser

# Explicit parser
parser = create_parser("pacbio")       # or "modkit", "ipd_summary", "combined"
motif_string = parser.parse("motifs.csv", min_fraction=0.40, min_detected=20)

# Auto-detect from file content
parser = auto_detect_parser("output.csv")

# List registered parsers
list_parsers()  # ['combined', 'ipd_summary', 'modkit', 'pacbio']
```

**BaseOutputParser ABC** (`base.py`):
- `name: ClassVar[str]` — registry key
- `supported_mods: ClassVar[list[str]]` — mod types this format carries
- `parse(filepath, min_fraction, min_detected) -> str` — file -> motif string
- `is_file_for_this_parser(filepath) -> bool` — heuristic for auto-detection

**PacBioParser** (`pacbio.py`): Handles motifs.csv with variable columns.
Required: `motifString`, `centerPos`. Optional: `modificationType`, `fraction`, `nDetected`.

**ModkitParser** (`modkit.py`): Handles modkit pileup `--bedMethyl` TSV (11+ columns).

**IpdSummaryParser** (`ipd_summary.py`): Auto-detects CSV vs GFF3 from ipdSummary.

**CombinedParser** (`combined.py`): Handles combined methylation CSV with columns:
`mod_type,motif,offset,frac_mod,n_sites,source`. Auto-detected when CSV header
contains both `mod_type` and `frac_mod`.

**Integration**: `load_motif_string()` in `kinsim/utils/motifs.py` accepts optional
`parser_name` kwarg. When provided, bypasses auto-detection and uses the named parser.

#### `prep/rebase.py`
Converts REBASE notation (1-based) to KinSim notation (0-based).

```python
parse_rebase_annotation(recognition_seq, meth_annotation) -> list[str]
parse_rebase_simple(filepath)       # 2-column TSV format
parse_rebase_withrefm(filepath)     # Format #19 tagged fields (RS=, MS=)
parse_rebase_file(filepath)         # auto-detect format
parse_rebase_isoschizomers(filepath) -> dict[str, list[str]]
write_fuzznuc_pattern_file(motif_string, filepath) -> dict
decode_fuzznuc_pattern_name(name) -> (meth_id, mod_pos)
fetch_rebase_org(org_num, output_path) -> list[dict]   # web fetch
```

CLI:
- `kinsim-prep rebase fetch <org_num>` — fetch from REBASE website, write CSV
- `kinsim-prep rebase parse <file>` — parse local REBASE file, print motif string
- `kinsim-prep rebase patterns <motifs> <outfile>` — write fuzznuc pattern file

#### `prep/motif_merge.py`
Merges, filters, and deduplicates motifs from calling-derived CSV and REBASE
into a single standard PacBio `motifs.csv`.

CLI: `kinsim-prep merge-motifs species_motifs.csv rebase_motifs.csv --output final_motifs.csv`

#### `prep/filter.py`
Filter .pkl by coverage, mod type, or max keys.

CLI: `kinsim-prep filter general.pkl training.pkl --min-coverage 50 --mod-type m6A,m5C`

#### `prep/manifest.py`
Manifest CSV inspection utilities.

CLI:
```
kinsim-prep manifest count <csv>       # prints integer for SLURM --array
kinsim-prep manifest validate <csv>    # checks duplicates, file existence
kinsim-prep manifest list <csv>        # tabular display
```

---

## Data Flow

```
Real PacBio BAMs (raw HiFi or bystrandified — never aligned post-pbmm2)
      |
      +-- Motif discovery (jasmine + ipdSummary + pbmotifmaker, threshold 0.7)
      |   produces a merged motifs.csv per sample
      |
      v  manifest.csv  [sample_id, bam_path, motifs]
      |
      |  ---- v4 EXTRACT (single pass, no bootstrap) ----
      v  kinsim extract --v4 --manifest manifest.csv --task $TASK --output-dir shards_v4/
strain1_shard_v4.pkl  strain2_shard_v4.pkl  ...
      |   (v4: 36 cols + CATEGORY column, key=kmer_id only)
      |   Each row tagged 0=baseline / 1=meth / 2=slowed
      |
      v  kinsim merge shards_v4/ master_v4.pkl
      |
      v  kinsim refine master_v4.pkl master_v4_clean.pkl
      |   (auto-detects v4 format)
      |   p95 filter on CATEGORY_SLOWED only — drops slowed samples
      |   whose IPD < p95(baseline). Baseline + near_meth pass through.
master_v4_clean.pkl
      |
      v  kinsim train master_v4_clean.pkl checkpoints/
      v
checkpoint_epoch50.pt + model_config.json
      |
      v  kinsim generate <pbsim3_reads> <ref> <ckpt.pt> <motifs> <output.bam>
      v
species_mlp.bam
  flag=4  fi:B:C (IPD uint8)  fp:B:C (PW uint8)
      |
      v  kinsim verify-generate <ref.bam> <gen.bam> <motifs> <report.tsv>
      v  (per-(kmer, meth) mean/sd comparison — Pearson r + MAE summary)
verify_report.tsv
```

Legacy v3 chain (still works for backward compat):
```
  kinsim extract                      (no --v4)             → shard.pkl
  kinsim merge shards/ master.pkl
  kinsim refine master.pkl master_clean.pkl                 → v3 GMM only
  kinsim train master_clean.pkl checkpoints/
```

---

## BAM Output Contract

| Field | Value |
|---|---|
| `flag` | `4` (unmapped) |
| `fi:B:C` | IPD per base, uint8, length == read length |
| `fp:B:C` | PW per base, uint8, length == read length |
| N-context positions | signal = `1` (not `0`, which means "no data" in PacBio) |
| Header | `HD VN:1.6 SO:unknown` |

---

## CLI Command Map

```
# kinsim -- ML pipeline --------------------------------------------------
kinsim extract                -> kinsim/extract.py
kinsim merge                  -> kinsim/extract.py
kinsim refine                 -> kinsim/refine.py           (auto v3=GMM only / v4=GMM+p95)
kinsim train                  -> kinsim/train.py
kinsim generate               -> kinsim/generate.py
kinsim evaluate               -> kinsim/evaluate.py
kinsim verify-generate        -> kinsim/verify_generate.py  (ref vs gen BAM comparison)
kinsim analyze                -> kinsim/analyze.py
kinsim sample                 -> kinsim/sample.py
kinsim strip-kinetics         -> kinsim/strip_kinetics.py

# kinsim-prep -- data preparation ----------------------------------------
kinsim-prep parse             -> kinsim/utils/motifs.py        (unified motif parser)
kinsim-prep rebase            -> prep/rebase.py                (REBASE fetch + parse)
kinsim-prep merge-motifs      -> prep/motif_merge.py           (merge + filter + dedup)
kinsim-prep manifest          -> prep/manifest.py              (count/validate/list)
kinsim-prep filter            -> prep/filter.py                (filter .pkl)
kinsim-prep balance           -> prep/balance.py               (balance .pkl by mod type)
```

Typo suggestions via `difflib.get_close_matches` in both `__main__.py` files.
If a user types `kinsim prep`, `kinsim rebase`, etc., a helpful redirect message
points them to `kinsim-prep`.

---

## Import Path Rules

### Within `kinsim/` package

Top-level modules (`extract.py`, `train.py`, `generate.py`, etc.) import from sub-packages:

```python
# From kinsim/ top-level modules:
from .utils.encoding import K, KMER_MASK, BASE_MAP, METH_IDS
from .utils.motifs import parse_motifs, scan_sequence, build_reference_meth_map
from .utils.config import setup_logging, load_manifest, load_yaml_config
from .utils.io import load_reference, parse_maf, get_extended_context
from .data.dataset import log_transform, inv_log_transform, MLPSignalDataset
from .models.predictor import create_from_config
```

Files under `kinsim/models/` are **1 level** below `kinsim/`:

```python
# From kinsim/models/:
from ..utils.encoding import K, kmer_mask
from ..data.dataset import log_transform, inv_log_transform
```

### Within `prep/` package

Files under `prep/callers/` are **1 level** below `prep/`.
Use absolute imports for `kinsim` modules:

```python
# From prep/callers/:
from kinsim.utils.encoding import METH_IDS
from .base import BaseOutputParser
from .registry import register
```

Files directly under `prep/` use absolute imports for `kinsim`:

```python
# From prep/:
from kinsim.utils.encoding import METH_IDS
from kinsim.utils.motifs import load_motif_string
from kinsim.utils.config import setup_logging, load_manifest
```

### Cross-package lazy imports (in `kinsim/utils/motifs.py`)

`load_motif_string()` and `_build_meth_map_fuzznuc()` use lazy imports from `prep`
to avoid circular imports at module-load time. Both packages are always co-installed
(same `pyproject.toml`), so these imports always succeed at runtime:

```python
from prep.callers import create_parser      # lazy, inside function
from prep.callers import auto_detect_parser # lazy, inside function
from prep.rebase import parse_rebase_file   # lazy, inside function
from prep.rebase import write_fuzznuc_pattern_file  # lazy, inside function
```

**Never use bare `print()` for operational output.**
Always use `log = logging.getLogger(__name__)` and `log.info()`, `log.warning()`, `log.error()`.

---

## Methylation State IDs

```python
METH_IDS = {'none': 0, 'm6A': 1, 'm4C': 2, 'm5C': 3}
```

Consistent everywhere. Defined in `kinsim/utils/encoding.py`.

---

## Coding Conventions

### General
- Python 3.10+. Type hints on public function signatures in `models/`.
- No global mutable state. All functions take explicit arguments.
- `Path` objects for all file I/O (not `os.path.join`), except in SLURM scripts.
- `sys.exit(1)` on fatal errors; always print a message to `stderr` first.
- Never catch `Exception` broadly. Catch specific exceptions.

### Neural Models
- Xavier uniform init for `nn.Linear`, small normal (`std=0.02`) for `nn.Embedding`.
- Always write `model.eval()` before inference; `model.train()` at start of epoch.
- `@torch.no_grad()` on all inference functions (`sample`, `predict_mean`, `generate_signals_batch`).
- Use `ReduceLROnPlateau` — never `verbose=True` (removed in PyTorch >= 2.1). Log LR drops via `log.info(...)`.
- Wrap training loops in `try/finally` to ensure CSV and TensorBoard are closed on crash.
- Always save `model_config.json` **before** the first epoch, not after training.
- Always include `"scheduler"` key in checkpoint dict for resume support.
- Use `create_from_config()` factory to load models — never hardcode architecture.

### Logging
- Every module uses `log = logging.getLogger(__name__)`.
- Never use bare `print()` for operational output — use `log.info()`, `log.warning()`, `log.error()`.
- `setup_logging()` from `kinsim.utils.config` is called once in each CLI `main()`.
- Format: `"%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"` — timestamps for SLURM logs.

### Checkpoints
```python
torch.save({
    'epoch': epoch + 1,
    'step':  global_step,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),   # always include
}, path)
```

### Signal Space
- **Training**: always in log1p space (`log_transform` applied by `MLPSignalDataset`).
- **Inference**: model outputs in log1p space -> `inv_log_transform` -> uint8 [0, 255].
- **Storage (.pkl)**: raw values (not transformed) + stoichiometric fraction column.
  Format: `np.ndarray(N, 3)` with columns [IPD, PW, fraction]. Legacy 2-col supported.
- **Metadata**: every .pkl has a `"__meta__"` string key with provenance. Dataset classes skip it.

### SLURM Scripts
- Array jobs for per-species tasks (`#SBATCH --array=1-N`).
- Use `kinsim_extract.slurm` with manifest CSV for extract jobs.
- Auto-detect flat vs subdirectory layout in `generate.py` (not in SLURM scripts).
- All SLURM scripts include diagnostics: date, hostname, GPU info, timing, exit codes.

### Manifest CSV
- Manifest columns: `sample_id`, `bam_path`, `motifs` (CSV with header).
- Count rows for `--array`: `N=$(kinsim-prep manifest count manifest.csv)`.
- Output shard naming: `shards/<sample_id>_shard.pkl` (derived from manifest `sample_id`).

---

## What NOT To Do

- **Do not** store log-transformed data in `.pkl` files — raw values only.
- **Do not** use `verbose=True` in `ReduceLROnPlateau` — crashes on PyTorch >= 2.1.
- **Do not** hardcode architecture params in `generate.py` — always read from `model_config.json` via `create_from_config()`.
- **Do not** modify `motifs.py` for stoichiometric fraction handling — fractions are parsed at the storage level (`_build_fraction_lookup` in `extract.py` and `generate.py`).
- **Do not** add new callers/parsers outside of `prep/callers/` — use the `@register` decorator pattern.
- **Do not** add data preparation commands to `kinsim/__main__.py` — they belong in `prep/__main__.py`.
- **Do not** use relative imports (`from ..prep`) from `kinsim` to reach `prep` — use absolute `from prep.` imports (lazy, inside functions only).

---

## Adding a New Motif Parser

1. Create `prep/callers/<name>.py` with a `@register` class inheriting `BaseOutputParser`.
2. Define `name`, `supported_mods`, `parse()`, and `is_file_for_this_parser()`.
3. Import the new module in `prep/callers/__init__.py` to trigger registration.
4. The parser is immediately available via `create_parser("name")` and auto-detection.

---

## Key Numbers

| Constant | Value | Where |
|---|---|---|
| K (k-mer size) | 11 | `kinsim/utils/encoding.py` |
| Total possible k-mers | 4,194,304 (4^11) | `kinsim/utils/encoding.py` |
| Methylation states | 4 (none/m6A/m4C/m5C) | `kinsim/utils/encoding.py` |
| MID (flanking bases) | 5 | `kinsim/utils/io.py` |
| Reservoir cap (extract) | 10,000 per (kmer, meth) | `kinsim/extract.py` |
| Reservoir cap (merge) | 50,000 per (kmer, meth) | `kinsim/extract.py` |
| MLPSignalDataset cap (unmeth) | 20 per key | `kinsim/data/dataset.py` |
| MLPSignalDataset cap (meth) | 100 per key | `kinsim/data/dataset.py` |
| BAM signal range | [0, 255] uint8 | `kinsim/generate.py` |
| N-context default signal | 1 (not 0) | `kinsim/generate.py` |
| log_sigma clamp range | [-6, 3] | `kinsim/models/predictor.py` |
| Train/val split | 90% / 10% | `kinsim/train.py` |
| LR patience | 5 epochs | `kinsim/train.py` |
| LR factor | 0.5 | `kinsim/train.py` |
| meth_proj_dim | 8 | `kinsim/models/predictor.py` |
| ConvPredictor params | ~140K | `kinsim/models/predictor.py` |
| MLPPredictor params | ~268M | `kinsim/models/predictor.py` |

---

## Future Work / v4 Roadmap

### Complementary-strand methylation channel
**Problem:** for palindromic methylation sites (Type II R-M systems), both strands carry an m6A. The polymerase generating one strand's read physically contacts both strands of the duplex (~12 bp footprint), so the methylation on the OPPOSITE strand also affects IPD/PW. Currently only forward-strand methylation is encoded in the FiLM input — the model has no signal about the complementary-strand methylation at the same genomic position.

**Observation:** on bc2033 (HMB-10), the m6A profile shows a strong IPD spike at +8 in addition to the expected peak at +0. This is the kinetic footprint of the bilateral methylation: the reverse strand's m6A sits 8 bp downstream of the forward m6A on the same site, and its steric effect on the polymerase is what produces the +8 spike.

**Proposed addition:**
- Extract: store rev_meth[-1], rev_meth[0], rev_meth[+1] per sample (3 extra columns) — meth_id of the complementary strand at the prediction position and its immediate neighbours.
- Dataset / model: feed those into FiLM alongside the forward meth context.
- Generate: build two reference meth maps (forward, reverse) and look up both at each prediction position.

**Cost:** ~+3 columns in the .pkl, ~100 lines of code modified, retraining required.

**Benefit:** the model learns bilateral methylation patterns explicitly. Predictions on palindromic sites reproduce the double IPD peak seen in real data (e.g. bc2033's +0/+8 signature). Also captures hemimethylation asymmetries (relevant for cell-cycle studies).
