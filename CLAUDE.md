# KinSim — Developer Reference for Claude

## Project Summary

KinSim simulates PacBio HiFi kinetic signals (IPD and PW) for metagenomic binning research.
Given PBSIM3-simulated reads and a reference genome, KinSim injects biologically realistic
per-base IPD/PW values into unaligned BAM files using a supervised MLP/Conv model that
predicts N(mu, sigma^2) per 11-mer context with methylation conditioning.

Output BAMs carry standard PacBio tags: `fi:B:C` (IPD) and `fp:B:C` (PW).

A single CLI is installed:
- **`kinsim`** — ML pipeline: extract, refine, train, generate, evaluate, analyze, verify-generate

Convenience offline tools live in `scripts/` (run via `python scripts/<name>.py`):
manifest count/validate/list, balance, filter, rebase fetch/parse, motif_merge.

## Sharded mode (preferred for ≥ 10 strains)

`merge` collapses N shards into a monolithic master.pkl whose RAM footprint
scales linearly with the corpus. For larger runs the entire pipeline now
supports a **sharded mode** that never holds the corpus in RAM:

```
extract  ──► <shards>/sample_id_shard.pkl       (one per strain, parallel via SLURM array)
refine   ──► <refined>/sample_id_shard_clean.pkl
            (auto-detected when input is a directory; pool-harvest IPDs across
             shards → fit GMMs once globally → apply per-shard, atomic write)
train    ──► reads <refined>/ via ShardedSignalDataset (PyTorch IterableDataset).
             Worker-aware shard partition; per-epoch shard + row shuffling.
analyze  ──► concatenates shards in-memory before stats (analyze is run once,
             can afford full-corpus RAM; refine + train are the hot paths)
```

Train/test splitting (sharded mode):

- `--test-strains bc2080,bc2081,bc2082` — explicit by-sample-id holdout. Those
  shards never enter training. Real generalisation metric.
- `--test-fraction 0.10` — random per-shard split, reproducible via `--split-seed`.

`merge` and the in-memory `MLPSignalDataset` are kept for small datasets.
Auto-detection on input path: directory → sharded path; file → in-memory path.
No CLI changes for the small-data case.

## Training Set

`extract` is a single-pass pipeline that captures **positions of
expected kinetic signature** in addition to methylation positions
themselves. Rationale: m6A/m4C/m5C signatures are observed at
configured downstream offsets (m6A at 0 and +5, m5C at +2 and +6, etc.,
declared in `kinsim_config.yaml`). Training only on methylation centers
would deprive the model of the slowing signal at those offsets.

```
  aligned BAM + ref + motifs --> kinsim extract  --> shards/<sample>_shard.pkl
                             --> kinsim refine   --> refined/<sample>_clean.pkl
                                                    (per-(meth, offset) GMM filter)
                             --> kinsim train    --> checkpoints/  (model_config.json
                                                                    carries p_fire)
```

**Three categories** (col 17 of the 20-col layout):
- `0` BASELINE  — far from any methylation; meth_context window is empty.
- `1` SLOWED    — at a signature offset of a methylation. Includes the
  methylation itself when `0 ∈ signature_offsets[T]` (m6A, m4C). For m5C
  the methylation centre is NEAR_METH, since 0 is not in `[2, 6]`.
- `2` NEAR_METH — close to a methylation (within `[+1, near_meth_max_dist]`)
  but NOT at a signature offset of it. Negative control: meth in mc but
  IPD should look baseline.

**Parent meth attribution** (col 36 PARENT_METH + col 37 PARENT_OFFSET):
extract knows which methylation produced each SLOWED/NEAR_METH row (it's
iterating "for each motif match of type T at position p"). Both columns
are written at emission time, so analyze can bucket rows per
`(category, parent meth)` cleanly with a vectorised mask — no need to
re-infer the parent from the mc[] window.

Precedence on overlapping motif claims (rare):
- SLOWED beats NEAR_METH (existing rule).
- Within SLOWED: last writer wins (deterministic given the YAML's
  meth-type iteration order).
- Within NEAR_METH: first writer wins (only assign where neither
  `slowed[c]` nor `near[c]` is set yet).

**Per-position emission rules in `extract`:**
For each motif-match position `p` of type `T`:
- For each `k ∈ signature_offsets[T]` (including 0): position `p+k` →
  CATEGORY_SLOWED.
- For each `k ∈ [0, near_meth_max_dist]` NOT in `signature_offsets[T]`:
  position `p+k` → CATEGORY_NEAR_METH (slowed wins on conflict).
Positions far from any methylation (distance ≥ `baseline_min_dist_to_meth`,
default = K = 11) → CATEGORY_BASELINE candidate, capped per kmer at
`n_baseline_per_kmer` via streaming reservoir sampling.

**Refine** has two methods, both keep BASELINE and NEAR_METH unchanged
and only filter CATEGORY_SLOWED rows:

- **`--method gmm` (default, `slowed_split_gmm`)** — for each meth type
  T present in the slowed rows, fit a 2-component GaussianMixture on
  the combined `baseline + slowed_by_T` IPD pool (baseline subsampled
  to match `slowed_by_T` count). Validate that ≥ 85 % of the baseline
  subsample lands in the lower-mean component; if so, drop slowed_by_T
  rows whose posterior in that component exceeds 0.5. If validation
  fails or the type has < `min_samples_for_gmm` slowed rows, keep all
  rows of that type. Per-type GMM params (means, sigmas, weights, lower
  index, baseline-in-lower fraction) are recorded in
  `__meta__["stats"]["per_type"]`.
- **`--method p95` (legacy, `slowed_split`)** — single global threshold
  = `secondary_percentile`-th percentile of the per-kmer baseline-mean
  distribution. Same threshold for every meth type. Kept as a fallback
  / comparison method.

The GMM is the default because the p95 threshold is unfair to
low-baseline kmers and to weak-signal meth types (m4C with
fraction 0.5, m5C with sig=[2,6]). Per-type GMM lets each type's
boundary be chosen by data, and the validation step rejects fits where
the baseline doesn't cluster cleanly (defensive).

---

## Repository Layout

```
KinSim/
├── pyproject.toml                  single entry point: kinsim
│
├── kinsim/                         ML pipeline package (v0.4.0)
│   ├── __init__.py
│   ├── __main__.py                 CLI router
│   │
│   ├── extract.py                  BAM extraction + shard merging (motif-based, manifest-driven)
│   ├── refine.py                   slowed_split: per-kmer baseline mean p95 filter
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
│       ├── encoding.py             11-mer bit-packing + get_meth_ids() (YAML-driven)
│       ├── sample_layout.py        20-col row layout constants (COL_*, CATEGORY_*)
│       ├── motifs.py               IUPAC motif parsing, sequence scanning, meth maps
│       ├── config.py               manifest CSV loader, YAML config, logging setup
│       ├── io.py                   FASTA loading, MAF parsing, PBSIM3 discovery
│       └── parsers/                methylation caller output parsers (plugin registry)
│           ├── __init__.py         exports: BaseOutputParser, create_parser, list_parsers, auto_detect_parser
│           ├── base.py             BaseOutputParser ABC
│           ├── registry.py         @register decorator, factory functions
│           ├── pacbio.py           PacBioParser -- motifs.csv with variable columns
│           ├── modkit.py           ModkitParser -- modkit pileup --bedMethyl TSV (per-site, requires post motif discovery)
│           ├── combined.py         CombinedParser -- mod_type,motif,offset,frac_mod,n_sites,source
│           ├── rebase.py           REBASE web fetch + file parsing + fuzznuc patterns
│           └── motif_merge.py      merge/filter/dedup motifs -> standard PacBio CSV
│
├── scripts/                        offline utilities (run with python scripts/X.py)
│   ├── manifest.py                 manifest CSV CLI (count / validate / list)
│   ├── balance.py                  balance .pkl by methylation type
│   └── filter.py                   filter .pkl by coverage, mod type, max keys
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
    │   ├── 00_extract.slurm        kinsim extract per manifest row (array job)
    │   ├── 02_refine.slurm         kinsim refine shards/ → refined/
    │   ├── 03_train.slurm          kinsim train (1 GPU)
    │   ├── 04_generate.slurm       kinsim generate on PBSIM3 reads (array)
    │   ├── 05_evaluate.slurm       kinsim evaluate
    │   ├── 06_verify_generate.slurm   kinsim verify-generate (array)
    │   └── run.sh                  orchestrator — extract/refine/train/evaluate chain
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

`load_motif_string()` uses lazy imports from `kinsim.utils.parsers` for
file-based motif loading. Lazy because the parser module path was moved
from the dissolved `prep/` package — the lazy form keeps the import
graph DAG-safe.

`build_reference_meth_map()` uses EMBOSS fuzznuc as primary backend for genome-wide
motif scanning, with automatic fallback to Python regex if fuzznuc is not installed
or returns empty results (fuzznuc can silently fail on some IUPAC patterns).

### `kinsim/utils/config.py`
Manifest CSV parsing, YAML config loading, and logging setup.

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

refine:
  slowed_split:
    secondary_percentile: 95   # p95 of per-kmer baseline mean

extract:
  n_baseline_per_kmer:        50   # per-kmer baseline cap (reservoir)
  baseline_min_dist_to_meth:  11   # >= K so meth_context never overlaps a meth
  baseline_sample_rate:       0.10 # front-end skip rate before reservoir
  near_meth_max_dist:         7    # 1..K-1; positions p+k for k in this range
                                   # are NEAR_METH unless k in signature_offsets[T]
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
Per-sample column layout, category enum, slice helpers. Pure Python
(no pysam) so refine/dataset/tests can import it without the
extract/BAM dependency.

```python
SAMPLE_NCOLS = 20
COL_IPD            = 0
COL_PW             = 1
COL_FRACTION       = 2
COL_METH_CTX_*     = 3..14    # mc_0..mc_10 (offsets [-7..+3])
COL_REV_METH       = 14..17   # complementary-strand meth at offsets [-1, 0, +1]
COL_CATEGORY       = 17
COL_PARENT_METH    = 18       # meth_id of the parent meth (0 for baseline)
COL_PARENT_OFFSET  = 19       # row_pos − parent_meth_pos (small int, [0, 7])

CATEGORY_BASELINE  = 0
CATEGORY_SLOWED    = 1   # at a signature offset (incl. meth itself if 0 ∈ sig)
CATEGORY_NEAR_METH = 2   # close to meth but not at signature offset

get_categories(arr) -> int8 ndarray  # arr[:, COL_CATEGORY]

slice_meth_context(meth_status, center) -> list[11]
slice_rev_meth(meth_status_complement, center) -> list[3]
slice_kinetic_profile(ipds, pws, center) -> list[18]
```

**`PARENT_METH` (col 36)** is written by extract at emission time —
analyze reads it directly to bucket rows per (category, parent meth)
without inferring from the meth_context window. This both fixes the
overlapping-motif ambiguity of mc-based attribution and lets analyze
vectorise per-bucket accumulators (~75 min → seconds on master pkls).

### `kinsim/data/dataset.py`
Dataset class and signal transforms. Never import transforms from model files.

```python
log_transform(x: Tensor) -> Tensor      # log1p — raw [0,255] -> training space
inv_log_transform(x: Tensor) -> Tensor  # expm1 clamped to [0,255] — inference

class MLPSignalDataset(Dataset):
    # Iterates int-keyed kmers, derives meth_id at centre from
    # mc[KMER_PRED_IDX], pre-flattens all rows into contiguous arrays
    # so every sample is seen once per epoch.
    # Returns (kmer_id, meth_full, log_signal, meth_id) tuples.
```

### `kinsim/extract.py`
The data preparation pipeline. **Motif-based extraction only**: the
model learns kinetic signatures from the per-position methylation
context fed to FiLM, so no aligned-BAM path is needed.

```python
validate_bam_kinetics(bam_path, n_check=10)
    # Returns "fi" (unaligned) or "ip" (aligned) — used to route the
    # forward extraction; reverse-strand pass gated by read.has_tag("ri").

# Single-pass extract emits dict[kmer_id (int)] -> ndarray(N, 20).
# Per row: IPD, PW, fraction, mc_0..10, rev_meth_-1/0/+1, CATEGORY,
# PARENT_METH, PARENT_OFFSET (see kinsim.utils.sample_layout).
extract_samples_from_bam(bam_path, motif_string,
                         n_baseline_per_kmer=50,
                         baseline_min_dist_to_meth=K,
                         baseline_sample_rate=0.10,
                         near_meth_max_dist=7, ...)
extract_from_manifest_task(manifest_path, task_index, output_dir, ...)

merge_shards(input_dir, output_file, max_samples_per_key=50000)
    # Concatenates per-sample shards (matching int-keyed kmer dicts).
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

### `kinsim/utils/parsers/` — Methylation Caller Parsing Library

#### `kinsim/utils/parsers/`
Read-only parsing library for methylation caller output files. Plugin registry
with `@register` decorator — adding a new format = one file + `@register` class.

```python
from kinsim.utils.parsers import create_parser, list_parsers, auto_detect_parser

# Explicit parser
parser = create_parser("pacbio")       # or "modkit", "combined"
motif_string = parser.parse("motifs.csv", min_fraction=0.40, min_detected=20)

# Auto-detect from file content
parser = auto_detect_parser("output.csv")

# List registered parsers
list_parsers()  # ['combined', 'modkit', 'pacbio']
```

**BaseOutputParser ABC** (`base.py`):
- `name: ClassVar[str]` — registry key
- `supported_mods: ClassVar[list[str]]` — mod types this format carries
- `parse(filepath, min_fraction, min_detected) -> str` — file -> motif string
- `is_file_for_this_parser(filepath) -> bool` — heuristic for auto-detection

**PacBioParser** (`pacbio.py`): Handles motifs.csv with variable columns.
Required: `motifString`, `centerPos`. Optional: `modificationType`, `fraction`, `nDetected`.

**ModkitParser** (`modkit.py`): Handles modkit pileup `--bedMethyl` TSV (11+ columns).
**Note**: emits per-site pseudo-motifs (`chrom:start:strand`), NOT real motif
patterns. End users feeding raw modkit output need a separate motif-discovery
step (or use the upstream PacBio motifmaker output if available).

**CombinedParser** (`combined.py`): Handles combined methylation CSV with columns:
`mod_type,motif,offset,frac_mod,n_sites,source`. Auto-detected when CSV header
contains both `mod_type` and `frac_mod`.

**Integration**: `load_motif_string()` in `kinsim/utils/motifs.py` accepts optional
`parser_name` kwarg. When provided, bypasses auto-detection and uses the named parser.

#### `kinsim/utils/parsers/rebase.py`
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

CLI (run via `python -m kinsim.utils.parsers.rebase ...`):
- `fetch <org_num> --output <csv>` — fetch from REBASE website, write CSV
- `parse <file>` — parse local REBASE file, print motif string
- `patterns <motifs> <outfile>` — write fuzznuc pattern file

#### `kinsim/utils/parsers/motif_merge.py`
Merges, filters, and deduplicates motifs from calling-derived CSV and REBASE
into a single standard PacBio `motifs.csv`.

CLI: `python -m kinsim.utils.parsers.motif_merge species_motifs.csv rebase_motifs.csv --output final_motifs.csv`

#### `scripts/filter.py`
Filter .pkl by coverage, mod type, or max keys.

CLI: `python scripts/filter.py general.pkl training.pkl --min-coverage 50 --mod-type m6A,m5C`

#### `scripts/manifest.py`
Manifest CSV inspection utilities.

CLI:
```
python scripts/manifest.py count <csv>       # prints integer for SLURM --array
python scripts/manifest.py validate <csv>    # checks duplicates, file existence
python scripts/manifest.py list <csv>        # tabular display
```

---

## Data Flow

```
Aligned bystrandified BAMs (sorted, indexed, ip/pw tags)
      |
      +-- Motif discovery (jasmine + ipdSummary + pbmotifmaker, threshold 0.7)
      |   produces a merged motifs.csv per sample
      |
      v  manifest.csv  [sample_id, bam_path, motifs, ref_path]
      |
      v  kinsim extract --manifest manifest.csv --task $TASK --output-dir shards/
strain1_shard.pkl  strain2_shard.pkl  ...
      |   (38 cols, key=kmer_id only, CATEGORY at col 35,
      |    PARENT_METH at 36, PARENT_OFFSET at 37)
      |   Each row tagged 0=baseline / 1=slowed / 2=near_meth
      |
      v  kinsim refine shards/ refined/
      |   Per-(meth, offset) GMM filter on slowed rows;
      |   baseline + near_meth pass through unchanged;
      |   p_fire = n_kept / n_in stored per bucket in __meta__.
refined/<sample>_clean.pkl
      |
      v  kinsim train refined/ checkpoints/
      v
checkpoint_epoch50.pt + model_config.json (carries p_fire)
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
# kinsim -- ML pipeline (7 verbs) ----------------------------------------
kinsim extract                -> kinsim/extract.py            (aligned BAM → shard.pkl)
kinsim refine                 -> kinsim/refine.py             (per-(meth, offset) GMM)
kinsim train                  -> kinsim/train.py
kinsim generate               -> kinsim/generate.py
kinsim evaluate               -> kinsim/evaluate.py
kinsim verify-generate        -> kinsim/verify_generate.py    (per-(kmer, meth) ref vs gen)
kinsim analyze                -> kinsim/analyze.py

# Offline tooling (run with python, not via the CLI) --------------------
python scripts/manifest.py        — count / validate / list manifest CSV
python scripts/balance.py         — balance .pkl by mod type
python scripts/filter.py          — filter .pkl by coverage / mod type / max keys
python scripts/sample.py          — subsample a shard pkl
python scripts/strip_kinetics.py  — strip fi/fp/ri/rp from a BAM
python scripts/compare.py         — cross-dataset kinetic comparison
python scripts/inspect_null_model.py — inspect an ipdSummary .npz.gz null model

python -m kinsim.utils.parsers.rebase       — REBASE fetch / parse
python -m kinsim.utils.parsers.motif_merge  — merge + filter + dedup motifs
```

Typo suggestions via `difflib.get_close_matches` in `kinsim/__main__.py`.
The `kinsim-prep` console script was removed during the prep/ dissolution;
its functionality is in `scripts/` and `kinsim/utils/parsers/`.

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

### Within `kinsim/utils/parsers/` package

Use relative imports for sibling modules, and absolute imports for the
rest of `kinsim`:

```python
# From kinsim/utils/parsers/<name>.py:
from kinsim.utils.encoding import METH_IDS, get_meth_ids
from .base import BaseOutputParser
from .registry import register
```

`scripts/` files are run as standalone python scripts, so they import
`kinsim` absolutely:

```python
# From scripts/<name>.py:
from kinsim.utils.encoding import METH_IDS
from kinsim.utils.motifs import load_motif_string
from kinsim.utils.config import setup_logging, load_manifest
```

### Lazy imports in `kinsim/utils/motifs.py`

`load_motif_string()` and `_build_meth_map_fuzznuc()` use lazy imports from
`kinsim.utils.parsers` to keep the import graph DAG-safe (the parsers
module can transitively touch `motifs` for IUPAC helpers):

```python
from kinsim.utils.parsers import create_parser      # lazy, inside function
from kinsim.utils.parsers import auto_detect_parser # lazy, inside function
from kinsim.utils.parsers.rebase import parse_rebase_file   # lazy
from kinsim.utils.parsers.rebase import write_fuzznuc_pattern_file  # lazy
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
- **Inference**: model outputs in log1p space → `inv_log_transform` → uint8 [0, 255].
- **Storage (.pkl)**: raw values (not transformed). 36-col layout from
  `kinsim.utils.sample_layout`; see col reference there.
- **Metadata**: every .pkl has a `"__meta__"` string key with provenance.
  Dataset classes skip it.

### SLURM Scripts
- Array jobs for per-species tasks (`#SBATCH --array=1-N`).
- Use `kinsim_extract.slurm` with manifest CSV for extract jobs.
- Auto-detect flat vs subdirectory layout in `generate.py` (not in SLURM scripts).
- All SLURM scripts include diagnostics: date, hostname, GPU info, timing, exit codes.

### Manifest CSV
- Manifest columns: `sample_id`, `bam_path`, `motifs` (CSV with header).
- Count rows for `--array`: `N=$(python scripts/manifest.py count manifest.csv)`.
- Output shard naming: `shards/<sample_id>_shard.pkl` (derived from manifest `sample_id`).

---

## What NOT To Do

- **Do not** store log-transformed data in `.pkl` files — raw values only.
- **Do not** use `verbose=True` in `ReduceLROnPlateau` — crashes on PyTorch >= 2.1.
- **Do not** hardcode architecture params in `generate.py` — always read from `model_config.json` via `create_from_config()`.
- **Do not** modify `motifs.py` for stoichiometric fraction handling — fractions are parsed at the storage level (`_build_fraction_lookup` in `extract.py` and `generate.py`).
- **Do not** add new callers/parsers outside of `kinsim/utils/parsers/` — use the `@register` decorator pattern.
- **Do not** add data preparation logic to `kinsim/__main__.py` — convenience CLIs live in `scripts/` (run via `python scripts/<name>.py`).
- **Do not** import parsers eagerly at module top — use lazy imports inside the function that needs them (the parser package transitively touches `kinsim.utils.motifs`).

---

## Adding a New Motif Parser

1. Create `kinsim/utils/parsers/<name>.py` with a `@register` class inheriting `BaseOutputParser`.
2. Define `name`, `supported_mods`, `parse()`, and `is_file_for_this_parser()`.
3. Import the new module in `kinsim/utils/parsers/__init__.py` to trigger registration.
4. The parser is immediately available via `create_parser("name")` and auto-detection.

---

## Key Numbers

| Constant | Value | Where |
|---|---|---|
| K (k-mer size) | 11 | `kinsim/utils/encoding.py` |
| Total possible k-mers | 4,194,304 (4^11) | `kinsim/utils/encoding.py` |
| Methylation states | 4 (none/m6A/m4C/m5C) | `kinsim/utils/encoding.py` |
| MID (flanking bases) | 5 | `kinsim/utils/io.py` |
| Reservoir cap (extract baseline) | 50 per kmer | `kinsim/extract.py` |
| Reservoir cap (merge) | 50,000 per kmer | `kinsim/extract.py` |
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

## Future Work

### Per-occupancy p_fire curve
Today's decomposition uses ``p_fire(target site) = target_frac × p_efficiency``,
which is linear in occupancy. For weak-signal types where the relationship
isn't linear, stratify per-bucket survival by ``frac`` bin (e.g. 0–0.3,
0.3–0.6, 0.6–1.0) and store a curve. Generate looks up by target frac.
Worth doing only if cross-strain verify shows occupancy mismatch.

### Wider rev_meth window for distal palindromes
Today's rev_meth captures only [-1, 0, +1] active-site neighbours.
8+ bp palindromic motifs (some Type II R-M) can place the partner meth
at ±3 to ±5. Extending the window is a layout change + retrain — wait
for evidence the model fails on those motif classes before doing this.
