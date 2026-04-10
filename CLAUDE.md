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
│   ├── extract.py                  BAM extraction + shard merging; manifest + GFF mode
│   ├── train.py                    supervised training loop (ConvPredictor/MLPPredictor)
│   ├── generate.py                 BAM generation with trained model
│   ├── evaluate.py                 calibration report + per-kmer distribution plots
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
    ├── run_pipeline.sh             submit full pipeline with dependency chain
    ├── pbsim3_simulate.slurm       PBSIM3 read simulation
    │
    ├── 00_extract.slurm            extract (array job, manifest mode)
    ├── 01_train.slurm              train (1 GPU, 24h)
    ├── 02_generate.slurm           generate (array job)
    ├── 03a_validate_generate.slurm validation: generate
    ├── 03b_validate_align.slurm    validation: pbmm2 alignment
    ├── 03c_validate_ipdsummary.slurm validation: ipdSummary
    ├── 03d_validate_pbmotifmaker.slurm validation: pbmotifmaker
    ├── 04_evaluate.slurm           evaluate
    ├── 05_baselines.slurm          run all 3 baseline models
    │
    ├── config/                     example configuration files
    │   ├── config_example.yaml     training config example
    │   └── manifest_example.csv    manifest CSV example
    │
    └── msa1003/                    MSA1003 data extraction pipeline
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
    gff:       str = ""  # optional ipdSummary GFF3 path (enables GFF extraction mode)

load_manifest(manifest_path) -> list[SampleEntry]
validate_manifest(entries, check_files=True) -> list[str]
load_yaml_config(path) -> dict
setup_logging(verbose=False)
```

### `kinsim/utils/io.py`
File I/O for FASTA references, MAF alignments, GFF annotations, and PBSIM3 directory discovery.

```python
load_reference(fasta_path) -> dict[str, str]       # contig_name -> sequence
parse_maf(maf_path) -> iterator                     # MAF alignment records
get_extended_context(ref, pos, k) -> str            # extract 11-mer from reference
discover_pbsim3_layout(directory) -> list[dict]     # auto-detect flat vs subdirectory

# GFF-based methylation annotations (ipdSummary output)
load_gff_annotations(gff_path, min_score=20.0, min_ipd_ratio=0.0)
    -> dict[(contig, pos_0based, strand), meth_id]  # GFF3 → position lookup
build_read_meth_array(annotations, contig, ref_start, read_len, strand)
    -> np.ndarray(int8)                             # per-base meth_id for one read
```

### `kinsim/data/dataset.py`
Dataset class and signal transforms. Never import transforms from model files.

```python
log_transform(x: Tensor) -> Tensor      # log1p -- raw [0,255] -> training space
inv_log_transform(x: Tensor) -> Tensor  # expm1 clamped to [0,255] -- inference

class MLPSignalDataset(Dataset):   # flat-sample dataset with dynamic capping
    # Returns (kmer_id, meth_probs, log_signal, meth_id) tuples
    # Dynamic capping: unmeth <= 20, meth <= 100 samples per key
    # Stoichiometric soft labels from per-sample fraction column
```

### `kinsim/extract.py`
The data preparation pipeline. Supports two extraction modes:

1. **Motif-based** (original): scans sequence for motif patterns, labels by regex match
2. **GFF-based** (recommended): uses ipdSummary GFF3 annotations for methylation labels

```python
validate_bam_kinetics(bam_path, n_check=10)
extract_samples_from_bam(bam_path, motif_string, ...)    # motif-based (unaligned BAM)
extract_from_aligned_bam(bam_path, gff_path, ...)        # GFF-based (aligned BAM)
merge_shards(input_dir, output_file, max_samples_per_key=50000)
extract_from_manifest_task(manifest_path, task_index, output_dir, ...)
    # Auto-selects GFF or motif mode based on manifest gff column
```

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
Real PacBio BAMs
      |
      +---- Motif-based path ----+---- GFF-based path (recommended) ----+
      |                          |                                       |
      v                          v  (align BAM + run ipdSummary SP3-C3)  |
rebase_motifs.csv          aligned.bam + annotations.gff                 |
      |                          |                                       |
      v  (merge-motifs)          v                                       |
final_motifs.csv                 |                                       |
      |                          |                                       |
      +---- manifest.csv -------+                                       |
             [sample_id, bam_path, motifs, gff]                         |
      |                                                                  |
      v  kinsim extract --manifest manifest.csv --task $TASK --output-dir shards/
      v  (auto-selects GFF or motif mode per row)
strain1_shard.pkl  strain2_shard.pkl  ...
      |
      v  kinsim merge shards/ master_data.pkl
      |
      v  (optional) kinsim-prep filter master_data.pkl training_data.pkl --min-coverage 50
      v
training_data.pkl
      |
      v  kinsim train training_data.pkl checkpoints/
      v
checkpoint_epoch50.pt + model_config.json
      |
      v  kinsim generate <pbsim3_reads> <ref> <ckpt.pt> <motifs> <output.bam>
      v
species_mlp.bam
  flag=4  fi:B:C (IPD uint8)  fp:B:C (PW uint8)
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
kinsim extract                -> kinsim/extract.py  (--gff for GFF mode)
kinsim merge                  -> kinsim/extract.py
kinsim train                  -> kinsim/train.py
kinsim generate               -> kinsim/generate.py
kinsim evaluate               -> kinsim/evaluate.py
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
- Manifest columns: `sample_id`, `bam_path`, `motifs`, `gff` (CSV with header, commas OK in quoted fields).
- The `gff` column is optional. When present and non-empty, GFF-based extraction is used.
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
