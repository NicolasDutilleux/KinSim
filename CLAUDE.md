# KinSim — Developer Reference for Claude

## Project Summary

KinSim simulates PacBio HiFi kinetic signals (IPD and PW) for metagenomic binning research.
Given PBSIM3-simulated reads and a reference genome, KinSim injects biologically realistic
per-base IPD/PW values into unaligned BAM files using one of three modes:

- **dictionary** — Gaussian sampling from per-k-mer accumulators (fast, no GPU)
- **mlp**        — Supervised MLP/Conv predicting N(mu, sigma^2) per context (Level 1 AI)
- **cgan**       — Conditional WGAN-GP (Level 2 AI, captures non-Gaussian distributions)

All three modes output BAMs with standard PacBio tags: `fi:B:C` (IPD) and `fp:B:C` (PW).

Two CLI tools are installed from the same repository:
- **`kinsim`**      — ML pipeline: extract, merge, train, generate, evaluate, analyze
- **`kinsim-prep`** — Data preparation: rebase, merge-motifs, manifest, filter, prepare, parse

---

## Repository Layout

```
KinSim/
├── pyproject.toml                  entry points: kinsim + kinsim-prep
├── requirements.txt
│
├── kinsim/                         ML pipeline package
│   ├── __init__.py
│   ├── __main__.py                 CLI router (v0.3.0) — core commands only
│   │
│   ├── encoding.py                 11-mer bit-packing (no dependencies)
│   ├── motifs.py                   IUPAC motif parsing, sequence scanning, meth maps
│   ├── config.py                   manifest CSV loader, YAML config, logging setup
│   │
│   ├── common/                     shared data pipeline (used by MLP and cGAN)
│   │   ├── __init__.py
│   │   ├── dataset.py              log_transform, inv_log_transform, KmerSignalDataset, MLPSignalDataset
│   │   └── extract.py              BAM extraction + shard merging; manifest mode; fail-fast validation
│   │
│   ├── dictionary/                 dictionary mode (no neural network)
│   │   ├── __init__.py
│   │   ├── train.py                builds accumulator .pkl from real PacBio BAMs
│   │   ├── inject.py               injects signals using dictionary lookup
│   │   └── analyze.py              coverage stats, HTML/text reports
│   │
│   └── models/                     all neural model implementations
│       ├── __init__.py
│       ├── cgan/
│       │   ├── __init__.py
│       │   ├── model.py            Generator + Discriminator (WGAN-GP)
│       │   ├── train.py            WGAN-GP training loop
│       │   ├── generate.py         BAM generation with trained Generator
│       │   └── parse_train.py      shim -> delegates to common/extract.py
│       └── mlp/
│           ├── __init__.py
│           ├── model.py            ConvPredictor + MLPPredictor + create_from_config()
│           ├── train.py            supervised training loop (--architecture conv|mlp)
│           ├── generate.py         BAM generation with trained model
│           └── evaluate.py         calibration report + per-kmer distribution plots
│
├── prep/                           data preparation package (kinsim-prep CLI)
│   ├── __init__.py
│   ├── __main__.py                 CLI router for kinsim-prep
│   │
│   ├── rebase.py                   REBASE web fetch + file parsing + fuzznuc patterns
│   ├── motif_merge.py              merge/filter/dedup motifs -> standard PacBio CSV
│   ├── manifest.py                 manifest CSV CLI (count / validate / list)
│   ├── prepare.py                  legacy BAM/motif pair validation (alternating-line format)
│   ├── filter.py                   General Dictionary -> Training Dictionary filtering
│   └── callers/                    methylation caller output parsers (plugin registry)
│       ├── __init__.py             exports: BaseOutputParser, create_parser, list_parsers, auto_detect_parser
│       ├── base.py                 BaseOutputParser ABC
│       ├── registry.py             @register decorator, factory functions
│       ├── pacbio.py               PacBioParser -- motifs.csv with variable columns
│       ├── modkit.py               ModkitParser -- modkit pileup --bedMethyl TSV
│       ├── ipd_summary.py          IpdSummaryParser -- ipdSummary CSV/GFF3
│       └── combined.py             CombinedParser -- mod_type,motif,offset,frac_mod,n_sites,source
│
└── slurm_kinsim/                   HPC SLURM job scripts
    ├── kinsim_pipeline.sh          master script (mode: dictionary | cgan | mlp)
    ├── kinsim_extract.slurm        extract (array job, manifest mode)
    ├── kinsim_mlp_train.slurm      mlp train (1 GPU, 24h)
    ├── kinsim_mlp_generate.slurm   mlp generate (array job)
    ├── kinsim_mlp_evaluate.slurm   mlp evaluate
    ├── kinsim_cgan_train.slurm     cgan train (1 GPU, 24h)
    ├── kinsim_cgan_generate.slurm  cgan generate (array job)
    ├── prep_MSA1003_rebase.sh      fetch REBASE motifs for MSA1003 species
    ├── prep_MSA1003_merge.sh       merge calling + REBASE motifs, build manifest
    └── ...                         dictionary and other scripts
```

---

## Key Files — What Each Does

### `kinsim/encoding.py`
Pure functions, no external dependencies. The foundation of all k-mer logic.

```python
K = 11                          # window size (fixed everywhere)
KMER_MASK = (1 << 22) - 1      # 22-bit mask for sliding window
BASE_MAP  = {'A':0,'C':1,'G':2,'T':3}
METH_IDS  = {'none':0,'m6A':1,'m4C':2,'m5C':3}

encode_kmer(seq: str) -> int    # 11-char string -> 22-bit integer
decode_kmer(val: int) -> str    # 22-bit integer -> 11-char string
get_ipd_stats(acc) -> (mean, std)  # extract IPD mean/std from accumulator
get_pw_stats(acc)  -> (mean, std)  # extract PW  mean/std from accumulator
```

Accumulator format: `np.array([n, sum_ipd, sum_ipd2, sum_pw, sum_pw2])` (Welford-style).

### `kinsim/motifs.py`
Everything methylation-motif related. Used by all three modes.

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

**Integration**: `load_motif_string()` in `kinsim/motifs.py` accepts optional `parser_name` kwarg.
When provided, bypasses auto-detection and uses the named parser from the registry.

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

```python
motif_contains(longer, offset_longer, shorter, offset_shorter) -> bool
deduplicate_motifs(entries: list[dict]) -> list[dict]
write_pacbio_motifs_csv(entries: list[dict], filepath: str)
merge_motifs(input_files, output_path, *, min_frac=0.8, min_sites=300,
             deduplicate=True) -> dict
```

Input formats (auto-detected per file):
- Combined CSV: `mod_type,motif,offset,frac_mod,n_sites,source`
- PacBio CSV: `motifString,centerPos,modificationType,fraction,...`

CLI: `kinsim-prep merge-motifs species_motifs.csv rebase_motifs.csv --output final_motifs.csv`

#### `prep/filter.py`
Two-dictionary architecture: General Dictionary -> Training Dictionary.

```python
filter_pkl(input_path, output_path, *, min_coverage=0, mod_types=None, max_keys=0) -> dict
    # Filters .pkl by: min samples per key, mod type, max total keys
    # Returns stats: {keys_in, keys_out, samples_in, samples_out}
```

CLI: `kinsim-prep filter general.pkl training.pkl --min-coverage 50 --mod-type m6A,m5C`

#### `prep/manifest.py`
Manifest CSV inspection utilities.

CLI:
```
kinsim-prep manifest count <csv>       # prints integer for SLURM --array
kinsim-prep manifest validate <csv>    # checks duplicates, file existence
kinsim-prep manifest list <csv>        # tabular display
```

#### `prep/prepare.py`
Legacy BAM + motif-source pair validation (alternating-line text format).

### `kinsim/config.py`
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
setup_logging(verbose=False)
```

### `kinsim/common/dataset.py`
The shared data contract between cGAN and MLP. Never import transforms from model files.

```python
log_transform(x: Tensor) -> Tensor      # log1p -- raw [0,255] -> training space
inv_log_transform(x: Tensor) -> Tensor  # expm1 clamped to [0,255] -- inference

class KmerSignalDataset(Dataset):  # cGAN: (kmer_id, meth_id, signal)
class MLPSignalDataset(Dataset):   # MLP: random-shot per key, stoichiometric meth vector
```

### `kinsim/common/extract.py`
The data preparation pipeline shared by all neural modes.

```python
validate_bam_kinetics(bam_path, n_check=10)
extract_samples_from_bam(bam_path, motif_string, max_samples_per_key=10000)
merge_shards(input_dir, output_file, max_samples_per_key=50000)
extract_from_manifest_task(manifest_path, task_index, output_dir, ...)
```

### `kinsim/models/mlp/model.py`
Dual architecture with factory pattern:

```python
class ConvPredictor(nn.Module):    # NEW DEFAULT (~140K params)
    # Per-base embedding (4x16) + positional embedding (11x16)
    # FiLM conditioning: methylation modulates base embeddings
    # Conv1D backbone (3 layers, k=3, BatchNorm, GELU)
    # Dual readout: center position + global average pool
    # -> [mu_ipd, mu_pw, log_sigma_ipd, log_sigma_pw]

class MLPPredictor(nn.Module):     # LEGACY (~268M params)
    # Flat 4.2M-row kmer embedding + 2-layer MLP

def create_from_config(config: dict) -> nn.Module:
    # Reads "architecture" key ("conv" or "mlp"), defaults to "mlp"
```

Both share interface: `forward(kmer_ids, meth_probs)`, `sample()`, `predict_mean()`, `get_config()`.

### `kinsim/models/mlp/train.py`
Supervised training with Gaussian NLL loss.

```python
Loss = 0.5 * (2*log_sigma + (target - mu)^2 / sigma^2)
```

CLI: `kinsim train --model mlp --architecture conv --config config_mlp.yaml`

### `kinsim/models/mlp/generate.py`
BAM generation with trained MLP/Conv model. Uses `create_from_config()` for loading.
Output files: `<species>_mlp.bam`.

### `kinsim/models/mlp/evaluate.py`
Calibration report + per-kmer distribution plots.
CLI: `kinsim evaluate --model mlp <ckpt_dir> <pkl>`

---

## Data Flow

```
Real PacBio BAMs
      |
      v  (once per species: kinsim-prep rebase fetch <org_num>)
rebase_motifs.csv
      |
      v  (kinsim-prep merge-motifs calling.csv rebase.csv --output final_motifs.csv)
final_motifs.csv
      |
      v  (kinsim-prep manifest count manifest.csv  ->  N for SLURM --array)
manifest.csv  [sample_id, bam_path, motifs]
      |
      v  kinsim extract --manifest manifest.csv --task $TASK --output-dir shards/
      v
strain1_shard.pkl  strain2_shard.pkl  ...
      |
      v  kinsim merge shards/ general_data.pkl     <- General Dictionary
      |
      v  kinsim-prep filter general_data.pkl training_data.pkl --min-coverage 50
      v                                             <- Training Dictionary
training_data.pkl
      |
      +-->  kinsim train --model mlp   -> checkpoint.pt + model_config.json
      +-->  kinsim train --model cgan  -> checkpoint.pt + model_config.json

PBSIM3 reads + reference + checkpoint.pt
      |
      v  kinsim generate --model mlp  / cgan
      v
species_mlp.bam / species_cgan.bam
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
kinsim extract                -> kinsim/common/extract.py
kinsim merge                  -> kinsim/common/extract.py
kinsim train --model <m>      -> kinsim/models/<m>/train.py  or  dictionary/train.py
kinsim generate --model <m>   -> kinsim/models/<m>/generate.py  or  dictionary/inject.py
kinsim evaluate --model mlp   -> kinsim/models/mlp/evaluate.py
kinsim analyze                -> kinsim/dictionary/analyze.py

# kinsim-prep -- data preparation ----------------------------------------
kinsim-prep parse             -> kinsim/motifs.py              (unified motif parser)
kinsim-prep rebase            -> prep/rebase.py                (REBASE fetch + parse)
kinsim-prep merge-motifs      -> prep/motif_merge.py           (merge + filter + dedup)
kinsim-prep manifest          -> prep/manifest.py              (count/validate/list)
kinsim-prep prepare           -> prep/prepare.py               (legacy BAM/motif pairs)
kinsim-prep filter            -> prep/filter.py                (General -> Training .pkl)
```

Typo suggestions via `difflib.get_close_matches` in both `__main__.py` files.
If a user types `kinsim prep`, `kinsim rebase`, etc., a helpful redirect message
points them to `kinsim-prep`.

---

## Import Path Rules

### Within `kinsim/` package

Files under `kinsim/models/cgan/` and `kinsim/models/mlp/` are **2 levels** below `kinsim/`.
Use 3 dots to reach `kinsim/`-level modules:

```python
# From kinsim/models/cgan/ or kinsim/models/mlp/:
from ...encoding import BASE_MAP, K, KMER_MASK
from ...motifs import parse_motifs, scan_sequence, build_reference_meth_map
from ...common.dataset import log_transform, inv_log_transform, KmerSignalDataset
from ...common.extract import extract_samples_from_bam, merge_shards
from ...dictionary.inject import load_reference, parse_maf, get_extended_context, MID
from ...config import setup_logging, load_manifest, load_yaml_config
from .model import ...      # within the same sub-package (1 dot)
```

### Within `prep/` package

Files under `prep/callers/` are **1 level** below `prep/`.
Use absolute imports for `kinsim` modules:

```python
# From prep/callers/:
from kinsim.encoding import METH_IDS
from .base import BaseOutputParser
from .registry import register
```

Files directly under `prep/` use absolute imports for `kinsim`:

```python
# From prep/:
from kinsim.encoding import METH_IDS
from kinsim.motifs import load_motif_string
from kinsim.config import setup_logging, load_manifest
```

### Cross-package lazy imports (in `kinsim/motifs.py`)

`load_motif_string()` and `_build_meth_map_fuzznuc()` use lazy imports from `prep`
to avoid circular imports at module-load time. Both packages are always co-installed
(same `pyproject.toml`), so these imports always succeed at runtime:

```python
from prep.callers import create_parser      # lazy, inside function
from prep.callers import auto_detect_parser # lazy, inside function
from prep.rebase import parse_rebase_file   # lazy, inside function
from prep.rebase import write_fuzznuc_pattern_file  # lazy, inside function
```

**Never import `log_transform`/`inv_log_transform` from `cgan/model.py`.**
Always import from `common/dataset.py`.

**Never use bare `print()` for operational output.**
Always use `log = logging.getLogger(__name__)` and `log.info()`, `log.warning()`, `log.error()`.

---

## Methylation State IDs

```python
METH_IDS = {'none': 0, 'm6A': 1, 'm4C': 2, 'm5C': 3}
```

Consistent across all modes. Defined in `kinsim/encoding.py`, used everywhere.

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
- `setup_logging()` from `kinsim.config` is called once in each CLI `main()`.
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
- **Training**: always in log1p space (`log_transform` applied by `KmerSignalDataset`).
- **Inference**: model outputs in log1p space -> `inv_log_transform` -> uint8 [0, 255].
- **Storage (.pkl)**: raw values (not transformed) + stoichiometric fraction column.
  Format: `np.ndarray(N, 3)` with columns [IPD, PW, fraction]. Legacy 2-col supported.
- **Metadata**: every .pkl has a `"__meta__"` string key with provenance. Dataset classes skip it.

### SLURM Scripts
- Array jobs for per-species tasks (`#SBATCH --array=1-N`).
- Use `kinsim_extract.slurm` with manifest CSV for extract jobs.
- Auto-detect flat vs subdirectory layout in `generate.py` (not in SLURM scripts).
- `kinsim_pipeline.sh` is the single source of truth for the full pipeline order.
- All SLURM scripts include diagnostics: date, hostname, GPU info, timing, exit codes.

### Manifest CSV
- Manifest columns: `sample_id`, `bam_path`, `motifs` (CSV with header, commas OK in quoted fields).
- Count rows for `--array`: `N=$(kinsim-prep manifest count manifest.csv)`.
- Output shard naming: `shards/<sample_id>_shard.pkl` (derived from manifest `sample_id`).

---

## What NOT To Do

- **Do not** add logic to `models/cgan/parse_train.py` — it is a shim only.
- **Do not** store log-transformed data in `.pkl` files — raw values only.
- **Do not** import `KmerSignalDataset` or transforms from `cgan/train.py` — use `common/dataset.py`.
- **Do not** use `verbose=True` in `ReduceLROnPlateau` — crashes on PyTorch >= 2.1.
- **Do not** add a new mode without adding a corresponding entry to `kinsim/__main__.py` and `kinsim_pipeline.sh`.
- **Do not** hardcode architecture params in `generate.py` — always read from `model_config.json` via `create_from_config()`.
- **Do not** use `..encoding` or `..motifs` from within `models/cgan/` or `models/mlp/` — use 3 dots (`...`).
- **Do not** modify `motifs.py` for stoichiometric fraction handling — fractions are parsed at the storage level (`_build_fraction_lookup` in `extract.py` and `generate.py`).
- **Do not** add new callers/parsers outside of `prep/callers/` — use the `@register` decorator pattern.
- **Do not** add data preparation commands to `kinsim/__main__.py` — they belong in `prep/__main__.py`.
- **Do not** use relative imports (`from ..prep`) from `kinsim` to reach `prep` — use absolute `from prep.` imports (lazy, inside functions only).

---

## Adding a New Model Mode

1. Create `kinsim/models/<mode>/` with `__init__.py`, `model.py`, `train.py`, `generate.py`.
2. Import `KmerSignalDataset` from `...common.dataset` (not from cgan).
3. Import `log_transform`/`inv_log_transform` from `...common.dataset`.
4. Import BAM/genome utilities from `...dictionary.inject` and `...motifs`.
5. Save `model_config.json` with all architecture hyperparameters before training.
6. Add the new command to `kinsim/__main__.py` (COMMANDS list + routing block).
7. Add a SLURM train + generate script under `slurm_kinsim/`.
8. Add the new mode to `slurm_kinsim/kinsim_pipeline.sh`.
9. Data prep reuses `kinsim extract` + `kinsim merge` — no separate extraction needed.
10. Add a `--config` YAML example to `slurm_kinsim/config_<mode>_example.yaml`.

## Adding a New Motif Parser

1. Create `prep/callers/<name>.py` with a `@register` class inheriting `BaseOutputParser`.
2. Define `name`, `supported_mods`, `parse()`, and `is_file_for_this_parser()`.
3. Import the new module in `prep/callers/__init__.py` to trigger registration.
4. The parser is immediately available via `create_parser("name")` and auto-detection.

---

## Key Numbers

| Constant | Value | Where |
|---|---|---|
| K (k-mer size) | 11 | `kinsim/encoding.py` |
| Total possible k-mers | 4,194,304 (4^11) | `kinsim/encoding.py` |
| Methylation states | 4 (none/m6A/m4C/m5C) | `kinsim/encoding.py` |
| MID (flanking bases) | 5 | `kinsim/dictionary/inject.py` |
| Reservoir cap (extract) | 10,000 per (kmer, meth) | `kinsim/common/extract.py` |
| Reservoir cap (merge) | 50,000 per (kmer, meth) | `kinsim/common/extract.py` |
| MLPSignalDataset cap (unmeth) | 20 per key | `kinsim/common/dataset.py` |
| MLPSignalDataset cap (meth) | 100 per key | `kinsim/common/dataset.py` |
| BAM signal range | [0, 255] uint8 | all generate files |
| N-context default signal | 1 (not 0) | all generate files |
| log_sigma clamp range | [-6, 3] | `kinsim/models/mlp/model.py` |
| cGAN n_critic | 5 | `kinsim/models/cgan/train.py` |
| cGAN lambda_gp | 10 | `kinsim/models/cgan/train.py` |
| MLP train/val split | 90% / 10% | `kinsim/models/mlp/train.py` |
| LR patience (MLP) | 5 epochs | `kinsim/models/mlp/train.py` |
| LR factor (MLP) | 0.5 | `kinsim/models/mlp/train.py` |
| meth_proj_dim | 8 | `kinsim/models/mlp/model.py` |
| ConvPredictor params | ~140K | `kinsim/models/mlp/model.py` |
| MLPPredictor params | ~268M | `kinsim/models/mlp/model.py` |
