# KinSim — Developer Reference for Claude

## Project Summary

KinSim simulates PacBio HiFi kinetic signals (IPD and PW) for metagenomic binning research.
Given PBSIM3-simulated reads and a reference genome, KinSim injects biologically realistic
per-base IPD/PW values into unaligned BAM files using one of three modes:

- **dictionary** — Gaussian sampling from per-k-mer accumulators (fast, no GPU)
- **mlp**        — Supervised MLP predicting N(μ, σ²) per context (Level 1 AI)
- **cgan**       — Conditional WGAN-GP (Level 2 AI, captures non-Gaussian distributions)

All three modes output BAMs with standard PacBio tags: `fi:B:C` (IPD) and `fp:B:C` (PW).

---

## Repository Layout

```
KinSim/
├── pyproject.toml                  entry point: kinsim = "kinsim.__main__:main"
├── requirements.txt
│
├── kinsim/                         main Python package
│   ├── __init__.py
│   ├── __main__.py                 CLI router — all commands dispatched here; sets up logging
│   │
│   ├── encoding.py                 11-mer bit-packing (no dependencies)
│   ├── motifs.py                   IUPAC motif parsing, sequence scanning, meth maps
│   ├── rebase_parser.py            REBASE → KinSim motif string conversion
│   ├── prepare.py                  kinsim prepare — validates BAM/motif config files
│   ├── config.py                   manifest CSV loader, YAML config, logging setup
│   │
│   ├── callers/                    methylation caller output parsers (read-only)
│   │   ├── __init__.py             exports: BaseOutputParser, create_parser, list_parsers, auto_detect_parser
│   │   ├── base.py                 BaseOutputParser ABC
│   │   ├── registry.py             @register decorator, create_parser(), auto_detect_parser()
│   │   ├── pacbio.py               PacBioParser — motifs.csv with variable columns
│   │   ├── modkit.py               ModkitParser — modkit pileup --bedMethyl TSV
│   │   └── ipd_summary.py          IpdSummaryParser — ipdSummary CSV/GFF3
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
│       │   └── parse_train.py      shim → delegates to common/extract.py
│       └── mlp/
│           ├── __init__.py
│           ├── model.py            MLPPredictor (Gaussian NLL head)
│           ├── train.py            supervised training loop (Lightning + Optuna)
│           ├── generate.py         BAM generation with trained MLPPredictor
│           └── evaluate.py         calibration report + per-kmer distribution plots
│
├── slurm_kinsim/                   HPC SLURM job scripts
│   ├── kinsim_pipeline.sh          master script (mode: dictionary | cgan | mlp)
│   ├── pbsim3_simulate.slurm
│   ├── kinsim_prepare.slurm
│   ├── kinsim_train.slurm          dictionary train
│   ├── kinsim_merge.slurm          dictionary merge
│   ├── kinsim_inject.slurm         dictionary inject (array job)
│   ├── kinsim_inject_alone.slurm
│   ├── kinsim_analyze.slurm
│   ├── kinsim_cgan_extract.slurm   cgan extract (array job)
│   ├── kinsim_cgan_train.slurm     cgan train (1 GPU, 24h)
│   ├── kinsim_cgan_generate.slurm  cgan generate (array job)
│   ├── kinsim_mlp_train.slurm      mlp train (1 GPU, 24h)
│   └── kinsim_mlp_generate.slurm   mlp generate (array job)
│
└── cluster_tests/                  integration test data and old validation scripts
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

encode_kmer(seq: str) -> int    # 11-char string → 22-bit integer
decode_kmer(val: int) -> str    # 22-bit integer → 11-char string
get_ipd_stats(acc) -> (μ, σ)   # extract IPD mean/std from accumulator
get_pw_stats(acc)  -> (μ, σ)   # extract PW  mean/std from accumulator
```

Accumulator format: `np.array([n, Σipd, Σipd², Σpw, Σpw²])` (Welford-style).

### `kinsim/motifs.py`
Everything methylation-motif related. Used by all three modes.

```python
iupac_to_re(motif)                        # "RGATCY" → "[AG]GAT[CT][CT]"
reverse_complement(seq)                    # IUPAC-aware
parse_motifs(motif_string, revcomp=True)   # "m6A,GATC,1;..." → list of dicts
scan_sequence(seq, motifs) -> np.int8[]   # per-base methylation ID array
parse_motifs_csv(path, min_fraction=0.40)  # PacBio motifs.csv → KinSim string
load_motif_string(arg, ..., parser_name=None)  # auto-detect or explicit parser from callers registry
build_reference_meth_map(ref_seqs, motif_string)  # genome-wide O(1) lookup array
```

**Motif string format**: `"mod_type,pattern,position"` semicolon-delimited.
Example: `"m6A,GATC,1;m4C,CCWGG,2"`.
Position is 0-based index of the modified base within the pattern.

### `kinsim/rebase_parser.py`
Converts REBASE notation (1-based) to KinSim notation (0-based).

```python
parse_rebase_annotation(recognition_seq, meth_annotation) -> list[str]
parse_rebase_simple(filepath)       # 2-column TSV format
parse_rebase_withrefm(filepath)     # Format #19 tagged fields (RS=, MS=)
parse_rebase_isoschizomers(filepath) -> dict[str, list[str]]  # motif → [enzyme_IDs]
write_fuzznuc_pattern_file(motif_string, filepath) -> dict  # for EMBOSS fuzznuc
```

### `kinsim/callers/`
Read-only parsing library for methylation caller output files. Plugin registry
with `@register` decorator — adding a new format = one file + `@register` class.

```python
from kinsim.callers import create_parser, list_parsers, auto_detect_parser

# Explicit parser
parser = create_parser("pacbio")       # or "modkit", "ipd_summary"
motif_string = parser.parse("motifs.csv", min_fraction=0.40, min_detected=20)

# Auto-detect from file content
parser = auto_detect_parser("output.bed")

# List registered parsers
list_parsers()  # ['ipd_summary', 'modkit', 'pacbio']
```

**BaseOutputParser ABC** (`base.py`):
- `name: ClassVar[str]` — registry key
- `supported_mods: ClassVar[list[str]]` — mod types this format carries
- `parse(filepath, min_fraction, min_detected) -> str` — file → motif string
- `is_file_for_this_parser(filepath) -> bool` — heuristic for auto-detection

**PacBioParser** (`pacbio.py`): Handles motifs.csv with variable columns.
Required: `motifString`, `centerPos`. Optional: `modificationType`, `fraction`, `nDetected`.
Missing `modificationType` → inferred from base at centerPos (A→m6A, C→m4C).

**ModkitParser** (`modkit.py`): Handles modkit pileup `--bedMethyl` TSV (11+ columns).

**IpdSummaryParser** (`ipd_summary.py`): Auto-detects CSV vs GFF3 from ipdSummary.

**Integration**: `load_motif_string()` in `motifs.py` accepts optional `parser_name` kwarg.
When provided, bypasses auto-detection and uses the named parser from the registry.

### `kinsim/config.py`
Manifest CSV parsing, YAML config loading, and logging setup.

```python
@dataclass
class SampleEntry:
    sample_id: str
    bam_path:  str
    motifs:    str    # KinSim string or path (resolved by load_motif_string)

load_manifest(manifest_path) -> list[SampleEntry]
    # reads manifest CSV (sample_id, bam_path, motifs)
    # skips comment rows (#) and empty rows
    # raises FileNotFoundError or ValueError on format errors

load_yaml_config(path) -> dict
    # loads YAML training config; requires PyYAML
    # used by train.py --config flag

setup_logging(verbose=False)
    # configures root logger with timestamp format for SLURM logs
    # format: "2026-03-03 14:32:01 [INFO]    kinsim.common.extract: ..."
    # call once in main() of each CLI module
```

**Manifest CSV format**:
```csv
sample_id,bam_path,motifs
strain1,/data/bams/strain1.bam,"m6A,GATC,1"
strain2,/data/bams/strain2.bam,/data/motifs/strain2.csv
```
The `motifs` field accepts quoted KinSim strings (commas are fine) or file paths.

### `kinsim/common/dataset.py`
The shared data contract between cGAN and MLP. Never import transforms from model files.
Both dataset classes skip the `"__meta__"` string key automatically.

```python
log_transform(x: Tensor) -> Tensor      # log1p — raw [0,255] → training space
inv_log_transform(x: Tensor) -> Tensor  # expm1 clamped to [0,255] — inference

class KmerSignalDataset(Dataset):
    # Input:  dict[(kmer_id, meth_id)] -> np.ndarray(N, 2 or 3)
    # Output: (kmer_id: long, meth_id: long, signal: float32 in log1p)
    # Uses only [IPD, PW] columns (fraction ignored — cGAN uses integer meth_id)
    # log_transform applied once at load time, not per-epoch

class MLPSignalDataset(Dataset):
    # Input:  dict[(kmer_id, meth_id)] -> np.ndarray(N, 3) [IPD, PW, fraction]
    # __len__ = number of unique (kmer_id, meth_id) keys
    # __getitem__ draws ONE random (IPD, PW, fraction) per key (random-shot)
    # meth output = Float[4] stoichiometric vector (NOT one-hot):
    #   meth_id=0 → [0, 0, 0, 0]  (no methylation signal)
    #   meth_id=1, fraction=0.75 → [0, 0.75, 0, 0]
    # dynamic capping: meth_id=0 → max 20 samples; meth_id∈{1,2,3} → max 100
    # Backward compat: 2-column .pkl → fraction defaults to 1.0 (meth) / 0.0 (unmeth)
```

**Raw values are stored in .pkl files** (not log-transformed). The transform happens
inside the dataset class. This allows inspection and different model-specific transforms.
The third column (fraction) is the stoichiometric methylation fraction from PacBio
motifs.csv (default 1.0 when not available).

### `kinsim/common/extract.py`
The data preparation pipeline shared by all neural modes.

```python
validate_bam_kinetics(bam_path, n_check=10)
    # fail-fast: raises ValueError if BAM has no fi/fp kinetic tags
    # call before starting extraction loop to avoid silent failures

extract_samples_from_bam(bam_path, motif_string, max_samples_per_key=10000, revcomp=True)
    # calls validate_bam_kinetics first
    # sliding 11-mer window → encode_kmer() → kmer_id
    # scan_sequence() → meth_id per position
    # _build_fraction_lookup(motif_string) → {meth_id: fraction} from motif string field 5
    # reservoir sampling (capped at max_samples_per_key per (kmer, meth) key)
    # returns dict with (kmer_id, meth_id) tuple keys → np.ndarray(N, 3) [IPD, PW, fraction]
    #   + "__meta__" provenance key

merge_shards(input_dir, output_file, max_samples_per_key=50000, glob_pattern="auto")
    # "auto" tries *_shard.pkl (new) then *_cgan.pkl (legacy)
    # concatenates arrays per key; subsamples if needed
    # includes merged "__meta__" with list of source BAMs

extract_from_manifest_task(manifest_path, task_index, output_dir, ...)
    # reads manifest CSV, picks row task_index (1-based = SLURM_ARRAY_TASK_ID)
    # writes shards/<sample_id>_shard.pkl
```

CLI (single-BAM mode — unchanged):
```
kinsim extract reads.bam "m6A,GATC,1" shard.pkl
kinsim merge   shards/   master_data.pkl
```

CLI (manifest mode — recommended for SLURM):
```
kinsim extract --manifest manifest.csv --task $SLURM_ARRAY_TASK_ID --output-dir shards/
kinsim merge   shards/ master_data.pkl
```

**Provenance in .pkl**: every shard and master file now contains a `"__meta__"` key
(string, never a tuple) with `kinsim_version`, `source_bam`, `motifs`, `created` timestamp.
Dataset classes automatically skip non-tuple keys.

### `kinsim/models/cgan/model.py`
WGAN-GP Generator and Discriminator. Re-exports transforms for backward compatibility.

```python
from ...common.dataset import log_transform, inv_log_transform  # noqa: F401

class Generator(nn.Module):
    # noise_dim=32, kmer_embed_dim=64, hidden_dim=128
    # forward(z, kmer_ids, meth_ids) -> (batch, 2)  in log1p space

class Discriminator(nn.Module):
    # kmer_embed_dim=64, hidden_dim=128, dropout=0.3
    # forward(signals, kmer_ids, meth_ids) -> (batch, 1)  raw WGAN score (no sigmoid)
```

Both classes use Xavier init. Embeddings: `Embedding(4_194_304, kmer_embed_dim)` and
`Embedding(4, 8)`. No weight sharing between G and D.

### `kinsim/models/cgan/train.py`
WGAN-GP training. Key hyperparameters:

```
Adam(betas=(0.0, 0.9))     no momentum — required for WGAN stability
n_critic = 5               discriminator updates per generator update
lambda_gp = 10             gradient penalty weight
```

Saves: `checkpoint_epochN.pt`, `model_config.json`, `training_log.csv`, `runs/` (TensorBoard).
`model_config.json` keys: `noise_dim`, `kmer_embed_dim`, `hidden_dim`, `n_critic`, `lambda_gp`.

### `kinsim/models/cgan/generate.py`
Two auto-detected calling modes (directory vs per-genome, detected by `os.path.isdir(argv[0])`).
Output files: `<species>_cgan.bam`.

Methylation scanning: EMBOSS fuzznuc primary backend, Python regex fallback.
The reference is pre-scanned **once** → `meth_map[ref_name][ref_pos]` = meth_id (O(1) per read).

### `kinsim/models/cgan/parse_train.py`
**Thin shim only.** All logic is in `common/extract.py`.

```python
from ...common.extract import extract_samples_from_bam, merge_shards, main
```

Exists so `kinsim cgan extract/merge` commands (and existing SLURM scripts) keep working.
Do not add logic here.

### `kinsim/models/mlp/model.py`

```python
class MLPPredictor(nn.Module):
    # kmer_embed_dim=64, hidden_dim=128, meth_proj_dim=8, dropout=0.0
    # Methylation input: nn.Linear(4, meth_proj_dim, bias=False)
    #   accepts Float[batch, 4] stoichiometric probability vector (not integer id)
    #   e.g. m6A at 75% → [0, 0.75, 0, 0]; unmethylated → [0, 0, 0, 0]
    #
    # forward(kmer_ids, meth_probs) -> (batch, 4)
    #   [:, 0:2] = [μ_ipd, μ_pw]        in log1p space
    #   [:, 2:4] = [log_σ_ipd, log_σ_pw]
    #
    # @torch.no_grad()
    # sample(kmer_ids, meth_probs) -> (batch, 2)  stochastic, values in [0,255]
    # predict_mean(kmer_ids, meth_probs) -> (batch, 2)  deterministic, values in [0,255]
```

`log_σ` is clamped to [-6, 3] in `sample()` for numerical stability.

### `kinsim/models/mlp/train.py`
Supervised training with Gaussian NLL loss. Supports `--config` YAML for reproducibility.

```python
Loss = 0.5 * (2*log_σ + (target - μ)² / σ²)  # jointly learns μ and σ
```

Alternatives: `--loss mse` or `--loss huber` (use μ head only, ignore variance).
Scheduler: `ReduceLROnPlateau(factor=0.5, patience=5)`. LR drops logged at INFO level.
`model_config.json` keys: `kmer_embed_dim`, `hidden_dim`, `meth_proj_dim`.

`--config config_mlp.yaml` overrides all default values; CLI flags override YAML.
Example: `kinsim train --model mlp --config config_mlp.yaml --epochs 100`

### `kinsim/models/mlp/generate.py`
Mirrors `cgan/generate.py`. Key difference: no noise vector.

```python
generate_signals_batch(model, kmer_ids, meth_ids, fractions, device, deterministic=False)
    # Builds stoichiometric meth_probs from (meth_id, fraction) pairs via scatter_()
    # e.g. meth_id=1, fraction=0.75 → meth_probs = [0, 0.75, 0, 0]
    # deterministic=False → model.sample()       (default, stochastic)
    # deterministic=True  → model.predict_mean() (same result every run)

_build_fraction_lookup(motif_string) -> dict[int, float]
    # Parses motif string field 5 (fraction) into {meth_id: fraction} dict
    # Defaults to 1.0 when fraction field absent; meth_id=0 always maps to 0.0
    # Used in both generate.py and extract.py
```

`_load_model(checkpoint_path, device)` reads `model_config.json` and hard-errors if missing.
Output files: `<species>_mlp.bam`.

---

## Data Flow

```
Real PacBio BAMs + Manifest CSV (sample_id, bam_path, motifs)
      │
      ▼  kinsim extract --manifest manifest.csv --task $TASK --output-dir shards/
      │  (common/extract.py)
      │  validate_bam_kinetics() — fail-fast before extraction loop
      │  sliding 11-mer → encode_kmer → kmer_id
      │  scan_sequence → meth_id per position
      │  _build_fraction_lookup → meth_id → stoichiometric fraction
      │  reservoir sampling → cap 10,000 per (kmer, meth)
      │  stores [IPD, PW, fraction] per sample (3 columns)
      │  writes shards/<sample_id>_shard.pkl  (includes __meta__ provenance)
      ▼
strain1_shard.pkl  strain2_shard.pkl  ...
      │
      ▼  kinsim merge <shards_dir/> <master.pkl>
      │  auto-detects *_shard.pkl (new) or *_cgan.pkl (legacy)
      ▼
master_data.pkl
    "__meta__": {kinsim_version, merged_from, created, ...}
    (kmer_id, meth_id) → np.ndarray(N, 3)  [IPD, PW, fraction] raw
      │
      ├──▶  kinsim train --model mlp  [--config config_mlp.yaml]
      │         → checkpoint.pt + model_config.json + training_log.csv
      └──▶  kinsim train --model cgan [--config config_cgan.yaml]
                → checkpoint.pt + model_config.json + training_log.csv

PBSIM3 reads (.fq.gz + .maf.gz) + reference (.fna) + checkpoint.pt
      │
      ▼  kinsim generate --model mlp  / kinsim generate --model cgan
      │  load_reference → pre-scan meth_map (O(1) lookup)
      │  parse_maf → read→reference alignment
      │  for each batch: encode kmer_ids + lookup meth_ids → model.sample()
      ▼
species_mlp.bam / species_cgan.bam
  flag=4  fi:B:C (IPD uint8)  fp:B:C (PW uint8)
```

---

## BAM Output Contract

Every KinSim output BAM (all three modes) must satisfy:

| Field | Value |
|---|---|
| `flag` | `4` (unmapped) |
| `fi:B:C` | IPD per base, uint8, length == read length |
| `fp:B:C` | PW per base, uint8, length == read length |
| N-context positions | signal = `1` (not `0`, which means "no data" in PacBio) |
| Header | `HD VN:1.6 SO:unknown` |

---

## Import Path Rules

All files under `kinsim/models/cgan/` and `kinsim/models/mlp/` are **2 levels** below `kinsim/`.
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

**Never import `log_transform`/`inv_log_transform` from `cgan/model.py`.**
Always import from `common/dataset.py`. The re-export in `cgan/model.py` exists only for
backward compatibility with old code.

**Never use bare `print()` for operational output.**
Always use `log = logging.getLogger(__name__)` and `log.info()`, `log.warning()`, `log.error()`.

---

## Methylation State IDs

```python
METH_IDS = {'none': 0, 'm6A': 1, 'm4C': 2, 'm5C': 3}
```

Consistent across all modes. Defined in `encoding.py`, used everywhere.

---

## CLI Command Map

```
kinsim prepare               → kinsim/prepare.py
kinsim motifs                → kinsim/motifs.py
kinsim rebase parse|patterns → kinsim/rebase_parser.py

kinsim dictionary train      → kinsim/dictionary/train.py        main(["train", ...])
kinsim dictionary merge      → kinsim/dictionary/train.py        main(["merge", ...])
kinsim dictionary inject     → kinsim/dictionary/inject.py       main(...)
kinsim dictionary metagenome → kinsim/dictionary/inject.py       metagenome_main(...)
kinsim dictionary analyze    → kinsim/dictionary/analyze.py      main(...)

kinsim cgan extract          → kinsim/models/cgan/parse_train.py main(["extract", ...])
kinsim cgan merge            → kinsim/models/cgan/parse_train.py main(["merge", ...])
kinsim cgan train            → kinsim/models/cgan/train.py       main(...)
kinsim cgan generate         → kinsim/models/cgan/generate.py    main(...)

kinsim mlp train             → kinsim/models/mlp/train.py        main(...)
kinsim mlp generate          → kinsim/models/mlp/generate.py     main(...)
kinsim mlp evaluate          → kinsim/models/mlp/evaluate.py     main(...)
```

Typo suggestions via `difflib.get_close_matches` in `__main__.py`.

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
- Use `ReduceLROnPlateau` — never `verbose=True` (removed in PyTorch ≥ 2.1). Log LR drops via `log.info(...)`.
- Wrap training loops in `try/finally` to ensure CSV and TensorBoard are closed on crash.
- Always save `model_config.json` **before** the first epoch, not after training.
- Always include `"scheduler"` key in checkpoint dict for resume support.

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
- **Inference**: model outputs in log1p space → `inv_log_transform` → uint8 [0, 255].
- **Storage (.pkl)**: raw values (not transformed) + stoichiometric fraction column.
  Format: `np.ndarray(N, 3)` with columns [IPD, PW, fraction]. Legacy 2-col supported.
- **Metadata**: every .pkl has a `"__meta__"` string key with provenance. Dataset classes skip it.

### SLURM Scripts
- Array jobs for per-species tasks (`#SBATCH --array=1-N`).
- Use `kinsim_extract.slurm` with manifest CSV for extract jobs (replaces `kinsim_cgan_extract.slurm`).
- Auto-detect flat vs subdirectory layout in `generate.py` (not in SLURM scripts).
- `kinsim_pipeline.sh` is the single source of truth for the full pipeline order.

### Manifest CSV
- Manifest columns: `sample_id`, `bam_path`, `motifs` (CSV with header, commas OK in quoted fields).
- Count rows for `--array`: `N=$(tail -n +2 manifest.csv | grep -cv '^#')`.
- Output shard naming: `shards/<sample_id>_shard.pkl` (derived from manifest `sample_id`).

---

## What NOT To Do

- **Do not** add logic to `models/cgan/parse_train.py` — it is a shim only.
- **Do not** store log-transformed data in `.pkl` files — raw values only.
- **Do not** import `KmerSignalDataset` or transforms from `cgan/train.py` — use `common/dataset.py`.
- **Do not** use `verbose=True` in `ReduceLROnPlateau` — crashes on PyTorch ≥ 2.1.
- **Do not** add a new mode without adding a corresponding entry to `__main__.py` and `kinsim_pipeline.sh`.
- **Do not** hardcode `kmer_embed_dim` or `hidden_dim` in `generate.py` — always read from `model_config.json` and hard-error if missing.
- **Do not** use `..encoding` or `..motifs` from within `models/cgan/` or `models/mlp/` — use 3 dots (`...`).
- **Do not** modify `motifs.py` for stoichiometric fraction handling — fractions are parsed from the motif string at the storage level (`_build_fraction_lookup` in `extract.py` and `generate.py`), not in the motif scanning code.

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

---

## Key Numbers

| Constant | Value | Where |
|---|---|---|
| K (k-mer size) | 11 | `encoding.py` |
| Total possible k-mers | 4,194,304 (4^11) | `encoding.py` |
| Methylation states | 4 (none/m6A/m4C/m5C) | `encoding.py` |
| MID (flanking bases) | 5 | `dictionary/inject.py` |
| Reservoir cap (extract) | 10,000 per (kmer, meth) | `common/extract.py` |
| Reservoir cap (merge) | 50,000 per (kmer, meth) | `common/extract.py` |
| MLPSignalDataset cap (unmeth) | 20 per key | `common/dataset.py` |
| MLPSignalDataset cap (meth) | 100 per key | `common/dataset.py` |
| BAM signal range | [0, 255] uint8 | all generate files |
| N-context default signal | 1 (not 0) | all generate files |
| log_σ clamp range | [-6, 3] | `models/mlp/model.py` |
| cGAN n_critic | 5 | `models/cgan/train.py` |
| cGAN lambda_gp | 10 | `models/cgan/train.py` |
| MLP train/val split | 90% / 10% | `models/mlp/train.py` |
| LR patience (MLP) | 5 epochs | `models/mlp/train.py` |
| LR factor (MLP) | 0.5 | `models/mlp/train.py` |
| meth_proj_dim (MLP V2) | 8 | `models/mlp/model.py` |
