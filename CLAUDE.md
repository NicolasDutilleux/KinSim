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
│   ├── __main__.py                 CLI router — all commands dispatched here
│   │
│   ├── encoding.py                 11-mer bit-packing (no dependencies)
│   ├── motifs.py                   IUPAC motif parsing, sequence scanning, meth maps
│   ├── rebase_parser.py            REBASE → KinSim motif string conversion
│   ├── prepare.py                  kinsim prepare — validates BAM/motif config files
│   │
│   ├── common/                     shared data pipeline (used by MLP and cGAN)
│   │   ├── __init__.py
│   │   ├── dataset.py              log_transform, inv_log_transform, KmerSignalDataset
│   │   └── extract.py              BAM extraction + shard merging (kinsim cgan extract/merge)
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
│           ├── train.py            supervised training loop
│           └── generate.py         BAM generation with trained MLPPredictor
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
load_motif_string(arg, ...)                # auto-detect: string | .csv | REBASE | per-species file
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
write_fuzznuc_pattern_file(motif_string, filepath) -> dict  # for EMBOSS fuzznuc
```

### `kinsim/common/dataset.py`
The shared data contract between cGAN and MLP. Never import transforms from model files.

```python
log_transform(x: Tensor) -> Tensor      # log1p — raw [0,255] → training space
inv_log_transform(x: Tensor) -> Tensor  # expm1 clamped to [0,255] — inference

class KmerSignalDataset(Dataset):
    # Input:  dict[(kmer_id, meth_id)] -> np.ndarray(N, 2)  raw IPD/PW
    # Output: (kmer_id: long, meth_id: long, signal: float32 in log1p)
    # log_transform applied once at load time, not per-epoch
```

**Raw values are stored in .pkl files** (not log-transformed). The transform happens
inside KmerSignalDataset. This allows inspection and different model-specific transforms.

### `kinsim/common/extract.py`
The data preparation pipeline shared by all neural modes.

```python
extract_samples_from_bam(bam_path, motif_string, output_pkl, max_samples_per_key=10000)
    # reads BAM fi/fp tags
    # sliding 11-mer window → encode_kmer() → kmer_id
    # scan_sequence()  → meth_id per position
    # reservoir sampling (capped at 10,000 per (kmer, meth) key)
    # output: shard .pkl

merge_shards(input_dir, output_path, glob_pattern="*_cgan.pkl", max_per_key=50000)
    # loads all shards matching glob
    # concatenates arrays per key
    # optional subsampling
    # output: master .pkl
```

CLI: `kinsim cgan extract` and `kinsim cgan merge` (routed via parse_train.py shim).

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
    # kmer_embed_dim=64, hidden_dim=128
    # forward(kmer_ids, meth_ids) -> (batch, 4)
    #   [:, 0:2] = [μ_ipd, μ_pw]        in log1p space
    #   [:, 2:4] = [log_σ_ipd, log_σ_pw]
    #
    # @torch.no_grad()
    # sample(kmer_ids, meth_ids) -> (batch, 2)  stochastic, values in [0,255]
    # predict_mean(kmer_ids, meth_ids) -> (batch, 2)  deterministic, values in [0,255]
```

`log_σ` is clamped to [-6, 3] in `sample()` for numerical stability.

### `kinsim/models/mlp/train.py`
Supervised training with Gaussian NLL loss.

```python
Loss = 0.5 * (2*log_σ + (target - μ)² / σ²)  # jointly learns μ and σ
```

Alternatives: `--loss mse` or `--loss huber` (use μ head only, ignore variance).
Scheduler: `ReduceLROnPlateau(factor=0.5, patience=5)`. LR drops printed manually.
`model_config.json` keys: `kmer_embed_dim`, `hidden_dim`.

### `kinsim/models/mlp/generate.py`
Mirrors `cgan/generate.py`. Key difference: no noise vector.

```python
generate_signals_batch(model, kmer_ids, meth_ids, device, deterministic=False)
    # deterministic=False → model.sample()       (default, stochastic)
    # deterministic=True  → model.predict_mean() (same result every run)
```

`_load_model(checkpoint_path, device)` reads `model_config.json` and hard-errors if missing.
Output files: `<species>_mlp.bam`.

---

## Data Flow

```
Real PacBio BAMs
      │
      ▼  kinsim cgan extract <bam> <motif> <shard.pkl>
      │  (common/extract.py)
      │  sliding 11-mer → encode_kmer → kmer_id
      │  scan_sequence → meth_id per base
      │  reservoir sampling → cap 10,000 per (kmer, meth)
      ▼
shard_001.pkl  shard_002.pkl  ...
      │
      ▼  kinsim cgan merge <shards_dir/> <master.pkl>
      ▼
master_data.pkl    ←  dict[(kmer_id, meth_id)] → np.ndarray(N, 2)  [IPD, PW] raw
      │
      ├──▶  kinsim mlp train  → checkpoint.pt + model_config.json
      └──▶  kinsim cgan train → checkpoint.pt + model_config.json

PBSIM3 reads (.fq.gz + .maf.gz) + reference (.fna) + checkpoint.pt
      │
      ▼  kinsim mlp generate  / kinsim cgan generate
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
from .model import ...      # within the same sub-package (1 dot)
```

**Never import `log_transform`/`inv_log_transform` from `cgan/model.py`.**
Always import from `common/dataset.py`. The re-export in `cgan/model.py` exists only for
backward compatibility with old code.

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
- Use `ReduceLROnPlateau` — never `verbose=True` (removed in PyTorch ≥ 2.1). Print LR drops manually.
- Wrap training loops in `try/finally` to ensure CSV and TensorBoard are closed on crash.
- Always save `model_config.json` **before** the first epoch, not after training.
- Always include `"scheduler"` key in checkpoint dict for resume support.

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
- **Storage (.pkl)**: raw values (not transformed) — transforms are model-specific.

### SLURM Scripts
- Array jobs for per-species tasks (`#SBATCH --array=1-N`).
- Auto-detect flat vs subdirectory layout in `generate.py` (not in SLURM scripts).
- `kinsim_pipeline.sh` is the single source of truth for the full pipeline order.

---

## What NOT To Do

- **Do not** add logic to `models/cgan/parse_train.py` — it is a shim only.
- **Do not** store log-transformed data in `.pkl` files — raw values only.
- **Do not** import `KmerSignalDataset` or transforms from `cgan/train.py` — use `common/dataset.py`.
- **Do not** use `verbose=True` in `ReduceLROnPlateau` — crashes on PyTorch ≥ 2.1.
- **Do not** add a new mode without adding a corresponding entry to `__main__.py` and `kinsim_pipeline.sh`.
- **Do not** hardcode `kmer_embed_dim` or `hidden_dim` in `generate.py` — always read from `model_config.json` and hard-error if missing.
- **Do not** use `..encoding` or `..motifs` from within `models/cgan/` or `models/mlp/` — use 3 dots (`...`).

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
9. Data prep reuses `kinsim cgan extract` + `kinsim cgan merge` — no separate extraction needed.

---

## Key Numbers

| Constant | Value | Where |
|---|---|---|
| K (k-mer size) | 11 | `encoding.py` |
| Total possible k-mers | 4,194,304 (4^11) | `encoding.py` |
| Methylation states | 4 (none/m6A/m4C/m5C) | `encoding.py` |
| MID (flanking bases) | 5 | `dictionary/inject.py` |
| Reservoir cap | 10,000 per (kmer, meth) | `common/extract.py` |
| BAM signal range | [0, 255] uint8 | all generate files |
| N-context default signal | 1 (not 0) | all generate files |
| log_σ clamp range | [-6, 3] | `models/mlp/model.py` |
| cGAN n_critic | 5 | `models/cgan/train.py` |
| cGAN lambda_gp | 10 | `models/cgan/train.py` |
| MLP train/val split | 90% / 10% | `models/mlp/train.py` |
| LR patience (MLP) | 5 epochs | `models/mlp/train.py` |
| LR factor (MLP) | 0.5 | `models/mlp/train.py` |
