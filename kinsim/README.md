# `kinsim/` — ML pipeline

The supervised learning pipeline that turns a kinetic library into a
trained model and uses it to inject realistic IPD/PW into PBSIM3 reads.

This package exposes the `kinsim` CLI (entry point in `pyproject.toml`).

For project-level overview see the [top-level README](../README.md).

---

## Pipeline stages

```
   manifest.csv  ──extract──►  shards/*.pkl
                  ──merge────► master.pkl
                  ──refine───► master_clean.pkl
                  ──train────► checkpoints/best.pt
                  ──generate─► simulated BAM
                  ──evaluate─► calibration report
                  ──verify───► ref vs gen comparison
```

| Stage | Module | What it does |
|---|---|---|
| `extract` | [extract.py](extract.py) | Parse a BAM read by read, scan for motifs, emit `(IPD, PW, profile, mc, rev_meth, CATEGORY)` rows per kmer |
| `merge` | [extract.py](extract.py) (`merge_shards`) | Concatenate per-sample shards into one master `.pkl` |
| `refine` | [refine.py](refine.py) | Drop CATEGORY_SLOWED rows whose IPD falls below the per-kmer-baseline-mean p95 (motif false positives) |
| `train` | [train.py](train.py) | Gaussian-NLL training of `ConvPredictor` (default) or `MLPPredictor` |
| `generate` | [generate.py](generate.py) | For each input read, predict `(IPD, PW)` per position; write a synthetic BAM |
| `evaluate` | [evaluate.py](evaluate.py) | Per-key calibration plots, μ / σ residuals |
| `verify-generate` | [verify_generate.py](verify_generate.py) | Per-(kmer, meth) distribution comparison: real vs generated BAM |
| `analyze` | [analyze.py](analyze.py) | Stats on a `.pkl`: coverage, signal distributions, signature profiles |

---

## Architecture

KinSim uses three asymmetric design choices:

### 1. Asymmetric kmer / methylation window — `[−7, +3]`

The polymerase has read more bases upstream than downstream at any moment,
and all kinetic signatures are downstream of the modification (m6A at +5,
m5C at +2/+6). To predict IPD at a position Y, the model needs upstream
context (modifications whose downstream effect reaches Y).

Inspired by Feng *et al.* 2013 (`kineticsTools`/`ipdSummary`, `[−7, +2]`
for unmodified DNA), extended to 11 positions to fit a single packed-integer
11-mer encoding. The prediction position sits at index 7 of the kmer.

### 2. Kinetic profile storage — `[0, +8]`

Every sample stores not just the `(IPD, PW)` at the prediction position
but also the **profile** (IPD and PW at offsets 0 to +8). The downstream
profile is what makes the methylation signature visible — analyse uses
it to confirm a motif's real biological effect.

### 3. Three-category single-pass extract + p95 refine

`extract` emits one of three category labels per sample (col 35 of the
36-col layout):

| Category | Code | Meaning |
|---|---|---|
| `BASELINE` | 0 | Far from any methylation; meth-context window is empty |
| `SLOWED` | 1 | At a signature offset of a methylation (or the methylation itself when `0 ∈ signature_offsets[T]`) |
| `NEAR_METH` | 2 | Close to a methylation but NOT at a signature offset — negative control |

`refine` then runs a single pass — `slowed_split` — that pools every
`BASELINE` sample's IPD per kmer, computes the per-kmer mean, and uses
the `secondary_percentile`-th percentile (default p95) of the per-kmer
mean distribution as the lower threshold for `SLOWED`. Slowed samples
below the threshold are motif false positives (the polymerase did not
actually slow). `BASELINE` and `NEAR_METH` pass through unchanged.

Per-kmer mean (rather than per-sample) avoids the natural per-sample
Poisson tail that PacBio polymerase pauses produce on unmodified DNA;
under CLT the per-kmer mean is much tighter, so the p95 reflects truly
anomalous kmers and weak signals (m4C, m5C) survive.

### Storage layout (`.pkl`, 38 columns)

| Cols | Contents |
|---|---|
| 0–1 | `IPD`, `PW` at the prediction position (uint8 [0, 255]) |
| 2 | stoichiometric `fraction` (soft label from upstream caller) |
| 3–13 | `mc_0..mc_10` — meth_id at offsets `[−7, +3]` from prediction position |
| 14–22 | `profile_IPD_0..+8` — IPD profile at downstream offsets |
| 23–31 | `profile_PW_0..+8` — PW profile |
| 32–34 | `rev_meth_−1, 0, +1` — complementary-strand methylation |
| 35 | `CATEGORY` — 0=baseline, 1=slowed, 2=near_meth |
| 36 | `PARENT_METH` — meth_id of the parent meth (0 for baseline) |
| 37 | `PARENT_OFFSET` — offset (row pos − parent meth pos), in `[0, 7]` |

`kmer_id` is a 22-bit integer encoding 11 bases (A=0, C=1, G=2, T=3).
`meth_id ∈ {0=none, 1=m6A, 2=m4C, 3=m5C}`.

---

## Models

| Model | Params | Strategy |
|---|---|---|
| **ConvPredictor** (default) | ~140K | Per-base + positional embeddings, **global FiLM** methylation conditioning, Conv1D backbone, dual readout (centre + global pool) |
| **MLPPredictor** (legacy) | ~268M | Flat 4M-row k-mer embedding table + 2-layer MLP |

ConvPredictor uses **global FiLM** (Feature-wise Linear Modulation):

```
meth_full (B, K, M)  ──flatten──►  Linear(K·M → 32)
                                          │
                                          ▼
                                    (γ, β) broadcast over kmer positions
                                          │
base_embed + pos_embed  ─────► (1 + γ) ⊙ x + β  ─►  Conv1D backbone  ─►  (μ, log σ)
```

Loss: Gaussian NLL in log1p space.

```
L = ½ · (2 · log σ + (y − μ)² / σ²)
```

At inference, `inv_log_transform = expm1` returns the value to the raw
[0, 255] uint8 scale of `fi/fp` tags.

---

## CLI reference

### `kinsim extract`

```bash
# Manifest mode (recommended for SLURM array jobs)
kinsim extract --manifest manifest.csv --task ${SLURM_ARRAY_TASK_ID} --output-dir shards/

# Single-BAM mode
kinsim extract <bam> <motifs_source> <output.pkl>
```

Supported BAM formats:

| Format | Tags | Reverse-strand support |
|---|---|---|
| Raw HiFi (unaligned) | `fi/fp` + `ri/rp` | ✅ Both strands per read |
| Bystrandified | `ip/pw` (×2 reads) | ✅ Each strand = own read |
| Aligned (post-pbmm2) | `ip/pw` only | ❌ `ri/rp` dropped — half the data |

Tag pair auto-detected (`validate_bam_kinetics`). **Aligned BAMs
are not supported** — pass an unaligned (raw or bystrandified) BAM.

### `kinsim merge`

```bash
kinsim merge shards/ master.pkl
```

Concatenates per-sample shards into one master `.pkl`.

### `kinsim refine`

```bash
kinsim refine master.pkl master_clean.pkl [--secondary-percentile 95] [-v]
```

Single pass: drops `SLOWED` samples whose IPD falls below the
`secondary_percentile`-th percentile of the per-kmer baseline mean
distribution (default 95, configurable in `kinsim_config.yaml`).
`BASELINE` and `NEAR_METH` pass through unchanged.

### `kinsim train`

```bash
kinsim train master_clean.pkl checkpoints/ [--architecture conv|mlp] [--config training.yaml]
```

Saves `model_config.json` + per-epoch checkpoints + `metrics.csv` +
TensorBoard logs. Resumable.

```yaml
# training.yaml (optional)
architecture: conv          # or 'mlp'
batch_size: 4096
epochs: 50
lr: 0.001
weight_decay: 1e-5
val_split: 0.1
patience: 5
factor: 0.5
```

### `kinsim generate`

Three call modes auto-detected:

```bash
# PBSIM3 directory (one BAM per genome subdirectory)
kinsim generate <pbsim3_dir> <ref.fna> <checkpoint.pt> <motifs> <output_dir>

# Single BAM (e.g. an existing simulated BAM)
kinsim generate <input.bam> <ref.fna> <checkpoint.pt> <motifs> <output.bam>

# Per-genome simulator output
kinsim generate <fq.gz> <maf.gz> <ref.fna> <ckpt.pt> <motifs> <out.bam>
```

`--deterministic` writes μ directly; default is stochastic sampling
from `N(μ, Σ)`.

For each input read, `generate.py` calls the model twice:

1. Forward kmers → produces `fi:B:C` / `fp:B:C` tags
2. Reverse-complement kmers → produces `ri:B:C` / `rp:B:C` tags

This produces a single-read-per-molecule BAM (raw HiFi format).

### `kinsim evaluate`

```bash
kinsim evaluate <ckpt_dir> <pkl>
```

Reads the latest checkpoint, runs inference on the validation split,
produces calibration plots and per-key μ / σ residuals.

### `kinsim verify-generate`

```bash
kinsim verify-generate <ref.bam> <gen.bam> <motifs> <report.tsv>
```

Per-(kmer, meth) bucket comparison: extracts samples from both BAMs,
computes Pearson correlation and MAE on per-key means and sigmas.

### `kinsim analyze`

```bash
kinsim analyze <master_clean.pkl> [--output-dir reports/] [--no-html]
```

Writes `<basename>_report.txt` and `<basename>_report.html`. The HTML
report focuses on four verification figures:

1. IPD distribution per category (with refine threshold)
2. Per-kmer baseline-mean distribution (where the threshold sits)
3. Kinetic signature profiles per bucket (offsets 0 to +8)
4. Sample counts per bucket

---

## Module layout

```
kinsim/
├── __init__.py
├── __main__.py             — CLI router
│
├── extract.py              — BAM extraction, shard merging, sample storage layout
├── refine.py               — slowed_split: per-kmer baseline mean p95 filter
├── train.py                — supervised training loop (Gaussian NLL)
├── generate.py             — BAM generation with trained model
├── evaluate.py             — calibration report
├── verify_generate.py      — ref vs generated BAM comparison
├── analyze.py              — distribution stats + signature profiles
├── sample.py               — random subsampling of .pkl files
├── strip_kinetics.py       — remove fi/fp/ri/rp tags from a BAM copy
│
├── data/
│   ├── __init__.py
│   └── dataset.py          — log_transform, inv_log_transform, MLPSignalDataset
│
├── models/
│   ├── __init__.py
│   └── predictor.py        — ConvPredictor + MLPPredictor + create_from_config
│
└── utils/
    ├── __init__.py
    ├── encoding.py         — 11-mer bit-packing (K, KMER_PRED_IDX, BASE_MAP, METH_IDS)
    ├── motifs.py           — IUPAC parsing, sequence scanning, meth maps
    ├── config.py           — manifest CSV loader, kinsim_config loader, logging setup
    ├── io.py               — FASTA loading, MAF parsing, PBSIM3 discovery
    └── sample_layout.py    — 36-col layout, CATEGORY enum, slice helpers
```

Imports follow:
- top-level kinsim modules → `from .utils.X import …`, `from .data.dataset import …`
- `kinsim/models/` → `from ..utils.encoding import …`
- `kinsim/data/` → only standard library + numpy + torch
