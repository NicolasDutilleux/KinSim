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
| `extract` | `extract.py` | Parse a BAM read by read, scan for motifs in the read sequence, record `(IPD, PW)` per `(11-mer, methylation)` bucket |
| `merge` | `extract.py:merge_shards` | Concatenate per-sample shards into one master `.pkl` |
| `refine` | `refine.py` | Joint-GMM on combined `(None + Meth)` populations per signature offset; drops contamination |
| `train` | `train.py` | Gaussian-NLL training of `ConvPredictor` or `MLPPredictor` |
| `generate` | `generate.py` | For each input read, predict `(IPD, PW)` per position; write a synthetic BAM |
| `evaluate` | `evaluate.py` | Per-key calibration plots, μ/σ residuals |
| `verify-generate` | `verify_generate.py` | Per-`(kmer, meth)` distribution comparison: real vs generated BAM |
| `analyze` | `analyze.py` | Stats on a `.pkl`: coverage, signal distributions, signature profiles, neighbor sensitivity |

---

## Architecture (v3)

KinSim v3 uses three asymmetric design choices:

### 1. Asymmetric kmer / methylation window — `[−7, +3]`

The polymerase has read more bases upstream than downstream at any moment,
and all kinetic signatures are downstream of the modification (m6A at +5,
m5C at +2/+6). To predict IPD at a position Y, the model needs upstream
context (modifications whose downstream effect reaches Y).

Inspired by Feng *et al.* 2013 (`kineticsTools`/`ipdSummary`, `[−7, +2]`
for unmodified DNA), extended to 11 positions to fit a single packed-integer
11-mer encoding. Prediction position sits at index 7 of the kmer.

### 2. Kinetic profile storage — `[0, +8]`

Every sample stores not just the `(IPD, PW)` at the prediction position
but also the **profile** (IPD and PW at offsets 0 to +8). This enables
refine to validate methylation events by checking whether the kinetic
signature is actually present at the configured offsets.

### 3. Joint-GMM refine per signature offset

For each `(kmer, meth_id)` bucket, refine fits one **joint Gaussian
Mixture Model** per signature offset on the combined `(None + Meth)`
sample population:

- **m6A** — fits a GMM on IPD@+0, another on IPD@+5; intersection of "real meth" assignments kept
- **m4C** — single GMM on IPD@+0
- **m5C** — GMMs on IPD@+2 and IPD@+6

A sample is kept only if it falls in the **highest-IPD cluster** at
*every* signature offset. Methylation only **slows** the polymerase
(IPD goes up, never down) — so contamination always sits at lower IPD
than real meth.

Older `--method em`, `--method clustered`, and `--method mahalanobis`
remain available as fallbacks.

### Storage layout (`.pkl`, 35 columns)

| Cols | Contents |
|---|---|
| 0–1 | `IPD`, `PW` at the prediction position (uint8 [0, 255]) |
| 2 | stoichiometric `fraction` (soft label from upstream caller) |
| 3–13 | `mc_0..mc_10` — methylation context across 11 positions `[−7, +3]` |
| 14–22 | `profile_IPD_0..+8` — IPD profile at downstream offsets |
| 23–31 | `profile_PW_0..+8` — PW profile |
| 32–34 | `rev_meth_−1, 0, +1` — complementary-strand methylation (active-site footprint) |

`kmer_id` is a 22-bit integer encoding 11 bases (A=0, C=1, G=2, T=3).
`meth_id ∈ {0=none, 1=m6A, 2=m4C, 3=m5C}`.

---

## Models

| Model | Params | Strategy |
|---|---|---|
| **ConvPredictor** (default) | ~140K | Per-base + positional embeddings, **global FiLM** methylation conditioning, Conv1D backbone, dual readout (center + global pool) |
| **MLPPredictor** (legacy) | ~268M | Flat 4M-row k-mer embedding table + 2-layer MLP |

ConvPredictor uses **global FiLM** (Feature-wise Linear Modulation):

```
meth_full (B, N, M)  ──flatten──►  Linear(N·M → 32)
                                          │
                                          ▼
                                    (γ, β) broadcast over kmer positions
                                          │
base_embed + pos_embed  ─────► (1 + γ) ⊙ x + β  ─►  Conv1D backbone  ─►  (μ, log σ)
```

The global embedding decouples methylation context length from kmer length —
useful when extending the meth context to include complementary-strand info.

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
# Manifest mode (SLURM array)
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

The tag pair is auto-detected (`validate_bam_kinetics`). **Aligned BAMs
are not supported** — pass an unaligned (raw or bystrandified) BAM.

Useful flags:

- `--max-reads N` — smoke-test mode (early stop, biased)
- `--max-samples N` — reservoir cap per `(kmer, meth)` key (default 10000)
- `--no-binarize` — keep raw fractions instead of forcing 0/1

### `kinsim refine`

```bash
kinsim refine in.pkl out.pkl --report report.tsv [--method gmm_signature|mahalanobis|em] [-v]
```

Default `--method gmm_signature`. Reads signature offsets from
`kinsim_config.yaml`. Status codes in the report TSV:

| Status | Meaning |
|---|---|
| `gmm_2_2_real_kept` | K=2 at both offsets, samples in real-meth cluster kept |
| `gmm_3_2_real_kept` | K=3 at first offset, K=2 at second |
| `gmm_all1_kept` | All offsets resulted in K=1 (no separation possible — kept) |
| `skip_gmm_no_valid` | No real-meth cluster found → bucket rejected |
| `lowN_mahal_kept` | Too few samples for GMM, Mahalanobis fallback kept some |
| `no_none_pair` | No `(kmer, none)` baseline available — bucket skipped |

### `kinsim train`

```bash
kinsim train master_clean.pkl checkpoints/ \
    [--architecture conv|mlp] [--config training.yaml]
```

Saves `model_config.json` + per-epoch checkpoints + `metrics.csv` +
TensorBoard logs. Resumable.

Training config YAML (optional):

```yaml
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

`--deterministic` writes μ directly. Default is stochastic sampling from
`N(μ, Σ)`.

For each input read, `generate.py` calls the model twice:

1. Forward kmers → produces `fi:B:C` / `fp:B:C` tags
2. Reverse-complement kmers → produces `ri:B:C` / `rp:B:C` tags

This produces a single-read-per-molecule BAM (raw HiFi format), ready
to be re-bystrandified for downstream tools.

### `kinsim evaluate`

```bash
kinsim evaluate <ckpt_dir> <pkl>
```

Reads the latest checkpoint, runs inference on the validation split,
produces calibration plots and per-key μ/σ residuals.

### `kinsim verify-generate`

```bash
kinsim verify-generate <ref.bam> <gen.bam> <motifs> <report.tsv>
```

Per-`(kmer, meth)` bucket comparison: extracts samples from both BAMs,
computes Pearson correlation and MAE on per-key means and sigmas.

### `kinsim analyze`

```bash
kinsim analyze <pkl> [--output-dir reports/] [--no-html]
```

Outputs in the chosen directory:

- `<basename>_report.txt` — text summary
- `<basename>_report.html` — full interactive report
- `figures/00..12_*.html` — 13 individual Plotly figures

Sections: overview, per-meth coverage, signal statistics, **kinetic
signature profile per type**, low-coverage warnings, neighbor
sensitivity, 3D density surfaces.

---

## Module layout

```
kinsim/
├── __init__.py
├── __main__.py             — CLI router
│
├── extract.py              — BAM extraction, shard merging, sample storage layout
├── refine.py               — joint-GMM cleanup per signature offset
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
    └── io.py               — FASTA loading, MAF parsing, PBSIM3 discovery
```

Imports follow:
- top-level kinsim modules → `from .utils.X import …`, `from .data.dataset import …`
- `kinsim/models/` → `from ..utils.encoding import …`
- `kinsim/data/` → only standard library + numpy + torch
