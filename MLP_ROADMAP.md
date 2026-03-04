# KinSim MLP — Reading Roadmap

> **Goal:** Understand the MLP pipeline end-to-end in 6 steps, starting from pure math
> and ending at generating a BAM file. Each step is self-contained; you can read them
> in order or jump to the one you need.

---

## The Big Picture

```
Real PacBio BAM
    │
    │  Step 1: Biology + Encoding
    │  kinsim/encoding.py + kinsim/motifs.py
    ▼
(kmer_id, meth_id) keys  ──────────────────────────────────────────┐
    │                                                               │
    │  Step 2: Extraction                                           │
    │  kinsim/common/extract.py                                     │
    ▼                                                               │
shard.pkl  →  master_data.pkl                                       │
    │                                                               │
    │  Step 3: Dataset                                              │
    │  kinsim/common/dataset.py                                     │
    ▼                                                               │
DataLoader batches  (kmer_id, meth_id, log1p_signal)               │
    │                                                               │
    │  Step 4: Model Architecture                                   │
    │  kinsim/models/mlp/model.py                                   │
    ▼                                                               │
MLPPredictor → (μ_ipd, μ_pw, log_σ_ipd, log_σ_pw)                 │
    │                                                               │
    │  Step 5: Training Loop                                        │
    │  kinsim/models/mlp/train.py                                   │
    ▼                                                               │
checkpoint_epoch50.pt + model_config.json                          │
    │                                                               │
    │  Step 6: Signal Generation                                    │
    │  kinsim/models/mlp/generate.py  ◄──────────────────────────-─┘
    ▼
species_mlp.bam  (fi:B:C + fp:B:C tags)
```

---

## Step 1 — Biology and Encoding

**Files:** [kinsim/encoding.py](kinsim/encoding.py), [kinsim/motifs.py](kinsim/motifs.py)

### What you need to understand

KinSim's core insight: the kinetic signal at base position `i` depends on **two things**:
1. The 11-mer surrounding that base (the sequence context)
2. Whether that base is methylated (and what type)

These two pieces of information are compressed into two integers:

| Integer | Name | Range | Meaning |
|---|---|---|---|
| `kmer_id` | 22-bit integer | `[0, 4_194_303]` | The 11-mer as a base-4 number |
| `meth_id` | Integer | `{0, 1, 2, 3}` | `none=0`, `m6A=1`, `m4C=2`, `m5C=3` |

### Key functions to read

```python
# encoding.py
encode_kmer("ACGTACGTACG") -> int    # 11-char string → 22-bit integer
decode_kmer(1234567)       -> str    # 22-bit integer → 11-char string

# motifs.py
iupac_to_re("RGATCY")  -> "[AG]GAT[CT][CT]"   # IUPAC → regex
scan_sequence(seq, motifs) -> np.int8[:]       # per-base meth_id array
```

### Mental model

```
Read:   A C G A T C A G T T A ...
kmer:   [ACGATCAGTTA]             # centered on position i
meth:   0 0 0 1 0 0 0 0 0 0 0    # 1 = m6A at position 3 (GATC motif)

→ key = (encode_kmer("ACGATCAGTTA"), 1)
→ training sample: IPD=12, PW=8
```

**Start here if you want to understand:** why k-mer size is 11, how IUPAC codes work,
how a read is scanned base by base.

---

## Step 2 — Extraction (BAM → .pkl)

**File:** [kinsim/common/extract.py](kinsim/common/extract.py)

### What it does

Reads a PacBio BAM, slides an 11-mer window across every read, records `(IPD, PW)` values
in a dictionary keyed by `(kmer_id, meth_id)`.

```python
# Pseudocode
for read in bam:
    for i in range(5, len(read) - 5):          # skip N-context edges
        kmer_id = encode_kmer(read.seq[i-5:i+6])
        meth_id = scan_sequence(read.seq, motifs)[i]
        key = (kmer_id, meth_id)
        samples[key].append((fi[i], fp[i]))     # IPD, PW from fi/fp tags
```

Reservoir sampling caps each key at `max_samples_per_key=10_000` to bound memory.

### Output format

```python
# shard.pkl content
{
    (kmer_id, meth_id): np.ndarray(N, 2),   # columns: [IPD, PW] raw uint8
    (kmer_id, meth_id): np.ndarray(N, 2),
    ...
    "__meta__": {                            # provenance (skipped by datasets)
        "source_bam": "/data/Ecoli.bam",
        "motifs": "m6A,GATC,1",
        "created": "2025-01-15T10:30:00"
    }
}
```

### Key functions

```python
validate_bam_kinetics(bam_path)               # fail-fast: checks fi/fp tags
extract_samples_from_bam(bam_path, motif_str, output_pkl)
extract_from_manifest_task(manifest, task_idx, output_dir)  # SLURM array entry point
merge_shards(input_dir, output_path)          # concatenate shards per key
```

**Read `extract_samples_from_bam()` if you want to understand** the inner loop, reservoir
sampling, and how `fi`/`fp` BAM tags become numpy arrays.

---

## Step 3 — Dataset (raw values → training tensors)

**File:** [kinsim/common/dataset.py](kinsim/common/dataset.py)

### The signal space problem

Raw IPD/PW values are integers in `[0, 255]`. They are **right-skewed** (most reads cluster
near 0, rare long pauses reach 255). Training a neural network on raw integers works poorly.

**Solution:** `log1p` transform.

```python
log_transform(x)     = log(1 + x)       # [0, 255] → [0, ~5.5]  symmetric, float
inv_log_transform(x) = clamp(expm1(x), 0, 255)  # back to [0, 255] uint8
```

The transform is applied **once at dataset load time**, stored in the dataset, not in the
`.pkl` file. This means `.pkl` files always contain raw values and can be inspected directly.

### KmerSignalDataset

```python
ds = KmerSignalDataset("master_data.pkl")
kmer_id, meth_id, signal = ds[0]
# kmer_id: LongTensor scalar
# meth_id: LongTensor scalar
# signal:  FloatTensor([log_ipd, log_pw])  — log1p space
```

`__len__` = total number of (IPD, PW) observation pairs across all keys.

### Non-tuple keys

The dataset explicitly skips non-tuple keys (like `"__meta__"`):
```python
for key, samples in data_dict.items():
    if not isinstance(key, tuple):
        continue  # skip "__meta__" and any other metadata
    kmer_id, meth_id = key
    ...
```

**Read `dataset.py` if you want to understand** the log1p transform, why raw values are
kept in `.pkl`, and how the DataLoader receives batches.

---

## Step 4 — Model Architecture

**File:** [kinsim/models/mlp/model.py](kinsim/models/mlp/model.py)

### Architecture

```
kmer_id  ──► Embedding(4_194_304, kmer_embed_dim=64) ──►  kmer_vec  (B, 64)
meth_id  ──► Embedding(4, 8)                         ──►  meth_vec  (B,  8)
                                                               │
                                                    concat → (B, 72)
                                                               │
                                                    Linear(72, 128) + ReLU
                                                    Linear(128, 128) + ReLU
                                                               │
                                                    Linear(128, 4)
                                                               │
                              ┌────────────────────────────────┘
                              │
                    [:, 0:2] = [μ_ipd, μ_pw]      in log1p space
                    [:, 2:4] = [log_σ_ipd, log_σ_pw]
```

### Three output methods

```python
model.forward(kmer_ids, meth_ids)
# → (B, 4): raw network outputs in log1p space

model.predict_mean(kmer_ids, meth_ids)
# → (B, 2): deterministic, inv_log_transform applied → [0, 255]

model.sample(kmer_ids, meth_ids)
# → (B, 2): stochastic (N(μ, σ²) sample) → inv_log_transform → [0, 255]
```

The key distinction: `predict_mean()` always returns the same result; `sample()` is
stochastic. In production, `sample()` is used by default (controlled by `--deterministic`).

### Why Gaussian NLL?

The model doesn't just predict μ — it also predicts σ. This is important because:
- IPD distributions are **heteroscedastic**: uncertainty varies by (kmer, meth) context
- A context near a strongly-modified motif may have high variance (multi-modal distribution)
- Predicting σ lets the model communicate uncertainty during generation

**Read `model.py` if you want to understand** the embedding dimensions, the 4-output head,
why `log_σ` is clamped to `[-6, 3]`, and the `predict_mean` vs `sample` distinction.

---

## Step 5 — Training Loop

**File:** [kinsim/models/mlp/train.py](kinsim/models/mlp/train.py)

### Loss function

```python
# Gaussian Negative Log-Likelihood (gnll)
log_σ = clamp(params[:, 2:4], -6, 3)
σ²    = exp(2 * log_σ)
μ     = params[:, 0:2]

loss = mean(0.5 * (2 * log_σ + (target - μ)² / σ²))
```

This jointly optimizes μ (push toward the data mean) and σ (shrink uncertainty where
the data is consistent; allow higher uncertainty where it is noisy).

### Training decisions at a glance

| Decision | Value | Why |
|---|---|---|
| Optimizer | Adam | Standard for MLPs |
| Betas | (0.9, 0.999) | Adam defaults |
| LR scheduler | ReduceLROnPlateau | Halves LR when val_loss stalls |
| Patience | 5 epochs | Avoids premature LR drops |
| Val split | 10% | Held out before any training |
| Default loss | gnll | Jointly learns μ and σ |
| Batch size | 4096 | Large — IPD/PW are cheap to compute |

### Checkpoint structure

```python
torch.save({
    'epoch':     epoch + 1,
    'step':      global_step,
    'model':     model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),   # always saved for resume
}, "checkpoint_epoch50.pt")
```

`model_config.json` is written **before the first epoch** (architecture hyperparameters only):
```json
{"kmer_embed_dim": 64, "hidden_dim": 128, "meth_proj_dim": 8}
```

### YAML config and CLI precedence

`main()` supports two calling modes:
1. Positional: `kinsim train master_data.pkl checkpoints/ --model mlp --epochs 50`
2. YAML: `kinsim train --model mlp --config config_mlp.yaml [--epochs 100]`

CLI flags always win. This is implemented with `_get(cli_val, yaml_key, default)`:
```python
epochs = _get(args.epochs, "epochs", 50)  # CLI > YAML > default
```

**Read `train.py` if you want to understand** the Gaussian NLL loss, the LR scheduler
behaviour, how checkpoints are saved and resumed, and the YAML config merge logic.

---

## Step 6 — Signal Generation (model → BAM)

**File:** [kinsim/models/mlp/generate.py](kinsim/models/mlp/generate.py)

### What it does

1. Load reference genome → pre-scan methylation positions → `meth_map[ref_name][ref_pos]`
2. Parse MAF alignment file → for each simulated read: `(ref_name, ref_start, strand, length)`
3. For each batch of reads: compute `kmer_ids` + look up `meth_ids` from `meth_map`
4. Call `model.sample(kmer_ids, meth_ids)` → `(B, 2)` signals in `[0, 255]`
5. Write BAM record: `flag=4`, `fi:B:C` (IPD), `fp:B:C` (PW)

### Loading the model

```python
# Always reads model_config.json — hard-errors if missing
cfg = json.load(open(checkpoint_dir / "model_config.json"))
model = MLPPredictor(
    kmer_embed_dim = cfg["kmer_embed_dim"],
    hidden_dim     = cfg["hidden_dim"],
)
state = torch.load("checkpoint_epoch50.pt")
model.load_state_dict(state["model"])
model.eval()
```

### The methylation map (O(1) lookup)

The reference is pre-scanned **once** before processing any reads:
```python
# build_reference_meth_map returns:
meth_map[ref_name] = np.int8[ref_length]  # value = meth_id per position
```
This avoids re-scanning the reference for every read.

### N-context positions

The first and last 5 bases of each read (where the 11-mer window falls off the edge of the
reference alignment) cannot have a real k-mer. They receive signal `= 1` (PacBio convention
for "minimal valid signal", not "no data").

**Read `generate.py` if you want to understand** the two auto-detected input modes
(directory vs single-genome), how the MAF file is parsed, batch inference, and
the BAM tag encoding.

---

---

## Step 7 — Evaluation Metrics

**Files:** [kinsim/models/mlp/train.py](kinsim/models/mlp/train.py) (`_compute_metrics`),
[kinsim/models/mlp/evaluate.py](kinsim/models/mlp/evaluate.py)

### The three questions a probabilistic model must answer

| Question | Metric | Target |
|---|---|---|
| Does μ track the real signal? | Pearson r | > 0.9 |
| How far off is μ on average? | MAE (log1p space) | < 0.1 is excellent |
| Does σ actually cover the real noise? | 2σ Calibration | ≈ 95 % |

### Calibration — the metric specific to probabilistic models

A purely deterministic model (MSE loss, μ only) can achieve low MAE but tells you nothing about uncertainty. KinSim's Gaussian NLL loss trains both μ and σ, so we can ask:

> "For a given context, does the model's confidence interval actually contain the data?"

The 2σ calibration check:
```python
in_2sigma = |actual - μ| ≤ 2σ
coverage  = mean(in_2sigma)   # expected: 95.4% for a perfect Gaussian
```

**Interpreting the number:**
```
coverage < 90%  →  σ too small — overconfident, underestimates PacBio noise
coverage ≈ 95%  →  well-calibrated — model has learned the correct noise level
coverage > 99%  →  σ too large — over-dispersed, signals will look too "noisy"
```

The training loop reports this every epoch:
```
Epoch [ 50/50]  train_loss=0.3821  val_mse=(0.0412, 0.0389)
                pearson=(0.923, 0.911)  calib=(94.7%, 95.1%)  lr=5.00e-04
```

### Heteroscedasticity check

A key feature of the MLP is that σ is **not constant** — it varies by context. After training, you should see:

```
mean σ  IPD:  0.18     ← average uncertainty across all contexts
median σ IPD: 0.12     ← most contexts are low-uncertainty
```

If `mean σ ≈ median σ`, the model is behaving like a constant-variance model. If `mean σ >> median σ`, the model correctly learned that some contexts (e.g., m6A in GATC) are much noisier than background.

### K-mer distribution plot — the visual sanity check

```bash
kinsim mlp evaluate checkpoints_mlp/ master_data.pkl \
    --kmer GGATCCTGCAT --meth m6A --plot gatc_m6A.png
```

The plot shows:
- **Blue histogram** — actual IPD/PW values from the training data (log1p space)
- **Red curve** — predicted N(μ, σ²) PDF
- **Orange shaded band** — μ ± 2σ (should contain ~95% of the histogram)
- **Corner annotation** — actual 2σ coverage for this specific context

A well-trained model on a methylated k-mer (m6A) should show:
- μ shifted RIGHT compared to unmethylated contexts (longer IPD pause)
- σ larger than for unmethylated contexts (more noise at modification sites)
- The red curve closely following the blue histogram shape

### Running the full evaluation report

```bash
# Full report (Pearson, MAE, calibration at 1σ/2σ/3σ, σ statistics)
kinsim mlp evaluate checkpoints_mlp/ master_data.pkl

# Save to file
kinsim mlp evaluate checkpoints_mlp/ master_data.pkl \
    --output my_eval_results.txt

# Example output:
# ============================================================
#   KinSim MLP — Evaluation Report
# ============================================================
#   Contexts evaluated : 12,847
#
#   ── Mean prediction quality (log1p space) ──────────────
#   MAE   IPD / PW  :  0.0823  /  0.0791
#   Pearson IPD / PW:  0.9134  /  0.9087
#
#   ── Calibration coverage (% within nσ of μ) ─────────────
#   Coverage     IPD     PW    Expected
#   1σ (68%)   67.8%   68.2%    68.3%
#   2σ (95%)   94.7%   95.1%    95.4%   ← target
#   3σ (99%)   99.4%   99.5%    99.7%
# ============================================================
```

**Read `evaluate.py` if you want to understand** the full calibration sweep, the matplotlib visualization, and the `_load_model()` function that reconstructs the architecture from `model_config.json`.

---

## Quick Reference

### Where does each concept live?

| Concept | File | Key symbol |
|---|---|---|
| 11-mer encoding | `kinsim/encoding.py` | `encode_kmer`, `decode_kmer` |
| Methylation IDs | `kinsim/encoding.py` | `METH_IDS = {none:0, m6A:1, ...}` |
| IUPAC motif parsing | `kinsim/motifs.py` | `parse_motifs`, `scan_sequence` |
| Reference meth map | `kinsim/motifs.py` | `build_reference_meth_map` |
| BAM → .pkl extraction | `kinsim/common/extract.py` | `extract_samples_from_bam` |
| Manifest CSV support | `kinsim/common/extract.py` | `extract_from_manifest_task` |
| log1p transform | `kinsim/common/dataset.py` | `log_transform`, `inv_log_transform` |
| Dataset | `kinsim/common/dataset.py` | `KmerSignalDataset` |
| Model architecture | `kinsim/models/mlp/model.py` | `MLPPredictor` |
| Training loop | `kinsim/models/mlp/train.py` | `main()` |
| Signal generation | `kinsim/models/mlp/generate.py` | `main()` |
| CLI config/logging | `kinsim/config.py` | `load_manifest`, `load_yaml_config` |

### Signal space cheat-sheet

```
.pkl storage   →  raw uint8 [0, 255]          (never log-transformed)
Dataset output →  log1p float [0, ~5.5]       (log_transform applied at load time)
Model input    →  log1p float [0, ~5.5]
Model output   →  log1p float (μ, σ)
BAM output     →  raw uint8 [0, 255]          (inv_log_transform applied at inference)
```

### Invariant tests (what to verify when debugging)

```python
# 1. Encoding round-trip
assert decode_kmer(encode_kmer("ACGTACGTACG")) == "ACGTACGTACG"

# 2. Transform round-trip
x = torch.tensor([0.0, 128.0, 255.0])
assert torch.allclose(inv_log_transform(log_transform(x)), x, atol=1e-4)

# 3. Model determinism vs stochasticity
torch.manual_seed(0)
out1 = model.predict_mean(kmer_ids, meth_ids)
out2 = model.predict_mean(kmer_ids, meth_ids)
assert torch.allclose(out1, out2)    # deterministic

out3 = model.sample(kmer_ids, meth_ids)
out4 = model.sample(kmer_ids, meth_ids)
assert not torch.allclose(out3, out4)  # stochastic

# 4. Output range
signals = model.sample(kmer_ids, meth_ids)
assert signals.min() >= 0 and signals.max() <= 255
```
