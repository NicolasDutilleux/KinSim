# kinsim_NN — Architecture Summary (May 2026)

End-to-end pipeline for simulating PacBio HiFi kinetic signals (IPD/PW) conditional on per-base methylation context, for metagenomic binning research.

---

## 1. Problem Statement

**Goal**: generate realistic PacBio HiFi kinetic tags (`fi`/`fp`/`ri`/`rp`, i.e. IPD/PW per strand) for a given reference genome + motif methylation map, such that downstream PacBio tools (ipdSummary, pbmotifmaker) detect the same methylation patterns as on real data.

**Why**: serves as a controlled benchmark for metagenomic methylation-based binning algorithms — we can inject known methylation landscapes into simulated reads and measure how well binners recover species identity from methylation signatures.

---

## 2. Pipeline Overview

```
Reference genomes + motifs.gff (PacBio motifmaker output)
        │
        ▼  kinsim_NN extract
shards/<strain>_shard.pkl  (3-category labeled training data)
        │
        ▼  kinsim_NN train
checkpoints/best_G.pt  (transformer cGAN, WGAN-GP)
        │
        ▼  kinsim_NN generate (precompute + multiprocess)
SIM.bam  (unmapped HiFi with generated fi/fp/ri/rp)
        │
        ▼  ccs-kinetics-bystrandify → pbmm2 align → ipdSummary → pbmotifmaker
SIM motifs.csv  (compare vs real motifs.csv)
```

---

## 3. Data Extraction (`kinsim_NN extract`)

### 3.1 Source data
- **Aligned bystrandified BAMs** (2 records per ZMW: `/ccs/fwd` and `/ccs/rev`, each with `ip`/`pw` kinetic tags)
- **Reference FASTA** (assembly per strain)
- **Methylation labels**: `motifs.gff` from `pbmotifmaker reprocess` (filtered to motif-confirmed lines via `require_motif: true` in GFFLabeler)

### 3.2 Three-category labeling (v3-style)
For each motif-confirmed methylation position `p` of type `T` (m6A, m4C, m5C), the extract expands into emission candidates at offsets `k ∈ [0, near_meth_max_dist=10]`:

| Category | Condition | Meaning |
|---|---|---|
| **SLOWED** (1) | `k ∈ T.signal_offsets` | Position where IPD slowdown is expected per the SP3-C3 model |
| **NEAR_METH** (2) | `k ∉ T.signal_offsets` | Position close to a meth but NOT at a signature offset (negative control) |
| **BASELINE** (0) | far from any methylation | True baseline kinetics |

**Signal offsets per meth type** (from PacBio SP3-C3):
- m6A: `[0, 5]` (peak at center + 5 bp downstream)
- m4C: `[0]` (peak at center only)
- m5C: `[2, 6]` (peaks at +2 and +6, NOT at center)

**Conflict resolution**: SLOWED beats NEAR_METH; last writer wins within same category.

### 3.3 Window construction
- **K = 21 bp window** (±10 around the emission position)
- 4 channels: `IPD_fwd, PW_fwd, IPD_rev, PW_rev` (bilateral)
- Strand routing follows kinsim2 convention (verified against real bystrandified BAMs)

### 3.4 Baseline sampling
- Random positions `≥ baseline_min_dist=20` from ANY high-QV ipdSummary call
- Uses `avoid_gff_pattern` (full motifs.gff, all high-QV positions) → baseline pool avoids both motif-confirmed AND ambiguous positions
- ~50k baseline positions per strain

### 3.5 Reads cap
- `reads_cap_per_position = 20` (reservoir sampling)
- Each emission position contributes up to 20 ZMW samples

### 3.6 Total corpus (after full extraction)
- **51 Strepto strains** (Revio P2-C2)
- **13 Vega strains** (Revio P1-C1, weaker SP3-C3 model match)
- **~5.6 billion samples** total (~10 GB shards each, ~600 GB total)
- Distribution: ~12% SLOWED, ~65% NEAR_METH, ~23% BASELINE

### 3.7 Shard schema (`kinsim_NN-2`)
Per-sample arrays in pickle:
- `base_fwd[K]`: A/C/G/T codes
- `meth_fwd[K]`, `meth_rev[K]`: per-position meth IDs (per strand)
- `signal[K, 4]`: uint8 PacBio codec
- `category` (uint8): 0/1/2
- `parent_meth` (uint8), `parent_offset` (int8): which methylation produced this sample, at what offset
- `ref_id`, `ref_pos`, `strand`, `zmw` (metadata)

---

## 4. Model Architecture (`kinsim_NN/models/`)

### 4.1 Conditional GAN (cGAN) with WGAN-GP loss

**Generator** (`TransformerGenerator`):
- Architecture: per-position embedding + positional embedding + AdaLN-Zero FiLM (meth conditioning) + Transformer encoder backbone + linear readout
- Hidden dim `d_model = 192`
- `n_layers = 6`, `n_heads = 6`
- Noise dim `z_dim = 64` (latent Gaussian for stochasticity)
- Output: `(B, K=21, 4)` predicted IPD/PW values in `log1p(frames)` space
- **~4.17 M parameters**

**Discriminator** (`TransformerDiscriminator`):
- Same backbone as generator but smaller: `d_model = 128`, `n_layers = 4`, `n_heads = 4`
- Spectral normalization on linear layers
- Output: scalar realness score
- **~0.80 M parameters**

### 4.2 Conditioning inputs (FiLM-modulated)
- `base_fwd`, `base_rev`: per-position one-hot bases (K, 4)
- `meth_fwd`, `meth_rev`: per-position meth labels one-hot (K, M=4)
- `z`: noise vector (B, z_dim)

### 4.3 Loss
- **WGAN-GP**: standard Wasserstein loss + gradient penalty (`λ = 10`)
- `n_critic = 5` (5 D updates per G update)
- TTUR (two-time-scale update): `lr_d = 4e-4`, `lr_g = 1e-4`
- Adam optimizer (`β1 = 0.0`, `β2 = 0.9`, WGAN-GP standard)

---

## 5. Training (`kinsim_NN train`)

### 5.1 Setup
- **Corpus**: 195M samples across 57 train strains + 4 held-out test strains (`bc2034`, `bc2045`, `bc2048`, `bc2082`)
- **Batch size**: 256
- **Total steps**: 200k planned, cancelled at 112k (model plateaued)
- **Hardware**: 1 × A100 GPU
- **Wall time at cancel**: ~11h

### 5.2 Metrics
- Train losses: D, G, gradient penalty (logged every 100 steps)
- Eval every 5000 steps: Wasserstein-1 distance on held-out test strains
  - Overall (`w1_overall`)
  - Per meth type (`w1_meth0/1/2/3` for none/m6A/m4C/m5C)
  - Per category (`w1_baseline/slowed/near_meth`)

### 5.3 Results

**Best checkpoint**: step 90k, `w1_overall = 1.32`

W1 in bytes (PacBio uint8 codec):

| Metric | Best W1 | % error vs mean IPD (~40) |
|---|---|---|
| overall | 1.32 | 3.3% |
| baseline | 1.49 | 3.7% |
| slowed | 1.91 | 4.8% |
| near_meth | 1.57 | 3.9% |
| m4C (single peak) | 2.09 | 5.2% |
| m6A (bimodal [0, 5]) | 4.62 | 11.5% |
| m5C (bimodal [2, 6]) | 4.36 | 10.9% |

**Interpretation**: model captures the overall IPD distribution well. Bimodal meth types (m6A, m5C) are harder due to two-peak signature structure.

---

## 6. Generation (`kinsim_NN generate`)

### 6.1 Inputs
- Input BAM (raw HiFi, ideally aligned for the fast path)
- Reference FASTA
- Trained checkpoint directory (`best_G.pt` + `model_config.json`)
- Motifs CSV (PacBio format: motif, centerPos, modType, fraction)

### 6.2 Algorithm — precompute path (28-90× faster than naive)

**Step 1 — Global Bernoulli draw on motif sites**
- For each motif occurrence in the reference, Bernoulli(`fraction`) → marked methylated or not
- One decision per genomic site, shared by all reads (matches PacBio fraction semantic)

**Step 2 — Precompute kinetics map per contig**
- Walk reference at **stride K = 21** (each inference covers K positions)
- For each window center, predict `(K, 4)` via generator
- Repeat for `n_z_samples = 32` independent noise samples per position
- Output: `kin_map[L_contig, 4 channels, 32 z_samples]` uint8
- Memory: ~1 GB per 8 Mbp contig
- Time: ~2-5 min total

**Step 3 — Multiprocess read processing**
- Master forks 8 workers (workers inherit `kin_map` via copy-on-write — zero-copy)
- For each read, master extracts picklable descriptor (qlen, ref_start, cigar, is_rev) and ships to worker pool
- Workers walk CIGAR (skip pysam overhead) → vectorized lookup `kin_map[r, :, random_z]`
- Per-position random z_idx → high apparent variance
- Strand routing matches training convention:
  - `is_rev=False`: `fi←ipd_rev(ch2), fp←pw_rev(ch3), ri←ipd_fwd(ch0), rp←pw_fwd(ch1)`
  - `is_rev=True`: `fi←ipd_fwd(ch0), fp←pw_fwd(ch1), ri←ipd_rev(ch2), rp←pw_rev(ch3)`
- Master writes BAM in submission order (preserves read order)

### 6.3 Performance benchmark (bc2034 test holdout)
- 124,446 input reads (3.2 GB stripped BAM)
- Precompute: 2 min (3M GPU inferences batched at 256)
- Multiprocess read fill: 7 min (8 workers)
- **Total: 9 minutes** vs 14h baseline (per-position GPU calls)
- **~93× speedup**

---

## 7. Validate Chain

```
SIM.bam (unaligned HiFi with fi/fp/ri/rp)
    │
    ▼ ccs-kinetics-bystrandify
SIM_bystr.bam (2 records/ZMW with ip/pw)
    │
    ▼ pbmm2 align (--preset CCS --sort)
SIM_aligned.bam + .pbi
    │
    ▼ ipdSummary SP3-C3 (m6A + m4C identification)
SIM_ipdsummary.gff + .csv (per-position methylation calls)
    │
    ▼ pbmotifmaker find
SIM_motifs.csv (discovered motif patterns + fractions)
    │
    ▼ pbmotifmaker reprocess
SIM_motifs.gff (annotated GFF)
```

---

## 8. Current Results — bc2034 Validation

**Real bc2034 motifs.csv** (11 motifs, 4 with high confidence):
- CTGAAG/CTTCAG (m6A, 100% fraction, IPDRatio ~5) — Type I R-M system pair
- CCGANNNNNNNCTCG/CGAGNNNNNNNTCGG (m6A, 100% fraction, IPDRatio ~4) — symmetric Type II
- VNCNK (m4C, 4.6% fraction, 129k detections) — weak abundant
- 6 other modified_base motifs at fractions 0.2-0.6

**SIM motifs.csv** (only 3 motifs, all weak):
- TVVVB (modified_base, 0.5% fraction) — generic AT-rich
- CTCTB (m4C, 3% fraction)
- HCDWTBND (m4C, 1.5% fraction)

**Status**: ❌ **SIM does not reproduce real motifs**.

The model generates IPD distributions matching average kinetics (W1=1.32 on test holdout), but per-position predictions are not consistent enough to reproduce strong, structured patterns detectable by ipdSummary + pbmotifmaker.

---

## 9. Limitations Identified

1. **Per-position random z dilutes motif signal**: each base of each read picks an independent z_idx from the 32 precomputed samples. For ipdSummary to detect a motif, it needs CONSISTENT slowdown across many reads at the same position. Per-base random z gives the right MEAN but high variance — ipdSummary's statistical tests don't find the consensus.

2. **Bimodal meth signatures (m6A, m5C) harder**: model captures single-peak m4C well (W1 ~2) but bimodal m6A/m5C signatures plateau at W1 ~4-5.

3. **Training plateaued at step 90k** (W1=1.32) and never improved significantly past step 50k. Suggests architecture capacity limit for this data size + 57 strain diversity.

4. **No per-strain conditioning**: model averages across all 57 train strains. Per-strain variance (chemistry version, polymerase batch effects) is mixed into the noise.

5. **GAN training instability**: WGAN-GP eval W1 oscillates between 1.3 and 5 even after convergence. Not a divergence (D losses stable) but indicates the equilibrium is fragile.

---

## 10. Proposed Improvements (v2)

### Generation-level (cheap, no retraining)
1. **Per-read z instead of per-base z**: pick ONE z_idx for the entire read → all bases share the same noise realization → preserves the per-molecule coherence ipdSummary expects. Single CLI flag swap.
2. **Smaller `n_z_samples` (8 instead of 32)** with per-read z: less apparent variance but more meaningful per-read consistency.
3. **Increase Bernoulli fraction**: bias toward methylated-true to amplify motif signal.

### Training-level (requires retraining)
4. **Per-strain embedding**: add a strain ID embedding → model can capture per-strain variance instead of dumping it into z noise.
5. **Longer training**: 500k+ steps with cosine LR decay near end.
6. **Bigger model**: d_model 192 → 256, n_layers 6 → 8. ~10M params for the generator.
7. **Better cGAN tricks**: relativistic discriminator (RaSGAN), R1 regularization, EMA generator.
8. **Per-position L2 auxiliary loss**: add MSE on center-position IPD alongside WGAN-GP to enforce sharper per-position predictions.

### Architecture-level
9. **Diffusion model**: replace WGAN-GP with a denoising diffusion process. Diffusion models match per-position distributions much better than GANs at the cost of slower sampling.
10. **Sequence-to-sequence transformer**: instead of per-window prediction, model the entire read as a sequence and predict IPD/PW as autoregressive tokens. Captures longer-range correlations.

---

## 11. Code Organization

```
kinsim_NN/
├── __main__.py          CLI dispatcher (extract, train, generate, evaluate, analyze)
├── extract.py           Aligned bystrandified BAM → shard.pkl (3-cat labels)
├── train.py             WGAN-GP training loop (Beta-NLL also supported as fallback)
├── generate.py          Precompute + multiprocess generation
├── analyze.py           HTML dashboard for shard QC (per-category distributions)
├── evaluate.py          Held-out W1 metrics
├── data/
│   ├── dataset.py       PyTorch IterableDataset (multi-shard streaming)
│   └── shard.py         Schema (kinsim_NN-2) + read/write helpers
├── models/
│   ├── generator.py     TransformerGenerator (4.17M params)
│   └── discriminator.py TransformerDiscriminator (0.80M params)
├── labelers/
│   ├── gff.py           GFFLabeler (motif=-required mode)
│   ├── jasmine_mm_ml.py jasmine 5mC labeler
│   └── registry.py      Plugin registry
└── utils/
    ├── config.py        YAML schema + frozen dataclasses
    ├── bam_io.py        Bystrandified pair walking + bilateral extraction
    ├── pacbio_codec.py  uint8 ↔ frames conversion
    └── encoding.py      Base encoding (single source of truth)

kinsim_nn_config.yaml    Single source of biology truth (signal_offsets, caps, etc.)
slurm_kinsim/            HPC SLURM scripts (prep, callers, ml, dataset orchestrators)
```

---

## 12. Numbers Summary

| Quantity | Value |
|---|---|
| Training strains | 57 (51 Strepto + 13 Vega - 4 holdout - some excluded) |
| Test holdout strains | 4 (bc2034, bc2045, bc2048, bc2082) |
| Total training samples | ~195 M (after motif filter) |
| Window K | 21 bp |
| Channels per sample | 4 (IPD_fwd, PW_fwd, IPD_rev, PW_rev) |
| Methylation types | 4 (none, m6A, m4C, m5C) |
| Generator params | 4.17 M |
| Discriminator params | 0.80 M |
| Best W1 | 1.32 (step 90k, ~3.3% error on IPD distribution) |
| Generate time (bc2034, 124k reads) | 9 min (93× faster than baseline) |
| ipdSummary on SIM | 3h 55min (200x coverage) |
| Real bc2034 motifs detected | 11 (4 high-confidence) |
| SIM bc2034 motifs detected | 3 (all weak, no overlap with real) |
