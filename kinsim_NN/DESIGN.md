# kinsim_NN — design document

This document records the architectural decisions for `kinsim_NN`, a
transformer-based conditional GAN that replaces the parametric Beta-NLL
of `kinsim` / `kinsim2` with an adversarial loss to avoid σ-collapse and
class-collapse pathologies observed in v12_run3.

Authoritative spec. Anything not stated here is open for future
discussion; anything stated here is committed unless we revisit.

## 1. Problem & motivation

`kinsim` v12_run3 (Beta-NLL ConvPredictor, single strand) and `kinsim2`
(Beta-NLL bilateral) both suffer from:

1. **σ-inflation:** σ_baseline ≈ 18 in uint8 space vs. real PacBio
   σ_baseline ≈ 5. The model preferred inflating variance over fitting
   μ. Beta-NLL's β=0.5 was insufficient.
2. **m4C blind spot:** model output for m4C in `predict_kmers` is
   ×0.998 vs real corpus ×1.93 — the m4C dimension of FiLM
   conditioning effectively collapsed to identity.
3. **Downstream consequence:** σ inflation pollutes every position →
   ipdSummary calls 70% of positions as modified → motifmaker drowns
   in noise → 0 motifs detected at threshold 0.7 on 3 of 4 validate
   strains.

The root cause is *parametric loss* on noisy data. cGAN sidesteps this
by replacing the (μ, σ) prediction with sampling from a learned
distribution adversarially matched to real per-read kinetics.

## 2. Window geometry

- **Reference window:** ±10 around the methylated center → **K = 21
  reference positions**.
- **Bilateral signal:** at each ref position we extract 4 channels
  (`IPD_fwd`, `PW_fwd`, `IPD_rev`, `PW_rev`). Equivalent to "42
  position-strand slots" in the user's intuition: 21 positions × 2
  strands.
- **Transformer tokens:** K = 21. Each token represents one ref
  position; the 4 signal channels live as token features (D) or token
  outputs (G).

## 3. Methylation conditioning

Per-position, per-strand methylation is encoded as **two integer
arrays of length K=21**:

```
meth_fwd[K]   uint8   meth_id at this position on forward strand
meth_rev[K]   uint8   meth_id at this position on reverse strand
```

Meth_id ∈ {0=none, 1=m6A, 2=m4C, 3=m5C}. Default M = 4. Extensible via
`kinsim_nn_config.yaml`:

```yaml
methylation_types:
  none: {id: 0}
  m6A:  {id: 1, modified_base: A, label_sources: [gff]}
  m4C:  {id: 2, modified_base: C, label_sources: [gff]}
  m5C:  {id: 3, modified_base: C, label_sources: [gff, jasmine_mm_ml]}
  # adding a new type is trivial:
  # m1A: {id: 4, modified_base: A, label_sources: [custom_caller]}
```

`label_sources` references entries in the `labelers` section. New
sources are added by writing a `BaseLabeler` subclass and registering
it via the `@register` decorator (see `labelers/registry.py`).

**Critical:** `meth_fwd` and `meth_rev` are independent. GATC palindrome
→ m6A on fwd at position 3 AND m6A on rev at position 2 are two
biologically distinct events.

The model receives them as separate one-hot tensors (`B, K, M`) inside
each token's feature vector, then dynamically expanded by `Dataset` at
training time. Stored compactly in shards (uint8 per position).

## 4. Token structure

### Generator (input)

```
Token i (input to G) =
  base_fwd[4 one-hot]
  base_rev[4 one-hot]     ← redundant with base_fwd but free; aids learning
  meth_fwd[M one-hot]
  meth_rev[M one-hot]
  position_embed[16 learned]
  → projected to d_model_g = 192

Latent noise z (z_dim=64) feeds two-step conditioning vector:
    cond_emb = MLP(z) + cond_pool_proj(mean_pool(input_tokens))
This `cond_emb` is what AdaLN-Zero modulates with in every transformer
block. The pooled condition term is a deliberate deviation from a
"pure z" AdaLN: it helps stabilise training by giving the modulation
some awareness of the conditioning context. AdaLN's zero-initialisation
keeps the early-training behaviour identical to vanilla self-attention
either way.

Token i (output of G) =
  signal[4 channels in log1p(frames) space]
  → 4 floats per token
```

### Discriminator (input)

```
Token i (input to D) =
  base_fwd[4]
  base_rev[4]
  meth_fwd[M]
  meth_rev[M]
  signal[4]               ← from real or fake
  position_embed[16]
  → projected to d_model_d = 128

Output of D:
  Critic score (scalar) via CLS-token pooling
```

Plus a CLS token prepended to the sequence (D only) so attention can
aggregate global info into one slot for the final scalar.

## 5. Bilateral BAM extraction (no raw HiFi re-alignment)

We have bystrandified+aligned BAMs (2 records per ZMW: `ccs/fwd` and
`ccs/rev`, each with `ip` + `pw`), NOT raw HiFi aligned BAMs (which
would have `fi/fp/ri/rp` in 1 record). Extracting 4 channels requires
pairing the two records per ZMW:

```
For each ZMW with both fwd and rev records:
  For each ref position p in window:
    IPD_fwd[p] = fwd.ip[map_q_to_r(fwd, p)]   if fwd covers p
    PW_fwd[p]  = fwd.pw[map_q_to_r(fwd, p)]
    IPD_rev[p] = rev.ip[map_q_to_r(rev, p)]   if rev covers p
    PW_rev[p]  = rev.pw[map_q_to_r(rev, p)]
  Emit one sample per ZMW (one per pair).
```

ZMW pairing is done by suffix-stripping the read name:
`m84151_240303_022646_s4/106171215/ccs/fwd` and `…/ccs/rev` share the
prefix up to `/ccs/`. Reads without a pair are skipped (rare, only at
chromosome ends).

## 6. Frame conversion (PacBio codec)

PacBio stores IPD/PW as `uint8` in BAMs via a non-linear codec:

| byte range | frame range | step |
|---|---|---|
| 0..63 | 0..63 | 1 |
| 64..127 | 64..190 | 2 |
| 128..191 | 192..444 | 4 |
| 192..255 | 448..952 | 8 |

Decode formula:

```
if b < 64:   frames = b
if b < 128:  frames = 64 + 2*(b - 64)
if b < 192:  frames = 192 + 4*(b - 128)
else:        frames = 448 + 8*(b - 192)
```

Stored compactly in shards as uint8 (1 byte/value). Converted to
`frames` (float32) at `Dataset.__getitem__()` time via a 256-entry
lookup table. **Training is done in `log1p(frames)` space.** At
generation, model output → exp - 1 → nearest-bucket lookup → uint8 for
BAM tags.

## 7. Labels: where they come from (modular)

The extract step takes a list of labelers in order. Each labeler emits
`(ref_id, ref_pos, meth_id, strand)` records over a reference. Sources
unioned, conflicts resolved by precedence (first labeler wins —
configurable).

Default chain:

1. **`GFFLabeler`**: parse the per-strain `motifs.gff` from
   `pbmotifmaker reprocess`. Filter by `score (QV) ≥ qv_threshold`.
   Maps GFF `type` column (`m6A`, `m4C`, `modified_base`) to meth_id.
   `modified_base` is treated by default as m5C if `--treat-modbase-as
   m5C` (default true since modkit/jasmine 5mC results merge here).

2. **`JasmineMMMLLabeler`**: parse `MM`/`ML` tags from the jasmine 5mC
   BAM (used when 5mC calls are not in the GFF). Threshold by
   `ml_threshold` (0-255, default 200 = ~78% confidence).

Adding a new labeler:

```python
# kinsim_NN/labelers/my_labeler.py
from .base import BaseLabeler
from .registry import register

@register
class MyLabeler(BaseLabeler):
    name = "my_labeler"
    def __init__(self, file, **kwargs):
        ...
    def label(self, ref_id, ref_seq, **kwargs) -> Iterable[tuple[str, int, int, str]]:
        # yield (ref_id, pos_0based, meth_id, strand) tuples
        ...
```

Import in `labelers/__init__.py` to register. Done.

## 8. Baseline samples (negative)

To teach G to produce baseline kinetics when `meth_fwd` and `meth_rev`
are zero, we mix in:

- **Random positions ≥ 20 bp from any labeled meth position**
- **Default count:** `baseline_per_strain = 50_000` (configurable)
- **Mixed in shards alongside meth-positive positions** — no separate
  class, just samples where the meth tensor is all zeros

There is no "baseline class" in the model — the meth tensor is the
condition, and baseline is "meth tensor is zeros".

## 9. Shard schema

One pkl file per strain, located at `shards/<strain>_shard.pkl`. Format:

```python
{
    "__meta__": {
        "config_version": "kinsim_NN-1",
        "extraction_params": ExtractionParams(...),  # K, methylation_types
        "strain_id": "strepto_bc2033",
        "git_sha": "...",
        "kinsim_nn_version": "0.1.0",
        "timestamp_utc": "...",
        "label_sources": [...],  # which labelers were active
    },
    "base_fwd":  np.ndarray(N, K)  uint8  # 0-3 base codes
    "meth_fwd":  np.ndarray(N, K)  uint8  # 0..M-1 meth_id
    "meth_rev":  np.ndarray(N, K)  uint8
    "signal":    np.ndarray(N, K, 4) uint8  # IPD_fwd, PW_fwd, IPD_rev, PW_rev
    "category":  np.ndarray(N) uint8  # 0=baseline 1=meth (for diagnostics only)
    "ref_id":    np.ndarray(N) uint16
    "ref_pos":   np.ndarray(N) int32
    "strand":    np.ndarray(N) int8  # +1 or -1
    "zmw":       np.ndarray(N) int64
}
```

`N` ≈ 3M samples per strain (100k positions × 20 reads cap). Total per
strain ≈ 500 MB. Total corpus (65 strains) ≈ 32 GB.

The arrays are sharded as a `dict[str, np.ndarray]` pickled with
protocol 5. At training time, `Dataset` numpy-memmaps each array → O(1)
random access without loading all in RAM.

## 10. Architecture

### Generator (DiT-style)

```python
TransformerGenerator(
    K              = 21,        # tokens
    M              = 4,         # meth types
    d_model        = 192,
    n_layers       = 6,
    n_heads        = 6,
    z_dim          = 64,
    drop_rate      = 0.0,
)
```

Block (DiT-style with AdaLN-Zero conditioning):

```
DiTBlock(x, z_emb):
    shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = silu(z_proj(z_emb)).chunk(6, dim=-1)
    x = x + gate_msa * MultiHeadAttention(LayerNorm(x) * (1 + scale_msa) + shift_msa)
    x = x + gate_ffn * FFN(LayerNorm(x) * (1 + scale_ffn) + shift_ffn)
    return x
```

Parameter count: ~3M.

### Discriminator

```python
TransformerDiscriminator(
    K              = 21,
    M              = 4,
    d_model        = 128,
    n_layers       = 4,
    n_heads        = 4,
    spectral_norm  = True,
)
```

Spectral norm on every Linear layer (Lipschitz-bounded for WGAN-GP).
CLS token prepended → critic score is `linear(CLS_token_h)`.

Parameter count: ~1.5M.

## 11. Loss & optimization

- **Loss:** WGAN-GP (Wasserstein with gradient penalty, λ=10).
- **n_critic:** 5 D updates per G update.
- **Optimizer:** Adam(β1=0.0, β2=0.9) for both G and D — WGAN-GP standard.
- **LR:** G = 1e-4, D = 4e-4 (TTUR: D learns 4× faster).
- **Batch size:** 256 (configurable, scaled to GPU memory).
- **Total steps:** 200k.
- **Checkpoint:** every 5k steps.
- **Eval:** every 5k steps — compute per-kmer Wasserstein-2 distance
  on held-out test strains.

Stop criterion: fixed 200k steps. No early stopping (GAN losses
oscillate; absolute loss value is uninformative).

## 12. Generation

Input:
- `motifs.csv` (motif strings with `meanIpdRatio`, `fraction`, etc.)
- Reference FASTA(s)
- Input BAM (either PBSIM3 raw HiFi or `strip_kinetics`-cleaned HiFi)
- Trained `G.pt` + `model_config.json`

Process:

```
For each input BAM read:
    For each query position q (skip first/last K/2):
        ref_pos = map(q)
        window = [ref_pos-10, ref_pos+10]
        base_fwd[K] = ref_seq[window]
        meth_fwd[K], meth_rev[K] = scan_motifs(motifs.csv, window)
            # for each motif overlap, place meth_id at meth_offset
            # Bernoulli with motif's fraction to decide if THIS read is methylated
        z = randn(z_dim)
        signal[K, 4] = G(z, base_fwd, meth_fwd, meth_rev)
        signal_center = signal[K // 2]   # = (IPD_fwd, PW_fwd, IPD_rev, PW_rev)
        # Convert log1p(frames) → frames → uint8
        fi[q] = frames_to_uint8(exp(signal_center[0]) - 1)
        fp[q] = frames_to_uint8(exp(signal_center[1]) - 1)
        ri[q] = frames_to_uint8(exp(signal_center[2]) - 1)
        rp[q] = frames_to_uint8(exp(signal_center[3]) - 1)
    Write read with fi/fp/ri/rp to output BAM.
```

Output: unmapped HiFi BAM with `fi:B:C`, `fp:B:C`, `ri:B:C`, `rp:B:C`
tags. Downstream pipeline (bystrandify → align → ipdSummary) is
unchanged from `kinsim` / `kinsim2`.

## 13. Train / test split

Default test holdout: `bc2034, bc2045, bc2048, bc2082` (same as
v12_run3, allowing comparison). Configurable via
`--test-strains` CLI flag or `train.test_strains` YAML field.

## 14. Corpus

Default: `$PREFIX/manifest.csv` (kinsim training manifest, both Strepto
+ Vega = 65 strains). Configurable via `--manifest` CLI flag.

## 15. CLI

Same shape as `kinsim` / `kinsim2`:

```
kinsim_nn extract   --manifest <path> --output-dir <dir> [--config <yaml>]
kinsim_nn train     <shards_dir> <ckpt_dir> [--config <yaml>]
kinsim_nn generate  <input.bam> <ref.fna> <ckpt> <motifs.csv> <out.bam>
kinsim_nn evaluate  <ckpt_dir> <shards_dir> --output <report.html>
```

CLI dispatcher in `kinsim_NN/__main__.py`.

## 16. Repo layout

```
kinsim_NN/
├── DESIGN.md                      this document
├── __init__.py                    version
├── __main__.py                    CLI dispatcher
│
├── extract.py                     GFF/labeler-driven extract → shards
├── train.py                       cGAN training loop (pure PyTorch)
├── generate.py                    motifs.csv → unmapped BAM
├── evaluate.py                    Wasserstein-2 per-kmer report
│
├── labelers/
│   ├── __init__.py                register all labelers
│   ├── base.py                    BaseLabeler ABC
│   ├── registry.py                @register + factory
│   ├── gff.py                     GFFLabeler (motifs.gff parser)
│   └── jasmine_mm_ml.py           JasmineMMMLLabeler (MM/ML BAM tags)
│
├── data/
│   ├── __init__.py
│   ├── shard.py                   Shard schema + read/write
│   └── dataset.py                 ShardedDataset (memmap + collate)
│
├── models/
│   ├── __init__.py
│   ├── blocks.py                  DiTBlock + AdaLN-Zero + SpectralNorm util
│   ├── generator.py               TransformerGenerator
│   └── discriminator.py           TransformerDiscriminator
│
└── utils/
    ├── __init__.py
    ├── config.py                  YAML loader + dataclass schema
    ├── losses.py                  WGAN-GP loss + gradient penalty
    ├── bam_io.py                  bystrandified pair combiner + tag extraction
    └── pacbio_codec.py            uint8 ↔ frames lookup

kinsim_nn_config.yaml              YAML at repo root (default config)
```

## 17. Open questions for future work

These are tracked but NOT done now:

- **Multi-GPU training:** single GPU for v1 (PyTorch DDP later if needed).
- **Variable K (K ≠ 21):** hardcoded; future work for K=11 or K=31 retrain.
- **Cross-attention conditioning** (vs current concat): start with concat, swap to cross-attention if mode collapse observed.
- **EMA on G weights:** standard GAN trick, add if convergence is unstable.
- **Mixed-precision training (bf16):** GPU permitting, can speed 2-3x.

## 18. Quality bar

This is the reference model for the thesis defence. Code must be:

- Type-hinted on all public function signatures.
- `Path` objects for file I/O.
- Explicit exception types (no bare `except Exception`).
- `setup_logging()` called in every CLI `main()`.
- `model_config.json` written **before** the first epoch.
- Resume-friendly checkpoints (G state, D state, optimizer states, step).
- Determinism: torch/numpy/random seeded via `--seed` flag (default 42).
- Logging: CSV at `logs/<run>/metrics.csv` + TensorBoard at `logs/<run>/tb/`.
