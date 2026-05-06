# `kinsim/` — ML pipeline

Supervised learning that turns a real kinetic library into a trained
model, then uses that model to inject realistic IPD/PW into synthetic
reads. Exposes the `kinsim` CLI (entry point in `pyproject.toml`).

For project-level overview see the [top-level README](../README.md).

---

## The pipeline (one verb per stage)

```
manifest.csv ──┐
ref.fasta    ──┤
motifs.csv   ──┴──► kinsim extract  ──► shards/*.pkl
                   kinsim refine    ──► refined/*.pkl
                   kinsim train     ──► checkpoints/best.pt + model_config.json
                   kinsim generate  ──► synthetic BAM (fi/fp tags)
                   kinsim evaluate  ──► calibration report
                   kinsim verify-   ──► per-(kmer, meth) ref-vs-gen comparison
                          generate
                   kinsim analyze   ──► HTML + TXT diagnostic dashboard
```

Each stage consumes the previous stage's output. Shards are the canonical
on-disk format throughout — `dict[kmer_id (int) → np.ndarray(N, 20)]`,
plus a `__meta__` provenance entry. The 20-column layout is defined in
[`utils/sample_layout.py`](utils/sample_layout.py).

| Verb | Module | Input | Output |
|---|---|---|---|
| `extract` | [extract.py](extract.py) | aligned bystrandified BAM + ref + motifs | `<sample>_shard.pkl` |
| `refine` | [refine.py](refine.py) | shards/ (or single shard.pkl) | refined/ shards |
| `train` | [train.py](train.py) | refined/ (or single shard.pkl) | checkpoints/ + `model_config.json` |
| `generate` | [generate.py](generate.py) | PBSIM3 reads + checkpoint + motifs | unaligned BAM with kinetics |
| `evaluate` | [evaluate.py](evaluate.py) | checkpoint + shard.pkl | calibration report + plots |
| `verify-generate` | [verify_generate.py](verify_generate.py) | two shards (ref vs gen) | TSV with per-(kmer, meth) Pearson r + MAE |
| `analyze` | [analyze.py](analyze.py) | shard.pkl OR shards/ directory | HTML + TXT report |

---

## Repository layout

```
kinsim/
├── __init__.py
├── __main__.py            CLI router — dispatches each verb to its module
│
├── extract.py             "kinsim extract" — orientation-aware extraction
├── refine.py              "kinsim refine" — per-(meth, offset) GMM filter
├── train.py               "kinsim train" — supervised training
├── generate.py            "kinsim generate" — model + reads → simulated BAM
├── evaluate.py            "kinsim evaluate" — calibration report
├── verify_generate.py     "kinsim verify-generate" — ref vs gen comparison
├── analyze.py             "kinsim analyze" — diagnostic dashboard
│
├── data/dataset.py        ShardedSignalDataset / MLPSignalDataset
├── models/predictor.py    ConvPredictor + MLPPredictor
└── utils/
    ├── encoding.py        kmer bit-packing (K=11, asymmetric window)
    ├── motifs.py          motif parsing + scan + IUPAC
    ├── sample_layout.py   the 20-column row contract
    ├── config.py          YAML + manifest loader
    └── io.py              FASTA loader + atomic pickle writes
```

Auxiliary scripts (not pipeline stages — run directly with `python`) live
in [`scripts/`](../scripts/) at the repo root.

---

## Architecture invariants

### 1. Asymmetric kmer + meth context window — `[−7, +3]`

The polymerase has read more bases upstream than downstream at any moment,
and all kinetic signatures are downstream of the modification (m6A at +5,
m5C at +2/+6). To predict IPD at position Y, the model needs upstream
context — methylations whose downstream effect reaches Y.

Inspired by Feng *et al.* 2013 (`kineticsTools` / `ipdSummary`, `[−7, +2]`
for unmodified DNA), extended to 11 positions to match a single packed-
integer 11-mer encoding. The prediction position sits at index 7 of the
kmer.

### 2. Strand-aware extraction (HiFi only)

Raw HiFi BAMs cannot be processed directly: `query_sequence` orientation is
arbitrary per-read, so `fi`/`ri` are swapped relative to the reference for
~50% of reads. KinSim consumes **aligned bystrandified BAMs** with `ip`/`pw`
tags, where `read.is_reverse` disambiguates which kinetic tag carries the
methylation signal at each position. See `kinsim/extract.py` top docstring
for the full convention.

### 3. Three categories per row

```
CATEGORY  COL_PARENT_METH  COL_PARENT_OFFSET
   0      0                0      → BASELINE  (far from any motif)
   1      meth_id          k      → SLOWED    (at p+k of a motif, k in signal_offsets[T])
   2      meth_id          k      → NEAR_METH (close to a motif but not at a signature offset)
```

Refine GMM separates per-(meth_type, offset) buckets independently — a
noisy offset of one meth type can fail validation and pass through
unfiltered without contaminating a clean offset of the same type.

### 4. Sharded mode

For corpora ≥ a handful of strains, `kinsim refine`, `train`, `analyze`
all consume **directories of `*_shard.pkl`** directly. Memory peak is
bounded by one shard, never the whole corpus.

---

## Configuration

Single YAML at the repo root: [`kinsim_config.yaml`](../kinsim_config.yaml).
Loaded via `utils.config.load_kinsim_config()`. Knobs:

```yaml
kinetic_signatures:
  m6A:
    modified_base:  A
    signal_offsets: [0, 5]
    aliases:        [6mA, 6-mA]
  m4C:
    modified_base:  C
    signal_offsets: [0]
  m5C:
    modified_base:  C
    signal_offsets: [2, 6]

meth_context: { left: 7, right: 3 }

extract:
  n_baseline_per_kmer:        50
  baseline_min_dist_to_meth:  11
  baseline_sample_rate:       0.10
  near_meth_max_dist:         7
```

Adding a new methylation type is a YAML edit only — `extract`, `refine`,
`train`, `analyze` all pick it up at runtime via
`utils.encoding.get_meth_ids()` and `utils.config.get_signature_offsets()`.

---

## Manifest format

```
sample_id,bam_path,motifs,ref_path
bc2034,/path/to/bc2034_aligned.bam,/path/to/motifs.csv,/path/to/genome.fasta
bc2045,/path/to/bc2045_aligned.bam,/path/to/motifs.csv,/path/to/genome.fasta
```

`ref_path` is REQUIRED. The `bam_path` must be an aligned bystrandified
BAM (sorted by coordinate, with `ip`/`pw` tags, indexed). Use the prep
pipeline in [`slurm_kinsim/strepto/`](../slurm_kinsim/strepto/) to build
these from raw HiFi.
