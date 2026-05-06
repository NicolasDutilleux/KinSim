# `slurm_kinsim/` — HPC job scripts

SLURM-ready job scripts for running KinSim end-to-end on the IBU cluster
(or any SLURM-managed HPC with comparable partitions).

For project-level overview see the [top-level README](../README.md).

---

## Pipeline structure

KinSim runs in two stages: **prep** (per dataset, methylation discovery)
and **ML** (shared across datasets, training and generation).

```
1. Prep pipelines (per dataset — each emits manifest_<dataset>.csv)
   ┌─────────────────────────┐ ┌───────────────────────┐ ┌─────────────────┐
   │ slurm_kinsim/strepto/   │ │ slurm_kinsim/vega/    │ │ slurm_kinsim/   │
   │  hifiasm asm (no asm    │ │  hifiasm assembly     │ │  sequel/        │
   │  for Strepto since      │ │   ↓                   │ │   subreads → CCS│
   │  refs already exist)    │ │  bystrandify          │ │   ↓             │
   │   ↓                     │ │   ↓                   │ │  bystrandify    │
   │  bystrandify            │ │  pbmm2 align          │ │   ↓             │
   │   ↓                     │ │   ↓                   │ │  pbmm2 align    │
   │  pbmm2 align            │ │  ipdSummary SP3-C3    │ │   ↓             │
   │   ↓                     │ │   ↓                   │ │  ipdSummary     │
   │  ipdSummary  +  modkit  │ │  pbmotifmaker         │ │   ↓             │
   │   ↓             (from   │ │   ↓                   │ │  pbmotifmaker   │
   │  pbmotifmaker  jasmine  │ │  merge_motifs         │ │   ↓             │
   │   ↓             tags)   │ │   ↓                   │ │  merge_motifs   │
   │  merge_motifs           │ │  build_manifest       │ │   ↓             │
   │   ↓                     │ │   ↓                   │ │  build_manifest │
   │  build_manifest         │ │ manifest_vega.csv     │ │   ↓             │
   │   ↓                     │ │                       │ │ manifest_       │
   │ manifest_strepto.csv    │ │                       │ │  sequel.csv     │
   └─────────────────────────┘ └───────────────────────┘ └─────────────────┘
                              ↓
2. ML pipeline (shared, manifest-driven — sharded end-to-end)
   ┌────────────────────────────────────────────────────────────┐
   │ slurm_kinsim/ml/                                           │
   │  00_extract.slurm  →  shards/<sample_id>_shard.pkl  (array)│
   │  02_refine.slurm   →  refined/   (per-(meth, offset) GMM)  │
   │  03_train.slurm    →  checkpoints/  (1 GPU)                │
   │  04_generate.slurm →  simulated BAM (per genome / array)   │
   │  05_evaluate.slurm →  calibration report                   │
   │  06_verify_generate.slurm → ref vs gen comparison          │
   └────────────────────────────────────────────────────────────┘
```

---

## Prep pipelines

Each prep pipeline is launched by its `run.sh` orchestrator with one of
three commands:

```bash
# Per-dataset orchestrator
sbatch slurm_kinsim/<dataset>/run.sh all                # all samples
sbatch slurm_kinsim/<dataset>/run.sh call <sample_id>   # one sample
sbatch slurm_kinsim/<dataset>/run.sh manifest           # rebuild manifest from existing motifs CSVs
```

The `run.sh` chains the per-step `*.slurm` jobs with `afterok`/`afterany`
dependencies. Each step runs from inside an Apptainer image
(`/containers/apptainer/pacbio-smrt-tools-25.3.sif`) for reproducibility.

### Per-step scripts (Vega / Strepto / Sequel)

| Step | Script | Tool | Purpose |
|---|---|---|---|
| 0 | `00_assembly.slurm` (Vega only) | hifiasm | Draft assembly from raw HiFi |
| 0/1 | `00_bystrandify.slurm` | ccs-kinetics-bystrandify | Split each read into per-strand reads with ip/pw |
| 1/2 | `01_align.slurm` | pbmm2 | Align to reference |
| 2/3 | `02_index.slurm` | samtools + pbindex | Build .bai and .pbi indexes |
| 3/4 | `03_ipdsummary.slurm` | ipdSummary SP3-C3 | Statistical m6A / m4C calling |
| 3b | `03_jasmine_modkit.slurm` (Strepto) | jasmine + modkit | 5mC calling from MM/ML tags |
| 4/5 | `04_motifmaker.slurm` | pbmotifmaker | Consensus motif discovery from ipdSummary GFF |
| 5/6 | `05_merge_motifs.slurm` | `prep/motif_merge.py` | Merge / filter / dedup at threshold 0.7 |
| 6/7 | `06_build_manifest.sh` | bash | Emit `manifest_<dataset>.csv` |

The exact step numbering varies slightly by dataset (Vega adds an
assembly step; Strepto skips it since references are pre-existing).

---

## ML pipeline

Shared across all datasets — drives off any `manifest_<dataset>.csv`
emitted by a prep pipeline.

```bash
# Compute manifest size for SLURM array
N=$(kinsim-prep manifest count manifest.csv)

SHARDS=/path/to/shards
REFINED=/path/to/refined

# 1. Extract — array job, one task per sample
EX=$(sbatch --parsable --array=1-${N}%8 slurm_kinsim/ml/00_extract.slurm \
    manifest.csv $SHARDS)

# 2. Refine — pool harvest across shards, fit GMMs once, apply per-shard
RF=$(sbatch --parsable --dependency=afterany:$EX slurm_kinsim/ml/02_refine.slurm \
    $SHARDS $REFINED)

# 3. Train — directory input, ShardedSignalDataset reads from refined/
TR=$(sbatch --parsable --dependency=afterok:$RF slurm_kinsim/ml/03_train.slurm \
    $REFINED checkpoints/)

# 4. Generate (typically a separate chain on PBSIM3 reads)
sbatch --dependency=afterok:$TR slurm_kinsim/ml/04_generate.slurm \
    <pbsim3_dir> checkpoints/best.pt motifs.csv output_dir/
```

The `slurm_kinsim/ml/run.sh` orchestrator wraps the extract → refine →
train → evaluate chain.

### Resource defaults

| Step | Mem | CPUs | GPU | Time | Notes |
|---|---|---|---|---|---|
| `00_extract` | 128 GB | 1 | — | 24h | Per-sample array task |
| `02_refine` | 96 GB | 4 | — | 6h | Pool-harvest pass + per-shard GMM apply (peak ≈ one shard) |
| `03_train` | 32 GB | 8 | 1 | 24h | A100 / H100 / V100 — auto-detected |
| `04_generate` | 32 GB | 4 | 1 | 4h | One job per simulated genome |

These are SLURM `#SBATCH --mem=…` defaults — tune in each script header
if your cluster has different node sizes.

---

## Cluster partition guidance (IBU example)

The IBU cluster offers several partitions:

| Partition | Time limit | Use case |
|---|---|---|
| `pibu_el8` | 28 days | Long jobs (extract array, train) |
| `pshort_el8` | 2 hours | Quick tests, refine on small datasets |
| `pgpu` | 28 days | Train / generate (1 GPU per node) |

Use `pshort_el8` for smoke tests (`--max-reads 200000`) — they queue
faster than the long `pibu_el8`.

---

## File layout

```
slurm_kinsim/
├── pbsim3_simulate.slurm           — PBSIM3 read simulation (input for generate)
├── jasmine_5mc.slurm               — jasmine + modkit 5mC discovery (array)
│
├── vega/                           — Vega prep pipeline (15 samples)
│   ├── 00_assembly.slurm
│   ├── 01_bystrandify.slurm
│   ├── 02_align.slurm
│   ├── 03_index.slurm
│   ├── 04_ipdsummary.slurm
│   ├── 05_motifmaker.slurm
│   ├── 06_build_manifest.sh
│   └── run.sh                      ← orchestrator
│
├── strepto/                        — Strepto prep pipeline (52 samples)
│   ├── 00_bystrandify.slurm
│   ├── 01_align.slurm
│   ├── 02_index.slurm
│   ├── 03_ipdsummary.slurm
│   ├── 04_motifmaker.slurm
│   ├── 05_build_manifest.sh
│   └── run.sh
│
├── sequel/                         — Sequel prep pipeline (subread-based)
│   ├── 00_ccs.slurm
│   ├── 01_bystrandify.slurm
│   ├── … (same chain as vega)
│   └── run.sh
│
├── ml/                             — Shared ML pipeline
│   ├── 00_extract.slurm
│   ├── 02_refine.slurm
│   ├── 03_train.slurm
│   ├── 04_generate.slurm
│   ├── 05_evaluate.slurm
│   ├── 06_verify_generate.slurm
│   └── run.sh                      ← orchestrator
│
├── config/                         — Example configs
│   ├── config_example.yaml
│   └── manifest_example.csv
│
└── msa1003/                        — Legacy MSA1003 mock community pipeline
    ├── prep_rebase.sh
    ├── prep_merge.sh
    ├── 00_align_split.slurm
    ├── 00b_add_ippw.slurm
    ├── 01_ipdsummary.slurm
    ├── 01b_modkit.slurm
    └── 02_pbmotifmaker.slurm
```

---

## Logging conventions

All SLURM scripts write logs to:

```
/data/projects/p774_MARSD/NDutilleux/logs/
├── ml_00_extract_<JOBID>_<TASKID>.log
├── ml_02_refine_<JOBID>.log
├── ml_03_train_<JOBID>.log
└── …
```

Each script's header includes diagnostics (date, hostname, GPU info,
SLURM env vars, timing, exit code) for post-mortem.
