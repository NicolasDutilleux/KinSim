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

Prep is now structured as **shared modules** that any per-dataset
orchestrator (`<dataset>/run.sh`) chains together via `sbatch
--dependency`. The numbered scripts have been replaced by reusable
named ones:

| Module | Tool | Purpose |
|---|---|---|
| `prep/assembly_hifiasm.slurm` | hifiasm | Draft assembly from raw HiFi (Vega only) |
| `prep/bystrandify.slurm` | ccs-kinetics-bystrandify | Split each HiFi read into per-strand reads with ip/pw |
| `prep/align_pbmm2.slurm` | pbmm2 | Align to reference (SKIP-first, filter unmapped, pbindex output) |
| `prep/index_bam.slurm` | samtools + pbindex | Build `.bai` and `.pbi` indexes |
| `callers/ipdsummary.slurm` | ipdSummary SP3-C3 | Statistical m6A / m4C calling |
| `callers/jasmine_modkit.slurm` | jasmine + modkit | 5mC calling from MM/ML tags |
| `callers/pbmotifmaker.slurm` | pbmotifmaker | Consensus motif discovery from ipdSummary GFF |
| `callers/merge_motifs.slurm` | `python -m kinsim.utils.parsers.motif_merge` | Merge / filter / dedup at threshold 0.7 |
| `<dataset>/06_build_manifest.sh` | bash | Emit `manifest_<dataset>.csv` |

Each per-dataset orchestrator (`vega/run.sh`, `sequel/run.sh`,
`strepto/run.sh`) chains the relevant modules in the correct order with
`afterok` dependencies.

---

## ML pipeline

Shared across all datasets — drives off any `manifest_<dataset>.csv`
emitted by a prep pipeline.

```bash
# Compute manifest size for SLURM array
N=$(python scripts/manifest.py count manifest.csv)

SHARDS=/path/to/shards
REFINED=/path/to/refined

# 1. Extract — array job, one task per sample
EX=$(sbatch --parsable --array=1-${N}%8 slurm_kinsim/ml/00_extract.slurm \
    manifest.csv $SHARDS)

# 2. Refine — pool harvest across shards, fit GMMs once, apply per-shard
RF=$(sbatch --parsable --dependency=afterany:$EX slurm_kinsim/ml/02_refine.slurm \
    $SHARDS $REFINED)

# 3. Train — directory input, SignalDataset reads from refined/
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
├── pbsim3_simulate.slurm           — PBSIM3 read simulation
├── ccs_subreads.slurm              — ccs → HiFi BAM with fi/fp/ri/rp
├── validate.sh                     — per-strain validate chain orchestrator
│
├── prep/                           — shared prep modules
│   ├── bystrandify.slurm
│   ├── align_pbmm2.slurm
│   ├── index_bam.slurm
│   ├── assembly_hifiasm.slurm
│   └── README.md
│
├── callers/                        — methylation callers (any aligned BAM)
│   ├── ipdsummary.slurm
│   ├── pbmotifmaker.slurm
│   ├── jasmine_modkit.slurm
│   ├── merge_motifs.slurm
│   └── README.md
│
├── validate/                       — per-task SLURM for validate.sh
│   ├── prep.slurm
│   ├── generate.slurm
│   ├── merge.slurm
│   └── write_regions.py
│
├── ml/                             — shared ML pipeline
│   ├── 00_extract.slurm
│   ├── 02_refine.slurm
│   ├── 03_train.slurm
│   ├── 04_generate.slurm
│   ├── 05_evaluate.slurm
│   ├── 06_verify_generate.slurm
│   └── run.sh                      ← orchestrator
│
├── vega/                           — Vega per-dataset orchestrator
│   ├── 06_build_manifest.sh
│   └── run.sh
│
├── sequel/                         — Sequel per-dataset orchestrator
│   ├── 06_build_manifest.sh
│   └── run.sh
│
├── strepto/                        — Strepto per-dataset orchestrator
│   ├── 05_build_manifest.sh
│   └── run.sh
│
└── config/                         — example configs
    ├── config_example.yaml
    └── manifest_example.csv
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
