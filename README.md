# KinSim

<p align="center">
  <img src="./images/unifr_logo.svg" alt="University of Fribourg" width="300">
</p>

<p align="center">
  <strong>PacBio HiFi kinetic signal simulator for methylation-aware metagenomic binning.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-orange">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20SLURM-lightgrey">
</p>

---

## What is KinSim?

KinSim learns per-context **IPD/PW kinetic distributions** from real PacBio HiFi reads and injects realistic kinetic signals into PBSIM3-simulated reads. The output is a synthetic BAM that mimics real PacBio kinetics — including methylation-driven shifts — usable as input for any downstream tool that consumes kinetic information (modkit, ipdSummary, jasmine, motif callers).

**Why?** Standard simulators (PBSIM3, badread) generate sequence + quality but ignore kinetics. Methylation detection in metagenomics relies on kinetic signals; without them, simulated reads are useless for benchmarking methylation-aware binners. KinSim fills this gap.

```
Real PacBio BAMs              Synthetic reads (PBSIM3)
        │                              │
        ▼                              ▼
[motif discovery]              [PBSIM3 sequences]
        │                              │
        └─► kinetic library ──► [trained NN] ──► simulated BAM with fi/fp tags
```

---

## Tools used

### Upstream callers (methylation motif discovery)

| Tool | Tested version | Role |
|---|---|---|
| **PacBio SMRT-Link** | 25.1 (containerised: `pacbio-smrt-tools-25.3.sif` on IBU) | Bundle providing all PacBio CLIs below |
| `ccs-kinetics-bystrandify` | SMRT-Link 25.1 | Split each HiFi read into two per-strand reads carrying ip/pw |
| `pbindex` | SMRT-Link 25.1 | Build `.pbi` index required by ipdSummary |
| `pbmm2` | SMRT-Link 25.1 | Reference alignment (when GFF mode was used; now optional) |
| `ipdSummary` (kineticsTools) | SMRT-Link 25.1, SP3-C3 chemistry model | m6A / m4C statistical caller |
| `pbmotifmaker` | SMRT-Link 25.1 | Consensus motif discovery from ipdSummary GFF |
| `jasmine` | SMRT-Link 25.1 (Revio P2-C2 model) | 5mC ML caller producing MM/ML tags |
| `modkit` | 0.4.x | Convert MM/ML tags into per-position bedMethyl / motif fractions |
| `hifiasm` | 0.19.x | Draft assembly (Vega prep pipeline only) |
| `samtools` | 1.18+ | BAM indexing |

### Read simulation

| Tool | Tested version | Role |
|---|---|---|
| `PBSIM3` | 3.0.4 | HiFi read simulator (sequence-only, no kinetics) — input to `kinsim generate` |

### KinSim model + pipeline

| Component | Tested version | Role |
|---|---|---|
| **Python** | 3.10+ | runtime |
| **PyTorch** | ≥ 2.0 | neural network framework |
| **pysam** | 0.22+ | BAM I/O |
| **numpy** | 1.26+ | numerical core |
| **scikit-learn** | 1.4+ | utilities used by baselines |
| **pandas** | 2.2+ | report tables |
| **plotly** | 5.20+ | interactive HTML reports |
| **pyyaml** | 6.0+ | `kinsim_config.yaml` loader |
| **EMBOSS fuzznuc** *(optional)* | 6.6.0 | fast genome-wide motif scanning (Python regex used as fallback) |
| **Apptainer / Singularity** *(HPC)* | 1.x | container runtime for SMRT-Link image |

The IBU cluster ships `/containers/apptainer/pacbio-smrt-tools-25.3.sif` —
note the image label says 25.3 but the embedded SMRT-Link release is 25.1
for `ipdSummary`/`pbmotifmaker`. Pinning to 25.1 was a deliberate choice:
SMRT-Link 12.x (the system module) detected only ~73% of TCGCGA m4C sites,
versus ~95% with 25.1 (matching L. Falquet's reference output).

---

## How to reproduce

### 1. Install

```bash
git clone https://github.com/NicolasDutilleux/KinSim.git
cd KinSim
pip install -e .
```

The main CLI entry point is **`kinsim`** — the full ML pipeline (extract /
merge / refine / train / generate / evaluate / analyze). A complementary
`kinsim-prep` CLI is also installed for ancillary data-preparation tools
(motif parsing, manifest CSV checks, sample filtering / balancing); see
[`prep/README.md`](prep/README.md) for details.

### 2. Discover methylation motifs (upstream callers)

For each sample, run the prep pipeline that combines jasmine, ipdSummary and pbmotifmaker into one `motifs_merged.csv` (kept at ≥ 70% confidence):

```bash
sbatch slurm_kinsim/strepto/run.sh all       # 52 Streptomyces samples
sbatch slurm_kinsim/vega/run.sh    all       # 15 Vega samples
```

See [`slurm_kinsim/README.md`](slurm_kinsim/README.md) for details.

### 3. Run the ML pipeline

```bash
N=$(kinsim-prep manifest count manifest.csv)
SHARDS=/path/to/shards
MASTER=/path/to/master.pkl

EX=$(sbatch --parsable --array=1-${N}%8 slurm_kinsim/ml/00_extract.slurm \
    manifest.csv $SHARDS)
MG=$(sbatch --parsable --dependency=afterany:$EX slurm_kinsim/ml/01_merge.slurm \
    $SHARDS $MASTER)
RF=$(sbatch --parsable --dependency=afterok:$MG slurm_kinsim/ml/02_refine.slurm \
    $MASTER ${MASTER%.pkl}_clean.pkl)
sbatch --dependency=afterok:$RF slurm_kinsim/ml/03_train.slurm \
    ${MASTER%.pkl}_clean.pkl checkpoints/
```

See [`kinsim/README.md`](kinsim/README.md) for the full ML pipeline reference.

### 4. Generate synthetic BAMs

```bash
kinsim generate <pbsim3_dir> <ref.fna> checkpoints/best.pt motifs.csv output/
```

Output: PacBio-style BAMs with `fi:B:C` (IPD) and `fp:B:C` (PW) tags, ready for any kinetics-aware downstream tool.

---

## Repository layout

```
KinSim/
├── README.md                ← this file (overview)
├── CLAUDE.md                ← in-depth developer reference
├── kinsim_config.yaml       ← biology / refine parameters (signature offsets, etc.)
├── pyproject.toml           ← entry points: kinsim + kinsim-prep
│
├── kinsim/                  ← ML pipeline package      [docs: kinsim/README.md]
├── prep/                    ← Data preparation package [docs: prep/README.md]
├── slurm_kinsim/            ← HPC SLURM scripts        [docs: slurm_kinsim/README.md]
│
├── archive/                 ← legacy code (dictionary, cGAN — not active)
├── baseline/                ← baseline models for comparison
└── images/                  ← logos / figures
```

Each subpackage has its own README with the tools-specific details:

- **[kinsim/README.md](kinsim/README.md)** — ML pipeline, models, refine algorithm, signature profiles
- **[prep/README.md](prep/README.md)** — caller parsers, motif merging, manifest tools, balance/filter
- **[slurm_kinsim/README.md](slurm_kinsim/README.md)** — per-dataset prep pipelines, ML SLURM chain, resource defaults

For implementation details (data flow, file format, conventions), see [`CLAUDE.md`](CLAUDE.md).

---

## Configuration

A single `kinsim_config.yaml` at the repo root holds the biology- and refine-related parameters that the user must keep up-to-date:

```yaml
kinetic_signatures:
  m6A: { signal_offsets: [0, 5] }    # at modified A AND +5 downstream
  m4C: { signal_offsets: [0] }       # at modified C only
  m5C: { signal_offsets: [2, 6] }    # +2 and +6 downstream — NOT at the C itself

meth_context:    { left: 7, right: 3 }     # asymmetric kmer / FiLM window
kinetic_profile: { start: 0, end: 8 }      # downstream profile stored per sample
```

Strain-specific signatures (e.g. m6A at +8 instead of +5 for some methyltransferases) are handled by editing the YAML — no code change.

---

## Output BAM format

| Field | Value |
|---|---|
| `flag` | `4` (unmapped) |
| `fi:B:C` | per-base IPD, uint8, length = read length |
| `fp:B:C` | per-base PW, uint8, length = read length |
| `ri:B:C` / `rp:B:C` | reverse-strand kinetics (when `--reverse` set during generation) |
| Header | `HD VN:1.6 SO:unknown` |

The output is single-read-per-molecule (raw HiFi format). Pass it through `ccs-kinetics-bystrandify` if downstream tools expect a bystrandified BAM.

---

## License

MIT — see `LICENSE`.

## Citation

If you use KinSim in your work, please cite:

```
Dutilleux, N. (2026). KinSim: PacBio HiFi kinetic signal simulator for
methylation-aware metagenomic binning. University of Fribourg.
https://github.com/NicolasDutilleux/KinSim
```
