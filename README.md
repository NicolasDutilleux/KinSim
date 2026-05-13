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
REFINED=/path/to/refined

EX=$(sbatch --parsable --array=1-${N}%8 slurm_kinsim/ml/00_extract.slurm \
    manifest.csv $SHARDS)
RF=$(sbatch --parsable --dependency=afterany:$EX slurm_kinsim/ml/02_refine.slurm \
    $SHARDS $REFINED)
sbatch --dependency=afterok:$RF slurm_kinsim/ml/03_train.slurm \
    $REFINED checkpoints/
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
├── scripts/                 ← auxiliary one-off tools (compare, sample, strip-kinetics, …)
├── baseline/                ← baseline models for comparison
└── images/                  ← logos / figures
```

Each subpackage has its own README with the tools-specific details:

- **[kinsim/README.md](kinsim/README.md)** — ML pipeline, models, refine algorithm, signature profiles
- **[prep/README.md](prep/README.md)** — caller parsers, motif merging, manifest tools, balance/filter
- **[slurm_kinsim/README.md](slurm_kinsim/README.md)** — per-dataset prep pipelines, ML SLURM chain, resource defaults

For implementation details (data flow, file format, conventions), see [`CLAUDE.md`](CLAUDE.md).

---

## Input requirements

`kinsim extract` consumes **aligned bystrandified BAMs only**. Other BAM formats either fail fast (a sniff check rejects raw HiFi BAMs that still carry `ri` tags) or fall back to half-data with a warning.

Required preprocessing chain (versions in the [Tools used](#tools-used) table above):

```
raw HiFi BAM
  → ccs-kinetics-bystrandify   (split each CCS into 2 reads, one per polymerase pass)
  → pbmm2 align                (produces aligned bystrandified BAM)
  → samtools index + pbindex
  → kinsim extract             (this repo)
```

Implemented end-to-end by the prep pipelines under [`slurm_kinsim/<dataset>/`](slurm_kinsim/).

---

## Reproducibility / Benchmarking

The pipeline is built to be re-runnable from the same starting state. Each stage is deterministic given a seed (refine, train accept `--seed`); the generative samples in `generate` are explicitly stochastic but seedable too.

### Reference dataset

Streptomyces collection (52 strains, ~250k–500k primary aligned reads each) — internal IBU paths:

| Resource | Path |
|---|---|
| Bystrandified aligned BAMs | `/data/projects/p774_MARSD/NDutilleux/training/Strepto/pipeline/<sample>/<sample>_aligned.bam` |
| Reference assemblies | `/data/projects/p774_MARSD/NDutilleux/training/Strepto/<sample>/final_assembly.fasta` |
| Merged motifs (per-sample CSV) | adjacent to each BAM |
| Manifest CSV (52 samples) | `/data/projects/p774_MARSD/NDutilleux/runs/v?_strepto/manifest_aligned.csv` |
| SMRT-Tools container | `/containers/apptainer/pacbio-smrt-tools-25.3.sif` |

### Expected runtimes (single bc2034 strain on `pshort_el8`)

Validated 2026-05-06 with the v7 / 20-col / vectorised extract:

| Stage | Wall time | Memory | Notes |
|---|---|---|---|
| `00_extract` | ~50 min | 32 GB | 248k primary reads, vectorised inner loop |
| `02_refine` | ~10 min | 16 GB | per-(meth, offset) GMM, BIC over K∈{2,3} |
| `03_train` | ~2-4 h | 32 GB + 1 GPU | ConvPredictor, 50 epochs, ReduceLROnPlateau |
| `05_evaluate` | ~10 min | 32 GB | calibration plots + per-kmer report |

For the **full Strepto corpus** (52 strains via SLURM array), wall time is bounded by the slowest single strain — typically ~60-90 min for extract.

### One-shot reproduction

After cloning + `pip install -e .` and verifying the cluster paths above:

```bash
# bc2034 single-strain verification (extract → refine, ~1h total)
PREFIX=/path/to/run_dir
MANIFEST=/path/to/manifest_bc2034_only.csv
N=$(kinsim-prep manifest count $MANIFEST)
J0=$(sbatch --parsable --array=1-${N} \
    --partition=pshort_el8 --mem=32G --time=02:00:00 \
    slurm_kinsim/ml/00_extract.slurm $MANIFEST $PREFIX/shards)
sbatch --parsable --dependency=afterok:$J0 \
    --partition=pshort_el8 --mem=16G --time=01:00:00 \
    slurm_kinsim/ml/02_refine.slurm $PREFIX/shards $PREFIX/refined

# full corpus: replace manifest, drop --time bound, use pibu_el8 if needed
```

### Versioning / change history

- Architectural decisions with rationale: [`DECISIONS.md`](DECISIONS.md).
- User-facing change log: [`CHANGELOG.md`](CHANGELOG.md).
- For the exact code state used to produce a result, capture the commit SHA next to your results: `git rev-parse HEAD`.

---

## Configuration

A single `kinsim_config.yaml` at the repo root holds the biology- and refine-related parameters that the user must keep up-to-date:

```yaml
kinetic_signatures:
  m6A: { signal_offsets: [0, 5] }    # at modified A AND +5 downstream
  m4C: { signal_offsets: [0] }       # at modified C only
  m5C: { signal_offsets: [2, 6] }    # +2 and +6 downstream — NOT at the C itself

meth_context:    { left: 7, right: 3 }     # asymmetric kmer / FiLM window
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

## Acknowledgements — Tooling

The author (Nicolas Dutilleux) developed this project during a stage de
master at the University of Fribourg (February – May 2026). The code,
the scientific approach, the architectural decisions, the validation
methodology, and the interpretation of results are the author's
contribution and responsibility.

**Tooling used during development:**
**Claude (Anthropic)**, accessed via the **Claude Code** CLI, was used
as a coding assistant — similar in role to an IDE, an autocomplete, or
a documentation lookup. The AI accelerated tasks like vectorising
numpy loops, scaffolding CLI options and SLURM submission scripts,
drafting helper utilities, surfacing potential failure modes in code
review, and producing first-draft docstrings.

**Review and validation:**
Every block of AI-suggested code was read, validated, and integrated
by the author before being committed. AI suggestions that did not match
the intended design were rejected or rewritten. Issues identified
during review and debug iterations on the cluster were diagnosed and
fixed by the author.

**Residual bugs:**
Like any software project of this scope developed under stage-de-master
time constraints, residual bugs may remain. Reviewers and re-users
of this code are encouraged to read critically, run their own
validations, and open issues on the tracker for anything that looks
suspicious.

This disclosure follows current best practice (2026) for AI-assisted
research software and aligns with the spirit of academic honesty for
master-level work. The author can explain and defend every design
decision and every line of code in this repository.

For implementation-level details of where AI was used, the development
log [`CHANGELOG_TFE.md`](CHANGELOG_TFE.md) (kept locally, not committed)
breaks down each improvement with its rationale and the corresponding
academic reference.

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
