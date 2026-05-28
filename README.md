# KinSim

Generative tool that injects biologically realistic PacBio HiFi kinetic
signals (Inter-Pulse Duration and Pulse Width) into a target BAM,
conditioned on the local sequence context and per-position methylation
state. The output BAM carries the canonical PacBio raw-HiFi tag layout
(`fi`, `fp`, `ri`, `rp`) and is drop-in compatible with the standard
downstream chain (`ccs-kinetics-bystrandify` → `pbmm2 align` →
`ipdSummary` → `pbmotifmaker`, and the jasmine / modkit path for 5mC).

The model is a conditional WGAN-GP with a transformer generator and a
transformer critic; see [`DECISIONS.md`](DECISIONS.md) for the design
rationale and [`CLAUDE.md`](CLAUDE.md) for the developer reference.

---

## Installation

Requires Python 3.9 or above. The package is a standard
`pip install -e .` away once dependencies are in place. CUDA-enabled
PyTorch is required for training; CPU PyTorch suffices for inference
on the held-out scale tested.

```bash
git clone https://github.com/NicolasDutilleux/KinSim.git
cd KinSim
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[plot]"   # matplotlib / plotly extras for the analysis dashboards
pip install -e ".[dev]"    # ruff, pytest, pre-commit
```

For a fully pinned environment, regenerate `requirements.lock.txt` from
the cluster `kinsim_env` conda environment:

```bash
ssh <cluster>
conda activate kinsim_env
pip freeze > requirements.lock.txt
```

### Component versions

The pipeline relies on the following exact versions, validated on the
production cluster.

| Component | Version | Source |
|---|---|---|
| Python | 3.9.25 | conda env `kinsim_env` on the production cluster |
| PyTorch | 2.x with CUDA 12.1 wheels | `pyproject.toml` |
| numpy | 1.26 or above | `pyproject.toml` |
| pysam | 0.22 or above | `pyproject.toml` |
| PyYAML | 6.0 or above | `pyproject.toml` |
| TensorBoard | 2.10 or above | `pyproject.toml` |
| matplotlib | 3.8 or above | `pyproject.toml [plot]` |
| EMBOSS fuzznuc | 6.6.0 | system binary on the cluster |
| SMRT-Link Apptainer image | 25.3 | `pacbio-smrt-tools-25.3.sif` (cluster) |
| `pbmm2`, `ccs-kinetics-bystrandify`, `samtools`, `pbindex` | shipped with SMRT-Link 25.3 SIF | `slurm/prep/*.slurm`, jasmine wrapper |
| `ipdSummary` (kineticsTools) | 3.0, SP3-C3 model, **from SMRT-Link 25.1** | L. Falquet install — `slurm/callers/ipdsummary.slurm` (matches his reference detection rates on the production corpus) |
| `pbmotifmaker` | shipped with SMRT-Link 25.1 | same Falquet install |
| `jasmine` | to be pinned from the cluster | `slurm/callers/jasmine_modkit.slurm` |
| `modkit` | to be pinned from the cluster | `slurm/callers/jasmine_modkit.slurm` |

The PacBio binaries are not pip-installable. Most are routed through
the SMRT-Link 25.3 Apptainer SIF on the cluster; `ipdSummary` and
`pbmotifmaker` are pinned to SMRT-Link 25.1 to match the reference
detection rates of the production pipeline.

---

## Pipeline at a glance

```
Aligned bystrandified BAM (ip/pw, 2 records per ZMW)
  + reference FASTA
  + GFF / jasmine MM-ML labels
                                       │
                          kinsim_nn extract
                                       │
                       shards/<strain>_shard.pkl
                                       │
                            kinsim_nn train
                                       │
                  ckpts/{G.pt, D.pt, best_G.pt,
                         model_config.json, metrics.csv, tb/}
                                       │
              ┌────────────────────────┴────────────────────────┐
              │                                                 │
       kinsim_nn evaluate                              kinsim_nn generate
       W1 on held-out shards                  BAM(stripped) → BAM(fi/fp/ri/rp)
```

The same SLURM building blocks in `slurm/prep/` and `slurm/callers/`
are used to validate a generated BAM end-to-end against the standard
PacBio methylation-calling chain.

---

## Command-line interface

```text
kinsim_nn extract  --manifest <csv> --output-dir <dir> [--task <i>] [--config <yaml>]
kinsim_nn train    <shards_dir> <ckpt_dir> [--resume] [--config <yaml>]
kinsim_nn generate <input.bam> <ref.fa> <ckpt_dir> <motifs.csv> <out.bam>
kinsim_nn evaluate <ckpt_dir> <shards_dir> --output-prefix <prefix>
kinsim_nn analyze  <shards_dir_or_file> [--output-dir <dir>] [--no-html]
```

### Typical usage

```bash
# 1. Extract shards from the production manifest, one SLURM array task per strain.
kinsim_nn extract --manifest manifest.csv --output-dir shards/ --task ${SLURM_ARRAY_TASK_ID}

# 2. Train.
kinsim_nn train shards/ ckpts/ --config kinsim_nn_config.yaml

# 3. Generate kinetics into a stripped HiFi BAM.
kinsim_nn generate input_stripped.bam reference.fa ckpts/ motifs.csv output.bam

# 4. Validate the generated BAM end-to-end against the upstream methylation-calling chain.
sbatch slurm/prep/bystrandify.slurm     output.bam              output_bys.bam
sbatch slurm/prep/align_pbmm2.slurm     output_bys.bam reference.fa   output_aln.bam
sbatch slurm/callers/ipdsummary.slurm   output_aln.bam reference.fa   output.gff output.csv
sbatch slurm/callers/pbmotifmaker.slurm reference.fa   output.gff     output_motifs.csv
```

---

## Repository layout

```
KinSim/
├── kinsim_NN/                       core package (extract / train / generate / evaluate / analyze)
├── kinsim_nn_config.yaml            single source of truth for all stages
├── slurm/
│   ├── prep/                        bystrandify, pbmm2 alignment, indexing, hifiasm
│   └── callers/                     ipdSummary, pbmotifmaker, jasmine + modkit, motif merge
├── scripts/
│   ├── strip_kinetics.py            remove fi/fp/ri/rp from a BAM in place
│   ├── plot_perread_ipd_at_gff_sites.py  per-read IPD histograms at top-N GFF positions
│   ├── plot_motif_ipdratios_corpus.py    cross-corpus motif IPDratio aggregator
│   ├── inspect_null_model.py        ipdSummary null-model inspector
│   ├── check_motifs_palindromes.py  motif-catalogue QC
│   └── manifest.py                  manifest CSV utilities
├── tests/                           pytest smoke tests
├── images/                          figures for the thesis
├── reports/                         outputs of the analysis dashboards
├── BUGS_FOUND.md                    BAM-emission boundary bugs and their corrections
├── CHANGELOG.md                     release notes
├── CLAUDE.md                        developer reference
├── CONTRIBUTING.md                  contribution guidelines
├── DECISIONS.md                     architectural decisions log
├── LICENSE                          MIT
├── CITATION.cff                     citation metadata
└── pyproject.toml
```

---

## Validation chain in one paragraph

The validation chain is the same that a real PacBio dataset goes
through. After the generator emits `<sample>_simulated.bam` (unaligned
HiFi with `fi/fp/ri/rp`), the chain is:

1. `ccs-kinetics-bystrandify` to split each ZMW into one
   `/ccs/fwd` and one `/ccs/rev` record carrying per-strand `ip` / `pw`.
2. `pbmm2 align` to the strain's reference assembly.
3. `ipdSummary` with the SP3-C3 model to call modifications, then
   `pbmotifmaker` to extract motif catalogues.
4. In parallel, `jasmine` (5mC) followed by `modkit pileup` and
   `modkit find-motifs` for 5mC motifs.
5. The two catalogues are merged via `slurm/callers/merge_motifs.slurm`
   with a default fraction threshold of `0.7`.

The merged catalogue is then compared site-by-site to the
real-data catalogue using the per-read IPD distribution at the GFF
positions called from the real run
(`scripts/plot_perread_ipd_at_gff_sites.py`).

---

## Citation

If KinSim contributes to your work, please cite this repository
(see [`CITATION.cff`](CITATION.cff) for machine-readable metadata) and
the foundational references listed in [`DECISIONS.md`](DECISIONS.md).

## License

MIT — see [`LICENSE`](LICENSE).
