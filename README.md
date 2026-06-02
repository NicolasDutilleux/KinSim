# KinSim

PacBio HiFi kinetic-signal generator — a conditional WGAN-GP that injects
biologically realistic IPD and PW kinetics into HiFi BAMs so they pass
through the standard PacBio methylation-calling chain
(`ccs-kinetics-bystrandify` → `pbmm2` → `ipdSummary` → `pbmotifmaker`,
and the jasmine + modkit path for 5mC) as if they were sequencer
output.

```
real bystrandified+aligned BAM ─┐
        + ref.fasta             ├─► kinsim_nn extract  ► shards/*.pkl
        + motifs.gff / MM-ML    ┘                              │
                                                               ▼
                                                    kinsim_nn train ► ckpts/best_G.pt
                                                               │
                  stripped HiFi BAM + ref + motifs.csv ─────────┤
                                                               ▼
                                          kinsim_nn generate ► BAM with fi/fp/ri/rp
                                                               │
                                                ┌──────────────┴──────────────┐
                                       bystrandify ► pbmm2 ► ipdSummary ► pbmotifmaker
                                       (validation chain, same as real data)
```

## Documentation map

| Doc | What's inside |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | developer reference: package layout, data flow, conventions |
| [`DECISIONS.md`](DECISIONS.md) | architectural rationale (WGAN-GP, transformer, AdaLN-Zero, …) |
| [`PACBIO_COMPATIBILITY.md`](PACBIO_COMPATIBILITY.md) | **read this before touching the BAM emission path or chaining tools** — version matrix, bystrandify silent-drop rules, BAM-format conventions, apptainer/conda mixing rules |
| [`BUGS_FOUND.md`](BUGS_FOUND.md) | historical record of every silent-drop bug we hit on the bc2034 validation chain |
| [`CHANGELOG.md`](CHANGELOG.md) | release notes |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | dev setup, lint/test commands, PR workflow |

## Install

Python 3.9 or above. The production cluster runs 3.9.25 inside the
`kinsim_env` conda environment.

```bash
git clone https://github.com/NicolasDutilleux/KinSim.git
cd KinSim
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,plot]"
pre-commit install
```

PacBio binaries (`pbmm2`, `ccs-kinetics-bystrandify`, `pbindex`,
`pbmotifmaker`, `ipdSummary`) are **not** pip-installable — they come
from the SMRT-Tools Apptainer SIF (and one host install for
`ipdSummary` / `pbmotifmaker` 25.1). See
[`PACBIO_COMPATIBILITY.md`](PACBIO_COMPATIBILITY.md) for the exact
sources and the apptainer-first policy.

## CLI

```text
kinsim_nn extract  --manifest <csv> --output-dir <dir> [--task <i>] [--config <yaml>]
kinsim_nn train    <shards_dir> <ckpt_dir> [--resume] [--config <yaml>]
kinsim_nn generate <input.bam> <ref.fa> <ckpt_dir> <motifs.csv> <out.bam>
kinsim_nn evaluate <ckpt_dir> <shards_dir> --output-prefix <prefix>
kinsim_nn analyze  <shards_dir_or_file> [--output-dir <dir>] [--no-html]
```

`extract` takes named flags; every other subcommand is positional.
Defaults for all stages live in
[`kinsim_nn_config.yaml`](kinsim_nn_config.yaml) — the single source of
truth, frozen into `model_config.json` at training start so a
checkpoint cannot be silently broken by a later YAML edit.

## Validation chain

After `kinsim_nn generate` emits a BAM with `fi/fp/ri/rp`, run it
through the same chain as the real data:

```bash
sbatch slurm/prep/bystrandify.slurm     output.bam              output_bys.bam
sbatch slurm/prep/align_pbmm2.slurm     output_bys.bam ref.fa   output_aln.bam
sbatch slurm/callers/ipdsummary.slurm   output_aln.bam ref.fa   output.gff output.csv
sbatch slurm/callers/pbmotifmaker.slurm ref.fa         output.gff output_motifs.csv
```

Per-read IPD comparison against the real chain:
`scripts/plot_perread_ipd_at_gff_sites.py`.

For the production v6 validation that splits CPU/GPU work and chains
strip → generate → downstream via `--dependency=afterok`, see
[`slurm/validate/`](slurm/validate/) and
[`PACBIO_COMPATIBILITY.md`](PACBIO_COMPATIBILITY.md).

## Repository layout

```
KinSim/
├── kinsim_NN/                  core package (extract / train / generate / evaluate / analyze)
├── kinsim_nn_config.yaml       YAML defaults, frozen into model_config.json
├── slurm/
│   ├── prep/                   bystrandify, pbmm2, indexing, hifiasm
│   ├── callers/                ipdSummary, pbmotifmaker, jasmine + modkit, motif merge
│   ├── validate/               v6 validation chain (strip → generate → downstream)
│   └── eval/                   held-out W1 eval drivers
├── scripts/                    pysam utilities, analysis plots, manifest CLI
├── tests/                      pytest smoke tests
├── PACBIO_COMPATIBILITY.md     compatibility rules between PacBio tools (read first)
├── BUGS_FOUND.md               every silent-drop bug we have hit (historical)
├── CLAUDE.md                   developer reference
├── DECISIONS.md                architectural rationale
├── CHANGELOG.md                release notes
├── CONTRIBUTING.md             dev workflow
├── CITATION.cff                citation metadata
└── pyproject.toml              build + lint + test configuration
```

## License

MIT — see [`LICENSE`](LICENSE).
