# `slurm/` — HPC job scripts

SLURM-ready job scripts for the preprocessing and methylation-calling
chain used by KinSim, both for preparing real-data corpora and for
validating generated BAMs against the standard PacBio analytical
chain.

For project-level overview see the [top-level README](../README.md).
For the package itself see [`CLAUDE.md`](../CLAUDE.md).

---

## Layout

```
slurm/
├── prep/                     — shared preprocessing modules
│   ├── bystrandify.slurm     — ccs-kinetics-bystrandify
│   ├── align_pbmm2.slurm     — pbmm2 alignment (SKIP-first, pbindex output)
│   ├── index_bam.slurm       — samtools index + pbindex
│   ├── assembly_hifiasm.slurm — draft assembly from raw HiFi
│   └── README.md
│
└── callers/                  — methylation callers (any aligned BAM)
    ├── ipdsummary.slurm      — ipdSummary SP3-C3 (m6A + m4C)
    ├── pbmotifmaker.slurm    — consensus motif discovery from ipdSummary GFF
    ├── pbmotifmaker_reprocess.slurm — re-run with relaxed thresholds
    ├── jasmine_modkit.slurm  — jasmine + modkit (5mC via CpG model)
    ├── jasmine_align_only.slurm — jasmine without modkit pile-up
    ├── merge_motifs.slurm    — union with fraction threshold and dedup
    └── README.md
```

The `prep/` scripts cover everything between a raw HiFi BAM and an
aligned, indexed BAM. The `callers/` scripts run methylation callers
against an aligned BAM and produce motif catalogues.

---

## Validation chain (typical usage)

Given a generated `kinsim_nn generate` output (`output.bam`):

```bash
J1=$(sbatch --parsable slurm/prep/bystrandify.slurm \
    output.bam output_bys.bam)

J2=$(sbatch --parsable --dependency=afterok:$J1 \
    slurm/prep/align_pbmm2.slurm \
    output_bys.bam reference.fa output_aln.bam)

J3=$(sbatch --parsable --dependency=afterok:$J2 \
    slurm/callers/ipdsummary.slurm \
    output_aln.bam reference.fa output.gff output.csv)

J4=$(sbatch --parsable --dependency=afterok:$J3 \
    slurm/callers/pbmotifmaker.slurm \
    reference.fa output.gff output_motifs.csv)
```

In parallel for 5mC:

```bash
J5=$(sbatch --parsable --dependency=afterok:$J2 \
    slurm/callers/jasmine_modkit.slurm \
    output_aln.bam reference.fa output_jasmine.csv)
```

Final merged catalogue:

```bash
sbatch --dependency=afterok:$J4:$J5 \
    slurm/callers/merge_motifs.slurm \
    output_merged.csv 0.7 output_motifs.csv output_jasmine.csv
```

---

## Cluster partition guidance (IBU cluster)

| Partition | Time limit | Use case |
|---|---|---|
| `pibu_el8` | 28 days | Long callers (ipdSummary, jasmine + modkit) |
| `pshort_el8` | 2 hours | Smoke tests, quick prep on small inputs |
| `pgpu` | 28 days | `kinsim_nn train` and `kinsim_nn generate` |
| `phighmem` | 110 days | Memory-heavy refines on large corpora |

All scripts use SMRT-Link 25.1 via the Apptainer SIF for the PacBio
binaries (`ccs-kinetics-bystrandify`, `pbmm2`, `ipdSummary`,
`pbmotifmaker`). `jasmine` and `modkit` are invoked from the conda
environment.

---

## Logging conventions

All scripts route logs to a central directory through the
`#SBATCH --output=…/%x_%J.log` directive in their header. Each
script's header includes diagnostics (date, hostname, GPU info, SLURM
env vars, timing, exit code) for post-mortem inspection.
