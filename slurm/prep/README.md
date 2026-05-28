# `slurm/prep/` — shared preprocessing helpers

Scripts that transform a raw HiFi BAM into an aligned, indexed BAM
ready to be consumed by the callers in `../callers/`. Callable from
any dataset orchestration via positional arguments — no hard-coded
paths.

| Script | Input | Output |
|---|---|---|
| `assembly_hifiasm.slurm` | raw HiFi BAM | assembly FASTA |
| `bystrandify.slurm` | raw HiFi BAM (fi/fp/ri/rp) | bystrandified BAM (ip/pw) |
| `align_pbmm2.slurm` | any HiFi BAM + reference | aligned sorted BAM |
| `index_bam.slurm` | aligned BAM | .bai + .pbi sidecars |

Toutes les invocations PacBio passent par la SIF apptainer
`/containers/apptainer/pacbio-smrt-tools-25.3.sif`.
