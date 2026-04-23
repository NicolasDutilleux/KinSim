# slurm_kinsim/prep/ — shared preprocessing helpers

Scripts génériques qui transforment un raw HiFi BAM en un BAM aligné+indexé
prêt à être callé par `callers/`. Appelables depuis n'importe quel dataset
(vega, sequel, strepto) via arguments positionnels — pas de hardcoding de chemins.

| Script | Input | Output |
|---|---|---|
| `assembly_hifiasm.slurm` | raw HiFi BAM | assembly FASTA |
| `bystrandify.slurm` | raw HiFi BAM (fi/fp/ri/rp) | bystrandified BAM (ip/pw) |
| `align_pbmm2.slurm` | any HiFi BAM + reference | aligned sorted BAM |
| `index_bam.slurm` | aligned BAM | .bai + .pbi sidecars |

Toutes les invocations PacBio passent par la SIF apptainer
`/containers/apptainer/pacbio-smrt-tools-25.3.sif`.
