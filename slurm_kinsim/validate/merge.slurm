#!/bin/bash
#SBATCH --job-name=val_merge
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#
# validate/merge — concat all shard BAMs into one + run pbindex
# (required by ipdSummary downstream).
#
# Usage:
#   sbatch slurm_kinsim/validate/merge.slurm <SHARD_DIR> <OUT_BAM>
set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SHARD_DIR=${1:?shard dir required}
OUT_BAM=${2:?output BAM required}

# Match only canonical shard_NNN.bam — NOT shard_NNN_*.bam per-region
# intermediates that may linger from earlier failed generate runs.
ls "$SHARD_DIR"/shard_[0-9][0-9][0-9].bam
# -c (combine-rg) collapses identical @RG IDs across shards into one.
samtools merge -c -@ "$SLURM_CPUS_PER_TASK" -f "$OUT_BAM" "$SHARD_DIR"/shard_[0-9][0-9][0-9].bam

apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.3.sif \
          pbindex "$OUT_BAM"
[ -f "${OUT_BAM}.pbi" ] || { echo "ERROR: pbindex did not produce .pbi" >&2; exit 1; }
echo "Merged + indexed: $OUT_BAM"
