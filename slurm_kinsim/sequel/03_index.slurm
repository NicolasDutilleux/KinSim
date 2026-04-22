#!/bin/bash
#SBATCH --job-name=sequel_03_idx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_03_idx_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel step 03 — samtools index + pbindex
#   sbatch --array=1-48%8 slurm_kinsim/sequel/03_index.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
OUTBASE=${SEQUEL}/ipdsummary_ccs
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

mapfile -t GZ_BAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
[ $IDX -lt ${#GZ_BAMS[@]} ] || exit 0

BC=$(basename "${GZ_BAMS[$IDX]}" .bam.gz | grep -oP 'bc\d+' | head -1)
ALIGNED="${OUTBASE}/${BC}/${BC}_aligned.bam"

echo "=== Sequel 03/index — $BC ==="
[ -f "$ALIGNED" ] || { echo "ERROR: aligned BAM missing: $ALIGNED"; exit 1; }

[ -f "${ALIGNED}.bai" ] || samtools index "$ALIGNED"
[ -f "${ALIGNED}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$ALIGNED"

echo "OK"
