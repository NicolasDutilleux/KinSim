#!/bin/bash
#SBATCH --job-name=prep_index
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# prep/index_bam — samtools index (.bai) + pbindex (.pbi)
#
# Usage:
#   sbatch -J prep_idx_<sample> slurm_kinsim/prep/index_bam.slurm <bam>
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

BAM=${1:?"Usage: sbatch $0 <bam>"}
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
[ -f "$BAM" ] || { echo "ERROR: BAM missing: $BAM"; exit 1; }

echo "=== prep/index_bam — $(basename "$BAM") ==="
[ -f "${BAM}.bai" ] || samtools index "$BAM"
[ -f "${BAM}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$BAM"
ls -l "$BAM"{,.bai,.pbi} 2>/dev/null | awk '{print "  "$NF" "$5}'
