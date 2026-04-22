#!/bin/bash
#SBATCH --job-name=sequel_01_by
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_01_by_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel step 01 — ccs-kinetics-bystrandify (fi/fp → ip/pw per strand)
#
# Usage:
#   sbatch --array=1-48%4 slurm_kinsim/sequel/01_bystrandify.slurm
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
HIFI="${OUTBASE}/${BC}/${BC}_hifi.bam"
BYSTRAND="${OUTBASE}/${BC}/${BC}_bystrandify.bam"

echo "=== Sequel 01/bystrandify — $BC ==="
[ -f "$HIFI" ] || { echo "ERROR: HiFi BAM missing: $HIFI"; exit 1; }

if [ -f "$BYSTRAND" ] && [ -s "$BYSTRAND" ]; then
    echo "SKIP — bystrandify BAM exists"; exit 0
fi

apptainer exec --bind /data "$SMRT_SIF" \
    ccs-kinetics-bystrandify "$HIFI" "$BYSTRAND"

[ -s "$BYSTRAND" ] || { echo "FAILED: bystrandify empty"; exit 1; }
echo "OK"
