#!/bin/bash
#SBATCH --job-name=strepto_02_idx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_02_idx_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Strepto step 02 — samtools index + pbindex
#   sbatch --array=1-N%8 slurm_kinsim/strepto/02_index.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
MANIFEST=${STREPTO}/manifest_strepto.csv
OUTBASE=${STREPTO}/gff_pipeline
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

ROW=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')
i = ${SLURM_ARRAY_TASK_ID} - 1
if i >= len(e): raise SystemExit(0)
print(f'{e[i].sample_id}')
")
[ -n "$ROW" ] || exit 0

SAMPLE_ID="$ROW"
ALIGNED="${OUTBASE}/${SAMPLE_ID}/${SAMPLE_ID}_aligned.bam"

echo "=== Strepto 02/index — $SAMPLE_ID ==="
[ -f "$ALIGNED" ] || { echo "ERROR: aligned BAM missing: $ALIGNED"; exit 1; }

[ -f "${ALIGNED}.bai" ] || samtools index "$ALIGNED"
[ -f "${ALIGNED}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$ALIGNED"

echo "OK"
