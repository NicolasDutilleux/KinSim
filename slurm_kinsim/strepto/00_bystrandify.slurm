#!/bin/bash
#SBATCH --job-name=strepto_00_by
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_00_by_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Strepto step 00 — ccs-kinetics-bystrandify (fi/fp → ip/pw per strand)
#
# Manifest-driven: reads ${STREPTO}/manifest_strepto.csv; the array index
# selects a row (sample_id, bam_path, motifs).
#
#   sbatch --array=1-N%4 slurm_kinsim/strepto/00_bystrandify.slurm
#
# Output: ${STREPTO}/gff_pipeline/<sample_id>/<sample_id>_bystrandify.bam
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
MANIFEST=${STREPTO}/manifest_strepto.csv
OUTBASE=${STREPTO}/gff_pipeline
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST"; exit 1; }

ROW=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')
i = ${SLURM_ARRAY_TASK_ID} - 1
if i >= len(e): raise SystemExit(0)
print(f'{e[i].sample_id}|{e[i].bam_path}')
")
[ -n "$ROW" ] || { echo "array index past end of manifest"; exit 0; }

SAMPLE_ID=$(echo "$ROW" | cut -d'|' -f1)
BAM_PATH=$(echo "$ROW" | cut -d'|' -f2)
SPECIES_DIR="${OUTBASE}/${SAMPLE_ID}"
BYSTRAND="${SPECIES_DIR}/${SAMPLE_ID}_bystrandify.bam"

mkdir -p "$SPECIES_DIR"

echo "=== Strepto 00/bystrandify — $SAMPLE_ID ==="
START=$(date +%s)

[ -f "$BAM_PATH" ] || { echo "ERROR: input BAM missing: $BAM_PATH"; exit 1; }

if [ -s "$BYSTRAND" ]; then
    echo "SKIP — bystrandify BAM exists"; exit 0
fi

apptainer exec --bind /data "$SMRT_SIF" \
    ccs-kinetics-bystrandify "$BAM_PATH" "$BYSTRAND"

[ -s "$BYSTRAND" ] || { echo "FAILED: bystrandify produced no output"; exit 1; }
echo "Elapsed: $(( $(date +%s) - START ))s"
