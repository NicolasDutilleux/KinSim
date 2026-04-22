#!/bin/bash
#SBATCH --job-name=strepto_01_aln
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_01_aln_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Strepto step 01 — pbmm2 align (bystrandified BAM → pre-existing reference)
#   sbatch --array=1-N%4 slurm_kinsim/strepto/01_align.slurm
#
# Reference: ${STREPTO}/<sample_id>/final_assembly.fasta
# Output:    ${STREPTO}/gff_pipeline/<sample_id>/<sample_id>_aligned.bam
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
MANIFEST=${STREPTO}/manifest_strepto.csv
OUTBASE=${STREPTO}/gff_pipeline

ROW=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')
i = ${SLURM_ARRAY_TASK_ID} - 1
if i >= len(e): raise SystemExit(0)
print(f'{e[i].sample_id}')
")
[ -n "$ROW" ] || exit 0

SAMPLE_ID="$ROW"
SPECIES_DIR="${OUTBASE}/${SAMPLE_ID}"
REF="${STREPTO}/${SAMPLE_ID}/final_assembly.fasta"
BYSTRAND="${SPECIES_DIR}/${SAMPLE_ID}_bystrandify.bam"
ALIGNED="${SPECIES_DIR}/${SAMPLE_ID}_aligned.bam"

echo "=== Strepto 01/align — $SAMPLE_ID ==="
START=$(date +%s)

[ -f "$REF" ]      || { echo "ERROR: reference missing: $REF"; exit 1; }
[ -f "$BYSTRAND" ] || { echo "ERROR: bystrandify missing: $BYSTRAND"; exit 1; }

if [ -s "$ALIGNED" ]; then
    echo "SKIP — aligned BAM exists"; exit 0
fi

[ -f "${REF}.fai" ] || samtools faidx "$REF"

pbmm2 align "$REF" "$BYSTRAND" "$ALIGNED" \
    --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"

[ -s "$ALIGNED" ] || { echo "FAILED: pbmm2 produced no output"; exit 1; }
samtools flagstat "$ALIGNED" | head -5
echo "Elapsed: $(( $(date +%s) - START ))s"
