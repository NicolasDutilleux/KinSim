#!/bin/bash
#SBATCH --job-name=strepto_03_ipd
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_03_ipd_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Strepto step 03 — ipdSummary SP3-C3 (m6A + m4C with methylFraction)
#   sbatch --array=1-N%4 slurm_kinsim/strepto/03_ipdsummary.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true
set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
MANIFEST=${STREPTO}/manifest_strepto.csv
OUTBASE=${STREPTO}/gff_pipeline
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

ROW=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')
i = ${SLURM_ARRAY_TASK_ID} - 1
if i >= len(e): raise SystemExit(0)
print(f'{e[i].sample_id}')
")
[ -n "$ROW" ] || exit 0

SAMPLE_ID="$ROW"
REF="${STREPTO}/${SAMPLE_ID}/final_assembly.fasta"
SPECIES_DIR="${OUTBASE}/${SAMPLE_ID}"
ALIGNED="${SPECIES_DIR}/${SAMPLE_ID}_aligned.bam"
GFF="${SPECIES_DIR}/${SAMPLE_ID}_ipdSummary.gff"
CSV="${SPECIES_DIR}/${SAMPLE_ID}_ipdSummary.csv"

echo "=== Strepto 03/ipdSummary — $SAMPLE_ID ==="
START=$(date +%s)

[ -f "$REF" ] || { echo "ERROR: reference missing: $REF"; exit 1; }
[ -f "${ALIGNED}.pbi" ] || { echo "ERROR: pbindex missing (run 02_index first)"; exit 1; }

if [ -s "$GFF" ]; then
    echo "SKIP — GFF exists ($(grep -c '^[^#]' "$GFF" || true) records)"; exit 0
fi

ipdSummary "$ALIGNED" \
    --reference "$REF" \
    --identify m6A,m4C \
    --methylFraction \
    --csv "$CSV" \
    --gff "$GFF" \
    --numWorkers "${SLURM_CPUS_PER_TASK}" \
    --ipdModel "$SP3_MODEL"

[ -s "$GFF" ] || { echo "WARNING: GFF empty"; exit 1; }
echo "GFF records: $(grep -c '^[^#]' "$GFF" || true)"
echo "Elapsed: $(( $(date +%s) - START ))s"
