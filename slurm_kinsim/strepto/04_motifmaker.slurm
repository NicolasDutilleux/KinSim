#!/bin/bash
#SBATCH --job-name=strepto_04_mot
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/strepto_04_mot_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Strepto step 04 — pbmotifmaker find (GFF → motifs.csv)
#   sbatch --array=1-N%8 slurm_kinsim/strepto/04_motifmaker.slurm
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
REF="${STREPTO}/${SAMPLE_ID}/final_assembly.fasta"
SPECIES_DIR="${OUTBASE}/${SAMPLE_ID}"
GFF="${SPECIES_DIR}/${SAMPLE_ID}_ipdSummary.gff"
MOTIFS="${SPECIES_DIR}/${SAMPLE_ID}_motifs.csv"

echo "=== Strepto 04/motifmaker — $SAMPLE_ID ==="

[ -f "$REF" ] || { echo "ERROR: reference missing"; exit 1; }
[ -f "$GFF" ] || { echo "ERROR: GFF missing"; exit 1; }

if [ -s "$MOTIFS" ]; then
    echo "SKIP — motifs.csv exists ($(tail -n +2 "$MOTIFS" | wc -l) motifs)"; exit 0
fi

apptainer exec --bind /data "$SMRT_SIF" \
    pbmotifmaker find "$REF" "$GFF" "$MOTIFS"

[ -s "$MOTIFS" ] || { echo "WARNING: motifs.csv empty"; exit 1; }
echo "Motifs: $(tail -n +2 "$MOTIFS" | wc -l)"
cat "$MOTIFS"
