#!/bin/bash
#SBATCH --job-name=ml_06_verify
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_06_verify_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 06 — verify generated BAMs match reference kinetics (array)
#   sbatch --array=1-N%K slurm_kinsim/ml/06_verify_generate.slurm \
#       <manifest.csv> <gen_dir> <out_dir>
#
# For each manifest row:
#   ref_bam = manifest.bam_path
#   gen_bam = <gen_dir>/<sample_id>_mlp.bam
#   motifs  = manifest.motifs
#   report  = <out_dir>/<sample_id>_verify.tsv
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

MANIFEST=${1:?"manifest CSV required"}
GEN_DIR=${2:?"generated BAM dir required"}
OUT_DIR=${3:?"output dir required"}

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST"; exit 1; }
[ -d "$GEN_DIR" ]  || { echo "ERROR: gen dir missing: $GEN_DIR"; exit 1; }
mkdir -p "$OUT_DIR"

ROW=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')
i = ${SLURM_ARRAY_TASK_ID} - 1
if i >= len(e): raise SystemExit(0)
x = e[i]
print(f'{x.sample_id}|{x.bam_path}|{x.motifs}')
")
[ -n "$ROW" ] || exit 0

SAMPLE_ID=$(echo "$ROW" | cut -d'|' -f1)
REF_BAM=$(echo "$ROW"   | cut -d'|' -f2)
MOTIFS=$(echo "$ROW"    | cut -d'|' -f3-)
GEN_BAM="${GEN_DIR}/${SAMPLE_ID}_mlp.bam"
REPORT="${OUT_DIR}/${SAMPLE_ID}_verify.tsv"

echo "=== ML 06/verify-generate — $SAMPLE_ID ==="
echo "REF:    $REF_BAM"
echo "GEN:    $GEN_BAM"
echo "MOTIFS: $MOTIFS"
echo "REPORT: $REPORT"
START=$(date +%s)

[ -f "$REF_BAM" ] || { echo "ERROR: reference BAM missing"; exit 1; }
[ -f "$GEN_BAM" ] || { echo "ERROR: generated BAM missing"; exit 1; }

kinsim verify-generate "$REF_BAM" "$GEN_BAM" "$MOTIFS" "$REPORT" -v

echo "Elapsed: $(( $(date +%s) - START ))s"
