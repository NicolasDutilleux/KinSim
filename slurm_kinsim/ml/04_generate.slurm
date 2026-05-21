#!/bin/bash
#SBATCH --job-name=ml_04_generate
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_04_generate_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 04 — generate synthetic BAMs from PBSIM3 reads (array)
#   sbatch --array=1-N slurm_kinsim/ml/04_generate.slurm \
#       <pbsim3_dir> <checkpoint.pt> <motifs> <out_dir>
#
# Output: <out_dir>/<species>_kinsim.bam
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

PBSIM=${1:?"pbsim3 dir required"}
CKPT=${2:?"checkpoint .pt required"}
MOTIFS=${3:?"motifs arg required"}
OUT=${4:?"output dir required"}

[ -d "$PBSIM" ] || { echo "ERROR: pbsim3 dir missing"; exit 1; }
[ -f "$CKPT" ]  || { echo "ERROR: checkpoint missing"; exit 1; }
mkdir -p "$OUT"

DEVICE="cpu"
nvidia-smi > /dev/null 2>&1 && DEVICE="cuda"

# Auto-detect flat vs subdir layout
if ls "${PBSIM}"/*.fq.gz 1>/dev/null 2>&1; then
    FQ=$(ls "${PBSIM}"/*.fq.gz | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SP=$(basename "$FQ" .fq.gz)
    MAF="${PBSIM}/${SP}.maf.gz"
    REF="${PBSIM}/${SP}.fna"
else
    SPD=$(ls -d "${PBSIM}"/*/ 2>/dev/null | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
    [ -n "$SPD" ] || { echo "No species dir at index $SLURM_ARRAY_TASK_ID"; exit 1; }
    SP=$(basename "$SPD")
    FQ=$(ls "${SPD}"*.fq.gz  2>/dev/null | head -1)
    MAF=$(ls "${SPD}"*.maf.gz 2>/dev/null | head -1)
    REF=$(ls "${SPD}"*.fna "${SPD}"*.fa "${SPD}"*.fasta 2>/dev/null | head -1)
fi

[ -f "$FQ" ]  || { echo "ERROR: FASTQ not found: $FQ"; exit 1; }
[ -f "$MAF" ] || { echo "ERROR: MAF not found: $MAF"; exit 1; }
[ -f "$REF" ] || { echo "ERROR: Reference not found: $REF"; exit 1; }

BAM_OUT="${OUT}/${SP}_kinsim.bam"

# Resolve per-species motifs file (species|motif_string lines)
if [ -f "$MOTIFS" ]; then
    SP_MOT=$(grep "^${SP}|" "$MOTIFS" 2>/dev/null | cut -d'|' -f2)
    MOT_ARG=${SP_MOT:-"$MOTIFS"}
else
    MOT_ARG="$MOTIFS"
fi

echo "=== ML 04/generate — $SP ==="
echo "FASTQ:  $FQ"
echo "REF:    $REF"
echo "CKPT:   $CKPT"
echo "MOTIFS: $MOT_ARG"
echo "OUT:    $BAM_OUT"
echo "DEVICE: $DEVICE"
START=$(date +%s)

kinsim generate "$FQ" "$MAF" "$REF" "$CKPT" "$MOT_ARG" "$BAM_OUT" \
    --device "$DEVICE" --batch-reads 1000

echo "Elapsed: $(( $(date +%s) - START ))s"
