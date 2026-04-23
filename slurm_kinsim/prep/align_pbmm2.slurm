#!/bin/bash
#SBATCH --job-name=prep_align
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# prep/align_pbmm2 — pbmm2 CCS alignment via SMRT-Tools SIF
#
# Usage:
#   sbatch -J prep_aln_<sample> slurm_kinsim/prep/align_pbmm2.slurm \
#          <input_bam> <reference_fasta> <output_bam>
#
# Preserves kinetics tags (fi/fp/ri/rp or ip/pw) and MM/ML tags when present.
# Must go through apptainer SIF — conda pbmm2 26.x is incompatible with SIF
# bystrandify output (see feedback_apptainer_first memory).
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

IN_BAM=${1:?"Usage: sbatch $0 <input_bam> <reference_fasta> <output_bam>"}
REF=${2:?"reference missing"}
OUT_BAM=${3:?"output_bam missing"}

SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
[ -f "$IN_BAM" ] || { echo "ERROR: input BAM missing: $IN_BAM"; exit 1; }
[ -f "$REF" ]    || { echo "ERROR: reference missing: $REF"; exit 1; }
[ -f "${REF}.fai" ] || samtools faidx "$REF"

mkdir -p "$(dirname "$OUT_BAM")"

echo "========================================================"
echo "  prep/align_pbmm2 — $(basename "$OUT_BAM")"
echo "========================================================"
echo "Input BAM: $IN_BAM"
echo "Reference: $REF"
echo "Out BAM:   $OUT_BAM"

if [ -s "$OUT_BAM" ]; then
    echo "SKIP — aligned BAM exists"
    exit 0
fi

# pbmm2 may need .pbi on input; create if missing.
if [ ! -f "${IN_BAM}.pbi" ]; then
    echo "=== pbindex input (pbmm2 requires) ==="
    apptainer exec --bind /data "$SMRT_SIF" pbindex "$IN_BAM"
fi

T=$(date +%s)
apptainer exec --bind /data "$SMRT_SIF" \
    pbmm2 align "$REF" "$IN_BAM" "$OUT_BAM" \
    --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
echo "Elapsed: $(( $(date +%s) - T ))s"

[ -s "$OUT_BAM" ] || { echo "FAILED: pbmm2 empty output"; exit 1; }
samtools flagstat "$OUT_BAM" | head -5
