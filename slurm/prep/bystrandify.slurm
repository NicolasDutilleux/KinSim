#!/bin/bash
#SBATCH --job-name=prep_bystrandify
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# prep/bystrandify — ccs-kinetics-bystrandify (fi/fp/ri/rp → ip/pw per strand)
#
# Usage:
#   sbatch -J prep_by_<sample> slurm/prep/bystrandify.slurm \
#          <raw_bam> <out_bam>
#
# Runs the SMRT-Tools SIF. Produces a pbindex sidecar (.pbi) alongside
# the output BAM since downstream pbmm2 requires it.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

RAW_BAM=${1:?"Usage: sbatch $0 <raw_bam> <out_bam>"}
OUT_BAM=${2:?"out_bam missing"}

SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
mkdir -p "$(dirname "$OUT_BAM")"

echo "========================================================"
echo "  prep/bystrandify — $(basename "$OUT_BAM")"
echo "========================================================"
echo "Raw BAM: $RAW_BAM"
echo "Out BAM: $OUT_BAM"

# SKIP first — if the bystrandified output exists we don't need raw_bam.
# (Datasets with archived raw BAMs may have only the bystrandified BAM
# kept on disk.)
if [ -s "$OUT_BAM" ]; then
    echo "SKIP — output BAM exists"
else
    [ -f "$RAW_BAM" ] || { echo "ERROR: raw BAM missing: $RAW_BAM"; exit 1; }
    T=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$RAW_BAM" "$OUT_BAM"
    echo "Elapsed: $(( $(date +%s) - T ))s"
fi
[ -s "$OUT_BAM" ] || { echo "FAILED: empty output"; exit 1; }

[ -f "${OUT_BAM}.pbi" ] || {
    echo "=== pbindex ==="
    apptainer exec --bind /data "$SMRT_SIF" pbindex "$OUT_BAM"
}
echo "Done. Sizes:"
ls -l "$OUT_BAM"{,.pbi} 2>/dev/null | awk '{print "  "$NF" "$5}'
