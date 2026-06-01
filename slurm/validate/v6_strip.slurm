#!/bin/bash
#SBATCH --job-name=v6_strip
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# validate/v6_strip — bystrandified+aligned BAM → HiFi-like unaligned BAM
#
# Usage:
#   sbatch -J v6_strip_<strain> slurm/validate/v6_strip.slurm \
#          <input_bystrandified_bam> <out_dir>
#
# Output:
#   <out_dir>/<bc>_hifi_stripped.bam   (1 record/ZMW, unaligned, no kinetics)
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

IN_BAM=${1:?"Usage: sbatch $0 <input.bam> <out_dir>"}
OUT=${2:?"out_dir missing"}

mkdir -p "$OUT"
BC=$(basename "$IN_BAM" | sed -E 's/(_aligned)?\.bam$//')
STRIPPED="$OUT/${BC}_hifi_stripped.bam"

[ -f "$IN_BAM" ] || { echo "ERROR: input BAM missing: $IN_BAM"; exit 1; }

KINSIM_REPO=${KINSIM_REPO:-/data/users/ndutilleux/KinSim}
STRIP_PY="$KINSIM_REPO/scripts/strip_bystrandified_to_hifi.py"
[ -f "$STRIP_PY" ] || { echo "ERROR: strip script missing: $STRIP_PY"; exit 1; }

echo "========================================================"
echo "  validate/v6_strip — $BC"
echo "========================================================"
echo "Input BAM:  $IN_BAM"
echo "Output dir: $OUT"

if [ -s "$STRIPPED" ]; then
    echo "SKIP — stripped BAM exists: $STRIPPED"
else
    T=$(date +%s)
    python "$STRIP_PY" "$IN_BAM" "$STRIPPED"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$STRIPPED" ] || { echo "FAILED: strip produced empty BAM"; exit 1; }
fi
echo "Stripped BAM size: $(stat -c %s "$STRIPPED") B"
echo "Stripped reads:    $(samtools view -c "$STRIPPED")"
