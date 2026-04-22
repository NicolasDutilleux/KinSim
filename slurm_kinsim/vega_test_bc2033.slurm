#!/bin/bash
#SBATCH --job-name=vega_test_bc2033
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_test_bc2033_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Vega bc2033 single-species iteration test
#
# Runs: extract (--min-score 50) -> analyze -> compute_ceiling
# One step at a time, within a single job.
#
# Usage:
#   sbatch slurm_kinsim/vega_test_bc2033.slurm
#   sbatch slurm_kinsim/vega_test_bc2033.slurm 100   # override min-score
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

MIN_SCORE=${1:-50}

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
BC=bc2033
TEST=${VEGA}/test_${BC}_s${MIN_SCORE}
SHARD=${TEST}/shards/${BC}_s${MIN_SCORE}_shard.pkl

mkdir -p "$TEST/shards" "$TEST/analyze_report"

echo "========================================================"
echo "  Vega bc2033 test  (min-score ${MIN_SCORE})"
echo "========================================================"
echo "Date:      $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:      $(hostname)"
echo "Test dir:  $TEST"
echo "========================================================"
echo ""

# 1-row manifest
cat > "$TEST/manifest.csv" <<EOF
sample_id,bam_path,motifs,gff
${BC}_s${MIN_SCORE},${VEGA}/assembly/${BC}/${BC}_aligned.bam,,${VEGA}/assembly/${BC}/${BC}_ipdSummary.gff
EOF

# ---- Step 1: extract ----
echo "=== Step 1: extract (min-score ${MIN_SCORE}) ==="
START=$(date +%s)
kinsim extract \
    --manifest "$TEST/manifest.csv" \
    --task 1 \
    --output-dir "$TEST/shards" \
    --min-score "$MIN_SCORE"
echo "extract done in $(( $(date +%s) - START ))s"
echo ""

# ---- Step 2: analyze ----
echo "=== Step 2: analyze ==="
START=$(date +%s)
kinsim analyze "$SHARD" --output-dir "$TEST/analyze_report"
echo "analyze done in $(( $(date +%s) - START ))s"
echo ""

# ---- Step 3: ceiling ----
echo "=== Step 3: compute_ceiling --per-meth ==="
START=$(date +%s)
python slurm_kinsim/compute_ceiling.py "$SHARD" --per-meth
echo "ceiling done in $(( $(date +%s) - START ))s"
echo ""

echo "========================================================"
echo "  COMPLETE"
echo "========================================================"
echo "  Shard:   $SHARD"
echo "  Report:  $TEST/analyze_report"
echo "  Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
