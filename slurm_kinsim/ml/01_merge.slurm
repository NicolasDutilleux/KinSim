#!/bin/bash
#SBATCH --job-name=ml_01_merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_01_merge_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 01 — merge shards into master .pkl
#   sbatch slurm_kinsim/ml/01_merge.slurm <shards_dir> <master.pkl>
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SHARDS=${1:?"shards dir required"}
MASTER=${2:?"master .pkl path required"}

[ -d "$SHARDS" ] || { echo "ERROR: shards dir missing: $SHARDS"; exit 1; }
mkdir -p "$(dirname "$MASTER")"

echo "=== ML 01/merge ==="
echo "Shards: $SHARDS  ($(ls "$SHARDS"/*_shard.pkl 2>/dev/null | wc -l) files)"
echo "Master: $MASTER"
START=$(date +%s)

kinsim merge "$SHARDS" "$MASTER"

echo "Master size: $(ls -lh "$MASTER" | awk '{print $5}')"
echo "Elapsed: $(( $(date +%s) - START ))s"
