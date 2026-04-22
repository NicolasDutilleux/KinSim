#!/bin/bash
#SBATCH --job-name=ml_00_extract
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_00_extract_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 00 — extract per-species shards from a GFF manifest
#   sbatch --array=1-N%K slurm_kinsim/ml/00_extract.slurm <manifest.csv> <shards_dir>
#
# Output: <shards_dir>/<sample_id>_shard.pkl
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

MANIFEST=${1:?"manifest CSV required"}
SHARDS=${2:?"shards dir required"}

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST"; exit 1; }
mkdir -p "$SHARDS"

echo "=== ML 00/extract — task ${SLURM_ARRAY_TASK_ID} ==="
START=$(date +%s)

kinsim extract \
    --manifest "$MANIFEST" \
    --task     "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$SHARDS" \
    --max-samples 1000 \
    --no-binarize

echo "Elapsed: $(( $(date +%s) - START ))s"
