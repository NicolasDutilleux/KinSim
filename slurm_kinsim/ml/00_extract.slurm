#!/bin/bash
#SBATCH --job-name=ml_00_extract
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_00_extract_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 00 — extract per-sample shards
#   sbatch --array=1-N%K slurm_kinsim/ml/00_extract.slurm \
#       <manifest.csv> <shards_dir>
#
# Output: <shards_dir>/<sample_id>_shard.pkl
#
# Single-pass extract: emits BASELINE / SLOWED / NEAR_METH samples in
# the 36-col layout per kmer (see kinsim/utils/sample_layout.py).
# False-positive motif matches survive this pass; `kinsim refine`
# drops them downstream via the per-kmer-mean baseline IPD threshold.
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
echo "Manifest: $MANIFEST"
echo "Output:   $SHARDS"
START=$(date +%s)

kinsim extract \
    --manifest    "$MANIFEST" \
    --task        "$SLURM_ARRAY_TASK_ID" \
    --output-dir  "$SHARDS"

echo "Elapsed: $(( $(date +%s) - START ))s"
