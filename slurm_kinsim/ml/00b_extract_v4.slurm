#!/bin/bash
#SBATCH --job-name=ml_00b_extract_v4
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_00b_extract_v4_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 00b — v4 extract using a refined master_clean.pkl
#
#   sbatch --array=1-N%K slurm_kinsim/ml/00b_extract_v4.slurm \
#       <manifest.csv> <shards_v4_dir> <master_clean.pkl>
#
# Output: <shards_v4_dir>/<sample_id>_shard_v4.pkl
#
# The v4 extract emits METH samples at confirmed methylation centers
# (kmers that survived the pass-1 GMM in master_clean.pkl), SLOWED
# samples at signature offsets of those (from kinsim_config.yaml), and
# BASELINE samples elsewhere capped per kmer.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

MANIFEST=${1:?"manifest CSV required"}
SHARDS=${2:?"shards_v4 dir required"}
REFINED=${3:?"master_clean.pkl required"}

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST"; exit 1; }
[ -s "$REFINED"  ] || { echo "ERROR: refined pkl missing: $REFINED"; exit 1; }
mkdir -p "$SHARDS"

echo "=== ML 00b/extract-v4 — task ${SLURM_ARRAY_TASK_ID} ==="
echo "Manifest: $MANIFEST"
echo "Refined:  $REFINED"
echo "Output:   $SHARDS"
START=$(date +%s)

kinsim extract \
    --manifest    "$MANIFEST" \
    --task        "$SLURM_ARRAY_TASK_ID" \
    --output-dir  "$SHARDS" \
    --refined-pkl "$REFINED"

echo "Elapsed: $(( $(date +%s) - START ))s"
