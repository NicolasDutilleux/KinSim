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
# ML step 00b — v4 standalone extract
#
#   sbatch --array=1-N%K slurm_kinsim/ml/00b_extract_v4.slurm \
#       <manifest.csv> <shards_v4_dir>
#
# Output: <shards_v4_dir>/<sample_id>_shard_v4.pkl
#
# Single-pass v4 extract:
#   - METH samples at every motif-match center
#   - SLOWED samples at signature offsets (per kinsim_config.yaml)
#   - BASELINE samples elsewhere, capped at YAML extract.n_baseline_per_kmer
#
# False-positive motifs are dropped downstream by `kinsim refine` (GMM
# pass-1 on meth pool + p95 pass-2 on slowed). No bootstrap required.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

MANIFEST=${1:?"manifest CSV required"}
SHARDS=${2:?"shards_v4 dir required"}

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST"; exit 1; }
mkdir -p "$SHARDS"

echo "=== ML 00b/extract-v4 — task ${SLURM_ARRAY_TASK_ID} ==="
echo "Manifest: $MANIFEST"
echo "Output:   $SHARDS"
START=$(date +%s)

kinsim extract \
    --v4 \
    --manifest    "$MANIFEST" \
    --task        "$SLURM_ARRAY_TASK_ID" \
    --output-dir  "$SHARDS"

echo "Elapsed: $(( $(date +%s) - START ))s"
