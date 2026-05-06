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
# The manifest CSV MUST have these columns (all required):
#       sample_id,bam_path,motifs,ref_path
#
# kinsim extract is aligned-only: bam_path must be an aligned bystrandified
# BAM (sorted, indexed, ip/pw tags) and ref_path is the reference FASTA
# the BAM was mapped against. read.is_reverse drives per-strand methylation
# routing — raw HiFi without alignment is not supported (orientation
# ambiguity averages the per-row signal away).
#
# Output layout: 38-col rows per kmer (see kinsim/utils/sample_layout.py).
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
