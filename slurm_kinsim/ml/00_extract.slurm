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
# The manifest CSV MUST have these columns:
#       sample_id,bam_path,motifs[,ref_path]
#
# When the OPTIONAL `ref_path` column is provided AND non-empty for
# a row, kinsim extract auto-dispatches to the orientation-aware
# aligned-BAM path (kinsim/extract_aligned.py). This is the correct
# path for HiFi data because raw fi/ri without alignment cannot
# disambiguate which kinetic tag carries the methylation signal at
# each genomic position — half the reads end up reading the wrong
# tag and the per-row signal averages away.
#
# When `ref_path` is absent or empty, the legacy raw-HiFi extract
# path runs (motif scan on read sequence + fi/fp). Use this only for
# Sequel subreads or BAMs where strand orientation is unambiguous.
#
# Output layout: 38-col rows per kmer (see kinsim/utils/sample_layout.py).
# Refine on the resulting shards is identical for both paths; only
# the underlying source of per-row IPDs differs.
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
