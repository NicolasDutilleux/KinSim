#!/bin/bash
# ============================================================
# 01_analyze_array.sh — array-task helper for `kinsim analyze`.
#
# Submitted by _v11_orchestrator.sh as a SLURM array. Each task
# picks one .pkl from <input_dir> by SLURM_ARRAY_TASK_ID and writes
# its report to <output_root>/<sample_id>/.
#
# Args:
#   $1  Input directory (extract shards/ or refined refined/)
#   $2  Output root (one subdir per sample)
#
# Usage (from orchestrator):
#   sbatch --array=1-N --wrap="bash <repo>/slurm_kinsim/ml/01_analyze_array.sh \\
#       /path/to/shards /path/to/reports/extract"
# ============================================================
set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

IN_DIR=${1:?"input dir (shards/ or refined/) required"}
OUT_ROOT=${2:?"output root required"}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set — must run as an array task" >&2
    exit 1
fi

# Stable, sorted enumeration of .pkl files under IN_DIR.
PKL=$(ls "$IN_DIR"/*.pkl 2>/dev/null | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
if [ -z "$PKL" ] || [ ! -f "$PKL" ]; then
    echo "ERROR: no pkl at array index $SLURM_ARRAY_TASK_ID in $IN_DIR" >&2
    exit 1
fi

SAMPLE=$(basename "$PKL" .pkl)
OUTDIR="$OUT_ROOT/$SAMPLE"
mkdir -p "$OUTDIR"

echo "=== analyze array task $SLURM_ARRAY_TASK_ID ==="
echo "Input:  $PKL"
echo "Output: $OUTDIR"
START=$(date +%s)

kinsim analyze "$PKL" --output-dir "$OUTDIR"

echo "Elapsed: $(( $(date +%s) - START ))s"
