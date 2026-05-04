#!/bin/bash
#SBATCH --job-name=ml_02_refine
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_02_refine_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 02 — refine master .pkl: drop CATEGORY_SLOWED samples
#              below the per-kmer-mean baseline IPD percentile.
#   sbatch slurm_kinsim/ml/02_refine.slurm <master.pkl> <master_clean.pkl>
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

IN_PKL=${1:?"input master .pkl required"}
OUT_PKL=${2:?"output master_clean .pkl required"}

echo "=== ML 02/refine ==="
echo "Input:  $IN_PKL"
echo "Output: $OUT_PKL"
START=$(date +%s)

kinsim refine "$IN_PKL" "$OUT_PKL" -v

echo "Elapsed: $(( $(date +%s) - START ))s"
