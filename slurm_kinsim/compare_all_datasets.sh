#!/bin/bash
#SBATCH --job-name=kinsim_compare
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_compare_%J.log

# ============================================================
# Cross-dataset kinetic comparison
#
# Compares IPD/PW distributions across all datasets to evaluate
# chemistry and methylome differences:
#   - Streptomyces (Revio R/P1-C1) — 52 strains
#   - Vega HMB-16  (Vega R/P1-C1)  — 16 strains (reference mock)
#   - Sequel II    (SP2)           — 48 strains
#
# Output: text report + CSV + HTML (Plotly interactive plots)
#
# Usage:
#   sbatch slurm_kinsim/compare_all_datasets.sh
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

BASE=/data/projects/p774_MARSD/NDutilleux/training
OUTDIR=${BASE}/comparison_$(date +%Y%m%d)
mkdir -p "$OUTDIR"

STREPTO=${BASE}/Strepto/master_raw.pkl
VEGA=${BASE}/Vega/master_raw.pkl
SEQUEL=${BASE}/Sequel/master_raw.pkl

echo "========================================================"
echo "  Cross-Dataset Kinetic Comparison"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# Sanity check — only compare what exists
ARGS=()
for label_pkl in "Strepto:${STREPTO}" "Vega:${VEGA}" "Sequel:${SEQUEL}"; do
    label=${label_pkl%%:*}
    pkl=${label_pkl#*:}
    if [ -s "$pkl" ]; then
        echo "  ✓ $label  -> $pkl ($(du -h "$pkl" | cut -f1))"
        ARGS+=(--label "$label" "$pkl")
    else
        echo "  ✗ $label  -> NOT FOUND ($pkl)"
    fi
done

if [ ${#ARGS[@]} -lt 4 ]; then
    echo "ERROR: Need at least 2 datasets (4 args). Got ${#ARGS[@]}."
    exit 1
fi

echo ""
echo "Output dir: $OUTDIR"
echo ""

kinsim compare "${ARGS[@]}" --output-dir "$OUTDIR" --bimodality

echo ""
echo "========================================================"
echo "  Comparison complete"
echo "========================================================"
echo "Reports:"
ls -lh "$OUTDIR"/*
