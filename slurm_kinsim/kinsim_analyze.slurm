#!/bin/bash
#SBATCH --job-name=KinSim_Analyze
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=pshort_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_analyze_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim Dictionary — Analyze coverage statistics (single job)
#
# Usage:
#   sbatch kinsim_analyze.slurm <dict.pkl>
#
# Arguments:
#   $1  Path to trained dictionary (.pkl file)
#
# Example:
#   sbatch kinsim_analyze.slurm /data/kinsim/dicts/master_dict.pkl
#
# Expected directory structure:
#   project/
#     dicts/
#       master_dict.pkl           <- merged dictionary to analyze
#     logs/                       <- SLURM logs
# ============================================================

# ---- Usage check ----
if [ -z "$1" ]; then
    echo "Usage: sbatch kinsim_analyze.slurm <dict.pkl>"
    echo ""
    echo "Arguments:"
    echo "  dict.pkl    Path to trained dictionary (.pkl file)"
    echo ""
    echo "Expected structure:"
    echo "  project/"
    echo "    dicts/master_dict.pkl"
    echo "    logs/"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Config ----
PKL_PATH="$1"

if [ ! -f "$PKL_PATH" ]; then
    echo "ERROR: Dictionary file '$PKL_PATH' not found."
    exit 1
fi

echo "Analyzing: $PKL_PATH"

# ---- Run ----
kinsim dictionary analyze "$PKL_PATH"
