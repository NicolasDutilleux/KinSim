#!/bin/bash
#SBATCH --job-name=KinSim_Analyze
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=pshort_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_analyze_%j.log

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Args ----
PKL_PATH=${1:?"Usage: sbatch kinsim_analyze.slurm <dict.pkl>"}

echo "Analyzing: $PKL_PATH"

# ---- Run ----
python -m kinsim.dictionary.analyze "$PKL_PATH"
