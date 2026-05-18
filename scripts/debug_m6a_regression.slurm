#!/bin/bash
#SBATCH --job-name=dbg_m6a
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --account=p774
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/debug_m6a_%J.log

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

python /data/users/ndutilleux/KinSim/scripts/debug_m6a_regression.py
