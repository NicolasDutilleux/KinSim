#!/bin/bash
#SBATCH --job-name=ml_05_eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_05_eval_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 05 — calibration report on held-out .pkl
#   sbatch slurm_kinsim/ml/05_evaluate.slurm <ckpt_dir> <master.pkl> [extra_flags]
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

CKPT_DIR=${1:?"ckpt dir required"}
PKL=${2:?".pkl required"}
shift 2

[ -d "$CKPT_DIR" ] || { echo "ERROR: ckpt dir missing: $CKPT_DIR"; exit 1; }
[ -f "$CKPT_DIR/model_config.json" ] || { echo "ERROR: model_config.json missing"; exit 1; }
[ -f "$PKL" ] || { echo "ERROR: pkl missing: $PKL"; exit 1; }

DEVICE="cpu"
nvidia-smi > /dev/null 2>&1 && DEVICE="cuda"

echo "=== ML 05/evaluate ==="
echo "Ckpt: $CKPT_DIR   Pkl: $PKL   Device: $DEVICE"
START=$(date +%s)

kinsim evaluate "$CKPT_DIR" "$PKL" --device "$DEVICE" "$@"

echo "Elapsed: $(( $(date +%s) - START ))s"
