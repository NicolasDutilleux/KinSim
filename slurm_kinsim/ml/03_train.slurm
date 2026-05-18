#!/bin/bash
#SBATCH --job-name=ml_03_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --account=p774
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# Note: --mem must accommodate the BIGGEST shard × num_workers + the main
# process scanning shards at startup. v12 corpus has vega_bc2041 at ~330M
# rows ≈ 30-40 GB unpickled. With 2 DataLoader workers loading concurrently
# AND the main process keeping the previous shard alive during transitions,
# peak RSS can hit 100+ GB. 192 GB gives generous headroom.
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/ml_03_train_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# ML step 03 — train ConvPredictor on refined master .pkl
#   sbatch slurm_kinsim/ml/03_train.slurm <master.pkl> <ckpt_dir> [extra_flags]
#   sbatch slurm_kinsim/ml/03_train.slurm --config config.yaml [extra_flags]
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

echo "=== ML 03/train ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
START=$(date +%s)

if [ "${1:-}" = "--config" ]; then
    kinsim train "$@"
else
    IN_PKL=${1:?"input .pkl required"}
    CKPT_DIR=${2:?"checkpoint dir required"}
    shift 2
    mkdir -p "$CKPT_DIR"
    kinsim train "$IN_PKL" "$CKPT_DIR" "$@"
fi

echo "Elapsed: $(( $(date +%s) - START ))s"
