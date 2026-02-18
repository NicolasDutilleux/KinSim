#!/bin/bash
#SBATCH --job-name=KinSim_cGAN_Train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_cgan_train_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim cGAN — Train conditional GAN (WGAN-GP) on kinetic data
#
# This script trains the Generator and Discriminator on merged raw sample data.
# Requires a single GPU for efficient training. Outputs checkpoints and logs.
#
# Usage:
#   sbatch kinsim_cgan_train.slurm <merged_pkl> <checkpoint_dir> [--resume <ckpt.pt>]
#
# Arguments:
#   $1  Merged training data .pkl (from: kinsim cgan merge)
#   $2  Output directory for checkpoints and logs
#   $@  Additional flags (e.g., --epochs, --kmer-embed-dim, --resume)
#
# Example:
#   sbatch kinsim_cgan_train.slurm master_cgan_data.pkl ./checkpoints/ --epochs 100 --batch-size 4096
#
#   # Resume from checkpoint:
#   sbatch kinsim_cgan_train.slurm master_cgan_data.pkl ./checkpoints/ \
#       --resume ./checkpoints/checkpoint_epoch50.pt --epochs 100
#
# Output structure:
#   checkpoint_dir/
#     model_config.json           <- Architecture hyperparameters
#     checkpoint_epoch10.pt       <- Model weights + optimizer state
#     checkpoint_epoch20.pt
#     ...
#     runs/                       <- TensorBoard logs
#     training_log.csv            <- CSV fallback (if TensorBoard unavailable)
#
# Monitor training:
#   tensorboard --logdir checkpoint_dir/runs
#
# Memory estimate: ~1 GB for 64-dim embeddings, ~0.5 GB with --kmer-embed-dim 32
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage:"
    echo "  sbatch kinsim_cgan_train.slurm <merged_pkl> <checkpoint_dir> [--resume <ckpt.pt>] [...]"
    echo ""
    echo "Arguments:"
    echo "  merged_pkl         Merged training data .pkl (from: kinsim cgan merge)"
    echo "  checkpoint_dir     Output directory for checkpoints and logs"
    echo "  Additional flags:  --epochs, --batch-size, --kmer-embed-dim, --resume, etc."
    echo ""
    echo "Example:"
    echo "  sbatch kinsim_cgan_train.slurm master_cgan_data.pkl ./checkpoints/ --epochs 100"
    echo ""
    echo "  # Resume training:"
    echo "  sbatch kinsim_cgan_train.slurm master_cgan_data.pkl ./checkpoints/ \\"
    echo "      --resume ./checkpoints/checkpoint_epoch50.pt"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# Check for GPU
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: No GPU detected. This job requires --gres=gpu:1"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Training ----
MERGED_PKL="$1"
CKPT_DIR="$2"
shift 2  # Remove first two args, keep the rest for kinsim

if [ ! -f "$MERGED_PKL" ]; then
    echo "ERROR: Training data file '$MERGED_PKL' not found."
    exit 1
fi

mkdir -p "$CKPT_DIR"
mkdir -p logs

echo "=== KinSim cGAN Training ==="
echo "Training data: $MERGED_PKL"
echo "Checkpoint dir: $CKPT_DIR"
echo "Additional flags: $@"
echo ""

# Run training with all additional flags passed through
kinsim cgan train "$MERGED_PKL" "$CKPT_DIR" "$@"

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to: $CKPT_DIR"
echo "Monitor with: tensorboard --logdir $CKPT_DIR/runs"
