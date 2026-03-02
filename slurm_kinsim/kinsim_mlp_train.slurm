#!/bin/bash
#SBATCH --job-name=KinSim_MLP_Train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_mlp_train_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim MLP — Train MLP predictor on merged kinetic samples
#
# Trains the MLPPredictor with Gaussian NLL loss on the same data format
# produced by 'kinsim cgan extract' and 'kinsim cgan merge'.
# Requires a single GPU.  Outputs checkpoints, model_config.json, and logs.
#
# Usage:
#   sbatch kinsim_mlp_train.slurm <merged_pkl> <checkpoint_dir> [extra_flags]
#
# Arguments:
#   $1  Merged training data .pkl (from: kinsim cgan merge)
#   $2  Output directory for checkpoints and logs
#   $@  Additional flags (e.g., --epochs, --loss, --kmer-embed-dim, --resume)
#
# Examples:
#   sbatch kinsim_mlp_train.slurm master_cgan_data.pkl ./checkpoints_mlp/
#   sbatch kinsim_mlp_train.slurm master_cgan_data.pkl ./checkpoints_mlp/ \
#       --epochs 100 --loss gnll --batch-size 4096
#
#   # Resume from checkpoint:
#   sbatch kinsim_mlp_train.slurm master_cgan_data.pkl ./checkpoints_mlp/ \
#       --resume ./checkpoints_mlp/checkpoint_epoch25.pt --epochs 50
#
# Output structure:
#   checkpoint_dir/
#     model_config.json           <- Architecture hyperparameters (kmer_embed_dim, hidden_dim)
#     checkpoint_epoch10.pt       <- Model weights + optimizer state
#     checkpoint_epoch20.pt
#     ...
#     runs/                       <- TensorBoard logs (if tensorboard installed)
#     training_log.csv            <- CSV fallback (always written)
#
# Monitor training:
#   tensorboard --logdir checkpoint_dir/runs
#   tail -f checkpoint_dir/training_log.csv
#
# Memory estimate:
#   ~1 GB for default kmer_embed_dim=64
#   ~0.5 GB with --kmer-embed-dim 32
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage:"
    echo "  sbatch kinsim_mlp_train.slurm <merged_pkl> <checkpoint_dir> [extra_flags]"
    echo ""
    echo "Arguments:"
    echo "  merged_pkl         Merged training data .pkl (from: kinsim cgan merge)"
    echo "  checkpoint_dir     Output directory for checkpoints and logs"
    echo "  Additional flags:  --epochs, --batch-size, --lr, --loss, --kmer-embed-dim,"
    echo "                     --hidden-dim, --val-fraction, --checkpoint-every, --resume"
    echo ""
    echo "Examples:"
    echo "  sbatch kinsim_mlp_train.slurm master_cgan_data.pkl ./checkpoints_mlp/"
    echo "  sbatch kinsim_mlp_train.slurm master_cgan_data.pkl ./checkpoints_mlp/ \\"
    echo "      --epochs 100 --loss gnll"
    echo ""
    echo "  # Resume training:"
    echo "  sbatch kinsim_mlp_train.slurm master_cgan_data.pkl ./checkpoints_mlp/ \\"
    echo "      --resume ./checkpoints_mlp/checkpoint_epoch25.pt"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- GPU check ----
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: No GPU detected. This job requires --gres=gpu:1"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---- Training ----
MERGED_PKL="$1"
CKPT_DIR="$2"
shift 2  # Remove first two positional args; pass the rest to kinsim

if [ ! -f "$MERGED_PKL" ]; then
    echo "ERROR: Training data file '$MERGED_PKL' not found."
    exit 1
fi

mkdir -p "$CKPT_DIR"
mkdir -p logs

echo "=== KinSim MLP Training ==="
echo "Training data: $MERGED_PKL"
echo "Checkpoint dir: $CKPT_DIR"
echo "Additional flags: $@"
echo ""

# Run training — all remaining args passed through (e.g. --epochs, --resume)
kinsim mlp train "$MERGED_PKL" "$CKPT_DIR" "$@"

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to: $CKPT_DIR"
echo "Monitor with: tensorboard --logdir $CKPT_DIR/runs"
