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
# Trains the MLPPredictor with Gaussian NLL loss on the merged .pkl produced
# by kinsim_extract.slurm (or kinsim_cgan_extract.slurm for legacy data).
# Requires a single GPU.  Outputs checkpoints, model_config.json, and logs.
#
# Usage — with positional args:
#   sbatch kinsim_mlp_train.slurm <merged_pkl> <checkpoint_dir> [extra_flags]
#
# Usage — with YAML config (recommended for reproducibility):
#   sbatch kinsim_mlp_train.slurm --config config_mlp.yaml [extra_flags]
#
# Arguments:
#   $1  Merged training data .pkl  (or --config path.yaml as first arg)
#   $2  Output directory for checkpoints and logs
#   $@  Additional flags (e.g., --epochs, --loss, --kmer-embed-dim, --resume)
#
# Examples:
#   sbatch kinsim_mlp_train.slurm master_data.pkl ./checkpoints_mlp/
#   sbatch kinsim_mlp_train.slurm master_data.pkl ./checkpoints_mlp/ \
#       --epochs 100 --loss gnll --batch-size 4096
#
#   # With YAML config (all params in one file):
#   sbatch kinsim_mlp_train.slurm --config config_mlp.yaml
#
#   # Resume from checkpoint:
#   sbatch kinsim_mlp_train.slurm master_data.pkl ./checkpoints_mlp/ \
#       --resume ./checkpoints_mlp/checkpoint_epoch25.pt --epochs 50
#
# Output structure:
#   checkpoint_dir/
#     model_config.json           <- Architecture hyperparameters
#     checkpoint_epoch10.pt       <- Model weights + optimizer state
#     checkpoint_epoch20.pt
#     ...
#     runs/                       <- TensorBoard logs (if installed)
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
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  sbatch kinsim_mlp_train.slurm <merged_pkl> <checkpoint_dir> [extra_flags]"
    echo "  sbatch kinsim_mlp_train.slurm --config config_mlp.yaml [extra_flags]"
    echo ""
    echo "Arguments:"
    echo "  merged_pkl         Merged training data .pkl (from: kinsim merge)"
    echo "  checkpoint_dir     Output directory for checkpoints and logs"
    echo "  --config FILE      YAML config file (replaces positional args + sets all flags)"
    echo "  Additional flags:  --epochs, --batch-size, --lr, --loss, --kmer-embed-dim,"
    echo "                     --hidden-dim, --meth-proj-dim, --val-fraction,"
    echo "                     --checkpoint-every, --resume, --device"
    echo ""
    echo "Examples:"
    echo "  sbatch kinsim_mlp_train.slurm master_data.pkl ./checkpoints_mlp/"
    echo "  sbatch kinsim_mlp_train.slurm --config config_mlp.yaml"
    echo ""
    echo "  # Resume training:"
    echo "  sbatch kinsim_mlp_train.slurm master_data.pkl ./checkpoints_mlp/ \\"
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
echo ""

# ---- Dispatch ----
# Two calling conventions:
#   1.  kinsim_mlp_train.slurm <pkl> <ckpt_dir> [flags]
#   2.  kinsim_mlp_train.slurm --config config.yaml [flags]
if [ "$1" = "--config" ]; then
    # YAML config mode — pass everything directly to kinsim
    echo "=== KinSim MLP Training (YAML config) ==="
    echo "Config: $2"
    echo ""
    mkdir -p logs
    kinsim train --model mlp "$@"
else
    # Positional mode — pkl + checkpoint_dir as first two args
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
    echo "Training data:  $MERGED_PKL"
    echo "Checkpoint dir: $CKPT_DIR"
    if [ -n "$*" ]; then
        echo "Extra flags:    $*"
    fi
    echo ""

    # All remaining args forwarded (e.g. --epochs, --loss, --resume)
    kinsim train --model mlp "$MERGED_PKL" "$CKPT_DIR" "$@"
fi

echo ""
echo "=== Training complete ==="
