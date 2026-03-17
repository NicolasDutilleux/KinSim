#!/bin/bash
#SBATCH --job-name=KinSim_MLP_Train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_train_%j.log
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
#   sbatch kinsim_train.slurm <merged_pkl> <checkpoint_dir> [extra_flags]
#
# Usage — with YAML config (recommended for reproducibility):
#   sbatch kinsim_train.slurm --config config.yaml [extra_flags]
#
# Arguments:
#   $1  Merged training data .pkl  (or --config path.yaml as first arg)
#   $2  Output directory for checkpoints and logs
#   $@  Additional flags (e.g., --epochs, --loss, --kmer-embed-dim, --resume)
#
# Examples:
#   sbatch kinsim_train.slurm master_data.pkl ./checkpoints_mlp/
#   sbatch kinsim_train.slurm master_data.pkl ./checkpoints_mlp/ \
#       --epochs 100 --loss gnll --batch-size 4096
#
#   # With YAML config (all params in one file):
#   sbatch kinsim_train.slurm --config config.yaml
#
#   # Resume from checkpoint:
#   sbatch kinsim_train.slurm master_data.pkl ./checkpoints_mlp/ \
#       --resume ./checkpoints_mlp/checkpoint_epoch25.pt --epochs 50
#
#   # With Optuna HPO (searches lr, kmer_embed_dim, hidden_dim before final run):
#   sbatch kinsim_train.slurm master_data.pkl ./checkpoints_mlp/ \
#       --optuna --n-trials 20 --optuna-epochs 20 --epochs 50
#
# Output structure:
#   checkpoint_dir/
#     model_config.json           <- Architecture hyperparameters (written first)
#     checkpoint_epoch10.pt       <- Legacy format: weights + optimizer state
#     checkpoint_epoch20.pt         (used by kinsim generate / evaluate)
#     ...
#     lightning_ckpts/            <- Lightning checkpoint files (top-3 + last)
#     logs/version_0/metrics.csv  <- Per-epoch CSV log (always written)
#     runs/version_0/             <- TensorBoard logs (if tensorboard installed)
#     optuna/                     <- HPO trial logs (only when --optuna is used)
#     optuna_best_params.json     <- Best Optuna hyperparameters
#
# Monitor training:
#   tensorboard --logdir checkpoint_dir/runs
#   tail -f checkpoint_dir/logs/version_0/metrics.csv
#
# Memory estimate:
#   ~1 GB for default kmer_embed_dim=64
#   ~0.5 GB with --kmer-embed-dim 32
# ============================================================

# ---- Usage check ----
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  sbatch kinsim_train.slurm <merged_pkl> <checkpoint_dir> [extra_flags]"
    echo "  sbatch kinsim_train.slurm --config config.yaml [extra_flags]"
    echo ""
    echo "Arguments:"
    echo "  merged_pkl         Merged training data .pkl (from: kinsim merge)"
    echo "  checkpoint_dir     Output directory for checkpoints and logs"
    echo "  --config FILE      YAML config file (replaces positional args + sets all flags)"
    echo "  Additional flags:  --epochs, --batch-size, --lr, --loss, --kmer-embed-dim,"
    echo "                     --hidden-dim, --meth-proj-dim, --val-fraction,"
    echo "                     --checkpoint-every, --resume, --device,"
    echo "                     --optuna, --n-trials N, --optuna-epochs N"
    echo ""
    echo "Examples:"
    echo "  sbatch kinsim_train.slurm master_data.pkl ./checkpoints_mlp/"
    echo "  sbatch kinsim_train.slurm --config config.yaml"
    echo ""
    echo "  # Resume training:"
    echo "  sbatch kinsim_train.slurm master_data.pkl ./checkpoints_mlp/ \\"
    echo "      --resume ./checkpoints_mlp/checkpoint_epoch25.pt"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  KinSim MLP Training"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory:     ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Conda env:  ${CONDA_DEFAULT_ENV:-not set}"
echo "Python:     $(which python)"
echo "Python ver: $(python --version 2>&1)"
echo "KinSim ver: $(kinsim --version 2>&1 || echo 'NOT FOUND')"
echo ""

# ---- GPU check ----
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: No GPU detected. This job requires --gres=gpu:1"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""
echo "CUDA visible:  ${CUDA_VISIBLE_DEVICES:-not set}"
python -c "import torch; print(f'PyTorch:  {torch.__version__}'); print(f'CUDA:     {torch.version.cuda}'); print(f'cuDNN:    {torch.backends.cudnn.version()}'); print(f'GPU seen: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')" 2>&1
echo "========================================================"
echo ""

# ---- Dispatch ----
# Two calling conventions:
#   1.  kinsim_train.slurm <pkl> <ckpt_dir> [flags]
#   2.  kinsim_train.slurm --config config.yaml [flags]
if [ "$1" = "--config" ]; then
    # YAML config mode — pass everything directly to kinsim
    echo "=== KinSim MLP Training (YAML config) ==="
    echo "Config: $2"
    echo ""
    mkdir -p logs
    kinsim train "$@"
    EXIT_CODE=$?
else
    # Positional mode — pkl + checkpoint_dir as first two args
    MERGED_PKL="$1"
    CKPT_DIR="$2"
    shift 2  # Remove first two positional args; pass the rest to kinsim

    if [ ! -f "$MERGED_PKL" ]; then
        echo "ERROR: Training data file '$MERGED_PKL' not found."
        exit 1
    fi

    PKL_SIZE=$(ls -lh "$MERGED_PKL" | awk '{print $5}')
    echo "Training data:  $MERGED_PKL  ($PKL_SIZE)"

    mkdir -p "$CKPT_DIR"
    mkdir -p logs

    echo "=== KinSim MLP Training ==="
    echo "Checkpoint dir: $CKPT_DIR"
    if [ -n "$*" ]; then
        echo "Extra flags:    $*"
    fi
    echo ""

    # All remaining args forwarded (e.g. --epochs, --loss, --resume)
    kinsim train "$MERGED_PKL" "$CKPT_DIR" "$@"
    EXIT_CODE=$?
fi

# ---- Summary ----
echo ""
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $EXIT_CODE"
else
    echo "=== Training complete ==="
    # List checkpoints produced
    if [ -n "$CKPT_DIR" ]; then
        echo ""
        echo "Checkpoints:"
        ls -lhtr "$CKPT_DIR"/checkpoint_epoch*.pt 2>/dev/null || echo "  (none found)"
        echo ""
        echo "Config:"
        ls -lh "$CKPT_DIR"/model_config.json 2>/dev/null || echo "  model_config.json NOT FOUND"
    fi
fi

echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
exit $EXIT_CODE
