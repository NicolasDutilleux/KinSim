#!/bin/bash
#SBATCH --job-name=KinSim_MLP_Evaluate
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_evaluate_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim — MLP Post-Training Evaluation
#
# Runs the full calibration report (MAE, Pearson r, sigma calibration)
# and optionally generates per-kmer distribution plots.
#
# Usage:
#   sbatch kinsim_evaluate.slurm <checkpoint_dir> <master_data.pkl> [extra_flags]
#
# Arguments:
#   $1  Checkpoint directory (contains model_config.json + checkpoint_epoch*.pt)
#   $2  Merged training data .pkl file
#   Additional flags:  --kmer, --meth, --plot, --batch-size, --device
#
# Examples:
#   # Full calibration report:
#   sbatch kinsim_evaluate.slurm checkpoints_mlp/ master_data.pkl
#
#   # Per-kmer distribution plot:
#   sbatch kinsim_evaluate.slurm checkpoints_mlp/ master_data.pkl \
#       --kmer GGATCCTGCAT --meth m6A --plot gatc_m6A.png
#
# Output:
#   checkpoints_mlp/evaluation_report.txt  (full report)
#   <plot>.png                              (per-kmer plot, if --plot given)
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch kinsim_evaluate.slurm <checkpoint_dir> <master_data.pkl> [flags]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_dir     Directory with model_config.json and checkpoint_epoch*.pt"
    echo "  master_data.pkl    Merged training data .pkl file"
    echo ""
    echo "Examples:"
    echo "  sbatch kinsim_evaluate.slurm checkpoints_mlp/ master_data.pkl"
    echo "  sbatch kinsim_evaluate.slurm checkpoints_mlp/ master_data.pkl \\"
    echo "      --kmer GGATCCTGCAT --meth m6A --plot gatc_m6A.png"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  KinSim MLP Evaluate"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "Conda env:  ${CONDA_DEFAULT_ENV:-not set}"
echo "Python:     $(which python)"
echo "Python ver: $(python --version 2>&1)"
echo "KinSim ver: $(kinsim --version 2>&1 || echo 'NOT FOUND')"
echo ""

# ---- GPU check ----
if nvidia-smi > /dev/null 2>&1; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    DEVICE="cuda"
else
    echo "No GPU detected — using CPU."
    DEVICE="cpu"
fi
echo "========================================================"
echo ""

# ---- Config ----
CKPT_DIR="$1"
PKL="$2"
shift 2

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
    exit 1
fi
if [ ! -f "$CKPT_DIR/model_config.json" ]; then
    echo "ERROR: model_config.json not found in $CKPT_DIR"
    exit 1
fi
if [ ! -f "$PKL" ]; then
    echo "ERROR: Data file not found: $PKL"
    exit 1
fi

echo "=== KinSim MLP Evaluate ==="
echo "Checkpoint dir: $CKPT_DIR"
echo "Training data:  $PKL"
echo "Architecture:   $(python -c "import json; print(json.load(open('$CKPT_DIR/model_config.json')).get('architecture', 'mlp'))" 2>/dev/null || echo 'unknown')"
if [ -n "$*" ]; then
    echo "Extra flags:    $*"
fi
echo ""

# ---- Run ----
kinsim evaluate "$CKPT_DIR" "$PKL" --device "$DEVICE" "$@"
EXIT_CODE=$?

# ---- Summary ----
echo ""
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
else
    echo "=== Evaluation complete ==="
    if [ -f "$CKPT_DIR/evaluation_report.txt" ]; then
        echo ""
        echo "--- Report ---"
        cat "$CKPT_DIR/evaluation_report.txt"
    fi
fi

echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
exit $EXIT_CODE
