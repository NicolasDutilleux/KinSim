#!/bin/bash
#SBATCH --job-name=KinSim_cGAN_Generate
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_cgan_generate_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim cGAN — Generate signals for PBSIM3 reads (array job)
#
# Uses a trained Generator to inject kinetic signals into synthetic reads.
# Each array task processes one genome. Requires GPU for efficient batched generation.
#
# Usage:
#   sbatch --array=1-N kinsim_cgan_generate.slurm <config> <output_dir>
#
# Arguments:
#   $1  Config file (pipe-delimited, one line per genome):
#       FASTQ_GZ|MAF_GZ|REF_FNA|CHECKPOINT_PT|MOTIF_STRING
#   $2  Output directory for injected BAM files
#
# Example:
#   sbatch --array=1-10 kinsim_cgan_generate.slurm config_generate.txt /data/kinsim/generated/
#
# Expected directory structure:
#   project/
#     config_generate.txt       <- pipe-delimited config (one line per genome)
#     checkpoints/
#       checkpoint_epoch100.pt  <- trained Generator
#       model_config.json       <- architecture hyperparameters
#     pbsim3_output/            <- PBSIM3 simulated reads
#       genome1.fq.gz
#       genome1.maf.gz
#     references/               <- reference genomes
#       genome1.fna
#     generated/                <- output dir
#       genome1_cgan.bam
#     logs/                     <- SLURM logs
#
# Config format (config_generate.txt):
#   /path/to/genome1.fq.gz|/path/to/genome1.maf.gz|/path/to/genome1.fna|/path/to/checkpoint_epoch100.pt|m6A,GATC,2;m4C,CCWGG,1
#   /path/to/genome2.fq.gz|/path/to/genome2.maf.gz|/path/to/genome2.fna|/path/to/checkpoint_epoch100.pt|m6A,RGATCY,4
#
# Note: The checkpoint .pt file must be in the same directory as model_config.json
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch --array=1-N kinsim_cgan_generate.slurm <config_generate.txt> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  config_generate.txt  Pipe-delimited, one line per genome:"
    echo "                       FASTQ_GZ|MAF_GZ|REF_FNA|CHECKPOINT_PT|MOTIF_STRING"
    echo "  output_dir           Directory for generated BAM files"
    echo ""
    echo "Expected structure:"
    echo "  project/"
    echo "    config_generate.txt"
    echo "    checkpoints/checkpoint_epoch100.pt"
    echo "    checkpoints/model_config.json       <- must be in same dir as .pt"
    echo "    pbsim3_output/                      <- .fq.gz + .maf.gz"
    echo "    references/                         <- .fna files"
    echo "    generated/                          <- output dir"
    echo "    logs/"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env
# Note: fuzznuc (EMBOSS) is used by default for reference methylation scanning.
# Ensure EMBOSS is installed in kinsim_env: conda install -c bioconda emboss
# KinSim falls back to Python regex automatically if fuzznuc is unavailable.
# Add --no-fuzznuc to the kinsim command below to force regex mode.

# Check for GPU
if ! nvidia-smi > /dev/null 2>&1; then
    echo "WARNING: No GPU detected. Falling back to CPU (slower)."
    DEVICE="cpu"
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
    DEVICE="cuda"
fi

# ---- Config ----
CONFIG_FILE="$1"
OUTPUT_DIR="$2"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' not found."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")

FASTQ_IN=$(echo "$LINE" | cut -d'|' -f1)
MAF_IN=$(echo "$LINE" | cut -d'|' -f2)
REF_IN=$(echo "$LINE" | cut -d'|' -f3)
CKPT_IN=$(echo "$LINE" | cut -d'|' -f4)
MOTIFS=$(echo "$LINE" | cut -d'|' -f5)

BASENAME=$(basename "$FASTQ_IN" .fq.gz)
BAM_OUT="${OUTPUT_DIR}/${BASENAME}_cgan.bam"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "FASTQ: $FASTQ_IN"
echo "MAF: $MAF_IN"
echo "Reference: $REF_IN"
echo "Checkpoint: $CKPT_IN"
echo "Motifs: $MOTIFS"
echo "Output: $BAM_OUT"
echo "Device: $DEVICE"
echo ""

# ---- Verify inputs ----
if [ ! -f "$FASTQ_IN" ]; then
    echo "ERROR: FASTQ file not found: $FASTQ_IN"
    exit 1
fi
if [ ! -f "$MAF_IN" ]; then
    echo "ERROR: MAF file not found: $MAF_IN"
    exit 1
fi
if [ ! -f "$REF_IN" ]; then
    echo "ERROR: Reference file not found: $REF_IN"
    exit 1
fi
if [ ! -f "$CKPT_IN" ]; then
    echo "ERROR: Checkpoint file not found: $CKPT_IN"
    exit 1
fi

# Check for model_config.json in checkpoint directory
CKPT_DIR=$(dirname "$CKPT_IN")
if [ ! -f "$CKPT_DIR/model_config.json" ]; then
    echo "WARNING: model_config.json not found in $CKPT_DIR"
    echo "         Using default architecture (noise_dim=32, kmer_embed_dim=64)"
fi

# ---- Run ----
kinsim cgan generate "$FASTQ_IN" "$MAF_IN" "$REF_IN" "$CKPT_IN" "$MOTIFS" "$BAM_OUT" \
    --device "$DEVICE" \
    --batch-reads 1000

echo ""
echo "=== Generation complete ==="
echo "Output: $BAM_OUT"
