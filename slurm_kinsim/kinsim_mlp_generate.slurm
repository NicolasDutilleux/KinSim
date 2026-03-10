#!/bin/bash
#SBATCH --job-name=KinSim_MLP_Generate
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_mlp_generate_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim MLP — Generate signals for PBSIM3 reads (array job)
#
# Uses a trained MLPPredictor to inject kinetic signals into synthetic reads.
# Each array task processes one species.  Requires GPU for efficient batched
# generation (falls back to CPU automatically if unavailable).
#
# Usage:
#   sbatch --array=1-N kinsim_mlp_generate.slurm <pbsim3_dir> <checkpoint.pt> <motifs> <output_dir>
#
# Arguments:
#   $1  PBSIM3 output directory — two supported layouts (auto-detected):
#
#       Species subdirectories (recommended):
#         pbsim3_dir/
#           Ecoli/          <- species name = subdir name
#             reads.fq.gz
#             reads.maf.gz
#             Ecoli.fna
#           Salmonella/
#             ...
#
#       Flat layout (files matched by basename):
#         pbsim3_dir/
#           Ecoli.fq.gz   Ecoli.maf.gz   Ecoli.fna
#           Salmonella.fq.gz ...
#
#   $2  Trained MLP checkpoint (.pt)
#       model_config.json must be in the same directory as the .pt file.
#   $3  Motifs: KinSim string ("m6A,GATC,2;m4C,CCWGG,1") applied to all species,
#       or path to per-species file (one "species_name|motif_string" line per species)
#   $4  Output directory for generated BAM files
#
# Count N:
#   Species subdirs:  ls -d <pbsim3_dir>/*/  | wc -l
#   Flat layout:      ls <pbsim3_dir>/*.fq.gz | wc -l
#
# Examples:
#   sbatch --array=1-57 kinsim_mlp_generate.slurm pbsim3_output/ checkpoint.pt \
#       "m6A,GATC,2;m4C,CCWGG,1" generated_mlp/
#
#   # Per-species motifs file:
#   sbatch --array=1-57 kinsim_mlp_generate.slurm pbsim3_output/ checkpoint.pt \
#       motifs_map.txt generated_mlp/
#
#   # Deterministic (mean prediction, no sampling — useful for ablations):
#   sbatch --array=1-57 kinsim_mlp_generate.slurm pbsim3_output/ checkpoint.pt \
#       "m6A,GATC,1" generated_mlp_det/
#   # then add --deterministic to the kinsim command below
#
# Output: <output_dir>/<species>_mlp.bam  (unaligned, fi + fp tags)
# Note: checkpoint.pt must be in the same directory as model_config.json
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: sbatch --array=1-N kinsim_mlp_generate.slurm <pbsim3_dir> <checkpoint.pt> <motifs> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  pbsim3_dir      Species subdirectories OR flat .fq.gz files"
    echo "  checkpoint.pt   Trained MLP checkpoint (model_config.json must be alongside)"
    echo "  motifs          KinSim string OR per-species 'species|motif_string' file"
    echo "  output_dir      Directory for generated BAM files"
    echo ""
    echo "Count N (subdirs):  ls -d <pbsim3_dir>/*/ | wc -l"
    echo "Count N (flat):     ls <pbsim3_dir>/*.fq.gz | wc -l"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env
# Note: fuzznuc (EMBOSS) is used by default for reference methylation scanning.
# Ensure EMBOSS is installed in kinsim_env: conda install -c bioconda emboss
# KinSim falls back to Python regex automatically if fuzznuc is unavailable.
# Add --no-fuzznuc to the kinsim command below to force regex mode.

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  KinSim MLP Generate"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID:-none}"
echo "Partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory:     ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Conda env:  ${CONDA_DEFAULT_ENV:-not set}"
echo "Python:     $(which python)"
echo "Python ver: $(python --version 2>&1)"
echo "KinSim ver: $(kinsim --version 2>&1 || echo 'NOT FOUND')"
echo "fuzznuc:    $(which fuzznuc 2>/dev/null || echo 'NOT FOUND (will use regex fallback)')"
echo ""

# ---- GPU check ----
if ! nvidia-smi > /dev/null 2>&1; then
    echo "WARNING: No GPU detected. Falling back to CPU (slower)."
    DEVICE="cpu"
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
    echo "CUDA visible:  ${CUDA_VISIBLE_DEVICES:-not set}"
    python -c "import torch; print(f'PyTorch:  {torch.__version__}'); print(f'CUDA:     {torch.version.cuda}'); print(f'GPU seen: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')" 2>&1
    DEVICE="cuda"
fi
echo "========================================================"
echo ""

# ---- Config ----
PBSIM3_DIR="$1"
CKPT_IN="$2"
MOTIFS_ARG="$3"
OUTPUT_DIR="$4"

if [ ! -d "$PBSIM3_DIR" ]; then
    echo "ERROR: PBSIM3 directory not found: $PBSIM3_DIR"
    exit 1
fi
if [ ! -f "$CKPT_IN" ]; then
    echo "ERROR: Checkpoint file not found: $CKPT_IN"
    exit 1
fi

CKPT_DIR=$(dirname "$CKPT_IN")
if [ ! -f "${CKPT_DIR}/model_config.json" ]; then
    echo "WARNING: model_config.json not found in $CKPT_DIR"
    echo "         Using default architecture (kmer_embed_dim=64, hidden_dim=128)"
fi

mkdir -p "$OUTPUT_DIR"

# ---- Auto-detect layout and pick this task's species ----
if ls "${PBSIM3_DIR}"/*.fq.gz 1>/dev/null 2>&1; then
    # Flat layout: index into .fq.gz files sorted alphabetically
    FASTQ_IN=$(ls "${PBSIM3_DIR}"/*.fq.gz | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SPECIES=$(basename "$FASTQ_IN" .fq.gz)
    MAF_IN="${PBSIM3_DIR}/${SPECIES}.maf.gz"
    REF_IN="${PBSIM3_DIR}/${SPECIES}.fna"
else
    # Subdir layout: index into species subdirectories sorted alphabetically
    SPECIES_DIR=$(ls -d "${PBSIM3_DIR}"/*/ 2>/dev/null | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
    if [ -z "$SPECIES_DIR" ]; then
        echo "ERROR: No species directory at index $SLURM_ARRAY_TASK_ID in $PBSIM3_DIR"
        exit 1
    fi
    SPECIES=$(basename "$SPECIES_DIR")
    FASTQ_IN=$(ls "${SPECIES_DIR}"*.fq.gz  2>/dev/null | head -1)
    MAF_IN=$(ls   "${SPECIES_DIR}"*.maf.gz 2>/dev/null | head -1)
    REF_IN=$(ls   "${SPECIES_DIR}"*.fna \
                  "${SPECIES_DIR}"*.fa  \
                  "${SPECIES_DIR}"*.fasta 2>/dev/null | head -1)
fi

if [ -z "$FASTQ_IN" ]; then
    echo "ERROR: No .fq.gz found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

BAM_OUT="${OUTPUT_DIR}/${SPECIES}_mlp.bam"

# ---- Verify inputs ----
if [ ! -f "$FASTQ_IN" ]; then echo "ERROR: FASTQ file not found: $FASTQ_IN"; exit 1; fi
if [ ! -f "$MAF_IN" ];   then echo "ERROR: MAF file not found: $MAF_IN";     exit 1; fi
if [ ! -f "$REF_IN" ];   then echo "ERROR: Reference not found: $REF_IN";    exit 1; fi

# ---- Resolve motifs ----
# Three cases:
#   1. Per-species mapping file (lines: "species_name|motif_string") -> look up by species name
#   2. PacBio motifs.csv or REBASE file -> pass path directly to kinsim (auto-detected)
#   3. Inline KinSim motif string -> use as-is
if [ -f "$MOTIFS_ARG" ]; then
    SPECIES_MOTIFS=$(grep "^${SPECIES}|" "$MOTIFS_ARG" 2>/dev/null | cut -d'|' -f2)
    if [ -n "$SPECIES_MOTIFS" ]; then
        MOTIFS="$SPECIES_MOTIFS"
    else
        # CSV or REBASE file: pass the path directly to kinsim (auto-detected)
        MOTIFS="$MOTIFS_ARG"
    fi
else
    MOTIFS="$MOTIFS_ARG"
fi

echo "=== Task $SLURM_ARRAY_TASK_ID: $SPECIES ==="
echo "FASTQ:      $FASTQ_IN"
echo "MAF:        $MAF_IN"
echo "Ref:        $REF_IN"
echo "Checkpoint: $CKPT_IN"
echo "Motifs:     $MOTIFS"
echo "Output:     $BAM_OUT"
echo "Device:     $DEVICE"
echo ""

# ---- Run ----
# Remove --deterministic below for stochastic generation (default, recommended).
# Add --deterministic for ablation studies (identical signal per context).
kinsim mlp generate "$FASTQ_IN" "$MAF_IN" "$REF_IN" "$CKPT_IN" "$MOTIFS" "$BAM_OUT" \
    --device "$DEVICE" \
    --batch-reads 1000
EXIT_CODE=$?

# ---- Summary ----
echo ""
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Generation failed with exit code $EXIT_CODE for $SPECIES"
else
    echo "=== Generation complete: $BAM_OUT ==="
    BAM_SIZE=$(ls -lh "$BAM_OUT" 2>/dev/null | awk '{print $5}')
    echo "BAM size: ${BAM_SIZE:-file not found}"
fi

echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
exit $EXIT_CODE
