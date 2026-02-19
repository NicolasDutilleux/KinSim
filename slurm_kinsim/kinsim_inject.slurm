#!/bin/bash
#SBATCH --job-name=KinSim_Inject
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_inject_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim Dictionary — Inject signals into PBSIM3 reads (array job)
#
# Usage:
#   sbatch --array=1-N kinsim_inject.slurm <config> <output_dir>
#
# Arguments:
#   $1  Config file (pipe-delimited, one line per genome):
#       FASTQ_GZ|MAF_GZ|REF_FNA|DICT_PKL|MOTIF_STRING
#   $2  Output directory for injected BAM files
#
# Example:
#   sbatch --array=1-10 kinsim_inject.slurm config_inject.txt /data/kinsim/injected/
#
# Expected directory structure:
#   project/
#     config_inject.txt         <- pipe-delimited config (one line per genome)
#     dicts/
#       master_dict.pkl         <- merged dictionary
#     pbsim3_output/            <- PBSIM3 simulated reads
#       genome1.fq.gz
#       genome1.maf.gz
#     references/               <- reference genomes
#       genome1.fna
#     injected/                 <- output dir
#       genome1_kinsim.bam
#     logs/                     <- SLURM logs
#
# Config format (config_inject.txt):
#   /path/to/genome1.fq.gz|/path/to/genome1.maf.gz|/path/to/genome1.fna|/path/to/master_dict.pkl|m6A,GATC,2;m4C,CCWGG,1
#   /path/to/genome2.fq.gz|/path/to/genome2.maf.gz|/path/to/genome2.fna|/path/to/master_dict.pkl|m6A,RGATCY,4
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch --array=1-N kinsim_inject.slurm <config_inject.txt> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  config_inject.txt    Pipe-delimited, one line per genome:"
    echo "                       FASTQ_GZ|MAF_GZ|REF_FNA|DICT_PKL|MOTIF_STRING"
    echo "  output_dir           Directory for injected BAM files"
    echo ""
    echo "Expected structure:"
    echo "  project/"
    echo "    config_inject.txt"
    echo "    dicts/master_dict.pkl"
    echo "    pbsim3_output/     <- .fq.gz + .maf.gz"
    echo "    references/        <- .fna files"
    echo "    injected/          <- output dir"
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
PKL_IN=$(echo "$LINE" | cut -d'|' -f4)
MOTIFS=$(echo "$LINE" | cut -d'|' -f5)

BASENAME=$(basename "$FASTQ_IN" .fq.gz)
BAM_OUT="${OUTPUT_DIR}/${BASENAME}_kinsim.bam"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "FASTQ: $FASTQ_IN"
echo "MAF:   $MAF_IN"
echo "Ref:   $REF_IN"
echo "Dict:  $PKL_IN"
echo "Motifs: $MOTIFS"
echo "Output: $BAM_OUT"
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
if [ ! -f "$PKL_IN" ]; then
    echo "ERROR: Dictionary file not found: $PKL_IN"
    exit 1
fi

# ---- Run ----
kinsim dictionary inject "$FASTQ_IN" "$MAF_IN" "$REF_IN" "$PKL_IN" "$MOTIFS" "$BAM_OUT"
