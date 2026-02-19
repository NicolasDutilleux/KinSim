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
#   sbatch --array=1-N kinsim_inject.slurm <pbsim3_dir> <dict.pkl> <motifs> <output_dir>
#
# Arguments:
#   $1  Directory with PBSIM3 output files (.fq.gz + .maf.gz + .fna, same basename)
#   $2  Path to trained dictionary (.pkl)
#   $3  Motifs: KinSim string ("m6A,GATC,2;m4C,CCWGG,1") or path to a
#       per-genome motifs file (one "basename|motif_string" line per genome)
#   $4  Output directory for injected BAM files
#
# Count N (number of genomes):
#   ls <pbsim3_dir>/*.fq.gz | wc -l
#
# Example:
#   sbatch --array=1-12 kinsim_inject.slurm pbsim3_output/ master_dict.pkl \
#       "m6A,GATC,2;m4C,CCWGG,1" injected/
#
# Expected directory structure:
#   project/
#     pbsim3_output/        <- PBSIM3 output (one set per genome)
#       genome1.fq.gz
#       genome1.maf.gz
#       genome1.fna
#       genome2.fq.gz
#       ...
#     master_dict.pkl       <- trained dictionary from kinsim_train.slurm
#     injected/             <- output BAMs written here
#       genome1_kinsim.bam
#     logs/                 <- SLURM logs
#
# Per-genome motifs file format (optional, instead of a single motif string):
#   genome1|m6A,GATC,2;m4C,CCWGG,1
#   genome2|m6A,RGATCY,4
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: sbatch --array=1-N kinsim_inject.slurm <pbsim3_dir> <dict.pkl> <motifs> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  pbsim3_dir    Directory with .fq.gz, .maf.gz, .fna files (same basename)"
    echo "  dict.pkl      Trained kinetic dictionary"
    echo "  motifs        KinSim motif string OR path to basename|motif_string file"
    echo "  output_dir    Directory for injected BAM files"
    echo ""
    echo "Count N: ls <pbsim3_dir>/*.fq.gz | wc -l"
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
PBSIM3_DIR="$1"
DICT_PKL="$2"
MOTIFS_ARG="$3"
OUTPUT_DIR="$4"

if [ ! -d "$PBSIM3_DIR" ]; then
    echo "ERROR: PBSIM3 directory not found: $PBSIM3_DIR"
    exit 1
fi
if [ ! -f "$DICT_PKL" ]; then
    echo "ERROR: Dictionary file not found: $DICT_PKL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---- Pick genome for this task ----
FASTQ_IN=$(ls "${PBSIM3_DIR}"/*.fq.gz 2>/dev/null | sed -n "${SLURM_ARRAY_TASK_ID}p")

if [ -z "$FASTQ_IN" ]; then
    echo "ERROR: No .fq.gz file at index $SLURM_ARRAY_TASK_ID in $PBSIM3_DIR"
    exit 1
fi

BASENAME=$(basename "$FASTQ_IN" .fq.gz)
MAF_IN="${PBSIM3_DIR}/${BASENAME}.maf.gz"
REF_IN="${PBSIM3_DIR}/${BASENAME}.fna"
BAM_OUT="${OUTPUT_DIR}/${BASENAME}_kinsim.bam"

# ---- Verify inputs ----
if [ ! -f "$MAF_IN" ]; then
    echo "ERROR: MAF file not found: $MAF_IN"
    exit 1
fi
if [ ! -f "$REF_IN" ]; then
    echo "ERROR: Reference file not found: $REF_IN"
    exit 1
fi

# ---- Resolve motifs ----
# If MOTIFS_ARG is a file, look up this genome's motif string by basename.
# Otherwise, use it as a motif string for all genomes.
if [ -f "$MOTIFS_ARG" ]; then
    MOTIFS=$(grep "^${BASENAME}|" "$MOTIFS_ARG" | cut -d'|' -f2)
    if [ -z "$MOTIFS" ]; then
        echo "ERROR: No motifs found for basename '$BASENAME' in $MOTIFS_ARG"
        exit 1
    fi
else
    MOTIFS="$MOTIFS_ARG"
fi

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "FASTQ:  $FASTQ_IN"
echo "MAF:    $MAF_IN"
echo "Ref:    $REF_IN"
echo "Dict:   $DICT_PKL"
echo "Motifs: $MOTIFS"
echo "Output: $BAM_OUT"
echo ""

# ---- Run ----
kinsim dictionary inject "$FASTQ_IN" "$MAF_IN" "$REF_IN" "$DICT_PKL" "$MOTIFS" "$BAM_OUT"
