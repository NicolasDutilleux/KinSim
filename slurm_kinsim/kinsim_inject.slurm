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
#   $2  Path to trained dictionary (.pkl)
#   $3  Motifs: KinSim string ("m6A,GATC,2;m4C,CCWGG,1") applied to all species,
#       or path to per-species file (one "species_name|motif_string" line per species)
#   $4  Output directory for injected BAM files
#
# Count N:
#   Species subdirs:  ls -d <pbsim3_dir>/*/  | wc -l
#   Flat layout:      ls <pbsim3_dir>/*.fq.gz | wc -l
#
# Examples:
#   # Species subdirectories
#   sbatch --array=1-57 kinsim_inject.slurm pbsim3_output/ master_dict.pkl \
#       "m6A,GATC,2;m4C,CCWGG,1" injected/
#
#   # Per-species motifs file
#   sbatch --array=1-57 kinsim_inject.slurm pbsim3_output/ master_dict.pkl \
#       motifs_map.txt injected/
# ============================================================

# ---- Usage check ----
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
    echo "Usage: sbatch --array=1-N kinsim_inject.slurm <pbsim3_dir> <dict.pkl> <motifs> <output_dir>"
    echo ""
    echo "Arguments:"
    echo "  pbsim3_dir    Species subdirectories OR flat .fq.gz files"
    echo "  dict.pkl      Trained kinetic dictionary"
    echo "  motifs        KinSim string OR per-species 'species|motif_string' file"
    echo "  output_dir    Directory for injected BAM files"
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

# ---- Auto-detect layout and pick this task's genome ----
if ls "${PBSIM3_DIR}"/*.fq.gz 1>/dev/null 2>&1; then
    # Flat layout: index into .fq.gz files
    FASTQ_IN=$(ls "${PBSIM3_DIR}"/*.fq.gz | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
    SPECIES=$(basename "$FASTQ_IN" .fq.gz)
    MAF_IN="${PBSIM3_DIR}/${SPECIES}.maf.gz"
    REF_IN="${PBSIM3_DIR}/${SPECIES}.fna"
else
    # Subdir layout: index into species subdirectories
    SPECIES_DIR=$(ls -d "${PBSIM3_DIR}"/*/ 2>/dev/null | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
    if [ -z "$SPECIES_DIR" ]; then
        echo "ERROR: No species directory at index $SLURM_ARRAY_TASK_ID in $PBSIM3_DIR"
        exit 1
    fi
    SPECIES=$(basename "$SPECIES_DIR")
    FASTQ_IN=$(ls "${SPECIES_DIR}"*.fq.gz 2>/dev/null | head -1)
    MAF_IN=$(ls   "${SPECIES_DIR}"*.maf.gz 2>/dev/null | head -1)
    REF_IN=$(ls   "${SPECIES_DIR}"*.fna "${SPECIES_DIR}"*.fa "${SPECIES_DIR}"*.fasta 2>/dev/null | head -1)
fi

if [ -z "$FASTQ_IN" ]; then
    echo "ERROR: No .fq.gz found for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

BAM_OUT="${OUTPUT_DIR}/${SPECIES}_kinsim.bam"

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
        # CSV or REBASE file: pass the path to kinsim (auto-detected by load_motif_string)
        MOTIFS="$MOTIFS_ARG"
    fi
else
    MOTIFS="$MOTIFS_ARG"
fi

echo "=== Task $SLURM_ARRAY_TASK_ID: $SPECIES ==="
echo "FASTQ:  $FASTQ_IN"
echo "MAF:    $MAF_IN"
echo "Ref:    $REF_IN"
echo "Dict:   $DICT_PKL"
echo "Motifs: $MOTIFS"
echo "Output: $BAM_OUT"
echo ""

# ---- Run ----
kinsim dictionary inject "$FASTQ_IN" "$MAF_IN" "$REF_IN" "$DICT_PKL" "$MOTIFS" "$BAM_OUT"
