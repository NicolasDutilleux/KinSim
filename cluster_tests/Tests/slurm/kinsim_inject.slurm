#!/bin/bash
#SBATCH --job-name=KinSim_Inject
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=pshort_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_inject_%A_%a.log

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Config ----
# config_inject.txt: pipe-delimited, one line per job
#   FASTQ_GZ|MAF_GZ|REF_FNA|DICT_PKL|MOTIF_STRING
# Example:
#   /path/to/reads.fq.gz|/path/to/reads.maf.gz|/path/to/ref.fna|/path/to/dict.pkl|m6A,GATC,2;m4C,CCWGG,1

CONFIG_FILE=${1:-"config_inject.txt"}
OUTPUT_DIR=${2:-"./"}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' not found."
    exit 1
fi

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_FILE")

FASTQ_IN=$(echo "$LINE" | cut -d'|' -f1)
MAF_IN=$(echo "$LINE" | cut -d'|' -f2)
REF_IN=$(echo "$LINE" | cut -d'|' -f3)
PKL_IN=$(echo "$LINE" | cut -d'|' -f4)
MOTIFS=$(echo "$LINE" | cut -d'|' -f5)

BASENAME=$(basename "$FASTQ_IN" .fq.gz)
BAM_OUT="${OUTPUT_DIR}/${BASENAME}_kinsim.bam"

echo "Task $SLURM_ARRAY_TASK_ID: $FASTQ_IN"
echo "Output: $BAM_OUT"

# ---- Run ----
python -m kinsim.dictionary.inject "$FASTQ_IN" "$MAF_IN" "$REF_IN" "$PKL_IN" "$MOTIFS" "$BAM_OUT"
