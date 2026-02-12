#!/bin/bash
#SBATCH --job-name=KinSim_Train
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=24:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_train_%A_%a.log

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Config ----
# config_strains.txt: 2-line-per-entry format
#   odd lines  = BAM path
#   even lines = motif string (e.g. "m6A,GATC,2;m4C,CCWGG,1")
# Generate with cluster/scripts/generate_config.sh

CONFIG_FILE=${1:-"config_strains.txt"}
OUTPUT_DIR=${2:-"./"}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file '$CONFIG_FILE' not found."
    exit 1
fi

BAM_PATH=$(sed -n "$(( ($SLURM_ARRAY_TASK_ID * 2) - 1 ))p" "$CONFIG_FILE")
MOTIFS=$(sed -n "$(( $SLURM_ARRAY_TASK_ID * 2 ))p" "$CONFIG_FILE")

BASENAME=$(basename "$BAM_PATH" .bam)
OUTPUT_PKL="${OUTPUT_DIR}/${BASENAME}_binary.pkl"

echo "Task $SLURM_ARRAY_TASK_ID: $BAM_PATH"
echo "Motifs: $MOTIFS"
echo "Output: $OUTPUT_PKL"

# ---- Run ----
python -m kinsim.dictionary.train "$BAM_PATH" "$MOTIFS" "$OUTPUT_PKL"
