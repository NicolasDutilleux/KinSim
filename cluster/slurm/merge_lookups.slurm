#!/bin/bash
#SBATCH --job-name=BinKmer_Merge
#SBATCH --mem=48G
#SBATCH --partition=pshort_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/merge_%j.log

# 1. Environment
source ~/.bashrc
conda activate kinsim_env

# 2. Récupération des arguments
# On vérifie si les arguments sont présents pour éviter les crashs
INPUT_DIR=${1:-"/default/path/if/empty"}
OUTPUT_FILE=${2:-"default_output.pkl"}

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch merge_lookups.slurm <input_dir> <output_file>"
    exit 1
fi

# 3. Run
echo "Starting merge from $INPUT_DIR to $OUTPUT_FILE"

python /data/users/ndutilleux/KinSim/cluster/scripts/merge_lookups.py \
    --input_dir "$INPUT_DIR" \
    --output_file "$OUTPUT_FILE"
