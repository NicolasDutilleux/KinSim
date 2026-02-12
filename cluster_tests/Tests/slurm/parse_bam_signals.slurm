#!/bin/bash
#SBATCH --job-name=KinSim_DataDump
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --partition=pshort_el8
#SBATCH --time=00:20:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/extract_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# 1. Environment
source ~/.bashrc
conda activate kinsim_env

# 2. Resolve paths from config
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/../config.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/../output"
mkdir -p "$OUTPUT_DIR"

BAM_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['bam_signals']['bam_dir'])")
LIST_FILE="${OUTPUT_DIR}/$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['output']['bam_signals_list'])")"

# 3. Build BAM file list
echo "INFO: Creating BAM file list..."
ls "${BAM_DIR}"/*.bam > "$LIST_FILE"

# 4. Execute (all print output goes to the SLURM log)
PYTHON_SCRIPT="${SCRIPT_DIR}/../scripts/parse_bam_signals.py"
echo "INFO: Starting data extraction..."
python "$PYTHON_SCRIPT" "$LIST_FILE"

echo "INFO: Job finished at $(date)"
