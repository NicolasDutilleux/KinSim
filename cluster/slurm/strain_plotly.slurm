#!/bin/bash
#SBATCH --job-name=strain_stats
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=pshort_el8
#SBATCH --time=30
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/strain_barplot_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# 1. Environment
source ~/.bashrc
conda activate kinsim_env

# 2. Resolve paths from config
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/../config.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/../output"

INPUT_CSV="${OUTPUT_DIR}/$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['output']['stats_csv'])")"
OUTPUT_HTML="${OUTPUT_DIR}/$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['output']['barplot_html'])")"

# 3. Execute
PYTHON_SCRIPT="${SCRIPT_DIR}/../scripts/strain_plotly.py"
python "$PYTHON_SCRIPT" "$INPUT_CSV" "$OUTPUT_HTML"
