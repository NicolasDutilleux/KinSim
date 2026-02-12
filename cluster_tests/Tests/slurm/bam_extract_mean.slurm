#!/bin/bash
#SBATCH --job-name=KinSim_Parallel
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=pshort_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/extract_stats_%j.log
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

BAM_DIR=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['bam_extract']['bam_dir'])")
LIST_FILE="${OUTPUT_DIR}/$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['output']['bam_extract_list'])")"
OUTPUT_CSV="${OUTPUT_DIR}/$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['output']['stats_csv'])")"

# 3. Build BAM file list
ls "${BAM_DIR}"/*.bam > "$LIST_FILE"

# 4. Execute
PYTHON_SCRIPT="${SCRIPT_DIR}/../scripts/bam_extract_mean.py"
echo "Starting parallel extraction for $(wc -l < "$LIST_FILE") files using $SLURM_CPUS_PER_TASK CPUs..."
python "$PYTHON_SCRIPT" "$LIST_FILE" "$OUTPUT_CSV"

echo "Job finished at $(date)"
