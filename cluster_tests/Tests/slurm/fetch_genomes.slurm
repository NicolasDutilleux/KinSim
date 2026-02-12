#!/bin/bash
#SBATCH --job-name=PBSIM3_Complete_Workflow
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=pshort_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sim_bact_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# 1. Environment
source ~/.bashrc
conda activate kinsim_env

# 2. Read simulation parameters from config
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/../config.yaml"

WORKDIR=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['simulation']['workdir'])")
MODEL=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['simulation']['pbsim3_model'])")
DEPTH=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['simulation']['depth'])")
STRATEGY=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['simulation']['strategy'])")
METHOD=$(python -c "import yaml; c=yaml.safe_load(open('${CONFIG}')); print(c['simulation']['method'])")

# Read genome list as "name<TAB>url" lines
GENOMES=$(python -c "
import yaml
c = yaml.safe_load(open('${CONFIG}'))
for g in c['simulation']['genomes']:
    print(g['name'] + '\t' + g['url'])
")

# 3. Working directory
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 4. Download genomes in parallel
echo "Downloading genomes..."
while IFS=$'\t' read -r NAME URL; do
    wget -q -nc -O "${NAME}.fna.gz" "$URL" &
done <<< "$GENOMES"
wait

echo "Decompressing..."
for gz in *.fna.gz; do
    gunzip -f "$gz" &
done
wait

# 5. Run PBSIM3 simulations in parallel
echo "Launching pbsim3 simulations..."
while IFS=$'\t' read -r NAME URL; do
    pbsim --strategy "$STRATEGY" \
          --method "$METHOD" \
          --errhmm "$MODEL" \
          --depth "$DEPTH" \
          --genome "${NAME}.fna" \
          --prefix "${NAME}_sim" &
done <<< "$GENOMES"
wait

echo "Workflow complete. Files in $WORKDIR"
