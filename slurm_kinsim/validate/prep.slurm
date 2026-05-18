#!/bin/bash
#SBATCH --job-name=val_prep
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#
# validate/prep — strip kinetics from aligned BAM, index it, and write
# the per-shard regions file used by validate/generate (array).
#
# Usage:
#   sbatch slurm_kinsim/validate/prep.slurm \
#          <REAL_BAM> <STRIPPED_BAM> <REGIONS_FILE> <N_SHARDS>
set -euo pipefail
source ~/.bashrc; conda activate kinsim_env

REAL_BAM=${1:?aligned BAM required}
STRIPPED_BAM=${2:?stripped BAM path required}
REGIONS_FILE=${3:?regions file path required}
N_SHARDS=${4:?n_shards required}
REPO=${REPO:-/data/users/ndutilleux/KinSim}

mkdir -p "$(dirname "$STRIPPED_BAM")" "$(dirname "$REGIONS_FILE")"

if [ ! -f "$STRIPPED_BAM" ]; then
  python "$REPO/scripts/strip_kinetics.py" "$REAL_BAM" "$STRIPPED_BAM"
fi
if [ ! -f "${STRIPPED_BAM}.bai" ]; then
  samtools index -@ "$SLURM_CPUS_PER_TASK" "$STRIPPED_BAM"
fi

python3 "$REPO/slurm_kinsim/validate/write_regions.py" \
        "$STRIPPED_BAM" "$REGIONS_FILE" "$N_SHARDS"
