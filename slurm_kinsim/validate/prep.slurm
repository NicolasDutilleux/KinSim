#!/bin/bash
#SBATCH --job-name=val_prep
#SBATCH --partition=pibu_el8
#SBATCH --account=p774
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#
# Strip fi/fp/ri/rp from a raw HiFi BAM and pbindex the result.
# Usage: sbatch validate/prep.slurm <RAW_BAM> <STRIPPED_BAM>
set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

RAW_BAM=${1:?raw HiFi BAM required}
STRIPPED_BAM=${2:?stripped BAM path required}
REPO=${REPO:-/data/users/ndutilleux/KinSim}
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

[ -f "$RAW_BAM" ] || { echo "ERROR: raw BAM missing: $RAW_BAM"; exit 1; }
mkdir -p "$(dirname "$STRIPPED_BAM")"

if [ -s "$STRIPPED_BAM" ] && [ "${FORCE_STRIP:-0}" != "1" ]; then
  echo "SKIP strip — $STRIPPED_BAM exists. Set FORCE_STRIP=1 to re-run."
else
  T=$(date +%s)
  python "$REPO/scripts/strip_kinetics.py" "$RAW_BAM" "$STRIPPED_BAM"
  echo "strip elapsed: $(( $(date +%s) - T ))s"
fi
[ -s "$STRIPPED_BAM" ] || { echo "ERROR: strip produced empty output"; exit 1; }

if [ ! -f "${STRIPPED_BAM}.pbi" ] || [ ! "${STRIPPED_BAM}.pbi" -nt "$STRIPPED_BAM" ]; then
  apptainer exec --bind /data "$SMRT_SIF" pbindex "$STRIPPED_BAM"
fi

ls -l "$STRIPPED_BAM"{,.pbi} 2>/dev/null | awk '{print "  "$NF" "$5}'
