#!/bin/bash
#SBATCH --job-name=val_generate
#SBATCH --partition=pgpu
#SBATCH --account=p774
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%A_%a.log
#
# validate/generate — one array task per region. Reads its region from
# REGIONS_FILE (column 1 == SLURM_ARRAY_TASK_ID) and writes a shard BAM.
# Array surplus (genome too small for the requested N_SHARDS) is handled
# by emitting nothing for that line in regions.txt — the task exits 0.
#
# Usage (set --array=0-N-1 in the sbatch call):
#   sbatch --array=0-9 slurm_kinsim/validate/generate.slurm \
#          <STRIPPED_BAM> <REF> <CKPT_DIR> <MOTIFS_CSV> \
#          <REGIONS_FILE> <SHARD_DIR>
set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

STRIPPED_BAM=${1:?stripped BAM required}
REF=${2:?reference required}
CKPT_DIR=${3:?ckpt dir required}
MOTIFS=${4:?motifs CSV required}
REGIONS_FILE=${5:?regions file required}
SHARD_DIR=${6:?shard dir required}

mkdir -p "$SHARD_DIR"

# Clean this task's stale intermediates from earlier failed runs (per-region
# .bam files that should have been rm'd after the inner-merge but weren't
# if the prior task crashed). Leaves other tasks' shards untouched.
TASK_PAT=$(printf 'shard_%03d' "$SLURM_ARRAY_TASK_ID")
rm -f "${SHARD_DIR}/${TASK_PAT}.bam" "${SHARD_DIR}/${TASK_PAT}_"*.bam

REGION=$(awk -v t="$SLURM_ARRAY_TASK_ID" '$1==t {print $2}' "$REGIONS_FILE" | paste -sd,)
if [ -z "$REGION" ]; then
  echo "no region for task $SLURM_ARRAY_TASK_ID — exiting cleanly"
  exit 0
fi

CKPT=$(ls -t "$CKPT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "ERROR: no checkpoint_epoch*.pt in $CKPT_DIR" >&2; exit 1; }

SHARD_OUT=$(printf '%s/shard_%03d.bam' "$SHARD_DIR" "$SLURM_ARRAY_TASK_ID")
TMP_OUTS=()
for R in ${REGION//,/ }; do
  SAFE=$(echo "$R" | tr ':/-' '___')
  OUT="${SHARD_DIR}/shard_$(printf %03d "$SLURM_ARRAY_TASK_ID")_${SAFE}.bam"
  kinsim generate "$STRIPPED_BAM" "$REF" "$CKPT" "$MOTIFS" "$OUT" --region "$R"
  TMP_OUTS+=("$OUT")
done

if [ "${#TMP_OUTS[@]}" -eq 1 ]; then
  mv "${TMP_OUTS[0]}" "$SHARD_OUT"
else
  samtools merge -c -@ "$SLURM_CPUS_PER_TASK" -f "$SHARD_OUT" "${TMP_OUTS[@]}"
  rm "${TMP_OUTS[@]}"
fi
echo "Done: $SHARD_OUT"
