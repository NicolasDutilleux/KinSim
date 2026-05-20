#!/bin/bash
# validate.sh — submit a validation chain for every strain in a manifest.
#
# Each row of the manifest (CSV with header `sample_id,lineage`) gets the
# chain:
#   prep → generate (array, fi/fp/ri/rp) → merge → bystrandify (ip/pw)
#                                                       │
#                                                       ├→ align → ipdSummary → motifmaker
#                                                       │
#                                                       └→ jasmine → final motif merge
#
# Same chain as the real-data pipeline: raw HiFi (fi/fp/ri/rp) →
# ccs-kinetics-bystrandify (one record per strand with ip/pw) → pbmm2
# align (preserves ip/pw) → ipdSummary (consumes ip).
#
# jasmine takes the bystrandified BAM (its own pbmm2 align inside the
# script will preserve ip/pw too).
#
# Usage:
#   bash slurm_kinsim/validate.sh <manifest.csv>
#
# Manifest:
#   sample_id,lineage
#   bc2034,Strepto
#   bc2046,Vega
#
# Env (optional):
#   CKPT_DIR    Default: $PREFIX/checkpoints/v12_run3
#   N_SHARDS    Default: 10  (kinsim generate array width per strain)
set -euo pipefail

MANIFEST=${1:?"usage: bash $0 <manifest.csv>"}
[ -f "$MANIFEST" ] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1; }

PREFIX=/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega
CKPT_DIR="${CKPT_DIR:-$PREFIX/checkpoints/v12_run3}"
N_SHARDS="${N_SHARDS:-10}"

[ -d "$CKPT_DIR" ] || { echo "ERROR: CKPT_DIR missing: $CKPT_DIR" >&2; exit 1; }
ls "$CKPT_DIR"/checkpoint_epoch*.pt >/dev/null 2>&1 \
  || { echo "ERROR: no checkpoint_epoch*.pt in $CKPT_DIR" >&2; exit 1; }

VAL_LABEL=$(basename "$CKPT_DIR")
REPO=/data/users/ndutilleux/KinSim
VAL_SLURM="$REPO/slurm_kinsim/validate"
CALLERS="$REPO/slurm_kinsim/callers"

# Walk the manifest. Skip header + comment + empty lines.
declare -a FINAL_JOBS=()
while IFS=, read -r SAMPLE LINEAGE _rest; do
  SAMPLE="${SAMPLE//[$'\t\r\n ']/}"
  LINEAGE="${LINEAGE//[$'\t\r\n ']/}"
  [ -z "$SAMPLE" ] && continue
  [ "${SAMPLE:0:1}" = "#" ] && continue
  [ "$SAMPLE" = "sample_id" ] && continue

  case "$LINEAGE" in
    Strepto|Vega) ;;
    *) echo "ERROR: row '$SAMPLE': lineage must be Strepto|Vega, got '$LINEAGE'" >&2; exit 1;;
  esac

  LINEAGE_DIR="/data/projects/p774_MARSD/NDutilleux/training/${LINEAGE}"
  PIPE_DIR="$LINEAGE_DIR/pipeline/$SAMPLE"
  REF="$PIPE_DIR/${SAMPLE}_assembly.fasta"
  [ -f "$REF" ] || REF="$LINEAGE_DIR/$SAMPLE/final_assembly.fasta"
  MOTIFS="$PIPE_DIR/${SAMPLE}_motifs_merged.csv"
  REAL_BAM="$PIPE_DIR/${SAMPLE}_aligned.bam"

  VAL_DIR="$PREFIX/validate_${SAMPLE}_${VAL_LABEL}"
  STRIPPED_BAM="$VAL_DIR/${SAMPLE}_stripped.bam"
  SHARD_DIR="$VAL_DIR/shards"
  REGIONS_FILE="$SHARD_DIR/regions.txt"
  SIM_BAM="$VAL_DIR/${SAMPLE}_simulated.bam"
  SIM_BYS_BAM="$VAL_DIR/${SAMPLE}_simulated_bystrandified.bam"
  SIM_ALIGNED_BAM="$VAL_DIR/${SAMPLE}_simulated_aligned.bam"
  SIM_GFF="$VAL_DIR/${SAMPLE}_simulated.gff"
  SIM_IPD_CSV="$VAL_DIR/${SAMPLE}_simulated_ipdSummary.csv"
  SIM_MM_CSV="$VAL_DIR/${SAMPLE}_simulated_motifs_ipdsummary.csv"
  SIM_JM_CSV="$VAL_DIR/${SAMPLE}_simulated_motifs_jasmine.csv"
  SIM_MERGED_CSV="$VAL_DIR/${SAMPLE}_simulated_motifs_merged.csv"

  for f in "$REF" "$MOTIFS"; do
    [ -f "$f" ] || { echo "ERROR: $SAMPLE: missing $f" >&2; exit 1; }
  done
  [ -f "$STRIPPED_BAM" ] || [ -f "$REAL_BAM" ] \
    || { echo "ERROR: $SAMPLE: neither $STRIPPED_BAM nor $REAL_BAM present" >&2; exit 1; }
  mkdir -p "$VAL_DIR" "$SHARD_DIR"

  echo "── $SAMPLE ($LINEAGE) ──────────────────────────────────"

  PREP_DEP=""
  if [ ! -f "$STRIPPED_BAM" ] || [ ! -f "${STRIPPED_BAM}.bai" ] || [ ! -f "$REGIONS_FILE" ]; then
    J_PREP=$(sbatch --parsable --job-name="val_prep_$SAMPLE" \
      "$VAL_SLURM/prep.slurm" "$REAL_BAM" "$STRIPPED_BAM" "$REGIONS_FILE" "$N_SHARDS")
    PREP_DEP="--dependency=afterok:${J_PREP}"
    echo "  prep       $J_PREP"
  fi

  J_GEN=$(sbatch --parsable $PREP_DEP --array=0-$((N_SHARDS - 1)) \
    --job-name="val_gen_$SAMPLE" \
    "$VAL_SLURM/generate.slurm" \
    "$STRIPPED_BAM" "$REF" "$CKPT_DIR" "$MOTIFS" "$REGIONS_FILE" "$SHARD_DIR")
  J_MERGE=$(sbatch --parsable --dependency=afterok:$J_GEN \
    --job-name="val_merge_$SAMPLE" \
    "$VAL_SLURM/merge.slurm" "$SHARD_DIR" "$SIM_BAM")
  # bystrandify — converts raw-HiFi (fi/fp/ri/rp on flag=4) to bystrandified
  # (one record per strand with ip/pw), matching the real-data pipeline.
  # pbmm2 then preserves ip/pw, and ipdSummary consumes them directly.
  J_BYS=$(sbatch --parsable --dependency=afterok:$J_MERGE \
    --job-name="val_bys_$SAMPLE" \
    "$REPO/slurm_kinsim/prep/bystrandify.slurm" "$SIM_BAM" "$SIM_BYS_BAM")
  J_ALIGN=$(sbatch --parsable --dependency=afterok:$J_BYS \
    --job-name="val_align_$SAMPLE" \
    "$REPO/slurm_kinsim/prep/align_pbmm2.slurm" "$SIM_BYS_BAM" "$REF" "$SIM_ALIGNED_BAM")
  J_IPD=$(sbatch --parsable --dependency=afterok:$J_ALIGN \
    --job-name="val_ipd_$SAMPLE" \
    "$CALLERS/ipdsummary.slurm" "$SIM_ALIGNED_BAM" "$REF" "$SIM_GFF" "$SIM_IPD_CSV")
  J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD \
    --job-name="val_mm_$SAMPLE" \
    "$CALLERS/pbmotifmaker.slurm" "$REF" "$SIM_GFF" "$SIM_MM_CSV")
  J_JM=$(sbatch --parsable --dependency=afterok:$J_BYS \
    --job-name="val_jm_$SAMPLE" \
    "$CALLERS/jasmine_modkit.slurm" "$SIM_BYS_BAM" "$REF" "$SIM_JM_CSV")
  J_FINAL=$(sbatch --parsable --dependency=afterok:${J_MM}:${J_JM} \
    --job-name="val_final_$SAMPLE" \
    "$CALLERS/merge_motifs.slurm" "$SIM_MERGED_CSV" 0.7 "$SIM_MM_CSV" "$SIM_JM_CSV")

  echo "  generate    $J_GEN (array 0-$((N_SHARDS - 1)))"
  echo "  merge       $J_MERGE"
  echo "  bystrandify $J_BYS"
  echo "  align       $J_ALIGN"
  echo "  ipdSummary  $J_IPD"
  echo "  motifmaker  $J_MM"
  echo "  jasmine     $J_JM"
  echo "  final       $J_FINAL"
  echo "  output:    $SIM_MERGED_CSV"
  FINAL_JOBS+=("$J_FINAL")
done < "$MANIFEST"

echo
echo "Submitted ${#FINAL_JOBS[@]} validation chains. Final job IDs: ${FINAL_JOBS[*]}"
