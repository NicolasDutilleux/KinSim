#!/bin/bash
# Submit validate chain per strain: raw HiFi → strip → align → kinsim generate
# → merge → jasmine ‖ bystrandify+align+ipdSummary+motifmaker → merge_motifs.
#
# Usage: bash slurm_kinsim/validate.sh <manifest.csv>
# manifest columns: sample_id,lineage  (lineage in {Strepto,Vega})
# Env: CKPT_DIR (default $PREFIX/checkpoints/v12_run3), N_SHARDS (default 10)
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
PREP_SLURM="$REPO/slurm_kinsim/prep"
CALLERS="$REPO/slurm_kinsim/callers"

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

  LINEAGE_LC=$(echo "$LINEAGE" | tr '[:upper:]' '[:lower:]')
  RAW_MANIFEST="$LINEAGE_DIR/manifest_${LINEAGE_LC}.csv"
  [ -f "$RAW_MANIFEST" ] || { echo "ERROR: $SAMPLE: lineage manifest missing: $RAW_MANIFEST" >&2; exit 1; }
  RAW_BAM=$(awk -F, -v s="$SAMPLE" '$1==s {print $2; exit}' "$RAW_MANIFEST")
  [ -n "$RAW_BAM" ] || { echo "ERROR: $SAMPLE not found in $RAW_MANIFEST" >&2; exit 1; }
  [ -f "$RAW_BAM" ] || { echo "ERROR: $SAMPLE: raw BAM missing: $RAW_BAM" >&2; exit 1; }

  VAL_DIR="$PREFIX/validate_${SAMPLE}_${VAL_LABEL}"
  # Rename vs old chain (used to be ${SAMPLE}_stripped.bam) so old artifacts can't be reused.
  STRIPPED_BAM="$VAL_DIR/${SAMPLE}_raw_stripped.bam"
  STRIPPED_ALIGNED_BAM="$VAL_DIR/${SAMPLE}_raw_stripped_aligned.bam"
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
  mkdir -p "$VAL_DIR" "$SHARD_DIR"

  echo "── $SAMPLE ($LINEAGE) — raw: $RAW_BAM"

  PREP_DEP=""
  if [ ! -s "$STRIPPED_BAM" ] || [ ! -f "${STRIPPED_BAM}.pbi" ]; then
    J_PREP=$(sbatch --parsable --job-name="val_prep_$SAMPLE" \
      "$VAL_SLURM/prep.slurm" "$RAW_BAM" "$STRIPPED_BAM")
    PREP_DEP="--dependency=afterok:${J_PREP}"
    echo "  prep        $J_PREP"
  fi

  ALIGN_DEP=""
  if [ ! -s "$STRIPPED_ALIGNED_BAM" ] || [ ! -f "${STRIPPED_ALIGNED_BAM}.pbi" ]; then
    J_ALIGN_RAW=$(sbatch --parsable $PREP_DEP --job-name="val_align_raw_$SAMPLE" \
      "$PREP_SLURM/align_pbmm2.slurm" "$STRIPPED_BAM" "$REF" "$STRIPPED_ALIGNED_BAM")
    ALIGN_DEP="--dependency=afterok:${J_ALIGN_RAW}"
    echo "  align_raw   $J_ALIGN_RAW"
  fi

  REGIONS_DEP=""
  if [ ! -f "${STRIPPED_ALIGNED_BAM}.bai" ] || [ ! -s "$REGIONS_FILE" ]; then
    J_REGIONS=$(sbatch --parsable $ALIGN_DEP --job-name="val_regions_$SAMPLE" \
      --partition=pshort_el8 --account=p774 \
      --cpus-per-task=2 --mem=4G --time=00:15:00 \
      --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log \
      --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
              [ -f ${STRIPPED_ALIGNED_BAM}.bai ] || samtools index -@ 2 ${STRIPPED_ALIGNED_BAM}; \
              python3 $VAL_SLURM/write_regions.py ${STRIPPED_ALIGNED_BAM} ${REGIONS_FILE} ${N_SHARDS}")
    REGIONS_DEP="--dependency=afterok:${J_REGIONS}"
    echo "  regions     $J_REGIONS"
  fi

  J_GEN=$(sbatch --parsable $REGIONS_DEP --array=0-$((N_SHARDS - 1)) \
    --job-name="val_gen_$SAMPLE" \
    "$VAL_SLURM/generate.slurm" \
    "$STRIPPED_ALIGNED_BAM" "$REF" "$CKPT_DIR" "$MOTIFS" "$REGIONS_FILE" "$SHARD_DIR")

  J_MERGE=$(sbatch --parsable --dependency=afterok:$J_GEN \
    --job-name="val_merge_$SAMPLE" \
    "$VAL_SLURM/merge.slurm" "$SHARD_DIR" "$SIM_BAM")

  J_JM=$(sbatch --parsable --dependency=afterok:$J_MERGE \
    --job-name="val_jm_$SAMPLE" \
    "$CALLERS/jasmine_modkit.slurm" "$SIM_BAM" "$REF" "$SIM_JM_CSV")

  J_BYS=$(sbatch --parsable --dependency=afterok:$J_MERGE \
    --job-name="val_bys_$SAMPLE" \
    "$PREP_SLURM/bystrandify.slurm" "$SIM_BAM" "$SIM_BYS_BAM")
  J_ALIGN_BYS=$(sbatch --parsable --dependency=afterok:$J_BYS \
    --job-name="val_align_bys_$SAMPLE" \
    "$PREP_SLURM/align_pbmm2.slurm" "$SIM_BYS_BAM" "$REF" "$SIM_ALIGNED_BAM")
  J_IPD=$(sbatch --parsable --dependency=afterok:$J_ALIGN_BYS \
    --job-name="val_ipd_$SAMPLE" \
    "$CALLERS/ipdsummary.slurm" "$SIM_ALIGNED_BAM" "$REF" "$SIM_GFF" "$SIM_IPD_CSV")
  J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD \
    --job-name="val_mm_$SAMPLE" \
    "$CALLERS/pbmotifmaker.slurm" "$REF" "$SIM_GFF" "$SIM_MM_CSV")

  J_FINAL=$(sbatch --parsable --dependency=afterok:${J_MM}:${J_JM} \
    --job-name="val_final_$SAMPLE" \
    "$CALLERS/merge_motifs.slurm" "$SIM_MERGED_CSV" 0.7 "$SIM_MM_CSV" "$SIM_JM_CSV")

  echo "  generate    $J_GEN (array 0-$((N_SHARDS - 1)))"
  echo "  merge       $J_MERGE   jasmine $J_JM"
  echo "  bys $J_BYS   align $J_ALIGN_BYS   ipd $J_IPD   mm $J_MM"
  echo "  final       $J_FINAL → $SIM_MERGED_CSV"
  FINAL_JOBS+=("$J_FINAL")
done < "$MANIFEST"

echo
echo "Submitted ${#FINAL_JOBS[@]} validation chains. Final job IDs: ${FINAL_JOBS[*]}"
