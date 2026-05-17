#!/bin/bash
# ============================================================
# launch_validate_strain.sh — end-to-end validation of a trained
# KinSim model on ONE held-out strain. Pipeline:
#
#   0. samtools index on stripped BAM (if missing)
#   1. N parallel array tasks, each --region chr:A-B → shard.bam
#      (kinsim generate, using the trained checkpoint)
#   2. samtools merge all shards → <holdout>_simulated.bam + pbindex
#   3. ipdSummary  → per-base IPD CSV + GFF
#   4. motifmaker  → motifs from ipdSummary GFF
#   4b. jasmine    → m5C motifs from BAM directly
#   5. merge motifs → final detected motifs CSV
#
# The detected motifs CSV can be diff'd against the ground-truth
# motifs CSV to score the model end-to-end.
#
# Usage:
#   bash launch_validate_strain.sh <HOLDOUT> <LINEAGE> [N_SHARDS]
#
# Args:
#   HOLDOUT    Strain id, e.g. bc2046, bc2082, bc2075.
#   LINEAGE    Strepto | Vega — selects the source pipeline path.
#   N_SHARDS   Optional number of parallel kinsim generate shards.
#              Default 10 (region-sharded across the genome).
#
# Env overrides:
#   CKPT_DIR             Directory with checkpoint_epoch*.pt + model_config.json
#                        Default: $PREFIX/checkpoints/v12_run3
#   MERGE_THRESHOLD      Min fraction for motif merge (default 0.7)
#   MOTIFMAKER_MIN_SCORE Min score for motifmaker (default 25)
#   VAL_LABEL            Label suffix appended to validate_<HOLDOUT>_<VAL_LABEL>
#                        Default: <CKPT_DIR_basename> (e.g. v12_run3)
#
# Example:
#   # Validate v12_run3 model on E. coli bc2046 (m6A + m5C):
#   CKPT_DIR=/data/.../checkpoints/v12_run3 \
#       bash launch_validate_strain.sh bc2046 Vega 10
#
#   # Validate same model on strepto bc2082 (m5C-rich):
#   CKPT_DIR=/data/.../checkpoints/v12_run3 \
#       bash launch_validate_strain.sh bc2082 Strepto 10
# ============================================================
set -euo pipefail
export TMPDIR=/tmp

HOLDOUT="${1:?holdout strain id required (e.g. bc2046)}"
LINEAGE="${2:?lineage required (Strepto | Vega)}"
N_SHARDS="${3:-10}"

case "$LINEAGE" in
  Strepto|Vega) ;;
  *) echo "ERROR: LINEAGE must be 'Strepto' or 'Vega', got '$LINEAGE'." >&2; exit 1 ;;
esac

# ── Paths ───────────────────────────────────────────────────────────
PREFIX=/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega
LINEAGE_DIR="/data/projects/p774_MARSD/NDutilleux/training/${LINEAGE}"
LOGDIR=/data/projects/p774_MARSD/NDutilleux/logs
PIPE_DIR="$LINEAGE_DIR/pipeline/$HOLDOUT"
REF="$PIPE_DIR/${HOLDOUT}_assembly.fasta"
[ -f "$REF" ] || REF="$LINEAGE_DIR/$HOLDOUT/final_assembly.fasta"   # Strepto layout fallback
GROUND_TRUTH_MOTIFS="$PIPE_DIR/${HOLDOUT}_motifs_merged.csv"

# Default checkpoint: latest run if not specified
CKPT_DIR="${CKPT_DIR:-$PREFIX/checkpoints/v12_run3}"
VAL_LABEL="${VAL_LABEL:-$(basename "$CKPT_DIR")}"
VAL_DIR="$PREFIX/validate_${HOLDOUT}_${VAL_LABEL}"

REPO=/data/users/ndutilleux/KinSim
CALLERS="$REPO/slurm_kinsim/callers"
MERGE_THRESHOLD="${MERGE_THRESHOLD:-0.7}"
export MOTIFMAKER_MIN_SCORE="${MOTIFMAKER_MIN_SCORE:-25}"

STRIPPED_BAM="$VAL_DIR/${HOLDOUT}_stripped.bam"
SHARD_DIR="$VAL_DIR/shards"
SIM_BAM="$VAL_DIR/${HOLDOUT}_simulated.bam"
SIM_GFF="$VAL_DIR/${HOLDOUT}_simulated.gff"
SIM_IPD_CSV="$VAL_DIR/${HOLDOUT}_simulated_ipdSummary.csv"
SIM_MM_CSV="$VAL_DIR/${HOLDOUT}_simulated_motifs_ipdsummary.csv"
SIM_JM_CSV="$VAL_DIR/${HOLDOUT}_simulated_motifs_jasmine.csv"
SIM_MERGED_CSV="$VAL_DIR/${HOLDOUT}_simulated_motifs_merged.csv"

mkdir -p "$VAL_DIR" "$SHARD_DIR" "$LOGDIR"

# ── Sanity checks ──────────────────────────────────────────────────
for f in "$REF" "$GROUND_TRUTH_MOTIFS"; do
  [ -f "$f" ] || { echo "ERROR: required file missing: $f" >&2; exit 1; }
done
[ -d "$CKPT_DIR" ] || { echo "ERROR: CKPT_DIR missing: $CKPT_DIR" >&2; exit 1; }
ls "$CKPT_DIR"/checkpoint_epoch*.pt 2>/dev/null >/dev/null || {
  echo "ERROR: no checkpoint_epoch*.pt in $CKPT_DIR — run training to completion or convert last.ckpt." >&2
  exit 1
}

# The stripped BAM is the input KinSim generate consumes. If it's not
# yet present, the user must produce it from the real aligned BAM via
# `kinsim strip-kinetics` (or scripts/strip_kinetics.py).
if [ ! -f "$STRIPPED_BAM" ]; then
  REAL_BAM="$PIPE_DIR/${HOLDOUT}_aligned.bam"
  if [ -f "$REAL_BAM" ]; then
    echo "INFO: stripped BAM missing — auto-stripping from $REAL_BAM"
    srun --partition=pibu_el8 --mem=8G --time=00:30:00 --cpus-per-task=4 \
         python "$REPO/scripts/strip_kinetics.py" "$REAL_BAM" "$STRIPPED_BAM"
  else
    echo "ERROR: stripped BAM missing AND no aligned BAM found at $REAL_BAM" >&2
    echo "Provide $STRIPPED_BAM manually, or fix the aligned BAM path." >&2
    exit 1
  fi
fi

echo "── Validation config ───────────────────────────────────────────"
echo "  HOLDOUT            : $HOLDOUT"
echo "  LINEAGE            : $LINEAGE"
echo "  REF                : $REF"
echo "  GROUND_TRUTH_MOTIFS: $GROUND_TRUTH_MOTIFS"
echo "  CKPT_DIR           : $CKPT_DIR"
echo "  VAL_DIR            : $VAL_DIR"
echo "  N_SHARDS           : $N_SHARDS"
echo "────────────────────────────────────────────────────────────────"

# ── 0. Index stripped BAM ──────────────────────────────────────────
[ -f "${STRIPPED_BAM}.bai" ] || {
  echo "Indexing $STRIPPED_BAM ..."
  srun --partition=pibu_el8 --mem=4G --time=00:15:00 \
       samtools index -@ 4 "$STRIPPED_BAM"
}

# Build N approximately equal regions across all contigs in the reference.
REGIONS_FILE="$SHARD_DIR/regions.txt"
python3 -c "
import pysam, math
N = $N_SHARDS
bam = pysam.AlignmentFile('$STRIPPED_BAM', 'rb')
contigs = list(zip(bam.references, bam.lengths))
total = sum(L for _, L in contigs)
per_shard = math.ceil(total / N)
out = open('$REGIONS_FILE', 'w')
shard = 0
cursor = 0
for name, L in contigs:
    pos = 0
    while pos < L:
        remaining = per_shard - cursor
        take = min(remaining, L - pos)
        end = pos + take
        out.write(f'{shard}\t{name}:{pos+1}-{end}\n')
        pos += take
        cursor += take
        if cursor >= per_shard:
            shard += 1
            cursor = 0
out.close()
print(f'Wrote {shard+1} regions to $REGIONS_FILE')
"

N_REGIONS=$(awk '{print $1}' "$REGIONS_FILE" | sort -u | wc -l)
ARRAY_MAX=$((N_REGIONS - 1))
echo "Submitting array job 0-${ARRAY_MAX} on $N_REGIONS regions"

# ── 1. Array job — N parallel `kinsim generate` tasks ─────────────
J_GEN=$(sbatch --parsable \
  --array=0-${ARRAY_MAX} \
  --partition=pgpu --account=p774 \
  --gres=gpu:1 --mem=24G --cpus-per-task=4 --time=01:00:00 \
  --job-name=val_gen_${HOLDOUT}_${VAL_LABEL} \
  --output="$LOGDIR/val_gen_${HOLDOUT}_${VAL_LABEL}_%A_%a.log" \
  --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
          REGION=\$(awk -v t=\$SLURM_ARRAY_TASK_ID '\$1==t {print \$2}' '$REGIONS_FILE' | paste -sd,); \
          echo \"Shard \$SLURM_ARRAY_TASK_ID region(s): \$REGION\"; \
          SHARD_OUT='$SHARD_DIR'/shard_\$(printf %03d \$SLURM_ARRAY_TASK_ID).bam; \
          TMP_OUTS=(); \
          CKPT=\$(ls -t '$CKPT_DIR'/checkpoint_epoch*.pt 2>/dev/null | head -1); \
          for R in \$(echo \$REGION | tr ',' ' '); do \
            OUT='$SHARD_DIR'/shard_\$(printf %03d \$SLURM_ARRAY_TASK_ID)_\$(echo \$R | tr ':-' '_').bam; \
            kinsim generate '$STRIPPED_BAM' '$REF' \"\$CKPT\" '$GROUND_TRUTH_MOTIFS' \"\$OUT\" --region \"\$R\"; \
            TMP_OUTS+=(\"\$OUT\"); \
          done; \
          if [ \${#TMP_OUTS[@]} -eq 1 ]; then mv \"\${TMP_OUTS[0]}\" \"\$SHARD_OUT\"; \
          else samtools merge -@ 4 -f \"\$SHARD_OUT\" \"\${TMP_OUTS[@]}\" && rm \"\${TMP_OUTS[@]}\"; fi; \
          echo Done: \$SHARD_OUT")

# ── 2. Merge all shards into one BAM ───────────────────────────────
J_MERGE=$(sbatch --parsable \
  --dependency=afterok:$J_GEN \
  --partition=pibu_el8 --account=p774 \
  --mem=16G --cpus-per-task=4 --time=00:30:00 \
  --job-name=val_merge_${HOLDOUT}_${VAL_LABEL} \
  --output="$LOGDIR/val_merge_${HOLDOUT}_${VAL_LABEL}_%J.log" \
  --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
          ls '$SHARD_DIR'/shard_*.bam; \
          samtools merge -@ 4 -f '$SIM_BAM' '$SHARD_DIR'/shard_*.bam; \
          apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.3.sif pbindex '$SIM_BAM' || true; \
          echo Merged: '$SIM_BAM'")

# ── 3. Downstream chain ────────────────────────────────────────────
J_IPD=$(sbatch --parsable --dependency=afterok:$J_MERGE \
  --job-name=val_ipd_${HOLDOUT}_${VAL_LABEL} \
  --output="$LOGDIR/val_ipd_${HOLDOUT}_${VAL_LABEL}_%J.log" \
  "$CALLERS/ipdsummary.slurm" "$SIM_BAM" "$REF" "$SIM_GFF" "$SIM_IPD_CSV")
J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD \
  --job-name=val_mm_${HOLDOUT}_${VAL_LABEL} \
  --output="$LOGDIR/val_mm_${HOLDOUT}_${VAL_LABEL}_%J.log" \
  "$CALLERS/pbmotifmaker.slurm" "$REF" "$SIM_GFF" "$SIM_MM_CSV")
J_JM=$(sbatch --parsable --dependency=afterok:$J_MERGE \
  --job-name=val_jm_${HOLDOUT}_${VAL_LABEL} \
  --output="$LOGDIR/val_jm_${HOLDOUT}_${VAL_LABEL}_%J.log" \
  "$CALLERS/jasmine_modkit.slurm" "$SIM_BAM" "$REF" "$SIM_JM_CSV")
J_FINAL=$(sbatch --parsable --dependency=afterok:${J_MM}:${J_JM} \
  --job-name=val_finalmerge_${HOLDOUT}_${VAL_LABEL} \
  --output="$LOGDIR/val_finalmerge_${HOLDOUT}_${VAL_LABEL}_%J.log" \
  "$CALLERS/merge_motifs.slurm" "$SIM_MERGED_CSV" "$MERGE_THRESHOLD" "$SIM_MM_CSV" "$SIM_JM_CSV")

echo "── Validation chain submitted ──"
printf '  %-25s : %s\n' "1. generate (array×$N_REGIONS)" "$J_GEN"
printf '  %-25s : %s\n' "2. merge shards"        "$J_MERGE (after $J_GEN)"
printf '  %-25s : %s\n' "3. ipdSummary"          "$J_IPD   (after $J_MERGE)"
printf '  %-25s : %s\n' "4. motifmaker"          "$J_MM    (after $J_IPD)"
printf '  %-25s : %s\n' "4b. jasmine"            "$J_JM    (after $J_MERGE)"
printf '  %-25s : %s\n' "5. final merge"         "$J_FINAL (after $J_MM + $J_JM)"
echo ""
echo "Output: $SIM_MERGED_CSV"
echo "Compare against: $GROUND_TRUTH_MOTIFS"
