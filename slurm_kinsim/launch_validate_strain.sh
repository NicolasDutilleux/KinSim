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
#   DEPENDS_ON           Optional SLURM job ID — the first job of this run
#                        (the generate array) waits for that job to complete
#                        successfully (--dependency=afterok). Use this to
#                        manually serialise multiple strain runs on the same
#                        shared GPU node:
#                          JID=$(... bash launch_validate_strain.sh A | grep -oP 'FINAL_JOB=\K[0-9]+')
#                          DEPENDS_ON=$JID bash launch_validate_strain.sh B
#                        The script prints `FINAL_JOB=<id>` on its last line
#                        for easy capture.
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

REAL_BAM="$PIPE_DIR/${HOLDOUT}_aligned.bam"

echo "── Validation config ───────────────────────────────────────────"
echo "  HOLDOUT            : $HOLDOUT"
echo "  LINEAGE            : $LINEAGE"
echo "  REF                : $REF"
echo "  GROUND_TRUTH_MOTIFS: $GROUND_TRUTH_MOTIFS"
echo "  CKPT_DIR           : $CKPT_DIR"
echo "  VAL_DIR            : $VAL_DIR"
echo "  N_SHARDS           : $N_SHARDS"
echo "────────────────────────────────────────────────────────────────"

REGIONS_FILE="$SHARD_DIR/regions.txt"

# ── 0. PREP — strip BAM (if missing) + index + write regions file ──
# Single sbatch step so the whole chain is non-blocking. Everything that
# the generate array needs ($STRIPPED_BAM, ${STRIPPED_BAM}.bai,
# $REGIONS_FILE) is produced here. The array waits for this job via
# --dependency=afterok and reads its assigned region(s) from $REGIONS_FILE
# at runtime. If both stripped BAM and index already exist AND the
# regions file is already there, no prep job is submitted and the array
# goes straight in (saves ~30 min on re-runs).
PREP_DEP=""
PREP_NEEDED=0
[ -f "$STRIPPED_BAM" ]      || PREP_NEEDED=1
[ -f "${STRIPPED_BAM}.bai" ] || PREP_NEEDED=1
[ -f "$REGIONS_FILE" ]      || PREP_NEEDED=1

if [ "$PREP_NEEDED" = "1" ]; then
  if [ ! -f "$STRIPPED_BAM" ] && [ ! -f "$REAL_BAM" ]; then
    echo "ERROR: stripped BAM missing AND no aligned BAM found at $REAL_BAM" >&2
    echo "Place $STRIPPED_BAM manually, or fix the aligned BAM path." >&2
    exit 1
  fi
  echo "Submitting prep job (strip if needed → index → write regions file) ..."
  J_PREP=$(sbatch --parsable \
    --partition=pibu_el8 --account=p774 \
    --mem=16G --cpus-per-task=4 --time=01:00:00 \
    --job-name=val_prep_${HOLDOUT}_${VAL_LABEL} \
    --output="$LOGDIR/val_prep_${HOLDOUT}_${VAL_LABEL}_%J.log" \
    --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
            if [ ! -f '$STRIPPED_BAM' ]; then \
              echo Stripping $REAL_BAM ...; \
              python '$REPO/scripts/strip_kinetics.py' '$REAL_BAM' '$STRIPPED_BAM'; \
            fi; \
            if [ ! -f '${STRIPPED_BAM}.bai' ]; then \
              echo Indexing $STRIPPED_BAM ...; \
              samtools index -@ 4 '$STRIPPED_BAM'; \
            fi; \
            echo Writing regions file ...; \
            python3 -c \"
import pysam, math
N = $N_SHARDS
bam = pysam.AlignmentFile('$STRIPPED_BAM', 'rb')
contigs = list(zip(bam.references, bam.lengths))
total = sum(L for _, L in contigs)
per_shard = max(1, math.ceil(total / N))
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
\"")
  PREP_DEP="--dependency=afterok:${J_PREP}"
  echo "  prep job: $J_PREP"
else
  echo "Prep already done — skipping (stripped BAM + .bai + regions.txt all present)."
fi

# Optional --dependency on a previous run's terminal job, for manual
# serialisation across strains on a shared GPU node. Combined with prep
# dependency below if both are active.
GEN_DEP="$PREP_DEP"
if [ -n "${DEPENDS_ON:-}" ]; then
  if [ -n "$GEN_DEP" ]; then
    GEN_DEP="--dependency=afterok:${J_PREP}:${DEPENDS_ON}"
  else
    GEN_DEP="--dependency=afterok:${DEPENDS_ON}"
  fi
  echo "Chaining: first generate array waits for job $DEPENDS_ON"
fi

# Array size: N_SHARDS upper bound. If the genome was too small to fill
# all N regions, the prep job wrote fewer lines — the surplus array
# tasks exit cleanly (no region for their task id, see ARRAY_GUARD below).
ARRAY_MAX=$((N_SHARDS - 1))
echo "Submitting array job 0-${ARRAY_MAX} (up to $N_SHARDS regions)"

# ── 1. Array job — N parallel `kinsim generate` tasks ─────────────
J_GEN=$(sbatch --parsable \
  $GEN_DEP \
  --array=0-${ARRAY_MAX} \
  --partition=pgpu --account=p774 \
  --gres=gpu:1 --mem=24G --cpus-per-task=4 --time=01:00:00 \
  --job-name=val_gen_${HOLDOUT}_${VAL_LABEL} \
  --output="$LOGDIR/val_gen_${HOLDOUT}_${VAL_LABEL}_%A_%a.log" \
  --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
          REGION=\$(awk -v t=\$SLURM_ARRAY_TASK_ID '\$1==t {print \$2}' '$REGIONS_FILE' | paste -sd,); \
          if [ -z \"\$REGION\" ]; then \
            echo \"ARRAY_GUARD: no region for task \$SLURM_ARRAY_TASK_ID (genome too small for $N_SHARDS shards) — skipping cleanly\"; \
            exit 0; \
          fi; \
          echo \"Shard \$SLURM_ARRAY_TASK_ID region(s): \$REGION\"; \
          SHARD_OUT='$SHARD_DIR'/shard_\$(printf %03d \$SLURM_ARRAY_TASK_ID).bam; \
          TMP_OUTS=(); \
          CKPT=\$(ls -t '$CKPT_DIR'/checkpoint_epoch*.pt 2>/dev/null | head -1); \
          for R in \$(echo \$REGION | tr ',' ' '); do \
            OUT='$SHARD_DIR'/shard_\$(printf %03d \$SLURM_ARRAY_TASK_ID)_\$(echo \$R | tr ':-/' '___').bam; \
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
if [ -n "${J_PREP:-}" ]; then
  printf '  %-30s : %s\n' "0. prep (strip+index+regions)" "$J_PREP"
fi
printf '  %-30s : %s\n' "1. generate (array, ≤$N_SHARDS tasks)" "$J_GEN"
printf '  %-30s : %s\n' "2. merge shards"        "$J_MERGE (after $J_GEN)"
printf '  %-30s : %s\n' "3. ipdSummary"          "$J_IPD   (after $J_MERGE)"
printf '  %-30s : %s\n' "4. motifmaker"          "$J_MM    (after $J_IPD)"
printf '  %-30s : %s\n' "4b. jasmine"            "$J_JM    (after $J_MERGE)"
printf '  %-30s : %s\n' "5. final merge"         "$J_FINAL (after $J_MM + $J_JM)"
echo ""
echo "Output: $SIM_MERGED_CSV"
echo "Compare against: $GROUND_TRUTH_MOTIFS"

# Machine-parseable line for caller chaining (used by DEPENDS_ON pattern).
# Must be on its own line as the LAST line; downstream consumers grep for it.
echo "FINAL_JOB=${J_FINAL}"
