#!/bin/bash
# ============================================================
# launch_v12_validate_parallel.sh — bc2046 holdout validation via
# region-sharded parallel kinsim generate.
#
# Pipeline:
#   0. samtools index on stripped BAM (if missing)
#   1. N parallel array tasks, each --region chr:A-B → shard.bam
#   2. samtools merge all shards → bc2046_simulated.bam
#   3. pbindex on merged BAM
#   4. ipdSummary → motifmaker / jasmine → merge motifs
#
# Usage: bash launch_v12_validate_parallel.sh [N_SHARDS]
# ============================================================
set -euo pipefail
export TMPDIR=/tmp

N_SHARDS="${1:-10}"

# ── Paths ───────────────────────────────────────────────────────────
PREFIX=/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega
HOLDOUT=bc2046
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
LOGDIR=/data/projects/p774_MARSD/NDutilleux/logs
PIPE_DIR="$VEGA/pipeline/$HOLDOUT"
REF="$PIPE_DIR/${HOLDOUT}_assembly.fasta"
GROUND_TRUTH_MOTIFS="$PIPE_DIR/${HOLDOUT}_motifs_merged.csv"
VAL_DIR="$PREFIX/validate_${HOLDOUT}"
CKPT_DIR="$PREFIX/checkpoints"
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

# ── 0. Index stripped BAM + compute genome regions ─────────────────
[ -f "${STRIPPED_BAM}.bai" ] || {
  echo "Indexing $STRIPPED_BAM ..."
  srun --partition=pibu_el8 --mem=4G --time=00:15:00 \
       samtools index -@ 4 "$STRIPPED_BAM"
}

# Build N approximately equal regions across all contigs in the reference.
REGIONS_FILE="$SHARD_DIR/regions.txt"
python3 -c "
import pysam, sys, math
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
  --job-name=v12val_genshard_${HOLDOUT} \
  --output="$LOGDIR/v12val_genshard_${HOLDOUT}_%A_%a.log" \
  --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
          REGION=\$(awk -v t=\$SLURM_ARRAY_TASK_ID '\$1==t {print \$2}' '$REGIONS_FILE' | paste -sd,); \
          echo \"Shard \$SLURM_ARRAY_TASK_ID region(s): \$REGION\"; \
          # If multiple regions per shard, run kinsim per region then concat
          FIRST=1; SHARD_OUT='$SHARD_DIR'/shard_\$(printf %03d \$SLURM_ARRAY_TASK_ID).bam; \
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
  --job-name=v12val_mergeshards_${HOLDOUT} \
  --output="$LOGDIR/v12val_mergeshards_${HOLDOUT}_%J.log" \
  --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
          ls '$SHARD_DIR'/shard_*.bam; \
          samtools merge -@ 4 -f '$SIM_BAM' '$SHARD_DIR'/shard_*.bam; \
          apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.3.sif pbindex '$SIM_BAM' || true; \
          echo Merged: '$SIM_BAM'")

# ── 3. Downstream chain (same as before) ──────────────────────────
J_IPD=$(sbatch --parsable --dependency=afterok:$J_MERGE \
  --job-name=v12val_ipd_${HOLDOUT} --output="$LOGDIR/v12val_ipd_${HOLDOUT}_%J.log" \
  "$CALLERS/ipdsummary.slurm" "$SIM_BAM" "$REF" "$SIM_GFF" "$SIM_IPD_CSV")
J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD \
  --job-name=v12val_mm_${HOLDOUT} --output="$LOGDIR/v12val_mm_${HOLDOUT}_%J.log" \
  "$CALLERS/pbmotifmaker.slurm" "$REF" "$SIM_GFF" "$SIM_MM_CSV")
J_JM=$(sbatch --parsable --dependency=afterok:$J_MERGE \
  --job-name=v12val_jasmine_${HOLDOUT} --output="$LOGDIR/v12val_jasmine_${HOLDOUT}_%J.log" \
  "$CALLERS/jasmine_modkit.slurm" "$SIM_BAM" "$REF" "$SIM_JM_CSV")
J_FINAL=$(sbatch --parsable --dependency=afterok:${J_MM}:${J_JM} \
  --job-name=v12val_merge_${HOLDOUT} --output="$LOGDIR/v12val_merge_${HOLDOUT}_%J.log" \
  "$CALLERS/merge_motifs.slurm" "$SIM_MERGED_CSV" "$MERGE_THRESHOLD" "$SIM_MM_CSV" "$SIM_JM_CSV")

echo "── Parallel validation chain submitted ──"
printf '  %-25s : %s\n' "1. generate (array×$N_REGIONS)" "$J_GEN"
printf '  %-25s : %s\n' "2. merge shards"        "$J_MERGE (after $J_GEN)"
printf '  %-25s : %s\n' "3. ipdSummary"          "$J_IPD   (after $J_MERGE)"
printf '  %-25s : %s\n' "4. motifmaker"          "$J_MM    (after $J_IPD)"
printf '  %-25s : %s\n' "4b. jasmine"            "$J_JM    (after $J_MERGE)"
printf '  %-25s : %s\n' "5. final merge"         "$J_FINAL (after $J_MM + $J_JM)"
echo ""
echo "Output: $SIM_MERGED_CSV"
