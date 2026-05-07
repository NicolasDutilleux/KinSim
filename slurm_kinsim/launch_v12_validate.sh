#!/bin/bash
# ============================================================
# launch_v12_validate.sh — end-to-end holdout validation for v12.
#
# After v12 training finishes, this script regenerates kinetics
# on the bc2046 (E. coli) holdout BAM and re-runs the methylation
# caller pipeline on the simulated kinetics. The resulting motif
# file can be compared to the ground-truth motif file to verify
# that KinSim produces detectable methylation signatures.
#
# Pipeline:
#   1. strip kinetics from bc2046_aligned.bam → stripped.bam
#   2. kinsim generate (BAM mode) — inject simulated kinetics
#                       using the trained checkpoint + bc2046 motifs
#                       → simulated.bam
#   3. ipdSummary on simulated.bam → simulated.gff
#   4. pbmotifmaker on simulated.gff → simulated_motifs_ipdsummary.csv
#   5. merge with existing jasmine motifs.csv
#                       → simulated_motifs_merged.csv
#
# Optional dependency on training: pass the train job ID as $1
# (afterok). Without it the script submits with no dependency
# (assumes train is already done).
#
# Usage:
#   bash slurm_kinsim/launch_v12_validate.sh [TRAIN_JID]
# ============================================================

set -euo pipefail
export TMPDIR=/tmp

# ── Config ────────────────────────────────────────────────────────────
PREFIX=/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega
HOLDOUT_VEGA=bc2046
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
LOGDIR=/data/projects/p774_MARSD/NDutilleux/logs

# Holdout inputs (already on disk from v12 prep)
PIPE_DIR="$VEGA/pipeline/$HOLDOUT_VEGA"
ALIGNED_BAM="$PIPE_DIR/${HOLDOUT_VEGA}_aligned.bam"
REF="$PIPE_DIR/${HOLDOUT_VEGA}_assembly.fasta"
JASMINE_CSV="$PIPE_DIR/${HOLDOUT_VEGA}_motifs_jasmine.csv"
GROUND_TRUTH_MOTIFS="$PIPE_DIR/${HOLDOUT_VEGA}_motifs_merged.csv"

# Output directory (a separate validate/ subdir under v12)
VAL_DIR="$PREFIX/validate_${HOLDOUT_VEGA}"
mkdir -p "$VAL_DIR"

# Trained checkpoint (latest .pt in checkpoints/)
CKPT_DIR="$PREFIX/checkpoints"

# Optional dependency on training
TRAIN_DEP=""
if [ -n "${1:-}" ]; then
    TRAIN_DEP="--dependency=afterok:$1"
fi

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
CALLERS="$REPO/slurm_kinsim/callers"
MERGE_THRESHOLD=${MERGE_THRESHOLD:-0.7}
export MOTIFMAKER_MIN_SCORE=${MOTIFMAKER_MIN_SCORE:-25}

echo "================================================================"
echo "  KinSim v12 holdout validation — $HOLDOUT_VEGA"
echo "  Aligned BAM:    $ALIGNED_BAM"
echo "  Reference:      $REF"
echo "  Ground-truth:   $GROUND_TRUTH_MOTIFS"
echo "  Jasmine input:  $JASMINE_CSV"
echo "  Validate dir:   $VAL_DIR"
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Train dep:      ${TRAIN_DEP:-(none — train assumed done)}"
echo "================================================================"
echo ""

# Sanity checks
[ -f "$ALIGNED_BAM" ]  || { echo "ERROR: missing $ALIGNED_BAM"; exit 1; }
[ -f "$REF" ]          || { echo "ERROR: missing $REF"; exit 1; }
[ -s "$JASMINE_CSV" ]  || { echo "ERROR: missing or empty $JASMINE_CSV"; exit 1; }
[ -s "$GROUND_TRUTH_MOTIFS" ] || echo "WARN: ground-truth motifs file missing — comparison will be skipped"

# ── Output paths ───────────────────────────────────────────────────────
STRIPPED_BAM="$VAL_DIR/${HOLDOUT_VEGA}_stripped.bam"
SIM_BAM="$VAL_DIR/${HOLDOUT_VEGA}_simulated.bam"
SIM_GFF="$VAL_DIR/${HOLDOUT_VEGA}_simulated.gff"
SIM_IPD_CSV="$VAL_DIR/${HOLDOUT_VEGA}_simulated_ipdSummary.csv"
SIM_MM_CSV="$VAL_DIR/${HOLDOUT_VEGA}_simulated_motifs_ipdsummary.csv"
SIM_MERGED_CSV="$VAL_DIR/${HOLDOUT_VEGA}_simulated_motifs_merged.csv"

# ── 1. Strip kinetics ─────────────────────────────────────────────────
J_STRIP=$(sbatch --parsable \
    $TRAIN_DEP \
    --partition=pibu_el8 --account=p774 \
    --mem=8G --cpus-per-task=2 --time=01:00:00 \
    --job-name=v12val_strip_${HOLDOUT_VEGA} \
    --output="$LOGDIR/v12val_strip_${HOLDOUT_VEGA}_%J.log" \
    --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
            python $REPO/scripts/strip_kinetics.py '$ALIGNED_BAM' '$STRIPPED_BAM'")

# ── 2. KinSim generate (BAM mode) ─────────────────────────────────────
# Pick the latest checkpoint at job-start time (after train completes).
J_GEN=$(sbatch --parsable \
    --dependency=afterok:$J_STRIP \
    --partition=pgpu --account=p774 \
    --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=02:00:00 \
    --job-name=v12val_generate_${HOLDOUT_VEGA} \
    --output="$LOGDIR/v12val_generate_${HOLDOUT_VEGA}_%J.log" \
    --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
            CKPT=\$(ls -t '$CKPT_DIR'/checkpoint_epoch*.pt 2>/dev/null | head -1); \
            [ -f \"\$CKPT\" ] || { echo 'ERROR: no checkpoint in $CKPT_DIR'; exit 1; }; \
            echo \"Using checkpoint: \$CKPT\"; \
            kinsim generate '$STRIPPED_BAM' '$REF' \"\$CKPT\" '$GROUND_TRUTH_MOTIFS' '$SIM_BAM'; \
            apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.3.sif pbindex '$SIM_BAM' || true")

# ── 3. ipdSummary on simulated BAM ────────────────────────────────────
J_IPD=$(sbatch --parsable \
    --dependency=afterok:$J_GEN \
    --job-name=v12val_ipd_${HOLDOUT_VEGA} \
    --output="$LOGDIR/v12val_ipd_${HOLDOUT_VEGA}_%J.log" \
    "$CALLERS/ipdsummary.slurm" "$SIM_BAM" "$REF" "$SIM_GFF" "$SIM_IPD_CSV")

# ── 4. pbmotifmaker on simulated GFF ──────────────────────────────────
J_MM=$(sbatch --parsable \
    --dependency=afterok:$J_IPD \
    --job-name=v12val_mm_${HOLDOUT_VEGA} \
    --output="$LOGDIR/v12val_mm_${HOLDOUT_VEGA}_%J.log" \
    "$CALLERS/pbmotifmaker.slurm" "$REF" "$SIM_GFF" "$SIM_MM_CSV")

# ── 5. merge with existing jasmine output ─────────────────────────────
# Reuses the original jasmine call (from the real bc2046 BAM) — the
# generated kinetics shouldn't change the 5mC modkit answer because
# bc2046 is the E. coli holdout (Dam/Dcm only, no 5mC by design).
J_MG=$(sbatch --parsable \
    --dependency=afterok:$J_MM \
    --job-name=v12val_merge_${HOLDOUT_VEGA} \
    --output="$LOGDIR/v12val_merge_${HOLDOUT_VEGA}_%J.log" \
    "$CALLERS/merge_motifs.slurm" "$SIM_MERGED_CSV" "$MERGE_THRESHOLD" "$SIM_MM_CSV" "$JASMINE_CSV")

echo "── Validation chain submitted ──"
printf '  %-25s : %s\n' "1. strip kinetics"     "$J_STRIP"
printf '  %-25s : %s\n' "2. kinsim generate"    "$J_GEN  (after $J_STRIP)"
printf '  %-25s : %s\n' "3. ipdSummary"         "$J_IPD  (after $J_GEN)"
printf '  %-25s : %s\n' "4. pbmotifmaker"       "$J_MM   (after $J_IPD)"
printf '  %-25s : %s\n' "5. merge motifs"       "$J_MG   (after $J_MM)"
echo ""
echo "Final output:"
echo "  $SIM_MERGED_CSV"
echo "Compare with ground truth:"
echo "  $GROUND_TRUTH_MOTIFS"
echo ""
echo "Estimated wall (after train):"
echo "  strip          : ~10–20 min"
echo "  generate       : ~10–30 min  (GPU)"
echo "  ipdSummary     : ~2–4 h"
echo "  pbmotifmaker   : ~1–2 h"
echo "  merge          : <5 min"
echo "  Total          : ~3–6 h"
