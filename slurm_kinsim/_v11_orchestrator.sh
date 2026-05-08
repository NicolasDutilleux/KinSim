#!/bin/bash
# ============================================================
# _v11_orchestrator.sh — runs after prep finishes (Strepto + Vega).
#
# Builds the v11 manifest from regenerated motifs_merged.csv files,
# then submits extract → refine → train chained on each other.
# Called from launch_v11.sh as a sbatch --wrap job.
#
# Args:  PREFIX  STREPTO_DIR  VEGA_DIR  HOLDOUT_STREPTO  HOLDOUT_VEGA
# ============================================================

PREFIX=${1:?"PREFIX missing"}
STREPTO=${2:?"STREPTO dir missing"}
VEGA=${3:?"VEGA dir missing"}
HOLDOUT_STREPTO=${4:?"HOLDOUT_STREPTO missing"}
HOLDOUT_VEGA=${5:?"HOLDOUT_VEGA missing"}

# /etc/bashrc has unbound vars; toggle -u around bashrc/conda init.
set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

REPO=/data/users/ndutilleux/KinSim
MANIFEST=$PREFIX/manifest.csv
LOGDIR=/data/projects/p774_MARSD/NDutilleux/logs

echo "=== v11 orchestrator ==="
echo "Date:     $(date)"
echo "PREFIX:   $PREFIX"
echo ""

# ── Build v11 manifest ─────────────────────────────────────────
echo "── Building v11 manifest ──"
mkdir -p "$PREFIX"
echo "sample_id,bam_path,motifs,ref_path" > "$MANIFEST"

n_strepto=0; skipped_strepto=()
for d in "$STREPTO"/pipeline/bc20*/; do
    bc=$(basename "$d")
    [ "$bc" = "$HOLDOUT_STREPTO" ] && { skipped_strepto+=("$bc:HOLDOUT"); continue; }
    bam="$d/${bc}_aligned.bam"
    motifs="$d/${bc}_motifs_merged.csv"
    ref="$STREPTO/$bc/final_assembly.fasta"
    if [ -f "$bam" ] && [ -s "$motifs" ] && [ -f "$ref" ]; then
        echo "strepto_$bc,$bam,$motifs,$ref" >> "$MANIFEST"
        n_strepto=$((n_strepto + 1))
    else
        skipped_strepto+=("$bc:missing-or-empty")
    fi
done

n_vega=0; skipped_vega=()
for d in "$VEGA"/pipeline/bc20*/; do
    bc=$(basename "$d")
    [ "$bc" = "$HOLDOUT_VEGA" ] && { skipped_vega+=("$bc:HOLDOUT_E.coli"); continue; }
    bam="$d/${bc}_aligned.bam"
    motifs="$d/${bc}_motifs_merged.csv"
    ref="$d/${bc}_assembly.fasta"
    if [ -f "$bam" ] && [ -s "$motifs" ] && [ -f "$ref" ]; then
        echo "vega_$bc,$bam,$motifs,$ref" >> "$MANIFEST"
        n_vega=$((n_vega + 1))
    else
        skipped_vega+=("$bc:missing-or-empty")
    fi
done

echo "  Strepto: $n_strepto strains  (skipped: ${skipped_strepto[*]:-none})"
echo "  Vega:    $n_vega strains  (skipped: ${skipped_vega[*]:-none})"

python "$REPO/scripts/manifest.py" validate "$MANIFEST"
N=$(python "$REPO/scripts/manifest.py" count "$MANIFEST")
echo "  Total: $N strains"
echo ""

if [ "$N" -lt 1 ]; then
    echo "ERROR: manifest is empty after filter — no strains to extract on."
    exit 1
fi

# ── Submit extract → (analyze ‖ refine → (train ‖ analyze)) chain ──
echo "── Submitting extract → refine → train chain (with parallel analyze branches) ──"

mkdir -p "$PREFIX/shards" "$PREFIX/refined" "$PREFIX/checkpoints" \
         "$PREFIX/reports/extract" "$PREFIX/reports/refined"

J_EXTRACT=$(sbatch --parsable \
    --array=1-${N} \
    --partition=pibu_el8 --account=p774 \
    --mem=192G --cpus-per-task=1 --time=12:00:00 \
    --job-name=v11_extract \
    --output="$LOGDIR/ml_00_extract_v11_%A_%a.log" \
    "$REPO/slurm_kinsim/ml/00_extract.slurm" "$MANIFEST" "$PREFIX/shards")

# Analyze pre-refine: one task per strain shard, runs in parallel with refine.
J_AN_EXTRACT=$(sbatch --parsable \
    --array=1-${N} \
    --dependency=afterok:$J_EXTRACT \
    --partition=pibu_el8 --account=p774 \
    --mem=128G --cpus-per-task=4 --time=00:45:00 \
    --job-name=v11_analyze_extract \
    --output="$LOGDIR/ml_01_analyze_extract_v11_%A_%a.log" \
    --wrap="bash $REPO/slurm_kinsim/ml/01_analyze_array.sh '$PREFIX/shards' '$PREFIX/reports/extract'")

J_REFINE=$(sbatch --parsable \
    --dependency=afterany:$J_EXTRACT \
    --partition=pibu_el8 --account=p774 \
    --mem=96G --cpus-per-task=4 --time=06:00:00 \
    --job-name=v11_refine \
    --output="$LOGDIR/ml_02_refine_v11_%J.log" \
    "$REPO/slurm_kinsim/ml/02_refine.slurm" "$PREFIX/shards" "$PREFIX/refined")

# Analyze post-refine: one task per refined shard, runs in parallel with train.
J_AN_REFINED=$(sbatch --parsable \
    --array=1-${N} \
    --dependency=afterok:$J_REFINE \
    --partition=pibu_el8 --account=p774 \
    --mem=128G --cpus-per-task=4 --time=00:45:00 \
    --job-name=v11_analyze_refined \
    --output="$LOGDIR/ml_01_analyze_refined_v11_%A_%a.log" \
    --wrap="bash $REPO/slurm_kinsim/ml/01_analyze_array.sh '$PREFIX/refined' '$PREFIX/reports/refined'")

J_TRAIN=$(sbatch --parsable \
    --dependency=afterok:$J_REFINE \
    --partition=pgpu --account=p774 \
    --gres=gpu:1 --mem=64G --cpus-per-task=4 --time=24:00:00 \
    --job-name=v11_train \
    --output="$LOGDIR/ml_03_train_v11_%J.log" \
    "$REPO/slurm_kinsim/ml/03_train.slurm" "$PREFIX/refined" "$PREFIX/checkpoints")

echo "  extract array       : $J_EXTRACT       ($N tasks)"
echo "  analyze pre-refine  : $J_AN_EXTRACT   ($N tasks, parallel with refine)"
echo "  refine              : $J_REFINE       (after $J_EXTRACT)"
echo "  analyze post-refine : $J_AN_REFINED   ($N tasks, parallel with train)"
echo "  train               : $J_TRAIN        (after $J_REFINE)"
echo ""
echo "=== orchestrator done ==="
