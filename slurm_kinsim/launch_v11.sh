#!/bin/bash
# ============================================================
# launch_v11.sh — full re-run with score=25 + threshold=0.7
#
# Pipeline:
#   1. Verify jasmine outputs exist for all strains
#   2. Wipe ipdSummary + motifmaker + merge outputs (keep BAMs+refs+jasmine)
#   3. Run prep chain per-strain (strepto + vega run.sh):
#        ipdSummary → pbmotifmaker(score=25)
#                          ↓
#                       merge(thresh=0.7) ← jasmine (existing)
#   4. After ALL merges done: build v11 manifest, then chain:
#        extract → refine → train
#
# Dependencies are wired via SLURM afterok/afterany so the whole pipeline
# is launched in one shot — the user can disconnect.
#
# Resource budget (verified against pibu_el8 + pgpu):
#   ipdSummary  : 48G mem, 8 cpu, 12h    (callers/ipdsummary.slurm)
#   motifmaker  : 16G mem, 4 cpu, 12h    (callers/pbmotifmaker.slurm, score=25)
#   jasmine     : SKIPPED (output exists)
#   merge       :  2G mem, 1 cpu, 15min  (callers/merge_motifs.slurm)
#   extract     : 192G mem, 1 cpu, 12h   (--mem override on ml/00_extract.slurm)
#   refine      : 96G mem, 4 cpu, 6h     (ml/02_refine.slurm)
#   train       : 64G mem, GPU, 24h      (ml/03_train.slurm)
#
# Holdouts (excluded from v11 manifest, stay in raw datasets):
#   strepto bc2080
#   vega    bc2046  (E. coli)
#
# Usage:
#   bash slurm_kinsim/launch_v11.sh
# ============================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
PREFIX=/data/projects/p774_MARSD/NDutilleux/runs/v11_strepto_vega_score25
STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
HOLDOUT_STREPTO=bc2080
HOLDOUT_VEGA=bc2046
LOGDIR=/data/projects/p774_MARSD/NDutilleux/logs

# Prep params (env-overridable on this script's own command line)
export MOTIFMAKER_MIN_SCORE=${MOTIFMAKER_MIN_SCORE:-25}
export MERGE_THRESHOLD=${MERGE_THRESHOLD:-0.7}

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
mkdir -p "$PREFIX" "$LOGDIR"

echo "================================================================"
echo "  KinSim v11 launcher"
echo "  PREFIX:           $PREFIX"
echo "  MOTIFMAKER_MIN_SCORE: $MOTIFMAKER_MIN_SCORE"
echo "  MERGE_THRESHOLD:      $MERGE_THRESHOLD"
echo "  Holdouts:         strepto=$HOLDOUT_STREPTO  vega=$HOLDOUT_VEGA"
echo "================================================================"
echo ""

# ── 1. Verify jasmine outputs exist for all strains ─────────────────
echo "── 1/6  Verifying jasmine_motifs.csv exists for every strain ──"
missing=()
for d in "$STREPTO"/pipeline/bc20*/ "$VEGA"/pipeline/bc20*/; do
    bc=$(basename "$d")
    f="$d/${bc}_motifs_jasmine.csv"
    [ -f "$f" ] || missing+=("$bc:$d")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: ${#missing[@]} strains missing jasmine output:"
    for m in "${missing[@]}"; do echo "  $m"; done
    echo ""
    echo "Re-run jasmine first (slurm_kinsim/callers/jasmine_modkit.slurm) or"
    echo "remove those strains from the manifests."
    exit 1
fi
echo "  All jasmine outputs present."
echo ""

# ── 2. Wipe ipdSummary + motifmaker + merge outputs ─────────────────
echo "── 2/6  Wiping prep outputs (keep BAMs + refs + jasmine) ──"
n_wiped=0
for d in "$STREPTO"/pipeline/bc20*/ "$VEGA"/pipeline/bc20*/; do
    bc=$(basename "$d")
    rm -f "$d/${bc}_ipdSummary.gff" "$d/${bc}_ipdSummary.csv"
    rm -f "$d/${bc}_motifs_ipdsummary.csv"
    rm -f "$d/${bc}_motifs_merged.csv"
    n_wiped=$((n_wiped + 1))
done
echo "  Wiped outputs for $n_wiped strain dirs."
echo ""

# ── 3. Strepto prep chain ──────────────────────────────────────────
echo "── 3/6  Submitting Strepto prep chain ──"
STREPTO_OUT=$(mktemp)
bash "$REPO/slurm_kinsim/strepto/run.sh" all > "$STREPTO_OUT" 2>&1 || {
    echo "ERROR: Strepto run.sh failed:"
    cat "$STREPTO_OUT"
    exit 1
}
cat "$STREPTO_OUT"
STREPTO_FENCE=$(grep "strepto.manifest:" "$STREPTO_OUT" | awk '{print $2}')
[ -n "$STREPTO_FENCE" ] || { echo "ERROR: could not capture strepto fence job ID"; exit 1; }
rm -f "$STREPTO_OUT"
echo "  → Strepto fence (manifest builder): $STREPTO_FENCE"
echo ""

# ── 4. Vega prep chain ─────────────────────────────────────────────
echo "── 4/6  Submitting Vega prep chain ──"
VEGA_OUT=$(mktemp)
bash "$REPO/slurm_kinsim/vega/run.sh" all > "$VEGA_OUT" 2>&1 || {
    echo "ERROR: Vega run.sh failed:"
    cat "$VEGA_OUT"
    exit 1
}
cat "$VEGA_OUT"
VEGA_FENCE=$(grep "vega.manifest:" "$VEGA_OUT" | awk '{print $2}')
[ -n "$VEGA_FENCE" ] || { echo "ERROR: could not capture vega fence job ID"; exit 1; }
rm -f "$VEGA_OUT"
echo "  → Vega fence (manifest builder): $VEGA_FENCE"
echo ""

# ── 5. Build v11 manifest + submit extract+refine+train chain ──────
# This is a single sbatch job that runs after both fences complete.
# Inside it: build manifest, count strains, submit extract array
# with --array=1-N, then refine + train chained.
echo "── 5/6  Submitting orchestrator job (builds manifest + extract+refine+train chain) ──"
ORCH_SCRIPT="$REPO/slurm_kinsim/_v11_orchestrator.sh"
J_ORCH=$(sbatch --parsable \
    --dependency=afterany:${STREPTO_FENCE}:${VEGA_FENCE} \
    --partition=pibu_el8 --account=p774 \
    --mem=4G --cpus-per-task=1 --time=00:30:00 \
    --job-name=v11_orchestrator \
    --output="$LOGDIR/v11_orchestrator_%J.log" \
    --wrap="bash $ORCH_SCRIPT '$PREFIX' '$STREPTO' '$VEGA' '$HOLDOUT_STREPTO' '$HOLDOUT_VEGA'")
echo "  → Orchestrator: $J_ORCH (depends on $STREPTO_FENCE + $VEGA_FENCE)"
echo ""

# ── 6. Summary ─────────────────────────────────────────────────────
echo "── 6/6  Chain submitted ──"
echo "  Strepto fence:    $STREPTO_FENCE"
echo "  Vega fence:       $VEGA_FENCE"
echo "  Orchestrator:     $J_ORCH (will print extract/refine/train IDs into its log)"
echo ""
echo "Watch progress with:"
echo "  squeue -u \$USER | head -30"
echo "  tail -f $LOGDIR/v11_orchestrator_${J_ORCH}.log    # extract+refine+train IDs once orchestrator runs"
echo ""
echo "Estimated wall clock:"
echo "  prep         : ~3-4 days"
echo "  extract      : ~6h after prep"
echo "  refine+train : ~25h after extract"
echo "  Total        : ~5 days"
