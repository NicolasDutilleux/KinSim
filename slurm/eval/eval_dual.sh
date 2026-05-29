#!/bin/bash
# Submit a parallel pair of `kinsim_nn evaluate` jobs on a single run
# directory: one for the in-training "best" generator (best_G.pt) and one
# for the latest training snapshot (G.pt). The two TSVs produced are then
# directly comparable.
#
# Usage:
#   bash slurm/eval/eval_dual.sh [<run_dir>]
#
# <run_dir> defaults to the production v12_strepto_vega run. It must
# contain:
#   - ckpts_v2_big/{G.pt, best_G.pt, model_config.json}
#   - shards_nn/*_shard.pkl
#
# Outputs:
#   <run_dir>/eval_best_G/best_G_stats.tsv
#   <run_dir>/eval_current_G/current_G_stats.tsv
#
# After the two jobs finish, read the results with:
#   bash slurm/eval/eval_show.sh [<run_dir>]
set -euo pipefail

RUN_DIR=${1:-/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega}
CKPT_DIR=$RUN_DIR/ckpts_v2_big
SHARDS=$RUN_DIR/shards_nn
LOG_DIR=/data/projects/p774_MARSD/NDutilleux/logs

# --- preflight ----------------------------------------------------------
for required in "$CKPT_DIR/G.pt" "$CKPT_DIR/best_G.pt" "$CKPT_DIR/model_config.json" "$SHARDS"; do
    if [ ! -e "$required" ]; then
        echo "ERROR: missing $required" >&2
        exit 1
    fi
done

N_SHARDS=$(ls "$SHARDS"/*_shard.pkl 2>/dev/null | wc -l)
if [ "$N_SHARDS" -eq 0 ]; then
    echo "ERROR: no *_shard.pkl in $SHARDS" >&2
    exit 1
fi

echo "Run dir:    $RUN_DIR"
echo "Ckpt dir:   $CKPT_DIR (best_G + G + D)"
echo "Shards:     $SHARDS ($N_SHARDS shards)"
echo

# --- eval best_G.pt ----------------------------------------------------
# `_find_checkpoint` picks best_G.pt > G.pt > most recent *.pt, so pointing
# directly at $CKPT_DIR gets best_G.pt by default.
OUT_BEST=$RUN_DIR/eval_best_G
mkdir -p "$OUT_BEST"

J_BEST=$(sbatch --parsable \
    --partition=pgpu --gres=gpu:1 --mem=48G --time=01:00:00 \
    --account=p774 \
    --output=$LOG_DIR/%x_%J.log \
    --job-name=eval_best_G \
    --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
            kinsim_nn evaluate '$CKPT_DIR' '$SHARDS' --output-prefix '$OUT_BEST/best_G' -v")

# --- eval current G.pt -------------------------------------------------
# Expose only G.pt + model_config.json in a sub-dir so `_find_checkpoint`
# falls through to G.pt (best_G.pt is not present in this dir).
OUT_CUR=$RUN_DIR/eval_current_G
WORK=$OUT_CUR/ckpt
mkdir -p "$WORK"
ln -sf "$CKPT_DIR/G.pt"              "$WORK/G.pt"
ln -sf "$CKPT_DIR/model_config.json" "$WORK/model_config.json"

J_CUR=$(sbatch --parsable \
    --partition=pgpu --gres=gpu:1 --mem=48G --time=01:00:00 \
    --account=p774 \
    --output=$LOG_DIR/%x_%J.log \
    --job-name=eval_current_G \
    --wrap="set +u; source ~/.bashrc; conda activate kinsim_env; set -euo pipefail; \
            kinsim_nn evaluate '$WORK' '$SHARDS' --output-prefix '$OUT_CUR/current_G' -v")

# --- summary -----------------------------------------------------------
echo "Submitted:"
echo "  best_G    job=$J_BEST  out=$OUT_BEST/best_G_stats.tsv"
echo "  current_G job=$J_CUR   out=$OUT_CUR/current_G_stats.tsv"
echo
squeue -u "$USER"
echo
echo "When both finish, read the results with:"
echo "  bash slurm/eval/eval_show.sh $RUN_DIR"
