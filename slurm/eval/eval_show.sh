#!/bin/bash
# Read the two TSVs produced by slurm/eval/eval_dual.sh, side-by-side, and
# tail the SLURM logs of the two most recent eval jobs.
#
# Usage:
#   bash slurm/eval/eval_show.sh [<run_dir>]
set -euo pipefail

RUN_DIR=${1:-/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega}
LOG_DIR=/data/projects/p774_MARSD/NDutilleux/logs

BEST_TSV=$RUN_DIR/eval_best_G/best_G_stats.tsv
CUR_TSV=$RUN_DIR/eval_current_G/current_G_stats.tsv

echo "########## best_G_stats.tsv ##########"
if [ -s "$BEST_TSV" ]; then
    column -t -s $'\t' "$BEST_TSV"
else
    echo "(missing or empty: $BEST_TSV)"
fi

echo
echo "########## current_G_stats.tsv ##########"
if [ -s "$CUR_TSV" ]; then
    column -t -s $'\t' "$CUR_TSV"
else
    echo "(missing or empty: $CUR_TSV)"
fi

# Side-by-side delta: meth_id, W1(best), W1(current), delta
if [ -s "$BEST_TSV" ] && [ -s "$CUR_TSV" ]; then
    echo
    echo "########## W1 delta (current_G − best_G) ##########"
    python3 - <<PY
import csv
def load(p):
    with open(p) as f:
        return {row["meth_id"]: row for row in csv.DictReader(f, delimiter="\t")}
b = load("$BEST_TSV")
c = load("$CUR_TSV")
print(f"{'meth_id':<8} {'meth_name':<10} {'W1(best)':>10} {'W1(current)':>12} {'Δ':>10}  verdict")
for k in sorted(set(b) | set(c)):
    rb, rc = b.get(k), c.get(k)
    if not (rb and rc): continue
    wb = float(rb['wasserstein_1d']); wc = float(rc['wasserstein_1d'])
    d = wc - wb
    v = 'best_G better' if d > 0.05 * max(wb, 1e-9) else ('current_G better' if d < -0.05 * max(wb, 1e-9) else 'comparable')
    print(f"{rb['meth_id']:<8} {rb['meth_name']:<10} {wb:>10.3f} {wc:>12.3f} {d:>+10.3f}  {v}")
PY
fi

echo
echo "########## last 20 lines of most recent eval_best_G log ##########"
LOG_BEST=$(ls -t "$LOG_DIR"/eval_best_G_*.log 2>/dev/null | head -1)
if [ -n "$LOG_BEST" ]; then
    echo "log: $LOG_BEST"
    tail -20 "$LOG_BEST"
else
    echo "(no eval_best_G log found)"
fi

echo
echo "########## last 20 lines of most recent eval_current_G log ##########"
LOG_CUR=$(ls -t "$LOG_DIR"/eval_current_G_*.log 2>/dev/null | head -1)
if [ -n "$LOG_CUR" ]; then
    echo "log: $LOG_CUR"
    tail -20 "$LOG_CUR"
else
    echo "(no eval_current_G log found)"
fi
