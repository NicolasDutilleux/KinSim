#!/bin/bash
# ============================================================
# launch_v12.sh — v12 lean relaunch (no bystrandify/align/index/jasmine).
#
# Architecture differences vs v11:
#   - Refine: 1D-IPD GMM, K∈{1,2,3} (K=1 wins on parsimony when no
#     separable contamination cluster exists; flags k1_null_signal).
#   - Refine drop rule: similarity-margin tunable (0.0 = strict
#     above-baseline, default for v12 first run).
#   - Train: kmer-aware FiLM ConvPredictor — γ, β depend on
#     concat(meth_proj, kmer_summary) so methylation modulation
#     varies by sequence context.
#   - 130 per-strain HTML reports written automatically
#     ($PREFIX/reports/{extract,refined}/).
#
# Preconditions (already on disk, kept):
#   - <strain>_aligned.bam  (per strain)
#   - <strain>_motifs_jasmine.csv  (per strain)
#   - <ref>.fasta  (per strain)
#
# Per training strain (51 Strepto + 14 Vega = 65, holdouts excluded):
#   1. ipdSummary       (--pvalue 0.2)
#   2. pbmotifmaker     (--min-score 25)              afterok ipd
#   3. merge_motifs     (--threshold 0.7, + jasmine)  afterok motifmaker
#
# Then ONE orchestrator job afterany on all 65 merges:
#   4. Build v12 manifest from regenerated motifs_merged.csv
#   5. extract array (per-strain, parallel)
#   6. analyze pre-refine array (per-strain, runs in parallel with refine)
#   7. refine (1D-IPD GMM, K∈{1,2,3}, similarity-margin=0.0)
#   8. analyze post-refine array (per-strain, runs in parallel with train)
#   9. train (kmer-aware FiLM ConvPredictor, GPU)
#
# Manifest excludes:
#   strepto bc2080 (Strepto holdout)
#   vega    bc2046 (E. coli holdout)
# ============================================================

set -euo pipefail
export TMPDIR=/tmp

PREFIX=/data/projects/p774_MARSD/NDutilleux/runs/v12_strepto_vega
STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
HOLDOUT_STREPTO=bc2080
HOLDOUT_VEGA=bc2046
LOGDIR=/data/projects/p774_MARSD/NDutilleux/logs

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
mkdir -p "$PREFIX" "$LOGDIR"

export MOTIFMAKER_MIN_SCORE=${MOTIFMAKER_MIN_SCORE:-25}
MERGE_THRESHOLD=${MERGE_THRESHOLD:-0.7}

CALLERS="$REPO/slurm_kinsim/callers"

echo "================================================================"
echo "  KinSim v12 launcher (1D-IPD GMM + kmer-aware FiLM)"
echo "  Orchestrator submits: extract → (analyze ‖ refine →"
echo "                        (analyze ‖ train))"
echo "  Refine: 1D-IPD GMM, K∈{1,2,3}, similarity-margin=0.0"
echo "  Train:  kmer-aware FiLM ConvPredictor"
echo ""
echo "  PREFIX:               $PREFIX"
echo "  MOTIFMAKER_MIN_SCORE: $MOTIFMAKER_MIN_SCORE"
echo "  MERGE_THRESHOLD:      $MERGE_THRESHOLD"
echo "  Holdouts (excluded):  strepto=$HOLDOUT_STREPTO  vega=$HOLDOUT_VEGA"
echo "================================================================"
echo ""

# Returns the merge job ID for one strain. Wipes GFF/motifmaker/merge outputs
# (NOT jasmine) and submits the lean chain ipd -> mm -> merge.
chain_one() {
    local pipe_dir=$1   # /pipeline/bcXXXX
    local aligned_bam=$2
    local ref=$3
    local sample_id=$4  # e.g. strepto_bc2033 (for the v12 manifest)

    local bc=$(basename "$pipe_dir")
    local gff="$pipe_dir/${bc}_ipdSummary.gff"
    local ipd_csv="$pipe_dir/${bc}_ipdSummary.csv"
    local mm_csv="$pipe_dir/${bc}_motifs_ipdsummary.csv"
    local jm_csv="$pipe_dir/${bc}_motifs_jasmine.csv"
    local merged_csv="$pipe_dir/${bc}_motifs_merged.csv"

    [ -f "$aligned_bam" ] || { echo "  SKIP $sample_id — no aligned BAM" >&2; echo ""; return; }
    [ -f "$ref" ]         || { echo "  SKIP $sample_id — no reference" >&2; echo ""; return; }
    [ -s "$jm_csv" ]      || { echo "  SKIP $sample_id — no motifs_jasmine.csv" >&2; echo ""; return; }

    # SKIP-first: if motifs_merged.csv already exists from a prior prep,
    # don't re-run ipd+mm+merge. Saves hours of compute on re-launches.
    # The merged.csv file is the only output the orchestrator/manifest
    # actually consumes; intermediate gff/ipd_csv/mm_csv aren't needed
    # downstream. Pass FORCE_PREP=1 to override (full wipe + re-run).
    if [ -s "$merged_csv" ] && [ "${FORCE_PREP:-0}" != "1" ]; then
        echo "  SKIP-PREP $sample_id — $merged_csv exists ($(wc -l < "$merged_csv") rows). Pass FORCE_PREP=1 to re-run." >&2
        echo ""
        return
    fi

    # Wipe everything we're about to regenerate (keep aligned BAM, ref, jasmine).
    rm -f "$gff" "$ipd_csv" "$mm_csv" "$merged_csv"

    local J_IPD=$(sbatch --parsable -J ${bc}_ipd \
        --output="$LOGDIR/${bc}_ipd_%J.log" \
        "$CALLERS/ipdsummary.slurm" "$aligned_bam" "$ref" "$gff" "$ipd_csv")

    local J_MM=$(sbatch --parsable --dependency=afterok:$J_IPD -J ${bc}_mm \
        --output="$LOGDIR/${bc}_mm_%J.log" \
        "$CALLERS/pbmotifmaker.slurm" "$ref" "$gff" "$mm_csv")

    local J_MG=$(sbatch --parsable --dependency=afterok:$J_MM -J ${bc}_merge \
        --output="$LOGDIR/${bc}_merge_%J.log" \
        "$CALLERS/merge_motifs.slurm" "$merged_csv" "$MERGE_THRESHOLD" "$mm_csv" "$jm_csv")

    printf '  %-15s :  ipd=%s  mm=%s  MERGE=%s\n' "$sample_id" "$J_IPD" "$J_MM" "$J_MG" >&2
    echo "$J_MG"
}

ALL_MERGES=()

# --- Strepto (52 in pipeline/ - 1 holdout = 51) ---
echo "── Strepto chains ──"
for d in "$STREPTO"/pipeline/bc20*/; do
    bc=$(basename "$d")
    [ "$bc" = "$HOLDOUT_STREPTO" ] && { echo "  SKIP strepto_$bc (HOLDOUT)" >&2; continue; }
    bam="$d/${bc}_aligned.bam"
    ref="$STREPTO/$bc/final_assembly.fasta"
    jid=$(chain_one "$d" "$bam" "$ref" "strepto_$bc")
    [ -n "$jid" ] && ALL_MERGES+=("$jid")
done
echo ""

# --- Vega (15 in pipeline/ - 1 holdout = 14) ---
echo "── Vega chains ──"
for d in "$VEGA"/pipeline/bc20*/; do
    bc=$(basename "$d")
    [ "$bc" = "$HOLDOUT_VEGA" ] && { echo "  SKIP vega_$bc (HOLDOUT_E.coli)" >&2; continue; }
    bam="$d/${bc}_aligned.bam"
    ref="$d/${bc}_assembly.fasta"
    jid=$(chain_one "$d" "$bam" "$ref" "vega_$bc")
    [ -n "$jid" ] && ALL_MERGES+=("$jid")
done
echo ""

N=${#ALL_MERGES[@]}
DEPS=$(IFS=:; echo "${ALL_MERGES[*]}")

echo "================================================================"
echo "  Submitted $N strain chains (ipd + mm + merge)"
echo "================================================================"

# --- Orchestrator: builds v12 manifest then chains extract+analyze+refine+analyze+train ---
J_ORCH=$(sbatch --parsable \
    --dependency=afterany:${DEPS} \
    --partition=pibu_el8 --account=p774 \
    --mem=4G --cpus-per-task=1 --time=00:30:00 \
    --job-name=v12_orchestrator \
    --output="$LOGDIR/v12_orchestrator_%J.log" \
    --wrap="bash $REPO/slurm_kinsim/_v12_orchestrator.sh '$PREFIX' '$STREPTO' '$VEGA' '$HOLDOUT_STREPTO' '$HOLDOUT_VEGA'")

echo ""
echo "  Orchestrator: $J_ORCH (after $N merges)"
echo "    → extract array (N=$N)"
echo "    → analyze pre-refine array (parallel with refine)"
echo "    → refine"
echo "    → analyze post-refine array (parallel with train)"
echo "    → train"
echo ""
echo "Watch:"
echo "  squeue -u \$USER | head -20"
echo "  tail -f $LOGDIR/v12_orchestrator_${J_ORCH}.log"
echo ""
echo "Reports will land at:"
echo "  $PREFIX/reports/extract/<sample_id>/  (pre-refine)"
echo "  $PREFIX/reports/refined/<sample_id>/  (post-refine)"
