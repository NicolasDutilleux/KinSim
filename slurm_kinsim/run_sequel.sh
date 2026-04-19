#!/bin/bash
# ============================================================
# Sequel II 48-plex — pipeline orchestrator
#
# Note: process (subread align + ipdSummary) is already done for the
# 48 barcodes on disk. Keep the step here so it can be re-run if needed.
#
# Steps:
#   process   pbmm2 align subreads + ipdSummary    (array 1-N_SPECIES)
#   manifest  build GFF manifest CSV (local, fast)
#   extract   kinsim extract per species           (array)
#   merge     kinsim merge → master pkl
#   all       manifest → extract → merge chained with --dependency
#
# Usage:
#   bash slurm_kinsim/run_sequel.sh <step>
#   bash slurm_kinsim/run_sequel.sh all
#
# Edit the CONFIG block below to modulate paths / concurrency.
# ============================================================

set -euo pipefail
source ~/.bashrc
conda activate kinsim_env

# ============ CONFIG ============
BASE=/data/projects/p774_MARSD/NDutilleux/training
SEQUEL=${BASE}/Sequel
MANIFEST=${SEQUEL}/manifest_sequel_gff.csv
SHARDS=${SEQUEL}/shards_gff
MASTER=${SEQUEL}/master_sequel_gff.pkl
LOGS=${BASE}/logs

N_SPECIES=48
PROC_CONCURRENT=5
EXTRACT_CONCURRENT=4
# ================================

mkdir -p "$LOGS"

# ============ STEP SUBMITTERS ============

submit_process() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d --array=1-${N_SPECIES}%${PROC_CONCURRENT} \
        slurm_kinsim/sequel_02_ipdsummary.slurm
}

# Manifest build is fast (reads 48 dirs, writes CSV). Run as a tiny
# sbatch job so the workflow is uniform and captures a jobid.
submit_manifest() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=sequel_manifest \
        --output=${LOGS}/sequel_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
bash slurm_kinsim/sequel_04_build_manifest.sh"
}

submit_extract() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    [ -s "$MANIFEST" ] || { echo "ERROR: manifest missing ($MANIFEST). Run: manifest" >&2; exit 1; }
    local n; n=$(kinsim-prep manifest count "$MANIFEST")
    KINSIM_NO_AUTOMERGE=1 sbatch --parsable $d \
        --array=1-${n}%${EXTRACT_CONCURRENT} \
        slurm_kinsim/00_extract.slurm "$MANIFEST" "$SHARDS" "$MASTER"
}

submit_merge() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterany:${dep}"
    sbatch --parsable $d \
        slurm_kinsim/00_extract.slurm "$MANIFEST" "$SHARDS" "$MASTER"
}

submit_extract_merge_wrapper() {
    local dep=$1
    sbatch --parsable --dependency=afterok:${dep} \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=sequel_extract_submit \
        --output=${LOGS}/sequel_extract_submit_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
N=\$(kinsim-prep manifest count '${MANIFEST}') && \
EXT=\$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable --array=1-\${N}%${EXTRACT_CONCURRENT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"sequel.extract: \$EXT\" && \
MRG=\$(sbatch --parsable --dependency=afterany:\${EXT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"sequel.merge:   \$MRG\""
}

# ============ DISPATCH ============
STEP=${1:-}

case "$STEP" in
    process)  J=$(submit_process);  echo "sequel.process:  $J (array 1-${N_SPECIES})" ;;
    manifest) J=$(submit_manifest); echo "sequel.manifest: $J" ;;
    extract)  J=$(submit_extract);  echo "sequel.extract:  $J" ;;
    merge)    J=$(submit_merge);    echo "sequel.merge:    $J" ;;
    all)
        # process is already done on disk → skip by default. If you need
        # to re-run it, call: bash run_sequel.sh process
        J1=$(submit_manifest);                    echo "sequel.manifest:  $J1"
        J2=$(submit_extract_merge_wrapper "$J1"); echo "sequel.ext+mrg:   $J2 (after $J1)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/sequel_*.log | head"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/run_sequel.sh <step>

Steps:
  process    pbmm2 align subreads + ipdSummary   (array 1-${N_SPECIES})
             already complete on disk — only re-run if you need to redo it.
  manifest   build GFF manifest CSV
  extract    kinsim extract per species          (array)
  merge      kinsim merge → master pkl
  all        manifest → extract → merge chained with --dependency

Paths (edit CONFIG at top of file to change):
  manifest   $MANIFEST
  shards     $SHARDS
  master     $MASTER
EOF
        exit 1
        ;;
esac
