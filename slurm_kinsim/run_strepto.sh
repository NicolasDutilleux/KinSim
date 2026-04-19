#!/bin/bash
# ============================================================
# Streptomyces — pipeline orchestrator
#
# Steps:
#   process   bystrandify + align + ipdSummary    (array, 1 per strain in ORIG_MANIFEST)
#   manifest  build GFF manifest CSV
#   extract   kinsim extract per species          (array)
#   merge     kinsim merge → master pkl
#   all       chain everything with --dependency=afterok
#
# Usage:
#   bash slurm_kinsim/run_strepto.sh <step>
#   bash slurm_kinsim/run_strepto.sh all
#
# Edit the CONFIG block below to modulate paths / concurrency.
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# ============ CONFIG ============
BASE=/data/projects/p774_MARSD/NDutilleux/training
STREPTO=${BASE}/Strepto

# Input: raw-HiFi sample manifest (sample_id, bam_path, motifs)
ORIG_MANIFEST=${STREPTO}/manifest_strepto.csv

# Working dir for bystrandify + aligned BAMs + GFF
IPD_DIR=${STREPTO}/gff_pipeline

# GFF manifest (built after process)
MANIFEST=${STREPTO}/manifest_strepto_gff.csv

# Extract outputs
SHARDS=${STREPTO}/shards_gff
MASTER=${STREPTO}/master_strepto_gff.pkl

LOGS=${BASE}/logs

PROC_CONCURRENT=4
EXTRACT_CONCURRENT=4
# ================================

mkdir -p "$LOGS"

count_rows() { kinsim-prep manifest count "$1"; }

# ============ STEP SUBMITTERS ============

submit_process() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    [ -s "$ORIG_MANIFEST" ] || { echo "ERROR: ORIG_MANIFEST missing ($ORIG_MANIFEST)" >&2; exit 1; }
    local n; n=$(count_rows "$ORIG_MANIFEST")
    sbatch --parsable $d --array=1-${n}%${PROC_CONCURRENT} \
        slurm_kinsim/strepto_00_bystrandify_ipd.slurm \
        "$ORIG_MANIFEST" "$IPD_DIR"
}

submit_manifest() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=strepto_manifest \
        --output=${LOGS}/strepto_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
bash slurm_kinsim/strepto_01_build_gff_manifest.sh \
    '${ORIG_MANIFEST}' '${IPD_DIR}' '${MANIFEST}'"
}

submit_extract() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    [ -s "$MANIFEST" ] || { echo "ERROR: manifest missing ($MANIFEST). Run: manifest" >&2; exit 1; }
    local n; n=$(count_rows "$MANIFEST")
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
        --job-name=strepto_extract_submit \
        --output=${LOGS}/strepto_extract_submit_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
N=\$(kinsim-prep manifest count '${MANIFEST}') && \
EXT=\$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable --array=1-\${N}%${EXTRACT_CONCURRENT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"strepto.extract: \$EXT\" && \
MRG=\$(sbatch --parsable --dependency=afterany:\${EXT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"strepto.merge:   \$MRG\""
}

# ============ DISPATCH ============
STEP=${1:-}

case "$STEP" in
    process)  J=$(submit_process);  echo "strepto.process:  $J" ;;
    manifest) J=$(submit_manifest); echo "strepto.manifest: $J" ;;
    extract)  J=$(submit_extract);  echo "strepto.extract:  $J" ;;
    merge)    J=$(submit_merge);    echo "strepto.merge:    $J" ;;
    all)
        J1=$(submit_process);                     echo "strepto.process:   $J1"
        J2=$(submit_manifest "$J1");              echo "strepto.manifest:  $J2 (after $J1)"
        J3=$(submit_extract_merge_wrapper "$J2"); echo "strepto.ext+mrg:   $J3 (after $J2)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/strepto_*.log | head"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/run_strepto.sh <step>

Steps:
  process    bystrandify + align + ipdSummary    (array from ORIG_MANIFEST)
  manifest   build GFF manifest from processed outputs
  extract    kinsim extract per species          (array)
  merge      kinsim merge → master pkl
  all        chain everything with --dependency=afterok

Paths (edit CONFIG at top of file to change):
  orig_manifest $ORIG_MANIFEST
  ipd_dir       $IPD_DIR
  manifest      $MANIFEST
  shards        $SHARDS
  master        $MASTER
EOF
        exit 1
        ;;
esac
