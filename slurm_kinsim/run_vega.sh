#!/bin/bash
# ============================================================
# Vega HMB — pipeline orchestrator
#
# Steps (each submits a single SLURM job you can copy the ID from):
#   process   assemble + bystrandify + align + ipdSummary   (array 1-N_SPECIES)
#   manifest  build GFF manifest CSV
#   extract   kinsim extract per species                    (array)
#   merge     kinsim merge → master pkl
#   all       chain all of the above with --dependency=afterok
#
# Usage:
#   bash slurm_kinsim/run_vega.sh <step>
#   bash slurm_kinsim/run_vega.sh all
#
# Any step can be re-run independently — each is idempotent and skips
# outputs that already exist. Edit the CONFIG block below to modulate
# paths / concurrency / species count.
# ============================================================

set -euo pipefail
source ~/.bashrc
conda activate kinsim_env

# ============ CONFIG ============
BASE=/data/projects/p774_MARSD/NDutilleux/training
VEGA=${BASE}/Vega
MANIFEST=${VEGA}/manifest_vega_gff.csv
SHARDS=${VEGA}/shards_gff
MASTER=${VEGA}/master_vega_gff.pkl
LOGS=${BASE}/logs

N_SPECIES=16
PROC_CONCURRENT=4
EXTRACT_CONCURRENT=4
# ================================

mkdir -p "$LOGS"

# ============ STEP SUBMITTERS ============
# Each submitter optionally accepts a dependency jobid as $1.
# They echo only the submitted jobid (so callers can capture it).

submit_process() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d --array=1-${N_SPECIES}%${PROC_CONCURRENT} \
        slurm_kinsim/vega_01_process_all.slurm
}

submit_manifest() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=vega_manifest \
        --output=${LOGS}/vega_manifest_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
bash slurm_kinsim/vega_02_build_manifest.sh"
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

# "extract+merge after manifest" as a single wrapper — needed when
# you run `all` because the extract array size isn't known until the
# manifest file is built.
submit_extract_merge_wrapper() {
    local dep=$1
    sbatch --parsable --dependency=afterok:${dep} \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=vega_extract_submit \
        --output=${LOGS}/vega_extract_submit_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
N=\$(kinsim-prep manifest count '${MANIFEST}') && \
EXT=\$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable --array=1-\${N}%${EXTRACT_CONCURRENT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"vega.extract: \$EXT\" && \
MRG=\$(sbatch --parsable --dependency=afterany:\${EXT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"vega.merge:   \$MRG\""
}

# ============ DISPATCH ============
STEP=${1:-}

case "$STEP" in
    process)  J=$(submit_process);  echo "vega.process:  $J (array 1-${N_SPECIES})" ;;
    manifest) J=$(submit_manifest); echo "vega.manifest: $J" ;;
    extract)  J=$(submit_extract);  echo "vega.extract:  $J (array)" ;;
    merge)    J=$(submit_merge);    echo "vega.merge:    $J" ;;
    all)
        J1=$(submit_process);                    echo "vega.process:   $J1 (array 1-${N_SPECIES})"
        J2=$(submit_manifest "$J1");             echo "vega.manifest:  $J2 (after $J1)"
        J3=$(submit_extract_merge_wrapper "$J2");echo "vega.ext+mrg:   $J3 (submits extract+merge after $J2)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/vega_*.log | head"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/run_vega.sh <step>

Steps:
  process    assemble + bystrandify + align + ipdSummary  (array 1-${N_SPECIES})
  manifest   build GFF manifest from processed outputs
  extract    kinsim extract per species                   (array)
  merge      kinsim merge → master pkl
  all        chain everything with --dependency=afterok

Paths (edit CONFIG at top of file to change):
  manifest   $MANIFEST
  shards     $SHARDS
  master     $MASTER
EOF
        exit 1
        ;;
esac
