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

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# ============ CONFIG ============
BASE=/data/projects/p774_MARSD/NDutilleux/training
SEQUEL=${BASE}/Sequel

# --- Track A: subreads → pbmm2 SUBREAD → ipdSummary (default model) ---
MANIFEST=${SEQUEL}/manifest_sequel_gff.csv
SHARDS=${SEQUEL}/shards_gff
MASTER=${SEQUEL}/master_sequel_gff.pkl

# --- Track B: subreads → ccs → bystrandify → pbmm2 CCS → ipdSummary SP3-C3 ---
MANIFEST_CCS=${SEQUEL}/manifest_sequel_ccs_gff.csv
SHARDS_CCS=${SEQUEL}/shards_ccs_gff
MASTER_CCS=${SEQUEL}/master_sequel_ccs_gff.pkl

LOGS=${BASE}/logs

N_SPECIES=48
PROC_CONCURRENT=5
CCS_CONCURRENT=4
EXTRACT_CONCURRENT=4

# Nextflow (alternative to the bash `process` + `manifest` steps)
NF_OUTDIR=${SEQUEL}/prepare
NF_PARAMS=nextflow/params/sequel.yaml
NF_PROFILE=sequel,slurm
NF_MANIFEST_NAME=manifest_sequel_gff.csv
# ================================

mkdir -p "$LOGS"

# ============ STEP SUBMITTERS ============

submit_process() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d --array=1-${N_SPECIES}%${PROC_CONCURRENT} \
        slurm_kinsim/sequel_02_ipdsummary.slurm
}

# Track B: ccs → bystrandify → pbmm2 CCS → ipdSummary SP3-C3 (parallel to process)
submit_process_ccs() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d --array=1-${N_SPECIES}%${CCS_CONCURRENT} \
        slurm_kinsim/sequel_03_ccs_pipeline.slurm
}

# Nextflow launcher — runs the PREPARE workflow (subread → pbmm2 SUBREAD →
# index → ipdSummary container) and writes the GFF manifest.
submit_nf() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=8G --cpus-per-task=2 --time=24:00:00 \
        --job-name=sequel_nf \
        --output=${LOGS}/sequel_nf_%J.log \
        --wrap="set +u && source ~/.bashrc && conda activate kinsim_env && set -euo pipefail && \
cd \"\$(git rev-parse --show-toplevel 2>/dev/null || pwd)\" && \
nextflow run nextflow/main.nf -profile ${NF_PROFILE} \
    -params-file ${NF_PARAMS} \
    --outdir '${NF_OUTDIR}' \
    --manifest_name '${NF_MANIFEST_NAME}' \
    -resume && \
ln -sf '${NF_OUTDIR}/${NF_MANIFEST_NAME}' '${MANIFEST}'"
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

submit_manifest_ccs() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=sequel_manifest_ccs \
        --output=${LOGS}/sequel_manifest_ccs_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
bash slurm_kinsim/sequel_04_build_manifest_ccs.sh"
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

submit_extract_merge_wrapper_ccs() {
    local dep=$1
    sbatch --parsable --dependency=afterok:${dep} \
        --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
        --job-name=sequel_extract_submit_ccs \
        --output=${LOGS}/sequel_extract_submit_ccs_%J.log \
        --wrap="source ~/.bashrc && conda activate kinsim_env && \
N=\$(kinsim-prep manifest count '${MANIFEST_CCS}') && \
EXT=\$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable --array=1-\${N}%${EXTRACT_CONCURRENT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST_CCS}' '${SHARDS_CCS}' '${MASTER_CCS}') && \
echo \"sequel_ccs.extract: \$EXT\" && \
MRG=\$(sbatch --parsable --dependency=afterany:\${EXT} \
    slurm_kinsim/00_extract.slurm '${MANIFEST_CCS}' '${SHARDS_CCS}' '${MASTER_CCS}') && \
echo \"sequel_ccs.merge:   \$MRG\""
}

# ============ DISPATCH ============
STEP=${1:-}

case "$STEP" in
    process)      J=$(submit_process);      echo "sequel.process:      $J (subread track, array 1-${N_SPECIES})" ;;
    process_ccs)  J=$(submit_process_ccs);  echo "sequel.process_ccs:  $J (CCS track, array 1-${N_SPECIES})" ;;
    manifest)     J=$(submit_manifest);     echo "sequel.manifest:     $J" ;;
    manifest_ccs) J=$(submit_manifest_ccs); echo "sequel.manifest_ccs: $J" ;;
    nf)           J=$(submit_nf);           echo "sequel.nf:           $J (Nextflow PREPARE → $NF_OUTDIR)" ;;
    extract)      J=$(submit_extract);      echo "sequel.extract:      $J (subread track)" ;;
    merge)        J=$(submit_merge);        echo "sequel.merge:        $J" ;;
    all)
        # subread process is already done on disk → skip by default. If you need
        # to re-run it, call: bash run_sequel.sh process
        J1=$(submit_manifest);                    echo "sequel.manifest:  $J1"
        J2=$(submit_extract_merge_wrapper "$J1"); echo "sequel.ext+mrg:   $J2 (after $J1)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/sequel_*.log | head"
        ;;
    all_ccs)
        # Track B: ccs → bystrandify → pbmm2 CCS → ipdSummary SP3-C3 → extract+merge
        J1=$(submit_process_ccs);                     echo "sequel.process_ccs:  $J1 (array 1-${N_SPECIES})"
        J2=$(submit_manifest_ccs "$J1");              echo "sequel.manifest_ccs: $J2 (after $J1)"
        J3=$(submit_extract_merge_wrapper_ccs "$J2"); echo "sequel.ext+mrg_ccs:  $J3 (after $J2)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/sequel_ccs_*.log ${LOGS}/sequel_manifest_ccs_*.log | head"
        ;;
    all_nf)
        J1=$(submit_nf);                          echo "sequel.nf:        $J1"
        J2=$(submit_extract_merge_wrapper "$J1"); echo "sequel.ext+mrg:   $J2 (after $J1)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/sequel_*.log | head"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/run_sequel.sh <step>

Two independent tracks — each builds its own manifest/shards/master:

  Track A (subreads, default model) — maximum statistical power
    process        pbmm2 align subreads + ipdSummary  (already done on disk)
    manifest       build GFF manifest CSV
    extract/merge  kinsim extract + merge → master
    all            chain: manifest → extract → merge

  Track B (ccs → bystrandify, SP3-C3 model) — apples-to-apples vs Vega/Strepto
    process_ccs    ccs + bystrandify + pbmm2 CCS + ipdSummary SP3-C3  (array 1-${N_SPECIES})
    manifest_ccs   build CCS GFF manifest CSV
    all_ccs        chain: process_ccs → manifest_ccs → extract+merge

  Other
    nf        Nextflow PREPARE — replaces Track A's process+manifest
    all_nf    chain nf: nf → extract → merge

Paths (edit CONFIG at top of file to change):
  A — manifest   $MANIFEST
      shards     $SHARDS
      master     $MASTER
  B — manifest   $MANIFEST_CCS
      shards     $SHARDS_CCS
      master     $MASTER_CCS
  nf outdir      $NF_OUTDIR
  nf params      $NF_PARAMS
EOF
        exit 1
        ;;
esac
