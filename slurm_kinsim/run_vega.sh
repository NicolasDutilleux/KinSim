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

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

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

# Nextflow (alternative to the bash `process` + `manifest` steps).
# When using `nf`, the pipeline writes its own manifest at
#   ${NF_OUTDIR}/${NF_MANIFEST_NAME}
# which is what MANIFEST above should point to if you want `extract`/`merge`
# to consume it directly. Tweak these + nextflow/params/vega.yaml together.
NF_OUTDIR=${VEGA}/prepare
NF_PARAMS=nextflow/params/vega.yaml
NF_PROFILE=vega,slurm
NF_MANIFEST_NAME=manifest_vega_gff.csv
# ================================

mkdir -p "$LOGS"

# ============ STEP SUBMITTERS ============
# Each submitter optionally accepts a dependency jobid as $1.
# They echo only the submitted jobid (so callers can capture it).

submit_process() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d --array=1-${N_SPECIES}%${PROC_CONCURRENT} \
        slurm_kinsim/vega_01_assembly_pipeline.slurm
}

# Nextflow launcher — runs the full PREPARE workflow (decompress → bystrandify
# → hifiasm → pbmm2 → index → ipdSummary) and writes the GFF manifest to
# $NF_OUTDIR/$NF_MANIFEST_NAME. Replaces bash `process` + `manifest` steps.
submit_nf() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=8G --cpus-per-task=2 --time=48:00:00 \
        --job-name=vega_nf \
        --output=${LOGS}/vega_nf_%J.log \
        --wrap="set +u && source ~/.bashrc && conda activate kinsim_env && set -euo pipefail && \
cd \"\$(git rev-parse --show-toplevel 2>/dev/null || pwd)\" && \
nextflow run nextflow/main.nf -profile ${NF_PROFILE} \
    -params-file ${NF_PARAMS} \
    --outdir '${NF_OUTDIR}' \
    --manifest_name '${NF_MANIFEST_NAME}' \
    -resume && \
ln -sf '${NF_OUTDIR}/${NF_MANIFEST_NAME}' '${MANIFEST}'"
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
    nf)       J=$(submit_nf);       echo "vega.nf:       $J (Nextflow PREPARE → $NF_OUTDIR)" ;;
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
    all_nf)
        J1=$(submit_nf);                         echo "vega.nf:        $J1 (Nextflow PREPARE)"
        J2=$(submit_extract_merge_wrapper "$J1");echo "vega.ext+mrg:   $J2 (submits extract+merge after $J1)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/vega_*.log | head"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/run_vega.sh <step>

Steps:
  process    assemble + bystrandify + align + ipdSummary  (bash, array 1-${N_SPECIES})
  manifest   build GFF manifest from processed outputs    (bash)
  nf         Nextflow PREPARE — replaces process+manifest (writes $NF_OUTDIR/$NF_MANIFEST_NAME
             and symlinks it to \$MANIFEST)
  extract    kinsim extract per species                   (array)
  merge      kinsim merge → master pkl
  all        chain bash: process → manifest → extract → merge
  all_nf     chain nf:   nf      → extract → merge

Paths (edit CONFIG at top of file to change):
  manifest       $MANIFEST
  shards         $SHARDS
  master         $MASTER
  nf outdir      $NF_OUTDIR
  nf params      $NF_PARAMS
EOF
        exit 1
        ;;
esac
