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

# Nextflow (alternative to the bash `process` + `manifest` steps)
NF_OUTDIR=${STREPTO}/prepare
NF_PARAMS=nextflow/params/strepto.yaml
NF_PROFILE=strepto,slurm
NF_MANIFEST_NAME=manifest_strepto_gff.csv
# ================================

mkdir -p "$LOGS"

count_rows() { kinsim-prep manifest count "$1"; }

# Apptainer `--bind /data` can only follow symlinks whose real path is
# also under /data. ORIG_MANIFEST bam_path entries are sometimes symlinks
# out of /data (e.g. to /home/...). Rewrite bam_path in-place with
# `readlink -f` so the container can actually see the file.
resolve_manifest_symlinks() {
    local m="$1"
    [ -s "$m" ] || { echo "ERROR: manifest missing ($m)" >&2; exit 1; }
    if ! grep -q '^[^,]*,/,' <(sed -n '2p' "$m") 2>/dev/null && \
       [ "$(awk -F, 'NR>1 && $2 ~ /^\// {c++} END{print c+0}' "$m")" -gt 0 ]; then
        cp -n "$m" "${m}.bak" 2>/dev/null || true
        awk -F, 'BEGIN{OFS=","} NR==1{print; next}
                 {cmd="readlink -f "$2; cmd|getline real; close(cmd);
                  if (real != "") $2=real; print}' \
            "${m}.bak" > "$m.tmp" && mv "$m.tmp" "$m"
        echo "Resolved bam_path symlinks in $m (backup: ${m}.bak)"
    fi
}

# ============ STEP SUBMITTERS ============

submit_process() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    [ -s "$ORIG_MANIFEST" ] || { echo "ERROR: ORIG_MANIFEST missing ($ORIG_MANIFEST)" >&2; exit 1; }
    resolve_manifest_symlinks "$ORIG_MANIFEST"
    local n; n=$(count_rows "$ORIG_MANIFEST")
    sbatch --parsable $d --array=1-${n}%${PROC_CONCURRENT} \
        slurm_kinsim/strepto_00_bystrandify_ipd.slurm \
        "$ORIG_MANIFEST" "$IPD_DIR"
}

# Nextflow launcher — runs the full PREPARE workflow (bystrandify → pbmm2 →
# index → ipdSummary) and writes the GFF manifest.  Replaces bash process+manifest.
submit_nf() {
    local dep=${1:-}; local d=""; [ -n "$dep" ] && d="--dependency=afterok:${dep}"
    sbatch --parsable $d \
        --partition=pibu_el8 --account=p774 --mem=8G --cpus-per-task=2 --time=24:00:00 \
        --job-name=strepto_nf \
        --output=${LOGS}/strepto_nf_%J.log \
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
    nf)       J=$(submit_nf);       echo "strepto.nf:       $J (Nextflow PREPARE → $NF_OUTDIR)" ;;
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
    all_nf)
        J1=$(submit_nf);                          echo "strepto.nf:        $J1"
        J2=$(submit_extract_merge_wrapper "$J1"); echo "strepto.ext+mrg:   $J2 (after $J1)"
        echo ""
        echo "monitor: squeue -u \$USER"
        echo "logs:    ls -lt ${LOGS}/strepto_*.log | head"
        ;;
    *)
        cat <<EOF
Usage: bash slurm_kinsim/run_strepto.sh <step>

Steps:
  process    bystrandify + align + ipdSummary    (bash, array from ORIG_MANIFEST)
  manifest   build GFF manifest from processed outputs  (bash)
  nf         Nextflow PREPARE — replaces process+manifest
  extract    kinsim extract per species          (array)
  merge      kinsim merge → master pkl
  all        chain bash: process → manifest → extract → merge
  all_nf     chain nf:   nf      → extract → merge

Paths (edit CONFIG at top of file to change):
  orig_manifest $ORIG_MANIFEST
  ipd_dir       $IPD_DIR
  manifest      $MANIFEST
  shards        $SHARDS
  master        $MASTER
  nf outdir     $NF_OUTDIR
  nf params     $NF_PARAMS
EOF
        exit 1
        ;;
esac
