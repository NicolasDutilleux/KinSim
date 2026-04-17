#!/bin/bash
# ============================================================
# Master orchestration — Vega + Sequel extraction pipelines
#
# Prerequisites:
#   Vega:    assembly + bystrandify + align + ipdSummary complete (vega_01)
#   Sequel:  CCS complete, assemblies downloaded, ipdSummary on subreads done
#
# This script submits with dependency chains:
#
#   [Sequel]  align_hifi (array) -> build_manifest -> extract (array) -> merge
#   [Vega]                           build_manifest -> extract (array) -> merge
#   [Final]                                                       -> compare_all
#
# Usage:
#   bash slurm_kinsim/run_all_extraction.sh
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

REPO=/data/users/ndutilleux/KinSim
SLURMDIR=${REPO}/slurm_kinsim

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel

echo "============================================================"
echo "  KinSim Extraction Orchestration"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ============================================================
# SEQUEL pipeline
# ============================================================
echo ""
echo "=== [Sequel] Step 1: Align HiFi BAMs (48 barcodes, 5 parallel) ==="
SEQUEL_ALIGN_JOB=$(sbatch --parsable --array=1-48%5 "${SLURMDIR}/sequel_03_align_hifi.slurm")
echo "  Job ID: $SEQUEL_ALIGN_JOB"

echo ""
echo "=== [Sequel] Step 2: Build manifest (after align) ==="
SEQUEL_MANIFEST_JOB=$(sbatch --parsable --dependency=afterok:${SEQUEL_ALIGN_JOB} \
    --partition=pibu_el8 --account=p774 \
    --mem=2G --time=00:30:00 \
    --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_manifest_%J.log \
    --wrap="bash ${SLURMDIR}/sequel_04_build_manifest.sh")
echo "  Job ID: $SEQUEL_MANIFEST_JOB"

echo ""
echo "=== [Sequel] Step 3: Extract (array, after manifest) ==="
# Number of rows known only after manifest — submit a wrapper job
SEQUEL_EXTRACT_SUBMIT=$(sbatch --parsable --dependency=afterok:${SEQUEL_MANIFEST_JOB} \
    --partition=pibu_el8 --account=p774 \
    --mem=2G --time=00:05:00 \
    --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_extract_submit_%J.log \
    --wrap="
        set -euo pipefail
        source ~/.bashrc
        conda activate kinsim_env
        N=\$(kinsim-prep manifest count ${SEQUEL}/manifest_sequel_gff.csv)
        echo \"Submitting Sequel extract: N=\$N tasks\"
        sbatch --array=1-\${N}%5 ${SLURMDIR}/00_extract.slurm \
            ${SEQUEL}/manifest_sequel_gff.csv \
            ${SEQUEL}/shards \
            ${SEQUEL}/master_raw.pkl
    ")
echo "  Wrapper Job ID: $SEQUEL_EXTRACT_SUBMIT"

# ============================================================
# VEGA pipeline
# ============================================================
echo ""
echo "=== [Vega] Step 1: Build manifest ==="
VEGA_MANIFEST_JOB=$(sbatch --parsable \
    --partition=pibu_el8 --account=p774 \
    --mem=2G --time=00:30:00 \
    --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_manifest_%J.log \
    --wrap="bash ${SLURMDIR}/vega_02_build_manifest.sh")
echo "  Job ID: $VEGA_MANIFEST_JOB"

echo ""
echo "=== [Vega] Step 2: Extract (array, after manifest) ==="
VEGA_EXTRACT_SUBMIT=$(sbatch --parsable --dependency=afterok:${VEGA_MANIFEST_JOB} \
    --partition=pibu_el8 --account=p774 \
    --mem=2G --time=00:05:00 \
    --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_extract_submit_%J.log \
    --wrap="
        set -euo pipefail
        source ~/.bashrc
        conda activate kinsim_env
        N=\$(kinsim-prep manifest count ${VEGA}/manifest_vega_gff.csv)
        echo \"Submitting Vega extract: N=\$N tasks\"
        sbatch --array=1-\${N}%5 ${SLURMDIR}/00_extract.slurm \
            ${VEGA}/manifest_vega_gff.csv \
            ${VEGA}/shards \
            ${VEGA}/master_raw.pkl
    ")
echo "  Wrapper Job ID: $VEGA_EXTRACT_SUBMIT"

# ============================================================
# COMPARISON (after all extractions)
# ============================================================
echo ""
echo "=== [Final] Cross-dataset comparison ==="
echo "  NOTE: submit manually after both master_raw.pkl are ready."
echo "  Command: sbatch ${SLURMDIR}/compare_all_datasets.sh"

echo ""
echo "============================================================"
echo "  All jobs submitted"
echo "============================================================"
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -u \$USER --format=JobID,JobName,State,Elapsed | head -30"
