#!/bin/bash
# ============================================================
# KinSim — Full Vega HMB pipeline
#
# End-to-end: BLAST ID → hifiasm → bystrandify → ipdSummary → extract → merge
#
# Usage:
#   bash slurm_kinsim/run_vega_pipeline.sh
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
BASEDIR=/data/projects/p774_MARSD/NDutilleux/training
MANIFEST=${VEGA}/manifest_vega_gff.csv
SHARDS=${VEGA}/shards_gff
MASTER=${VEGA}/master_vega_gff.pkl
LOGS=/data/projects/p774_MARSD/NDutilleux/logs

echo "========================================================"
echo "  Vega HMB Full Pipeline"
echo "========================================================"
echo ""

# ============================================================
# STEP 1: Process all 16 species (assembly + bystrandify + ipdSummary)
# ============================================================
echo "=== STEP 1: Assembly + Bystrandify + ipdSummary (16 species) ==="

PROCESS_JOB=$(sbatch --parsable --array=1-16%4 \
    slurm_kinsim/vega_01_process_all.slurm)
echo "  Job: $PROCESS_JOB (array 1-16, max 4 concurrent)"
echo ""

# ============================================================
# STEP 2: Build manifest (after all processing done)
# ============================================================
echo "=== STEP 2: Build GFF manifest ==="

MANIFEST_JOB=$(sbatch --parsable \
    --dependency=afterok:${PROCESS_JOB} \
    --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
    --job-name=vega_manifest \
    --output=${LOGS}/vega_manifest_%J.log \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
bash slurm_kinsim/vega_02_build_manifest.sh")
echo "  Job: $MANIFEST_JOB (after processing)"
echo ""

# ============================================================
# STEP 3+4: Extract + Merge (after manifest exists)
# ============================================================
echo "=== STEP 3+4: Extract + Merge ==="

EXTRACT_JOB=$(sbatch --parsable \
    --dependency=afterok:${MANIFEST_JOB} \
    --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
    --job-name=vega_extract_submit \
    --output=${LOGS}/vega_extract_submit_%J.log \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
N_VEGA=\$(kinsim-prep manifest count '${MANIFEST}') && \
echo \"Vega manifest: \$N_VEGA species\" && \
EXT_JOB=\$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable --array=1-\${N_VEGA}%4 \
    slurm_kinsim/00_extract.slurm \
    '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"Extract: \$EXT_JOB\" && \
MRG_JOB=\$(sbatch --parsable \
    --dependency=afterany:\${EXT_JOB} \
    slurm_kinsim/00_extract.slurm \
    '${MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"Merge: \$MRG_JOB\"")
echo "  Job: $EXTRACT_JOB (submits extract+merge after manifest)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "  Vega HMB pipeline submitted"
echo "=========================================="
echo "  1. Process (16 species): $PROCESS_JOB"
echo "  2. Build manifest:       $MANIFEST_JOB"
echo "  3+4. Extract+Merge:      $EXTRACT_JOB"
echo "=========================================="
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ls -lt ${LOGS}/vega_*.log | head"
echo ""
echo "After merge, compare Vega vs Strepto:"
echo "  kinsim compare --label Strepto ${BASEDIR}/master_strepto_train.pkl \\"
echo "                 --label Vega   ${MASTER} \\"
echo "                 -o ${BASEDIR}/compare_vega_strepto/"
