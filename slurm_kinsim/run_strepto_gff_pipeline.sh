#!/bin/bash
# ============================================================
# KinSim — Full Streptomyces GFF pipeline
#
# End-to-end: bystrandify + ipdSummary -> GFF extract -> merge -> train
#
# Steps:
#   1. Bystrandify + align + ipdSummary (array job, 1 per species)
#   2. Build GFF manifest from ipdSummary outputs
#   3. Extract with GFF mode (array job, 1 per species)
#   4. Merge shards into master .pkl
#   5. Train model (GPU)
#
# Usage:
#   bash slurm_kinsim/run_strepto_gff_pipeline.sh
#
# Edit the config section below before running.
# ============================================================

set -euo pipefail

# ============================================================
# CONFIG — Edit these paths
# ============================================================

BASEDIR=/data/projects/p774_MARSD/NDutilleux/training
STREPTO=${BASEDIR}/Strepto

# Input: existing Streptomyces sample manifest (raw HiFi BAMs per strain)
ORIG_MANIFEST=${STREPTO}/manifest_strepto.csv

# ipdSummary working directory (bystrandify + aligned BAMs + GFF outputs)
IPD_DIR=${STREPTO}/gff_pipeline

# GFF manifest (auto-generated after ipdSummary)
GFF_MANIFEST=${STREPTO}/manifest_strepto_gff.csv

# Extract shards + merged output
SHARDS=${STREPTO}/shards_gff
MASTER=${STREPTO}/master_strepto_gff.pkl

# Training checkpoints
CKPT=${STREPTO}/checkpoints_gff

# Validation (bc2036)
VALDIR=${BASEDIR}/Strepto/bc2036_validation
VAL_REF=${BASEDIR}/Strepto/bc2036/final_assembly.fasta
VAL_MOTIFS=${BASEDIR}/Strepto/bc2036/motifs.csv

# Test data (for --test-pkl during training)
TESTPKL=${BASEDIR}/master_binary_test.pkl

# Logs
LOGS=/data/projects/p774_MARSD/NDutilleux/logs

# Concurrency limits
IPD_CONCURRENT=4     # max simultaneous bystrandify+ipdSummary tasks
EXTRACT_CONCURRENT=4 # max simultaneous extract tasks

# ============================================================
# Count species
# ============================================================

source ~/.bashrc
conda activate kinsim_env

N=$(kinsim-prep manifest count "$ORIG_MANIFEST")
echo "Species in manifest: $N"
echo ""

# ============================================================
# STEP 1: Bystrandify + align + ipdSummary (array job)
# ============================================================
echo "=== STEP 1: Bystrandify + align + ipdSummary ==="

IPD_JOB=$(sbatch --parsable --array=1-${N}%${IPD_CONCURRENT} \
    slurm_kinsim/strepto_00_bystrandify_ipd.slurm \
    "$ORIG_MANIFEST" "$IPD_DIR")
echo "  Job: $IPD_JOB (array 1-$N, max $IPD_CONCURRENT concurrent)"
echo ""

# ============================================================
# STEP 2: Build GFF manifest (after all ipdSummary tasks finish)
# ============================================================
echo "=== STEP 2: Build GFF manifest ==="

BUILD_JOB=$(sbatch --parsable \
    --dependency=afterok:${IPD_JOB} \
    --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
    --job-name=strepto_manifest \
    --output=${LOGS}/strepto_build_manifest_%J.log \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
bash slurm_kinsim/strepto_01_build_gff_manifest.sh \
    '${ORIG_MANIFEST}' '${IPD_DIR}' '${GFF_MANIFEST}'")
echo "  Job: $BUILD_JOB (after ipdSummary)"
echo ""

# ============================================================
# STEP 3+4: Extract (GFF mode) + Merge
#   Submitted as a separate script that runs AFTER the GFF manifest exists.
#   This way we can count the actual GFF manifest rows.
# ============================================================
echo "=== STEP 3+4: Extract + Merge (submitted after manifest build) ==="

EXTRACT_MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --partition=pibu_el8 --account=p774 --mem=4G --cpus-per-task=1 --time=00:10:00 \
    --job-name=strepto_extract_submit \
    --output=${LOGS}/strepto_extract_submit_%J.log \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
N_GFF=\$(kinsim-prep manifest count '${GFF_MANIFEST}') && \
echo \"GFF manifest has \$N_GFF species\" && \
EXTRACT_JOB=\$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable --array=1-\${N_GFF}%${EXTRACT_CONCURRENT} \
    slurm_kinsim/00_extract.slurm \
    '${GFF_MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"Extract job: \$EXTRACT_JOB\" && \
MERGE_JOB=\$(sbatch --parsable \
    --dependency=afterany:\${EXTRACT_JOB} \
    slurm_kinsim/00_extract.slurm \
    '${GFF_MANIFEST}' '${SHARDS}' '${MASTER}') && \
echo \"Merge job: \$MERGE_JOB\" && \
TRAIN_JOB=\$(sbatch --parsable \
    --dependency=afterok:\${MERGE_JOB} \
    slurm_kinsim/01_train.slurm \
    '${MASTER}' '${CKPT}' \
    --epochs 50 --test-pkl '${TESTPKL}') && \
echo \"Train job: \$TRAIN_JOB\" && \
echo \"Pipeline submitted: extract=\$EXTRACT_JOB merge=\$MERGE_JOB train=\$TRAIN_JOB\"")
echo "  Job: $EXTRACT_MERGE_JOB (submits extract+merge+train after manifest is built)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=========================================="
echo "  Streptomyces GFF pipeline submitted"
echo "=========================================="
echo "  1. Bystrandify+ipdSummary: $IPD_JOB (array 1-$N)"
echo "  2. Build GFF manifest:     $BUILD_JOB"
echo "  3+4+5. Extract+Merge+Train: $EXTRACT_MERGE_JOB (auto-submits after manifest)"
echo "=========================================="
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls -lt ${LOGS}/strepto_*.log | head"
echo ""
echo "After training:"
echo "  kinsim evaluate ${CKPT} ${MASTER}"
echo ""
echo "Config:"
echo "  Original manifest: $ORIG_MANIFEST"
echo "  GFF manifest:      $GFF_MANIFEST"
echo "  ipdSummary dir:    $IPD_DIR"
echo "  Shards:            $SHARDS"
echo "  Master pkl:        $MASTER"
echo "  Checkpoints:       $CKPT"
