#!/bin/bash
# ============================================================
# KinSim v5 fracbin — Full pipeline with explicit dependencies
#
# Changes vs v4:
#   - Fraction-guided IPD binarization (no more Otsu)
#   - No min-sample filter (all keys kept)
#   - Binary coin-flip during generation (no fractional meth_probs)
#
# Usage: bash slurm_kinsim/pipeline_v5_fracbin.sh
# ============================================================

set -euo pipefail

BASEDIR=/data/projects/p774_MARSD/NDutilleux/training
MANIFEST=${BASEDIR}/manifest_combined_train.csv
SHARDS=${BASEDIR}/shards_fracbin_train
MASTER=${BASEDIR}/master_fracbin_train.pkl
CKPT_NEW=${BASEDIR}/checkpoints_fracbin
TESTPKL=${BASEDIR}/master_binary_test.pkl

VALDIR=${BASEDIR}/Strepto/bc2036_validation
REF=${BASEDIR}/Strepto/bc2036/final_assembly.fasta
MOTIFS_CSV=${BASEDIR}/Strepto/bc2036/motifs.csv
STRIPPED=${VALDIR}/bc2036_stripped.bam

# Output names — all contain v5_fracbin
GEN_BAM=${VALDIR}/bc2036_generated_v5_fracbin.bam
ALIGNED_TMP=${VALDIR}/bc2036_v5_fracbin_pbmm2_tmp.bam
ALIGNED=${VALDIR}/bc2036_v5_fracbin_aligned.bam
IPD_CSV=${VALDIR}/bc2036_v5_fracbin_ipdSummary.csv
IPD_GFF=${VALDIR}/bc2036_v5_fracbin_ipdSummary.gff
MOTIFS_OUT=${VALDIR}/bc2036_v5_fracbin_motifs.csv

LOGS=/data/projects/p774_MARSD/NDutilleux/logs

# ---- Count species ----
N=$(kinsim-prep manifest count $MANIFEST)
echo "Species in manifest: $N"

# ============================================================
# STEP 1: Extract (array job, auto-merge disabled)
# ============================================================
EXTRACT_JOB=$(KINSIM_NO_AUTOMERGE=1 sbatch --parsable --array=1-${N}%4 \
    slurm_kinsim/kinsim_extract.slurm \
    $MANIFEST $SHARDS $MASTER)
echo "1. Extract:      $EXTRACT_JOB (array 1-$N)"

# ============================================================
# STEP 2: Merge (explicit, depends on all extract tasks)
# ============================================================
MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:${EXTRACT_JOB} \
    slurm_kinsim/kinsim_extract.slurm \
    $MANIFEST $SHARDS $MASTER)
echo "2. Merge:        $MERGE_JOB (after extract)"

# ============================================================
# STEP 3: Train (depends on merge)
# ============================================================
TRAIN_JOB=$(sbatch --parsable \
    --dependency=afterok:${MERGE_JOB} \
    slurm_kinsim/kinsim_train.slurm \
    $MASTER $CKPT_NEW \
    --epochs 50 --test-pkl $TESTPKL)
echo "3. Train:        $TRAIN_JOB (after merge)"

# ============================================================
# STEP 4: Generate validation BAM (depends on train)
# ============================================================
GEN_JOB=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JOB} \
    --partition=pgpu --gres=gpu:1 --mem=16G --cpus-per-task=2 --time=02:00:00 \
    --job-name=kinsim_v5_generate \
    --output=${LOGS}/v5_fracbin_generate_%J.log \
    --account=p774 \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
kinsim generate \
    ${STRIPPED} ${REF} \
    ${CKPT_NEW}/checkpoint_epoch50.pt \
    ${MOTIFS_CSV} ${GEN_BAM} \
    --batch-reads 100")
echo "4. Generate:     $GEN_JOB (after train)"

# ============================================================
# STEP 5: pbmm2 align (depends on generate)
# ============================================================
ALIGN_JOB=$(sbatch --parsable \
    --dependency=afterok:${GEN_JOB} \
    --partition=pibu_el8 --mem=32G --cpus-per-task=8 --time=01:00:00 \
    --job-name=kinsim_v5_align \
    --output=${LOGS}/v5_fracbin_align_%J.log \
    --account=p774 \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
echo '=== pbmm2 align ===' && date && \
apptainer exec --bind /data /containers/apptainer/pbmm2-1.13.1.sif \
    pbmm2 align ${REF} ${GEN_BAM} ${ALIGNED_TMP} \
    --preset CCS --sort --num-threads 8 && \
date && echo '=== pbmm2 done ==='")
echo "5. Align:        $ALIGN_JOB (after generate)"

# ============================================================
# STEP 6: Add ip/pw tags + index (depends on align)
# ============================================================
IPPW_JOB=$(sbatch --parsable \
    --dependency=afterok:${ALIGN_JOB} \
    --partition=pibu_el8 --mem=16G --cpus-per-task=2 --time=01:00:00 \
    --job-name=kinsim_v5_ippw \
    --output=${LOGS}/v5_fracbin_ippw_%J.log \
    --account=p774 \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
echo '=== Add ip/pw tags from fi/fp/ri/rp ===' && date && \
python -c \"
import pysam, array, numpy as np
bam_in = pysam.AlignmentFile('${ALIGNED_TMP}')
bam_out = pysam.AlignmentFile('${ALIGNED}', 'wb', header=bam_in.header)
n = 0
for read in bam_in:
    if read.is_unmapped:
        bam_out.write(read)
        continue
    try:
        fi = np.array(read.get_tag('fi'), dtype=np.uint8)
        fp = np.array(read.get_tag('fp'), dtype=np.uint8)
        ri = np.array(read.get_tag('ri'), dtype=np.uint8)
        rp = np.array(read.get_tag('rp'), dtype=np.uint8)
    except KeyError:
        bam_out.write(read)
        continue
    if read.is_reverse:
        ip_vals = ri[::-1].copy()
        pw_vals = rp[::-1].copy()
    else:
        ip_vals = fi.copy()
        pw_vals = fp.copy()
    read.set_tag('ip', array.array('B', ip_vals.tolist()))
    read.set_tag('pw', array.array('B', pw_vals.tolist()))
    bam_out.write(read)
    n += 1
bam_in.close()
bam_out.close()
print(f'Added ip/pw tags to {n} reads')
\" && \
echo '=== Index aligned BAM ===' && \
apptainer exec --bind /data /containers/apptainer/samtools-1.19.sif \
    samtools index ${ALIGNED} && \
rm -f ${ALIGNED_TMP} ${ALIGNED_TMP}.bai && \
date && echo '=== Done ==='")
echo "6. ip/pw tags:   $IPPW_JOB (after align)"

# ============================================================
# STEP 7: ipdSummary (depends on ip/pw tags)
# ============================================================
IPD_JOB=$(sbatch --parsable \
    --dependency=afterok:${IPPW_JOB} \
    --partition=pibu_el8 --mem=32G --cpus-per-task=8 --time=02:00:00 \
    --job-name=kinsim_v5_ipdsummary \
    --output=${LOGS}/v5_fracbin_ipdsummary_%J.log \
    --account=p774 \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
echo '=== ipdSummary ===' && date && \
apptainer exec --bind /data /containers/apptainer/samtools-1.19.sif \
    samtools faidx ${REF} && \
apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.1.sif \
    pbindex ${ALIGNED} && \
apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.1.sif \
    ipdSummary ${ALIGNED} \
    --reference ${REF} \
    --identify m6A,m4C \
    --methylFraction \
    --csv ${IPD_CSV} \
    --gff ${IPD_GFF} \
    --numWorkers 8 && \
date && echo '=== Done ==='")
echo "7. ipdSummary:   $IPD_JOB (after ip/pw)"

# ============================================================
# STEP 8: pbmotifmaker (depends on ipdSummary)
# ============================================================
MOTIF_JOB=$(sbatch --parsable \
    --dependency=afterok:${IPD_JOB} \
    --partition=pibu_el8 --mem=16G --cpus-per-task=4 --time=04:00:00 \
    --job-name=kinsim_v5_pbmotifmaker \
    --output=${LOGS}/v5_fracbin_pbmotifmaker_%J.log \
    --account=p774 \
    --wrap="source ~/.bashrc && conda activate kinsim_env && \
echo '=== pbmotifmaker ===' && date && \
if [ ! -s '${IPD_GFF}' ]; then echo 'ERROR: GFF empty'; exit 1; fi && \
apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.1.sif \
    pbmotifmaker find '${REF}' '${IPD_GFF}' '${MOTIFS_OUT}' && \
echo '' && echo '=== Recovered motifs (v5 fracbin) ===' && \
cat '${MOTIFS_OUT}' && \
echo '' && echo '=== Real motifs ===' && \
cat '${MOTIFS_CSV}' && \
date && echo '=== Done ==='")
echo "8. pbmotifmaker: $MOTIF_JOB (after ipdSummary)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  v5 fracbin pipeline submitted"
echo "=========================================="
echo "  1. Extract:      $EXTRACT_JOB (array 1-$N)"
echo "  2. Merge:        $MERGE_JOB"
echo "  3. Train:        $TRAIN_JOB"
echo "  4. Generate:     $GEN_JOB"
echo "  5. Align:        $ALIGN_JOB"
echo "  6. ip/pw tags:   $IPPW_JOB"
echo "  7. ipdSummary:   $IPD_JOB"
echo "  8. pbmotifmaker: $MOTIF_JOB"
echo "=========================================="
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Compare:  diff <(cat ${VALDIR}/bc2036_generated_v4_motifs.csv) <(cat ${MOTIFS_OUT})"
echo ""
echo "Training metrics: cat ${CKPT_NEW}/logs/version_0/metrics.csv"
echo "v4 checkpoint:    ${BASEDIR}/checkpoints_binary/"
echo "v5 checkpoint:    ${CKPT_NEW}/"
