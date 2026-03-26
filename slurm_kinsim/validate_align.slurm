#!/bin/bash
#SBATCH --job-name=kinsim_validate_align
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/validate_align_%J.log

source ~/.bashrc
conda activate kinsim_env

VALDIR=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation
REF=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/final_assembly.fasta
INPUT=${VALDIR}/bc2036_generated_v4.bam
ALIGNED_TMP=${VALDIR}/bc2036_generated_v4_pbmm2.bam
ALIGNED=${VALDIR}/bc2036_generated_v4_aligned.bam

echo "=== Step 1: pbmm2 align ==="
date

apptainer exec --bind /data \
    /containers/apptainer/pbmm2-1.13.1.sif \
    pbmm2 align \
        $REF \
        $INPUT \
        $ALIGNED_TMP \
        --preset CCS --sort --num-threads 8

RC=$?
echo "pbmm2 exit code: $RC"
if [ $RC -ne 0 ]; then
    echo "ERROR: pbmm2 failed"
    exit 1
fi
date

echo "=== Step 2: Add ip/pw tags from fi/fp/ri/rp ==="
echo "pbmm2 does not convert fi/fp→ip/pw for KinSim-generated BAMs."
echo "Post-processing: forward reads get ip=fi,pw=fp; reverse reads get ip=rev(ri),pw=rev(rp)."

python -c "
import pysam
import array
import numpy as np

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
"

RC=$?
echo "Post-process exit code: $RC"
if [ $RC -ne 0 ]; then
    echo "ERROR: post-processing failed"
    exit 1
fi

# Index the final BAM
apptainer exec --bind /data \
    /containers/apptainer/samtools-1.19.sif \
    samtools index ${ALIGNED}

# Clean up temp
rm -f ${ALIGNED_TMP} ${ALIGNED_TMP}.bai

date
echo "=== Done ==="
