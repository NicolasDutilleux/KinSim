#!/bin/bash
#SBATCH --job-name=bc2036_sp3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=08:00:00
#SBATCH --output=/data/users/ndutilleux/kinsim_test/bc2036_sp3_%J.log

# Full bc2036 methylation calling pipeline using bystrandify + SP3-C3 null model
#
# Steps:
#   1. ccs-kinetics-bystrandify  — split HiFi CCS reads into 2 pseudo-subreads
#   2. pbmm2 align               — align pseudo-subreads to reference
#   3. pbindex                   — create .pbi index for ipdSummary
#   4. ipdSummary (SP3-C3 model) — call modified bases with explicit Sequel IIe model
#   5. pbmotifmaker              — cluster modified sites into motifs
#
# SP3-C3 is the closest available null model to Revio chemistry.
# Compare output to bc2036_bystrandify run (25.3 container default model).

module load SMRT-Link/12.0.0.177059-cli-tools-only

TESTDIR=/data/users/ndutilleux/kinsim_test
REF=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/final_assembly.fasta
INPUT=${TESTDIR}/bc2036.bam

BYSTRAND=${TESTDIR}/bc2036_sp3_bystrandify.bam
ALIGNED=${TESTDIR}/bc2036_sp3_aligned.bam
GFF=${TESTDIR}/bc2036_sp3.gff
CSV=${TESTDIR}/bc2036_sp3.csv
MOTIFS=${TESTDIR}/bc2036_sp3_motifs.csv

PARAMS=/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources

date && hostname
echo "========================================"
echo "=== bc2036 bystrandify + SP3-C3 pipeline ==="
echo "========================================"

echo ""
echo "=== Step 1: ccs-kinetics-bystrandify ==="
ccs-kinetics-bystrandify ${INPUT} ${BYSTRAND}
echo "Exit: $?"

echo ""
echo "=== Step 2: pbmm2 align ==="
pbmm2 align --sort --num-threads 8 --preset SUBREAD ${REF} ${BYSTRAND} ${ALIGNED}
echo "Exit: $?"

echo ""
echo "=== Step 3: pbindex ==="
apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.3.sif \
    pbindex ${ALIGNED}
echo "Exit: $?"

echo ""
echo "=== Step 4: ipdSummary (SP3-C3 model) ==="
samtools faidx ${REF}
ipdSummary ${ALIGNED} \
    --reference ${REF} \
    --identify m6A,m4C \
    --methylFraction \
    --csv ${CSV} \
    --gff ${GFF} \
    --numWorkers 8 \
    --paramsPath ${PARAMS}
echo "Exit: $?"

echo ""
echo "=== Step 5: pbmotifmaker ==="
apptainer exec --bind /data /containers/apptainer/pacbio-smrt-tools-25.3.sif \
    pbmotifmaker find ${REF} ${GFF} ${MOTIFS}
echo "Exit: $?"

echo ""
echo "=== Recovered motifs (SP3-C3) ==="
cat ${MOTIFS}

echo ""
echo "=== Expected motifs ==="
cat /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/motifs.csv 2>/dev/null \
    || echo "(no reference motifs.csv found)"

echo ""
echo "=== Comparison with previous bystrandify run (25.3 container) ==="
cat ${TESTDIR}/bc2036_bystrandify_motifs.csv 2>/dev/null \
    || echo "(previous run motifs not found)"

date
echo "=== Done ==="
