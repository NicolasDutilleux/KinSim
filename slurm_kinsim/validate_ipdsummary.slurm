#!/bin/bash
#SBATCH --job-name=kinsim_validate_ipdsummary
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/validate_ipdsummary_%J.log

BAM=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation/bc2036_generated_v4_aligned.bam
REF=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/final_assembly.fasta
CSV=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation/bc2036_generated_v4_ipdSummary.csv
GFF=/data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation/bc2036_generated_v4_ipdSummary.gff

# Index the reference FASTA (.fai)
apptainer exec --bind /data \
    /containers/apptainer/samtools-1.19.sif \
    samtools faidx $REF

# Index the BAM (.pbi)
apptainer exec --bind /data \
    /containers/apptainer/pacbio-smrt-tools-25.1.sif \
    pbindex $BAM

# Run ipdSummary
apptainer exec --bind /data \
    /containers/apptainer/pacbio-smrt-tools-25.1.sif \
    ipdSummary $BAM \
        --reference $REF \
        --identify m6A,m4C \
        --methylFraction \
        --csv $CSV \
        --gff $GFF \
        --numWorkers 8
