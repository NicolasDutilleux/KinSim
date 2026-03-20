#!/bin/bash
#SBATCH --job-name=kinsim_validate_align
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/validate_align_%J.log

apptainer exec --bind /data \
    /containers/apptainer/pbmm2-1.13.1.sif \
    pbmm2 align \
        /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/final_assembly.fasta \
        /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation/bc2036_generated_v2.bam \
        /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation/bc2036_generated_v2_aligned.bam \
        --preset CCS --sort --num-threads 8
