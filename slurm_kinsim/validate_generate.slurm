#!/bin/bash
#SBATCH --job-name=kinsim_validate_generate
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=p774
#SBATCH --partition=pgpu
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/validate_generate_%J.log

source ~/.bashrc
conda activate kinsim_env

kinsim generate \
    /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation/bc2036_stripped.bam \
    /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/final_assembly.fasta \
    /data/projects/p774_MARSD/NDutilleux/training/checkpoints_conv_v2/checkpoint_epoch20.pt \
    /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036/motifs.csv \
    /data/projects/p774_MARSD/NDutilleux/training/Strepto/bc2036_validation/bc2036_generated.bam \
    --batch-reads 100 \
    --deterministic
