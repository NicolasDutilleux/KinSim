#!/bin/bash
#SBATCH --job-name=KinSim_Train
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --account=p774
#SBATCH --partition=pshort_el8
#SBATCH --time=00:30:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/prepate_verif_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# 1. Environment

source ~/.bashrc
conda activate kinsim_env 

# 2. Command line

kinsim prepare cluster_tests/Tests/prepare_verification.test cluster_tests/Tests/prepare_verified.test
