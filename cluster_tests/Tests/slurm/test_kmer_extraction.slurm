#!/bin/bash
#SBATCH --job-name=Test_Kmer
#SBATCH --array=1-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                 
#SBATCH --partition=pshort_el8     
#SBATCH --time=00:30:00            
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/test_ext_%A_%a.log

source ~/.bashrc
conda activate kinsim_env

CONFIG_FILE=$1

cd /data/users/ndutilleux/KinSim/cluster/scripts/

# Extraction des paramètres du config
BAM_PATH=$(sed -n "$(( ($SLURM_ARRAY_TASK_ID * 2) - 1 ))p" "$CONFIG_FILE")
MOTIFS=$(sed -n "$(( $SLURM_ARRAY_TASK_ID * 2 ))p" "$CONFIG_FILE")

# Lancer le script de test
python test_kmer_extraction.py "$BAM_PATH" "$MOTIFS"
