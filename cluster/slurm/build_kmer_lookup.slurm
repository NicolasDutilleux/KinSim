#!/bin/bash
#SBATCH --job-name=BinKmer_Ext
#SBATCH --array=1-53
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8       
#SBATCH --time=24:00:00   
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/bin_ext2_%A_%a.log

# 1. Environnement
source ~/.bashrc
conda activate kinsim_env

# 2. Gestion de la configuration
# Si tu ne donnes pas d'argument, il prendra config_strains.txt par défaut
CONFIG_FILE=${1:-config_strains.txt}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERREUR : Le fichier de configuration '$CONFIG_FILE' est introuvable."
    exit 1
fi

# 3. Extraction des paramètres
BAM_PATH=$(sed -n "$(( ($SLURM_ARRAY_TASK_ID * 2) - 1 ))p" "$CONFIG_FILE")
MOTIFS=$(sed -n "$(( $SLURM_ARRAY_TASK_ID * 2 ))p" "$CONFIG_FILE")

echo "--- Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID ---"
echo "Processing BAM: $BAM_PATH"

# 4. Exécution
python /data/users/ndutilleux/KinSim/cluster/scripts/build_kmer_lookup.py "$BAM_PATH" "$MOTIFS"
