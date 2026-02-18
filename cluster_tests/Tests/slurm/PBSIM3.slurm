#!/bin/bash
#SBATCH --job-name=PBSIM3_Sim
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=pshort_el8
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/pbsim_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# --- 1. Environnement ---
source ~/.bashrc
conda activate kinsim_env

# --- 2. Gestion des Arguments ---
# On récupère le génome via la ligne de commande ($1)
GENOME_IN=$1

if [ -z "$GENOME_IN" ]; then
    echo "Erreur : Aucun fichier génome fourni."
    echo "Usage : sbatch simulate_pbsim3.slurm <chemin_vers_genome.fna>"
    exit 1
fi

# Définition du préfixe de sortie basé sur le nom du fichier d'entrée
PREFIX=$(basename "$GENOME_IN" .fna)
MODEL="/home/ndutilleux/.conda/envs/kinsim_env/data/ERRHMM-SEQUEL.model"

# --- 3. Exécution de la Simulation ---
echo "Lancement de pbsim3 sur : $GENOME_IN"
echo "Modèle utilisé : $MODEL"

pbsim --strategy wgs \
      --method errhmm \
      --errhmm "$MODEL" \
      --depth 50 \
      --genome "$GENOME_IN" \
      --prefix "${PREFIX}_sim"

echo "Simulation terminée pour $PREFIX"
