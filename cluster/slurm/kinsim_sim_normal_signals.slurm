#!/bin/bash
#SBATCH --job-name=KinSim_Signals
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=pshort_el8
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sim_%j_%a.log
#SBATCH --array=1-10%5 # Ajuste le nombre de fichiers à simuler ici

# --- 1. Environnement ---
source ~/.bashrc
conda activate kinsim_env

# --- 2. Récupération des paramètres ---
# On utilise le pipe | comme séparateur dans notre fichier config
CONFIG_FILE="config_sim.txt"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $CONFIG_FILE)

FASTQ_IN=$(echo $LINE | cut -d'|' -f1)
PKL_DICO=$(echo $LINE | cut -d'|' -f2)
MOTIFS=$(echo $LINE | cut -d'|' -f3)

# Définition des noms de sortie basés sur le FASTQ
BASENAME=$(basename $FASTQ_IN .fq.gz)
SAM_OUT="${BASENAME}_signals.sam"
BAM_OUT="${BASENAME}_signals.bam"

# --- 3. Exécution de la simulation ---
echo "--- Début de la simulation pour $FASTQ_IN ---"
python kinsim_sim_signals.py "$FASTQ_IN" "$PKL_DICO" "$MOTIFS" "$SAM_OUT"

# --- 4. Conversion et Nettoyage ---
echo "--- Conversion SAM -> BAM ---"
# On utilise samtools pour compresser et trier le résultat
samtools view -Sb "$SAM_OUT" | samtools sort -o "$BAM_OUT"

# On indexe le BAM pour pouvoir le visualiser dans IGV ou l'utiliser avec des outils de détection
samtools index "$BAM_OUT"

# Suppression du SAM (très lourd et inutile une fois le BAM créé)
rm "$SAM_OUT"

echo "--- Simulation terminée : $BAM_OUT généré ---"
