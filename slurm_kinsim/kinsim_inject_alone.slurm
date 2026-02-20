#!/bin/bash
#SBATCH --job-name=KinSim_Inject
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_inject_%A.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

set -e

# ---- Environnement ----
source ~/.bashrc
conda activate kinsim_env

# ---- Assignation des arguments pour la clarté ----
FASTQ="$1"
MAF="$2"
REF="$3"
DICT="$4"
MOTIFS="$5"
OUTPUT="$6"

# ---- Vérification rapide des fichiers ----
for f in "$FASTQ" "$MAF" "$REF" "$DICT" "$MOTIFS"; do
    if [ ! -f "$f" ]; then
        echo "❌ ERREUR : Fichier introuvable -> $f"
        exit 1
    fi
done

echo "🚀 Démarrage de l'injection KinSim..."
echo "Mode : Per-genome (6 arguments)"
echo "---------------------------------------------------"

# ---- Commande KinSim (Logique à 6 arguments) ----
kinsim dictionary inject \
    "$FASTQ" \
    "$MAF" \
    "$REF" \
    "$DICT" \
    "$MOTIFS" \
    "$OUTPUT"

echo "---------------------------------------------------"
echo "✅ Injection terminée avec succès."
echo "Fichier produit : $OUTPUT"
