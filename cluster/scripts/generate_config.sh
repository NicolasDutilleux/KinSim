#!/bin/bash

# 1. Définir les chemins de base
BASE_DIR="/data/projects/p774_MARSD/NDutilleux/training/Strepto"
BAM_DIR="$BASE_DIR/hifireads"
OUTPUT="/data/users/ndutilleux/KinSim/cluster/config_strains.txt"

# Vider le fichier s'il existe déjà
> $OUTPUT

# 2. Boucle sur les fichiers BAM en filtrant les souches exclues
ls $BAM_DIR/*.bam | grep -Ev "bc2035|bc2047|bc2050|bc2057|bc2060" | while read BAM_PATH; do
    
    # Extraire le numéro de barcode (ex: bc2033) du nom du fichier
    BC=$(echo $BAM_PATH | grep -oP "bc[0-9]{4}")
    MOTIF_FILE="$BASE_DIR/$BC/motifs.csv"
    
    # Vérifier si le fichier motifs.csv existe pour cette souche
    if [ -f "$MOTIF_FILE" ]; then
        # Extraire les colonnes 3 (type), 1 (sequence) et 2 (position) du CSV
        # On saute la première ligne (NR>1) et on formate avec des virgules et points-virgules
        MOTIF_STRING=$(awk -F',' 'NR>1 {printf "%s,%s,%s;", $3, $1, $2}' "$MOTIF_FILE" | sed 's/;$//')
        
        # Écrire dans le fichier config
        echo "$BAM_PATH" >> $OUTPUT
        echo "$MOTIF_STRING" >> $OUTPUT
        echo "✅ Ajouté $BC au config."
    else
        echo "⚠️  Attention : $MOTIF_FILE introuvable pour $BC."
    fi
done

echo "🚀 Fichier $OUTPUT généré avec succès !"
