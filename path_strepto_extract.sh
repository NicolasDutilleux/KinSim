#!/bin/bash

# Chemins de base
BASE_DIR="/data/projects/p774_MARSD/NDutilleux/training/Strepto"
BAM_DIR="$BASE_DIR/hifireads"
OUTPUT_FILE="/data/users/ndutilleux/KinSim/kinsim_input_strepto.txt"

# Liste des barcodes à exclure
EXCLUDE="bc2035 bc2047 bc2050 bc2057 bc2060"

# Vider le fichier de sortie
> "$OUTPUT_FILE"

echo "Traitement en cours..."

for bam_path in "$BAM_DIR"/*.bam; do
    # Extraire le barcode (ex: bc2033)
    bc_id=$(basename "$bam_path" | grep -o "bc[0-9]\+")
    
    # Vérifier si le barcode est dans la liste d'exclusion
    if [[ $EXCLUDE =~ $bc_id ]]; then
        # On passe au suivant sans rien écrire
        continue
    fi
    
    # Chemin vers le csv
    csv_path="$BASE_DIR/$bc_id/motifs.csv"
    
    # Écriture si le CSV existe
    if [[ -f "$csv_path" ]]; then
        echo "$bam_path" >> "$OUTPUT_FILE"
        echo "$csv_path" >> "$OUTPUT_FILE"
    fi
done

echo "Fichier $OUTPUT_FILE généré (barcodes exclus : $EXCLUDE)"
