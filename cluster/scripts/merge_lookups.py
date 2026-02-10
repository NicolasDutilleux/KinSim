import pickle
import glob
import os
import argparse
import numpy as np
from collections import defaultdict

def merge_binary_shards(input_dir, output_file):
    # Initialisation du dictionnaire maître
    master = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    
    # Construction du pattern de recherche
    pattern = os.path.join(input_dir, "*_binary.pkl")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Erreur : Aucun fichier '_binary.pkl' trouvé dans {input_dir}")
        return

    print(f"Fusion de {len(files)} fichiers trouvés dans {input_dir}...")

    for f_path in files:
        with open(f_path, 'rb') as f:
            shard = pickle.load(f)
            for key, data in shard.items():
                master[key] += data

    # Sauvegarde du résultat
    print(f"Sauvegarde du dictionnaire fusionné dans : {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(dict(master), f)
    print("Fusion terminée avec succès.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusionne les shards binaires .pkl de KinSim.")
    parser.add_argument("--input_dir", required=True, help="Dossier contenant les fichiers _binary.pkl")
    parser.add_argument("--output_file", required=True, help="Chemin du fichier de sortie (ex: master.pkl)")
    
    args = parser.parse_args()
    
    merge_binary_shards(args.input_dir, args.output_file)
