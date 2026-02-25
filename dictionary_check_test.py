import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import os
import argparse

# Import des constantes et fonctions de KinSim
from kinsim.encoding import get_ipd_stats, get_pw_stats, METH_IDS

def generate_variability_report(pkl_path, output_html):
    if not os.path.exists(pkl_path):
        print(f"Erreur : {pkl_path} introuvable.")
        return

    print(f"Chargement du dictionnaire : {pkl_path}")
    with open(pkl_path, 'rb') as f:
        lookup = pickle.load(f)

    id_to_name = {v: k for k, v in METH_IDS.items()}
    data = []

    print("Traitement des données de méthylation...")
    for (kmer, meth_id), acc in lookup.items():
        # On ignore les bases non-méthylées pour ce graphique spécifique
        if meth_id == 0:
            continue
            
        mu_ipd, _ = get_ipd_stats(acc)
        mu_pw, _ = get_pw_stats(acc)
        
        data.append({
            'Modification': id_to_name.get(meth_id, f"ID_{meth_id}"),
            'Mean_IPD': mu_ipd,
            'Mean_PW': mu_pw
        })

    df = pd.DataFrame(data)

    if df.empty:
        print("Aucune donnée de méthylation trouvée dans ce dictionnaire.")
        return

    # Création d'un graphique à deux volets (IPD et PW)
    # Le Violin plot montre la densité, le Box plot montre les quartiles
    fig_ipd = px.violin(df, x="Modification", y="Mean_IPD", color="Modification",
                        box=True, points="all", hover_data=df.columns,
                        title="Variabilité du signal IPD selon le contexte (K-mer)")

    fig_pw = px.violin(df, x="Modification", y="Mean_PW", color="Modification",
                       box=True, points="all", hover_data=df.columns,
                       title="Variabilité du signal PW selon le contexte (K-mer)")

    # Export vers un seul fichier HTML
    with open(output_html, 'w') as f:
        f.write(fig_ipd.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_pw.to_html(full_html=False, include_plotlyjs=False))

    print(f"Rapport de variabilité généré : {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse la variabilité contextuelle des méthylations.")
    parser.add_argument("dict_path", help="Chemin vers master_dict.plk")
    parser.add_argument("--output", default="meth_variability.html", help="Nom du fichier de sortie")
    
    args = parser.parse_args()
    generate_variability_report(args.dict_path, args.output)
