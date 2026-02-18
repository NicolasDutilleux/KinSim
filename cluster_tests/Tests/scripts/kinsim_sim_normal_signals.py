import sys
import gzip
import pickle
import numpy as np
import re
from collections import defaultdict

# --- CONFIGURATION ---
METH_IDS = {'none': 0, 'm6A': 1, 'm4C': 2, 'm5C': 3}
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
IUPAC_TO_REGEX = {
    'A':'A', 'C':'C', 'G':'G', 'T':'T', 'N':'.', 'R':'[AG]', 'Y':'[CT]',
    'S':'[GC]', 'W':'[AT]', 'K':'[GT]', 'M':'[AC]', 'B':'[CGT]', 'D':'[AGT]',
    'H':'[ACT]', 'V':'[ACG]'
}

def iupac_to_re(motif):
    return "".join(IUPAC_TO_REGEX.get(b, b) for b in motif)

def get_stats(stats_vec):
    n = stats_vec[0]
    if n == 0: return 1.0, 0.1 # Baseline si k-mer inconnu
    mu = stats_vec[1] / n
    var = max(0, (stats_vec[2] / n) - mu**2)
    return mu, np.sqrt(var)

def sample_signal(mu, sigma):
    return max(0, np.random.normal(mu, sigma))

# --- LOGIQUE PRINCIPALE ---

def run_simulation(fastq_path, pkl_path, motif_str, output_sam):
    print(f"--- Chargement du dictionnaire: {pkl_path} ---")
    with open(pkl_path, 'rb') as f:
        lookup = pickle.load(f)

    # Calcul des moyennes globales pour le bruit (6 premières/dernières bases)
    all_non_meth = [v for k, v in lookup.items() if k[1] == 0]
    global_mu_ipd = np.mean([v[1]/v[0] for v in all_non_meth]) if all_non_meth else 1.0
    global_mu_pw = np.mean([v[3]/v[0] for v in all_non_meth]) if all_non_meth else 1.0

    # Préparation des motifs de méthylation
    motifs = []
    for entry in motif_str.split(';'):
        if not entry: continue
        m_type, seq, pos = entry.split(',')
        pattern = re.compile(f'(?=({iupac_to_re(seq)}))')
        motifs.append({'pattern': pattern, 'id': METH_IDS.get(m_type, 0), 'pos': int(pos)})

    print(f"--- Simulation des signaux vers: {output_sam} ---")
    
    open_func = gzip.open if fastq_path.endswith('.gz') else open
    with open_func(fastq_path, 'rt') as f_in, open(output_sam, 'w') as f_out:
        # Header SAM minimal
        f_out.write("@HD\tVN:1.6\tSO:unknown\n")
        
        while True:
            header = f_in.readline().strip()
            if not header: break
            seq = f_in.readline().strip()
            plus = f_in.readline().strip()
            qual = f_in.readline().strip()
            
            read_id = header[1:].split()[0]
            L = len(seq)
            ipd_signals = []
            pw_signals = []

            # 1. Identifier les méthylations dans le read
            meth_status = np.zeros(L, dtype=np.int8)
            for m in motifs:
                for match in m['pattern'].finditer(seq):
                    target = match.start() + m['pos']
                    if target < L: meth_status[target] = m['id']

            # 2. Générer les signaux base par base
            for i in range(L):
                # Cas des bords (6 premières et 6 dernières bases) : Bruit Aléatoire
                if i < 5 or i > L - 6:
                    ipd_signals.append(int(sample_signal(global_mu_ipd, 0.2) * 255))
                    pw_signals.append(int(sample_signal(global_mu_pw, 0.2) * 255))
                else:
                    # Extraction du k-mer binaire (taille 11)
                    kmer_seq = seq[i-5:i+6]
                    bit_kmer = 0
                    for base in kmer_seq:
                        bit_kmer = (bit_kmer << 2) | BASE_MAP.get(base, 0)
                    
                    m_type = meth_status[i]
                    key = (bit_kmer, m_type)
                    
                    # Tirage dans la distribution du dictionnaire
                    mu, sigma = get_stats(lookup.get(key, [0]*5))
                    ipd_signals.append(int(sample_signal(mu, sigma) * 255))
                    
                    mu_pw, sigma_pw = get_stats(lookup.get(key, [0]*5)[3:]) # Stats PW sont aux indices 3/4
                    pw_signals.append(int(sample_signal(mu_pw, sigma_pw) * 255))

            # 3. Écriture au format SAM avec tags fi et fp (entiers compressés)
            fi_tag = ",".join(map(str, ipd_signals))
            fp_tag = ",".join(map(str, pw_signals))
            # Format SAM : QNAME FLAG RNAME POS MAPQ CIGAR RNEXT PNEXT TLEN SEQ QUAL TAGS
            sam_line = f"{read_id}\t4\t*\t0\t0\t*\t*\t0\t0\t{seq}\t{qual}\tfi:B:C,{fi_tag}\tfp:B:C,{fp_tag}\n"
            f_out.write(sam_line)

if __name__ == "__main__":
    run_simulation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
