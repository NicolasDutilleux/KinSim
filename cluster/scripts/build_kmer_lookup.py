import pysam
import numpy as np
import sys
import os
import pickle
import re  # Indispensable pour l'IUPAC
from collections import defaultdict

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
METH_IDS = {'none': 0, 'm6A': 1, 'm4C': 2, 'm5C': 3}

# Traduction IUPAC vers Regex
IUPAC_TO_REGEX = {
    'A':'A', 'C':'C', 'G':'G', 'T':'T', 'N':'.',
    'R':'[AG]', 'Y':'[CT]', 'S':'[GC]', 'W':'[AT]',
    'K':'[GT]', 'M':'[AC]', 'B':'[CGT]', 'D':'[AGT]',
    'H':'[ACT]', 'V':'[ACG]'
}

def iupac_to_re(motif):
    """Transforme GYCAGCYC en G[CT]CAG[CT]C"""
    return "".join(IUPAC_TO_REGEX.get(b, b) for b in motif)

def reverse_complement(seq):
    comp = {'A':'T', 'C':'G', 'G':'C', 'T':'A', 'N':'N', 
            'Y':'R', 'R':'Y', 'S':'S', 'W':'W', 'K':'M', 'M':'K',
            'B':'V', 'V':'B', 'D':'H', 'H':'D'}
    return "".join(comp.get(base, base) for base in reversed(seq))

def parse_motifs_v3(motif_string):
    motifs = []
    if not motif_string: return motifs
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry: continue
        m_type, seq, pos = entry.split(',')
        m_id = METH_IDS.get(m_type, 0)
        p = int(pos)
        
        # On compile le regex pour le forward ET le reverse
        # Le (?=(...)) permet de trouver les motifs qui se chevauchent
        for s, offset in [(seq, p), (reverse_complement(seq), len(seq)-1-p)]:
            regex_pattern = re.compile(f'(?=({iupac_to_re(s)}))')
            motifs.append({'pattern': regex_pattern, 'id': m_id, 'pos': offset})
    return motifs

def process_bam_binary(bam_path, motifs_list, k=11):
    mid = k // 2
    lookup = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    mask = (1 << (2 * k)) - 1

    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam:
            seq = read.query_sequence
            if not (seq and len(seq) >= k and read.has_tag('fi')): continue

            ipds, pws = read.get_tag('fi'), read.get_tag('fp')
            min_len = min(len(seq), len(ipds), len(pws))

            # --- LA PARTIE QUI CHANGE ---
            pos_status = np.zeros(min_len, dtype=np.int8)
            for m in motifs_list:
                # finditer avec lookahead (?=) trouve TOUTES les occurrences, même imbriquées
                for match in m['pattern'].finditer(seq):
                    idx = match.start()
                    target_pos = idx + m['pos']
                    if target_pos < min_len:
                        pos_status[target_pos] = m['id']

            # Fenêtre glissante bit-packing (Inchangé)
            current_bit_kmer = 0
            for i in range(min_len):
                base_val = BASE_MAP.get(seq[i], 0)
                current_bit_kmer = ((current_bit_kmer << 2) | base_val) & mask
                if i >= k - 1:
                    meth_type = pos_status[i - mid]
                    key = (current_bit_kmer, meth_type)
                    stats = lookup[key]
                    val_ipd, val_pw = float(ipds[i-mid]), float(pws[i-mid])
                    stats[0] += 1
                    stats[1] += val_ipd
                    stats[2] += val_ipd**2
                    stats[3] += val_pw
                    stats[4] += val_pw**2
    return lookup

if __name__ == "__main__":
    bam_file, motif_arg = sys.argv[1], sys.argv[2]
    out_name = os.path.splitext(bam_file)[0] + "_binary.pkl"
    final_lookup = process_bam_binary(bam_file, parse_motifs_v3(motif_arg))
    with open(out_name, 'wb') as f:
        pickle.dump(dict(final_lookup), f)
