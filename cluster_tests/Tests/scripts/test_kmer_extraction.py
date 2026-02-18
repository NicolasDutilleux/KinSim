import pysam
import numpy as np
import sys
import os
import pickle
from collections import defaultdict

# Mapping binaire : 2 bits par base
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
METH_IDS = {'none': 0, 'm6A': 1, 'm4C': 2, 'm5C': 3}

def parse_motifs_optimized(motif_string):
    """Retourne {motif_seq: (type_id, mod_pos)}"""
    motifs = {}
    if not motif_string:
        return motifs
    for entry in motif_string.split(';'):
        if not entry or ',' not in entry: continue
        m_type, seq, pos = entry.split(',')
        motifs[seq] = (METH_IDS.get(m_type, 0), int(pos))
    return motifs

def process_bam_test(bam_path, motifs, k=11, max_reads=50):
    mid = k // 2
    lookup = defaultdict(lambda: np.zeros(5, dtype=np.float64))
    mask = (1 << (2 * k)) - 1
    
    print(f"DEBUG: Analyse de {max_reads} reads dans {bam_path}...")
    
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        count = 0
        for read in bam:
            if count >= max_reads: 
                break
            
            seq = read.query_sequence
            if not seq or len(seq) < k: continue
            if not (read.has_tag('fi') and read.has_tag('fp')): continue
            
            ipds = read.get_tag('fi')
            pws = read.get_tag('fp')
            
            # --- SÉCURITÉ ANTI-BUG (IndexError) ---
            min_len = min(len(seq), len(ipds), len(pws))
            if min_len < k: continue

            # 1. Marquage des motifs
            pos_status = np.zeros(min_len, dtype=np.int8)
            for motif_seq, (m_id, mod_pos) in motifs.items():
                start = 0
                while True:
                    idx = seq.find(motif_seq, start)
                    if idx == -1: break
                    target_pos = idx + mod_pos
                    if target_pos < min_len:
                        pos_status[target_pos] = m_id
                    start = idx + 1

            # 2. Sliding window
            current_bit_kmer = 0
            reads_added = False
            for i in range(min_len):
                val = BASE_MAP.get(seq[i], 0)
                current_bit_kmer = ((current_bit_kmer << 2) | val) & mask
                
                if i >= k - 1:
                    meth_type = pos_status[i - mid]
                    key = (current_bit_kmer, meth_type)
                    stats = lookup[key]
                    
                    # Accumulation des sommes pour : n, sum(x), sum(x^2), sum(y), sum(y^2)
                    stats[0] += 1
                    stats[1] += float(ipds[i - mid])
                    stats[2] += float(ipds[i - mid])**2
                    stats[3] += float(pws[i - mid])
                    stats[4] += float(pws[i - mid])**2
                    reads_added = True
            
            if reads_added:
                count += 1
                    
    print(f"DEBUG: Test terminé. {count} reads valides extraits.")
    return lookup

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_kmer_extraction.py <file.bam> <motif_string>")
        sys.exit(1)

    bam_file, motif_arg = sys.argv[1], sys.argv[2]
    out_name = "test_output_" + os.path.basename(bam_file).replace(".bam", ".pkl")
    
    motifs_dict = parse_motifs_optimized(motif_arg)
    result = process_bam_test(bam_file, motifs_dict)
    
    with open(out_name, 'wb') as f:
        pickle.dump(dict(result), f)
    print(f"DEBUG: Dictionnaire de test sauvegardé dans -> {out_name}")
