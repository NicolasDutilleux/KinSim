import pysam
import sys
import os

def process_bam_list(list_file):
    # En-tête pour le log
    print("READ_NAME\tSEQUENCE\tFI_TAG\tRI_TAG\tFP_TAG\tRP_TAG")

    with open(list_file, 'r') as f:
        for i, line in enumerate(f):
            if i > 1: break
            bam_path = line.strip()
            if not bam_path or not os.path.exists(bam_path):
                continue

            # AJOUT DE check_sq=False ici pour les fichiers non-alignés
            with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
                for read in bam:
                    seq = read.query_sequence
                    if not seq: continue

                    fi = list(read.get_tag('fi')) if read.has_tag('fi') else "None"
                    ri = list(read.get_tag('ri')) if read.has_tag('ri') else "None"
                    fp = list(read.get_tag('fp')) if read.has_tag('fp') else "None"
                    rp = list(read.get_tag('rp')) if read.has_tag('rp') else "None"

                    print(f"{read.query_name}\t{seq}\t{fi}\t{ri}\t{fp}\t{rp}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_bam_list(sys.argv[1])
