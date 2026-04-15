#!/bin/bash
#SBATCH --job-name=vega_blast
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=02:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/vega_blast_%J.log

# BLAST missing Vega barcodes (bc2039/HMB-16) against full nt

set +u
source ~/.bashrc
conda activate kinsim_env
source /etc/profile.d/modules.sh 2>/dev/null || true
module load BLAST+/2.15.0-gompi-2021a
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
QUERY=/tmp/vega_missing_${SLURM_JOB_ID}.fasta
DB=/data/databases/ncbi-blastdbs/nt_2021_04_01/nt

echo "=== BLAST missing Vega species ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Extract queries for missing barcodes (bc2039 had no hit, bc2038 was low identity)
python -c "
import pysam, os
vega = '${VEGA}'
missing = [2038, 2039]
with open('${QUERY}', 'w') as out:
    for bc in missing:
        bam = pysam.AlignmentFile(os.path.join(vega, f'm21026_260313_002345.hifi_reads.bc{bc}.bam'), check_sq=False)
        read = next(iter(bam))
        out.write(f'>bc{bc}\n{read.query_sequence[:3000]}\n')
        bam.close()
        print(f'  bc{bc}: {read.query_length}bp')
"

echo ""
echo "Started: $(date '+%H:%M:%S')"

blastn -query "$QUERY" \
    -db "$DB" \
    -outfmt "6 qseqid stitle pident length evalue" \
    -max_target_seqs 3 \
    -num_threads 8 \
    -evalue 1e-10

echo ""
echo "Done: $(date '+%H:%M:%S')"

rm -f "$QUERY"
