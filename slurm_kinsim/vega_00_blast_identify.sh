#!/bin/bash
# ============================================================
# Identify Vega HMB species by BLASTing one read per barcode
#
# Usage:
#   srun --partition=pibu_el8 --account=p774 --mem=8G --time=00:30:00 \
#       bash slurm_kinsim/vega_00_blast_identify.sh
#
# Output:
#   /data/projects/p774_MARSD/NDutilleux/training/Vega/species_id.txt
#   /data/projects/p774_MARSD/NDutilleux/training/Vega/blast_results.txt
# ============================================================

set -euo pipefail
source ~/.bashrc
conda activate kinsim_env

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
BLASTDB=/data/databases/ncbi-blastdbs
QUERY=${VEGA}/blast_query.fasta
RESULTS=${VEGA}/blast_results.txt
SPECIES=${VEGA}/species_id.txt

echo "========================================================"
echo "  Vega HMB Species Identification via BLAST"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
echo ""

# Check BLAST db
echo "=== BLAST database ==="
ls "$BLASTDB"/ | head -20
echo ""

# Find the nt database
NTDB=""
for candidate in "${BLASTDB}/nt" "${BLASTDB}/nt/nt" "${BLASTDB}/core_nt/core_nt"; do
    if [ -f "${candidate}.nsq" ] || [ -f "${candidate}.00.nsq" ] || [ -f "${candidate}.nal" ]; then
        NTDB="$candidate"
        break
    fi
done

if [ -z "$NTDB" ]; then
    echo "Available databases:"
    find "$BLASTDB" -name "*.nal" -o -name "*.nsq" 2>/dev/null | head -20
    echo ""
    echo "ERROR: Could not find nt BLAST database. Check /data/databases/ncbi-blastdbs/"
    echo "You may need to adjust NTDB path in this script."
    exit 1
fi
echo "Using BLAST db: $NTDB"
echo ""

# Extract one read per barcode
echo "=== Extracting query reads ==="
python -c "
import pysam, os
vega = '${VEGA}'
with open('${QUERY}', 'w') as out:
    for bc in range(2033, 2049):
        bam_path = os.path.join(vega, f'm21026_260313_002345.hifi_reads.bc{bc}.bam')
        bam = pysam.AlignmentFile(bam_path, check_sq=False)
        read = next(iter(bam))
        # Use first 3000bp — enough for species ID, fast BLAST
        out.write(f'>bc{bc}\n')
        out.write(read.query_sequence[:3000] + '\n')
        bam.close()
        print(f'  bc{bc}: {read.query_length}bp read, using first 3000bp')
"
echo ""

# Run BLAST
echo "=== Running BLAST ==="
echo "Started: $(date '+%H:%M:%S')"

# Check if blastn is available
if command -v blastn &>/dev/null; then
    BLASTN=blastn
elif [ -f /usr/bin/blastn ]; then
    BLASTN=/usr/bin/blastn
else
    # Try via module
    module load BLAST+ 2>/dev/null || module load blast 2>/dev/null || true
    BLASTN=blastn
fi

$BLASTN -query "$QUERY" \
    -db "$NTDB" \
    -outfmt "6 qseqid sseqid stitle pident length evalue bitscore" \
    -max_target_seqs 1 \
    -num_threads 4 \
    -evalue 1e-10 \
    -out "$RESULTS"

echo "Done: $(date '+%H:%M:%S')"
echo ""

# Parse results into clean species table
echo "=== Species Identification ==="
echo ""
printf "%-8s  %-8s  %6s  %s\n" "BARCODE" "SAMPLE" "IDENT%" "SPECIES"
echo "--------------------------------------------------------------"

# Also write to file
echo "barcode,sample,pident,species,full_hit" > "$SPECIES"

python -c "
import csv

# Map barcode to sample name
bc_to_sm = {
    'bc2033': 'HMB-10', 'bc2034': 'HMB-11', 'bc2035': 'HMB-12', 'bc2036': 'HMB-13',
    'bc2037': 'HMB-14', 'bc2038': 'HMB-15', 'bc2039': 'HMB-16', 'bc2040': 'HMB-01',
    'bc2041': 'HMB-02', 'bc2042': 'HMB-03', 'bc2043': 'HMB-04', 'bc2044': 'HMB-05',
    'bc2045': 'HMB-06', 'bc2046': 'HMB-07', 'bc2047': 'HMB-08', 'bc2048': 'HMB-09',
}

with open('${RESULTS}') as f, open('${SPECIES}', 'a') as out:
    for line in f:
        parts = line.strip().split('\t')
        bc = parts[0]
        stitle = parts[2] if len(parts) > 2 else 'unknown'
        pident = parts[3] if len(parts) > 3 else '?'
        sm = bc_to_sm.get(bc, '?')
        # Extract genus species from title
        species = ' '.join(stitle.split()[:2]) if stitle else 'unknown'
        print(f'{bc:<8}  {sm:<8}  {pident:>6}  {species}  ({stitle[:60]})')
        out.write(f'{bc},{sm},{pident},{species},{stitle}\n')
"

echo ""
echo "Full results: $RESULTS"
echo "Species table: $SPECIES"
echo ""
echo "=== Done ==="
