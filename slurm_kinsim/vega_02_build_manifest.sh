#!/bin/bash
# ============================================================
# Build manifest for Vega HMB data after processing
#
# Usage:
#   bash slurm_kinsim/vega_02_build_manifest.sh
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
PROCESSED=${VEGA}/processed
MANIFEST=${VEGA}/manifest_vega_gff.csv
SPECIES_FILE=${VEGA}/species_id.txt

echo "=== Building Vega GFF Manifest ==="

python -c "
import csv
from pathlib import Path

processed = Path('${PROCESSED}')
manifest = '${MANIFEST}'
species_file = '${SPECIES_FILE}'

# Load species names if available
species_map = {}
if Path(species_file).exists():
    with open(species_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            species_map[row['barcode']] = row.get('species', row['barcode'])

barcodes = [f'bc{i}' for i in range(2033, 2049)]

with open(manifest, 'w') as out:
    out.write('sample_id,bam_path,motifs,gff\n')
    included = 0
    skipped = []

    for bc in barcodes:
        sp_dir = processed / bc
        aligned = sp_dir / f'{bc}_aligned.bam'
        gff = sp_dir / f'{bc}_ipdSummary.gff'

        if not gff.exists() or gff.stat().st_size == 0:
            skipped.append(bc)
            continue
        if not aligned.exists():
            skipped.append(bc)
            continue

        # Use species name as sample_id if available
        sample_id = species_map.get(bc, bc)
        # Sanitize for CSV (no commas in sample_id)
        sample_id = sample_id.replace(',', '_').replace(' ', '_')

        # motifs column empty — GFF mode doesn't need it
        out.write(f'{sample_id},{aligned},,{gff}\n')
        included += 1

print(f'Manifest: {manifest}')
print(f'  Included: {included}')
if skipped:
    print(f'  Skipped: {len(skipped)} — {skipped}')
"

echo ""
echo "Preview:"
head -5 "$MANIFEST"
echo "..."
echo "Total: $(tail -n +2 "$MANIFEST" | wc -l) species"
