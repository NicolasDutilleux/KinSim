#!/bin/bash
# ============================================================
# Build a GFF manifest CSV from ipdSummary outputs
#
# Reads the original training manifest and the ipdSummary output directory
# to produce a new manifest with the gff column populated.
#
# Usage:
#   bash strepto_01_build_gff_manifest.sh <original_manifest> <ipd_dir> <output_manifest>
#
# Example:
#   bash slurm_kinsim/strepto_01_build_gff_manifest.sh \
#       manifest_combined_train.csv \
#       gff_pipeline/ \
#       manifest_strepto_gff.csv
#
# The output manifest uses the ALIGNED BAMs (from bystrandify + pbmm2)
# as bam_path, and points gff to the ipdSummary GFF files.
# ============================================================

set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ] || [ -z "${3:-}" ]; then
    echo "Usage: bash strepto_01_build_gff_manifest.sh <original_manifest> <ipd_dir> <output_manifest>"
    echo ""
    echo "  original_manifest   Existing training manifest (sample_id, bam_path, motifs)"
    echo "  ipd_dir             Output directory from strepto_00_bystrandify_ipd.slurm"
    echo "  output_manifest     Path for the new GFF manifest CSV"
    exit 1
fi

ORIG_MANIFEST="$1"
IPD_DIR="$2"
OUT_MANIFEST="$3"

if [ ! -f "$ORIG_MANIFEST" ]; then
    echo "ERROR: Original manifest not found: $ORIG_MANIFEST"
    exit 1
fi
if [ ! -d "$IPD_DIR" ]; then
    echo "ERROR: ipdSummary output directory not found: $IPD_DIR"
    exit 1
fi

python -c "
import sys
from kinsim.utils.config import load_manifest
from pathlib import Path

orig = '${ORIG_MANIFEST}'
ipd_dir = Path('${IPD_DIR}')
out = '${OUT_MANIFEST}'

entries = load_manifest(orig)

with open(out, 'w') as f:
    f.write('sample_id,bam_path,motifs,gff\n')
    included = 0
    skipped = []
    for e in entries:
        species_dir = ipd_dir / e.sample_id
        aligned_bam = species_dir / f'{e.sample_id}_aligned.bam'
        gff_file    = species_dir / f'{e.sample_id}_ipdSummary.gff'

        if not gff_file.exists() or gff_file.stat().st_size == 0:
            skipped.append(e.sample_id)
            continue
        if not aligned_bam.exists():
            skipped.append(e.sample_id)
            continue

        # Quote motifs if they contain commas
        motifs = e.motifs
        if ',' in motifs:
            motifs = f'\"{motifs}\"'

        f.write(f'{e.sample_id},{aligned_bam},{motifs},{gff_file}\n')
        included += 1

print(f'GFF manifest written: {out}')
print(f'  Included: {included} species')
if skipped:
    print(f'  Skipped (no GFF): {len(skipped)}')
    for s in skipped:
        print(f'    - {s}')
"

echo ""
echo "Manifest preview:"
head -5 "$OUT_MANIFEST"
echo "..."
echo "Total data rows: $(tail -n +2 "$OUT_MANIFEST" | grep -v "^#" | grep -v "^$" | wc -l)"
