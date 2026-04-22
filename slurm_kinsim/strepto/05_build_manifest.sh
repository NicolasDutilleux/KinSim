#!/bin/bash
# ============================================================
# Strepto step 05 — build GFF manifest (sample_id, bam_path, motifs, gff)
#
# Walks ${STREPTO}/manifest_strepto.csv rows and keeps any whose aligned
# BAM + ipdSummary GFF exist. motifs column points to per-species
# motifs.csv from step 04 if present, else empty.
#
# Output: ${STREPTO}/manifest_strepto_gff.csv
# ============================================================

set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
ORIG_MANIFEST=${STREPTO}/manifest_strepto.csv
OUTBASE=${STREPTO}/gff_pipeline
OUT_MANIFEST=${STREPTO}/manifest_strepto_gff.csv

[ -s "$ORIG_MANIFEST" ] || { echo "ERROR: original manifest missing: $ORIG_MANIFEST"; exit 1; }

python - <<PY
from pathlib import Path
from kinsim.utils.config import load_manifest

orig    = '${ORIG_MANIFEST}'
ipd_dir = Path('${OUTBASE}')
out     = '${OUT_MANIFEST}'

entries = load_manifest(orig)
kept, skipped = 0, []
with open(out, 'w') as f:
    f.write('sample_id,bam_path,motifs,gff\n')
    for e in entries:
        d = ipd_dir / e.sample_id
        aligned = d / f'{e.sample_id}_aligned.bam'
        gff     = d / f'{e.sample_id}_ipdSummary.gff'
        motifs  = d / f'{e.sample_id}_motifs.csv'

        if not aligned.exists() or not gff.exists() or gff.stat().st_size == 0:
            skipped.append(e.sample_id); continue

        motifs_s = str(motifs) if motifs.exists() and motifs.stat().st_size > 0 else ''
        f.write(f'{e.sample_id},{aligned},{motifs_s},{gff}\n')
        kept += 1

print(f'Wrote {out} ({kept} rows, {len(skipped)} skipped)')
for s in skipped:
    print(f'  skip: {s}')
PY

kinsim-prep manifest validate "$OUT_MANIFEST" || true
