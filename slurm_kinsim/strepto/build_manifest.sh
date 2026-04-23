#!/bin/bash
# ============================================================
# Strepto manifest builder — new modular pipeline output
#
# Produces manifest_strepto_merged.csv with per-sample merged motifs CSVs
# (ipdSummary ∪ jasmine filtered at the configured threshold).
# ============================================================

set -euo pipefail

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
ORIG_MANIFEST=${STREPTO}/manifest_strepto.csv
OUTBASE=${STREPTO}/pipeline
OUT_MANIFEST=${STREPTO}/manifest_strepto_merged.csv

[ -s "$ORIG_MANIFEST" ] || { echo "ERROR: input manifest missing: $ORIG_MANIFEST"; exit 1; }

python3 - <<PY
from pathlib import Path
from kinsim.utils.config import load_manifest

orig    = '${ORIG_MANIFEST}'
pipeline_dir = Path('${OUTBASE}')
out     = '${OUT_MANIFEST}'

entries = load_manifest(orig)
kept, skipped = 0, []
with open(out, 'w') as f:
    f.write('sample_id,bam_path,motifs,gff\n')
    for e in entries:
        d       = pipeline_dir / e.sample_id
        aligned = d / f'{e.sample_id}_aligned.bam'
        merged  = d / f'{e.sample_id}_motifs_merged.csv'
        gff     = d / f'{e.sample_id}_ipdSummary.gff'

        if not aligned.exists():
            skipped.append((e.sample_id, 'no aligned BAM')); continue
        if not merged.exists() or merged.stat().st_size == 0:
            skipped.append((e.sample_id, 'no merged motifs')); continue

        gff_s = str(gff) if gff.exists() and gff.stat().st_size > 0 else ''
        f.write(f'{e.sample_id},{aligned},{merged},{gff_s}\n')
        kept += 1

print(f'Wrote {out} ({kept} rows, {len(skipped)} skipped)')
for sid, reason in skipped:
    print(f'  skip {sid}: {reason}')
PY

kinsim-prep manifest validate "$OUT_MANIFEST" || true
