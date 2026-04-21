#!/bin/bash
# ============================================================
# Build manifest for Sequel II 48-plex CCS-pipeline extraction
#
# Companion to sequel_04_build_manifest.sh — that one targets the
# subread pipeline (ipdsummary/), this one targets the CCS pipeline
# (ipdsummary_ccs/) so you get two independent manifests and can
# compare master_sequel_gff.pkl vs master_sequel_ccs_gff.pkl.
#
#   ipdsummary_ccs/bcXXXX/bcXXXX_aligned.bam
#   ipdsummary_ccs/bcXXXX/bcXXXX_ipdSummary.gff
#
# Output: manifest_sequel_ccs_gff.csv  (sample_id, bam_path, motifs, gff)
#
# Usage:
#   bash slurm_kinsim/sequel_04_build_manifest_ccs.sh
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
IPD_DIR=${SEQUEL}/ipdsummary_ccs
MANIFEST=${SEQUEL}/manifest_sequel_ccs_gff.csv
SAMPLE_LIST=${SEQUEL}/48-plex_sample_list.tsv

echo "=== Building Sequel 48-plex CCS GFF Manifest ==="

python - <<PYEOF
import csv
from pathlib import Path

ipd_dir = Path("${IPD_DIR}")
manifest = "${MANIFEST}"
sample_list = Path("${SAMPLE_LIST}")

# Optional species-name mapping from PacBio metadata
bc_to_species = {}
if sample_list.exists():
    with open(sample_list) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            bc = row.get("Barcode") or row.get("barcode") or row.get("Bc")
            sp = row.get("Species") or row.get("species") or row.get("Sample")
            if bc and sp:
                bc_key = "bc" + bc.lstrip("bc").lstrip("0")
                bc_to_species[bc_key] = sp.replace(" ", "_").replace(",", "_")

if not ipd_dir.exists():
    raise SystemExit(f"ERROR: CCS pipeline dir not found: {ipd_dir}")

barcodes = sorted(
    d.name for d in ipd_dir.iterdir()
    if d.is_dir() and d.name.startswith("bc")
)

with open(manifest, "w") as out:
    out.write("sample_id,bam_path,motifs,gff\n")
    included = 0
    skipped = []

    for bc in barcodes:
        aligned = ipd_dir / bc / f"{bc}_aligned.bam"
        gff     = ipd_dir / bc / f"{bc}_ipdSummary.gff"

        if not aligned.exists() or aligned.stat().st_size == 0:
            skipped.append((bc, "no aligned BAM"))
            continue
        if not gff.exists() or gff.stat().st_size == 0:
            skipped.append((bc, "no GFF"))
            continue

        sample_id = bc_to_species.get(bc, bc)
        if sample_id != bc:
            sample_id = f"{sample_id}_{bc}_ccs"
        else:
            sample_id = f"{bc}_ccs"

        out.write(f"{sample_id},{aligned},,{gff}\n")
        included += 1

print(f"Manifest: {manifest}")
print(f"  Included: {included}")
if skipped:
    print(f"  Skipped:  {len(skipped)}")
    for bc, reason in skipped:
        print(f"    - {bc}: {reason}")
PYEOF

echo ""
echo "Preview:"
head -5 "$MANIFEST"
echo "..."
echo "Total: $(tail -n +2 "$MANIFEST" | wc -l) species"
