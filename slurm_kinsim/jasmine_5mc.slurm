#!/bin/bash
#SBATCH --job-name=jasmine_5mc
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=08:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/jasmine_5mc_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# 5mC motif discovery pipeline — jasmine + pbmm2 + modkit
#
# Per-sample chain (array-indexed via manifest):
#   1. jasmine predict       unaligned HiFi BAM → MM/ML tagged BAM
#   2. pbmm2 align           align to reference
#   3. samtools index
#   4. modkit pileup         per-site bedMethyl
#   5. modkit find-motifs    motif enrichment from bedMethyl
#   6. emit motifs.csv       combined-parser format (kinsim-compatible)
#
#   sbatch --array=1-N%K slurm_kinsim/jasmine_5mc.slurm <manifest.csv> <out_dir>
#
# Manifest columns: sample_id, bam_path, motifs (motifs col ignored here).
# Reference is looked up at ${STREPTO}/<sample_id>/final_assembly.fasta by default;
# override via REF_TEMPLATE env (use %s for sample_id) or a REF= env var.
#
# Output per sample: <out_dir>/<sample_id>/
#   <sample_id>_jasmine.bam
#   <sample_id>_aligned.bam
#   <sample_id>_pileup.bed
#   <sample_id>_find_motifs.tsv
#   <sample_id>_motifs.csv   (combined format: mod_type,motif,offset,frac_mod,n_sites,source)
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

MANIFEST=${1:?"manifest CSV required"}
OUTBASE=${2:?"output dir required"}

SMRT_SIF=${SMRT_SIF:-/containers/apptainer/pacbio-smrt-tools-25.3.sif}
REF_TEMPLATE=${REF_TEMPLATE:-/data/projects/p774_MARSD/NDutilleux/training/Strepto/%s/final_assembly.fasta}

[ -s "$MANIFEST" ] || { echo "ERROR: manifest missing: $MANIFEST"; exit 1; }
mkdir -p "$OUTBASE"

ROW=$(python -c "
from kinsim.utils.config import load_manifest
e = load_manifest('${MANIFEST}')
i = ${SLURM_ARRAY_TASK_ID} - 1
if i >= len(e): raise SystemExit(0)
print(f'{e[i].sample_id}|{e[i].bam_path}')
")
[ -n "$ROW" ] || exit 0

SAMPLE_ID=$(echo "$ROW" | cut -d'|' -f1)
BAM_IN=$(echo "$ROW"    | cut -d'|' -f2)
REF=${REF:-$(printf "$REF_TEMPLATE" "$SAMPLE_ID")}

DIR="${OUTBASE}/${SAMPLE_ID}"
mkdir -p "$DIR"

JASMINE_BAM="${DIR}/${SAMPLE_ID}_jasmine.bam"
ALIGNED="${DIR}/${SAMPLE_ID}_aligned.bam"
PILEUP="${DIR}/${SAMPLE_ID}_pileup.bed"
MOTIFS_TSV="${DIR}/${SAMPLE_ID}_find_motifs.tsv"
MOTIFS_CSV="${DIR}/${SAMPLE_ID}_motifs.csv"

echo "========================================================"
echo "  jasmine 5mC — $SAMPLE_ID"
echo "========================================================"
echo "Input BAM:  $BAM_IN"
echo "Reference:  $REF"
echo "Output dir: $DIR"

[ -f "$BAM_IN" ] || { echo "ERROR: input BAM missing: $BAM_IN"; exit 1; }
[ -f "$REF" ]    || { echo "ERROR: reference missing: $REF"; exit 1; }

# ---- Step 1: jasmine predict ----
if [ -s "$JASMINE_BAM" ]; then
    echo "=== 1 SKIP — jasmine BAM exists ==="
else
    echo "=== 1 jasmine predict ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        jasmine "$BAM_IN" "$JASMINE_BAM" \
        --keep-kinetics
    [ -s "$JASMINE_BAM" ] || { echo "FAILED: jasmine produced no output"; exit 1; }
    echo "    $(( $(date +%s) - START ))s"
fi

# ---- Step 2: pbmm2 align ----
if [ -s "$ALIGNED" ]; then
    echo "=== 2 SKIP — aligned BAM exists ==="
else
    echo "=== 2 pbmm2 align ==="
    START=$(date +%s)
    [ -f "${REF}.fai" ] || samtools faidx "$REF"
    pbmm2 align "$REF" "$JASMINE_BAM" "$ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "    $(( $(date +%s) - START ))s"
fi

# ---- Step 3: index ----
[ -f "${ALIGNED}.bai" ] || samtools index "$ALIGNED"

# ---- Step 4: modkit pileup ----
if [ -s "$PILEUP" ]; then
    echo "=== 4 SKIP — pileup BED exists ==="
else
    echo "=== 4 modkit pileup ==="
    START=$(date +%s)
    modkit pileup "$ALIGNED" "$PILEUP" \
        --ref "$REF" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --filter-threshold 0.5 \
        --log-filepath "${DIR}/${SAMPLE_ID}_modkit.log"
    echo "    $(( $(date +%s) - START ))s  ($(wc -l < "$PILEUP") sites)"
fi

# ---- Step 5: modkit find-motifs ----
if [ -s "$MOTIFS_TSV" ]; then
    echo "=== 5 SKIP — find-motifs TSV exists ==="
else
    echo "=== 5 modkit find-motifs ==="
    START=$(date +%s)
    modkit find-motifs \
        --in-bedmethyl "$PILEUP" \
        --ref "$REF" \
        --out-table "$MOTIFS_TSV" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        || { echo "modkit find-motifs failed — writing empty TSV"; : > "$MOTIFS_TSV"; }
    echo "    $(( $(date +%s) - START ))s"
fi

# ---- Step 6: emit motifs.csv in combined-parser format ----
echo "=== 6 emit motifs.csv ==="
python - <<PY
# Convert modkit find-motifs TSV → combined-parser CSV (mod_type,motif,offset,frac_mod,n_sites,source)
import csv
from pathlib import Path

tsv = Path('${MOTIFS_TSV}')
out = Path('${MOTIFS_CSV}')

def _cand(d: dict, *names):
    for n in names:
        if n in d: return d[n]
    return None

rows = []
if tsv.exists() and tsv.stat().st_size > 0:
    with tsv.open() as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for r in rdr:
            motif  = _cand(r, 'motif', 'sequence')
            offset = _cand(r, 'mod_offset', 'offset', 'position')
            frac   = _cand(r, 'frac_mod', 'fraction', 'methylated_fraction')
            nsite  = _cand(r, 'n_sites', 'num_sites', 'total_sites')
            mod    = _cand(r, 'mod_code', 'mod_type') or 'm'
            if not motif or offset is None:
                continue
            mod_type = 'm5C' if mod in ('m', 'm5C', '5mC') else mod
            rows.append((mod_type, motif, int(offset), float(frac or 0.0), int(float(nsite or 0)), 'jasmine+modkit'))

with out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mod_type','motif','offset','frac_mod','n_sites','source'])
    for r in rows:
        w.writerow(r)

print(f'Wrote {out}  ({len(rows)} motifs)')
PY

echo "========================================================"
echo "  Done: $SAMPLE_ID"
echo "    jasmine BAM: $JASMINE_BAM"
echo "    aligned BAM: $ALIGNED"
echo "    pileup BED:  $PILEUP"
echo "    find-motifs: $MOTIFS_TSV"
echo "    motifs CSV:  $MOTIFS_CSV"
echo "========================================================"
