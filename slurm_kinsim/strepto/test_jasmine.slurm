#!/bin/bash
#SBATCH --job-name=strepto_jasmine
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=04:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Strepto jasmine+modkit 5mC test
#
# Usage:  sbatch -J strepto_jasmine_<bc> slurm_kinsim/strepto/test_jasmine.slurm <sample_id>
#
# Strepto raw BAMs already carry production jasmine 2.2.0 MM/ML tags
# (Revio GPU, default=5mC, chemistry-matched to the sample). We skip
# re-running jasmine entirely and go straight to:
#
#   1. pbmm2 align raw BAM → assembly (preserves MM/ML)
#   2. modkit pileup → bedMethyl
#   3. modkit find-motifs → motif TSV
#   4. emit motifs_jasmine.csv (combined-parser format)
#
# Output (${STREPTO}/test_jasmine/<sample_id>/):
#   <sample_id>_jas_aligned.bam (+ .bai)
#   <sample_id>_pileup.bed
#   <sample_id>_find_motifs.tsv
#   <sample_id>_motifs_jasmine.csv
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SAMPLE=${1:?"Usage: sbatch $0 <sample_id>   e.g. bc2071, bc2038, bc2046"}

STREPTO=/data/projects/p774_MARSD/NDutilleux/training/Strepto
SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
MANIFEST=${STREPTO}/manifest_strepto.csv

# Look up the raw BAM path from the manifest
RAW_BAM=$(python -c "
from kinsim.utils.config import load_manifest
for e in load_manifest('${MANIFEST}'):
    if e.sample_id == '${SAMPLE}':
        print(e.bam_path); break
")
[ -n "$RAW_BAM" ] || { echo "ERROR: sample '$SAMPLE' not in manifest $MANIFEST"; exit 1; }

REF="${STREPTO}/${SAMPLE}/final_assembly.fasta"
OUTDIR="${STREPTO}/test_jasmine/${SAMPLE}"
mkdir -p "$OUTDIR"

JAS_ALIGNED="${OUTDIR}/${SAMPLE}_jas_aligned.bam"
PILEUP="${OUTDIR}/${SAMPLE}_pileup.bed"
FIND_TSV="${OUTDIR}/${SAMPLE}_find_motifs.tsv"
MOTIFS_JAS="${OUTDIR}/${SAMPLE}_motifs_jasmine.csv"
IPDSUMMARY_CSV="${STREPTO}/${SAMPLE}/motifs.csv"   # pre-existing ipdSummary output for comparison

echo "========================================================"
echo "  Strepto jasmine+modkit test — $SAMPLE"
echo "========================================================"
echo "Date:           $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:           $(hostname)"
echo "Raw BAM:        $RAW_BAM"
echo "Reference:      $REF"
echo "Outdir:         $OUTDIR"
echo "ipdSummary ref: $IPDSUMMARY_CSV"
echo "========================================================"

[ -f "$RAW_BAM" ] || { echo "ERROR: raw BAM missing"; exit 1; }
[ -f "$REF" ]     || { echo "ERROR: reference missing"; exit 1; }
[ -f "${REF}.fai" ] || samtools faidx "$REF"

# Sanity check: raw BAM must carry jasmine 5mC tags (C+m in MM)
echo "=== verify raw BAM has 5mC (C+m) MM tag ==="
HAS_5MC=$(apptainer exec --bind /data "$SMRT_SIF" samtools view "$RAW_BAM" 2>/dev/null \
    | head -500 | grep -cE $'MM:Z:[^\t]*C\\+m' || true)
echo "  reads with C+m tag (sample of 500): $HAS_5MC"
if [ "$HAS_5MC" -eq 0 ]; then
    echo "  WARN: no C+m tag in first 500 reads — jasmine 5mC calls may be sparse"
    echo "        (or raw BAM was jasmine-called with --6ma --trt instead of default 5mC)"
fi

T0=$(date +%s)

# ---------- 1. pbmm2 align raw BAM (preserves MM/ML tags) ----------
if [ -s "$JAS_ALIGNED" ]; then
    echo "=== pbmm2 align SKIP — aligned BAM exists ==="
else
    echo "=== pbmm2 align (raw jasmine-tagged BAM → assembly) ==="
    START=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        pbmm2 align "$REF" "$RAW_BAM" "$JAS_ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "    $(( $(date +%s) - START ))s"
fi

[ -f "${JAS_ALIGNED}.bai" ] || samtools index "$JAS_ALIGNED"

# ---------- 2. modkit pileup ----------
if [ -s "$PILEUP" ]; then
    echo "=== modkit pileup SKIP — BED exists ($(wc -l < "$PILEUP") sites) ==="
else
    echo "=== modkit pileup ==="
    START=$(date +%s)
    modkit pileup "$JAS_ALIGNED" "$PILEUP" \
        --ref "$REF" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --filter-threshold 0.5 \
        --log-filepath "${OUTDIR}/${SAMPLE}_modkit.log"
    echo "    $(( $(date +%s) - START ))s  ($(wc -l < "$PILEUP") sites)"
fi

# ---------- 3. modkit find-motifs ----------
if [ -s "$FIND_TSV" ]; then
    echo "=== modkit find-motifs SKIP — TSV exists ==="
else
    echo "=== modkit find-motifs ==="
    START=$(date +%s)
    modkit find-motifs \
        --in-bedmethyl "$PILEUP" \
        --ref "$REF" \
        --out-table "$FIND_TSV" \
        --threads "${SLURM_CPUS_PER_TASK}" \
        || { echo "modkit find-motifs failed — writing empty TSV"; : > "$FIND_TSV"; }
    echo "    $(( $(date +%s) - START ))s"
fi

# ---------- 4. emit motifs.csv (combined-parser format) ----------
echo "=== emit ${SAMPLE}_motifs_jasmine.csv ==="
python3 - <<PY
import csv
from pathlib import Path

tsv = Path('${FIND_TSV}')
out = Path('${MOTIFS_JAS}')

def _cand(d, *names):
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
            # modkit find-motifs emits high_count / mid_count / low_count separately
            hi     = _cand(r, 'high_count')
            mid    = _cand(r, 'mid_count')
            lo     = _cand(r, 'low_count')
            nsite  = _cand(r, 'n_sites', 'num_sites', 'total_sites')
            if nsite is None:
                # sum high+mid+low if available
                parts = [int(x) for x in (hi, mid, lo) if x not in (None, '')]
                nsite = sum(parts) if parts else 0
            mod    = _cand(r, 'mod_code', 'mod_type') or 'm'
            if not motif or offset is None:
                continue
            mod_type = 'm5C' if mod in ('m', 'm5C', '5mC') else mod
            rows.append((mod_type, motif, int(offset),
                         float(frac or 0.0), int(float(nsite or 0)),
                         'jasmine+modkit'))

with out.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mod_type','motif','offset','frac_mod','n_sites','source'])
    for r in rows:
        w.writerow(r)

print(f'Wrote {out}  ({len(rows)} motifs)')
PY

# ---------- 5. Side-by-side comparison with ipdSummary motifs.csv ----------
echo ""
echo "========================================================"
echo "  COMPARISON — ipdSummary (pbmotifmaker) vs jasmine (modkit)"
echo "========================================================"
echo ""
echo "--- ipdSummary (SP3-C3) output, pre-existing ---"
if [ -s "$IPDSUMMARY_CSV" ]; then
    head -1 "$IPDSUMMARY_CSV"
    tail -n +2 "$IPDSUMMARY_CSV" | head -20
    echo "  (total $(tail -n +2 "$IPDSUMMARY_CSV" | wc -l) motifs)"
else
    echo "  (ipdSummary motifs.csv not found at $IPDSUMMARY_CSV)"
fi
echo ""
echo "--- jasmine+modkit output, just produced ---"
if [ -s "$MOTIFS_JAS" ]; then
    cat "$MOTIFS_JAS"
    echo "  (total $(tail -n +2 "$MOTIFS_JAS" | wc -l) motifs)"
else
    echo "  (empty output)"
fi
echo ""
echo "========================================================"
echo "  DONE — $SAMPLE, elapsed $(( $(date +%s) - T0 ))s"
echo "  Aligned BAM:   $JAS_ALIGNED"
echo "  Pileup BED:    $PILEUP"
echo "  Motifs CSV:    $MOTIFS_JAS"
echo "========================================================"
