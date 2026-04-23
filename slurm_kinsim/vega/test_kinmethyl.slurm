#!/bin/bash
#SBATCH --job-name=vega_kinmethyl
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=06:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinMethyl 5mC test on a Vega sample — proof of concept
#
# Usage:
#   sbatch -J vega_kinmethyl_bc2046 slurm_kinsim/vega/test_kinmethyl.slurm bc2046 [motif] [mod_loc]
#
# Default: motif=CCWGG, mod_loc=1 (E. coli Dcm m5C canonical)
#
# Pipeline:
#   1. extract_features.py on bc*_ipd_aligned.bam at CCWGG motif positions
#   2. kinmethyl-test with pretrained P6-C4 5mC classifier (ensemble of 9)
#   3. aggregate + report mean 5mC freq at CCWGG vs baseline
#
# Chemistry mismatch note: KinMethyl ships models for P6-C4 (RSII) and
# SP4-C2 (Sequel II). Our data is P1-C1 (Vega). Expected degradation of
# AUC from 0.82 → ~0.7 but still likely >> ipdSummary (0.62).
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

SAMPLE=${1:?"Usage: sbatch $0 <barcode> [motif=CCWGG] [mod_loc=1]"}
MOTIF=${2:-CCWGG}
MOD_LOC=${3:-1}
case "$SAMPLE" in bc*) ;; *) SAMPLE="bc${SAMPLE}" ;; esac

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
KM=/data/users/ndutilleux/tools/KinMethyl
SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif

ALIGNED="${VEGA}/test_dual/${SAMPLE}/${SAMPLE}_ipd_aligned.bam"
REF="${VEGA}/assembly/${SAMPLE}/${SAMPLE}_assembly.fasta"
OUTDIR="${VEGA}/test_dual/${SAMPLE}/kinmethyl_${MOTIF}"
mkdir -p "$OUTDIR"

FEAT_TSV="${OUTDIR}/${SAMPLE}_${MOTIF}_features.tsv"
PRED_PREFIX="${OUTDIR}/${SAMPLE}_${MOTIF}_pred"

echo "========================================================"
echo "  KinMethyl test — $SAMPLE, motif=$MOTIF, mod_loc=$MOD_LOC"
echo "========================================================"
echo "Date:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "Aligned:  $ALIGNED"
echo "Ref:      $REF"
echo "Outdir:   $OUTDIR"
echo "KM root:  $KM"
echo "========================================================"

[ -f "$ALIGNED" ]      || { echo "ERROR: aligned BAM missing: $ALIGNED"; exit 1; }
[ -f "$REF" ]          || { echo "ERROR: reference missing: $REF"; exit 1; }
[ -f "${REF}.fai" ]    || samtools faidx "$REF"
[ -d "$KM" ]           || { echo "ERROR: KinMethyl repo missing: $KM"; exit 1; }

T0=$(date +%s)

# ---------- 1. extract_features ----------
if [ -s "$FEAT_TSV" ]; then
    echo "=== SKIP extract_features — TSV exists ==="
else
    echo "=== extract_features.py (motif=$MOTIF mod_loc=$MOD_LOC) ==="
    START=$(date +%s)
    # the script has broken `from utils.process_utils import ...` imports;
    # running it from INSIDE the kinmethyl/ subdir puts utils/ on sys.path
    cd "$KM/kinmethyl"
    python extract_features.py \
        --input "$ALIGNED" \
        --output "$FEAT_TSV" \
        --mode align \
        --ref "$REF" \
        --motifs "$MOTIF" \
        --mod_loc "$MOD_LOC" \
        --seq_len 21 \
        --norm zscore \
        --threads "${SLURM_CPUS_PER_TASK}" \
        --is_mapfea 0
    cd - >/dev/null
    echo "    $(( $(date +%s) - START ))s"
fi

[ -s "$FEAT_TSV" ] || { echo "FAILED: feature TSV empty"; exit 1; }
N_SITES=$(wc -l < "$FEAT_TSV")
echo "    feature TSV: $FEAT_TSV ($N_SITES rows)"
echo ""

# ---------- 2. kinmethyl-test with 9-model ensemble (P6-C4 5mC) ----------
echo "=== kinmethyl-test ensemble (9 P6-C4 5mC classifiers) ==="
for i in 1 2 3 4 5 6 7 8 9; do
    PRED="${PRED_PREFIX}_id${i}.tsv"
    if [ -s "$PRED" ]; then
        echo "  Fusion_ID${i}: SKIP (exists)"
        continue
    fi
    echo "  Fusion_ID${i}:"
    START=$(date +%s)
    kinmethyl-test \
        --data_file "$FEAT_TSV" \
        --model_file "$KM/models/classification_models/P6C4/5mC/Fusion_ID${i}.ckpt" \
        --model_type combined \
        --batch_size 256 \
        --output "$PRED" 2>&1 | tail -3
    echo "    $(( $(date +%s) - START ))s"
done
echo ""

# ---------- 3. aggregate + per-motif stats ----------
echo "=== aggregate ensemble + CCWGG vs baseline stats ==="
python3 - <<PY
import glob, statistics
from pathlib import Path

OUTDIR = Path("${OUTDIR}")
PREDS  = sorted(OUTDIR.glob("${SAMPLE}_${MOTIF}_pred_id*.tsv"))
FEAT   = Path("${FEAT_TSV}")
MOTIFS_CSV = OUTDIR / "${SAMPLE}_motifs_kinmethyl.csv"

print(f"Found {len(PREDS)} prediction files")
for p in PREDS:
    print(f"  {p.name} ({p.stat().st_size} bytes)")

if not PREDS:
    print("ERROR: no predictions; aborting")
    raise SystemExit(1)

# Each prediction row is typically: chrom\tpos\tstrand\tkmer\tipd\tpw\tlabel\tprob
# Parse each model's output, keep (chrom,pos,strand) -> [probs across models]
site_probs = {}
for p in PREDS:
    with p.open() as f:
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 2:
                continue
            # try to detect columns: last float is probability
            try:
                prob = float(parts[-1])
            except ValueError:
                continue
            # position key: if align mode, usually parts[0]=chrom parts[1]=pos parts[2]=strand
            key = tuple(parts[:3]) if len(parts) >= 8 else (parts[0],)
            site_probs.setdefault(key, []).append(prob)

print(f"Parsed {len(site_probs)} unique sites across models")

if not site_probs:
    print("ERROR: cannot parse prediction TSV format — please inspect manually")
    for p in PREDS[:1]:
        with p.open() as f:
            for i, line in enumerate(f):
                if i > 2: break
                print("  sample line:", line.rstrip()[:200])
    raise SystemExit(1)

# Ensemble: mean probability across models
mean_probs = {k: sum(v)/len(v) for k, v in site_probs.items()}

n         = len(mean_probs)
probs     = list(mean_probs.values())
mean_p    = statistics.mean(probs)
median_p  = statistics.median(probs)
frac50    = sum(p >= 0.50 for p in probs) / n
frac80    = sum(p >= 0.80 for p in probs) / n

print()
print("=== KinMethyl ensemble 5mC predictions at ${MOTIF} sites ===")
print(f"  n sites       = {n}")
print(f"  mean prob     = {mean_p:.3f}")
print(f"  median prob   = {median_p:.3f}")
print(f"  frac >=0.50   = {frac50:.3f}")
print(f"  frac >=0.80   = {frac80:.3f}")
print()
print("=== Comparison with REBASE ground truth ===")
print(f"  REBASE says M.EcoKDcm methylates ${MOTIF} at ~92% on E. coli K-12")
print(f"  ipdSummary SP3-C3 on this same sample: CCTGG m4C 44.6%")
if frac50 >= 0.70:
    verdict = "STRONG — KinMethyl detects ${MOTIF} 5mC well above ipdSummary baseline"
elif frac50 >= 0.40:
    verdict = "MODEST — detection comparable to ipdSummary, moderate chem-mismatch degradation"
elif frac50 >= 0.20:
    verdict = "WEAK — partial signal, maybe retrain regression model for P1-C1"
else:
    verdict = "NO SIGNAL — P6-C4 model does not transfer to P1-C1; retraining needed"
print(f"  VERDICT: {verdict}")

# Emit a motifs.csv in the combined-parser format (for KinSim consumption)
with MOTIFS_CSV.open("w") as out:
    out.write("mod_type,motif,offset,frac_mod,n_sites,source\n")
    out.write(f"m5C,${MOTIF},${MOD_LOC},{mean_p:.4f},{n},kinmethyl_P6C4\n")
print()
print(f"Wrote {MOTIFS_CSV}")
PY

echo ""
echo "========================================================"
echo "  DONE — elapsed $(( $(date +%s) - T0 ))s"
echo "  Feature TSV:  $FEAT_TSV"
echo "  Predictions:  ${PRED_PREFIX}_id*.tsv"
echo "  Motifs CSV:   ${OUTDIR}/${SAMPLE}_motifs_kinmethyl.csv"
echo "========================================================"
