#!/bin/bash
#SBATCH --job-name=v6_down
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=36:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/%x_%J.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# validate/v6_downstream — bystrandify + pbmm2 + ipdSummary + pbmotifmaker
#
# Usage:
#   sbatch -J v6_down_<strain> slurm/validate/v6_downstream.slurm \
#          <kinsim_bam> <ref_fasta> <out_dir>
#
# Produces (in <out_dir>):
#   <bc>_kinsim_bys.bam              (bystrandified, 2 records/ZMW with ip/pw)
#   <bc>_kinsim_aligned.bam          (pbmm2 aligned)
#   <bc>_kinsim_basemods.gff         (ipdSummary basemods)
#   <bc>_kinsim_basemods.csv         (ipdSummary csv)
#   <bc>_kinsim_motifs.csv           (pbmotifmaker output)
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

KINSIM_BAM=${1:?"Usage: sbatch $0 <kinsim_bam> <ref_fasta> <out_dir>"}
REF=${2:?"ref missing"}
OUT=${3:?"out_dir missing"}

mkdir -p "$OUT"
BC=$(basename "$KINSIM_BAM" | sed -E 's/_kinsim\.bam$//')

SMRT_SIF=/containers/apptainer/pacbio-smrt-tools-25.3.sif
FALQUET_SMRT=/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R
IPDSUMMARY_BIN="$FALQUET_SMRT/smrtcmds/bin/ipdSummary"
SP3_MODEL="$FALQUET_SMRT/install/smrtlink-release_25.1.0.257715/bundles/smrttools/install/smrttools-release_25.1.0.257715/private/pacbio/python3pkgs/kineticstools-py3/lib/python3.9/site-packages/kineticsTools/resources/SP3-C3.npz.gz"
FALQUET_PBMM=/data/users/lfalquet/SOFTS/SMRTlink/smrtlink251R/smrtcmds/bin/pbmotifmaker

[ -f "$KINSIM_BAM" ]       || { echo "ERROR: kinsim BAM missing: $KINSIM_BAM"; exit 1; }
[ -f "$REF" ]              || { echo "ERROR: ref missing: $REF"; exit 1; }
[ -f "$SMRT_SIF" ]         || { echo "ERROR: SMRT SIF missing"; exit 1; }
[ -x "$IPDSUMMARY_BIN" ]   || { echo "ERROR: Falquet ipdSummary missing: $IPDSUMMARY_BIN"; exit 1; }
[ -f "$SP3_MODEL" ]        || { echo "ERROR: SP3-C3 model missing: $SP3_MODEL"; exit 1; }
[ -x "$FALQUET_PBMM" ]     || { echo "ERROR: Falquet pbmotifmaker missing: $FALQUET_PBMM"; exit 1; }
[ -f "${REF}.fai" ] || samtools faidx "$REF"

BYS="$OUT/${BC}_kinsim_bys.bam"
ALIGNED="$OUT/${BC}_kinsim_aligned.bam"
BASEMODS_GFF="$OUT/${BC}_kinsim_basemods.gff"
BASEMODS_CSV="$OUT/${BC}_kinsim_basemods.csv"
MOTIFS_CSV="$OUT/${BC}_kinsim_motifs.csv"

echo "========================================================"
echo "  validate/v6_downstream — $BC"
echo "========================================================"
echo "Kinsim BAM: $KINSIM_BAM"
echo "Reference:  $REF"
echo "Output dir: $OUT"

# --- STEP 3: bystrandify ---
echo
echo "=== STEP 3: bystrandify → $(basename "$BYS") ==="
if [ -s "$BYS" ]; then
    echo "SKIP — bystrandified BAM exists"
else
    T=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$KINSIM_BAM" "$BYS"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$BYS" ] || { echo "FAILED: bystrandify empty"; exit 1; }
fi
[ -f "${BYS}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$BYS"

# --- STEP 4: pbmm2 align ---
echo
echo "=== STEP 4: pbmm2 align → $(basename "$ALIGNED") ==="
if [ -s "$ALIGNED" ]; then
    echo "SKIP — aligned BAM exists"
else
    RAW="${ALIGNED%.bam}.with_unmapped.bam"
    T=$(date +%s)
    apptainer exec --bind /data "$SMRT_SIF" \
        pbmm2 align "$REF" "$BYS" "$RAW" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$RAW" ] || { echo "FAILED: pbmm2 empty"; exit 1; }
    # filter unmapped: ipdSummary needs alignment
    samtools view -F 4 -b -@ "$SLURM_CPUS_PER_TASK" -o "$ALIGNED" "$RAW"
    rm -f "$RAW" "${RAW}.pbi" "${RAW}.bai"
fi
[ -f "${ALIGNED}.pbi" ] || apptainer exec --bind /data "$SMRT_SIF" pbindex "$ALIGNED"

# --- STEP 5: ipdSummary (Falquet SMRT-Link 25.1) ---
echo
echo "=== STEP 5: ipdSummary → $(basename "$BASEMODS_GFF") ==="
if [ -s "$BASEMODS_GFF" ]; then
    echo "SKIP — basemods GFF exists ($(grep -c '^[^#]' "$BASEMODS_GFF") records)"
else
    T=$(date +%s)
    "$IPDSUMMARY_BIN" "$ALIGNED" \
        --reference "$REF" \
        --identify m6A,m4C \
        --methylFraction \
        --pvalue 0.2 \
        --methylMinCov 5 \
        --identifyMinCov 3 \
        --minCoverage 2 \
        --mapQvThreshold 10 \
        --csv "$BASEMODS_CSV" \
        --gff "$BASEMODS_GFF" \
        --numWorkers "${SLURM_CPUS_PER_TASK}" \
        --ipdModel "$SP3_MODEL"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$BASEMODS_GFF" ] || { echo "FAILED: ipdSummary GFF empty"; exit 1; }
fi
echo "Basemods records: $(grep -c '^[^#]' "$BASEMODS_GFF" || true)"

# --- STEP 6: pbmotifmaker (Falquet 1.1.0) ---
echo
echo "=== STEP 6: pbmotifmaker → $(basename "$MOTIFS_CSV") ==="
if [ -s "$MOTIFS_CSV" ]; then
    echo "SKIP — motifs.csv exists ($(tail -n +2 "$MOTIFS_CSV" | wc -l) motifs)"
else
    T=$(date +%s)
    MOTIFMAKER_MIN_SCORE=${MOTIFMAKER_MIN_SCORE:-40}
    "$FALQUET_PBMM" find --min-score "$MOTIFMAKER_MIN_SCORE" \
        "$REF" "$BASEMODS_GFF" "$MOTIFS_CSV"
    echo "Elapsed: $(( $(date +%s) - T ))s"
    [ -s "$MOTIFS_CSV" ] || { echo "FAILED: pbmotifmaker empty"; exit 1; }
fi
echo "Motifs discovered: $(tail -n +2 "$MOTIFS_CSV" | wc -l)"
echo "--- head ---"
head -6 "$MOTIFS_CSV"

echo
echo "========================================================"
echo "  v6_downstream DONE for $BC"
echo "========================================================"
ls -la "$OUT"
