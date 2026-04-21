#!/bin/bash
#SBATCH --job-name=sequel_ccs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=p774
#SBATCH --partition=pibu_el8
#SBATCH --time=12:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/sequel_ccs_%A_%a.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# Sequel II 48-plex — CCS → bystrandify → ipdSummary (SP3-C3)
#
# Parallel comparison baseline to sequel_02_ipdsummary.slurm:
#   02_ipdsummary.slurm: subreads → pbmm2 SUBREAD → ipdSummary (default model)
#   03_ccs_pipeline.slurm: subreads → ccs → bystrandify → pbmm2 CCS → ipdSummary SP3-C3
#                          (matches Vega/Strepto exactly — apples-to-apples)
#
# Pipeline per barcode:
#   1. Decompress subreads (.bam.gz → .bam)
#   2. ccs                        → HiFi reads with fi/fp kinetic tags
#   3. ccs-kinetics-bystrandify   → ip/pw tags (per-strand records)
#   4. pbmm2 align --preset CCS   → aligned BAM
#   5. samtools index + pbindex
#   6. ipdSummary --ipdModel SP3-C3 → GFF + CSV methylation calls
#   7. Cleanup decompressed .bam (keep .gz, keep HiFi, keep aligned)
#
# Output per barcode lands in a separate tree so it doesn't collide with 02:
#   ipdsummary_ccs/bcXXXX/bcXXXX_hifi.bam
#   ipdsummary_ccs/bcXXXX/bcXXXX_bystrandify.bam
#   ipdsummary_ccs/bcXXXX/bcXXXX_aligned.bam        (.bai + .pbi)
#   ipdsummary_ccs/bcXXXX/bcXXXX_ipdSummary.gff
#   ipdsummary_ccs/bcXXXX/bcXXXX_ipdSummary.csv
#
# Usage:
#   sbatch --array=1-48%4 slurm_kinsim/sequel_03_ccs_pipeline.slurm
# ============================================================

set +u
source ~/.bashrc
conda activate kinsim_env
set -euo pipefail

# Load SMRT-Link for ipdSummary (SP3-C3 model — host binary, not container)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load SMRT-Link/12.0.0.177059-cli-tools-only 2>/dev/null || true

# ---- Config ----
SEQUEL=/data/projects/p774_MARSD/NDutilleux/training/Sequel
ASM_DIR=${SEQUEL}/assemblies
OUTBASE=${SEQUEL}/ipdsummary_ccs
SMRT_SIF="/containers/apptainer/pacbio-smrt-tools-25.3.sif"

# SP3-C3 model — closest available null model to Sequel II P3/C3 chemistry
SP3_MODEL="/mnt/ss/sib/ibu/rocky8/2023072800/software/SMRT-Link/12.0.0.177059-cli-tools-only/install/smrtlink-release_12.0.0.177059/bundles/smrttools/install/smrttools-release_12.0.0.177059/private/thirdparty/python3/python3_3.9.6/site-packages/kineticsTools/resources/SP3-C3.npz.gz"

# ccs parameters — Q20 HiFi
CCS_MIN_RQ=0.99         # >= Q20
CCS_MIN_PASSES=3        # need at least 3 full passes for reliable CCS

mkdir -p "$OUTBASE"

# ---- Build barcode list from available .bam.gz files ----
mapfile -t GZ_BAMS < <(ls "${SEQUEL}"/lima.bc*--bc*.bam.gz 2>/dev/null | sort)
N_BAMS=${#GZ_BAMS[@]}

if [ "$N_BAMS" -eq 0 ]; then
    echo "ERROR: No lima.bc*--bc*.bam.gz files found in $SEQUEL"
    exit 1
fi

IDX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "$IDX" -ge "$N_BAMS" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID > $N_BAMS BAMs. Nothing to do."
    exit 0
fi

GZ_BAM="${GZ_BAMS[$IDX]}"
FNAME=$(basename "$GZ_BAM" .bam.gz)
BC=$(echo "$FNAME" | grep -oP 'bc\d+' | head -1)
SUBREADS_BAM="${SEQUEL}/${FNAME}.bam"

ASM_FASTA="${ASM_DIR}/${BC}_assembly.fsa"

SPECIES_DIR="${OUTBASE}/${BC}"
mkdir -p "$SPECIES_DIR"

HIFI="${SPECIES_DIR}/${BC}_hifi.bam"
BYSTRAND="${SPECIES_DIR}/${BC}_bystrandify.bam"
ALIGNED="${SPECIES_DIR}/${BC}_aligned.bam"
GFF="${SPECIES_DIR}/${BC}_ipdSummary.gff"
CSV="${SPECIES_DIR}/${BC}_ipdSummary.csv"

# ---- Diagnostics ----
START_TIME=$(date +%s)
echo "========================================================"
echo "  Sequel II CCS Pipeline — $BC (task $SLURM_ARRAY_TASK_ID)"
echo "========================================================"
echo "Date:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host:       $(hostname)"
echo "Subreads:   $GZ_BAM"
echo "Assembly:   $ASM_FASTA"
echo "Output:     $SPECIES_DIR/"
echo "ccs:        min-rq=$CCS_MIN_RQ  min-passes=$CCS_MIN_PASSES"
echo "========================================================"
echo ""

if [ ! -f "$ASM_FASTA" ]; then
    echo "ERROR: Assembly not found: $ASM_FASTA"
    echo "Run sequel_00b_download_meta.slurm first."
    exit 1
fi

# ---- Step 1: Decompress subreads ----
if [ -f "$SUBREADS_BAM" ]; then
    echo "=== Step 1: SKIP — subreads BAM already decompressed ==="
else
    echo "=== Step 1: Decompress subreads ==="
    echo "Started: $(date '+%H:%M:%S')"
    gunzip -k "$GZ_BAM"
    echo "  $(ls -lh "$SUBREADS_BAM" | awk '{print $5}')"
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 2: ccs — subreads → HiFi with fi/fp kinetic tags ----
if [ -f "$HIFI" ] && [ -s "$HIFI" ]; then
    echo "=== Step 2: SKIP — HiFi BAM already exists ==="
    echo "  $(samtools view -c "$HIFI" 2>/dev/null || echo '?') reads"
else
    echo "=== Step 2: ccs (HiFi + kinetics) ==="
    echo "Started: $(date '+%H:%M:%S')"

    # --hifi-kinetics produces per-base fi/fp (forward) and ri/rp (reverse)
    # derived by averaging per-pass ip/pw across all subread passes.
    apptainer exec --bind /data "$SMRT_SIF" \
        ccs "$SUBREADS_BAM" "$HIFI" \
            --min-rq "$CCS_MIN_RQ" \
            --min-passes "$CCS_MIN_PASSES" \
            --hifi-kinetics \
            -j "${SLURM_CPUS_PER_TASK}"

    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$HIFI" ]; then
        echo "FAILED: ccs exit=$RC"
        exit 1
    fi

    N_HIFI=$(samtools view -c "$HIFI" 2>/dev/null || echo 0)
    echo "  HiFi reads: $N_HIFI"
    if [ "$N_HIFI" -eq 0 ]; then
        echo "FAILED: ccs produced zero reads (not enough subread coverage?)"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 3: ccs-kinetics-bystrandify (fi/fp → ip/pw) ----
if [ -f "$BYSTRAND" ] && [ -s "$BYSTRAND" ]; then
    echo "=== Step 3: SKIP — bystrandify BAM already exists ==="
else
    echo "=== Step 3: ccs-kinetics-bystrandify ==="
    echo "Started: $(date '+%H:%M:%S')"

    apptainer exec --bind /data "$SMRT_SIF" \
        ccs-kinetics-bystrandify "$HIFI" "$BYSTRAND"

    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$BYSTRAND" ]; then
        echo "FAILED: bystrandify exit=$RC"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 4: pbmm2 align (bystrandified HiFi → assembly, CCS preset) ----
if [ -f "$ALIGNED" ] && [ -s "$ALIGNED" ]; then
    echo "=== Step 4: SKIP — aligned BAM already exists ==="
else
    echo "=== Step 4: pbmm2 align (CCS preset) ==="
    echo "Started: $(date '+%H:%M:%S')"

    if [ ! -f "${ASM_FASTA}.fai" ]; then
        samtools faidx "$ASM_FASTA"
    fi

    pbmm2 align "$ASM_FASTA" "$BYSTRAND" "$ALIGNED" \
        --preset CCS --sort --num-threads "${SLURM_CPUS_PER_TASK}"

    RC=$?
    if [ $RC -ne 0 ] || [ ! -f "$ALIGNED" ]; then
        echo "FAILED: pbmm2 align exit=$RC"
        exit 1
    fi
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 5: Index ----
echo "=== Step 5: Index BAM ==="

if [ ! -f "${ALIGNED}.bai" ]; then
    samtools index "$ALIGNED"
fi

if [ ! -f "${ALIGNED}.pbi" ]; then
    apptainer exec --bind /data "$SMRT_SIF" \
        pbindex "$ALIGNED"
fi

echo "Alignment stats:"
samtools flagstat "$ALIGNED" | head -5
echo ""

# ---- Step 6: ipdSummary (SP3-C3 model, host binary) ----
if [ -f "$GFF" ] && [ -s "$GFF" ]; then
    echo "=== Step 6: SKIP — GFF already exists ==="
    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  Existing GFF: $GFF_LINES records"
else
    echo "=== Step 6: ipdSummary (SP3-C3) ==="
    echo "Started: $(date '+%H:%M:%S')"

    ipdSummary "$ALIGNED" \
        --reference "$ASM_FASTA" \
        --identify m6A,m4C \
        --methylFraction \
        --csv "$CSV" \
        --gff "$GFF" \
        --numWorkers "${SLURM_CPUS_PER_TASK}" \
        --ipdModel "$SP3_MODEL"

    RC=$?
    if [ $RC -ne 0 ]; then
        echo "FAILED: ipdSummary exit=$RC"
        exit 1
    fi

    if [ ! -f "$GFF" ] || [ ! -s "$GFF" ]; then
        echo "WARNING: GFF file empty or missing"
        exit 1
    fi

    GFF_LINES=$(grep -c "^[^#]" "$GFF" || true)
    echo "  GFF records: $GFF_LINES"
    echo "Done: $(date '+%H:%M:%S')"
fi
echo ""

# ---- Step 7: Cleanup decompressed subreads (keep .gz, keep HiFi + aligned) ----
if [ -f "$GFF" ] && [ -s "$GFF" ] && [ -f "$GZ_BAM" ]; then
    echo "=== Step 7: Cleanup ==="
    rm -f "$SUBREADS_BAM"
    echo "Removed decompressed subreads (keeping .gz + HiFi + aligned)"
fi
echo ""

# ---- Summary ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "========================================================"
echo "  COMPLETE: $BC (CCS pipeline)"
echo "========================================================"
echo "  HiFi:        $HIFI"
echo "  Bystrandify: $BYSTRAND"
echo "  Aligned:     $ALIGNED"
echo "  GFF:         $GFF"
echo "  CSV:         $CSV"
echo ""
echo "Elapsed: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo "Ended:   $(date '+%Y-%m-%d %H:%M:%S')"
