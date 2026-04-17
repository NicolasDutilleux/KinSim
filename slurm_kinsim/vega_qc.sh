#!/bin/bash
# ============================================================
# Vega HMB-16 Assembly Quality Check
#
# Usage (from login node):
#   srun --partition=pibu_el8 --account=p774 --mem=2G --time=00:10:00 \
#        --pty bash slurm_kinsim/vega_qc.sh
# ============================================================

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega

# David's HMB-16 expected sizes (Mb) — order matches bc2033..bc2048
declare -A EXPECTED_MB=(
    [2033]="?" [2034]="?" [2035]="?" [2036]="?" [2037]="?" [2038]="?" [2039]="?"
    [2040]="?" [2041]="?" [2042]="?" [2043]="?" [2044]="?" [2045]="?" [2046]="?"
    [2047]="?" [2048]="?"
)

echo "============================================================"
echo "  Vega Assembly Quality Check"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
printf "%-8s %-8s %-10s %-10s %-12s %-12s\n" \
    "Barcode" "Contigs" "Total_Mb" "Largest_Mb" "N50_kb" "GFF_recs"
echo "------------------------------------------------------------"

for bc in 2033 2034 2035 2036 2037 2038 2039 2040 2041 2042 2043 2044 2045 2046 2047 2048; do
    d=${VEGA}/assembly/bc${bc}
    asm=${d}/bc${bc}_assembly.fasta
    gff=${d}/bc${bc}_ipdSummary.gff

    if [ ! -s "$asm" ]; then
        printf "%-8s %-8s %-10s %-10s %-12s %-12s\n" \
            "bc${bc}" "---" "---" "---" "---" "---"
        continue
    fi

    n_contigs=$(grep -c '^>' "$asm")

    stats=$(awk '
        /^>/ {if(seq) lens[++n]=length(seq); seq=""; next}
        {seq=seq$0}
        END {
            if(seq) lens[++n]=length(seq)
            total=0
            for(i=1;i<=n;i++) total+=lens[i]
            # sort descending
            asort(lens, sorted, "@val_num_desc")
            cum=0; n50=0
            for(i=1;i<=n;i++){
                cum+=sorted[i]
                if(cum >= total/2 && n50==0) n50=sorted[i]
            }
            printf "%.2f %.2f %d", total/1e6, sorted[1]/1e6, n50/1000
        }' "$asm")

    total_mb=$(echo "$stats" | cut -d' ' -f1)
    largest_mb=$(echo "$stats" | cut -d' ' -f2)
    n50_kb=$(echo "$stats" | cut -d' ' -f3)

    if [ -s "$gff" ]; then
        gff_recs=$(grep -cv "^#" "$gff" 2>/dev/null || echo 0)
    else
        gff_recs="---"
    fi

    printf "%-8s %-8s %-10s %-10s %-12s %-12s\n" \
        "bc${bc}" "$n_contigs" "$total_mb" "$largest_mb" "$n50_kb" "$gff_recs"
done

echo "------------------------------------------------------------"
echo ""
echo "Quality interpretation:"
echo "  Contigs ≤ 5   + N50 ≥ 1000 kb = EXCELLENT (chromosome-level)"
echo "  Contigs ≤ 20  + N50 ≥ 500 kb  = GOOD (usable for methylation)"
echo "  Contigs > 50                   = FRAGMENTED (consider NCBI ref)"
