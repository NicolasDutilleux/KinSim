#!/bin/bash
# ============================================================
# Download NCBI RefSeq genomes for the 16 Vega HMB ATCC strains
#
# Usage (on cluster, compute node via srun):
#   srun --partition=pibu_el8 --account=p774 --mem=4G --time=01:00:00 \
#        --pty bash slurm_kinsim/vega_00_download_references.sh
#
# Requires: ncbi-datasets-cli in kinsim_env
#   conda install -n kinsim_env -c conda-forge -c bioconda ncbi-datasets-cli
# ============================================================

set -euo pipefail
source ~/.bashrc
conda activate kinsim_env

VEGA=/data/projects/p774_MARSD/NDutilleux/training/Vega
REF_DIR=${VEGA}/references
mkdir -p "$REF_DIR"
cd "$REF_DIR"

# ATCC ID → search query (strain name + ATCC)
# Order matches David's HMB-16 list. Barcode mapping TBD.
declare -A STRAINS=(
    ["BAA-1710"]="Acinetobacter baumannii AYE"
    ["33560"]="Campylobacter jejuni CIP 702"
    ["13124"]="Clostridium perfringens ATCC 13124"
    ["BAA-894"]="Cronobacter sakazakii BAA-894"
    ["13047"]="Enterobacter cloacae ATCC 13047"
    ["51559"]="Enterococcus faecium MMC4"
    ["700926"]="Escherichia coli MG1655"
    ["35401"]="Escherichia coli H10407"
    ["BAA-2146"]="Klebsiella pneumoniae BAA-2146"
    ["19115"]="Listeria monocytogenes Li2"
    ["47085"]="Pseudomonas aeruginosa PAO1"
    ["700720"]="Salmonella enterica LT2"
    ["700930"]="Shigella flexneri 2457T"
    ["25923"]="Staphylococcus aureus Seattle 1945"
    ["BAA-1116"]="Vibrio harveyi BB120"
    ["17802"]="Vibrio parahaemolyticus ATCC 17802"
)

echo "========================================================"
echo "  Downloading 16 HMB references from NCBI"
echo "  Output: $REF_DIR"
echo "========================================================"

for atcc in "${!STRAINS[@]}"; do
    query="${STRAINS[$atcc]}"
    out_name=$(echo "$query" | tr ' ' '_')
    out_file="${REF_DIR}/${out_name}.fna"

    if [ -s "$out_file" ]; then
        echo "SKIP: $query (exists)"
        continue
    fi

    echo ""
    echo "=== $atcc: $query ==="

    # Find accession via NCBI datasets search
    acc=$(datasets summary genome taxon "$query" --reference --limit 1 2>/dev/null \
          | jq -r '.reports[0].accession' 2>/dev/null || echo "")

    if [ -z "$acc" ] || [ "$acc" = "null" ]; then
        echo "  NO REFERENCE FOUND — try manual search on NCBI Assembly"
        continue
    fi

    echo "  Accession: $acc"

    datasets download genome accession "$acc" --include genome --filename "${atcc}.zip"
    unzip -o -q "${atcc}.zip" -d "${atcc}_tmp"
    mv "${atcc}_tmp"/ncbi_dataset/data/${acc}/*.fna "$out_file"
    rm -rf "${atcc}_tmp" "${atcc}.zip"

    echo "  Saved: $(basename "$out_file") ($(du -h "$out_file" | cut -f1))"
done

echo ""
echo "========================================================"
echo "  Done: $(ls "$REF_DIR"/*.fna 2>/dev/null | wc -l)/16 references"
echo "========================================================"
