#!/bin/bash
#SBATCH --job-name=KinSim_Analyze
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=p774
#SBATCH --partition=pshort_el8
#SBATCH --time=01:00:00
#SBATCH --output=/data/projects/p774_MARSD/NDutilleux/logs/kinsim_analyze_%j.log
#SBATCH --mail-user=nicolas.dutilleux@unifr.ch
#SBATCH --mail-type=FAIL

# ============================================================
# KinSim Dictionary — Comprehensive analysis report (single job)
#
# Usage:
#   sbatch kinsim_analyze.slurm <dict.pkl> [output_dir]
#
# Arguments:
#   $1  Path to trained dictionary (.pkl file)
#   $2  (optional) Output directory for reports
#       Default: same directory as the .pkl file
#
# Output files:
#   <basename>_report.txt   — coverage, signals, neighbor sensitivity (also in SLURM log)
#   <basename>_report.html  — interactive Plotly visualizations
#
# Examples:
#   sbatch kinsim_analyze.slurm /data/kinsim/dicts/master_dict.pkl
#   sbatch kinsim_analyze.slurm /data/kinsim/dicts/master_dict.pkl /data/kinsim/reports/
#
# Performance notes:
#   - Neighbor sensitivity (500K cap): ~10-15 min, ~8 GB RAM
#   - HTML generation adds ~1-2 min (requires plotly: pip install -e .[analyze])
#   - Use --max-neighbor-entries 0 to skip neighbor analysis for faster runs
# ============================================================

# ---- Usage check ----
if [ -z "$1" ]; then
    echo "Usage: sbatch kinsim_analyze.slurm <dict.pkl> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  dict.pkl    Path to trained dictionary (.pkl file)"
    echo "  output_dir  (optional) Directory for output reports"
    echo ""
    echo "Output files:"
    echo "  <basename>_report.txt   TXT report (also in SLURM log)"
    echo "  <basename>_report.html  Interactive HTML report (requires plotly)"
    exit 1
fi

# ---- Environment ----
source ~/.bashrc
conda activate kinsim_env

# ---- Config ----
PKL_PATH="$1"
OUTPUT_DIR="${2:-}"

if [ ! -f "$PKL_PATH" ]; then
    echo "ERROR: Dictionary file '$PKL_PATH' not found."
    exit 1
fi

echo "Analyzing: $PKL_PATH"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output dir: $OUTPUT_DIR"
fi

# ---- Run ----
if [ -n "$OUTPUT_DIR" ]; then
    kinsim dictionary analyze "$PKL_PATH" -o "$OUTPUT_DIR"
else
    kinsim dictionary analyze "$PKL_PATH"
fi
