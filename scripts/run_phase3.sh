#!/bin/bash
# Run complete Phase 3 analysis
#
# This script runs all Phase 3 experiments:
# 1. Model-in-the-loop baseline (optional, ~$0.34)
# 2. Localization analysis (Block G)
# 3. Cost comparison (Block H)
# 4. Cold-start analysis (Block I)
#
# Usage:
#   ./scripts/run_phase3.sh              # Run all (including model-in-loop)
#   ./scripts/run_phase3.sh --skip-mil   # Skip model-in-loop baseline

set -e

SKIP_MIL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-mil)
            SKIP_MIL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Phase 3: Complete Analysis"
echo "=============================================="

# Check dependencies
if [ ! -f "results/phase2/tables/uncertainty_dispersion.csv" ]; then
    echo "Error: Phase 2 results not found."
    echo "Run phase2_analysis.py first."
    exit 1
fi

# Create output directory
mkdir -p results/phase3

# Step 1: Model-in-the-loop baseline (optional)
if [ "$SKIP_MIL" = false ]; then
    echo ""
    echo "Step 1/2: Running model-in-the-loop baseline..."
    echo "  (50 venues × 10 facets × 2 runs = 1000 queries)"
    echo "  Estimated cost: ~\$0.34"
    echo ""

    poetry run python scripts/phase3_model_in_loop.py \
        --data data/study1_venues_20250117.csv \
        --n-venues 50 \
        --n-runs 2 \
        --parallel 10 \
        --seed 42 \
        2>&1 | tee results/phase3/model_in_loop_run.log

    echo ""
    echo "Model-in-loop baseline complete!"
else
    echo ""
    echo "Step 1/2: Skipping model-in-the-loop baseline (--skip-mil)"
fi

# Step 2: Main Phase 3 analysis
echo ""
echo "Step 2/2: Running main Phase 3 analysis..."
echo "  - Block G: Localization (gentags vs embeddings)"
echo "  - Block H: Cost comparison"
echo "  - Block I: Cold-start analysis"
echo ""

poetry run python scripts/phase3_analysis.py \
    --run-id week2_run_20251223_191104 \
    --data data/study1_venues_20250117.csv \
    --results-dir results/phase1_downloaded \
    2>&1 | tee results/phase3/phase3_run.log

echo ""
echo "=============================================="
echo "✅ Phase 3 complete!"
echo "=============================================="
echo ""
echo "Results saved to: results/phase3/"
echo "  - localization.csv"
echo "  - facet_assignments.csv"
echo "  - cost_comparison.csv"
echo "  - cold_start.csv"
echo ""
echo "Next: Run phase3_plots.py to generate visualizations"
