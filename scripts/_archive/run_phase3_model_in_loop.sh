#!/bin/bash
# Run Phase 3 Model-in-the-Loop Baseline
#
# This script runs the model-in-the-loop baseline experiment.
# Configuration: 50 venues × 10 facets × 2 runs = 1000 LLM calls
#
# Estimated cost: ~$0.10-0.20 (using gpt-5-nano)
#
# Usage:
#   ./scripts/run_phase3_model_in_loop.sh
#
# Or with custom parameters:
#   poetry run python scripts/phase3_model_in_loop.py --n-venues 10 --n-runs 1

set -e

echo "=============================================="
echo "Phase 3: Model-in-the-Loop Baseline"
echo "=============================================="

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        echo "Loading API key from .env..."
        export $(grep -v '^#' .env | xargs)
    else
        echo "Error: OPENAI_API_KEY not set and .env not found"
        exit 1
    fi
fi

# Create output directory
mkdir -p results/phase3

# Run the experiment
poetry run python scripts/phase3_model_in_loop.py \
    --data data/study1_venues_20250117.csv \
    --n-venues 50 \
    --n-runs 2 \
    --seed 42 \
    2>&1 | tee results/phase3/model_in_loop_run.log

echo ""
echo "Done! Results saved to results/phase3/"
