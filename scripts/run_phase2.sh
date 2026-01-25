#!/bin/bash
# Phase 2 Analysis Runner
# 
# This script runs Phase 2 semantic stability analysis.
# 
# Phase 2 computes embeddings (requires OpenAI API key) for:
# - All reviews (to create venue representations)
# - All unique tags (to create extraction representations)
# 
# After first run, embeddings are cached and you can use --skip-embeddings
# to avoid API calls.

set -e

# Default values
RUN_ID="week2_run_20251223_191104"
DATA_FILE="data/study1_venues_20250117.csv"
RESULTS_DIR="results/phase1_downloaded"
SKIP_EMBEDDINGS=false
BATCH_SIZE=64
LOG_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --data)
            DATA_FILE="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --skip-embeddings)
            SKIP_EMBEDDINGS=true
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --run-id ID              Run ID prefix (default: week2_run_20251223_191104)"
            echo "  --data PATH              Path to venues CSV (default: data/study1_venues_20250117.csv)"
            echo "  --results-dir PATH       Directory with Phase 1 results (default: results/phase1_downloaded)"
            echo "  --skip-embeddings        Skip embedding computation (use cached)"
            echo "  --batch-size N           Embedding batch size (default: 64)"
            echo "  --log FILE               Write output to log file (also shows on screen)"
            echo "  --help                   Show this help"
            echo ""
            echo "Examples:"
            echo "  # First run (computes embeddings, requires OPENAI_API_KEY)"
            echo "  $0"
            echo ""
            echo "  # Subsequent runs (uses cached embeddings)"
            echo "  $0 --skip-embeddings"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Check if data files exist
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found: $DATA_FILE"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Check for extraction files
EXTRACTION_FILES=$(find "$RESULTS_DIR" -name "${RUN_ID}_extractions_*.csv" 2>/dev/null | wc -l | tr -d ' ')
if [ "$EXTRACTION_FILES" -eq 0 ]; then
    echo "❌ Error: No extraction files found for run_id: $RUN_ID"
    echo "   Looked in: $RESULTS_DIR"
    exit 1
fi

echo "✅ Found $EXTRACTION_FILES extraction file(s)"

# Check API key if not skipping embeddings (Python script will load from .env if needed)
if [ "$SKIP_EMBEDDINGS" = false ]; then
    echo "ℹ️  Computing embeddings (requires OPENAI_API_KEY in .env or environment)"
else
    echo "ℹ️  Skipping embeddings (using cache)"
fi

# Build command (use -u flag for unbuffered output so logs show immediately)
CMD="poetry run python -u scripts/phase2_analysis.py"
CMD="$CMD --run-id $RUN_ID"
CMD="$CMD --data $DATA_FILE"
CMD="$CMD --results-dir $RESULTS_DIR"
CMD="$CMD --embed-batch-size $BATCH_SIZE"

if [ "$SKIP_EMBEDDINGS" = true ]; then
    CMD="$CMD --skip-embeddings"
fi

echo ""
echo "🚀 Running Phase 2 analysis..."
echo "   Run ID: $RUN_ID"
echo "   Data: $DATA_FILE"
echo "   Results: $RESULTS_DIR"
echo "   Skip embeddings: $SKIP_EMBEDDINGS"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the command with output shown in real-time
if [ -n "$LOG_FILE" ]; then
    echo "📝 Logging output to: $LOG_FILE"
    echo ""
    # Use tee to show output AND write to log
    eval $CMD 2>&1 | tee "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
else
    # Just show output normally
    eval $CMD
    EXIT_CODE=$?
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Phase 2 complete!"
    echo "   Tables: results/phase2/tables/"
    echo "   Cache: results/phase2_cache/"
    echo "   Plots: results/phase2/plots/ (run phase2_plots.py to generate)"
else
    echo "❌ Phase 2 failed with exit code $EXIT_CODE"
    echo "   Check the output above for errors"
    exit $EXIT_CODE
fi
