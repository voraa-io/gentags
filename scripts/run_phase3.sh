#!/bin/bash
# Run Phase 3 Analysis (v2 - State-Gini)
#
# This script runs the corrected Phase 3 analysis:
# 1. Phase 3: State-Gini for gentags
# 2. Phase 3A: State-Gini for classical baselines (RAKE, TF-IDF, YAKE)
#
# Prerequisites:
# - Phase 1 results in results/phase1_downloaded/
# - Phase 2 embeddings cached in results/phase2_cache/
# - OPENAI_API_KEY set in .env

set -e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Phase 3 Analysis (v2 - State-Gini)"
echo "============================================================"
echo ""

# Check for Phase 1 results
if [ ! -d "results/phase1_downloaded" ]; then
    echo "ERROR: Phase 1 results not found at results/phase1_downloaded/"
    echo "Run: poetry run python scripts/run_phase1.py first"
    exit 1
fi

# Check for Phase 2 cache
if [ ! -f "results/phase2_cache/tag_embeddings_text_embedding_3_large_normeval.npz" ]; then
    echo "ERROR: Phase 2 embeddings not found"
    echo "Run Phase 2 analysis first"
    exit 1
fi

# Run Phase 3 (gentag State-Gini)
echo ""
echo "[1/2] Running Phase 3: Gentag State-Gini..."
echo ""
poetry run python scripts/state_gini_full.py

# Run Phase 3A (baseline State-Gini)
echo ""
echo "[2/2] Running Phase 3A: Baseline State-Gini..."
echo ""
poetry run python scripts/phase3a_baselines.py

echo ""
echo "============================================================"
echo "Phase 3 Analysis Complete"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - results/phase3/tables/state_localization.csv"
echo "  - results/phase3/tables/drift_localization.csv"
echo "  - results/phase3a/tables/baseline_state_gini.csv"
echo "  - results/phase3a/tables/state_gini_summary.csv"
echo ""
