#!/usr/bin/env python3
"""
Quick test of summarize_cost() function using existing smoke test results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import load_results, summarize_cost

def main():
    # Load the smoke test results
    results_path = Path(__file__).parent.parent / "results" / "smoke_test_20251221_201925.csv"
    
    if not results_path.exists():
        print(f"ERROR: Results file not found at {results_path}")
        return 1
    
    print("Loading results...")
    tags_df = load_results(str(results_path))
    print(f"Loaded {len(tags_df)} rows")
    
    print("\nComputing cost summary...")
    summary = summarize_cost(tags_df)
    
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"\nTotal cost: ${summary['total_cost_usd']:.6f}")
    print(f"Total extractions: {summary['total_extractions']}")
    print(f"Total tags: {summary['total_tags']}")
    print(f"Avg cost per extraction: ${summary['avg_cost_per_extraction_usd']:.6f}")
    print(f"Avg cost per tag: ${summary['avg_cost_per_tag_usd']:.6f}")
    
    print("\n" + "-" * 60)
    print("BY MODEL / PROMPT:")
    print("-" * 60)
    print(summary['by_model_prompt'].to_string(index=False))
    
    print("\n" + "-" * 60)
    print("EXTRACTION DETAILS (first 5):")
    print("-" * 60)
    extraction_cols = ['exp_id', 'model', 'prompt_type', 'cost_usd', 'n_tags', 'n_unique_tag_eval', 'cost_per_tag']
    print(summary['extractions'][extraction_cols].head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✓ Cost summary works!")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

