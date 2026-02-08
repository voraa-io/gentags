#!/usr/bin/env python3
"""
Sanity check: validate_tags() and basic stats.

This is the Week 1 completion smoke test:
- loads data/sample/venues_sample.csv
- runs 1 venue × 1 model × 1 prompt × 1 run
- validates output (validate_tags)
- prints get_version_info()

Usage:
    python scripts/sanity_check.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import (
    GentagExtractor,
    load_venue_data,
    run_experiment,
    validate_tags,
    get_version_info,
)


def main():
    print("=" * 60)
    print("GENTAGS WEEK 1 SMOKE TEST")
    print("=" * 60)
    
    # Print version info
    print("\n1. Version Information:")
    print("-" * 60)
    info = get_version_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Initialize extractor
    print("\n2. Initializing extractor...")
    print("-" * 60)
    extractor = GentagExtractor()
    available = extractor.available_models()
    print(f"  Available models: {available}")
    
    if not available:
        print("  ERROR: No models available. Check your API keys in .env file.")
        return 1
    
    # Use first available model
    test_model = available[0]
    print(f"  Using model: {test_model}")
    
    # Load sample data
    print("\n3. Loading sample data...")
    print("-" * 60)
    data_path = Path(__file__).parent.parent / "data" / "sample" / "venues_sample.csv"
    
    if not data_path.exists():
        print(f"  ERROR: Sample data not found at {data_path}")
        print("  Please create data/sample/venues_sample.csv")
        return 1
    
    venues_df = load_venue_data(str(data_path), sample_size=1, random_seed=42)
    print(f"  Loaded {len(venues_df)} venue(s)")
    print(f"  Venue: {venues_df.iloc[0]['name']}")
    
    # Run single extraction
    print("\n4. Running extraction...")
    print("-" * 60)
    results_df = run_experiment(
        extractor=extractor,
        venues_df=venues_df,
        models=[test_model],
        prompts=["minimal"],
        runs=1,
        verbose=True,
        save_raw_on_error=False  # Don't save raw for smoke test
    )
    
    # Validate results
    print("\n5. Validating results...")
    print("-" * 60)
    validation = validate_tags(results_df)
    
    print(f"  Total rows: {validation['total_rows']}")
    print(f"  Rows with tags: {validation['rows_with_tags']}")
    print(f"  Unique tags (raw): {validation['unique_tags_raw']}")
    print(f"  Unique tags (normalized): {validation['unique_tags_norm']}")
    print(f"  Unique venues: {validation['unique_venues']}")
    print(f"  Unique experiments: {validation['unique_experiments']}")
    
    if validation.get('tag_length'):
        print(f"  Tag length - Mean: {validation['tag_length']['mean']} words")
        print(f"  Tag length - Min: {validation['tag_length']['min']} words")
        print(f"  Tag length - Max: {validation['tag_length']['max']} words")
    
    if validation['issues']:
        print(f"\n  ⚠ Issues found:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    else:
        print(f"\n  ✓ No issues found")
    
    # Final status
    print("\n" + "=" * 60)
    if validation['passed']:
        print("✓ WEEK 1 SMOKE TEST PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ WEEK 1 SMOKE TEST FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
