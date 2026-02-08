#!/usr/bin/env python3
"""
Minimal smoke test: 1 venue × 1 prompt × 1 model × 1 run

Goal: Confirm we get:
- JSON list of gentags
- ExtractionResult
- Saved CSV row
"""

import sys
from pathlib import Path

# Add src to path (OK for smoke test; long-term prefer `poetry run`)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import GentagExtractor, load_venue_data, run_experiment, save_results


def main():
    print("=" * 60)
    print("MINIMAL SMOKE TEST")
    print("=" * 60)
    print("1 venue × 1 prompt (minimal) × 1 model (openai) × 1 run\n")
    
    # Initialize extractor
    print("1. Initializing extractor...")
    extractor = GentagExtractor()
    available = extractor.available_models()
    print(f"   Available models: {available}")
    
    if "openai" not in available:
        print("   ⚠ WARNING: OpenAI model not available. Skipping extraction.")
        print("   (This is OK for CI - tests pass without API keys)")
        print("   To run full test, set OPENAI_API_KEY in .env")
        return 0  # Exit gracefully, don't fail
    
    # Load sample data
    print("\n2. Loading sample data...")
    data_path = Path(__file__).parent.parent / "data" / "study1_venues_20250117.csv"
    
    if not data_path.exists():
        print(f"   ERROR: Sample data not found at {data_path}")
        return 1
    
    venues_df = load_venue_data(str(data_path), sample_size=1, random_seed=42)
    
    if len(venues_df) == 0:
        print("   ERROR: No venues loaded")
        return 1
    
    venue = venues_df.iloc[0]
    venue_name = venue["name"]
    reviews = venue["google_reviews"]

    print(f"   Venue: {venue_name}")
    print(f"   Reviews: {len(reviews)}")
    for i, review in enumerate(reviews[:3], 1):
        print(f"     Review {i}: {review[:60]}...")

    print("\n3. Running extraction...")
    results_df = run_experiment(
        extractor=extractor,
        venues_df=venues_df,
        models=["openai"],
        prompts=["minimal"],
        runs=1,
        verbose=True,
        save_raw_on_error=False,
    )

    print("\n4. Results:")
    if len(results_df) == 0:
        print("   ERROR: No results returned")
        return 1

    row = results_df.iloc[0]
    print(f"   Run ID: {row['run_id']}")
    print(f"   Venue ID: {row['venue_id']}")
    print(f"   Venue Name: {row['venue_name']}")
    print(f"   Model: {row['model']}")
    print(f"   Prompt: {row['prompt_type']}")
    print(f"   Status: {row['status']}")
    print(f"   Time: {row['time_seconds']}s")
    
    # Show tags (JSON list)
    tags_df = results_df[results_df["tag_raw"].notna()]
    if len(tags_df) > 0:
        tags = tags_df["tag_raw"].tolist()
        print(f"\n   Tags extracted: {len(tags)}")
        print(f"   JSON list: {tags}")

        print("\n   Individual tags:")
        for i, r in enumerate(tags_df.itertuples(index=False), 1):
            print(f"     {i}. {r.tag_raw} (norm: {r.tag_norm})")
    else:
        print("\n   ⚠ No tags extracted (check status/error)")
        if row["status"] != "success" and "error" in results_df.columns:
            print(f"   Error: {row.get('error')}")
    
    # Save to CSV
    print("\n5. Saving to CSV...")
    output_path = save_results(results_df, output_dir="results", prefix="smoke_test")
    print(f"   Saved to: {output_path}")
    
    print("\n✓ SMOKE TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

