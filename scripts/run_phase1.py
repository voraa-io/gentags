#!/usr/bin/env python3
"""
CLI runner: 500 venues x models x prompts x runs

Usage:
    python scripts/run_phase1.py --data data/external/venues_data.csv --output results/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import GentagExtractor, load_venue_data, run_experiment, save_results


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 extraction experiment")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to venues_data.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of venues to sample (default: 500)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to use (default: all available)"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Prompts to use (default: all)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of runs per combination (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to .env file (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    print("Initializing extractor...")
    extractor = GentagExtractor(env_path=args.env)
    available = extractor.available_models()
    print(f"Available models: {available}")
    
    if not available:
        print("ERROR: No models available. Check your API keys in .env file.")
        return 1
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    venues_df = load_venue_data(args.data, sample_size=args.sample_size, random_seed=args.seed)
    print(f"Loaded {len(venues_df)} venues")
    
    # Run experiment
    print(f"\nRunning experiment...")
    print(f"  Models: {args.models or 'all'}")
    print(f"  Prompts: {args.prompts or 'all'}")
    print(f"  Runs per combination: {args.runs}")
    
    results_df = run_experiment(
        extractor=extractor,
        venues_df=venues_df,
        models=args.models,
        prompts=args.prompts,
        runs=args.runs,
        verbose=True,
        raw_output_dir=f"{args.output}/raw"
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    output_path = save_results(results_df, output_dir=args.output, prefix="gentags")
    print(f"Results saved to: {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total rows: {len(results_df)}")
    print(f"  Unique venues: {results_df['venue_id'].nunique()}")
    print(f"  Unique experiments: {results_df['exp_id'].nunique()}")
    print(f"  Rows with tags: {results_df['tag_raw'].notna().sum()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
