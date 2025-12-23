#!/usr/bin/env python3
"""
Week 2: Full extraction run
500 venues x models x prompts x runs

Usage:
    poetry run python scripts/run_phase1.py --data data/study1_venues_20250117.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import (
    GentagExtractor, 
    load_venue_data, 
    run_experiment, 
    save_results,
    summarize_cost,
    PROMPTS
)


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 extraction experiment")
    parser.add_argument(
        "--data",
        type=str,
        default="data/study1_venues_20250117.csv",
        help="Path to venues_data.csv (default: data/study1_venues_20250117.csv)"
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
        default=None,
        help="Number of venues to sample (default: None = use all venues)"
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
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=50.0,
        help="Maximum cost in USD before requiring confirmation (default: 50.0)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N extractions (default: 50)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint (start fresh)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID prefix to resume/reuse outputs (e.g. week2_run_20251222_101530). If omitted, a new one is created."
    )
    parser.add_argument(
        "--pilot-venues",
        type=int,
        default=3,
        help="Number of venues to use for pilot cost estimation (default: 3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: validate everything but don't call APIs (no cost)"
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
    
    # Validate models arg
    if args.models:
        bad = [m for m in args.models if m not in available]
        if bad:
            print(f"ERROR: Unknown/unavailable models: {bad}. Available: {available}")
            return 1
    
    # Validate prompts arg
    if args.prompts:
        bad = [p for p in args.prompts if p not in PROMPTS]
        if bad:
            print(f"ERROR: Unknown prompts: {bad}. Available: {list(PROMPTS.keys())}")
            return 1
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    venues_df = load_venue_data(args.data, sample_size=args.sample_size, random_seed=args.seed)
    print(f"Loaded {len(venues_df)} venues")
    
    # Ensure output directories exist
    Path(args.output, "raw").mkdir(parents=True, exist_ok=True)
    Path(args.output, "meta").mkdir(parents=True, exist_ok=True)
    
    # Determine models and prompts to use (explicit)
    models_to_use = args.models or available
    prompts_to_use = args.prompts or list(PROMPTS.keys())
    
    total_extractions = len(venues_df) * len(models_to_use) * len(prompts_to_use) * args.runs
    
    # Generate run ID prefix (reuse if --run-id provided, otherwise create new)
    run_id_prefix = args.run_id or datetime.now().strftime("week2_run_%Y%m%d_%H%M%S")
    checkpoint_path = Path(args.output) / f"{run_id_prefix}_checkpoint.csv"
    
    # Dry run mode: skip API calls, just validate
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No API calls will be made")
        print(f"  Run ID: {run_id_prefix}")
        print(f"  Venues: {len(venues_df)}")
        print(f"  Models: {models_to_use}")
        print(f"  Prompts: {prompts_to_use}")
        print(f"  Runs: {args.runs}")
        print(f"  Total extractions: {total_extractions}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output dir: {args.output}")
        if args.run_id:
            print(f"  ⚠️  Using provided --run-id: {args.run_id}")
        print(f"\n✅ All validations passed. Remove --dry-run to run for real.")
        return 0
    
    # Cost guard: Run pilot to estimate cost
    pilot_venue_count = min(args.pilot_venues, len(venues_df))
    if pilot_venue_count == 0:
        print("ERROR: No venues available for pilot run.")
        return 1
    
    print(f"\n💰 Cost estimation (pilot run with {pilot_venue_count} venues)...")
    print("  (This will make real API calls and cost money)")
    pilot_df = venues_df.head(pilot_venue_count).copy()
    pilot_results = run_experiment(
        extractor=extractor,
        venues_df=pilot_df,
        models=models_to_use,
        prompts=prompts_to_use,
        runs=1,  # Just 1 run for pilot
        verbose=False,
        save_raw_on_error=False
    )
    pilot_summary = summarize_cost(pilot_results)
    pilot_extractions = len(pilot_df) * len(models_to_use) * len(prompts_to_use) * 1
    avg_cost_per_extraction = pilot_summary['avg_cost_per_extraction_usd']
    
    if pilot_extractions == 0:
        print("ERROR: Pilot run produced no extractions. Aborting.")
        return 1
    
    estimated_total_cost = avg_cost_per_extraction * total_extractions
    
    print(f"  Pilot: {pilot_extractions} extractions = ${pilot_summary['total_cost_usd']:.6f}")
    print(f"  Avg cost per extraction: ${avg_cost_per_extraction:.6f}")
    print(f"  Estimated total cost ({total_extractions} extractions): ${estimated_total_cost:.2f}")
    
    if estimated_total_cost > args.max_cost_usd:
        print(f"\n⚠️  WARNING: Estimated cost (${estimated_total_cost:.2f}) exceeds max (${args.max_cost_usd:.2f})")
        response = input(f"Continue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return 1
    
    # Display run ID (will be reused if --run-id was provided)
    if args.run_id:
        print(f"\n📌 Using existing run ID: {run_id_prefix}")
        print(f"   Checkpoint: {checkpoint_path}")
        if checkpoint_path.exists():
            print(f"   ✓ Checkpoint exists - will resume from it")
        else:
            print(f"   ⚠️  Checkpoint not found - starting fresh")
    else:
        print(f"\n📌 New run ID: {run_id_prefix}")
    
    # Run experiment
    print(f"\nRunning experiment...")
    print(f"  Models: {models_to_use}")
    print(f"  Prompts: {prompts_to_use}")
    print(f"  Runs per combination: {args.runs}")
    print(f"  Total extractions: {total_extractions}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Estimated cost: ${estimated_total_cost:.2f}")
    
    results_df = run_experiment(
        extractor=extractor,
        venues_df=venues_df,
        models=models_to_use,
        prompts=prompts_to_use,
        runs=args.runs,
        verbose=True,
        raw_output_dir=f"{args.output}/raw",
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume
    )
    
    # Save main results (tags CSV) with run ID prefix (atomic write)
    print(f"\nSaving results to {args.output}...")
    tags_path = Path(args.output) / f"{run_id_prefix}_tags.csv"
    tmp_tags = tags_path.with_suffix('.tmp')
    results_df.to_csv(tmp_tags, index=False)
    tmp_tags.replace(tags_path)
    print(f"Tags CSV saved to: {tags_path}")
    
    # Generate cost summary
    print("\nGenerating cost summary...")
    summary = summarize_cost(results_df)
    
    # Save extraction-level CSV with run ID prefix (atomic write)
    extractions_path = Path(args.output) / f"{run_id_prefix}_extractions.csv"
    tmp_extractions = extractions_path.with_suffix('.tmp')
    summary["extractions"].to_csv(tmp_extractions, index=False)
    tmp_extractions.replace(extractions_path)
    print(f"Extractions CSV saved to: {extractions_path}")
    
    # Save cost breakdown by model/prompt with run ID prefix (atomic write)
    cost_path = Path(args.output) / f"{run_id_prefix}_cost_by_model_prompt.csv"
    tmp_cost = cost_path.with_suffix('.tmp')
    summary["by_model_prompt"].to_csv(tmp_cost, index=False)
    tmp_cost.replace(cost_path)
    print(f"Cost breakdown saved to: {cost_path}")
    
    # Generate manifest with run_id_prefix
    print("\nGenerating reproducibility manifest...")
    manifest_path = Path(args.output) / "meta" / f"{run_id_prefix}_manifest.json"
    
    # Import generate_manifest function
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_manifest import generate_manifest
    except ImportError:
        print("ERROR: generate_manifest.py not found next to this script.")
        print("Fix: put scripts/generate_manifest.py in the same folder as run_phase1.py, or move it into src/gentags/.")
        return 1
    
    generate_manifest(
        dataset_name="study1_venues_20250117",
        dataset_path=args.data,
        row_count=len(venues_df),
        sample_size=args.sample_size,  # Fix: record actual sample_size used
        output_path=str(manifest_path)
    )
    
    # Print summary
    print(f"\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total rows (tags): {len(results_df)}")
    print(f"  Unique venues: {results_df['venue_id'].nunique()}")
    print(f"  Unique experiments: {results_df['exp_id'].nunique()}")
    print(f"  Rows with tags: {results_df['tag_raw'].notna().sum()}")
    print(f"\nCost Summary:")
    print(f"  Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"  Avg cost per extraction: ${summary['avg_cost_per_extraction_usd']:.6f}")
    print(f"  Avg cost per tag: ${summary['avg_cost_per_tag_usd']:.6f}")
    print(f"\nOutput files:")
    print(f"  {tags_path}")
    print(f"  {extractions_path}")
    print(f"  {cost_path}")
    print(f"  {manifest_path}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
