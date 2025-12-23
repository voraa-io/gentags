#!/usr/bin/env python3
"""
Week 2: Full extraction run
500 venues x models x prompts x runs

Usage:
    poetry run python scripts/run_phase1.py --data data/study1_venues_20250117.csv
"""

import argparse
import sys
import os
import secrets
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
        default=None,
        help="Maximum cost in USD for THIS shard before requiring confirmation. If omitted, default is 50."
    )
    parser.add_argument(
        "--total-max-cost-usd",
        type=float,
        default=None,
        help="Total maximum cost across ALL shards. If provided with --num-shards, automatically sets --max-cost-usd per shard."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of parallel shards running. Used with --total-max-cost-usd to calculate per-shard budget."
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
        "--shard-name",
        type=str,
        default=None,
        help="Shard identifier for parallel runs (e.g. 'openai', 'gemini', 'minimal_prompt'). Used in output filenames to avoid collisions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: validate everything but don't call APIs (no cost)"
    )
    
    args = parser.parse_args()
    
    # Budget logic
    if args.total_max_cost_usd is not None or args.num_shards is not None:
        # Require both if using total budget mode
        if args.total_max_cost_usd is None or args.num_shards is None:
            print("ERROR: --total-max-cost-usd requires --num-shards (and vice versa).")
            return 1
        
        # If user ALSO set per-shard explicitly, that's ambiguous
        if args.max_cost_usd is not None:
            print(f"ERROR: Cannot use both --max-cost-usd ({args.max_cost_usd}) and "
                  f"--total-max-cost-usd ({args.total_max_cost_usd}).")
            return 1
        
        args.max_cost_usd = args.total_max_cost_usd / args.num_shards
        print(f"💰 Total budget: ${args.total_max_cost_usd:.2f} across {args.num_shards} shards")
        print(f"   Per-shard budget: ${args.max_cost_usd:.2f}")
    
    # If still unset, use default
    if args.max_cost_usd is None:
        args.max_cost_usd = 50.0
    
    # Require --shard-name when --run-id is provided (to avoid output collisions)
    if args.run_id and not args.shard_name:
        print("ERROR: --run-id requires --shard-name to avoid output collisions.")
        print("  When coordinating parallel shards, each shard must have a unique --shard-name.")
        return 1
    
    # Warn if pilot will run in multiple shards (wasteful)
    if args.shard_name and args.total_max_cost_usd and args.pilot_venues > 0:
        print(f"\n⚠️  WARNING: Pilot will run in this shard (--pilot-venues {args.pilot_venues})")
        print("   For parallel shards, consider running pilot once manually, then use --pilot-venues 0")
        print("   in all shards to avoid redundant pilot runs.")
        response = input("Continue with pilot in this shard? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted. Use --pilot-venues 0 to skip pilot.")
            return 1
    
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
    
    # Add shard name to filenames if provided (for parallel runs)
    shard_suffix = f"_{args.shard_name}" if args.shard_name else ""
    checkpoint_path = Path(args.output) / f"{run_id_prefix}_checkpoint{shard_suffix}.csv"
    
    # Resume if checkpoint exists (regardless of --run-id) unless --no-resume is set
    is_resuming = checkpoint_path.exists() and (not args.no_resume)
    
    # Dry run mode: skip API calls, just validate (BEFORE lockfile creation)
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
    
    # Print run info at top
    print(f"\n{'='*60}")
    print(f"Run ID: {run_id_prefix} | Shard: {args.shard_name or 'none'}")
    print(f"{'='*60}")
    
    # Lockfile: always lock for shard runs (resume or not)
    lockfile_path = None
    lockfile_created = False
    lockfile_token = None
    
    if args.shard_name:
        lockfile_path = Path(args.output) / f"{run_id_prefix}_checkpoint{shard_suffix}.lock"
        
        if lockfile_path.exists():
            # Read existing lockfile to check if process is still running
            try:
                existing_lock = lockfile_path.read_text()
                existing_pid = None
                existing_shard = None
                existing_started = None
                
                for line in existing_lock.strip().split('\n'):
                    if line.startswith('pid='):
                        existing_pid = int(line.split('=')[1])
                    elif line.startswith('shard='):
                        existing_shard = line.split('=', 1)[1]
                    elif line.startswith('started='):
                        existing_started = line.split('=', 1)[1]
                    elif line.startswith('token='):
                        # Old format, ignore
                        pass
                
                # Check if process is still running (Unix only)
                if existing_pid:
                    try:
                        os.kill(existing_pid, 0)  # Signal 0 = check if process exists
                        # Process exists - warn user
                        print(f"\n⚠️  WARNING: Lockfile exists and process {existing_pid} appears to be running")
                        print(f"   Shard: {existing_shard}")
                        print(f"   Started: {existing_started}")
                        print(f"   Lockfile: {lockfile_path}")
                        print("   If you are sure it is not running, delete the lockfile and retry.")
                        response = input("Continue anyway? (yes/no): ")
                        if response.lower() not in ["yes", "y"]:
                            print("Aborted.")
                            return 1
                    except ProcessLookupError:
                        # Process doesn't exist - safe to overwrite
                        print(f"⚠️  Lockfile exists but process {existing_pid} is not running. Overwriting.")
                    except OSError:
                        # Windows or permission issue - assume safe to overwrite
                        print(f"⚠️  Lockfile exists. Unable to verify process status. Proceeding with caution.")
            except Exception as e:
                print(f"⚠️  Could not read lockfile: {e}. Proceeding.")
        
        # Generate unique token for this run
        lockfile_token = secrets.token_hex(16)
        
        # Always create lockfile for shard runs (even when resuming)
        lockfile_path.write_text(
            f"pid={os.getpid()}\nshard={args.shard_name}\nstarted={datetime.now().isoformat()}\ntoken={lockfile_token}\n"
        )
        lockfile_created = True
    
    # Display run ID (will be reused if --run-id was provided)
    if args.run_id:
        print(f"\n📌 Using existing run ID: {run_id_prefix}")
        if args.shard_name:
            print(f"   Shard: {args.shard_name}")
        print(f"   Checkpoint: {checkpoint_path}")
        if checkpoint_path.exists():
            print(f"   ✓ Checkpoint exists - will resume from it")
        else:
            print(f"   ⚠️  Checkpoint not found - starting fresh")
    else:
        print(f"\n📌 New run ID: {run_id_prefix}")
        if args.shard_name:
            print(f"   Shard: {args.shard_name}")
    
    # Cost guard: Run pilot to estimate cost (skip if resuming or --pilot-venues 0)
    estimated_total_cost = None
    if not is_resuming and args.pilot_venues > 0:
        pilot_venue_count = min(args.pilot_venues, len(venues_df))
        if pilot_venue_count == 0:
            print("ERROR: No venues available for pilot run.")
            return 1
        
        print(f"\n💰 Cost estimation (pilot run with {pilot_venue_count} venues)...")
        print("  (This will make real API calls and cost money)")
        # Randomize pilot selection to avoid bias from CSV ordering
        pilot_df = venues_df.sample(n=pilot_venue_count, random_state=args.seed).copy()
        pilot_results = run_experiment(
            extractor=extractor,
            venues_df=pilot_df,
            models=models_to_use,
            prompts=prompts_to_use,
            runs=1,  # Just 1 run for pilot
            verbose=False,
            raw_output_dir=f"{args.output}/raw",
            save_raw_on_error=True
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
        print(f"  Estimated shard cost ({total_extractions} extractions): ${estimated_total_cost:.2f}")
        
        # If total budget is set, show full run estimate
        if args.total_max_cost_usd and args.num_shards:
            full_run_estimate = estimated_total_cost * args.num_shards
            print(f"  Estimated full run cost (all {args.num_shards} shards): ${full_run_estimate:.2f}")
        
        if estimated_total_cost > args.max_cost_usd:
            print(f"\n⚠️  WARNING: Estimated cost (${estimated_total_cost:.2f}) exceeds max (${args.max_cost_usd:.2f})")
            response = input(f"Continue anyway? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Aborted.")
                return 1
    elif args.pilot_venues == 0:
        print(f"\n💰 Skipping pilot cost estimation (--pilot-venues 0)")
    else:
        print(f"\n💰 Skipping pilot cost estimation (resume mode)")
    
    # Run experiment (wrapped in try/finally for lockfile cleanup)
    try:
        # Run experiment
        print(f"\nRunning experiment...")
        print(f"  Models: {models_to_use}")
        print(f"  Prompts: {prompts_to_use}")
        print(f"  Runs per combination: {args.runs}")
        print(f"  Total extractions: {total_extractions}")
        print(f"  Checkpoint: {checkpoint_path}")
        if estimated_total_cost is not None:
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
        tags_path = Path(args.output) / f"{run_id_prefix}_tags{shard_suffix}.csv"
        tmp_tags = tags_path.with_suffix('.tmp')
        results_df.to_csv(tmp_tags, index=False)
        tmp_tags.replace(tags_path)
        print(f"Tags CSV saved to: {tags_path}")
        
        # Generate cost summary
        print("\nGenerating cost summary...")
        summary = summarize_cost(results_df)
        
        # Save extraction-level CSV with run ID prefix (atomic write)
        extractions_path = Path(args.output) / f"{run_id_prefix}_extractions{shard_suffix}.csv"
        tmp_extractions = extractions_path.with_suffix('.tmp')
        summary["extractions"].to_csv(tmp_extractions, index=False)
        tmp_extractions.replace(extractions_path)
        print(f"Extractions CSV saved to: {extractions_path}")
        
        # Save cost breakdown by model/prompt with run ID prefix (atomic write)
        cost_path = Path(args.output) / f"{run_id_prefix}_cost_by_model_prompt{shard_suffix}.csv"
        tmp_cost = cost_path.with_suffix('.tmp')
        summary["by_model_prompt"].to_csv(tmp_cost, index=False)
        tmp_cost.replace(cost_path)
        print(f"Cost breakdown saved to: {cost_path}")
        
        # Generate manifest with run_id_prefix
        print("\nGenerating reproducibility manifest...")
        manifest_path = Path(args.output) / "meta" / f"{run_id_prefix}_manifest{shard_suffix}.json"
        
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
    
    finally:
        # Always cleanup lockfile if we created it (only if token matches)
        if lockfile_created and lockfile_path and lockfile_path.exists() and lockfile_token:
            try:
                # Only delete if token matches (prevent deleting another process's lock)
                existing_lock = lockfile_path.read_text()
                if f"token={lockfile_token}" in existing_lock:
                    lockfile_path.unlink()
                else:
                    # Token mismatch - another process may have overwritten it
                    print(f"\n⚠️  Lockfile token mismatch - not deleting (another process may be running)")
            except Exception as e:
                # If we can't read/delete, that's okay - don't crash
                pass


if __name__ == "__main__":
    sys.exit(main())
