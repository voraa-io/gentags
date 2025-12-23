"""
Experiment runner: run_experiment().
"""

from typing import List, Optional
import pandas as pd

from .extractor import GentagExtractor, get_venue_id, generate_exp_id
from .config import PROMPTS
from .normalize import normalize_tag, normalize_tag_eval
from .io import save_raw_response


def run_experiment(
    extractor: GentagExtractor,
    venues_df: pd.DataFrame,
    models: List[str] = None,
    prompts: List[str] = None,
    runs: int = 2,
    verbose: bool = True,
    save_raw_on_error: bool = True,
    raw_output_dir: str = "results/raw",
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 50,
    resume: bool = True
) -> pd.DataFrame:
    """
    Run full experiment matrix and return results as DataFrame.
    
    Args:
        extractor: Initialized GentagExtractor
        venues_df: DataFrame with venue data (must have 'name' and 'google_reviews' columns)
        models: List of model keys (default: all available)
        prompts: List of prompt keys (default: all)
        runs: Number of runs per combination
        verbose: Print progress
        save_raw_on_error: Save raw responses on errors
        raw_output_dir: Directory for raw error responses
        checkpoint_path: Path to checkpoint CSV file (for resume/periodic saves)
        checkpoint_every: Save checkpoint every N extractions (default: 50)
        resume: If True and checkpoint exists, skip completed exp_ids
    
    Returns:
        DataFrame in tags_df format (one row per tag)
        Includes both tag_raw and tag_norm columns
    """
    from pathlib import Path
    
    models = models or extractor.available_models()
    prompts = prompts or list(PROMPTS.keys())
    
    total = len(venues_df) * len(models) * len(prompts) * runs
    completed = 0
    
    # Load existing checkpoint if resuming
    completed_exp_ids = set()
    all_results = []
    
    if resume and checkpoint_path and Path(checkpoint_path).exists():
        try:
            existing_df = pd.read_csv(checkpoint_path)
            completed_exp_ids = set(existing_df['exp_id'].unique())
            all_results = existing_df.to_dict('records')
            print(f"Resuming: Found {len(completed_exp_ids)} completed extractions in checkpoint")
            if verbose:
                print(f"  Loaded {len(all_results)} existing rows")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}. Starting fresh.")
    
    extraction_count = 0
    
    for _, venue_row in venues_df.iterrows():
        venue_name = venue_row['name']
        venue_reviews = venue_row['google_reviews']
        venue_id = venue_row.get('id') or get_venue_id(venue_name)
        
        for model in models:
            for prompt_type in prompts:
                for run_num in range(1, runs + 1):
                    # Generate exp_id to check if already completed (must match generate_exp_id format)
                    exp_id = generate_exp_id(venue_id, model, prompt_type, run_num)
                    
                    # Skip if already completed (resume mode)
                    if resume and exp_id in completed_exp_ids:
                        if verbose:
                            completed += 1
                            print(f"[{completed}/{total}] {venue_name} | {model} | {prompt_type} | run {run_num} (skipped - already done)")
                        continue
                    
                    if verbose:
                        completed += 1
                        print(f"[{completed}/{total}] {venue_name} | {model} | {prompt_type} | run {run_num}")
                    
                    extraction_count += 1  # Count actual extractions (not skipped)
                    result = extractor.extract(
                        model=model,
                        prompt_type=prompt_type,
                        venue_name=venue_name,
                        venue_reviews=venue_reviews,
                        run_number=run_num,
                        venue_id=venue_id
                    )
                    
                    # Convert to rows (one per tag)
                    base_row = {
                        "run_id": result.run_id,
                        "venue_id": result.venue_id,
                        "venue_name": result.venue_name,
                        "model": result.model,
                        "prompt_type": result.prompt_type,
                        "run_number": result.run_number,
                        "exp_id": result.exp_id,
                        "timestamp": result.timestamp,
                        "num_reviews": result.num_reviews,
                        "reviews_total_chars": result.reviews_total_chars,
                        "time_seconds": result.time_seconds,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "total_tokens": result.total_tokens,
                        "cost_usd": result.cost_usd,
                        "status": result.status,
                        "prompt_hash": result.prompt_hash,
                        "system_prompt_hash": result.system_prompt_hash,
                        "input_prompt_hash": result.input_prompt_hash,
                        "tags_filtered_count": len(result.tags_filtered_out),
                        "extraction_phase": "phase1"
                    }
                    
                    if result.tags:
                        for tag in result.tags:
                            row = base_row.copy()
                            row.update({
                                "tag_raw": tag,
                                "tag_norm": normalize_tag(tag),
                                "tag_norm_eval": normalize_tag_eval(tag),
                                "word_count": len(tag.split()),
                            })
                            all_results.append(row)
                    else:
                        # Log extraction with no tags (for tracking parse errors)
                        row = base_row.copy()
                        row.update({
                            "tag_raw": None,
                            "tag_norm": None,
                            "tag_norm_eval": None,
                            "word_count": None,
                        })
                        all_results.append(row)
                    
                    # Save raw response on error/parse_error
                    if save_raw_on_error and result.status in ("error", "parse_error") and result.raw_response:
                        save_raw_response(result.raw_response, result.exp_id, result.run_id, raw_output_dir)
                    
                    if verbose:
                        if result.status == "success":
                            print(f"    ✓ {len(result.tags)} tags extracted")
                        elif result.status == "parse_error":
                            print(f"    ⚠ Parse error (raw response saved to {raw_output_dir}/)")
                    else:
                        print(f"    ✗ Error: {result.error}")
                    
                    # Checkpoint periodically
                    if checkpoint_path and extraction_count % checkpoint_every == 0:
                        checkpoint_df = pd.DataFrame(all_results)
                        checkpoint_df.to_csv(checkpoint_path, index=False)
                        if verbose:
                            print(f"    💾 Checkpoint saved ({len(all_results)} rows)")
    
    # Final checkpoint save
    if checkpoint_path:
        checkpoint_df = pd.DataFrame(all_results)
        checkpoint_df.to_csv(checkpoint_path, index=False)
        if verbose:
            print(f"\n💾 Final checkpoint saved: {checkpoint_path}")
    
    return pd.DataFrame(all_results)

