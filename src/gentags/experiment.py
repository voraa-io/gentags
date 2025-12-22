"""
Experiment runner: run_experiment().
"""

from typing import List, Optional
import pandas as pd

from .extractor import GentagExtractor, get_venue_id
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
    raw_output_dir: str = "results/raw"
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
    
    Returns:
        DataFrame in tags_df format (one row per tag)
        Includes both tag_raw and tag_norm columns
    """
    models = models or extractor.available_models()
    prompts = prompts or list(PROMPTS.keys())
    
    total = len(venues_df) * len(models) * len(prompts) * runs
    completed = 0
    
    all_results = []
    
    for _, venue_row in venues_df.iterrows():
        venue_name = venue_row['name']
        venue_reviews = venue_row['google_reviews']
        venue_id = venue_row.get('id') or get_venue_id(venue_name)
        
        for model in models:
            for prompt_type in prompts:
                for run_num in range(1, runs + 1):
                    if verbose:
                        completed += 1
                        print(f"[{completed}/{total}] {venue_name} | {model} | {prompt_type} | run {run_num}")
                    
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
    
    return pd.DataFrame(all_results)

