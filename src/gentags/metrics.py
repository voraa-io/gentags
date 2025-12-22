"""
Metrics: jaccard, overlap helpers, validation.
"""

from typing import List, Dict, Any
import pandas as pd

from .normalize import normalize_tag


def compute_jaccard_similarity(tags1: List[str], tags2: List[str], normalized: bool = True) -> float:
    """
    Compute Jaccard similarity between two tag lists.
    
    Args:
        tags1, tags2: Lists of tags
        normalized: If True, normalize tags before comparison
    
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    if normalized:
        set1 = set(normalize_tag(t) for t in tags1)
        set2 = set(normalize_tag(t) for t in tags2)
    else:
        set1, set2 = set(tags1), set(tags2)
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def validate_tags(tags_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run sanity checks on extracted tags.
    
    Args:
        tags_df: DataFrame with extracted tags
    
    Returns:
        Dict with validation results
    """
    # Filter to rows with actual tags
    tags_only = tags_df[tags_df['tag_raw'].notna()].copy()
    
    results = {
        "total_rows": len(tags_df),
        "rows_with_tags": len(tags_only),
        "unique_tags_raw": tags_only['tag_raw'].nunique() if len(tags_only) > 0 else 0,
        "unique_tags_norm": tags_only['tag_norm'].nunique() if len(tags_only) > 0 else 0,
        "unique_venues": tags_df['venue_id'].nunique(),
        "unique_experiments": tags_df['exp_id'].nunique(),
        "issues": []
    }
    
    # Check for parse errors
    parse_errors = tags_df[tags_df['status'] == 'parse_error']
    if len(parse_errors) > 0:
        results["issues"].append(f"Found {len(parse_errors)} parse errors")
        results["parse_error_count"] = len(parse_errors)
    
    # Check for empty tags
    if len(tags_only) > 0:
        empty_tags = tags_only[tags_only['tag_raw'].str.strip() == '']
        if len(empty_tags) > 0:
            results["issues"].append(f"Found {len(empty_tags)} empty tags")
        
        # Check tag length distribution
        word_counts = tags_only['word_count']
        results["tag_length"] = {
            "mean": round(word_counts.mean(), 2),
            "min": int(word_counts.min()),
            "max": int(word_counts.max()),
            "over_4_words": int((word_counts > 4).sum())
        }
        
        if results["tag_length"]["over_4_words"] > 0:
            results["issues"].append(f"{results['tag_length']['over_4_words']} tags exceed 4 words")
        
        # Check for duplicates within same experiment
        dups = tags_only.groupby('exp_id')['tag_raw'].apply(lambda x: x.duplicated().sum()).sum()
        if dups > 0:
            results["issues"].append(f"Found {dups} duplicate tags within experiments")
    
    results["passed"] = len(results["issues"]) == 0
    
    return results


def summarize_cost(tags_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize cost metrics from tags DataFrame.
    
    tags_df is one row per tag. This summarizes cost at extraction level
    and returns a dict of useful rollups + extraction-level dataframe.
    
    Args:
        tags_df: DataFrame with one row per tag (includes cost_usd, tokens, etc.)
    
    Returns:
        Dict with:
        - total_cost_usd: Total cost across all extractions
        - total_extractions: Number of unique extractions
        - total_tags: Total number of tags
        - avg_cost_per_extraction_usd: Average cost per extraction
        - avg_cost_per_tag_usd: Average cost per tag
        - by_model_prompt: DataFrame with cost stats by model/prompt
        - extractions: DataFrame with one row per extraction (deduped)
    """
    df = tags_df.copy()
    
    # Rows that correspond to actual tags
    tags_only = df[df["tag_raw"].notna()].copy()
    
    group_cols = ["run_id", "exp_id", "venue_id", "model", "prompt_type", "run_number"]
    
    # One row per extraction (dedupe the repeated cost across tag rows)
    extractions = (
        df.groupby(group_cols, dropna=False)
        .agg(
            timestamp=("timestamp", "first"),
            status=("status", "first"),
            time_seconds=("time_seconds", "first"),
            input_tokens=("input_tokens", "first"),
            output_tokens=("output_tokens", "first"),
            total_tokens=("total_tokens", "first"),
            cost_usd=("cost_usd", "first"),
        )
        .reset_index()
    )
    
    # Attach tag counts
    tag_counts = (
        tags_only.groupby(group_cols, dropna=False)
        .agg(
            n_tags=("tag_raw", "count"),
            n_unique_tag_eval=("tag_norm_eval", "nunique"),
        )
        .reset_index()
    )
    
    extractions = extractions.merge(tag_counts, on=group_cols, how="left")
    extractions["n_tags"] = extractions["n_tags"].fillna(0).astype(int)
    extractions["n_unique_tag_eval"] = extractions["n_unique_tag_eval"].fillna(0).astype(int)
    
    # Cost efficiency
    extractions["cost_per_tag"] = extractions.apply(
        lambda r: (r["cost_usd"] / r["n_tags"]) if r["cost_usd"] and r["n_tags"] > 0 else None,
        axis=1
    )
    extractions["cost_per_unique_tag"] = extractions.apply(
        lambda r: (r["cost_usd"] / r["n_unique_tag_eval"]) if r["cost_usd"] and r["n_unique_tag_eval"] > 0 else None,
        axis=1
    )
    
    # Rollups
    total_cost = extractions["cost_usd"].dropna().sum()
    total_extractions = len(extractions)
    total_tags = int(tags_only.shape[0])
    
    by_model_prompt = (
        extractions.groupby(["model", "prompt_type"], dropna=False)
        .agg(
            n_extractions=("exp_id", "count"),
            cost_total=("cost_usd", "sum"),
            cost_mean=("cost_usd", "mean"),
            tokens_mean=("total_tokens", "mean"),
            tags_mean=("n_tags", "mean"),
            unique_tags_mean=("n_unique_tag_eval", "mean"),
            cost_per_tag_mean=("cost_per_tag", "mean"),
        )
        .reset_index()
        .sort_values(["cost_mean"], ascending=True)
    )
    
    summary = {
        "total_cost_usd": float(total_cost),
        "total_extractions": int(total_extractions),
        "total_tags": int(total_tags),
        "avg_cost_per_extraction_usd": float(total_cost / total_extractions) if total_extractions > 0 else 0.0,
        "avg_cost_per_tag_usd": float(total_cost / total_tags) if total_tags > 0 else 0.0,
        "by_model_prompt": by_model_prompt,
        "extractions": extractions,
    }
    return summary

