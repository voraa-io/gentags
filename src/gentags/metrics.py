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

