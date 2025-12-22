"""
Gentags: Research Tag Extraction Pipeline

A research project for extracting structured tags from unstructured text data.
"""

__version__ = "1.2.0"

from .pipeline import (
    GentagExtractor,
    PROMPTS,
    MODELS,
    get_version_info,
    load_venue_data,
    run_experiment,
    save_results,
    save_raw_response,
    load_results,
    validate_tags,
    compute_jaccard_similarity,
)

__all__ = [
    "GentagExtractor",
    "PROMPTS",
    "MODELS",
    "get_version_info",
    "load_venue_data",
    "run_experiment",
    "save_results",
    "save_raw_response",
    "load_results",
    "validate_tags",
    "compute_jaccard_similarity",
]
