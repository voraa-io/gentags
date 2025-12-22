"""
Data structures: ExtractionResult, ExperimentConfig.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .config import PROMPT_VERSION, PROMPT_HASH, SYSTEM_PROMPT_HASH, MODEL_VERSION


@dataclass
class ExtractionResult:
    """Single extraction result with full metadata."""
    run_id: str
    venue_id: str
    venue_name: str
    model: str
    prompt_type: str
    run_number: int
    exp_id: str
    timestamp: str
    tags: List[str]
    tags_filtered_out: List[str] = field(default_factory=list)  # Tags that violated constraints
    
    # Input metadata
    num_reviews: int = 0
    reviews_total_chars: int = 0
    input_prompt_hash: str = ""  # Hash of exact prompt sent to model
    
    # Extraction metadata
    time_seconds: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    status: str = "success"  # "success", "error", "parse_error"
    error: Optional[str] = None
    raw_response: Optional[str] = None
    
    # Version tracking (for reproducibility)
    prompt_version: str = PROMPT_VERSION
    prompt_hash: str = PROMPT_HASH
    system_prompt_hash: str = SYSTEM_PROMPT_HASH
    model_version: str = MODEL_VERSION
    pipeline_version: str = "1.2"


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment run."""
    venues: List[str]  # venue IDs
    models: List[str]  # model keys from MODELS dict
    prompts: List[str]  # prompt keys from PROMPTS dict
    runs_per_combination: int = 2
    random_seed: Optional[int] = 42
    
    # Inclusion rules
    min_reviews: int = 1
    max_reviews: Optional[int] = None  # None = no limit
    language: str = "en"  # Future: language filtering
    
    # Output
    output_dir: str = "results"
    save_intermediate: bool = True

