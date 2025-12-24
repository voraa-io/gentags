"""
GENTAGS EXTRACTION PIPELINE v1.2
================================
Week 1 Lock-In: Foundation & Methodological Commitments

This module provides a reproducible, importable pipeline for gentag extraction.
All definitions, prompts, and models are FROZEN for Study 1.

Note on "status" field:
- status="success" means FORMAT success (valid JSON list parsed)
- status="parse_error" means model output couldn't be parsed as JSON
- status="error" means API/network error
- This does NOT indicate semantic validity or hallucination-free output

Usage:
    from gentags import (
        GentagExtractor,
        PROMPTS,
        MODELS,
        load_venue_data,
        run_experiment,
    )
"""

import os
import json
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from uuid import uuid4
import pandas as pd

# =============================================================================
# 🔒 1. GENTAG DEFINITION (FROZEN)
# =============================================================================
"""
GENTAG DEFINITION (v1.0 - IMMUTABLE)
------------------------------------
A gentag is a semantic tag with the following properties:

✅ WHAT GENTAGS ARE:
- Atomized: One dominant semantic constraint per tag
- Short: 1-4 words
- Composable: Can be combined with other tags
- Embed-friendly: Suitable for vector similarity search
- Folksonomy: Emergent vocabulary, not predefined taxonomy
- Zero-shot emergent: Extracted without examples or ontology hints

❌ WHAT GENTAGS ARE NOT:
- NOT ratings (no stars, scores, or sentiment values)
- NOT categories (no predefined slots like "cuisine_type")
- NOT summaries (not paragraph-length descriptions)
- NOT sentiment labels (not "positive" / "negative")
- NOT schema-aligned attributes (no ontology)
"""

# =============================================================================
# 🔒 2. PROMPTS (FROZEN - v1.0)
# =============================================================================

PROMPTS = {
    "minimal": """Extract semantic tags ("gentags") for this venue based on the reviews.
A gentag is a short, meaningful semantic phrase (typically 1–4 words) that captures one idea expressed or strongly implied in the reviews.
Include any gentags that describe atmosphere, food, service, vibe, crowd, or typical occasions mentioned in the reviews.
Do not invent information beyond what the reviews support.
Return only a JSON list of gentags.""",

    "anti_hallucination": """Extract semantic tags ("gentags") for this venue based ONLY on what is explicitly stated or clearly implied in the reviews.
A gentag is a short, meaningful semantic phrase (typically 1–4 words) that captures a single idea grounded in the review text. It must not be a full sentence.
Do NOT infer, assume, generalize, or guess any information that is not directly supported by the reviews. 
If a concept is uncertain, ambiguous, or weakly implied, do NOT include it as a gentag.
Include only gentags that reflect concrete statements in the reviews.
Return only a JSON list of gentags.""",

    "short_phrase": """Extract semantic tags ("gentags") for this venue that summarize the key ideas expressed in the reviews.
A gentag must be a short phrase of 1–4 words that represents one clear semantic idea. 
Do not produce full sentences.
Tags must be grounded in the content of the reviews and should not rely on assumptions or outside knowledge.
Return only a JSON list of short gentags."""
}

PROMPT_VERSION = "1.0"
PROMPT_HASH = hashlib.md5(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest()[:8]

# System prompts per provider (FROZEN)
SYSTEM_PROMPTS = {
    "openai": "You extract only JSON lists of gentags based on reviews. No explanations.",
    "gemini": None,  # Gemini doesn't use system prompts
    "claude": None,   # Claude uses user prompt only for this task
    "grok": "You extract only JSON lists of gentags based on reviews. No explanations."
}
SYSTEM_PROMPT_HASH = hashlib.md5(json.dumps(SYSTEM_PROMPTS, sort_keys=True).encode()).hexdigest()[:8]

# =============================================================================
# 🔒 3. MODELS (FROZEN - v1.0)
# =============================================================================

MODELS = {
    "openai": {
        "name": "gpt-5-nano",
        "provider": "OpenAI",
        "short": "gpt5",
        "pricing": {"input_per_mtok": 0.05, "output_per_mtok": 0.40},
        "params": {
            "max_tokens": None,  # Provider default (no cap)
            "temperature": None  # Provider default
        }
    },
    "gemini": {
        "name": "gemini-2.5-flash",
        "provider": "Google",
        "short": "gemini25",
        "pricing": {"input_per_mtok": 0.25, "output_per_mtok": 0.50},
        "params": {
            "max_tokens": None,  # Provider default (no cap)
            "temperature": None  # Provider default
        }
    },
    "claude": {
        "name": "claude-sonnet-4-5",
        "provider": "Anthropic",
        "short": "claude45",
        "pricing": {"input_per_mtok": 3.00, "output_per_mtok": 15.00},
        "params": {
            "max_tokens": None,  # Claude API requires max_tokens; we use 8192 as fallback (not provider default)
            "temperature": None  # Provider default
        }
    },
    "grok": {
        "name": "grok-4",
        "provider": "xAI",
        "short": "grok4",
        "pricing": {"input_per_mtok": 2.00, "output_per_mtok": 10.00},  # Grok-4: $2/M input, $10/M output
        "params": {
            "max_tokens": None,  # Provider default
            "temperature": None  # Provider default
        },
        "base_url": "https://api.x.ai/v1"  # xAI uses OpenAI-compatible API
    }
}

# Study 1 extraction constraints
MAX_TAG_WORDS = 4  # Tags exceeding this are filtered out
MAX_TAGS_PER_EXTRACTION = None  # No cap - observe natural model behavior

MODEL_VERSION = "1.0"

# =============================================================================
# 🔒 4. DATA STRUCTURES
# =============================================================================

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


# =============================================================================
# 🔒 5. EXTRACTION FUNCTIONS
# =============================================================================

def extract_json_list(text: str) -> tuple[Optional[List[str]], str]:
    """
    Extract JSON list from model output.
    
    Returns:
        (tags, status): tags is list or None, status is "success" or "parse_error"
    """
    if not text:
        return None, "parse_error"
    
    text_stripped = text.strip()
    
    # Strategy 1: Try parsing entire text as JSON list
    try:
        parsed = json.loads(text_stripped)
        if isinstance(parsed, list):
            tags = [str(t).strip() for t in parsed if t and str(t).strip()]
            return tags, "success"
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Strip markdown code blocks and try again
    content = re.sub(r"^```(?:json)?\s*", "", text_stripped, flags=re.MULTILINE)
    content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, list):
            tags = [str(t).strip() for t in parsed if t and str(t).strip()]
            return tags, "success"
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Find the FIRST complete JSON array (greedy match for balanced brackets)
    # This is more robust than non-greedy regex
    bracket_start = text_stripped.find('[')
    if bracket_start != -1:
        depth = 0
        for i, char in enumerate(text_stripped[bracket_start:], start=bracket_start):
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    candidate = text_stripped[bracket_start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            tags = [str(t).strip() for t in parsed if t and str(t).strip()]
                            return tags, "success"
                    except json.JSONDecodeError:
                        pass
                    break
    
    # All strategies failed
    return None, "parse_error"


def normalize_tag(tag: str) -> str:
    """
    Light normalization for evaluation (tag_norm).
    - Lowercase
    - Strip punctuation (except internal hyphens/apostrophes)
    - Collapse whitespace
    """
    tag = tag.lower().strip()
    # Remove leading/trailing punctuation
    tag = re.sub(r'^[^\w]+|[^\w]+$', '', tag)
    # Collapse internal whitespace
    tag = re.sub(r'\s+', ' ', tag)
    return tag


def normalize_tag_eval(tag: str) -> str:
    """
    Stricter normalization for stability metrics (tag_norm_eval).
    - All of normalize_tag() plus:
    - Simple plural → singular
    - Remove common prefixes like "gets", "very", "really"
    """
    tag = normalize_tag(tag)
    
    # Remove common prefixes that don't change meaning
    prefixes = ['gets ', 'very ', 'really ', 'quite ', 'super ', 'pretty ']
    for prefix in prefixes:
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
    
    # Simple plural handling
    words = tag.split()
    normalized_words = []
    for word in words:
        if len(word) > 3:
            # Handle -ies → -y (pastries → pastry, fries → fry)
            if word.endswith('ies'):
                normalized_words.append(word[:-3] + 'y')
            # Handle -es after s/x/z/ch/sh (dishes → dish, boxes → box)
            elif word.endswith('es') and len(word) > 4 and word[-3] in 'sxz':
                normalized_words.append(word[:-2])
            elif word.endswith('ches') or word.endswith('shes'):
                normalized_words.append(word[:-2])
            # Handle regular -s (but not -ss like "glass")
            elif word.endswith('s') and not word.endswith('ss'):
                normalized_words.append(word[:-1])
            else:
                normalized_words.append(word)
        else:
            normalized_words.append(word)
    
    return ' '.join(normalized_words)


def filter_valid_tags(tags: List[str], max_words: int = MAX_TAG_WORDS) -> tuple[List[str], List[str]]:
    """
    Filter tags by word count constraint.
    
    Returns:
        (valid_tags, filtered_out_tags)
    """
    valid = []
    filtered = []
    for tag in tags:
        word_count = len(tag.split())
        if 1 <= word_count <= max_words:
            valid.append(tag)
        else:
            filtered.append(tag)
    return valid, filtered


def generate_exp_id(venue_id: str, model_key: str, prompt_type: str, run_number: int) -> str:
    """Generate experiment ID: e.g., BOU_gpt5_minimal_run1"""
    model_short = MODELS[model_key]["short"]
    prompt_short = {"minimal": "minimal", "anti_hallucination": "anti", "short_phrase": "short"}.get(prompt_type, prompt_type[:5])
    return f"{venue_id}_{model_short}_{prompt_short}_run{run_number}"


def generate_run_id() -> str:
    """Generate unique run ID (timestamp + pid + random suffix for concurrency safety)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{os.getpid()}_{uuid4().hex[:6]}"


def get_venue_id(venue_name: str) -> str:
    """Generate a short venue_id from venue name."""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', venue_name)
    words = clean.split()
    if len(words) == 1:
        return words[0][:8].upper()
    elif len(words) >= 2:
        return (words[0][:4] + "_" + words[1][:4]).upper()
    return clean[:8].upper() if clean else "VENUE"


class GentagExtractor:
    """
    Main extraction class. Handles all three model providers.
    
    Usage:
        extractor = GentagExtractor()
        result = extractor.extract(
            model="openai",
            prompt_type="minimal",
            venue_name="Bou",
            venue_reviews=["Review 1...", "Review 2..."],
            run_number=1
        )
    """
    
    def __init__(self, env_path: Optional[str] = None):
        """Initialize extractor with API clients."""
        self._load_env(env_path)
        self._init_clients()
    
    def _load_env(self, env_path: Optional[str] = None):
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            if env_path:
                load_dotenv(Path(env_path))
            else:
                # Try common locations (no hardcoded user paths)
                for path in [
                    Path(__file__).parent.parent.parent / ".env",
                    Path.cwd() / ".env"
                ]:
                    if path.exists():
                        load_dotenv(path)
                        break
        except ImportError:
            pass  # dotenv not installed, rely on system env vars
    
    def _init_clients(self):
        """Initialize API clients for all providers."""
        self.clients = {}
        
        # OpenAI
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.clients["openai"] = OpenAI(api_key=api_key)
        except ImportError:
            pass
        
        # Gemini
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self.clients["gemini"] = genai.Client(api_key=api_key)
        except ImportError:
            pass
        
        # Claude
        try:
            import anthropic
            api_key = os.environ.get("CLAUDE_API_KEY")
            if api_key:
                self.clients["claude"] = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            pass
        
        # Grok (xAI) - uses OpenAI-compatible API
        try:
            from openai import OpenAI
            api_key = os.environ.get("XAI_API_KEY")
            if api_key:
                base_url = MODELS["grok"].get("base_url", "https://api.x.ai/v1")
                self.clients["grok"] = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            pass
    
    def available_models(self) -> List[str]:
        """Return list of models with initialized clients."""
        return list(self.clients.keys())
    
    def extract(
        self,
        model: str,
        prompt_type: str,
        venue_name: str,
        venue_reviews: List[str],
        run_number: int = 1,
        venue_id: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract gentags from venue reviews using specified model and prompt.
        
        Args:
            model: Key from MODELS dict ("openai", "gemini", "claude")
            prompt_type: Key from PROMPTS dict ("minimal", "anti_hallucination", "short_phrase")
            venue_name: Human-readable venue name
            venue_reviews: List of review texts
            run_number: Run number for this experiment
            venue_id: Optional venue ID (generated from name if not provided)
        
        Returns:
            ExtractionResult with tags and metadata
        """
        if model not in self.clients:
            raise ValueError(f"Model '{model}' not available. Available: {self.available_models()}")
        
        if prompt_type not in PROMPTS:
            raise ValueError(f"Prompt type '{prompt_type}' not found. Available: {list(PROMPTS.keys())}")
        
        venue_id = venue_id or get_venue_id(venue_name)
        run_id = generate_run_id()
        exp_id = generate_exp_id(venue_id, model, prompt_type, run_number)
        timestamp = datetime.now().isoformat()
        
        # Input metadata
        num_reviews = len(venue_reviews)
        reviews_total_chars = sum(len(r) for r in venue_reviews)
        
        # Build prompt
        reviews_text = "\n\n".join([f"Review {i+1}: {review}" for i, review in enumerate(venue_reviews)])
        full_prompt = f"{PROMPTS[prompt_type]}\n\nReviews:\n{reviews_text}"
        input_prompt_hash = hashlib.md5(full_prompt.encode()).hexdigest()[:12]
        
        # Call appropriate extractor
        start_time = time.time()
        
        try:
            if model == "openai":
                result = self._extract_openai(full_prompt)
            elif model == "gemini":
                result = self._extract_gemini(full_prompt)
            elif model == "claude":
                result = self._extract_claude(full_prompt)
            elif model == "grok":
                result = self._extract_grok(full_prompt)
            else:
                raise ValueError(f"Unknown model: {model}")
            
            elapsed = time.time() - start_time
            
            # Filter tags by word count constraint
            raw_tags = result["tags"]
            valid_tags, filtered_tags = filter_valid_tags(raw_tags, MAX_TAG_WORDS)
            
            # Optional: cap total number of tags
            if MAX_TAGS_PER_EXTRACTION and len(valid_tags) > MAX_TAGS_PER_EXTRACTION:
                valid_tags = valid_tags[:MAX_TAGS_PER_EXTRACTION]
            
            return ExtractionResult(
                run_id=run_id,
                venue_id=venue_id,
                venue_name=venue_name,
                model=MODELS[model]["name"],
                prompt_type=prompt_type,
                run_number=run_number,
                exp_id=exp_id,
                timestamp=timestamp,
                tags=valid_tags,
                tags_filtered_out=filtered_tags,
                num_reviews=num_reviews,
                reviews_total_chars=reviews_total_chars,
                input_prompt_hash=input_prompt_hash,
                time_seconds=round(elapsed, 3),
                input_tokens=result.get("input_tokens"),
                output_tokens=result.get("output_tokens"),
                total_tokens=result.get("total_tokens"),
                cost_usd=result.get("cost_usd"),
                status=result.get("status", "success"),
                error=result.get("error"),
                raw_response=result.get("raw_response")
            )
        
        except Exception as e:
            elapsed = time.time() - start_time
            return ExtractionResult(
                run_id=run_id,
                venue_id=venue_id,
                venue_name=venue_name,
                model=MODELS[model]["name"],
                prompt_type=prompt_type,
                run_number=run_number,
                exp_id=exp_id,
                timestamp=timestamp,
                tags=[],
                tags_filtered_out=[],
                num_reviews=num_reviews,
                reviews_total_chars=reviews_total_chars,
                input_prompt_hash=input_prompt_hash,
                time_seconds=round(elapsed, 3),
                status="error",
                error=str(e)
            )
    
    def _extract_openai(self, full_prompt: str) -> Dict[str, Any]:
        """Extract using OpenAI."""
        client = self.clients["openai"]
        params = MODELS["openai"]["params"]
        
        kwargs = {
            "model": MODELS["openai"]["name"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["openai"]},
                {"role": "user", "content": full_prompt}
            ]
        }
        # Only add params if explicitly set (otherwise use provider defaults)
        if params.get("max_tokens") is not None:
            kwargs["max_tokens"] = params["max_tokens"]
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        
        response = client.chat.completions.create(**kwargs)
        
        raw = response.choices[0].message.content
        tags, parse_status = extract_json_list(raw)
        
        input_tokens = response.usage.prompt_tokens if response.usage else None
        output_tokens = response.usage.completion_tokens if response.usage else None
        
        cost = None
        if input_tokens and output_tokens:
            pricing = MODELS["openai"]["pricing"]
            cost = (input_tokens / 1_000_000) * pricing["input_per_mtok"] + \
                   (output_tokens / 1_000_000) * pricing["output_per_mtok"]
        
        return {
            "tags": tags or [],
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
            "cost_usd": round(cost, 6) if cost else None,
            "status": parse_status,
            "error": "Failed to parse JSON from response" if parse_status == "parse_error" else None
        }
    
    def _extract_gemini(self, full_prompt: str) -> Dict[str, Any]:
        """Extract using Gemini."""
        client = self.clients["gemini"]
        params = MODELS["gemini"]["params"]
        
        kwargs = {
            "model": MODELS["gemini"]["name"],
            "contents": full_prompt
        }
        
        # Only add config if params are explicitly set
        if params.get("max_tokens") is not None or params.get("temperature") is not None:
            from google.genai import types
            config_kwargs = {}
            if params.get("max_tokens") is not None:
                config_kwargs["max_output_tokens"] = params["max_tokens"]
            if params.get("temperature") is not None:
                config_kwargs["temperature"] = params["temperature"]
            kwargs["config"] = types.GenerateContentConfig(**config_kwargs)
        
        response = client.models.generate_content(**kwargs)
        
        raw = response.text
        tags, parse_status = extract_json_list(raw)
        
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None) if hasattr(response, 'usage_metadata') else None
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', None) if hasattr(response, 'usage_metadata') else None
        
        cost = None
        if input_tokens or output_tokens:
            pricing = MODELS["gemini"]["pricing"]
            cost = ((input_tokens or 0) / 1_000_000) * pricing["input_per_mtok"] + \
                   ((output_tokens or 0) / 1_000_000) * pricing["output_per_mtok"]
        
        return {
            "tags": tags or [],
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
            "cost_usd": round(cost, 6) if cost else None,
            "status": parse_status,
            "error": "Failed to parse JSON from response" if parse_status == "parse_error" else None
        }
    
    def _extract_claude(self, full_prompt: str) -> Dict[str, Any]:
        """Extract using Claude."""
        client = self.clients["claude"]
        params = MODELS["claude"]["params"]
        
        # Claude requires max_tokens - we set 8192 as a high ceiling (not a tuning choice)
        # This is documented as "as-is" policy: we don't pass temperature, top_p, etc.
        kwargs = {
            "model": MODELS["claude"]["name"],
            "max_tokens": 8192,  # Required by API, set high to not constrain output
            "messages": [{"role": "user", "content": full_prompt}]
        }
        # Note: We do NOT pass temperature, top_p, or other sampling parameters
        
        message = client.messages.create(**kwargs)
        
        raw = message.content[0].text
        tags, parse_status = extract_json_list(raw)
        
        input_tokens = getattr(message.usage, "input_tokens", None) if hasattr(message, "usage") else None
        output_tokens = getattr(message.usage, "output_tokens", None) if hasattr(message, "usage") else None
        
        cost = None
        if input_tokens or output_tokens:
            pricing = MODELS["claude"]["pricing"]
            cost = ((input_tokens or 0) / 1_000_000) * pricing["input_per_mtok"] + \
                   ((output_tokens or 0) / 1_000_000) * pricing["output_per_mtok"]
        
        return {
            "tags": tags or [],
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
            "cost_usd": round(cost, 6) if cost else None,
            "status": parse_status,
            "error": "Failed to parse JSON from response" if parse_status == "parse_error" else None
        }
    
    def _extract_grok(self, full_prompt: str) -> Dict[str, Any]:
        """Extract using Grok (xAI) - OpenAI-compatible API."""
        client = self.clients["grok"]
        params = MODELS["grok"]["params"]
        
        kwargs = {
            "model": MODELS["grok"]["name"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS["grok"]},
                {"role": "user", "content": full_prompt}
            ]
        }
        # Only add params if explicitly set (otherwise use provider defaults)
        if params.get("max_tokens") is not None:
            kwargs["max_tokens"] = params["max_tokens"]
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        
        response = client.chat.completions.create(**kwargs)
        
        raw = response.choices[0].message.content
        tags, parse_status = extract_json_list(raw)
        
        input_tokens = response.usage.prompt_tokens if response.usage else None
        output_tokens = response.usage.completion_tokens if response.usage else None
        
        cost = None
        if input_tokens and output_tokens:
            pricing = MODELS["grok"]["pricing"]
            cost = (input_tokens / 1_000_000) * pricing["input_per_mtok"] + \
                   (output_tokens / 1_000_000) * pricing["output_per_mtok"]
        
        return {
            "tags": tags or [],
            "raw_response": raw,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None,
            "cost_usd": round(cost, 6) if cost else None,
            "status": parse_status,
            "error": "Failed to parse JSON from response" if parse_status == "parse_error" else None
        }


# =============================================================================
# 🔒 6. DATA LOADING & PREPARATION
# =============================================================================

def load_venue_data(csv_path: str, sample_size: Optional[int] = None, random_seed: int = 42) -> pd.DataFrame:
    """
    Load venue data and prepare for extraction.
    
    Args:
        csv_path: Path to venues_data.csv
        sample_size: Optional number of venues to sample
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: id, name, google_reviews (as list of texts)
    
    Note: Ratings are explicitly excluded from the review data.
    """
    import ast
    
    df = pd.read_csv(csv_path)
    
    # Extract review texts from google_reviews column
    # IMPORTANT: Only extracts 'text' field, ratings are explicitly excluded
    def extract_review_texts(raw_val):
        if pd.isna(raw_val):
            return []
        reviews = raw_val
        if isinstance(raw_val, str):
            try:
                reviews = ast.literal_eval(raw_val)
            except Exception:
                return []
        if not isinstance(reviews, list):
            return []
        
        texts = []
        for review in reviews:
            if isinstance(review, dict):
                # Only extract text, explicitly ignore 'rating' field
                text = review.get('text', '')
                if text and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                # Note: review.get('rating') is present but intentionally ignored
        return texts
    
    df['google_reviews'] = df['google_reviews'].apply(extract_review_texts)
    
    # Filter to venues with reviews
    df = df[df['google_reviews'].apply(len) > 0].copy()
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed)
    
    # Select relevant columns
    cols = ['id', 'name', 'google_reviews']
    if 'place_description' in df.columns:
        cols.append('place_description')
    if 'googleMapsTags' in df.columns:
        cols.append('googleMapsTags')
    
    return df[cols].reset_index(drop=True)


# =============================================================================
# 🔒 7. EXPERIMENT RUNNER
# =============================================================================

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
            # Only skip exp_ids where status == "success" (rerun failed/parse_error)
            success_df = existing_df[existing_df['status'] == 'success']
            completed_exp_ids = set(success_df['exp_id'].unique())
            all_results = existing_df.to_dict('records')
            print(f"Resuming: Found {len(completed_exp_ids)} successful extractions in checkpoint")
            failed_count = len(existing_df) - len(success_df)
            if failed_count > 0:
                print(f"  Will rerun {failed_count} failed/parse_error extractions")
            if verbose:
                print(f"  Loaded {len(all_results)} existing rows")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}. Starting fresh.")
    
    extraction_count = 0
    
    # Initialize progress bar if verbose
    from tqdm import tqdm
    pbar = None
    if verbose:
        pbar = tqdm(
            total=total,
            desc="Extracting",
            unit="extraction",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        # Set initial progress for already completed items
        if completed_exp_ids:
            pbar.update(len(completed_exp_ids))
    
    try:
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
                            if pbar:
                                pbar.update(1)
                                pbar.set_postfix_str(f"⏭️  {venue_name[:30]} | {model} | {prompt_type} | run {run_num}")
                            continue
                        
                        if pbar:
                            pbar.set_postfix_str(f"🔄 {venue_name[:30]} | {model} | {prompt_type} | run {run_num}")
                        
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
                            "model": result.model,  # Full model name (e.g., "gpt-5-nano")
                            "model_key": model,  # Model key (e.g., "openai", "gemini", "claude", "grok")
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
                        
                        # Update progress bar with status
                        if pbar:
                            status_icon = {
                                "success": "✓",
                                "parse_error": "⚠",
                                "error": "✗"
                            }.get(result.status, "?")
                            
                            if result.status == "success":
                                cost_str = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "N/A"
                                status_msg = f"{status_icon} {len(result.tags)} tags | {cost_str} | {result.time_seconds:.1f}s"
                            elif result.status == "parse_error":
                                status_msg = f"{status_icon} Parse error"
                            else:
                                status_msg = f"{status_icon} Error: {result.error[:30] if result.error else 'Unknown'}"
                            
                            pbar.set_postfix_str(status_msg)
                            pbar.update(1)
                        
                        # Checkpoint periodically (atomic write)
                        if checkpoint_path and extraction_count % checkpoint_every == 0:
                            checkpoint_df = pd.DataFrame(all_results)
                            # Atomic write: write to temp file then rename
                            temp_path = Path(checkpoint_path).with_suffix('.tmp')
                            checkpoint_df.to_csv(temp_path, index=False)
                            temp_path.replace(checkpoint_path)
                            if pbar:
                                pbar.write(f"💾 Checkpoint saved ({len(all_results)} rows)")
    
    finally:
        if pbar:
            pbar.close()
    
    # Final checkpoint save (atomic write)
    if checkpoint_path:
        checkpoint_df = pd.DataFrame(all_results)
        temp_path = Path(checkpoint_path).with_suffix('.tmp')
        checkpoint_df.to_csv(temp_path, index=False)
        temp_path.replace(checkpoint_path)
        if verbose:
            print(f"💾 Final checkpoint saved: {checkpoint_path}")
    
    return pd.DataFrame(all_results)


# =============================================================================
# 🔒 8. VALIDATION & SANITY CHECKS
# =============================================================================

def validate_tags(tags_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run sanity checks on extracted tags.
    
    Returns dict with validation results.
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


def compute_jaccard_similarity(tags1: List[str], tags2: List[str], normalized: bool = True) -> float:
    """
    Compute Jaccard similarity between two tag lists.
    
    Args:
        tags1, tags2: Lists of tags
        normalized: If True, normalize tags before comparison
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
    
    group_cols = ["run_id", "exp_id", "venue_id", "model", "model_key", "prompt_type", "run_number"]
    
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
    
    # Attach tag counts and raw_tags_json
    import json as json_lib
    
    tag_counts = (
        tags_only.groupby(group_cols, dropna=False)
        .agg(
            n_tags=("tag_raw", "count"),
            n_unique_tag_eval=("tag_norm_eval", "nunique"),
        )
        .reset_index()
    )
    
    # Get raw_tags_json (stringified list of tags per extraction)
    def get_tags_json(group):
        """Get raw tags as JSON string for this extraction."""
        tags = group['tag_raw'].dropna().tolist()
        return json_lib.dumps(tags) if tags else None
    
    tags_json = (
        tags_only.groupby(group_cols, dropna=False)
        .apply(get_tags_json, include_groups=False)
        .reset_index(name='raw_tags_json')
    )
    
    extractions = extractions.merge(tag_counts, on=group_cols, how="left")
    extractions = extractions.merge(tags_json, on=group_cols, how="left")
    extractions["n_tags"] = extractions["n_tags"].fillna(0).astype(int)
    extractions["n_unique_tag_eval"] = extractions["n_unique_tag_eval"].fillna(0).astype(int)
    
    # Add tags_filtered_count (from first row of each extraction)
    filtered_counts = (
        df.groupby(group_cols, dropna=False)
        .agg(tags_filtered_count=("tags_filtered_count", "first"))
        .reset_index()
    )
    extractions = extractions.merge(filtered_counts, on=group_cols, how="left")
    
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


# =============================================================================
# 🔒 9. OUTPUT HELPERS
# =============================================================================

def save_results(tags_df: pd.DataFrame, output_dir: str = "results", prefix: str = "gentags") -> str:
    """Save results to CSV with timestamp."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{prefix}_{timestamp}.csv"
    
    tags_df.to_csv(filename, index=False)
    return str(filename)


def save_raw_response(raw_response: str, exp_id: str, run_id: str, output_dir: str = "results/raw") -> str:
    """Save raw model response to file (for debugging parse errors)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"{exp_id}_{run_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(raw_response or "")
    return str(filename)


def load_results(filepath: str) -> pd.DataFrame:
    """Load results from CSV."""
    return pd.read_csv(filepath)


# =============================================================================
# 🔒 10. VERSION INFO
# =============================================================================

def get_version_info() -> Dict[str, Any]:
    """Return version information for reproducibility."""
    return {
        "pipeline_version": "1.2",
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "system_prompt_hash": SYSTEM_PROMPT_HASH,
        "model_version": MODEL_VERSION,
        "models": {k: v["name"] for k, v in MODELS.items()},
        "model_params": {k: v["params"] for k, v in MODELS.items()},
        "prompts": list(PROMPTS.keys()),
        "constraints": {
            "max_tag_words": MAX_TAG_WORDS,
            "max_tags_per_extraction": MAX_TAGS_PER_EXTRACTION
        },
        "frozen_date": "2025-01-17"
    }


# =============================================================================
# BACKWARD COMPATIBILITY ALIAS
# =============================================================================

# Alias for backward compatibility with old typo
GenttagExtractor = GentagExtractor


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("GENTAGS Pipeline v1.2")
    print("=" * 50)
    info = get_version_info()
    print(f"Prompts: {info['prompts']}")
    print(f"Models: {info['models']}")
    print(f"Prompt hash: {info['prompt_hash']}")
    print(f"System prompt hash: {info['system_prompt_hash']}")
    print(f"Model params: {info['model_params']}")
    print(f"Constraints: {info['constraints']}")
    print("=" * 50)
    print("\nUsage:")
    print("  from gentags_pipeline import GentagExtractor, run_experiment, load_venue_data")
