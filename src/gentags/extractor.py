"""
Main extraction class: GentagExtractor + provider clients.
"""

import os
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import uuid4

from .config import PROMPTS, SYSTEM_PROMPTS, MODELS, MAX_TAG_WORDS, MAX_TAGS_PER_EXTRACTION
from .schema import ExtractionResult
from .parsing import extract_json_list
from .normalize import filter_valid_tags


def get_venue_id(venue_name: str) -> str:
    """Generate a short venue_id from venue name."""
    import re
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', venue_name)
    words = clean.split()
    if len(words) == 1:
        return words[0][:8].upper()
    elif len(words) >= 2:
        return (words[0][:4] + "_" + words[1][:4]).upper()
    return clean[:8].upper() if clean else "VENUE"


def generate_exp_id(venue_id: str, model_key: str, prompt_type: str, run_number: int) -> str:
    """Generate experiment ID: e.g., BOU_gpt5_minimal_run1"""
    model_short = MODELS[model_key]["short"]
    prompt_short = {"minimal": "minimal", "anti_hallucination": "anti", "short_phrase": "short"}.get(prompt_type, prompt_type[:5])
    return f"{venue_id}_{model_short}_{prompt_short}_run{run_number}"


def generate_run_id() -> str:
    """Generate unique run ID (timestamp + pid + random suffix for concurrency safety)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{os.getpid()}_{uuid4().hex[:6]}"


class GentagExtractor:
    """
    Main extraction class. Handles all model providers.
    
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
            model: Key from MODELS dict ("openai", "gemini", "claude", "grok")
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
        
        # Claude requires max_tokens - use large default if not specified
        kwargs = {
            "model": MODELS["claude"]["name"],
            "max_tokens": params.get("max_tokens") or 8192,
            "messages": [{"role": "user", "content": full_prompt}]
        }
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        
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

