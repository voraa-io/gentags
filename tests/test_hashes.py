"""
Unit tests for hash stability (prompts and system prompts).
No API keys required.
"""

import pytest
from gentags.config import PROMPT_HASH, SYSTEM_PROMPT_HASH, PROMPTS, SYSTEM_PROMPTS, PROMPT_VERSION, MODEL_VERSION
import hashlib
import json


def test_prompt_hash_stable():
    """Test that PROMPT_HASH matches expected computation."""
    # Recompute hash the same way as config.py
    prompt_str = json.dumps(PROMPTS, sort_keys=True)
    expected_hash = hashlib.md5(prompt_str.encode()).hexdigest()[:8]
    
    assert PROMPT_HASH == expected_hash, \
        f"PROMPT_HASH mismatch. Expected {expected_hash}, got {PROMPT_HASH}"


def test_system_prompt_hash_stable():
    """Test that SYSTEM_PROMPT_HASH matches expected computation."""
    # Recompute hash the same way as config.py
    system_prompt_str = json.dumps(SYSTEM_PROMPTS, sort_keys=True)
    expected_hash = hashlib.md5(system_prompt_str.encode()).hexdigest()[:8]
    
    assert SYSTEM_PROMPT_HASH == expected_hash, \
        f"SYSTEM_PROMPT_HASH mismatch. Expected {expected_hash}, got {SYSTEM_PROMPT_HASH}"


def test_prompt_version_frozen():
    """Test that PROMPT_VERSION is set."""
    assert PROMPT_VERSION == "1.0", "PROMPT_VERSION should be 1.0 for Study 1"


def test_model_version_frozen():
    """Test that MODEL_VERSION is set."""
    assert MODEL_VERSION == "1.0", "MODEL_VERSION should be 1.0 for Study 1"


def test_prompts_exist():
    """Test that all expected prompts exist."""
    expected_prompts = ["minimal", "anti_hallucination", "short_phrase"]
    assert set(PROMPTS.keys()) == set(expected_prompts), \
        f"Expected prompts {expected_prompts}, got {list(PROMPTS.keys())}"


def test_system_prompts_exist():
    """Test that system prompts are defined for all providers."""
    expected_providers = ["openai", "gemini", "claude", "grok"]
    assert set(SYSTEM_PROMPTS.keys()) == set(expected_providers), \
        f"Expected providers {expected_providers}, got {list(SYSTEM_PROMPTS.keys())}"

