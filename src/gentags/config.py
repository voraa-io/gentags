"""
Configuration: PROMPTS, MODELS, hashes, frozen versions.

All prompts and models are FROZEN for Study 1.
"""

import json
import hashlib

# =============================================================================
# GENTAG DEFINITION (FROZEN)
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
# PROMPTS (FROZEN - v1.0)
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
# MODELS (FROZEN - v1.0)
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
# VERSION INFO
# =============================================================================

def get_version_info():
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

