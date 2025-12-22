# Frozen Prompts

This document contains the exact prompt texts used in Study 1, along with their hashes for reproducibility.

## Prompt Version

**Version:** 1.0  
**Hash:** See individual prompts below  
**Frozen Date:** 2025-01-17

## Extraction Prompts

### Minimal Prompt

```
Extract semantic tags ("gentags") for this venue based on the reviews.
A gentag is a short, meaningful semantic phrase (typically 1–4 words) that captures one idea expressed or strongly implied in the reviews.
Include any gentags that describe atmosphere, food, service, vibe, crowd, or typical occasions mentioned in the reviews.
Do not invent information beyond what the reviews support.
Return only a JSON list of gentags.
```

**Hash:** [Computed in config.py]

### Anti-Hallucination Prompt

```
Extract semantic tags ("gentags") for this venue based ONLY on what is explicitly stated or clearly implied in the reviews.
A gentag is a short, meaningful semantic phrase (typically 1–4 words) that captures a single idea grounded in the review text. It must not be a full sentence.
Do NOT infer, assume, generalize, or guess any information that is not directly supported by the reviews.
If a concept is uncertain, ambiguous, or weakly implied, do NOT include it as a gentag.
Include only gentags that reflect concrete statements in the reviews.
Return only a JSON list of gentags.
```

**Hash:** [Computed in config.py]

### Short Phrase Prompt

```
Extract semantic tags ("gentags") for this venue that summarize the key ideas expressed in the reviews.
A gentag must be a short phrase of 1–4 words that represents one clear semantic idea.
Do not produce full sentences.
Tags must be grounded in the content of the reviews and should not rely on assumptions or outside knowledge.
Return only a JSON list of short gentags.
```

**Hash:** [Computed in config.py]

## System Prompts

### OpenAI / Grok

```
You extract only JSON lists of gentags based on reviews. No explanations.
```

### Gemini / Claude

No system prompt (uses user prompt only).

## Hash Computation

Prompts are hashed using MD5 and truncated to 8 characters for compact representation. The full hash is computed in `src/gentags/config.py` and included in all extraction results for reproducibility.

## Reproducibility

All prompts are frozen for Study 1. Any changes require:

1. Creating a new study version
2. Updating version numbers
3. Documenting the rationale for changes
