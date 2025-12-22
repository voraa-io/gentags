# STUDY 1 LOCK — Methodological Commitments

This document defines the **frozen methodological configuration** for **Study 1** of the Gentags project.

All definitions, prompts, models, and constraints described here are **fixed** for Study 1 and MUST NOT be modified after data collection begins.

Any future changes belong to **Study 2+** and will be explicitly versioned.

---

## 1. Scope of Study 1

Study 1 evaluates the feasibility and behavior of **machine-generated folksonomies** (gentags) extracted from sparse venue reviews using large language models.

**Study 1 does NOT evaluate:**

- Recommendation accuracy
- User engagement
- Downstream ranking models
- UX or product outcomes

The goal is **semantic extraction analysis**, not system performance optimization.

---

## 2. Definition of Gentags (Frozen)

A **gentag** is defined as:

> A short (1–4 words), atomized semantic phrase extracted or synthesized by a large language model that represents a single interpretable semantic constraint expressed or implied in venue reviews.

### Required Properties

- Atomized (one dominant semantic idea)
- Short (1–4 words)
- Composable with other gentags
- Human-readable and interpretable
- Embed-friendly (suitable for vector similarity)
- Emergent (not drawn from predefined categories)

### Explicit Exclusions

Gentags are NOT:

- Ratings or numerical scores
- Sentiment labels
- Ontology-aligned attributes
- Predefined schema fields
- Summaries or descriptions

This definition is **immutable** for Study 1.

---

## 3. Model Usage Policy: "As-Is Defaults"

**CRITICAL:** We do not set temperature, top_p, seed, frequency_penalty, presence_penalty, or any other sampling parameters.

We call each provider with **provider defaults only**.

### Explicit Policy

- **Temperature:** `None` (not passed, provider default)
- **Top-p / Top-k:** Not passed (provider default)
- **Max tokens:** `None` for OpenAI/Gemini/Grok (not passed, provider default)
- **Max tokens for Claude:** `8192` (required by API, set high to not constrain output - this is still "as-is" sampling, not tuning)
- **All other parameters:** Not passed (provider default)

### Rationale

This ensures:

- Fair comparison across models (all use defaults)
- No post-hoc optimization
- Reproducibility (defaults are stable)
- Scientific validity (no cherry-picking)

### Model-Specific Notes

- **OpenAI (gpt-5-nano):** No parameters passed except model name and messages
- **Gemini (gemini-2.5-flash):** No parameters passed except model name and contents
- **Claude (claude-sonnet-4-5):** Only `max_tokens=8192` passed (API requirement, not a tuning choice)
- **Grok (grok-4):** No parameters passed except model name and messages

---

## 4. Data Inputs (Frozen)

### Review Data

- Source: Google Maps reviews
- Content used: **text only**
- Explicitly excluded: ratings, likes, timestamps, reviewer metadata

### Venue Inclusion

- Minimum reviews: 1
- No upper bound on number of reviews
- Language: English only

No additional preprocessing, filtering, or weighting is applied.

---

## 5. Prompts (Frozen)

Three prompts are used:

1. **Minimal Extraction Prompt**
2. **Strict Anti-Hallucination Prompt**
3. **Short-Phrase Constraint Prompt**

All prompts:

- Are zero-shot
- Contain no examples
- Contain no ontology hints
- Require JSON list output only

Prompt texts and hashes are recorded and versioned in the extraction pipeline.

Prompts MUST NOT be altered during Study 1.

See `docs/PROMPTS.md` for full prompt texts.

---

## 6. Models (Frozen)

Study 1 evaluates the following models **as-is**:

- OpenAI: `gpt-5-nano`
- Google: `gemini-2.5-flash`
- Anthropic: `claude-sonnet-4-5`
- xAI: `grok-4`

Models are treated as **black boxes** with provider defaults.

---

## 7. Extraction Constraints (Frozen)

- Tag length: **1–4 words**
- No hard cap on number of tags per extraction
- Tags exceeding length constraints are filtered and logged
- No deduplication across models or runs
- No semantic normalization prior to extraction

Light normalization is applied **only for evaluation metrics**, not for tag generation.

---

## 8. Output Validity Definition

The `status` field reflects **format validity only**:

- `success`: Valid JSON list parsed
- `parse_error`: Output could not be parsed as JSON
- `error`: API or execution failure

**Semantic correctness, grounding, or hallucination-free status is NOT implied.**

---

## 9. Evaluation Dimensions (Study 1)

Study 1 evaluates:

- **Stability:** run-to-run consistency
- **Model agreement:** cross-model overlap
- **Prompt sensitivity:** effect of prompt constraints
- **Tag distribution:** length, diversity, specificity
- **Cost & performance:** latency and token usage

No downstream task performance is evaluated.

---

## 10. Reproducibility Guarantees

Each extraction records:

- Prompt hash
- System prompt hash
- Model identifier and model_key
- Pipeline version
- Timestamp and run ID

This ensures full experimental traceability.

---

## 11. Change Policy

After Study 1 begins:

- ❌ Prompts may NOT change
- ❌ Models may NOT change
- ❌ Constraints may NOT change
- ❌ Definitions may NOT change
- ❌ Model parameters may NOT be tuned

Any modification requires:

- New pipeline version
- New study designation (e.g., Study 2)

---

## Status

- Locked on: **2025-01-17**
- Pipeline version: **v1.2**
- Applicable to: **Study 1 only**
