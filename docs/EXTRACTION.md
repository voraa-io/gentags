# Gentags: Extraction

> **Paper section:** Section 3 (Method — Extraction)
> **Phase:** Phase 1
> **Status:** Complete
> **Last updated:** 2026-03-06

---

## 1. Definition

A **gentag** is a short, evidence-conditioned semantic unit extracted from natural language text by a large language model. Gentags are not drawn from a predefined ontology or label set; they are generated freely by the model in response to the source text.

Each gentag is:
- **Short** — typically 1-4 words (76.8% are 2 words; 17.0% are 3 words)
- **Evidence-conditioned** — grounded in the provided source text, not hallucinated
- **Externalized** — stored as persistent state outside the model
- **Inspectable** — readable strings, not dense vectors

### 1.1 Semantic Taxonomy

Empirical analysis of 975 unique gentags across 50 venues reveals three distinct categories:

| Category | Proportion | Definition | Examples |
|----------|-----------|------------|----------|
| **Descriptive attributes** | 60.0% | Describes a feature, activity, or characteristic | "outdoor seating", "live music", "big screen sport", "watching sport" |
| **Evaluative propositions** | 24.5% | Asserts a quality judgment | "amazing food", "slow service", "overpriced", "friendly" |
| **Entity mentions** | 11.4% | Names a specific item or offering | "brisket", "margarita", "bacon burger" |
| **Mixed (evaluative + entity)** | 4.1% | Evaluates a specific entity | "delicious pizza", "bad pizza", "crispy fry" |

Each gentag contributes a unit to the semantic state of the represented entity. Evaluative propositions (24.5%) and mixed units (4.1%) directly assert evaluable claims. Descriptive attributes (60.0%) implicitly assert "this entity is characterized by [attribute]." Entity mentions (11.4%) assert presence. All three categories function as evidence-conditioned semantic state.

### 1.2 Word Count Distribution

| Words | Count | Proportion |
|-------|-------|------------|
| 1 | 48 | 4.0% |
| 2 | 919 | 76.8% |
| 3 | 204 | 17.0% |
| 4 | 26 | 2.2% |
| **Total** | **1,197** | |

The modal form is a two-word adjective-noun or noun-noun phrase. Single-word units are either evaluative adjectives ("noisy", "overpriced") or entity nouns ("brisket", "paella").

### 1.3 Contrast with Keyword Baselines

Gentags differ from lexical keyword methods (RAKE, YAKE, TF-IDF) in what they produce:

| Aspect | Gentags | Keywords (RAKE/YAKE/TF-IDF) |
|--------|---------|----------------------------|
| **Generation** | Model-synthesized | Statistically extracted from surface text |
| **Semantic clarity** | "fast service", "game-day vibe" | "relative quick time", "fashioned regional mexican music" |
| **Granularity** | Atomized units | Variable-length fragments |
| **Grounding** | Evidence-conditioned (prompted) | Frequency/co-occurrence based |
| **Coverage** | Synthesizes across evidence | Reflects dominant surface patterns |

**Concrete example (Colton's Arcadia):**

Gentags:
> "amazing atmosphere", "fast service", "favorite sport bar", "game-day vibe", "delicious food", "coordination issue", "delivery delay"

RAKE:
> "fashioned regional mexican music", "dishes without coordinating first", "really cool vibe", "also delicious —", "watching sports", "several times", "screens everywhere"

YAKE:
> "Excellent place to watch", "place to watch", "watch the games", "games", "service", "atmosphere", "amazing"

TF-IDF:
> "service", "amazing", "atmosphere", "delicious", "pizza", "time", "attentive"

### 1.4 What Gentags Are Not

- Not a belief state (no probabilities, no uncertainty quantification)
- Not an ontology (no predefined categories or schema)
- Not a retrieval index
- Not a summarization method (they atomize rather than compress)

---

## 2. Extraction Protocol

### 2.1 Overview

Gentags are extracted by prompting a large language model with venue reviews and requesting short semantic tags. No predefined label set, ontology, or examples are provided. The model freely generates tags conditioned on the evidence. Output is constrained to a JSON list. Tags exceeding 4 words are filtered post-hoc.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Source data | 553 venues, each with 1-20 user reviews |
| Models | 4 (see Section 2.4) |
| Prompts | 3 variants (see Section 2.2) |
| Runs per configuration | 2 |
| Total extractions | 13,272 (553 x 4 x 3 x 2) |
| Total tags extracted | ~265,842 |
| Max tag words | 4 (enforced post-hoc) |
| Max tags per extraction | None (observe natural model behavior) |
| Temperature | Provider default (not overridden) |
| Total cost | $12.61 |
| Completion rate | 100% |
| Pipeline version | 1.2 (prompts frozen at v1.0) |
| Source file | `src/gentags/pipeline.py` |

### 2.2 Extraction Prompts (Frozen v1.0)

Three prompt variants are used. All prompts are frozen and hashed for reproducibility (`PROMPT_HASH` = MD5 of serialized prompt dict).

**Minimal:**
```
Extract semantic tags ("gentags") for this venue based on the reviews.
A gentag is a short, meaningful semantic phrase (typically 1–4 words) that
captures one idea expressed or strongly implied in the reviews.
Include any gentags that describe atmosphere, food, service, vibe, crowd,
or typical occasions mentioned in the reviews.
Do not invent information beyond what the reviews support.
Return only a JSON list of gentags.
```

**Anti-hallucination:**
```
Extract semantic tags ("gentags") for this venue based ONLY on what is
explicitly stated or clearly implied in the reviews.
A gentag is a short, meaningful semantic phrase (typically 1–4 words) that
captures a single idea grounded in the review text. It must not be a full
sentence.
Do NOT infer, assume, generalize, or guess any information that is not
directly supported by the reviews.
If a concept is uncertain, ambiguous, or weakly implied, do NOT include it
as a gentag.
Include only gentags that reflect concrete statements in the reviews.
Return only a JSON list of gentags.
```

**Short phrase:**
```
Extract semantic tags ("gentags") for this venue that summarize the key
ideas expressed in the reviews.
A gentag must be a short phrase of 1–4 words that represents one clear
semantic idea.
Do not produce full sentences.
Tags must be grounded in the content of the reviews and should not rely on
assumptions or outside knowledge.
Return only a JSON list of short gentags.
```

### 2.3 System Prompts

| Provider | System Prompt |
|----------|--------------|
| OpenAI | `"You extract only JSON lists of gentags based on reviews. No explanations."` |
| Grok | `"You extract only JSON lists of gentags based on reviews. No explanations."` |
| Gemini | None (user prompt only) |
| Claude | None (user prompt only) |

### 2.4 Models

| Key | Model | Provider | Input $/Mtok | Output $/Mtok |
|-----|-------|----------|-------------|---------------|
| openai | `gpt-5-nano` | OpenAI | $0.05 | $0.40 |
| gemini | `gemini-2.5-flash` | Google | $0.25 | $0.50 |
| claude | `claude-sonnet-4-5` | Anthropic | $3.00 | $15.00 |
| grok | `grok-4` | xAI | $2.00 | $10.00 |

All models use provider-default temperature and max_tokens (except Claude, which requires explicit max_tokens; 8192 used as fallback).

### 2.5 Prompt Behavior

| Prompt | Behavior | Effect |
|--------|----------|--------|
| `minimal` | Balanced extraction | Moderate tag count, general coverage |
| `anti_hallucination` | Emphasizes grounding | More tags, higher granularity, explicit grounding instruction |
| `short_phrase` | Emphasizes brevity | Fewer tags, more compressed forms |

Prompt variant affects tag count and surface style but not core semantic content (see `docs/STABILITY.md`, Section S2).

### 2.6 Output Parsing

Model output is parsed as a JSON list with three fallback strategies:
1. Direct `json.loads()` on raw response
2. Markdown stripping (remove ` ```json ``` ` wrappers)
3. Bracket-balance matching (extract first `[...]` from response)

Tags exceeding `MAX_TAG_WORDS=4` are filtered post-hoc and recorded in `tags_filtered_out`.

Each extraction produces an `ExtractionResult` dataclass containing: tags, filtered tags, input metadata (review count, char count, prompt hash), extraction metadata (timing, tokens, cost, status), and version tracking (prompt version, model version, pipeline version).

### 2.7 Quality Filtering

| Stage | Count |
|-------|-------|
| Total extractions | 13,272 |
| Error extractions removed | 2,898 (21.8%) |
| Venues with all 4 models successful | 230 |
| Final extractions for analysis | 5,517 |
| Final tag rows | 118,832 |

**Errors by model:**

| Model | Errors | Error Rate |
|-------|--------|------------|
| Claude | 1,697 | 51.1% |
| Grok | 1,201 | 36.2% |
| OpenAI | 0 | 0% |
| Gemini | 0 | 0% |

Analysis was restricted to 230 venues with successful extractions from all 4 models to ensure fair cross-model comparison.

### 2.8 Cost and Scalability

Gentag extraction is cheap ($12.61 for 13,272 extractions), scalable (100% completion on non-error runs), and model-agnostic (4 models produce comparable semantic content — see `docs/STABILITY.md`).

---

## File References

- Pipeline source: `src/gentags/pipeline.py`
- Extraction results: `results/phase1/*_tags.csv`, `*_extractions.csv`
- Manifests: `results/phase1/meta/*_manifest.json`
- Data: `data/study1_venues_20250117.csv` (553 venues)
