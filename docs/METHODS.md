# Methods

This section describes the data, models, prompts, extraction pipeline, and evaluation procedures used in **Study 1**. All methodological choices were frozen prior to large-scale extraction to ensure reproducibility and prevent post-hoc optimization.

---

## 1. Study Design Overview

Study 1 is a **descriptive, comparative study** that analyzes how large language models extract short-form semantic metadata (“gentags”) from sparse venue reviews.

The study evaluates:

- Cross-model agreement
- Prompt sensitivity
- Run-to-run stability
- Semantic coverage and variability

The study does **not** evaluate downstream recommendation accuracy or user behavior.

---

## 2. Data Sources

### 2.1 Venue Reviews

Venue data was collected from Google Maps and includes:

- Venue identifier and name
- A list of user review texts

**Only review text was used.**  
The following were explicitly excluded:

- Ratings or star scores
- Reviewer metadata
- Timestamps
- Review likes or engagement signals

This design choice avoids sentiment leakage and rating-induced bias.

### 2.2 Inclusion Criteria

- Minimum number of reviews per venue: **1**
- No upper bound on number of reviews
- Language: English

No additional filtering or weighting was applied.

---

## 3. Gentag Definition

A **gentag** is defined as:

> A short (1–4 words), atomized semantic phrase extracted or synthesized by a large language model that represents a single interpretable semantic constraint expressed or implied in venue reviews.

Gentags are designed to function as:

- A machine-generated folksonomy
- Human-readable semantic metadata
- Embed-friendly units for downstream retrieval

Gentags are **not** ratings, sentiment labels, categories, or ontology-aligned attributes.

This definition is fixed for Study 1.

---

## 4. Language Models

The following models were evaluated:

- **OpenAI:** gpt-5-nano
- **Google:** gemini-2.5-flash
- **Anthropic:** claude-sonnet-4-5
- **xAI:** grok-4

All models were used **as-is**, without:

- Temperature tuning
- Top-p tuning
- Decoding parameter optimization
- Fine-tuning or adapters

Provider defaults were used wherever applicable. Models were treated as black-box generators.

---

## 5. Prompt Design

Three zero-shot prompts were used:

1. **Minimal Extraction Prompt**  
   Designed to observe unconstrained emergent semantics.

2. **Strict Anti-Hallucination Prompt**  
   Designed to limit inferred or weakly grounded tags.

3. **Short-Phrase Constraint Prompt**  
   Designed to enforce compact, embedding-friendly output.

All prompts:

- Contain no examples
- Contain no predefined schema or ontology
- Require output as a JSON list of strings only

Prompt texts and hashes were versioned and frozen prior to experimentation.

---

## 6. Extraction Pipeline

For each venue, gentags were extracted using all combinations of:

- Model
- Prompt
- Run number

Each configuration was executed multiple times to measure run-to-run variability.

### 6.1 Output Parsing

Model outputs were parsed into JSON lists using a multi-stage parser:

1. Direct JSON parsing
2. Markdown block stripping
3. Balanced bracket extraction

Each extraction was assigned a status:

- `success`
- `parse_error`
- `error`

This status reflects **format validity only**, not semantic correctness.

---

## 7. Constraints and Filtering

Post-extraction constraints:

- Tags must contain **1–4 words**
- Tags exceeding length constraints were filtered and logged
- No deduplication was applied across models or runs
- No semantic merging was performed

All raw outputs were preserved for analysis.

---

## 8. Normalization (Evaluation Only)

Two normalization schemes were applied **only for evaluation metrics**:

- **Light normalization:** lowercasing and whitespace cleanup
- **Evaluation normalization:** plural reduction and removal of non-semantic prefixes

Normalization was never applied during extraction.

---

## 9. Evaluation Metrics

Study 1 evaluates gentags using the following metrics:

### 9.1 Stability

- Run-to-run Jaccard similarity
- Within-model variability

### 9.2 Model Agreement

- Cross-model tag overlap
- Shared vs unique tag analysis

### 9.3 Prompt Sensitivity

- Tag count differences
- Semantic specificity variation

### 9.4 Distributional Properties

- Tag length distribution
- Vocabulary diversity
- Frequency of contradictions

### 9.5 Cost and Performance

- Latency per extraction
- Token usage
- Estimated API cost

No downstream task metrics were evaluated.

---

## 10. Reproducibility

Each extraction records:

- Model identifier
- Prompt type and hash
- System prompt hash
- Pipeline version
- Timestamp and run ID

All experiments are fully reproducible given API access.

---

## 11. Limitations

Study 1 does not evaluate:

- Recommendation accuracy
- User satisfaction
- Real-world behavioral impact

These are addressed in future work.

---

## Summary

This methodology isolates the behavior of LLMs as semantic extractors under sparse data conditions, enabling a controlled analysis of emergent machine-generated folksonomies without conflating extraction quality with downstream system performance.
