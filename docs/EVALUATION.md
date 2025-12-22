# Evaluation

This section evaluates the properties of gentags produced in Study 1.  
The evaluation focuses on **stability, agreement, sensitivity, semantic coverage, and information retention**, rather than downstream recommendation performance.

---

## 1. Evaluation Overview

Gentags are intended to function as a **compressed semantic representation** of venue reviews.  
Accordingly, evaluation focuses on the following research questions:

- Do gentags remain stable across runs?
- Do different models agree on extracted semantics?
- How sensitive are gentags to prompt constraints?
- How much semantic information is retained after compression?
- What are the distributional properties of the resulting folksonomy?

No evaluation in this study measures recommendation accuracy, ranking performance, or user behavior.

---

## 2. Stability Analysis (Run-to-Run Consistency)

To assess extraction stability, each model–prompt–venue configuration was executed multiple times.

### Metric

- **Jaccard similarity** between tag sets across runs
- Computed on lightly normalized tags

### Analysis

- High similarity indicates deterministic or stable extraction
- Low similarity indicates stochastic variation or semantic ambiguity in reviews

This analysis quantifies how reliably a model extracts gentags from the same input under identical conditions.

---

## 3. Cross-Model Agreement

To evaluate semantic agreement across models, gentags extracted by different LLMs from the same venue and prompt were compared.

### Metrics

- Pairwise Jaccard similarity
- Shared vs unique tag counts
- Intersection size across 2+ models

### Interpretation

- Shared tags indicate robust semantic signals
- Model-specific tags reveal differences in abstraction, specificity, or world knowledge priors

This analysis helps identify which semantic attributes consistently emerge independent of model choice.

---

## 4. Prompt Sensitivity Analysis

Gentags were compared across different prompt formulations for the same model and venue.

### Dimensions Analyzed

- Number of tags extracted
- Average tag length
- Semantic specificity
- Degree of abstraction

### Purpose

This analysis measures how prompt constraints shape the emergent folksonomy and whether certain prompts bias toward concreteness, conservatism, or verbosity.

---

## 5. Distributional Properties of Gentags

We analyze global properties of the extracted gentag corpus, including:

- Tag length distribution
- Vocabulary size and diversity
- Frequency of repeated tags
- Occurrence of contradictory tags (e.g., “slow service” and “quick service”)

Contradictions are not treated as errors, as they often reflect differing reviewer experiences.

---

## 6. Semantic Reconstruction Evaluation (Information Retention)

Gentags act as a **lossy semantic compression** of review text.  
To assess how much semantic information is retained, we conduct a **semantic reconstruction evaluation**.

### 6.1 Procedure

1. Original reviews are compressed into gentags using the extraction pipeline.
2. A language model is prompted to generate a natural-language description of the venue **using only the gentags** as input.
3. The original reviews are withheld during reconstruction.

The reconstruction model is not fine-tuned or optimized for this task.

---

### 6.2 Metrics

Reconstructed text is compared to the original reviews using:

- **Embedding similarity** (semantic proximity)
- **BERTScore** (semantic overlap)
- **Human judgment** of semantic fidelity

Evaluation focuses on preservation of:

- Atmosphere
- Service patterns
- Food characteristics
- Use-case suitability

Exact textual recovery is neither expected nor required.

---

### 6.3 Interpretation

This evaluation measures **semantic information retention**, not Shannon information.

A successful reconstruction:

- Captures high-level meaning
- Preserves decision-relevant attributes
- Remains interpretable to humans

This test functions as a **diagnostic probe** of representational sufficiency, not as a decoding objective.

---

## 7. Cost and Performance Analysis

For each extraction, we record:

- Latency
- Token usage
- Estimated API cost

This allows comparison of semantic yield relative to computational expense across models.

---

## 8. Summary of Evaluation Scope

Study 1 evaluation establishes:

- Whether gentags are stable
- Whether they generalize across models
- Whether they retain decision-relevant semantics
- Whether they form a coherent emergent folksonomy

These results motivate future work on synthetic gentags and downstream recommendation systems.

---

## 9. Limitations

This evaluation does not assess:

- User satisfaction
- Recommendation accuracy
- Behavioral outcomes
- Long-term learning dynamics

These are addressed in future studies.
