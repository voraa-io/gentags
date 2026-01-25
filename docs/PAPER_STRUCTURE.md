# Paper Structure: Gentags as Semantic Belief Representations

**Core Principle:**
> Gentags without pdensity are publishable. Gentags with pdensity are differentiated.

---

## What This Paper IS

A **systems + representation paper** showing:

1. LLMs can externalize latent semantics as discrete, inspectable tags
2. These representations are semantically stable despite lexical variation
3. Variability correlates with evidence sparsity (uncertainty signal)
4. Multiple models agree on extracted semantics

**This is already publishable.** No pdensity required.

---

## What This Paper is NOT

- ❌ A retrieval benchmark paper
- ❌ A recommender systems paper
- ❌ A user study paper
- ❌ A theoretical information theory paper

---

## Contribution Hierarchy

### Tier 1: Core Contributions (Required for Publication)

1. **Gentags** — A new primitive
   - Machine-generated folksonomy-like semantic units
   - Zero-shot extraction from LLMs
   - Reproducible pipeline across 4 models × 3 prompts

2. **Semantic Stability** — The key claim
   - "Lexically unstable but semantically stable"
   - Gap of 0.504 between cosine (0.977) and Jaccard (0.471)
   - Proves surface variation ≠ semantic variation

3. **Evidence-Sensitive Variability** — The S4 result
   - Correlation = -0.230 (negative)
   - More evidence → less variability
   - Representation variability as proxy for epistemic insufficiency

### Tier 2: Supporting Evidence (Strengthens the Paper)

4. **Model Agreement** — Cross-model validation
   - 4 different LLMs produce semantically similar outputs
   - Gentags reflect shared linguistic priors, not model artifacts

5. **Prompt Robustness** — Practical reliability
   - Different prompts change style/granularity, not meaning
   - Core semantics preserved across prompt variations

6. **Retention** — Meaning preservation
   - Gentags capture review meaning (+0.164 above random)
   - Not arbitrary text fragments

### Tier 3: Differentiation (Phase 3 + Exploratory)

7. **State Observability / Localization** (Phase 3)
   - Gentags enable localized change attribution
   - Embeddings produce diffuse, unattributable change
   - Model-in-the-loop lacks persistent state
   - Supports state observability for downstream decision systems

8. **Propositional Density (pdensity)** — Interpretive Construct
   - Explanatory concept for semantic constraint strength
   - Not directly measurable, not central contribution
   - Provides intuition for why certain gentags collapse semantic space more aggressively

---

## Paper Outline

### Abstract
LLMs can produce stable, compressed semantic representations (gentags) that behave like folksonomies under perturbation. We show these representations are lexically variable but semantically stable, and that representation variability correlates with evidence sparsity—providing a proxy for epistemic insufficiency. We introduce propositional density (pdensity) as an explanatory construct for understanding semantic constraint strength.

### 1. Introduction
- Problem: How to represent venue semantics for systems requiring persistent semantic state?
- Challenge: Raw text is expensive; embeddings are opaque; model-in-the-loop lacks persistence
- Contribution: Gentags as factorized, inspectable, persistent semantic state

### 2. Related Work
- Aspect extraction
- Keyphrase generation
- Synthetic captions
- Folksonomy research
- LLM-based summarization

### 3. Method: Gentag Extraction
- Pipeline description
- Prompt design (3 variants)
- Multi-model setup (4 LLMs)
- Normalization and deduplication

### 4. Experiments

#### 4.1 Semantic Stability (Phase 2 - S1)
- Run-to-run stability
- Key result: Cosine 0.977, Jaccard 0.471, Gap 0.504
- Claim validated: "Lexically unstable but semantically stable"

#### 4.2 Prompt and Model Sensitivity (Phase 2 - S2, S3)
- Cross-prompt comparison
- Cross-model comparison
- Result: Style varies, meaning stable

#### 4.3 Variability and Evidence Sparsity (Phase 2 - S4)
- Token count vs. variability correlation
- Result: -0.230 (negative)
- Interpretation: Representation variability as proxy for epistemic insufficiency

#### 4.4 Retention Analysis
- Comparison to random baseline
- Result: +0.164 above random

#### 4.5 State Observability (Phase 3)
- Localization experiment
- Gentags: high Gini (localized change)
- Embeddings: low Gini (diffuse change)
- Model-in-the-loop: no persistent state (31.6% stability)
- Note: Facets introduced solely for evaluation, not part of gentag representation

### 5. Discussion
- Implications for systems requiring persistent semantic state
- State observability for downstream decision systems
- Limitations
- When NOT to use gentags

### 6. Conclusion
- Gentags are a valid semantic representation
- Variability is signal, not noise
- Future: Control applications, pdensity exploration

### Appendix (Optional)
- A. Propositional Density: Exploratory Analysis
- B. Full prompt templates
- C. Per-model detailed results

---

## What We've Completed

| Component | Status | Phase |
|-----------|--------|-------|
| Gentag extraction pipeline | ✅ | Phase 1 |
| Semantic stability (S1) | ✅ | Phase 2 |
| Prompt sensitivity (S2) | ✅ | Phase 2 |
| Model sensitivity (S3) | ✅ | Phase 2 |
| Sparsity/uncertainty (S4) | ✅ | Phase 2 |
| Retention analysis | ✅ | Phase 2 |
| Stability report | ✅ | Phase 2 |
| All plots | ✅ | Phase 2 |
| **Localization (Block G)** | ✅ | Phase 3 |
| **Cost comparison (Block H)** | ✅ | Phase 3 |
| **Cold-start (Block I)** | ✅ | Phase 3 |
| **Model-in-loop baseline** | ✅ | Phase 3 |
| pdensity (exploratory) | 🔮 | Future |

### Phase 3 Key Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Gentag Gini (localization) | **0.657** | > 0.5 | ✅ |
| Embedding Gini | **0.361** | < 0.5 | ✅ |
| Gini difference | **+0.297** | > 0.2 | ✅ |
| % gentag more localized | **90.1%** | > 80% | ✅ |
| Cold-start correlation | **-0.317** | < 0 | ✅ |

---

## Key Metrics (Paper-Ready)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Cosine (semantic stability) | **0.977** | > 0.9 | ✅ |
| Jaccard (surface) | **0.471** | 0.3-0.6 | ✅ |
| Gap (cosine - jaccard) | **0.504** | > 0.3 | ✅ |
| Retention delta | **+0.164** | > 0.1 | ✅ |
| S4 correlation | **-0.230** | < 0 | ✅ |
| Model agreement | High | — | ✅ |
| Prompt robustness | High | — | ✅ |

---

## pdensity: Proper Positioning

### What pdensity IS
- An **interpretive construct** (not a metric)
- Explanatory concept for semantic constraint strength
- "How much does this tag narrow the possibility space?"
- Provides intuition, not measurement

### What pdensity is NOT
- ❌ Uniquely defined
- ❌ Directly measurable
- ❌ Central contribution
- ❌ Control layer or agent policy

### How to Introduce in Paper

```
Propositional Density (PDensity)

We introduce propositional density (pdensity) as an explanatory concept:
the number of independent semantic constraints encoded in a compact
linguistic unit.

We do not claim pdensity is uniquely defined or directly measurable.
Rather, it provides intuition for why certain gentags exert stronger
generative constraints and collapse semantic space more aggressively
than others.
```

### Safe Framing
> "pdensity is not objective, not universal—it is an interpretive construct for understanding semantic constraint in LLM-based systems."

This disarms reviewers. They can't attack what you explicitly position as interpretive.

---

## Reviewer-Safe Claims

### Strong (Defensible)
- "Gentags are lexically variable but semantically stable"
- "Representation variability correlates with evidence sparsity"
- "Multiple LLMs produce semantically similar gentags"
- "Gentags preserve review meaning better than random"
- "Model-in-the-loop systems are unstable across repeated queries"

### Moderate (Supported)
- "Gentags enable localized change attribution"
- "Dense embeddings exhibit diffuse, non-attributable drift"
- "Gentags provide persistent semantic state"

### Interpretive (Explicitly Flagged)
- "pdensity is an interpretive construct for semantic constraint strength"
- "Representation variability serves as a proxy for epistemic insufficiency"

### NOT Claimed (Important)
- ❌ Calibrated uncertainty estimation
- ❌ Bayesian posteriors
- ❌ Decision-making policies
- ❌ Control loops or action selection
- ❌ Full autonomous agent

---

## The Bottom Line

**Gentags stand on their own.**

The paper is publishable with:
1. Gentags (new primitive)
2. Semantic stability (key claim)
3. Evidence-sensitive variability (S4)
4. Localized change attribution (Phase 3)

**The Core Claim (Correct Framing):**

> Gentags provide a factorized, persistent semantic representation that enables localized change attribution and evidence-sensitive variability, which dense embeddings and model-in-the-loop architectures cannot provide.

**What We Show:**
- Semantic stability
- Localized drift
- Persistent state
- Evidence-sensitive variability
- Cost efficiency

**Together:** Gentags act as an **observable semantic state layer** for downstream systems.

**NOT:** Full decision-making, full uncertainty quantification, full control. Those come later.

pdensity is an interpretive construct—powerful for narrative, not required for validity.
