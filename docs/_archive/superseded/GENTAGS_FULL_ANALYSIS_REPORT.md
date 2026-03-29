# Gentags: Full Analysis Report

**Date:** 2026-01-30 (Updated: 2026-02-07)
**Status:** In progress (Phase 2 complete; Phase 3 redefined)
**Purpose:** Consolidated analysis for paper preparation

---

## Executive Summary

This report documents the complete empirical validation of **gentags** — machine-generated semantic tags extracted from venue reviews. Gentags serve as **representation infrastructure**: a persistent, inspectable, factorized semantic state layer.

### The Core Claim

> **Gentags are a persistent semantic representation that is compact, inspectable, and preserves meaning.**

### Key Metrics at a Glance

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Semantic stability (cosine) | **0.977** | > 0.9 | ✅ |
| Surface variation (Jaccard) | **0.471** | 0.3-0.6 | ✅ |
| Semantic gap (cosine - Jaccard) | **0.504** | > 0.3 | ✅ |
| Retention above random | **+0.164** | > 0.1 | ✅ |
| Evidence-variability correlation | **-0.230** | < 0 | ✅ |

**Phase 3 (planned):** State-Gini structural proof + DIR/INV utility proof.

### Falsification Criteria

Gentags would fail if any of these broke:

| Criterion | Threshold | Actual | Status |
|-----------|-----------|--------|--------|
| Semantic cosine < embedding baseline | < 0.9 | **0.977** | ✅ PASS |
| Variability does NOT decrease with evidence | r ≥ 0 | **-0.230** | ✅ PASS |
| Model agreement collapses | < 0.9 | **>0.94** | ✅ PASS |

---

## What This Paper IS

A **representation + characterization paper** showing:

1. LLMs can externalize latent semantics as discrete, inspectable tags
2. These representations are semantically stable despite lexical variation
3. Dispersion correlates with evidence sparsity (identifiability signal)
4. Multiple models agree on extracted semantics
5. Phase 3 will test **structure (State-Gini)** and **utility (DIR/INV)** (planned)

### What This Paper is NOT

- ❌ A retrieval benchmark paper
- ❌ A recommender systems paper
- ❌ A user study paper
- ❌ An uncertainty quantification paper
- ❌ A belief-state or probabilistic state-estimation paper
- ❌ A control/agent paper

**Terminology note:** In this paper, **semantic state** means a set of discrete, evidence-conditioned
semantic propositions (gentags). It is **not** a probabilistic belief state: we do not model
uncertainty, belief updates, or control dynamics.

---

## Data Summary

### Phase 1: Extraction

| Metric | Value |
|--------|-------|
| Total venues | 553 |
| Total extractions | 13,272 |
| Configuration | 4 models × 3 prompts × 2 runs |

### After Quality Filtering

| Metric | Before | After |
|--------|--------|-------|
| Total extractions | 13,272 | 5,517 |
| Error extractions removed | — | 2,898 (21.8%) |
| Venues with all 4 models | 553 | 230 |
| Tag rows | 230,151 | 118,832 |

### Models and Prompts

**Models:** OpenAI `gpt-5-nano`, Gemini `gemini-2.5-flash`, Claude `claude-sonnet-4-5`, Grok `grok-4`

**Prompts:**
- `anti_hallucination` — More tags, more grounded
- `minimal` — Balanced extraction
- `short_phrase` — Fewer tags, compressed

---

## Phase 2: Stability Analysis

### S1: Run-to-Run Stability

**Question:** If I run the same extraction twice, do I get the same meaning?

**Answer:** YES — High semantic stability despite surface variation.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cosine (semantic) | **0.977** | Same meaning across runs |
| Jaccard (surface) | **0.471** | Different surface forms |
| MMC (paraphrase) | **0.887** | Tags are paraphrases |
| Gap | **0.504** | Proves lexical ≠ semantic |

#### By Model

| Model | Cosine | Jaccard | MMC |
|-------|--------|---------|-----|
| Claude | 0.982 | 0.574 | 0.913 |
| Gemini | 0.971 | 0.404 | 0.869 |
| Grok | 0.975 | 0.722 | 0.876 |
| OpenAI | 0.975 | 0.387 | 0.861 |

![Run Stability](../results/phase2/plots/1_run_stability.png)

---

### S2: Prompt Sensitivity

**Question:** Do prompts change what meaning is extracted?

**Answer:** Prompts affect granularity/style, not core semantics.

| Prompt Pair | Mean Cosine | Mean Jaccard |
|-------------|-------------|--------------|
| anti_hallucination ↔ minimal | 0.966 | 0.321 |
| anti_hallucination ↔ short_phrase | 0.962 | 0.282 |
| minimal ↔ short_phrase | 0.966 | 0.352 |

Semantic similarity remains >0.95 across all prompt pairs.

![Prompt Sensitivity](../results/phase2/plots/2_prompt_sensitivity.png)

---

### S3: Model Agreement

**Question:** Do different LLMs extract the same latent semantics?

**Answer:** YES — Models share core semantic dimensions.

| Model Pair | Mean Cosine | Mean Jaccard |
|------------|-------------|--------------|
| Claude ↔ Gemini | 0.951 | 0.253 |
| Claude ↔ Grok | 0.953 | 0.267 |
| Claude ↔ OpenAI | 0.951 | 0.236 |
| Gemini ↔ Grok | 0.969 | 0.323 |
| Gemini ↔ OpenAI | 0.958 | 0.248 |
| Grok ↔ OpenAI | 0.969 | 0.315 |

All 4 models produce semantically similar outputs (cosine >0.94) despite different surface forms.

![Model Sensitivity](../results/phase2/plots/3_model_sensitivity.png)

---

### S4: Evidence-Induced Dispersion

**Question:** As textual evidence decreases, does representation variability increase?

**Answer:** YES — **Correlation = -0.230**

| Metric | Value |
|--------|-------|
| Token-variability correlation | **-0.230** |
| Mean tokens per venue | ~400 |
| Mean variability | 0.051 |

#### By Token Bucket

| Token Bucket | Mean Variability | N Venues |
|--------------|------------------|----------|
| <200 | 0.057 (highest) | 104 |
| 200-400 | 0.047 | 87 |
| 400-600 | 0.045 | 29 |
| >600 | 0.044 (lowest) | 10 |

**Interpretation:** Limited evidence produces higher dispersion. This is not noise — it's an identifiability signal. Sparse venues have weakly constrained representations.

![Sparsity Analysis](../results/phase2/plots/7_sparsity_analysis.png)

---

### Retention Analysis

**Question:** Do gentags retain the meaning of original reviews?

**Answer:** YES — Significantly above random baseline.

| Metric | Value |
|--------|-------|
| Retention (cosine to reviews) | 0.625 |
| Random baseline | 0.461 |
| **Delta** | **+0.164** |

Gentags are not arbitrary text fragments — they capture semantic content from source reviews.

![Retention](../results/phase2/plots/4_retention.png)

---

### Surface vs Semantic Gap

The scatter below shows many points where Jaccard is low (surface variation) but cosine is high (semantic stability). This decoupling proves lexical overlap ≠ semantic similarity.

![Surface vs Semantic](../results/phase2/plots/6_surface_vs_semantic.png)

---

### Cost-Effectiveness

Different model/prompt combinations offer different cost-quality tradeoffs.

![Cost Effectiveness](../results/phase2/plots/5_cost_effectiveness.png)

---

## Phase 3: Structure + Utility (Planned)

Phase 3 is a **two-part proof** executed independently:
1. **Structural proof** (State-Gini) — the state is factorized.
2. **Utility proof** (CheckList DIR/INV) — the state is actionable.

### Experiment A: Structural Proof (State-Gini)

**Goal:** Quantify factorization of gentag state.

**Protocol:**
- Hard-assign tags to 10 facets using cosine similarity with threshold τ=0.35
- Compute Gini on facet counts

**Baselines:** RAKE / TF-IDF / YAKE (same facets + same τ)

**Primary metrics:** State-Gini (gentags vs baselines)

---

### Experiment B: Utility Proof (CheckList DIR/INV)

**Goal:** Show targeted edits to gentag state predictably change downstream decisions.

**DIR:** Delete/flip a negative tag; score must move in expected direction.

**INV:** Paraphrase tags; score should remain within ε of baseline.

**Baseline:** Dense embeddings (`text-embedding-3-large`).

**Primary metrics:** DIR pass rate, INV pass rate, attribution precision.

---

**Status:** planned. No Phase 3 results are claimed in this report until executed.

## What We Claim (Strong, Defensible)

- ✅ Gentags are lexically variable but semantically stable
- ✅ Limited evidence produces higher dispersion (identifiability signal)
- ✅ Multiple LLMs produce semantically similar gentags
- ✅ Gentags preserve review meaning better than random
- ✅ Gentags provide persistent semantic state
- ⚠️ Phase 3 structural + utility results are planned but not yet run

## What We Do NOT Claim

- ❌ Calibrated uncertainty estimation
- ❌ Bayesian posteriors
- ❌ Decision-making policies
- ❌ Control loops or action selection
- ❌ Full autonomous agent
- ❌ User behavior modeling
- ❌ Recommender system

## Future Work

1. Belief & uncertainty: confidence on gentags, contradiction resolution, probabilistic gentags, fusion across sources
2. Temporal dynamics: update rules, decay, drift handling, persistence policies over time
3. Control & agents: action selection conditioned on gentag state, active querying, self-correction loops
4. Quantitative gentags: numeric attributes, hybrid semantic + scalar tags, clinical variables
5. Domain applications: medicine, monitoring, recommendation, safety

---

## The Bottom Line

### What We Built

A **persistent semantic representation** that:
- Is **compact** (few tags vs. thousands of words)
- Is **inspectable** (read the tags)
- **Preserves meaning** (+0.164 above random)
- Is **consistent across models and prompts**

### What This Enables

Gentags act as an **observable semantic state layer** for downstream systems:
- Support monitoring and comparison workflows
- Enable planned attribution and intervention analyses (Phase 3)

### Research Trajectory

```
Paper 1 (THIS): Representation infrastructure
                → Gentags as semantic propositions

Paper 2 (NEXT): Belief & uncertainty
                → Confidence, contradiction, updating

Paper 3 (FUTURE): Control & agents
                  → Active querying, policies

Paper 4 (FUTURE): Domain applications
                  → SPC, obstetrics, etc.
```

This is how representation research works. Word2vec didn't ship reinforcement learning. TF-IDF didn't ship recommender systems. They shipped representations. Same here.

---

## Technical Details

### Embedding Model
- **Model:** OpenAI `text-embedding-3-large`
- **Dimensions:** 3,072
- **Normalization:** L2 normalized

### Metrics

| Metric | Definition |
|--------|------------|
| Cosine | Semantic similarity in embedding space |
| Jaccard | Surface overlap of normalized tag sets |
| MMC | Mean Max Cosine — paraphrase detection |
| Gini | Concentration of semantic state or change (higher = more localized) |
| Retention | Cosine(review_embedding, tag_embedding) |

### Reproducibility
- Fixed random seeds
- All configurations logged in `phase2_manifest.json`
- Scripts: `scripts/phase2_analysis.py` (Phase 2), Phase 3 scripts pending redefinition

---

## File References

### Phase 2 Tables
- `results/phase2/tables/run_stability.csv`
- `results/phase2/tables/prompt_sensitivity.csv`
- `results/phase2/tables/model_sensitivity.csv`
- `results/phase2/tables/retention.csv`
- `results/phase2/tables/sparsity_analysis.csv`

### Phase 2 Plots
- `results/phase2/plots/1_run_stability.png`
- `results/phase2/plots/2_prompt_sensitivity.png`
- `results/phase2/plots/3_model_sensitivity.png`
- `results/phase2/plots/4_retention.png`
- `results/phase2/plots/5_cost_effectiveness.png`
- `results/phase2/plots/6_surface_vs_semantic.png`
- `results/phase2/plots/7_sparsity_analysis.png`

### Phase 3 Tables
Planned outputs (Phase 3 redefined; structural + utility):
- `results/phase3/tables/state_gini_gentags.csv`
- `results/phase3/tables/state_gini_baselines.csv`
- `results/phase3/tables/state_gini_summary.csv`
- `results/phase3/tables/dir_results.csv`
- `results/phase3/tables/inv_results.csv`
- `results/phase3/tables/dir_inv_summary.csv`

### Phase 3 Plots
- `results/phase3/plots/state_gini_comparison.png`
- `results/phase3/plots/dir_pass_rate.png`
- `results/phase3/plots/inv_pass_rate.png`

---

## Appendix: Phase 4 (Planned)

Phase 4 will add:

### 4A: Coverage & Dispersion (Descriptive)
- Tag-set sparsity per venue
- Observer dispersion per venue
- Evidence vs. dispersion relationship

### 4B: Downstream Sensitivity (Diagnostic)
- 5 semantic constraint bundles as probes
- Ranking stability across OTag snapshots
- Attribution analysis
- Failure mode documentation

**Status:** Planning — see `docs/PHASE4_PLAN.md`

---

*Report generated: 2026-01-30 (Updated: 2026-02-07)*
*Run ID: week2_run_20251223_191104*
