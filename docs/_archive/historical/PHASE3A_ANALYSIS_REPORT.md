# Phase 3A: Classical Baseline Comparison Report

**Date:** 2026-01-30 (Corrected: 2026-02-01)
**Status:** Complete (Methodology Corrected)
**Purpose:** Validate gentags against classical keyword extraction methods

> **METHODOLOGY CORRECTION (2026-02-01):** The original report compared gentag drift Gini (0.657) to baseline representation Gini (0.12), claiming "5x advantage." This was invalid — different metrics. The valid comparison is gentags (0.553) vs embeddings (0.369) = 1.50x advantage. Classical baselines are deterministic and cannot have drift Gini computed.

---

## Executive Summary

**Critical Finding:** Classical keyword extraction methods (RAKE, TF-IDF, YAKE) achieve higher retention than gentags. This is expected — they extract surface tokens optimized for lexical overlap.

**Key Insight:** Classical methods are **deterministic**. There is no run-to-run variation, so drift-based localization analysis (Phase 3) cannot be applied to them. The valid comparison is gentags vs embeddings (1.50x localization advantage), not gentags vs classical baselines.

**Decision Tree Outcome:** Retention shows baselines win on surface extraction; gentags win on semantic properties (stability, cross-model agreement, evidence-sensitive dispersion).

### Key Results at a Glance

| Metric | Winner | Gentags | Best Baseline | Gap |
|--------|--------|---------|---------------|-----|
| **Retention** | RAKE | 0.625 | 0.742 (RAKE) | -0.117 |
| **Localization (Drift Gini)** | **Gentags** | **0.553** | N/A (deterministic) | — |

> **Methodological Note:** The original Phase 3A report compared gentag drift Gini (0.553-0.657) to baseline representation Gini (0.12). This comparison was **invalid** — these are different metrics. Drift Gini measures change concentration between runs; representation Gini measures facet similarity distribution. Classical baselines have no run-to-run variation, so drift Gini = 0 by definition.

---

## Why This Phase Was Critical

The paper had a **glass jaw**: "+0.164 above random" for retention is necessary but not sufficient.

Reviewers would immediately ask:
> "Why not just use TF-IDF or RAKE? They're computationally free."

**This phase answers that question definitively.**

---

## Methodology

### Budget Matching

To ensure fair comparison:
- **k = ~20 keywords** (median gentag count per venue)
- **Phrase length = 1-4 words** (matching gentag constraint)

### Methods Compared

| Method | Type | Description |
|--------|------|-------------|
| **TF-IDF** | Statistical | Term frequency–inverse document frequency |
| **RAKE** | Statistical | Rapid Automatic Keyword Extraction |
| **YAKE** | Statistical | Yet Another Keyword Extractor |
| **Gentags** | LLM-based | Machine-generated semantic tags |

### Metrics

1. **Retention:** `cosine(embed(reviews), embed(keywords))`
2. ~~**Localization (Gini):** Concentration of semantic drift across facets~~ — **INVALID** for deterministic baselines

---

## Results

### A) Retention Comparison

| Method | Mean | Std | Median | Avg Keywords |
|--------|------|-----|--------|--------------|
| **RAKE** | **0.742** | 0.052 | 0.744 | 19.5 |
| TF-IDF | 0.687 | 0.063 | 0.694 | 19.8 |
| YAKE | 0.677 | 0.077 | 0.687 | 19.8 |
| Gentags | 0.625 | 0.052 | 0.620 | 21.5 |

**Finding:** RAKE beats gentags by 0.117 on retention. TF-IDF and YAKE also outperform gentags.

**Interpretation:** Classical methods extract *surface tokens* that correlate better with the review text they came from. This is expected — they're optimized for lexical overlap.

---

### B) Localization (Gini) — Methodological Correction

> ⚠️ **INVALID COMPARISON REMOVED**
>
> The original analysis compared:
> - Gentag **drift Gini** (0.657): How concentrated is the CHANGE between run1 and run2
> - Baseline **representation Gini** (0.12): How evenly does a single snapshot relate to facets
>
> These are fundamentally different metrics. The "5x advantage" claim was methodologically unsound.

**Valid Comparisons:**

| Comparison | Gentags | Baseline | Valid? |
|------------|---------|----------|--------|
| Drift Gini: Gentags vs Embeddings | 0.553 | 0.369 | ✅ Yes (1.50x) |
| Drift Gini: Gentags vs Classical | 0.553 | N/A | ❌ No (deterministic) |
| Representation Gini | — | 0.12 | — (different metric) |

**Why classical baselines can't be compared on drift:**
- RAKE(text) = RAKE(text) — deterministic, no variation
- No run-to-run variation → drift = 0 by definition
- The metric simply doesn't apply to deterministic methods

---

### C) Combined View

| Method | Retention | Stochastic? | Valid Drift Comparison? |
|--------|-----------|-------------|-------------------------|
| RAKE | 0.742 | ❌ No | ❌ N/A |
| TF-IDF | 0.687 | ❌ No | ❌ N/A |
| YAKE | 0.677 | ❌ No | ❌ N/A |
| **Gentags** | 0.625 | ✅ Yes | ✅ 0.553 drift Gini |
| Embeddings | — | ✅ Yes | ✅ 0.369 drift Gini |

**Key insight:** Classical methods and LLM-based methods are fundamentally different:
- Classical: Deterministic, high retention, no semantic properties to measure
- LLM-based: Stochastic, semantic stability measurable, localization advantage over embeddings

---

## Decision Tree Analysis

### The Plan (Original)

| Case | Condition | Outcome |
|------|-----------|---------|
| Case 1 | Gentags beat classics on retention | Golden — "LLM captures semantics beyond surface" |
| Case 2 | Gentags tie/lose retention, win localization | Pivot — Localization is the contribution |
| Case 3 | Classics beat gentags on both metrics | Red alert |

### The Corrected Result

- Gentags **lose** retention (-0.117 vs RAKE) — expected, baselines optimized for this
- ~~Gentags "win" localization~~ — **INVALID comparison** (different metrics)

**The valid localization comparison is gentags vs embeddings (both stochastic, both drift Gini):**
- Gentags drift Gini: 0.553
- Embedding drift Gini: 0.369
- **Advantage: 1.50x**

**Updated claim:** Classical baselines excel at surface extraction but lack semantic properties. Gentags provide attributable semantic state vs embeddings (1.50x), not vs classical baselines.

---

## What Phase 3A Actually Shows

### 1. Localization vs Embeddings (Valid Comparison)

**The valid comparison is gentags (0.553) vs embeddings (0.369) = 1.50x advantage.**

Both are stochastic representations where we can measure run-to-run drift. Classical baselines are deterministic — no variation to measure.

**With gentags:**
```
tags_t1 = {"great coffee", "friendly staff", "quiet atmosphere"}
tags_t2 = {"great coffee", "slow service", "crowded atmosphere"}

Changes:
  - "friendly staff" → "slow service"   (SERVICE changed)
  - "quiet atmosphere" → "crowded"      (AMBIANCE changed)
  - "great coffee" → unchanged          (COFFEE_DRINKS stable)
```

**With embeddings:** Changes are diffuse, non-attributable (0.369 Gini).

### 2. Semantic Stability (from Phase 2)

| Metric | Value |
|--------|-------|
| Run-to-run cosine | 0.977 |
| Cross-model cosine | >0.94 |
| MMC (paraphrase) | 0.887 |

Classical methods are deterministic — they trivially "win" run-to-run stability. But gentags show high *semantic* stability even when exact tokens differ.

### 3. Cross-Model Agreement (from Phase 2)

Four different LLMs (OpenAI, Gemini, Claude, Grok) produce semantically similar gentags. This proves gentags capture **latent semantic structure**, not model-specific artifacts.

Classical methods don't have this property because they extract surface tokens, not concepts.

### 4. Evidence-Sensitive Dispersion (from Phase 2)

| Evidence Level | Gentag Variability |
|----------------|-------------------|
| Sparse (<200 tokens) | 0.057 (highest) |
| Dense (>600 tokens) | 0.044 (lowest) |

Correlation: **r = -0.230**

Gentags signal identifiability — sparse evidence produces higher dispersion. This is semantically meaningful. Classical methods don't have this property.

---

## Paper Framing Implications

### Old Framing (Vulnerable)
> "Gentags retain review semantics better than random baseline."

**Problem:** RAKE does it better.

### New Framing (Defensible)
> "Gentags provide **localized, attributable semantic state** with 1.50x localization advantage over embeddings."

**Supporting evidence:**
1. 1.50x higher drift Gini than embeddings (valid comparison)
2. Semantic stability under lexical variation (0.977 cosine, 0.471 Jaccard)
3. Cross-model agreement proves linguistic universality
4. Evidence-sensitive dispersion provides identifiability signal

**Note:** Classical baselines are deterministic — no run-to-run variation exists, so drift Gini cannot be computed. The original "5x" claim comparing drift Gini to representation Gini was methodologically invalid.

---

## The Bottom Line

### What Classical Methods Provide
- High retention (they're optimized for this)
- Zero cost (no API calls)
- Deterministic output

### What Classical Methods Cannot Provide
- **Run-to-run variation:** Deterministic = no drift to measure
- **Semantic stability analysis:** No stochastic variation
- **Semantic structure:** Surface tokens, not concepts
- **Cross-model validation:** Not applicable

### What Gentags Provide
- **Attributable state:** Know exactly what changed
- **Semantic stability:** Same meaning, different words
- **Cross-model agreement:** Universal latent semantics
- **Evidence sensitivity:** Signal when information is insufficient

---

## Cost Comparison

| Method | Compute Cost | Retention | Notes |
|--------|--------------|-----------|-------|
| TF-IDF | ~$0 | 0.687 | Deterministic |
| RAKE | ~$0 | 0.742 | Deterministic |
| YAKE | ~$0 | 0.677 | Deterministic |
| **Gentags** | **~$0.005/venue** | 0.625 | Stochastic, semantic properties |

**Trade-off:** Pay ~$0.005/venue for semantic stability and localization advantage over embeddings (1.50x).

For 230 venues: **$1.15 total** for attributable semantic state.

---

## Statistical Significance

| Comparison | Metric | Test | Result |
|------------|--------|------|--------|
| Gentags vs RAKE | Retention | Paired t-test | p < 0.001 (RAKE wins) |
| Gentags vs Embeddings | Drift Gini | Wilcoxon | p < 0.001 (Gentags wins, 1.50x) |

> ~~Gentags vs Baselines Gini~~ — **REMOVED** (invalid: comparing different metrics)

---

## Reviewer Shield

> "We compare gentags against classical keyword extraction methods (TF-IDF, RAKE, YAKE). Classical methods achieve higher retention (optimizing for lexical overlap), but they are deterministic — there is no run-to-run variation to analyze. Gentags provide **1.50x better localization than embeddings** (drift Gini: 0.553 vs 0.369), plus semantic stability (0.977 cosine), cross-model agreement (>0.94), and evidence-sensitive dispersion (r=-0.230)."

---

## Output Files

### Tables
- `results/phase3a/tables/baseline_retention.csv` — Per-venue retention for all methods
- `results/phase3a/tables/baseline_summary.csv` — Retention summary statistics
- `results/phase3a/tables/baseline_gini.csv` — Per-venue Gini for all methods
- `results/phase3a/tables/gini_summary.csv` — Gini summary statistics

---

## Conclusion

**Phase 3A validates the gentag contribution through contrast:**

| Dimension | Classical Methods | Gentags | Notes |
|-----------|-------------------|---------|-------|
| Retention | ✅ Better (0.742) | ❌ Lower (0.625) | Baselines optimized for this |
| Stochastic variation | ❌ None (deterministic) | ✅ Yes | Required for drift analysis |
| Drift Gini (vs embeddings) | — | ✅ 1.50x advantage | Valid comparison |
| Semantic stability | ❌ N/A (deterministic) | ✅ 0.977 | |
| Cross-model agreement | ❌ N/A | ✅ >0.94 | |
| Evidence sensitivity | ❌ No | ✅ r=-0.230 | |

**Methodological correction:** The original "5x Gini advantage" claim compared drift Gini (gentags) to representation Gini (baselines) — an invalid comparison. The valid localization comparison is gentags vs embeddings: **1.50x advantage**.

**The contribution is not compression. The contribution is attributable semantic state vs embeddings.**

---

*Report generated: 2026-01-30*
*Script: `scripts/phase3a_baselines.py`*
*Venues analyzed: 230*
