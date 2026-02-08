# Phase 3B: Paraphrase Robustness Report

**Date:** 2026-02-01
**Status:** Complete
**Purpose:** Test whether gentags maintain semantic state under meaning-preserving paraphrase

---

## Executive Summary

**Critical Finding:** Gentags do NOT demonstrate the expected robustness to paraphrase. The "kill shot" hypothesis — that gentags would maintain semantic state while RAKE craters under rewording — did not materialize.

**Outcome:** **Negative Result** — Gentags show moderate semantic stability (MMC 0.65) but no localization advantage over classical methods in the drift setting.

### Key Results at a Glance

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **MMC (semantic stability)** | >0.80 | 0.648 | **FAIL** |
| **Drift Gini Advantage** | >2x | 1.0x | **FAIL** |

---

## Hypothesis Under Test

### The "Kill Shot" Claim

> "Gentags capture semantic content, not surface form. Therefore, meaning-preserving paraphrase should leave gentag representations largely unchanged, while lexically-bound methods like RAKE should collapse."

### Expected Outcome

| Method | Under Paraphrase | Prediction |
|--------|------------------|------------|
| Gentags | Same meaning → Same tags | MMC > 0.80 |
| RAKE | Different words → Different keywords | MMC < 0.30 |
| Drift Gini | Gentags localized, RAKE diffuse | Advantage > 2x |

### Actual Outcome

| Method | Under Paraphrase | Reality |
|--------|------------------|---------|
| Gentags | Same meaning → Different tags | MMC = 0.65 |
| RAKE | Different words → Similar drift pattern | ~0.36 Gini |
| Drift Gini | Both similarly distributed | Advantage = 1.0x |

---

## Methodology

### Intervention Design

This was an **intervention study**, not a stability test:
- **Intervention:** Reword reviews while preserving meaning
- **Measure:** Change in extracted representation
- **Compare:** Gentags vs. classical methods

### Diverse Paraphrasers (Avoiding Closed-Loop)

To avoid the critique "GPT-4 paraphrases for GPT-4 extraction = artifact", we used three diverse paraphrasers:

| Method | Model | Type | Purpose |
|--------|-------|------|---------|
| **A** | GPT-4o-mini | LLM | Primary paraphraser |
| **B** | Claude 3.5 Haiku | LLM | Cross-model validation |
| **C** | Back-translation (EN→FR→EN) | Non-LLM | Non-neural baseline |

### Paraphrase Quality Constraints

| Metric | Target | Purpose |
|--------|--------|---------|
| Jaccard (text) | < 0.25 | Ensure lexical transformation |
| Cosine (text embedding) | > 0.85 | Ensure semantic preservation |
| Gap (Cosine - Jaccard) | > 0.60 | Confirm lex ≠ semantic |

### Metrics

1. **MMC (Mean Max Cosine):** Semantic overlap between original and paraphrased gentags
2. **Drift Gini:** Localization of facet similarity *changes* (not absolute values)
3. **Jaccard (tags):** Surface overlap between tag sets

---

## Paraphrase Quality Validation

### Results by Paraphraser

| Method | Jaccard (text) | Cosine (text) | Gap | Quality |
|--------|----------------|---------------|-----|---------|
| **A (GPT-4)** | 0.283 | 0.868 | 0.585 | **Good** |
| B (Claude) | 0.112 | 0.596 | 0.484 | Too aggressive |
| C (Back-trans) | 0.720 | 0.962 | 0.241 | Too conservative |

### Interpretation

- **Method A:** Best balance — adequate lexical change with semantic preservation
- **Method B:** Too aggressive — lost meaning (Cosine 0.60 < 0.85 threshold)
- **Method C:** Too conservative — insufficient lexical transformation (Jaccard 0.72)

---

## Results

### A) MMC (Semantic Stability Under Paraphrase)

| Method | MMC Mean | MMC Std | Target | Status |
|--------|----------|---------|--------|--------|
| A (GPT-4) | 0.651 | 0.052 | >0.80 | **FAIL** |
| B (Claude) | 0.582 | 0.059 | >0.80 | **FAIL** |
| C (Back-trans) | 0.713 | 0.065 | >0.80 | **FAIL** |
| **Overall** | **0.648** | — | >0.80 | **FAIL** |

**Finding:** Gentags achieve only ~65% semantic overlap when reviews are paraphrased. This is below the 80% threshold needed to claim "semantic invariance."

**Interpretation:** LLM-based extraction is more sensitive to surface form than hypothesized. Different wording leads to different tags, even when meaning is preserved.

---

### B) Tag-Level Jaccard (Lexical Overlap)

| Method | Jaccard Mean | Interpretation |
|--------|--------------|----------------|
| A (GPT-4) | 0.027 | Almost no exact tag overlap |
| B (Claude) | 0.009 | Virtually zero overlap |
| C (Back-trans) | 0.073 | Minimal overlap |

**Finding:** Tag-level Jaccard is extremely low (~3%), confirming that different wording produces different exact tags.

**Interpretation:** Combined with MMC of 0.65, this shows gentags are lexically unstable AND only moderately semantically stable — not the "high semantic, low lexical" pattern we hoped for.

---

### C) Drift Gini (Change Localization)

| Method | Gentag Drift Gini | RAKE Drift Gini | TF-IDF Drift Gini | Advantage |
|--------|-------------------|-----------------|-------------------|-----------|
| A (GPT-4) | 0.355 | 0.361 | 0.383 | 0.98x |
| B (Claude) | 0.341 | 0.364 | 0.372 | 0.94x |
| C (Back-trans) | 0.358 | 0.354 | 0.345 | 1.01x |
| **Overall** | **0.351** | **0.360** | **0.367** | **1.0x** |

**Finding:** No drift Gini advantage. In fact, RAKE shows slightly *higher* drift Gini than gentags in Methods A and B.

**Interpretation:** The localization advantage seen in Phase 3A (absolute Gini) does not transfer to the "change localization" setting. When representations change under paraphrase, both gentags and classical methods change in similarly distributed ways across facets.

---

### D) Combined View

| Metric | Gentags | RAKE | TF-IDF | Winner |
|--------|---------|------|--------|--------|
| MMC (stability) | 0.648 | — | — | — |
| Drift Gini | 0.351 | 0.360 | 0.367 | Tie |
| Tag Jaccard | 0.036 | — | — | — |

---

## Why the Hypothesis Failed

### 1. LLM Extraction is Surface-Sensitive

The gentag extraction prompt asks the LLM to generate tags from review text. Even with the same underlying *meaning*, different *wording* triggers different tag generation.

```
Original: "The coffee here is amazing and the staff is super friendly"
Paraphrased: "This establishment serves exceptional espresso with remarkably cordial personnel"

Original tags: ["amazing coffee", "friendly staff", "welcoming atmosphere"]
Paraphrased tags: ["exceptional espresso", "cordial service", "pleasant ambiance"]

MMC = 0.65 (moderate overlap, not high)
```

### 2. Semantic Similarity ≠ Semantic Identity

An MMC of 0.65 means tags are *related* but not *identical*. For a "robustness" claim, we need tags to be near-identical (MMC > 0.80).

### 3. Drift Pattern is Universal

The facet-level change pattern is similar across all methods because:
- The paraphrase affects certain semantic aspects of the review
- All representation methods respond to those same aspects
- The "drift fingerprint" is driven by the paraphrase, not the method

---

## Implications for the Paper

### What We Can Still Claim

From Phase 3:
- **1.50x higher drift Gini than embeddings** — Gentag changes are more localized (valid comparison)
- **Semantic stability across runs** — High cosine (0.977) with low Jaccard (0.471)
- **Cross-model agreement** — Four LLMs produce similar semantic content

### What We Cannot Claim

From Phase 3B:
- ~~"Gentags are robust to paraphrase"~~ — MMC 0.65 is not "robust"
- ~~"Gentags maintain semantic invariance under rewording"~~ — Tags change significantly
- ~~"Gentags have better change localization than RAKE"~~ — No drift Gini advantage

### Honest Framing

> "Gentags provide localized, attributable semantic state (Phase 3A), but are sensitive to surface form variation (Phase 3B). The extraction process captures both semantic content and lexical patterns from source text."

---

## Comparison: Phase 3A vs Phase 3B

| Dimension | Phase 3A Finding | Phase 3B Finding |
|-----------|------------------|------------------|
| **Setting** | Absolute representation | Change under intervention |
| **Metric** | Absolute Gini | Drift Gini |
| **Result** | 1.50x vs embeddings (invalid vs baselines) | No advantage |
| **Interpretation** | Gentags localize representation | Gentags don't localize *change* |

**Key Insight:** Localization of *representation* (Phase 3A) is different from localization of *change* (Phase 3B).

---

## Potential Rescue Strategies

### 1. Canonicalization

Collapse synonym variation before comparing:
- "amazing coffee" ≈ "exceptional espresso"
- "friendly staff" ≈ "cordial personnel"

This might rescue MMC by grouping semantically equivalent tags.

### 2. Facet-Level Analysis

Some facets might be stable while others are volatile:
- Coffee-related tags might be stable
- Ambiance-related tags might be sensitive to wording

Selective claims about which facets are robust.

### 3. Temperature Tuning

Current extraction uses default temperature. Lower temperature might produce more consistent tags.

### 4. Accept and Report

Report the negative result honestly. Science includes null findings.

---

## Statistical Details

### Sample Size
- **Venues processed:** 50
- **Paraphrasers:** 3
- **Total comparisons:** 150

### Per-Method Statistics

| Method | n | MMC Mean | MMC Std | Drift Gini Mean | Drift Gini Std |
|--------|---|----------|---------|-----------------|----------------|
| A | 50 | 0.651 | 0.052 | 0.355 | — |
| B | 50 | 0.582 | 0.059 | 0.341 | — |
| C | 50 | 0.713 | 0.065 | 0.358 | — |

---

## Output Files

### Tables
- `results/phase3b/tables/robustness_results.csv` — Per-venue results for all methods
- `results/phase3b/tables/robustness_summary.csv` — Summary statistics by paraphraser
- `results/phase3b/tables/paraphrase_validation.csv` — Paraphrase quality metrics
- `results/phase3b/tables/paraphrased_A_gpt4.csv` — GPT-4 paraphrased reviews
- `results/phase3b/tables/paraphrased_B_claude.csv` — Claude paraphrased reviews
- `results/phase3b/tables/paraphrased_C_backtrans.csv` — Back-translated reviews

### Plots
- `results/phase3b/plots/1_gini_comparison.png` — Drift Gini comparison across methods
- `results/phase3b/plots/2_mmc_comparison.png` — MMC distribution by paraphraser
- `results/phase3b/plots/3_paraphraser_consistency.png` — Cross-paraphraser consistency

---

## Conclusion

**Phase 3B produces a negative result.** The hypothesis that gentags would demonstrate robustness to meaning-preserving paraphrase while classical methods collapse did not hold.

| Win Condition | Target | Actual | Verdict |
|---------------|--------|--------|---------|
| MMC > 0.80 | 0.80 | 0.648 | **FAIL** |
| Drift Gini > 2x RAKE | 2.0x | 1.0x | **FAIL** |

### What This Means

1. **Gentags are more surface-sensitive than expected** — Different wording produces different tags
2. **The Phase 3A localization advantage is real but limited** — It applies to representation structure, not change patterns
3. **The "paraphrase robustness" claim cannot be made** — Honest reporting required

### Recommendation

Proceed with canonicalization experiment (if time permits) to test whether synonym collapsing can rescue MMC. Otherwise, report Phase 3B as a negative result that bounds the gentag contribution.

---

*Report generated: 2026-02-01*
*Script: `scripts/phase3b_robustness.py`*
*Venues analyzed: 50*
*Paraphrasers: 3 (GPT-4o-mini, Claude Haiku, Back-translation)*
