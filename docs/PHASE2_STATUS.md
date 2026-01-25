# Phase 2 Analysis Status

**Last Updated:** 2026-01-24
**Status:** ✅ ALL COMPLETE (S1-S4)

---

## Executive Summary

Phase 2 semantic stability analysis is **largely complete**. The key claim has been validated:

> **"Gentags are lexically unstable but semantically stable"**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Cosine (semantic) | > 0.9 | **0.977** | ✅ |
| Jaccard (surface) | 0.3-0.6 | **0.471** | ✅ |
| Gap (cosine - jaccard) | > 0.3 | **0.504** | ✅ |
| Retention delta vs random | > +0.1 | **+0.164** | ✅ |

**All analyses complete**, including S4 (Sparsity Analysis).

---

## What's Been Completed

### S1: Run Stability ✅

**Question:** If I run the same extraction twice, do I get the same meaning?

**Result:** YES — High semantic stability despite surface variation.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cosine | 0.977 | Same meaning across runs |
| Jaccard | 0.471 | Different surface forms (expected) |
| MMC | 0.887 | Tags are paraphrases of each other |
| Gap | 0.504 | Proves lexical ≠ semantic |

**Conclusion:** Run-to-run variation is lexical (paraphrases), not semantic. Gentags are stable representations.

### S2: Prompt Sensitivity ✅

**Question:** Do prompts change what meaning is extracted?

**Result:** Prompts affect granularity/style but not core semantics.

- `anti_hallucination` → more tags, more grounded
- `short_phrase` → fewer tags, more compressed
- Semantic similarity across prompts remains high

**Tables:** `results/phase2/tables/prompt_sensitivity.csv`

### S3: Model Sensitivity ✅

**Question:** Do different LLMs extract the same latent semantics?

**Result:** Models share core semantic dimensions despite style differences.

- All 4 models (OpenAI, Gemini, Claude, Grok) produce semantically similar outputs
- Differences are in verbosity and granularity, not meaning

**Tables:** `results/phase2/tables/model_sensitivity.csv`

### Retention Analysis ✅

**Question:** Do gentags retain the meaning of the original reviews?

**Result:** YES — Gentags capture review meaning significantly better than random.

| Metric | Value |
|--------|-------|
| Retention (cosine to reviews) | 0.625 |
| Random baseline | 0.461 |
| Delta (above random) | **+0.164** |

**Tables:** `results/phase2/tables/retention.csv`

### Uncertainty Dispersion ✅

**Question:** What is the structure of variability across extractions?

**Result:** Computed per-venue variability metrics.

- Mean pairwise distance: 0.051
- Within-run, prompt, and model variance decomposition available

**Tables:** `results/phase2/tables/uncertainty_dispersion.csv`

### Plots Generated ✅

All 6 required plots have been generated:

1. `1_run_stability.png` — Run-to-run stability by model
2. `2_prompt_sensitivity.png` — Prompt sensitivity heatmaps
3. `3_model_sensitivity.png` — Model sensitivity heatmaps
4. `4_retention.png` — Retention vs baselines
5. `5_cost_effectiveness.png` — Cost-information tradeoff
6. `6_surface_vs_semantic.png` — Jaccard vs Cosine scatter

**Location:** `results/phase2/plots/`

---

### S4: Stability Under Sparsity ✅ COMPLETE

**Question:** As textual evidence decreases, does representation variability increase?

**Result:** YES — **Correlation = -0.230** (negative as expected)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Token-variability correlation | **-0.230** | More evidence → less variability |
| Mean tokens per venue | ~400 | Moderate evidence |
| Mean variability | 0.051 | Low overall variability |

**Conclusion:** Variability is an **uncertainty signal**, not noise. Venues with less textual evidence show more representation variability — exactly what we'd expect if variability reflects information need.

**Tables:** `results/phase2/tables/sparsity_analysis.csv`
**Plot:** `results/phase2/plots/7_sparsity_analysis.png`

---

## Data Summary

### After Filtering

| Metric | Before | After |
|--------|--------|-------|
| Total extractions | 13,272 | 5,517 |
| Error extractions removed | — | 2,898 (21.8%) |
| Venues with all 4 models | 553 | 230 |
| Tag rows | 230,151 | 118,832 |

### Why Filtering Was Needed

Phase 1 had API failures:
- **Claude:** 51% error rate (1,697 failures)
- **Grok:** 36% error rate (1,201 failures)
- **OpenAI/Gemini:** 0% errors

We filtered to venues with successful extractions from **all 4 models** to ensure fair comparisons.

See: `docs/PHASE1_EXTRACTION_ISSUES.md`

---

## Key Learnings

### 1. Surface Variation ≠ Semantic Variation

The 0.504 gap between cosine and Jaccard proves that different words can express the same meaning. "Great coffee" and "excellent espresso" are lexically different but semantically similar.

### 2. Gentags Are Robust Across Models

All 4 LLMs extract similar semantic content. This suggests gentags reflect shared cultural/linguistic priors, not model-specific artifacts.

### 3. Prompts Affect Style, Not Substance

Different prompts produce different numbers of tags and different phrasings, but the core meaning extracted is consistent.

### 4. Variability is Interpretable

When gentag representations vary across runs/prompts/models, this can be interpreted as uncertainty — a feature, not a bug.

---

## File Inventory

### Scripts
- `scripts/phase2_analysis.py` — Main analysis (S1-S3, retention, uncertainty)
- `scripts/phase2_plots.py` — Plot generation

### Results
```
results/phase2/
├── tables/
│   ├── run_stability.csv           # S1 results
│   ├── run_stability_summary.csv
│   ├── prompt_sensitivity.csv      # S2 results
│   ├── prompt_sensitivity_summary.csv
│   ├── model_sensitivity.csv       # S3 results
│   ├── model_sensitivity_summary.csv
│   ├── retention.csv               # Retention analysis
│   ├── compression_summary.csv     # Cost-effectiveness
│   ├── uncertainty_dispersion.csv  # Per-venue variability
│   └── sparsity_analysis.csv       # S4 results
├── plots/
│   ├── 1_run_stability.png
│   ├── 2_prompt_sensitivity.png
│   ├── 3_model_sensitivity.png
│   ├── 4_retention.png
│   ├── 5_cost_effectiveness.png
│   ├── 6_surface_vs_semantic.png
│   └── 7_sparsity_analysis.png     # S4 plot
├── phase2_manifest.json
└── phase2_cache/                   # Embedding cache
```

### Documentation
- `docs/PHASE2_PLAN.md` — Original analysis plan
- `docs/PHASE2_STATUS.md` — This document
- `docs/PHASE1_EXTRACTION_ISSUES.md` — Extraction failure documentation
- `issues/001_phase2_analysis_performance.md` — Technical issues and resolution

---

## Next Steps

Phase 2 is complete. Phase 3 is also complete:

1. **Phase 3: Baseline Comparison & Attribution Analysis** ✅ Complete
   - See `docs/PHASE3_ANALYSIS_REPORT.md` for full results with plots
   - Localization experiment: Gentag Gini 0.657 vs Embedding Gini 0.361
   - Model-in-loop baseline: 31.6% exact match rate (unstable)
   - Cost comparison complete
   - Cold-start analysis complete

2. **Write Complete Analysis Report** ✅ Complete
   - `docs/STABILITY_ANALYSIS_REPORT.md` — Phase 2 full report
   - `docs/PHASE3_ANALYSIS_REPORT.md` — Phase 3 full report

---

## Commands Reference

```bash
# Run full Phase 2 analysis
poetry run python scripts/phase2_analysis.py \
  --run-id week2_run_20251223_191104 \
  --data data/study1_venues_20250117.csv \
  --results-dir results/phase1_downloaded \
  --skip-embeddings

# Generate plots
poetry run python scripts/phase2_plots.py

# View plots
open results/phase2/plots/
```
