# Phase 2: Semantic Stability & Compression Analysis

## Overview

Phase 2 analyzes the semantic properties of gentags using embedding-based metrics, moving beyond surface-level Jaccard overlap to measure semantic stability, sensitivity, and compression efficiency.

## Key Shift from Phase 1

- **Phase 1:** Demonstrated that gentags emerge zero-shot across 4 models and 3 prompts
- **Phase 2:** Measures semantic meaning retention and stability, not just string overlap
- **Jaccard is reported as a surface diagnostic; semantic similarity is primary.**

## Analysis Blocks

### Block A: Run Semantic Stability (RQ1-S)

**Question:** Are gentags semantically stable across runs?

- **Unit:** venue × model × prompt
- **Comparison:** run1 vs run2
- **Representation:** T_mean_unique (dedupe by tag_norm_eval, then mean pool)
- **Primary metric:** `cosine(T_mean_unique(run1), T_mean_unique(run2))`
- **Robustness metric:** MMC (Mean Max Cosine) between tag embedding sets
- **Surface diagnostic:** Jaccard(norm_eval)

### Block B: Prompt Semantic Sensitivity (RQ2-S)

**Question:** How do tags differ semantically across prompts?

- **Unit:** venue × model × run
- **Comparison:** prompt A vs B (all pairs: minimal/anti_hallucination/short_phrase)
- **Representation:** T_mean_unique
- **Primary metric:** cosine between pooled representations
- **Surface diagnostic:** Jaccard

### Block C: Model Semantic Sensitivity (RQ3-S)

**Question:** How do tags differ semantically across models?

- **Unit:** venue × prompt × run
- **Comparison:** model A vs B (all pairs: openai/gemini/claude/grok)
- **Representation:** T_mean_unique
- **Primary metric:** cosine between pooled representations
- **Surface diagnostic:** Jaccard

### Block D: Semantic Retention

**Question:** Do gentags retain the meaning of the original reviews?

- **Unit:** venue × model × prompt × run
- **Representations:**
  - Reviews: R_mean = mean pooling of per-review embeddings
  - Tags: T_mean_unique = dedupe by tag_norm_eval, then mean pool
- **Primary metric:** `cosine(R_mean(venue), T_mean_unique(extraction))`
- **Null baselines:**
  - Baseline 1: Random tags (within-dataset, same count)
  - Baseline 2: Shuffled venue mapping (pair venue reviews with another venue's tags)
  - Baseline 3: Top-K frequent tags (global top-K as generic summary)
- **Delta retention:** `retention_cosine - retention_random_baseline`

### Block E: Compression Efficiency

**Question:** What is the cost-information tradeoff?

- **Unit:** extraction
- **Primary metrics:**
  - (E1) Delta retention per dollar: `delta_retention / cost_usd`
  - (E2) Pareto front: Plot retention vs cost (or tokens) showing Pareto curve by model/prompt
- **Supporting metrics:**
  - Delta retention per token: `delta_retention / total_tokens`
  - Delta retention per tag: `delta_retention / n_tags`

### Block F: Uncertainty Signal

**Question:** What is the structure of variability?

- **Unit:** venue
- **Definition:** Representation dispersion (not entropy)
- **For each venue (24 representations: 4 models × 3 prompts × 2 runs):**
  - Mean pairwise distance (1 - cosine) among all 24 vectors
  - Within-run variance (run1 vs run2 comparisons, averaged)
  - Prompt variance (within model, across prompts)
  - Model variance (within prompt, across models)
- **Interpretation:** High dispersion = representation instability = lower confidence

## Embedding Strategy

### Representations

**Reviews (CANONICAL):**

- (R1) Review-level embeddings: Embed each review separately → `E(r_i)`
- (R2) Venue pooled representation: `R_mean(v) = mean_i(E(r_i))` ← **PRIMARY**

**Tags (CANONICAL):**

- (T1) Tag-level embeddings: Embed each gentag separately → `E(t_j)`
- (T2) Extraction pooled representation: `T_mean_unique(e) = mean_j(E(t_j))` where tags are deduplicated by `tag_norm_eval` first ← **PRIMARY**
- (T3) Optional: Concatenated tags text embedding (baseline for sensitivity check)

### Embedding Model

- **Primary:** OpenAI `text-embedding-3-large` (3072 dimensions)
- **Rationale:** Stable, strong semantic performance, single model keeps evaluation consistent
- **Robustness check:** Recompute on 50 venues using alternative embedding model, show rank correlation between metrics (to demonstrate results aren't embedding-model-specific)

## Deliverables (6 Required Plots)

1. **Run stability distribution**

   - Violin/ECDF plot per model (optionally per prompt)
   - Shows: Distribution of cosine similarity between run1 and run2

2. **Prompt sensitivity heatmap**

   - 3×3 matrix per model
   - Shows: Mean semantic shift between prompt pairs

3. **Model sensitivity heatmap**

   - 4×4 matrix per prompt
   - Shows: Mean semantic shift between model pairs

4. **Retention plot**

   - Cosine(review, gentags) by model/prompt
   - Shows: How well gentags capture review meaning

5. **Cost-effectiveness plot**

   - Pareto front: retention vs cost (or tokens) by model/prompt
   - Shows: Cost-information tradeoff (more interpretable than raw ratios)

6. **Surface vs semantic scatter**
   - Jaccard vs cosine similarity
   - Shows: Why literal overlap is insufficient (demonstrates semantic equivalence despite surface variation)

## Results Framing

### Key Messages

1. **Surface variation is expected:** Different runs often paraphrase the same idea ("great coffee" vs "excellent coffee")
2. **Jaccard is a diagnostic, not the signal:** We report Jaccard as surface variability, but use embedding-based semantic similarity as the primary stability measure
3. **Three stability concepts:**
   - Run stability: same model/prompt, run1 vs run2
   - Prompt sensitivity: same model, different prompts
   - Model sensitivity: same prompt, different models
4. **Uncertainty as signal:** When gentag sets change substantially across runs/prompts/models, downstream systems can treat the venue representation as lower confidence
5. **Cost-information tradeoff:** Gentags offer favorable compression while retaining semantic meaning

## Implementation

### Scripts

- `scripts/phase2_analysis.py` - Main analysis pipeline

  - Loads Phase 1 results
  - Computes embeddings (with caching)
  - Computes all metrics
  - Generates tables

- `scripts/phase2_plots.py` - Plot generation (to be created)
  - Generates 6 required plots
  - Saves to `results/phase2/plots/`

### Output Structure

```
results/phase2/
├── cache/
│   ├── review_embeddings.npz (efficient NPZ format)
│   ├── review_embeddings_map.json (venue_id → indices)
│   ├── tag_embeddings.npz
│   └── tag_embeddings_map.json (tag_raw → index)
├── tables/
│   ├── run_stability.csv (cosine, mmc, jaccard)
│   ├── prompt_sensitivity.csv
│   ├── model_sensitivity.csv
│   ├── retention.csv (with baselines: random, shuffled, topk)
│   └── uncertainty_dispersion.csv
└── plots/
    ├── 1_run_stability.png
    ├── 2_prompt_sensitivity.png
    ├── 3_model_sensitivity.png
    ├── 4_retention.png
    ├── 5_cost_effectiveness.png (Pareto front)
    └── 6_surface_vs_semantic.png
```

## Usage

```bash
# Run full analysis (computes embeddings)
poetry run python scripts/phase2_analysis.py \
  --run-id week2_run_20251223_191104 \
  --data data/study1_venues_20250117.csv

# Skip embedding computation (use cached)
poetry run python scripts/phase2_analysis.py \
  --run-id week2_run_20251223_191104 \
  --skip-embeddings
```

## Next Steps

1. ✅ Created analysis script structure
2. ⏳ Create plotting script for 6 deliverables
3. ⏳ Test on small subset
4. ⏳ Run full analysis on Phase 1 results
5. ⏳ Generate paper-ready plots
