# Phase 2 Success Checklist

## Definition of "Done"

Phase 2 is **complete** when you can produce:
- ✅ 5 tables (CSV) with correct row counts + sane stats
- ✅ 6 plots saved to disk (paper-ready)
- ✅ Embedding cache saved + reload works (`--skip-embeddings`)
- ✅ Reproducibility: fixed seeds + config written to `phase2_manifest.json`
- ✅ Sanity checks pass and you can summarize results in 5–7 bullets

---

## A) Required Outputs (Files That Must Exist)

### Tables (CSV)
Saved in `results/phase2/tables/`:

#### 1. `run_stability.csv`
**Columns:**
- `venue_id`, `model_key`, `prompt_type`
- `cosine_similarity` (PRIMARY)
- `mmc` (ROBUSTNESS)
- `jaccard_norm_eval` (surface diagnostic)
- `n_tags_run1`, `n_tags_run2`

**Expected row count:** ~6,636 (553 venues × 4 models × 3 prompts)

#### 2. `prompt_sensitivity.csv`
**Columns:**
- `venue_id`, `model_key`, `run_number`
- `prompt1`, `prompt2`
- `cosine_similarity`
- `jaccard_norm_eval`

**Expected row count:** ~13,272 (553 × 4 × 2 × 3 pairs)

#### 3. `model_sensitivity.csv`
**Columns:**
- `venue_id`, `prompt_type`, `run_number`
- `model1`, `model2`
- `cosine_similarity`
- `jaccard_norm_eval`

**Expected row count:** ~19,908 (553 × 3 × 2 × 6 pairs)

#### 4. `retention.csv`
**Columns:**
- `venue_id`, `exp_id`, `model_key`, `prompt_type`, `run_number`
- `retention_cosine` (PRIMARY)
- `retention_random`, `retention_shuffled`, `retention_topk` (baselines)
- `delta_retention` (PRIMARY FOR compression)
- `cost_usd`, `total_tokens`, `n_tags`
- `delta_retention_per_dollar`, `delta_retention_per_token`, `delta_retention_per_tag`

**Expected row count:** ~13,272 (one per extraction)

#### 5. `uncertainty_dispersion.csv`
**Columns:**
- `venue_id`
- `mean_pairwise_distance` (PRIMARY)
- `within_run_mean_distance` (PRIMARY)
- `prompt_mean_distance` (PRIMARY)
- `model_mean_distance` (PRIMARY)
- `n_representations`

**Expected row count:** ~553 (one per venue)

### Plots (PNG)
Saved in `results/phase2/plots/`:

1. `1_run_stability.png` - Violin/ECDF per model
2. `2_prompt_sensitivity.png` - 3×3 heatmap per model
3. `3_model_sensitivity.png` - 4×4 heatmap per prompt
4. `4_retention.png` - Bar/box by model×prompt
5. `5_cost_effectiveness.png` - Pareto front (retention vs cost)
6. `6_surface_vs_semantic.png` - Scatter: Jaccard vs cosine

### Cache
Saved in `results/phase2_cache/`:

- `review_embeddings.npz`
- `review_embeddings_map.json`
- `tag_embeddings.npz`
- `tag_embeddings_map.json`

### Manifest
Saved in `results/phase2/phase2_manifest.json`:

```json
{
  "run_id": "week2_run_20251223_191104",
  "embedding_model": "text-embedding-3-large",
  "date": "2025-01-XX",
  "git_commit": "abc123...",
  "seeds": {"random_baseline": 42, "shuffled_venues": 42},
  "dataset_path": "data/study1_venues_20250117.csv",
  "venue_count": 553,
  "extraction_count": 13272,
  "tag_count": 123456
}
```

---

## B) Correctness Checks (Must Run These)

### 1. Data Completeness

```python
# Check extraction counts
extractions_df.groupby("venue_id").size().value_counts()
# Expect: most venues have 24 extractions (4 models × 3 prompts × 2 runs)
```

### 2. Run Stability Row Count

```python
len(run_stability)  # Should be ≈ 6,636
```

If way smaller, you're dropping data (check for missing embeddings or exp_id mismatch).

### 3. Prompt Sensitivity Row Count

```python
len(prompt_sensitivity)  # Should be ≈ 13,272
```

### 4. Model Sensitivity Row Count

```python
len(model_sensitivity)  # Should be ≈ 19,908
```

### 5. Retention Row Count

```python
len(retention)  # Should equal extractions: ~13,272
```

### 6. Sanity Ranges

```python
# Cosine similarities mostly in [0.4, 0.98]
run_stability["cosine_similarity"].describe()

# MMC should typically be ≥ pooled cosine (not always but often)
(run_stability["mmc"] >= run_stability["cosine_similarity"]).mean()

# Retention should beat baselines
(retention["retention_cosine"] > retention["retention_random"]).mean()  # Should be > 0.5
(retention["retention_cosine"] > retention["retention_shuffled"]).mean()  # Should be > 0.5

# Delta retention should be positive
retention["delta_retention"].mean()  # Should be > 0
```

If `delta_retention` mean is negative, your baseline is wrong or embedding mismatch.

---

## C) Code Fixes Implemented

✅ **1. Safe JSON tag parsing** - `safe_load_tags_json()` prevents crashes on NaN/None

✅ **2. Fixed pairwise comparisons** - Positional loops instead of `iterrows()` indices

✅ **3. Jaccard on tag_norm_eval** - `jaccard_from_norm_eval()` aligns with Phase 2 spec

✅ **4. Cache existence checks** - `--skip-embeddings` fails fast if cache missing

✅ **5. Block F mean distances** - Reports means, not variances (matches spec)

✅ **6. Derangement for shuffled baseline** - Avoids self-matches

✅ **7. Batch embeddings** - 128 texts per batch (critical for performance)

---

## D) "Phase 2 Plots" Script Requirements

Create `scripts/phase2_plots.py` that:

1. Loads all tables from `results/phase2/tables/`
2. Generates the 6 plots:
   - **Run stability:** Violin + ECDF or just ECDF
   - **Prompt sensitivity heatmap:** Mean cosine per (prompt1, prompt2)
   - **Model sensitivity heatmap:** Mean cosine per (model1, model2)
   - **Retention:** Bar/box by model×prompt
   - **Pareto:** Scatter retention vs cost (color by model, marker by prompt)
   - **Surface vs semantic:** Scatter `jaccard_norm_eval` vs `cosine_similarity`
3. Saves 6 PNG files to `results/phase2/plots/`

**Success check:** Script ends with 6 PNG files saved.

---

## E) Results Framing (One-Page Summary)

You must be able to report:

1. **Run stability (semantic)** is high/medium/low and differs by model/prompt
2. **Prompt sensitivity** exists but is smaller/larger than run variance
3. **Model sensitivity** is the largest driver (or not)
4. **Retention** is meaningfully above baselines (random/shuffled/topK)
5. **Compression:** Best model/prompt is on the Pareto front
6. **Dispersion** correlates with:
   - Review sparsity (few reviews)
   - Generic venues (if true)
   - Other interpretable factors

That's "uncertainty signal": dispersion = reliability proxy.

---

## F) Minimal "Done" Commands

When everything is wired:

```bash
# Run full analysis (computes embeddings)
poetry run python scripts/phase2_analysis.py \
  --run-id week2_run_20251223_191104 \
  --data data/study1_venues_20250117.csv

# Verify cache reload works
poetry run python scripts/phase2_analysis.py \
  --run-id week2_run_20251223_191104 \
  --skip-embeddings

# Generate plots
poetry run python scripts/phase2_plots.py
```

**If those run clean and produce all outputs with sane row counts, Phase 2 is complete.**

---

## G) Testing Sequence

1. **Test on 10 venues first:**
   ```bash
   # Modify script to sample 10 venues, run analysis
   # Verify tables have expected structure
   ```

2. **Run full analysis:**
   ```bash
   poetry run python scripts/phase2_analysis.py --run-id <run_id>
   ```

3. **Verify row counts match expectations**

4. **Check sanity ranges (cosine, delta_retention, etc.)**

5. **Generate plots:**
   ```bash
   poetry run python scripts/phase2_plots.py
   ```

6. **Review plots for obvious errors (all zeros, all ones, etc.)**

---

## H) Common Failure Modes

### Row counts too small
- **Cause:** Missing embeddings (exp_id mismatch, failed extractions)
- **Fix:** Check `extraction_embeddings` dict keys match `extractions_df["exp_id"]`

### All cosine similarities = 0.0
- **Cause:** Embedding dimension mismatch or zero vectors
- **Fix:** Check `EMBEDDING_DIM` matches model output (3072 for text-embedding-3-large)

### Delta retention negative
- **Cause:** Random baseline too high or embedding mismatch
- **Fix:** Verify random tags are truly random, check embedding alignment

### Cache reload fails
- **Cause:** NPZ file corruption or mapping mismatch
- **Fix:** Delete cache and recompute, check file sizes

---

## I) Next Steps After Phase 2

1. ✅ Analysis script complete
2. ⏳ Create plotting script (`phase2_plots.py`)
3. ⏳ Test on small subset (10 venues)
4. ⏳ Run full analysis
5. ⏳ Generate paper-ready plots
6. ⏳ Write results section

---

**Last Updated:** 2025-01-XX
**Status:** Code fixes complete, plotting script pending

