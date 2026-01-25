# Issue 001: Phase 2 Analysis Performance Bottleneck

**Created:** 2026-01-23
**Status:** Resolved (filtering for successful extractions implemented)
**Severity:** High (was blocking analysis)

---

## The Research Problem: Why Are We Doing This?

### Core Question

> **Are gentags a legitimate semantic representation, or just noisy text fragments?**

When we extract semantic tags from reviews using LLMs, the output varies across runs, prompts, and models. This variation could mean:

1. **Bad:** The extraction is unreliable noise — different runs give random, unrelated tags
2. **Good:** The extraction captures stable meaning but expresses it in different surface forms (paraphrases)

We need to prove it's #2, not #1.

### The Key Claim We Must Prove

> **"Gentags are lexically unstable but semantically stable"**

This means:
- **Low Jaccard similarity** (different words/phrases across runs) — expected and acceptable
- **High cosine similarity** (same meaning in embedding space) — this is what makes them useful

If we can show the gap between surface variation and semantic stability, we prove gentags are a coherent representation layer.

---

## Why Each Analysis Block Exists

### S1: Run Stability (same model, same prompt, different runs)

**Question:** If I run the same extraction twice, do I get the same meaning?

**Why it matters:**
- If run-to-run variation is semantic (not just lexical), gentags are unreliable
- If variation is only lexical (paraphrases), gentags are stable representations

**Expected outcome:**
- Jaccard: 0.3-0.5 (moderate surface overlap — different phrasings)
- Cosine: 0.9+ (high semantic overlap — same ideas)
- Gap: 0.4+ (proves surface ≠ semantic)

### S2: Prompt Sensitivity (same model, different prompts)

**Question:** Do prompts change what meaning is extracted, or only the granularity/style?

**Why it matters:**
- If prompts change core semantics, prompt choice is critical and results are fragile
- If prompts only change resolution (more/fewer tags), core meaning is robust

**Expected outcome:**
- `anti_hallucination` → more tags, higher granularity, more grounded
- `short_phrase` → fewer tags, more compressed
- Semantic similarity across prompts should still be high (0.8+)

### S3: Model Sensitivity (same prompt, different models)

**Question:** Do different LLMs extract the same latent semantics, or invent different worlds?

**Why it matters:**
- If models agree semantically, gentags reflect shared cultural/linguistic priors
- If models diverge, gentags are model-specific artifacts (less useful)

**Expected outcome:**
- Some style differences (verbosity, granularity)
- Core semantic dimensions overlap strongly
- No single "best" model — they're different lenses on the same meaning

### S4: Stability Under Sparsity (less evidence → more uncertainty)

**Question:** As textual evidence decreases, does representation variability increase?

**Why it matters:**
- This is the uncertainty signal — variability should increase with less evidence
- High variability = low confidence = information need
- This is NOT noise — it's a feature that exposes where more data is needed

**Original approach (problematic):**
- Subsample reviews (5, 3, 1 per venue)
- Problem: Most venues only have ~5 reviews anyway, not enough range
- Would require re-running extractions ❌

**Better approach: Token count as sparsity measure (NO NEW EXTRACTIONS)**
- Measure total tokens in accumulated reviews per venue
- Group venues by token count buckets (e.g., <500, 500-1000, 1000-2000, 2000+)
- Analyze how representation variability correlates with evidence amount
- Uses existing data — no new extractions needed

**Why token count is better than review count:**
- 5 reviews saying "Great!" = very sparse evidence
- 5 detailed reviews with 500 words each = rich evidence
- Token count captures actual information content

**Expected outcome:**
- Low token venues → higher within-venue variability (more uncertainty)
- High token venues → lower variability (more constrained representation)
- Variability is interpretable as uncertainty, not error

**Data already exists (no new extractions needed):**
```
results/phase1_downloaded/           # All extractions complete
├── *_extractions_*.csv              # 13,272 extractions (4 models × 3 prompts × 2 runs × 553 venues)
└── *_tags_*.csv                     # 230,151 tag rows

data/study1_venues_20250117.csv      # Source: google_reviews column with full text

results/phase2/tables/
└── uncertainty_dispersion.csv       # Per-venue variability (already computed)
```
- Just need to compute token count per venue and join with variability

**Implementation:**
```python
# For each venue, compute:
# 1. Total tokens across all reviews (sum of len(review.split()) for each review)
# 2. Mean pairwise distance across all 24 representations (already in uncertainty_dispersion.csv)
# 3. Correlation between tokens and variability

# Expected correlation: negative (more tokens → less variability)
# This proves variability is uncertainty signal, not noise
```

---

## What Success Looks Like

| Metric | Target | Meaning |
|--------|--------|---------|
| Run cosine | > 0.9 | Same meaning across runs |
| Run Jaccard | 0.3-0.6 | Surface variation (expected) |
| Cosine - Jaccard gap | > 0.3 | Proves lexical ≠ semantic |
| Prompt cosine | > 0.8 | Prompts don't change core meaning |
| Model cosine | > 0.7 | Models share latent semantics |
| Retention vs random | > +0.1 | Gentags capture review meaning |

---

## Current Results (After Filtering - 2026-01-24)

### Data After Filtering

```
Filtered out 2898 error extractions (21.8%)
   - claude: 1697 errors
   - grok: 1201 errors
Filtered to venues with all 4 models:
   - Venues: 553 → 230 (323 removed)
   - Extractions: 10374 → 5517
```

### Final Results ✅

```
KEY CLAIM: Lexically unstable but semantically stable
├─ Run stability - Jaccard (surface):  0.471 median
├─ Run stability - Cosine (semantic):  0.977 median
├─ Run stability - MMC (paraphrase):   0.887 median
└─ Gap (cosine - jaccard):             0.504

Retention - Mean: 0.625
Retention - Mean delta (vs random): 0.164
Redundancy - Mean rate: 0.001
Uncertainty - Mean pairwise distance: 0.051

✅ All success checks passed
```

**KEY CLAIM VALIDATED.** The **0.504 gap** between cosine and Jaccard proves that surface variation does not equal semantic variation.

### Results vs Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Run cosine | > 0.9 | **0.977** | ✅ |
| Run Jaccard | 0.3-0.6 | **0.471** | ✅ |
| Cosine - Jaccard gap | > 0.3 | **0.504** | ✅ |
| Retention vs random | > +0.1 | **+0.164** | ✅ |

### Issues Resolved

1. **Claude 0.0 metrics** — ✅ Fixed by filtering to complete venues only
2. **Cosines out of [-1, 1] range** — ✅ Fixed with `np.clip()` clamping
3. **77.1% cosine > jaccard** — ✅ Now passes (was dragged down by error extractions)

### Solution Applied

Modified `scripts/phase2_analysis.py` to:
1. Filter `extractions_df` to only `status="success"` rows
2. Filter to venues with successful extractions from **ALL 4 models**
3. Filter `tags_df` to only tags from complete venues
4. Added `np.clip()` to cosine similarity for floating point precision

---

## The Technical Problem (What Slowed Us Down)

### Summary

The Phase 2 stability analysis script was taking excessively long to run after adding new lexical diagnostic metrics.

---

## Current State

### Running Processes

| PID | CPU | Memory | Runtime | Memory (MB) |
|-----|-----|--------|---------|-------------|
| 80978 | 100% | 4.8% | 24:49 | 902 MB |
| 95038 | 98.7% | 0.5% | 14:07 | 91 MB |

### Command Being Run

```bash
poetry run python scripts/phase2_analysis.py \
  --run-id week2_run_20251223_191104 \
  --data data/study1_venues_20250117.csv \
  --results-dir results/phase1_downloaded \
  --skip-embeddings
```

---

## Root Cause Analysis

### The Problematic Function

**File:** `scripts/phase2_analysis.py`
**Function:** `compute_redundancy_rate()` (lines 299-320)

```python
def compute_redundancy_rate(
    tag_embs: List[np.ndarray],
    threshold: float = 0.9
) -> float:
    """
    Compute near-duplicate redundancy rate within an extraction.
    Returns the fraction of tag pairs with cosine > threshold.
    """
    if len(tag_embs) < 2:
        return 0.0

    n_pairs = 0
    n_redundant = 0

    for i in range(len(tag_embs)):
        for j in range(i + 1, len(tag_embs)):
            n_pairs += 1
            if cosine_similarity(tag_embs[i], tag_embs[j]) > threshold:
                n_redundant += 1

    return n_redundant / n_pairs if n_pairs > 0 else 0.0
```

### Why It's Slow

1. **O(n²) complexity per extraction** — For each tag set with `n` tags, computes `n*(n-1)/2` pairwise cosine similarities

2. **Called multiple times per comparison:**
   - `compute_run_stability()`: Called twice per comparison (run1, run2)
   - `compute_prompt_sensitivity()`: Called twice per comparison (prompt1, prompt2)
   - `compute_model_sensitivity()`: Called twice per comparison (model1, model2)

3. **Scale of the problem:**

| Function | Comparisons | Redundancy Calls | Tags per Set | Pairs per Call |
|----------|-------------|------------------|--------------|----------------|
| run_stability | ~6,636 | 13,272 | ~17 avg | ~136 |
| prompt_sensitivity | ~19,908 | 39,816 | ~17 avg | ~136 |
| model_sensitivity | ~19,908 | 39,816 | ~17 avg | ~136 |
| **Total** | ~46,452 | **92,904** | — | ~12.6M cosine ops |

4. **Pure Python loop** — No vectorization, no numpy broadcasting

### Estimated Time

- Each cosine similarity: ~0.1ms (with 3072-dim vectors)
- Total cosine operations: ~12.6 million
- Estimated time: ~21 minutes just for redundancy calculations

---

## Files Involved

| File | Role | Lines Changed |
|------|------|---------------|
| `scripts/phase2_analysis.py` | Main analysis script | Added ~100 lines |
| `results/phase2/tables/*.csv` | Output tables | Will be regenerated |
| `results/phase2_cache/*.npz` | Embedding cache | Already exists (not recomputed) |

### New Functions Added (causing slowdown)

1. **`compute_word_count_stats()`** (lines 279-297) — Fast, O(n)
2. **`compute_redundancy_rate()`** (lines 299-320) — SLOW, O(n²)

### Functions Modified

1. **`compute_run_stability()`** — Now calls `compute_redundancy_rate()` twice per comparison
2. **`compute_prompt_sensitivity()`** — Now calls `compute_redundancy_rate()` twice per comparison
3. **`compute_model_sensitivity()`** — Now calls `compute_redundancy_rate()` twice per comparison

---

## Data Scale

From `results/phase2/phase2_manifest.json`:

```json
{
  "counts": {
    "n_extractions": 13272,
    "n_tag_rows": 230151,
    "n_venues_in_reviews": 553,
    "n_venues_in_extractions": 553,
    "n_unique_tags_raw": 47976,
    "n_unique_tags_norm_eval": 35007
  }
}
```

- **13,272 extractions** across 4 models × 3 prompts × 2 runs × 553 venues
- **~17.3 tags per extraction** on average
- **35,007 unique normalized tags** with embeddings

---

## Solutions

### Option 1: Vectorize Redundancy Calculation (Recommended)

Replace Python loops with numpy matrix operations:

```python
def compute_redundancy_rate_fast(
    tag_embs: List[np.ndarray],
    threshold: float = 0.9
) -> float:
    if len(tag_embs) < 2:
        return 0.0

    # Stack into matrix (n_tags × embedding_dim)
    emb_matrix = np.vstack(tag_embs)

    # Normalize for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_normed = emb_matrix / (norms + 1e-10)

    # Compute all pairwise cosines at once
    cos_matrix = emb_normed @ emb_normed.T

    # Get upper triangle (excluding diagonal)
    upper_tri = np.triu(cos_matrix, k=1)

    # Count pairs above threshold
    n_pairs = len(tag_embs) * (len(tag_embs) - 1) // 2
    n_redundant = np.sum(upper_tri > threshold)

    return n_redundant / n_pairs
```

**Expected speedup:** 50-100x (seconds instead of minutes)

### Option 2: Make Redundancy Optional

Add `--skip-redundancy` flag to defer this computation:

```python
parser.add_argument(
    "--skip-redundancy",
    action="store_true",
    help="Skip redundancy rate calculation (slow)"
)
```

### Option 3: Sample-Based Estimation

Only compute redundancy for a sample of extractions:

```python
if compute_redundancy and (sample_rate == 1.0 or random.random() < sample_rate):
    redundancy = compute_redundancy_rate(embs)
else:
    redundancy = None
```

---

## Immediate Actions

1. **Kill running processes:**
   ```bash
   kill 80978 95038
   ```

2. **Apply vectorized fix** to `compute_redundancy_rate()`

3. **Re-run analysis** — should complete in 2-5 minutes instead of 30+

---

## Lessons Learned

1. **Profile before adding O(n²) operations** at scale
2. **Vectorize numpy operations** — Python loops over arrays are slow
3. **Test on subset first** — Run with `--sample-size 10` before full dataset
4. **Add timing logs** — Would have caught this earlier

---

## Related Files

- `scripts/phase2_analysis.py` — Main script (needs fix)
- `docs/PHASE2_PLAN.md` — Analysis requirements
- `results/phase2/phase2_manifest.json` — Previous successful run metadata
