# Phase 3 Anchor Fix Plan

**Date:** 2026-02-01
**Priority:** CRITICAL
**Issue:** Circular reasoning in anchor computation

---

## The Problem

### Current Implementation (BROKEN)

In `scripts/phase3_analysis.py`, the `compute_anchor_embeddings()` function:

1. **Averages tag embeddings to create anchors** (circular reasoning)
2. **Uses random vectors for facets with no coverage** (noise injection)

```python
# CURRENT (BAD) - lines 291-316
def compute_anchor_embeddings(tag_embeddings):
    for facet in FACETS:
        facet_tags = [tag for tag in tag_embeddings.keys() if assign_facet(tag) == facet]
        if len(facet_tags) >= 5:
            anchor_embeddings[facet] = mean_pool([...])  # CIRCULAR!
        else:
            anchor_embeddings[facet] = np.random.randn(...)  # NOISE!
```

### Why This Is Wrong

1. **Circularity:** Using tags to define anchors, then measuring tag localization against those anchors = self-fulfilling prophecy
2. **Noise:** Random vectors for missing facets dilute the Gini measurement
3. **Inconsistency:** Phase 3A and 3B use fixed phrases, but Phase 3 uses tag averaging

---

## The Fix

### Correct Implementation (from Phase 3A/3B)

Use the fixed descriptive phrases defined in `FACET_ANCHORS`:

```python
FACET_ANCHORS = {
    "food_quality": "food quality, taste, freshness, delicious meals",
    "coffee_drinks": "coffee, espresso, latte, beverages, drinks",
    "service": "service quality, staff friendliness, speed, waiters",
    "ambiance": "atmosphere, ambiance, vibe, decor, cozy environment",
    "price_value": "price, value for money, affordable, expensive",
    "crowding": "crowded, busy, wait times, lines, availability",
    "seating": "seating, tables, outdoor patio, indoor space",
    "dietary": "dietary options, vegan, vegetarian, gluten-free",
    "portions": "portion size, generous servings, filling meals",
    "location": "location, parking, accessibility, neighborhood",
}

# CORRECT - Embed fixed phrases once
def compute_anchor_embeddings_fixed(openai_client) -> Dict[str, np.ndarray]:
    """Compute anchor embeddings from fixed descriptive phrases."""
    anchor_embeddings = {}
    for facet, text in FACET_ANCHORS.items():
        anchor_embeddings[facet] = embed_text(openai_client, text)
    return anchor_embeddings
```

### Benefits

1. **No circularity:** Anchors are defined independently of any method's output
2. **No noise:** Every facet has a meaningful embedding
3. **Method-neutral:** Same yardstick for Gentags, Embeddings, RAKE, TF-IDF

---

## Action Items

### 1. Fix `phase3_analysis.py`

- [ ] Replace `compute_anchor_embeddings()` with fixed phrase embedding
- [ ] Add OpenAI client initialization for embedding
- [ ] Remove random fallback logic

### 2. Re-run Phase 3 Analysis

```bash
poetry run python scripts/phase3_analysis.py
```

### 3. Update Results

- [ ] Check if Gini numbers change significantly
- [ ] Update `docs/PHASE3_ANALYSIS_REPORT.md`
- [ ] Update `docs/GENTAGS_FULL_ANALYSIS_REPORT.md`
- [ ] Update `README.md` if numbers change

### 4. Verify Consistency

Ensure all scripts use the same anchor approach:

| Script | Anchor Source | Status |
|--------|---------------|--------|
| `phase3_analysis.py` | Fixed phrases | TODO: Fix |
| `phase3a_baselines.py` | Fixed phrases | ✅ Correct |
| `phase3b_robustness.py` | Fixed phrases | ✅ Correct |

---

## Expected Impact

### Best Case
- Gini numbers remain similar → Original finding stands, methodology now bulletproof

### Possible Case
- Gini advantage decreases but remains significant (e.g., 3x instead of 5x) → Adjust claims

### Worst Case
- Gini advantage disappears → Major revision needed, contribution shifts

---

## Reviewer Shield (After Fix)

> "Facet anchors are defined as fixed descriptive phrases (e.g., 'food quality, taste, freshness') embedded using text-embedding-3-large. These anchors are external to any extraction method, ensuring method-neutral evaluation. All representations (Gentags, Embeddings, RAKE, TF-IDF) are projected into the same fixed semantic space."

---

*Plan created: 2026-02-01*
