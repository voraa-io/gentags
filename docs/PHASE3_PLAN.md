# Phase 3: Representation Comparison for Semantic State Observability

**Status:** ✅ Complete
**Depends on:** Phase 2 (Stability) ✅ Complete
**Last Updated:** 2026-01-25

---

## Results Summary

### Block G: Localization / Attribution ✅

| Metric | Gentags | Embeddings | Interpretation |
|--------|---------|------------|----------------|
| **Gini coefficient** | **0.657** | **0.361** | Gentags: localized change |
| Gini difference | +0.297 | — | Gentags more localized |
| % gentag > embedding | **90.1%** | — | Highly consistent |
| Wilcoxon p-value | **< 0.001** | — | Statistically significant |

**Key Finding:** Gentags enable ATTRIBUTABLE change detection. When semantic state changes, gentags show *which facet* changed. Embeddings only show *that* something changed (diffuse).

### Block H: Cost Comparison ✅

| Method | Cost Type | Per-Venue Cost | Notes |
|--------|-----------|----------------|-------|
| Gentags | One-time | ~$0.001-0.01 | Extract once, query unlimited |
| Model-in-loop | Per-query | $0.0057 per venue | 1000 queries = $0.28 |
| Embeddings | Storage only | Negligible | No attribution capability |

### Model-in-Loop Stability ✅

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Exact match rate | **31.6%** | LLM responses NOT stable |
| No-info agreement | 95.0% | Consistent on "no info" |
| Mean length ratio | 0.855 | Response length varies |

**Key insight:** Model-in-loop has NO persistent state and is NOT stable across runs (only 31.6% exact match). Gentags provide stable, persistent semantic state.

### Block I: Cold-Start Analysis ✅

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Correlation (reviews vs variability) | **-0.317** | More evidence → less variability |
| Sparse venues variability | Higher | Uncertainty signal works |

---

## The Core Problem: Belief States for Autonomous Systems

**This is NOT about:**
- ❌ User search queries
- ❌ Retrieval benchmarks
- ❌ Ranking venues for humans
- ❌ Recommender UX

**This IS about:**
- ✅ Self-operating information systems
- ✅ Semantic belief representations
- ✅ State observability for control
- ✅ Attribution and monitoring

### The System Model

An LLM-based autonomous system has:
- **Environment:** Documents, reviews, logs, sensors
- **Goal:** Implicit or explicit decision-making
- **Constraints:** Limited context window, limited compute, costs for querying, costs for mistakes

**The key problem:**
> The system must decide when it knows enough to act and when it must seek more information.

That's **control**.

Gentags are NOT for retrieval UX. They are for:
> **Externalizing semantic state so the system can monitor itself.**

---

## What is a Belief State?

Given raw evidence:
```
E = {review_1, review_2, ..., review_n}
```

A representation extracts a **belief state**:
```
B = f(E)
```

This belief state is what the system uses to:
- Make decisions
- Detect change
- Attribute causes
- Monitor uncertainty

Different representations produce different belief states with different properties.

---

## The Four Systems (Correct Framing)

Each system produces a semantic belief representation from the same evidence.

### System A: Raw Evidence (Full Text)

**Belief state:**
```
B_raw = full review text (concatenated)
```

**Properties:**
| Property | Value |
|----------|-------|
| Storage | Huge (full text) |
| Structure | Unstructured |
| Reasoning cost | High (must re-read everything) |
| Uncertainty signal | None explicit |
| Attribution | None |

**Control implication:**
- LLM must re-read everything every time
- No persistent memory
- No change detection
- No monitoring possible

---

### System B: Dense Embedding

**Belief state:**
```
B_embed = vector(E) ∈ ℝ^3072
```

**Properties:**
| Property | Value |
|----------|-------|
| Storage | Compact (fixed size) |
| Structure | Dense, continuous |
| Reasoning cost | Low (similarity search) |
| Uncertainty signal | None |
| Attribution | **None** |

**Control implication:**
- You can observe: `||B_t - B_{t-1}||` (scalar drift)
- But you **cannot** answer: "What changed?"
- No semantic attribution
- No factorization
- Change is **entangled** — you see distance, not cause

---

### System C: Model-in-the-Loop (No Persistent State)

**Belief state:**
```
B = None (nothing stored)
```

**Every operation:**
```
LLM(E, goal) → action
```

**Properties:**
| Property | Value |
|----------|-------|
| Storage | None |
| Structure | N/A |
| Reasoning cost | **Highest** (full LLM call every time) |
| Uncertainty signal | None persistent |
| Attribution | Per-query only |

**Control implication:**
- **Impossible to do process control**
- Cannot detect drift (nothing is stored between calls)
- Cannot monitor state over time
- Every decision requires full re-computation

**What this means concretely:**

When we say "model-in-the-loop with no persistent representation," we mean:

1. **No pre-computed summary exists** — The system stores only raw evidence
2. **Every decision requires a fresh LLM call** — To answer "is this venue quiet?", the system must:
   - Load all reviews into context
   - Call the LLM with a prompt like: "Based on these reviews, is this venue quiet?"
   - Get a response
   - Discard the reasoning (nothing persisted)
3. **Repeated queries = repeated costs** — Asking 10 questions about the same venue = 10 full LLM calls
4. **No state tracking** — If a new review arrives, the system cannot say "what changed?" — it must re-derive everything

**Example comparison:**

| Operation | Gentags | Model-in-the-Loop |
|-----------|---------|-------------------|
| "Is venue quiet?" | Check if "quiet" ∈ tags | LLM call with all reviews |
| "What changed since yesterday?" | Diff tag sets | **Impossible** (no prior state) |
| "Which aspects are uncertain?" | Check tag variance | **Impossible** |
| 100 questions about same venue | 100 tag lookups | 100 LLM calls |

---

### System D: Gentags (Our Method)

**Belief state:**
```
B_tags = {tag_1, tag_2, ..., tag_k}
```

Plus optional pooled embedding for similarity.

**Properties:**
| Property | Value |
|----------|-------|
| Storage | Moderate (tags + optional embedding) |
| Structure | **Discrete, compositional, factorized** |
| Reasoning cost | Low (tag lookup + similarity) |
| Uncertainty signal | **Yes** (variance across extractions) |
| Attribution | **Yes** (tags map to facets) |

**Control implication:**
- Observable semantic variables
- Change detection: "quiet disappeared, crowded appeared"
- Uncertainty proxies: high variance = low confidence
- Attribution: know **what** changed, not just **that** something changed

---

## Why Localization Matters (The Key Insight)

When evidence changes:

**Embeddings:**
```
vector moves by δ = 0.15
```
You don't know why. Was it service? Coffee? Ambiance? All of them?

**Gentags:**
```
- "quiet atmosphere" → removed
+ "crowded afternoons" → added
- "friendly staff" → removed
+ "slow service" → added
```
Now you know **exactly what changed**.

That's **localization**.
That's **state observability**.
That's what makes **control possible**.

---

## Phase 3 Core Question

> **Which representation supports monitoring, attribution, and uncertainty-aware control?**

We evaluate along four axes:

| Axis | Question |
|------|----------|
| **Compactness** | Storage + compute cost |
| **Interpretability** | Can the system (and humans) inspect the belief state? |
| **Stability** | Does the representation change spuriously? (Phase 2) |
| **State Observability** | Can we localize semantic change? (Key experiment) |

---

## Comparison Dimensions

### Dimension 1: Cost Analysis

**Question:** What is the compute/storage tradeoff for each representation?

**Metrics:**
- Extraction cost ($/venue)
- Storage cost (bytes/venue)
- Per-operation cost ($/decision)
- Total cost at scale (1K, 10K, 100K venues)

**Data needed:**
- Token counts from Phase 1
- API pricing (already have)
- Storage measurements

**Output:**
- `cost_comparison.csv`
- Cost Pareto plot

### Dimension 2: State Observability / Localization

**This is the key differentiator.**

**Why it matters for autonomous systems:**

An autonomous system needs to answer:
- "What do I currently believe about this venue?"
- "What changed since my last observation?"
- "Which specific aspect changed?"
- "Should I seek more information?"

**Dense embeddings cannot answer these questions.** You only see `||B_t - B_{t-1}|| = 0.15`. Was it service? Coffee? Ambiance? Unknown.

**Gentags can answer these questions.** You see: `"quiet" removed, "crowded" added`. Now the system knows exactly what changed and can reason about it.

**Core Claim:**
> Dense embeddings entangle multiple semantic factors; gentags externalize and factorize them, enabling localized change attribution.

#### Experiment: Semantic Localization Test

**Goal:** Show that with embeddings, change is diffuse; with gentags, change is localized.

**Control system implication:** If a system cannot localize change, it cannot make targeted decisions about what to investigate or update.

**Step A — Define Semantic Facets (10 facets)**

| Facet | Description | Example Tags |
|-------|-------------|--------------|
| food_quality | Food taste, freshness | "fresh pastries", "bland food" |
| coffee_drinks | Coffee, beverages | "great espresso", "weak coffee" |
| service | Staff interaction, speed | "friendly staff", "slow service" |
| ambiance | Atmosphere, vibe | "cozy atmosphere", "noisy" |
| price_value | Cost, value for money | "affordable", "overpriced" |
| crowding | Busy-ness, wait times | "always crowded", "no wait" |
| seating | Indoor/outdoor, comfort | "outdoor seating", "cramped tables" |
| dietary | Vegan, allergies, options | "vegan options", "gluten-free" |
| portions | Size, quantity | "generous portions", "small servings" |
| location | Accessibility, parking | "easy parking", "hard to find" |

**Step B — Assign Tags to Facets**

Option 1: Keyword mapping (deterministic) — **Recommended**

```python
FACETS = [
    "food_quality", "coffee_drinks", "service", "ambiance",
    "price_value", "crowding", "seating", "dietary", "portions", "location"
]

FACET_KEYWORDS = {
    "food_quality": ["food", "fresh", "tasty", "delicious", "bland", "meal", "breakfast", "lunch"],
    "coffee_drinks": ["coffee", "espresso", "latte", "tea", "drink", "beverage", "cappuccino"],
    "service": ["staff", "service", "friendly", "rude", "slow", "fast", "waiter", "barista"],
    "ambiance": ["atmosphere", "vibe", "cozy", "noisy", "quiet", "decor", "music", "lighting"],
    "price_value": ["price", "expensive", "cheap", "affordable", "value", "worth", "overpriced"],
    "crowding": ["crowded", "busy", "wait", "line", "packed", "empty", "quiet"],
    "seating": ["seating", "outdoor", "patio", "table", "chair", "space", "indoor"],
    "dietary": ["vegan", "vegetarian", "gluten", "allergy", "organic", "healthy"],
    "portions": ["portion", "size", "generous", "small", "large", "filling"],
    "location": ["location", "parking", "accessible", "downtown", "corner", "find"],
}

def assign_facet(tag: str) -> str:
    """Assign a tag to a facet based on keyword matching."""
    tag_lower = tag.lower()
    for facet, keywords in FACET_KEYWORDS.items():
        if any(kw in tag_lower for kw in keywords):
            return facet
    return "other"
```

Option 2: Zero-shot LLM classifier (for validation)
```
Given this tag: "{tag}"
Classify into one of: food_quality, coffee_drinks, service, ambiance, price_value, crowding, seating, dietary, portions, location, other
```

**Recommendation:** Start with keyword mapping for speed; validate with LLM on sample of 100 tags.

**Step C — Compute Per-Facet Drift**

For each representation pair (run1 vs run2, prompt1 vs prompt2, model1 vs model2):

**Gentags — Per-facet Jaccard distance:**

```python
def compute_gentag_facet_drift(tags1: List[str], tags2: List[str]) -> np.ndarray:
    """
    Compute per-facet drift between two tag sets.
    Returns: array of drift values per facet (0-1 scale)
    """
    drift = []

    for facet in FACETS:
        # Get tags belonging to this facet
        facet_tags1 = set(t for t in tags1 if assign_facet(t) == facet)
        facet_tags2 = set(t for t in tags2 if assign_facet(t) == facet)

        # Compute Jaccard distance (1 - similarity) for this facet
        if len(facet_tags1) == 0 and len(facet_tags2) == 0:
            facet_drift = 0.0  # No tags in this facet, no change
        else:
            intersection = len(facet_tags1 & facet_tags2)
            union = len(facet_tags1 | facet_tags2)
            jaccard = intersection / union if union > 0 else 0
            facet_drift = 1.0 - jaccard  # Convert to distance

        drift.append(facet_drift)

    return np.array(drift)

# Example:
# tags1 = ["great coffee", "friendly staff", "cozy vibe"]
# tags2 = ["great coffee", "slow service", "cozy vibe"]  # Only service changed
#
# Result: [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]
#          ^food ^coffee ^service ^ambiance ...
#
# Change is LOCALIZED to service facet → High Gini
```

**Embeddings — Anchor similarity method:**

```python
# Pre-compute anchor embeddings (once)
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

# Embed all anchors (once at startup)
anchor_embeddings = {facet: embed(text) for facet, text in FACET_ANCHORS.items()}

def compute_embedding_facet_drift(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute per-facet drift between two venue embeddings.
    Uses anchor similarity to project onto facet dimensions.
    """
    drift = []

    for facet in FACETS:
        anchor_emb = anchor_embeddings[facet]

        # Compute similarity to anchor for each embedding
        sim1 = cosine_similarity(emb1, anchor_emb)
        sim2 = cosine_similarity(emb2, anchor_emb)

        # Drift = absolute change in facet similarity
        facet_drift = abs(sim1 - sim2)
        drift.append(facet_drift)

    return np.array(drift)

# Example:
# emb1 = embedding of "great coffee, friendly staff, cozy vibe"
# emb2 = embedding of "great coffee, slow service, cozy vibe"
#
# Result: [0.02, 0.01, 0.15, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01]
#          ^food ^coffee ^service ^ambiance ...
#
# Change is DIFFUSE across all facets → Low Gini
# (embeddings entangle everything, so changing "service" affects all dimensions)
```

**Step D — Localization Metric (Gini Coefficient)**

The Gini coefficient measures how concentrated the drift is:
- **High Gini (→1):** Change concentrated in few facets (LOCALIZED)
- **Low Gini (→0):** Change spread evenly across all facets (DIFFUSE)

```python
def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of a distribution.

    - High Gini (→1): Change concentrated in few facets (LOCALIZED)
    - Low Gini (→0): Change spread evenly (DIFFUSE)
    """
    values = np.abs(values)
    if values.sum() == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)

    # Gini formula
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return gini

# Example:
# Gentag drift:    [0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]  → Gini ≈ 0.9 (localized)
# Embedding drift: [0.02, 0.01, 0.15, 0.03, ...]     → Gini ≈ 0.3 (diffuse)
```

**Step E — Full Comparison Loop**

```python
def compute_localization_comparison(pairs_df, tags_df, extraction_embs):
    """
    For each comparison pair (run1 vs run2), compute localization for both
    gentags and embeddings.
    """
    results = []

    for _, row in tqdm(pairs_df.iterrows(), desc="Localization"):
        venue_id = row["venue_id"]
        exp_id1 = row["exp_id1"]
        exp_id2 = row["exp_id2"]

        # Get tags for each extraction
        tags1 = get_tags_for_extraction(tags_df, exp_id1)
        tags2 = get_tags_for_extraction(tags_df, exp_id2)

        # Get embeddings for each extraction
        emb1 = extraction_embs[exp_id1]
        emb2 = extraction_embs[exp_id2]

        # Compute per-facet drift
        gentag_drift = compute_gentag_facet_drift(tags1, tags2)
        embedding_drift = compute_embedding_facet_drift(emb1, emb2)

        # Compute localization (Gini)
        gentag_gini = gini_coefficient(gentag_drift)
        embedding_gini = gini_coefficient(embedding_drift)

        results.append({
            "venue_id": venue_id,
            "pair_type": row.get("pair_type", "run"),
            "gentag_gini": gentag_gini,
            "embedding_gini": embedding_gini,
        })

    return pd.DataFrame(results)
```

**Step F — Statistical Comparison**

```python
# After running on all pairs:
print(f"Gentag Gini (mean): {results['gentag_gini'].mean():.3f}")
print(f"Embedding Gini (mean): {results['embedding_gini'].mean():.3f}")

# Statistical test
from scipy.stats import mannwhitneyu
stat, pvalue = mannwhitneyu(results['gentag_gini'], results['embedding_gini'])
print(f"Mann-Whitney U p-value: {pvalue:.2e}")
```

**Expected Results:**

| Representation | Mean Gini | Interpretation |
|----------------|-----------|----------------|
| Gentags | High (~0.6-0.8) | Change is localized to few facets |
| Embeddings | Low (~0.2-0.4) | Change is diffuse across all facets |

**Visual Summary:**

```
GENTAGS:                          EMBEDDINGS:

Facet Drift Vector:               Facet Drift Vector:
[0, 0, 0.9, 0, 0, 0, 0, 0, 0, 0]  [0.05, 0.08, 0.12, 0.06, 0.04, ...]
     ↑                                 ↑    ↑    ↑    ↑    ↑
     service changed                   everything changes a little

Gini = 0.9 (LOCALIZED)            Gini = 0.3 (DIFFUSE)
```

**Why This Works:**

1. **Gentags are explicit:** Each tag maps to a specific facet. If only service tags change, only the service facet shows drift.

2. **Embeddings are entangled:** The vector encodes everything together. Changing "friendly staff" to "slow service" shifts the entire vector, affecting similarity to ALL anchors.

This is the **attribution/localization** difference we want to demonstrate.

**Output:**
- `localization.csv` — per-pair Gini scores
- Histogram plot: gentags vs embeddings localization distribution
- 2-3 qualitative examples showing facet-level attribution

### Dimension 3: Cold-Start / Uncertainty Behavior

**Question:** How do representations behave when evidence is sparse?

**Why it matters for autonomous systems:**

An autonomous system encountering a new venue with only 1-2 reviews must:
- Form an initial belief state
- Know how confident that belief is
- Decide whether to seek more information

**The key difference:**

| System | Sparse Data Behavior |
|--------|---------------------|
| Raw text | Works, but no confidence signal |
| Embeddings | Produces a vector, but **no uncertainty signal** |
| Model-in-the-loop | Can say "I'm uncertain" per-query, but **no persistent state** |
| **Gentags** | High variance across extractions = **explicit uncertainty signal** |

**Approach:**
- Use S4 sparsity results from Phase 2 (correlation = -0.230)
- Show: variability increases with sparse data
- This variability IS the uncertainty signal

**Control implication:**
- Low variance → high confidence → act
- High variance → low confidence → seek more information

**Metrics:**
- Variability at different sparsity levels
- Retention at different sparsity levels
- Stability at different sparsity levels

**Output:**
- Cold-start comparison plot
- Recommendation for minimum evidence threshold

### Dimension 4: Stability

**Already completed in Phase 2.**

- Run stability: ✅ Cosine 0.977
- Prompt sensitivity: ✅ High semantic similarity
- Model sensitivity: ✅ Models agree
- Sparsity correlation: ✅ -0.230

**For Phase 3:** Compare stability metrics across baselines.

---

## Implementation Plan

### Block G: Localization / Attribution

**Input:**
- Phase 2 outputs (run_stability, prompt_sensitivity, model_sensitivity)
- Phase 1 tags with facet assignments

**Steps:**
1. Load existing comparison pairs from Phase 2
2. Assign tags to facets (keyword or LLM)
3. Compute per-facet drift for gentags
4. Compute per-facet drift for embeddings (anchor method)
5. Compute Gini coefficient for each pair
6. Compare distributions

**Output:**
- `results/phase3/tables/facet_assignments.csv`
- `results/phase3/tables/localization.csv`
- `results/phase3/plots/localization_comparison.png`

### Block H: Cost Comparison

**Input:**
- Phase 1 extraction costs
- Token counts
- API pricing

**Output:**
- `results/phase3/tables/cost_comparison.csv`
- `results/phase3/plots/cost_pareto.png`

### Block I: Cold-Start Analysis

**Input:**
- Phase 2 sparsity analysis
- Retention by sparsity level

**Output:**
- `results/phase3/tables/cold_start.csv`
- `results/phase3/plots/cold_start_comparison.png`

---

## Expected Deliverables

### Tables
- `facet_assignments.csv` — tag → facet mapping
- `localization.csv` — per-pair Gini scores (gentags vs embeddings)
- `cost_comparison.csv` — cost by representation type
- `cold_start.csv` — performance by sparsity level

### Plots
1. **Localization histogram** — Gini distribution: gentags vs embeddings
2. **Facet drift heatmap** — example venues showing per-facet change
3. **Cost Pareto front** — cost vs quality tradeoff
4. **Cold-start curves** — performance vs evidence amount

### Qualitative Examples
- 2-3 venues showing facet-level attribution
- Side-by-side: "what changed?" for gentags vs embeddings

---

## Key Claims to Support

### For Autonomous Systems / Control

1. **State Observability:** Gentags externalize semantic state as inspectable, discrete variables. Embeddings hide state in opaque vectors.

2. **Change Attribution:** When evidence changes, gentags show **what** changed (which facets). Embeddings only show **that** something changed (scalar distance).

3. **Uncertainty Signal:** Gentag variance across extractions provides an uncertainty proxy. High variance = low confidence = seek more information.

4. **Persistent State:** Unlike model-in-the-loop, gentags provide persistent belief state that enables monitoring, drift detection, and process control.

### For Representation Quality

5. **Semantic Stability:** (From Phase 2) Gentags are lexically unstable but semantically stable. Surface variation, same meaning.

6. **Cost-effectiveness:** Gentags offer favorable cost-quality tradeoff. One-time extraction vs. repeated LLM calls.

---

## Safe Phrasing for Paper

**Use:**
> "Dense embeddings encode meaning in a distributed manner, making it difficult to attribute representational change to specific semantic factors. Gentags provide an explicit factorized representation, enabling localized analysis of semantic drift."

> "For autonomous decision systems, state observability is critical. Gentags externalize semantic belief as discrete, inspectable variables, supporting monitoring, attribution, and uncertainty-aware control."

**Avoid:**
- "Embeddings are bad"
- "Embeddings cannot explain"
- "Gentags outperform embeddings"
- Retrieval/search framing
- User query framing

---

## Timeline

| Block | Description | Complexity |
|-------|-------------|------------|
| G | Localization / Attribution | Medium |
| H | Cost Comparison | Low |
| I | Cold-Start Analysis | Low |

**Recommended order:** G → H → I

---

## Scripts to Create

- `scripts/phase3_analysis.py` — Main analysis (Blocks G, H, I)
- `scripts/phase3_plots.py` — Plot generation

---

## Questions to Resolve

1. **Facet assignment method:** Keyword mapping vs LLM classifier?
   - Recommendation: Start with keywords, validate on 100 tags with LLM

2. **Number of facets:** 10 sufficient?
   - Recommendation: Yes, 10 is standard and covers main dimensions

3. **Anchor texts for embeddings:** How to construct?
   - Recommendation: Short, descriptive phrases per facet

4. **Minimum venues for localization test:**
   - Recommendation: Use all 230 venues with complete data

---

## Connection to pdensity

The localization analysis directly supports pdensity:

- **High pdensity tags:** Touch many facets simultaneously
- **Low pdensity tags:** Affect only one facet

This can be shown qualitatively:
- "cozy cafe with great espresso" → affects ambiance + coffee (high pdensity)
- "slow service" → affects only service (low pdensity)

No need to compute pdensity scores yet — just demonstrate the concept.
