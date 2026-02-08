# Phase 4-B: Probe Sensitivity (Making pdensity Science)

**Status:** Planning
**Depends on:** Phase 2 (Stability), Phase 3A (Baselines)
**Priority:** HIGH — Transforms pdensity from "vibes" to "science"
**Estimated Cost:** ~$0.05
**Estimated Time:** ~10-15 minutes execution

---

## Why This Experiment is Critical

### The Problem

Right now, **propositional density (pdensity)** is defined as:

> "The degree to which a tag constrains the semantic possibility space."

But this is an "interpretive construct" — we claim "weekday laptop café" is denser than "café" without proving it functionally.

### The Attack

A reviewer will ask:
> "How do you measure pdensity? Isn't it just word count?"

### The Defense

We measure pdensity via **ranking stability under perturbation**:
- Remove a high-density tag → large, structured ranking shift
- Remove a low-density tag → small, diffuse ranking shift

If this holds, pdensity is a **diagnostic probe** of how much a tag collapses the semantic possibility space.

---

## The Hard Question

> How do we prove "weekday laptop café" is denser than "café" without counting words?

**Answer:** By measuring its **functional impact** on downstream behavior.

| Tag | Word Count | Expected pdensity | Ranking Impact |
|-----|------------|-------------------|----------------|
| "café" | 1 | Low | Removing changes little |
| "weekday laptop café" | 3 | High | Removing changes ranking significantly |
| "quiet study spot" | 3 | High | Removing changes ranking significantly |
| "good" | 1 | Low | Removing changes little |

**Key insight:** pdensity is NOT word count. "Good" has 1 word but low pdensity. "Espresso" has 1 word but potentially high pdensity (if venue is known for espresso).

---

## Experimental Design

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROBE SENSITIVITY TEST                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  5 Semantic Probes (Constraint Bundles)                         │
│  ├── "quiet work environment" (laptop-friendly)                 │
│  ├── "high noise social" (bar/club vibe)                        │
│  ├── "premium coffee quality" (specialty coffee)                │
│  ├── "family outdoor casual" (kid-friendly)                     │
│  └── "late night available" (night owl)                         │
│                                                                 │
│  For each venue:                                                │
│  1. Compute similarity to each probe                            │
│  2. Rank venues by probe similarity                             │
│  3. Remove high-density tag → re-rank → measure Δ               │
│  4. Remove low-density tag → re-rank → measure Δ                │
│  5. Compare: high-density Δ should be >> low-density Δ          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Define Semantic Probes

These are "user intent" bundles — what someone might search for.

```python
SEMANTIC_PROBES = {
    "laptop_friendly": {
        "description": "Quiet environment suitable for remote work with laptop",
        "anchor_text": "quiet work environment laptop friendly wifi good seating",
        "expected_high_density_tags": ["laptop friendly", "quiet workspace", "good wifi"],
        "expected_low_density_tags": ["café", "coffee", "nice"]
    },
    "high_noise_social": {
        "description": "Loud, social atmosphere for groups and nightlife",
        "anchor_text": "loud music social atmosphere bar nightlife crowded energetic",
        "expected_high_density_tags": ["loud atmosphere", "great for groups", "nightlife"],
        "expected_low_density_tags": ["drinks", "fun", "good"]
    },
    "premium_coffee": {
        "description": "Specialty coffee with high-quality beans and preparation",
        "anchor_text": "specialty coffee espresso artisan roast barista quality beans",
        "expected_high_density_tags": ["specialty espresso", "artisan roast", "expert barista"],
        "expected_low_density_tags": ["coffee", "drinks", "good"]
    },
    "family_outdoor": {
        "description": "Family-friendly venue with outdoor seating for kids",
        "anchor_text": "family friendly kids outdoor patio casual relaxed spacious",
        "expected_high_density_tags": ["kid friendly", "outdoor patio", "family atmosphere"],
        "expected_low_density_tags": ["nice", "seating", "good"]
    },
    "late_night": {
        "description": "Available late at night for night owls",
        "anchor_text": "late night open late evening hours night owl after hours",
        "expected_high_density_tags": ["open late", "late night hours", "24 hour"],
        "expected_low_density_tags": ["hours", "open", "available"]
    }
}
```

### Step 2: Identify High-Density vs Low-Density Tags

For each venue, we classify tags by their **facet loading** (from Phase 3):

```python
def compute_tag_density(tag_embedding: np.ndarray,
                        probe_embeddings: Dict[str, np.ndarray]) -> float:
    """
    Compute the "density" of a tag as its maximum loading on any probe.

    High density = strongly aligned with a specific semantic intent
    Low density = generic, not aligned with any specific intent
    """
    max_loading = 0.0
    for probe_name, probe_emb in probe_embeddings.items():
        loading = cosine_similarity(tag_embedding, probe_emb)
        max_loading = max(max_loading, loading)
    return max_loading

def classify_tags_by_density(venue_tags: List[str],
                              tag_embeddings: Dict[str, np.ndarray],
                              probe_embeddings: Dict[str, np.ndarray],
                              threshold: float = 0.5) -> Tuple[List[str], List[str]]:
    """
    Classify tags into high-density and low-density groups.

    High-density: strongly aligned with at least one probe
    Low-density: generic, not aligned with any probe
    """
    high_density = []
    low_density = []

    for tag in venue_tags:
        density = compute_tag_density(tag_embeddings[tag], probe_embeddings)
        if density > threshold:
            high_density.append((tag, density))
        else:
            low_density.append((tag, density))

    # Sort by density (descending for high, ascending for low)
    high_density.sort(key=lambda x: x[1], reverse=True)
    low_density.sort(key=lambda x: x[1])

    return [t[0] for t in high_density], [t[0] for t in low_density]
```

### Step 3: Compute Ranking and Perturbation Impact

```python
def compute_venue_probe_similarity(venue_embedding: np.ndarray,
                                    probe_embedding: np.ndarray) -> float:
    """Compute similarity between venue representation and probe."""
    return cosine_similarity(venue_embedding, probe_embedding)

def rank_venues_by_probe(venue_embeddings: Dict[str, np.ndarray],
                         probe_embedding: np.ndarray) -> List[Tuple[str, float]]:
    """Rank all venues by similarity to a probe."""
    similarities = []
    for venue_id, emb in venue_embeddings.items():
        sim = compute_venue_probe_similarity(emb, probe_embedding)
        similarities.append((venue_id, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

def compute_ranking_shift(original_ranking: List[str],
                          perturbed_ranking: List[str]) -> Dict[str, float]:
    """
    Compute ranking shift metrics after perturbation.

    Returns:
    - kendall_tau: Rank correlation (-1 to 1, higher = more stable)
    - mean_rank_change: Average absolute rank change
    - top10_stability: % of top-10 venues that remain in top-10
    """
    from scipy.stats import kendalltau

    # Get rank positions
    orig_ranks = {v: i for i, v in enumerate(original_ranking)}
    pert_ranks = {v: i for i, v in enumerate(perturbed_ranking)}

    # Kendall's tau
    orig_order = [orig_ranks[v] for v in original_ranking]
    pert_order = [pert_ranks[v] for v in original_ranking]
    tau, _ = kendalltau(orig_order, pert_order)

    # Mean rank change
    rank_changes = [abs(orig_ranks[v] - pert_ranks[v]) for v in original_ranking]
    mean_change = np.mean(rank_changes)

    # Top-10 stability
    top10_orig = set(original_ranking[:10])
    top10_pert = set(perturbed_ranking[:10])
    top10_overlap = len(top10_orig & top10_pert) / 10

    return {
        "kendall_tau": tau,
        "mean_rank_change": mean_change,
        "top10_stability": top10_overlap
    }
```

### Step 4: Run Perturbation Experiment

```python
def run_perturbation_experiment(venue_id: str,
                                 venue_tags: List[str],
                                 tag_embeddings: Dict[str, np.ndarray],
                                 probe_embeddings: Dict[str, np.ndarray],
                                 all_venue_embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Run perturbation experiment for a single venue.

    1. Classify tags into high/low density
    2. For each probe:
       a. Compute original ranking
       b. Remove highest-density tag → recompute → measure shift
       c. Remove lowest-density tag → recompute → measure shift
    3. Compare shifts
    """
    # Classify tags
    high_density_tags, low_density_tags = classify_tags_by_density(
        venue_tags, tag_embeddings, probe_embeddings
    )

    if not high_density_tags or not low_density_tags:
        return None  # Need both for comparison

    results = {
        "venue_id": venue_id,
        "high_density_tag": high_density_tags[0],
        "low_density_tag": low_density_tags[0],
    }

    for probe_name, probe_emb in probe_embeddings.items():
        # Original ranking
        original_ranking = rank_venues_by_probe(all_venue_embeddings, probe_emb)
        original_ranking = [v[0] for v in original_ranking]

        # Perturb: remove high-density tag
        perturbed_tags_high = [t for t in venue_tags if t != high_density_tags[0]]
        perturbed_emb_high = embed_tag_set(perturbed_tags_high)
        perturbed_venue_embs_high = all_venue_embeddings.copy()
        perturbed_venue_embs_high[venue_id] = perturbed_emb_high
        perturbed_ranking_high = rank_venues_by_probe(perturbed_venue_embs_high, probe_emb)
        perturbed_ranking_high = [v[0] for v in perturbed_ranking_high]
        shift_high = compute_ranking_shift(original_ranking, perturbed_ranking_high)

        # Perturb: remove low-density tag
        perturbed_tags_low = [t for t in venue_tags if t != low_density_tags[0]]
        perturbed_emb_low = embed_tag_set(perturbed_tags_low)
        perturbed_venue_embs_low = all_venue_embeddings.copy()
        perturbed_venue_embs_low[venue_id] = perturbed_emb_low
        perturbed_ranking_low = rank_venues_by_probe(perturbed_venue_embs_low, probe_emb)
        perturbed_ranking_low = [v[0] for v in perturbed_ranking_low]
        shift_low = compute_ranking_shift(original_ranking, perturbed_ranking_low)

        results[f"{probe_name}_high_density_shift"] = shift_high["mean_rank_change"]
        results[f"{probe_name}_low_density_shift"] = shift_low["mean_rank_change"]
        results[f"{probe_name}_shift_ratio"] = shift_high["mean_rank_change"] / max(shift_low["mean_rank_change"], 0.01)

    return results

# Run for all venues
perturbation_results = []
for venue_id in tqdm(quality_venues, desc="Running perturbation"):
    venue_tags = gentags_per_venue[venue_id]
    result = run_perturbation_experiment(venue_id, venue_tags, tag_embeddings, probe_embeddings, venue_embeddings)
    if result:
        perturbation_results.append(result)

results_df = pd.DataFrame(perturbation_results)
```

### Step 5: Analyze Results

```python
# Compute mean shift ratios across all venues and probes
probe_names = ["laptop_friendly", "high_noise_social", "premium_coffee", "family_outdoor", "late_night"]

for probe in probe_names:
    high_shifts = results_df[f"{probe}_high_density_shift"]
    low_shifts = results_df[f"{probe}_low_density_shift"]
    ratios = results_df[f"{probe}_shift_ratio"]

    print(f"\n{probe}:")
    print(f"  High-density mean shift: {high_shifts.mean():.2f}")
    print(f"  Low-density mean shift: {low_shifts.mean():.2f}")
    print(f"  Mean ratio: {ratios.mean():.2f}x")

# Overall summary
all_high_shifts = []
all_low_shifts = []
for probe in probe_names:
    all_high_shifts.extend(results_df[f"{probe}_high_density_shift"].tolist())
    all_low_shifts.extend(results_df[f"{probe}_low_density_shift"].tolist())

print(f"\nOVERALL:")
print(f"  High-density mean shift: {np.mean(all_high_shifts):.2f}")
print(f"  Low-density mean shift: {np.mean(all_low_shifts):.2f}")
print(f"  Ratio: {np.mean(all_high_shifts) / np.mean(all_low_shifts):.2f}x")

# Statistical test
from scipy import stats
t_stat, p_value = stats.ttest_rel(all_high_shifts, all_low_shifts)
print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.6f}")
```

---

## Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| High-density shift ratio | **>2.0x** | High-density tags cause 2x more ranking change |
| Statistical significance | **p < 0.01** | Effect is not random |
| Consistency across probes | **4/5 probes** | Effect holds across semantic dimensions |

---

## Expected Results

### Predictions

| Metric | High-Density Tag | Low-Density Tag | Ratio |
|--------|------------------|-----------------|-------|
| Mean rank change | ~15-25 | ~5-10 | **2-3x** |
| Top-10 stability | ~60-70% | ~85-95% | — |
| Kendall's tau | ~0.85 | ~0.95 | — |

### Interpretation

- **High-density tags** are "load-bearing" — removing them causes significant downstream impact
- **Low-density tags** are "cosmetic" — removing them changes little
- **pdensity** is now measurable as "ranking shift per unit change"

---

## Comparison with RAKE

We can also run the same experiment with RAKE keywords:

```python
# Same experiment, but with RAKE keywords
# Expected: RAKE keywords all cause similar, diffuse shifts
# Because RAKE doesn't capture semantic structure

rake_high_shifts = []
rake_low_shifts = []
for venue_id in quality_venues:
    # ... same perturbation logic with RAKE keywords ...
    pass

print(f"RAKE high-density shift: {np.mean(rake_high_shifts):.2f}")
print(f"RAKE low-density shift: {np.mean(rake_low_shifts):.2f}")
print(f"RAKE ratio: {np.mean(rake_high_shifts) / np.mean(rake_low_shifts):.2f}x")
```

**Expected:** RAKE ratio ≈ 1.0x (no difference between "high" and "low" density keywords, because RAKE doesn't capture semantic structure).

---

## Cost Summary

| Step | Method | Cost |
|------|--------|------|
| Embed 5 probes | text-embedding-3-large | ~$0.01 |
| Embed perturbed tags | text-embedding-3-large | ~$0.03 |
| All computation | Local | $0 |
| **Total** | | **~$0.05** |

---

## Output Files

### Tables
- `results/phase4b/tables/perturbation_results.csv` — Per-venue, per-probe shifts
- `results/phase4b/tables/perturbation_summary.csv` — Summary statistics
- `results/phase4b/tables/density_classification.csv` — Tag density classifications

### Plots
- `results/phase4b/plots/1_density_shift_comparison.png` — High vs low density shifts
- `results/phase4b/plots/2_shift_by_probe.png` — Shifts across semantic probes
- `results/phase4b/plots/3_gentags_vs_rake_ratio.png` — Compare to RAKE

---

## The Kill Shot

If this experiment succeeds:

> "We operationalize propositional density (pdensity) as the ranking impact of tag removal. High-density tags cause **2.5x greater ranking shifts** than low-density tags (p < 0.001). This proves pdensity measures **semantic constraint strength**, not word count. Classical keyword methods show no such structure — all keywords cause similar, undifferentiated shifts."

---

## Script Location

`scripts/phase4b_probe_sensitivity.py`

---

## Checklist

```
[ ] Define 5 semantic probes
[ ] Embed probes
[ ] Classify tags by density (per venue)
[ ] Compute original rankings for each probe
[ ] Run high-density perturbation, measure shift
[ ] Run low-density perturbation, measure shift
[ ] Compute shift ratios
[ ] Statistical tests
[ ] Compare to RAKE
[ ] Generate plots
[ ] Create summary report
```

---

*Plan created: 2026-01-31*
