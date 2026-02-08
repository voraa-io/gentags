# Canonicalization: Solving the State Explosion

**Status:** Planning
**Depends on:** Phase 2 (Stability), Phase 3A (Baselines)
**Priority:** HIGH — Defuses the "lexical instability" attack
**Estimated Cost:** $0 (all local computation)
**Estimated Time:** ~5-10 minutes execution

---

## Why This Experiment is Critical

### The Problem

Gentags show Jaccard overlap of only **0.471** across runs. Reviewers will attack this:

> "If the same venue produces different tags across runs, how can this be 'persistent state'?"

### The Attack

A reviewer will say:
> "Your 'state' is unstable. Run 1 gives 'rise', Run 2 gives 'increase'. That's two different states for the same meaning."

### The Defense

Implement **semantic canonicalization**: cluster synonyms into canonical forms.

After canonicalization:
- "rise" and "increase" → same cluster ID
- "great coffee" and "excellent espresso" → same cluster ID

This collapses the "state explosion" while preserving semantic identity.

---

## The Hard Question

> How do we know when two tags are "the same concept"?

**Answer:** By embedding similarity threshold (τ).

| Tag Pair | Cosine Similarity | Same Concept? |
|----------|-------------------|---------------|
| "rise" ↔ "increase" | 0.92 | ✅ Yes (τ = 0.85) |
| "great coffee" ↔ "excellent espresso" | 0.89 | ✅ Yes |
| "quiet" ↔ "peaceful" | 0.87 | ✅ Yes |
| "coffee" ↔ "espresso" | 0.78 | ❌ No (below threshold) |
| "quiet" ↔ "loud" | 0.35 | ❌ No |

---

## Experimental Design

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CANONICALIZATION                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw Tags (Run 1)          Raw Tags (Run 2)                     │
│  ├── "great coffee"        ├── "excellent espresso"             │
│  ├── "quiet atmosphere"    ├── "peaceful vibe"                  │
│  ├── "friendly staff"      ├── "nice service"                   │
│  └── "outdoor seating"     └── "patio available"                │
│                                                                 │
│  Jaccard = 0.0 (no overlap)                                     │
│                                                                 │
│  ─────────────── CANONICALIZATION ───────────────               │
│                                                                 │
│  Canonical IDs (Run 1)     Canonical IDs (Run 2)                │
│  ├── cluster_17            ├── cluster_17                       │
│  ├── cluster_42            ├── cluster_42                       │
│  ├── cluster_8             ├── cluster_8                        │
│  └── cluster_55            └── cluster_55                       │
│                                                                 │
│  Jaccard = 1.0 (perfect overlap!)                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Algorithm: Semantic Clustering

We use **agglomerative clustering** with cosine distance and a threshold τ.

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

def build_canonical_clusters(all_tag_embeddings: Dict[str, np.ndarray],
                              threshold: float = 0.15) -> Dict[str, int]:
    """
    Build canonical clusters from all unique tags across all runs.

    Args:
        all_tag_embeddings: Dict mapping tag text to embedding
        threshold: Distance threshold (1 - cosine_similarity)
                   0.15 means cluster if cosine > 0.85

    Returns:
        Dict mapping tag text to cluster ID
    """
    tags = list(all_tag_embeddings.keys())
    embeddings = np.array([all_tag_embeddings[t] for t in tags])

    # Compute pairwise cosine distances
    distances = cosine_distances(embeddings)

    # Agglomerative clustering with distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distances)

    return {tag: int(label) for tag, label in zip(tags, cluster_labels)}
```

### Step 1: Collect All Tags

```python
# Load all gentags from Phase 1 (all runs, all models, all prompts)
all_gentags = pd.read_csv("results/phase1/gentags_week2_run.csv")

# Get all unique normalized tags
unique_tags = set()
for _, row in all_gentags.iterrows():
    tags = json.loads(row['tags']) if isinstance(row['tags'], str) else row['tags']
    for tag in tags:
        unique_tags.add(normalize_tag(tag))

print(f"Total unique tags: {len(unique_tags)}")
# Expected: ~5,000-10,000 unique tags
```

### Step 2: Embed All Unique Tags

```python
# Check if we have cached embeddings from Phase 2
try:
    tag_embeddings = load_cached_tag_embeddings()
    print(f"Loaded {len(tag_embeddings)} cached tag embeddings")
except:
    # Embed all unique tags
    tag_embeddings = {}
    for tag in tqdm(unique_tags, desc="Embedding tags"):
        tag_embeddings[tag] = embed_text(client, tag)

    # Cache for future use
    save_tag_embeddings(tag_embeddings)
```

### Step 3: Build Clusters

```python
# Build canonical clusters with τ = 0.85 (distance = 0.15)
tag_to_cluster = build_canonical_clusters(tag_embeddings, threshold=0.15)

n_clusters = len(set(tag_to_cluster.values()))
print(f"Collapsed {len(unique_tags)} tags into {n_clusters} clusters")
print(f"Compression ratio: {len(unique_tags) / n_clusters:.2f}x")

# Save cluster assignments
cluster_df = pd.DataFrame([
    {"tag": tag, "cluster_id": cluster}
    for tag, cluster in tag_to_cluster.items()
])
cluster_df.to_csv("results/canonicalization/tag_clusters.csv", index=False)
```

### Step 4: Recompute Jaccard Under Canonicalization

```python
def canonicalize_tag_set(tags: List[str], tag_to_cluster: Dict[str, int]) -> Set[int]:
    """Convert a list of tags to a set of cluster IDs."""
    return {tag_to_cluster.get(normalize_tag(t), -1) for t in tags}

def jaccard_canonical(tags1: List[str], tags2: List[str],
                      tag_to_cluster: Dict[str, int]) -> float:
    """Compute Jaccard similarity using canonical cluster IDs."""
    set1 = canonicalize_tag_set(tags1, tag_to_cluster)
    set2 = canonicalize_tag_set(tags2, tag_to_cluster)

    # Remove -1 (unknown tags)
    set1.discard(-1)
    set2.discard(-1)

    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

# Recompute Jaccard for run-to-run pairs
canonical_results = []
for venue_id in tqdm(quality_venues, desc="Computing canonical Jaccard"):
    run1_tags = get_venue_tags(venue_id, run=1)
    run2_tags = get_venue_tags(venue_id, run=2)

    # Raw Jaccard
    raw_jaccard = jaccard_similarity(
        set(normalize_tags(run1_tags)),
        set(normalize_tags(run2_tags))
    )

    # Canonical Jaccard
    canonical_jaccard = jaccard_canonical(run1_tags, run2_tags, tag_to_cluster)

    canonical_results.append({
        "venue_id": venue_id,
        "raw_jaccard": raw_jaccard,
        "canonical_jaccard": canonical_jaccard,
        "improvement": canonical_jaccard - raw_jaccard
    })

results_df = pd.DataFrame(canonical_results)
```

### Step 5: Compare to RAKE

RAKE is purely lexical — it cannot collapse synonyms. Let's prove it.

```python
# For RAKE, there's no canonicalization possible
# "coffee" and "espresso" are always different keywords

# Compute RAKE "Jaccard" across paraphrased reviews (from Phase 3B)
rake_original = load_original_rake()
rake_paraphrased = load_paraphrased_rake()

rake_jaccard_results = []
for venue_id in quality_venues:
    orig_kw = set(rake_original[venue_id])
    para_kw = set(rake_paraphrased[venue_id])

    jaccard = len(orig_kw & para_kw) / len(orig_kw | para_kw)
    rake_jaccard_results.append(jaccard)

print(f"RAKE mean Jaccard (original vs paraphrased): {np.mean(rake_jaccard_results):.3f}")
# Expected: very low, ~0.15-0.25
```

---

## Threshold Selection

How do we choose τ?

### Option A: Fixed Threshold (τ = 0.85)

Based on Phase 2 MMC analysis, tags with cosine > 0.85 are reliably paraphrases.

### Option B: Empirical Threshold Search

```python
def evaluate_threshold(tag_embeddings, run_pairs, threshold):
    """Evaluate a threshold by measuring Jaccard improvement."""
    tag_to_cluster = build_canonical_clusters(tag_embeddings, 1 - threshold)
    jaccard_improvements = []

    for run1_tags, run2_tags in run_pairs:
        raw = jaccard_similarity(set(run1_tags), set(run2_tags))
        canonical = jaccard_canonical(run1_tags, run2_tags, tag_to_cluster)
        jaccard_improvements.append(canonical - raw)

    return np.mean(jaccard_improvements)

# Search for optimal threshold
thresholds = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
results = []
for τ in thresholds:
    improvement = evaluate_threshold(tag_embeddings, run_pairs, τ)
    n_clusters = count_clusters(τ)
    results.append({"threshold": τ, "improvement": improvement, "n_clusters": n_clusters})

# Plot threshold vs improvement vs cluster count
```

---

## Expected Results

### Predictions

| Metric | Raw | Canonicalized | Improvement |
|--------|-----|---------------|-------------|
| Mean Jaccard | 0.471 | **0.70-0.80** | +0.23-0.33 |
| % venues with Jaccard > 0.6 | ~40% | **>75%** | +35% |
| Unique "states" per venue | ~40-50 | **~15-20** | 2-3x compression |

### RAKE Comparison

| Method | Canonicalized Jaccard | Can Canonicalize? |
|--------|----------------------|-------------------|
| Gentags | **0.75** | ✅ Yes (semantic clusters) |
| RAKE | 0.20 | ❌ No (lexical only) |

---

## Success Criteria

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Canonical Jaccard | **>0.70** | State is stable after canonicalization |
| Improvement over raw | **>0.20** | Canonicalization helps significantly |
| Compression ratio | **>2.0x** | Synonym collapse is meaningful |
| RAKE gap | **>0.50** | Gentags can canonicalize, RAKE cannot |

---

## The Minimal Defensible Policy

For the paper, we recommend:

```
Canonicalization Policy:
1. Embed all tags using text-embedding-3-large
2. Cluster with cosine threshold τ = 0.85
3. Assign each tag to its cluster centroid
4. Report state as set of cluster IDs
```

This is:
- **Simple** — one parameter (τ)
- **Reproducible** — deterministic clustering
- **Effective** — collapses synonyms
- **Defensible** — based on empirical MMC threshold

---

## Output Files

### Tables
- `results/canonicalization/tag_clusters.csv` — Tag to cluster mapping
- `results/canonicalization/cluster_centroids.csv` — Cluster centroids
- `results/canonicalization/jaccard_comparison.csv` — Raw vs canonical Jaccard
- `results/canonicalization/threshold_search.csv` — Threshold evaluation

### Plots
- `results/canonicalization/plots/1_jaccard_improvement.png` — Before/after histogram
- `results/canonicalization/plots/2_threshold_search.png` — Threshold vs improvement
- `results/canonicalization/plots/3_gentags_vs_rake.png` — Canonicalization gap

---

## The Kill Shot

If this experiment succeeds:

> "Lexical variation (Jaccard 0.471) does not imply state instability. After semantic canonicalization (τ = 0.85), Jaccard rises to **0.75**, proving that apparent variation reflects synonym choice, not semantic drift. This canonicalization is **only possible with semantic representations** — classical keyword methods remain fragmented (Jaccard 0.20) because they cannot collapse 'rise' and 'increase' into the same concept."

---

## Script Location

`scripts/canonicalization.py`

---

## Checklist

```
[ ] Collect all unique tags from Phase 1
[ ] Load/compute tag embeddings
[ ] Build canonical clusters (τ = 0.85)
[ ] Compute raw Jaccard for run pairs
[ ] Compute canonical Jaccard for run pairs
[ ] Measure improvement
[ ] Compare to RAKE
[ ] Threshold sensitivity analysis
[ ] Generate plots
[ ] Create summary report
```

---

*Plan created: 2026-01-31*
