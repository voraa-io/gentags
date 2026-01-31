# Phase 3A: Classical Baseline Comparison

**Status:** In Progress
**Depends on:** Phase 2 (Stability) ✅, Phase 3 (Localization) ✅
**Last Updated:** 2026-01-30

---

## Why This Phase is Critical

The current paper has a **glass jaw**: "+0.164 above random" for retention is not enough.

Reviewers will immediately ask:
> "Why not just use TF-IDF or RAKE? They're computationally free."

If classical keyword methods achieve similar retention, then gentags are "an expensive way to generate keywords."

**This phase proves that gentags capture semantics beyond surface extraction.**

---

## The Hard Question

Statistical methods (TF-IDF, RAKE, YAKE) are purely lexical. If they achieve similar retention, it implies the "latent semantic externalization" we claim the LLM does is actually just sophisticated keyword extraction.

**We need to show:** Gentags capture semantic hypotheses that aren't explicitly in the text — something TF-IDF cannot do.

---

## Baselines

| Method | Type | Description |
|--------|------|-------------|
| **TF-IDF** | Statistical | Term frequency–inverse document frequency |
| **RAKE** | Statistical | Rapid Automatic Keyword Extraction |
| **YAKE** | Statistical | Yet Another Keyword Extractor |
| **Gentags** | LLM-based | Machine-generated semantic tags |

### Budget Matching

To ensure fair comparison:
- **k** = median number of unique gentags per venue (~25)
- **Phrase length** = 1-4 tokens (matching gentag constraint)

---

## Metrics

### A) Retention (Primary)

**Definition:** `cosine(embed(reviews), embed(representation_text))`

- For TF-IDF/RAKE/YAKE: `representation_text = " ".join(top_k_phrases)`
- For gentags: `representation_text = " ".join(unique_gentags)`

**Win condition (strong):** gentags retention > baselines by meaningful margin (effect size, not just p-value)

### B) Localization / Gini (Differentiator)

Same methodology as Phase 3:
- Build per-venue representation from top-k phrases
- Embed and compute facet drift via anchor similarity
- Compute Gini coefficient

**Expected:** TF-IDF/RAKE/YAKE will be more diffuse (lower Gini) because they aren't structured around semantic factors.

### C) Semantic Stability (Not Run Stability)

Baselines are deterministic — they trivially "win" run-to-run stability.

Instead, we use existing Phase 2 data:
- Gentags show high MMC even when Jaccard is low
- This proves semantic stability under lexical variation
- Classical methods don't have this property (they extract the same surface forms)

---

## Decision Tree

### Case 1: Gentags beat classics on retention
**→ Golden.** Claim: "LLM gentags capture semantics beyond surface extraction."

### Case 2: Gentags tie classics on retention (e.g., 0.625 vs 0.620)
**→ Pivot.** Retention becomes sanity check. Contribution shifts to:
1. Semantic stability under lexical variation
2. Localization (Gini)
3. Cross-model agreement
4. Persistent state

### Case 3: Classics beat gentags on retention
**→ Red alert.** Must answer: why pay LLM cost?

Survival conditions:
- Gentags dominate localization + cross-model agreement + interpretability
- Frame gentags as "state variables" not "compression"

---

## Implementation

### Cost & Time Estimate

| Step | Method | Time | Cost |
|------|--------|------|------|
| Load data | Local | ~5 sec | $0 |
| Extract TF-IDF | sklearn (local) | ~30 sec | $0 |
| Extract RAKE | rake-nltk (local) | ~30 sec | $0 |
| Extract YAKE | yake (local) | ~30 sec | $0 |
| Embed keywords | OpenAI API | ~30 sec | ~$0.02 |
| Compute retention | NumPy (local) | ~5 sec | $0 |
| **Total** | | **~2-3 min** | **~$0.02** |

**API calls:** 230 venues × 3 methods = 690 texts → batched into ~6 API calls

### Script: `scripts/phase3a_baselines.py`

#### Step-by-Step Code Walkthrough

**1. Load Data (already cached)**
```python
# Load venue data (553 venues)
venue_df = pd.read_csv("data/study1_venues_20250117.csv")

# Load gentag retention to get the 230 quality-filtered venues
gentag_retention = load_gentag_retention()
quality_venues = gentag_retention['venue_id'].unique()

# Filter to only those venues
venue_df = venue_df[venue_df['id'].isin(quality_venues)]

# Load cached review embeddings from Phase 2 (no API call)
review_embeddings = load_review_embeddings()
```

**2. Determine k (match gentag budget)**
```python
# Get median number of unique gentags per venue (~20-25)
median_gentags = gentag_retention.groupby('venue_id')['n_unique_norm_eval'].median().median()
k = min(median_gentags, 25)  # Use this as keyword count
```

**3. Extract Keywords (all local, no API)**
```python
# TF-IDF: Statistical term weighting
def extract_tfidf_keywords(text, k=25):
    vectorizer = TfidfVectorizer(ngram_range=(1, 4), stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    # Sort by score, return top-k phrases (1-4 words)
    return top_k_phrases

# RAKE: Graph-based keyword extraction
def extract_rake_keywords(text, k=25):
    rake = Rake(min_length=1, max_length=4)
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:k]

# YAKE: Unsupervised keyword extraction
def extract_yake_keywords(text, k=25):
    extractor = yake.KeywordExtractor(n=4, top=k)
    keywords = extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]
```

**4. Embed Keywords (API calls, batched)**
```python
# Concatenate keywords into text
kw_text = " ".join(keywords)  # e.g., "great coffee friendly staff quiet"

# Batch embed all keyword texts (690 texts → 6 batches)
keyword_embeddings = embed_texts_batch(client, texts_to_embed, batch_size=128)
```

**5. Compute Retention (local)**
```python
# For each venue-method pair:
review_emb = review_embeddings[venue_id]  # Cached from Phase 2
kw_emb = keyword_embeddings[i]            # Just computed
retention = cosine_similarity(review_emb, kw_emb)
```

**6. Compare to Gentags**
```python
# Gentag retention already in Phase 2 data
gentag_mean = gentag_retention.groupby('venue_id')['retention_cosine'].mean()

# Summary comparison
summary = all_results.groupby('method')['retention_cosine'].agg(['mean', 'std', 'median'])
```

#### Progress Bars

The script uses `tqdm` for progress tracking:
```python
for _, row in tqdm(venue_df.iterrows(), total=len(venue_df), desc="Processing venues"):
    # ...

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
    # ...
```

You'll see:
```
Processing venues: 100%|██████████| 230/230 [01:30<00:00, 2.55it/s]
Embedding: 100%|██████████| 6/6 [00:25<00:00, 4.17s/batch]
```

### Output Files

**Tables:**
- `results/phase3a/tables/baseline_retention.csv`
- `results/phase3a/tables/baseline_localization.csv`
- `results/phase3a/tables/comparison_summary.csv`

**Plots:**
- `results/phase3a/plots/1_retention_comparison.png`
- `results/phase3a/plots/2_localization_comparison.png`
- `results/phase3a/plots/3_cost_vs_retention.png`

---

## Expected Results (Predictions)

| Metric | Gentags | TF-IDF | RAKE | YAKE |
|--------|---------|--------|------|------|
| Retention | 0.625 | ~0.55-0.60 | ~0.50-0.55 | ~0.50-0.55 |
| Localization Gini | 0.657 | ~0.40-0.50 | ~0.35-0.45 | ~0.35-0.45 |
| Cross-model agreement | >0.94 | N/A | N/A | N/A |
| Persistent state | ✅ | ✅ | ✅ | ✅ |
| Interpretable | ✅ | ❌ | ✅ | ✅ |

**Key differentiators if retention ties:**
1. Gentags are paraphrase-robust (high MMC under lexical variation)
2. Gentags provide better localization (higher Gini)
3. Gentags show cross-model agreement (linguistic universality)

---

## Paper Integration

### Results Section Addition

```
4.X Classical Baseline Comparison

We compare gentags against classical keyword extraction methods
(TF-IDF, RAKE, YAKE) to evaluate whether LLM-based extraction
captures semantics beyond surface features.

[Table: Retention comparison]
[Table: Localization comparison]
[Discussion of results per decision tree]
```

### The Reviewer Shield

> "We evaluate whether gentags capture semantics beyond surface extraction
> by comparing against classical keyword baselines (TF-IDF, RAKE, YAKE)
> on retention, paraphrase robustness, and localization of representational drift."

---

## Implementation Checklist

```
[ ] Install dependencies (scikit-learn, yake, rake-nltk)
[ ] Load 230 quality-filtered venues
[ ] Load review embeddings from Phase 2 cache
[ ] Compute TF-IDF keywords per venue (top-k)
[ ] Compute RAKE keywords per venue (top-k)
[ ] Compute YAKE keywords per venue (top-k)
[ ] Embed baseline representations
[ ] Compute retention for each baseline
[ ] Compute localization Gini for each baseline
[ ] Generate comparison tables
[ ] Generate comparison plots
[ ] Update full analysis report
```

---

## What This Proves

If gentags beat baselines:
- LLM extracts **latent semantics**, not just surface keywords
- Cost is justified by semantic depth

If gentags tie on retention but win on localization:
- Gentags provide **structured semantic state**
- Value is in attribution, not compression

Either way, we have a defensible paper.
