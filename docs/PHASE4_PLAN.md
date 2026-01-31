# Phase 4: Downstream Sensitivity Analysis

**Status:** Planning
**Depends on:** Phase 3 (Representation Comparison) ✅ Complete
**Last Updated:** 2026-01-27

---

## What Changed vs Previous Phase 4

This section documents the framing correction applied after review.

### OLD Framing (Incorrect)

| Concept | Old Meaning |
|---------|-------------|
| System | Repeatedly runs LLM extractions |
| Variability | Internal ambiguity of the system |
| Monitoring | System re-checks itself via repeated runs |
| Info seeking | Model realizes it's unsure |
| Extraction | Part of the system |

**Problem:** This implied the system itself is re-running extractions to monitor its own beliefs. That's not what we built. The multiple extractions were research instrumentation for identifiability and reproducibility, not a runtime loop.

### NEW Framing (Correct)

| Concept | New Meaning |
|---------|-------------|
| System | Downstream consumer of gentags |
| Variability | Observer agreement / identifiability across samples |
| Monitoring | Offline analysis of collected representations |
| Info need | **Not modeled in this paper** |
| Extraction | Preprocessing pipeline (not the system itself) |

**Key insight:** We collected multiple OTags (observed tags) to study representation behavior. OTags are observer samples used to quantify identifiability, dispersion, and reproducibility. The system does NOT repeatedly check itself.

### What Stays the Same

Despite the framing fix, **all experiments stay identical**:

- ✅ Phase 2 stability — still valid
- ✅ Phase 3 localization — still valid
- ✅ Facet drift — still valid
- ✅ Downstream probe — still valid
- ✅ Ranking stability — still valid
- ✅ Attribution — still valid
- ✅ Sparsity correlation — still valid

**Same data. Same code. Different language.**

---

## Critical Framing

### What This Phase IS

We evaluate how different semantic state representations affect downstream behavior. OTags are **observer samples** used to quantify identifiability and representation dispersion; they are not a runtime mechanism.

### What This Phase is NOT

- ❌ The system re-running itself
- ❌ Online epistemic monitoring
- ❌ Active querying or agents
- ❌ User search behavior
- ❌ Recommender systems

### OTags and PTags (Definitions)

| Term | Meaning | This Paper? |
|------|---------|-------------|
| **OTags** | Tags extracted from evidence (reviews) = observed semantic state | ✅ Yes |
| **PTags** | Tags proposed/constructed by system to resolve gaps = candidate constraints | ❌ Future work |

Phase 4 is **OTags only**. PTags would start building an agent loop, expanding scope beyond this paper.

### The Correct One-Liner

> "We analyze how semantic state representations affect downstream behavior; repeated extractions are used solely to evaluate representational identifiability."

---

## Phase 4A: Representation Coverage & Dispersion (Descriptive)

We report **coverage** (number of unique tags) and **dispersion** (mean pairwise distance across OTag samples) as descriptive properties of the observed semantic state. These are offline measurements of identifiability and reproducibility, not epistemic monitoring.

### Question

> What descriptive properties (coverage and dispersion) characterize observed gentag state, and how do they relate to evidence amount?

### Inputs (Already Available)

- `tags_{model}.csv` — tag rows from Phase 1
- `extractions_{model}.csv` — extraction metadata
- `data/study1_venues_*.csv` — reviews text (evidence amount)
- Phase 2 dispersion data

### Measures

#### Measure 1: Tag Coverage

How many unique gentags does this venue have?

```python
for venue_id in venues:
    tag_count[venue_id] = count_unique_tags(venue_id)
```

#### Measure 2: Observer Dispersion

We use multiple independent extractions as **observer samples** of the same evidence to estimate representation dispersion.

For each venue:

```python
dispersion[venue_id] = mean_pairwise_distance_across_otag_sets(venue_id)
```

High dispersion means observer samples yield different tag sets, indicating lower identifiability under the same evidence.

### Phase 4A Deliverables

#### Tables

| Table | Contents |
|-------|----------|
| `phase4a_tag_coverage.csv` | venue_id, n_unique_tags, n_reviews, tokens_total |
| `phase4a_dispersion.csv` | venue_id, mean_pairwise_distance, n_otag_samples |

#### Plots

| Plot | Description |
|------|-------------|
| `1_tag_coverage_distribution.png` | Distribution of unique tag counts per venue |
| `2_dispersion_distribution.png` | Distribution of OTag dispersion per venue |
| `3_evidence_vs_dispersion.png` | Tokens/reviews vs dispersion |

---

## Phase 4B: Downstream Sensitivity (Diagnostic)

### Question

> Do representational differences (gentags vs embeddings) change downstream ranking stability and attribution under the same semantic probe?

### Framing

This is a **diagnostic probe**, NOT a product or user query.

> "We use synthetic semantic constraint bundles as diagnostic probes of representation stability, not as user-facing applications."

### The Probe Set (5 Probes)

Not one query — five **semantic constraint bundles**:

| Probe | Semantic Constraints |
|-------|---------------------|
| `"quiet work-friendly café"` | ambiance + seating/purpose |
| `"family-friendly outdoor seating"` | demographic + seating |
| `"cheap quick lunch"` | price + speed + food |
| `"romantic date night ambiance"` | social context + ambiance |
| `"vegan options"` | dietary |

These are **constraint bundles**, not "user search queries."

### What We Compute

#### Representation A: Gentags

For each OTag snapshot (venue_id, model, prompt, run):
1. Embed `" ".join(unique_tags)` → vector
2. Compute `similarity(probe_embedding, tags_embedding)` → score
3. Rank venues
4. Compute rank stability across snapshots

#### Representation B: Review Embeddings (Baseline)

1. Embed concatenated reviews once per venue
2. Rank venues once
3. Single ranking (no stability comparison — only one snapshot exists)

### Metrics

#### 1. Ranking Stability (Gentags)

```python
# Kendall τ or Spearman ρ across rankings from different OTag snapshots
stability = mean_pairwise_kendall_tau(rankings_per_snapshot)
```

Report: mean τ ± std per probe.

#### 2. Attribution

For top-k venues per probe, show:
- Which tags matched which probe terms (lexical matching)
- OR: top contributing tags by cosine similarity (if per-tag embeddings cached)

Gentags: matched tags visible.
Embeddings: nothing to show.

#### 3. Failure Mode Examples (3 Cases)

1. **Missing concept:** Probe asks for something no venue has tags about
2. **Spurious embedding match:** Venue ranked high by embedding, low by gentags — no relevant tags on inspection
3. **Dispersion propagation:** High-instability venue shows different ranks across snapshots

### Phase 4B Deliverables

#### Tables

| Table | Contents |
|-------|----------|
| `phase4b_probe_rank_stability.csv` | probe, model, prompt, mean_tau, std_tau |
| `phase4b_topk_attribution.csv` | probe, rank, venue_id, similarity, matched_tags |

#### Plots

| Plot | Description |
|------|-------------|
| `4_ranking_stability_by_probe.png` | Bar chart: τ per probe |
| `5_tau_distribution.png` | Distribution of pairwise τ values across snapshots |

#### Qualitative

| File | Contents |
|------|----------|
| `phase4b_attribution_examples.md` | probe → top venues → matched tags (3-5 examples) |
| `phase4b_failure_modes.md` | 3 documented failure cases |

---

## Implementation

### Script Structure

**`scripts/phase4_analysis.py`** — Single script with sections:

```
1. Load
   - Tags files across models
   - Venue reviews dataset

2. Build Snapshots
   - A snapshot = (venue_id, model, prompt, run_number) → set(tags)

3. Phase 4A
   - Tag-set coverage per venue (unique tag counts)
   - OTag dispersion per venue (pairwise across snapshots)
   - Descriptive coverage/dispersion tables

4. Phase 4B
   - Embed probes
   - Embed snapshot tag texts
   - Ranking per snapshot
   - Compute Kendall τ distribution
   - Export attribution examples
```

**`scripts/phase4_plots.py`** — Plotting script

**Outputs to:** `results/phase4/tables/` and `results/phase4/plots/`

### Implementation Steps

```
[ ] Load Phase 1 tags data + venue reviews
[ ] Build OTag snapshots (venue_id, model, prompt, run → tag set)
[ ] Compute tag-set coverage per venue (unique tag counts)
[ ] Compute OTag dispersion per venue (mean pairwise distance across snapshots)
[ ] Generate 4A plots (coverage distribution, dispersion distribution, evidence vs dispersion)
[ ] Embed 5 probe vectors
[ ] Embed OTag snapshots (use cached embeddings where possible)
[ ] Compute rankings per probe × per snapshot
[ ] Compute Kendall τ stability metrics
[ ] Extract attribution examples for top-k venues
[ ] Document 3 failure modes
[ ] Generate 4B plots (stability bars, τ distribution)
```

---

## Paper Integration

### Where This Fits

```
Section 4: Experiments

4.1 Semantic Stability (Phase 2)
4.2 Prompt and Model Sensitivity (Phase 2)
4.3 Variability and Evidence Sparsity (Phase 2)
4.4 Retention Analysis (Phase 2)
4.5 State Observability (Phase 3)
4.6 Representation Coverage & Dispersion (Phase 4A)
4.7 Downstream Sensitivity (Phase 4B)
```

### Section 4.6: Representation Coverage & Dispersion (~1 page)

- Define coverage and dispersion as descriptive properties of observed gentag state
- Ontology-free (no predefined facets — operates on raw gentag sets)
- Connect to Phase 2 sparsity results without invoking information seeking

### Section 4.7: Downstream Sensitivity (~1-2 pages)

- 5 diagnostic probes
- Ranking stability comparison
- Attribution analysis
- 3 failure mode examples

### Key Sentences (Reviewer Shields)

**For 4A:**
> "We report coverage and observer dispersion as descriptive properties of observed semantic state. Repeated extractions are used solely to evaluate identifiability; they are not a runtime mechanism or a model of information seeking."

**For 4B:**
> "We do not implement active querying or agents. Instead, we use synthetic semantic constraint bundles as diagnostic probes, demonstrating how gentag representations produce more stable and interpretable downstream rankings compared to dense embeddings."

---

## What This Validates

### Contribution Stack (Complete)

| Contribution | Phase | Status |
|--------------|-------|--------|
| Gentags form stable semantic representations | Phase 2 | ✅ |
| Variability = observer-estimated identifiability | Phase 2 | ✅ |
| Gentags enable localized attribution | Phase 3 | ✅ |
| Gentags provide observable state descriptors (coverage + dispersion) | Phase 4A | 🔜 |
| Representational structure affects downstream sensitivity | Phase 4B | 🔜 |

### The Final Narrative

```
Gentags → observable semantic state (OTags)
Tag-set sparsity → coverage descriptor
Observer dispersion → identifiability descriptor
Downstream probe → representational consequences visible
```

### What We Do NOT Claim

- ❌ The system repeatedly checks itself
- ❌ Calibrated probability estimates or Bayesian posteriors
- ❌ We implement information seeking
- ❌ We build agents or action policies
- ❌ We model user behavior

### What We DO Claim

- ✅ Coverage and dispersion are descriptive properties of observed state
- ✅ Repeated extractions are used solely to evaluate identifiability
- ✅ Representational structure affects downstream ranking stability and attribution
- ✅ Embeddings cannot provide equivalent state attribution signals

---

## Recommended Order

4A (sparsity + dispersion) → 4B (probes + ranking) → attribution → failure modes → plots
