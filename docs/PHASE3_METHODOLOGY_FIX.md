# Phase 3 Methodology (Structural + Utility Proof)

**Date:** 2026-02-07
**Status:** Redefined (planned; not yet run)

This document replaces the prior localization-only Phase 3 plan. Phase 3 now has two required
components executed independently:

1. **Structural Proof** — State-Gini factorization
2. **Utility Proof** — CheckList DIR/INV attribution tests

---

## 1) Core Claims

**Structural:** Gentags form a factorized semantic state (semantic mass concentrates into a few
interpretable facets).

**Utility:** Gentags are actionable state (targeted edits produce predictable downstream changes).

---

## 2) Experiment A — Structural Proof (State-Gini)

**Purpose:** Convert “semantic mass concentrates” into a hard number.

### Protocol
1. Hard-assign each gentag to one of 10 facets using cosine similarity with threshold τ=0.35.
2. Count tags per facet.
3. Compute Gini on the facet counts.

### Baselines
- RAKE
- TF-IDF
- YAKE

All baselines must use the **same** facet anchors and threshold τ.

### Metrics
- State-Gini (gentags)
- State-Gini (baselines)

### Expected Target (pre-registered)
- Gentags: 0.5–0.7
- Baselines: 0.1–0.3

### Outputs
- `results/phase3/tables/state_gini_gentags.csv`
- `results/phase3/tables/state_gini_baselines.csv`
- `results/phase3/tables/state_gini_summary.csv`
- `results/phase3/plots/state_gini_comparison.png`

---

## 3) Experiment B — Utility Proof (CheckList DIR/INV)

**Purpose:** Prove attribution-aware reasoning using gentag state.

### DIR (Directional Expectation)
- Delete or flip a negative tag (e.g., remove “no ramp”).
- Judge score must move in the expected direction.

### INV (Invariance)
- Paraphrase tags (e.g., “fast service” → “rapid response”).
- Judge score should remain within ε of baseline.

### Baseline to beat
- Dense embeddings (`text-embedding-3-large`)

### Metrics
- DIR pass rate
- INV pass rate (|Δscore| ≤ ε)
- Attribution precision (intervened tag appears in justification)

### Outputs
- `results/phase3/tables/dir_results.csv`
- `results/phase3/tables/inv_results.csv`
- `results/phase3/tables/dir_inv_summary.csv`
- `results/phase3/plots/dir_pass_rate.png`
- `results/phase3/plots/inv_pass_rate.png`

---

## 4) Prompt Templates (JSON-only)

**Template A — Baseline Decision**

System:
```
You are a deterministic Recommendation Judge. Use ONLY the provided gentags.
Do NOT use external knowledge. Output JSON only:
{"score": int 0-100, "justification": "one sentence", "tags_used": ["tag1","tag2",...]}
```

User:
```
User Profile:
{PROFILE_TEXT}

Venue Gentags:
{GENTAG_LIST}

Task:
Return a recommendation score for this user based only on the gentags.
```

**Template B — DIR Intervention**

System:
```
You are a deterministic Recommendation Judge. Use ONLY the provided gentags.
Do NOT use external knowledge. Output JSON only:
{"score": int 0-100, "justification": "one sentence", "tags_used": ["tag1","tag2",...]}
```

User:
```
User Profile:
{PROFILE_TEXT}

Original Gentags:
{ORIGINAL_GENTAGS}

Intervention:
{INTERVENTION_DESC}  # e.g., DELETE "no ramp"

Revised Gentags:
{REVISED_GENTAGS}

Task:
Recompute the score for the revised state.
```

**Template C — INV Intervention**

System:
```
You are a deterministic Recommendation Judge. Use ONLY the provided gentags.
Do NOT use external knowledge. Output JSON only:
{"score_a": int 0-100, "score_b": int 0-100, "justification": "one sentence", "tags_used": ["tag1","tag2",...]}
```

User:
```
User Profile:
{PROFILE_TEXT}

Set A Gentags:
{GENTAGS_A}

Set B Gentags (Paraphrase):
{GENTAGS_B}

Task:
Score both sets. Scores should be invariant within ε if meaning is preserved.
```

---

## 5) Required Inputs

- Gentags from Phase 1/2 extractions
- Fixed facets + anchors (10 facets)
- Fixed user profiles (`phase3_profiles.csv`)
- Dense embedding baseline (`text-embedding-3-large`)
- Keyword baselines (RAKE, TF-IDF, YAKE)

---

## 6) Run Individually (Required)

Each experiment must be executed **independently** (not as a combined pipeline) so that
failures, metrics, and learnings are attributable to a single protocol at a time.

---

## 7) Decisions Needed Before Running

1. Facet list and anchor phrases (frozen)
2. Similarity threshold τ (default 0.35)
3. Judge model(s) and versions
4. Output schema and validation rules
5. User profile set and selection criteria
6. Intervention rules (delete/flip + expected direction)
7. INV paraphrase source and acceptance criteria
8. Baseline implementations for RAKE/TF-IDF/YAKE and embeddings
9. Scoring tolerances (DIR threshold, INV epsilon)
10. Sampling plan (venues per experiment)
11. Reproducibility controls (temperature, seeds, caching)
