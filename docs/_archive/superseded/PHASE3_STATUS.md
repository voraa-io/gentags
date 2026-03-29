# Phase 3 Status (Structural + Utility Proof)

**Date:** 2026-02-07
**Status:** Redefined (planned; not yet run)

**Goal of this doc:** Define Phase 3 with concrete protocols, baselines, and metrics so the
paper can claim **factorized structure** and **actionable attribution**.

---

## 1) What Phase 3 Is Trying To Prove (Plain English)

Phase 1–2 show gentags are **consistent**. Phase 3 must show two things:

1. **Structural proof:** the state is **factorized** (semantic mass concentrates into a few facets)
2. **Utility proof:** the state is **actionable** (interventions cause predictable downstream change)

---

## 2) Experiment A — Structural Proof (State-Gini)

**Objective:** Quantify factorization of gentag state.

**Procedure:**
1. Hard-assign each gentag to 1 of 10 facets using cosine similarity with threshold τ=0.35.
2. Count tags per facet.
3. Compute Gini on facet counts.

**Baselines:** RAKE, TF-IDF, YAKE (same facet assignment + same τ).

**Primary metrics:**
- **State-Gini (gentags)**
- **State-Gini (baselines)**

**Target:** Gentags Gini 0.5–0.7 vs baselines 0.1–0.3 (pre-registered expectation).

**Outputs (planned):**
- `results/phase3/tables/state_gini_gentags.csv`
- `results/phase3/tables/state_gini_baselines.csv`
- `results/phase3/tables/state_gini_summary.csv`
- `results/phase3/plots/state_gini_comparison.png`

---

## 3) Experiment B — Utility Proof (CheckList DIR/INV)

**Objective:** Show gentags enable **attribution-aware interventions**.

**DIR (Directional Expectation):**
- Delete or flip a negative tag (e.g., remove “no ramp”).
- Judge score must move in the expected direction.

**INV (Invariance):**
- Paraphrase tags (e.g., “fast service” → “rapid response”).
- Judge score should remain within ε of baseline.

**Baseline to beat:** Dense embeddings (`text-embedding-3-large`).

**Primary metrics:**
- DIR pass rate
- INV pass rate (|Δscore| ≤ ε)
- Attribution precision (justification cites intervened tag)

**Outputs (planned):**
- `results/phase3/tables/dir_results.csv`
- `results/phase3/tables/inv_results.csv`
- `results/phase3/tables/dir_inv_summary.csv`
- `results/phase3/plots/dir_pass_rate.png`
- `results/phase3/plots/inv_pass_rate.png`

---

## 4) Run Individually (Required)

Each experiment must be executed **independently** (not as a combined pipeline) so that
failures, metrics, and learnings are attributable to a single protocol at a time.

---

## 5) Judge LLM Requirements (Non-Negotiable)

- Must be **deterministic** in output format (JSON only).
- Must use **only** provided gentags (no external knowledge).
- Must cite the **tags used** in justification for attribution scoring.

---

## 6) Data Inputs

- Gentags: Phase 1/2 extractions
- Facets + anchors: fixed list (10 facets)
- User profiles: curated, fixed list (`phase3_profiles.csv`)
- Embeddings baseline: `text-embedding-3-large`

---

## 7) Decisions Needed Before Running Phase 3

1. Facet list and anchor phrases (frozen for State-Gini)
2. Similarity threshold τ (default 0.35) for gentags and baselines
3. Judge model(s) and versions (primary + backup)
4. Output schema (fields and constraints for JSON-only responses)
5. User profile set (count, diversity, and selection criteria)
6. Intervention rules (which tags can be deleted/flipped; direction expectations)
7. INV paraphrase source (manual vs model, and acceptance criteria)
8. Baseline implementations for RAKE/TF-IDF/YAKE and embeddings
9. Scoring tolerances (DIR pass threshold, INV epsilon)
10. Sampling plan (venues per experiment, per model, per profile)
11. Reproducibility controls (temperature, seeds, caching policy)

---

## 8) Status Notes

- Prior Phase 3 localization/drift-only analysis is **deprecated** as the primary claim.
- Phase 3 now requires **both** State-Gini (structure) and DIR/INV (utility).
