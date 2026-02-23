# State-Gini Follow-Up Analyses: Plan and Rationale

The main State-Gini run showed gentags with **lower** Gini (0.60) than baselines (0.71–0.74) and 43.7% of gentags in "Other". To avoid the paper being rejected on a "narrative gap," we run four follow-up analyses to prove gentags are a superior **state object** (synthesis + coverage), not just "a different way to embed words."

---

## 1. τ Sensitivity Sweep (Stability)

**Reviewer risk:** "τ = 0.35 is arbitrary."

**What we test:** The **coverage gap** (Gentags ~57% assigned vs Baselines ~33%) must be a **persistent structural advantage**, not a fluke of the 0.35 cutoff.

**Prediction:** At τ = 0.30, gentag "Other" rate drops; baselines stay spiky (high Gini). At τ = 0.40, gentag Other rises but coverage gap persists. That gives empirical proof that gentags capture more of the semantic mass humans expect.

**Actions:**
- Run `poetry run python scripts/state_gini_full.py --tau 0.30 --output-suffix _tau030` and same for `--tau 0.40 --output-suffix _tau040`.
- Run `poetry run python scripts/phase3a_baselines.py --tau 0.30 --output-suffix _tau030 --gentag-suffix _tau030` and same for 0.40.
- Produce a table: (τ, method, state_gini_mean, other_rate_mean) for τ ∈ {0.30, 0.35, 0.40}.

**Outputs:** `results/phase3/tables/state_localization_tau030.csv`, `_tau040.csv`; `results/phase3a/tables/baseline_state_gini_tau030.csv`, `_tau040.csv`; optional summary `results/phase3/tables/tau_sensitivity_summary.csv`.

---

## 2. Semantic Identity of the "Other" Bucket (Qualitative Probe)

**Worry:** 43.7% in "Other" makes the data look "weird."

**What we test:** Are the 99,411 tags in "Other" **noise**, or **long-tail semantic propositions** that don't fit our 10 facets?

**Examples:** If "Other" contains "dim lighting" (near ambiance but below τ) or "unfriendly cat" (domain nuance), then Other is a **success of granularity**, not a failure of coverage.

**PhD angle:** Like CheckList showing where models fail *specifically*: we show gentags capture **nuance** while RAKE captures **keywords**. Then the Other rate becomes our strongest argument for gentags as a "missing layer."

**Actions:**
- Run `poetry run python scripts/state_gini_other_probe.py` to export all tags in Other (tag, best_facet, best_sim).
- Optional: `--sample 500 --out results/phase3/other_bucket_sample.csv` for a coding sample.
- Document findings in `docs/phase3/other_bucket_probe.md`.

**Outputs:** `results/phase3/other_bucket_tags.csv` (tag, best_facet, best_sim); optional sample CSV; probe doc.

---

## 3. Cross-Facet Similarity Matrix ("Bleed" Check)

**What we test:** We know max pairwise anchor cosine is 0.5817 (Food vs Service). We need: **How many gentags are "near-misses"?** (e.g. 0.34 to Facet A and 0.33 to Facet B.)

**Why it matters:** If most gentags have a **clear primary** facet and a **low secondary** similarity, we have eliminated "embedding bleed" and proved the state is **factorized**, not a diffuse "meaning cloud."

**Actions:**
- Run `poetry run python scripts/state_gini_bleed_check.py` (optionally `--sample 5000` for speed).
- Script computes for each gentag: similarity to all 10 facets; primary = max, secondary = second max; gap = primary − secondary.
- Report: gap mean/std, % near-miss (gap < 0.05), % clear primary (gap ≥ 0.10).

**Outputs:** `results/phase3/bleed_check_summary.json`; optional per-tag CSV via `--out-csv`.

---

## 4. Like-for-Like Unit Alignment (Venue-Aggregated Gentag Gini)

**Problem:** Gentag Gini is per **extraction** (n = 10,373); baselines are per **venue** (n = 230). A reviewer can say the comparison is unfair.

**What we do:** **Venue-aggregated Gentag Gini.** For each venue, pool tags from all extractions (run1 + run2, all models), then compute facet counts and Gini on that pool. One State-Gini per venue for gentags.

**Goal:** Table comparing "1 Venue (Gentags)" vs "1 Venue (TF-IDF)" so the spikiness of baselines isn’t dismissed as a sample-size artifact.

**Actions:**
- Run `poetry run python scripts/state_gini_venue_aggregate.py`. Script loads extractions + tags, groups by venue_id, pools all gentags per venue, assigns at τ = 0.35, computes State-Gini per venue. Uses same quality-venue filter as phase3a (retention.csv) when available.
- Output: `results/phase3/tables/venue_gentag_state_gini.csv` (venue_id, n_tags, state_gini, other_count, other_rate_pct, ...).
- Compare to `results/phase3a/tables/baseline_state_gini.csv` (join on venue_id) for a like-for-like table.

**Outputs:** `results/phase3/tables/venue_gentag_state_gini.csv`; optional `venue_comparison_table.csv` (manual or scripted join).

---

## Mentor’s Verdict

Without these follow-ups, the "lower Gini" result will be read as **failure of gentags to concentrate meaning**. With the τ sweep and the Other probe, we shift the story to **Synthesis and Coverage**: gentags put more mass in the facet space (lower Other, stable across τ) and spread across more facets (balanced Gini); the bleed check and venue alignment close methodological objections.

**Execution order:** (1) τ sensitivity, (2) Other-bucket probe, (3) Bleed check, (4) Venue-aligned Gini. Then update `state_gini_full_run_analysis.md` with follow-up results and narrative.
