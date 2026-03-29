# Phase 3 — State-Gini: Runnable Protocol & Decision Log

This document is the **single source of truth** for the **State-Gini experiment** in Phase 3. State-Gini is **one** of the things we do in Phase 3; other Phase 3 work (e.g. utility proof / attribution experiments) is separate and documented elsewhere (e.g. `docs/_archive/superseded/PHASE3_STATUS.md`, `docs/PHASE3_METHODOLOGY_FIX.md`).

**Goals of this document:**

1. **Run:** You can run the State-Gini experiment from this doc alone—inputs, commands, code paths, and outputs are all specified.
2. **Understand:** When you come back later, every decision (why 10 facets, why τ=0.35, why hard assignment, etc.) is explained in one place.
3. **Preserve:** All information produced or assumed by the State-Gini experiment lives here, so the experiment is reproducible and the knowledge is not scattered.

Parameters are treated as **experimental controls**, not arbitrary settings; the doc maps each control to the code that enforces it.

**Scope:** This document covers **only** the State-Gini structural proof (gentags + RAKE/TF-IDF/YAKE baselines, facet assignment, Gini on counts). It does **not** define or run Phase 4; it does not define other Phase 3 experiments (e.g. Judge LLM, DIR/INV). Those have their own protocols.

**Document outline:**  
§0 Why parameters are controls — §1 Fixed structural anchors (facets, τ, hard assignment, rationale, orthogonality) — §2 Code mapping — §3 Other Phase 3 work (pointer only) — §4 Baseline war — §5 Reproducibility — §6 Execution order — §7 Summary table (appendix) — §8 Facet critique — §9 Methodology comparison — §10 Pre-flight — §11 Output artifacts — §12 Input/output schemas — §13 Success/failure criteria — §14 Data flow — §15 Gini formula — §16 Baseline params — §17 Required inputs — §18 Decisions log — §19 Reviewer Q&A — §20 One-page code summary — §21 End-to-end code flow.

**Phase 2 numbers (reference):** Semantic gap = 0.504 (cosine 0.977, Jaccard 0.471). Retention delta vs random = +0.164. Token-variability correlation = -0.230. See `docs/PHASE2_STABILITY.md`.

---

### How to run the State-Gini experiment

**Prerequisites:** Phase 1 extractions in `results/phase1_downloaded/`; Phase 2 tag embeddings in `results/phase2_cache/` (e.g. `tag_embeddings_text_embedding_3_large_normeval.npz` + `.map.json`); `data/study1_venues_20250117.csv`; `OPENAI_API_KEY` in `.env` (for anchor embeddings).

**Commands (from repo root):**

1. **Gentag State-Gini:**  
   `poetry run python scripts/state_gini_full.py`  
   Optional args: `--run-id`, `--data`, `--results-dir` (see script help).

2. **Baseline State-Gini (RAKE, TF-IDF, YAKE):**  
   `poetry run python scripts/phase3a_baselines.py`  
   Reads Phase 3 gentag results for comparison; needs venue data and Phase 2 cache.

Or run both in order:  
`./scripts/run_phase3.sh`

**Outputs:**  
- `results/phase3/tables/state_localization.csv` — State-Gini per extraction (gentags).  
- `results/phase3/tables/drift_localization.csv` — Drift-Gini (secondary).  
- `results/phase3a/tables/baseline_state_gini.csv` — State-Gini per venue/method (baselines).  
- `results/phase3a/tables/state_gini_summary.csv` — Summary by method.  
- `results/phase3/phase3_v2_manifest.json`, `results/phase3a/phase3a_v2_manifest.json` — Run metadata.

After the run, you can come back to this document to see why each parameter (facets, τ, hard assignment, baselines) was chosen and where it lives in the code.

---

### Plan to do State-Gini (order of operations)

Do **not** run the full State-Gini pipeline until you have:

1. **Checked anchor orthogonality** — so facet axes are not too similar in embedding space.
2. **Chosen the best τ** — so the threshold is justified, not arbitrary.

Then lock parameters and run the full experiment.

| Step | What to do | Why | How |
|------|------------|-----|-----|
| **0. Orthogonality** | Compute anchor embeddings; pairwise cosine between all 10 anchors; report max. | If two anchors are too similar (e.g. cosine > 0.55–0.60), hard assignment can flip between them and stability suffers. | Run `poetry run python scripts/state_gini_preflight.py --orthogonality`. Fix anchor phrases for any pair above the bound; re-run until max pairwise cosine is acceptable. |
| **1. τ sensitivity** | Run facet assignment only (no State-Gini) at τ ∈ {0.30, 0.35, 0.40}. Report for each τ: mean other_rate (fraction of tags below threshold). | Picks a defensible threshold: filters noise without killing coverage. State-Gini is not computed in preflight — only in the full run. | Run `poetry run python scripts/state_gini_preflight.py --tau-sweep [--sample 10]`. Inspect table; choose τ (default 0.35) and document in manifest. |
| **2. Lock** | Set τ and anchor list; record in manifest (e.g. `threshold`, `anchor_max_pairwise_cosine`). | Ensures the full run is auditable and reproducible. | Use `--tau` in `state_gini_full.py` (default 0.35); set `SEMANTIC_THRESHOLD` in `phase3a_baselines.py` to match if needed; write preflight results to `results/phase3/preflight_*.json`. |
| **3. Full run** | Run State-Gini for gentags, then baselines (or `./scripts/run_phase3.sh`). | Produces Table 1 and all outputs. | See "How to run" above. |

**Summary:** First run preflight (orthogonality + τ sweep), then lock τ and anchors, then run the full State-Gini experiment.

**Preflight script:** `scripts/state_gini_preflight.py` (standalone; same style as phase2_analysis / state_gini_full).

- **Orthogonality only:**  
  `poetry run python scripts/state_gini_preflight.py --orthogonality`  
  Writes `results/phase3/preflight_orthogonality.json`. Needs `OPENAI_API_KEY` (to embed anchors once).

- **τ sweep only:**  
  `poetry run python scripts/state_gini_preflight.py --tau-sweep [--sample 10]`  
  Uses Phase 1 extractions and Phase 2 tag cache; runs facet assignment at τ = 0.30, 0.35, 0.40. Optional `--sample 10` limits to 10 venues. Writes `results/phase3/preflight_tau_sweep.json`.

- **Both:**  
  `poetry run python scripts/state_gini_preflight.py --orthogonality --tau-sweep`

---

## 0. Why Parameters Are Controls, Not Settings

In top-tier research, the difference between a "project" and a "paper" is the rigor with which boundaries are defined **before** the first line of code runs. For the **State-Gini experiment** we lock:

- **Structural anchors** (facet list, threshold τ, hard assignment),
- **Baseline parity** (same methodology for gentags and RAKE/TF-IDF/YAKE),
- **Reproducibility** (seeds, caching, manifest),

so the result is defensible. This document locks those controls and shows how the code enforces them.

---

## 1. Fixed Structural Anchors (Phase 3 Fix)

### 1.1 Facet list and anchor phrases (frozen)

- **Claim:** Gentags are a **factorized representation**: semantic mass concentrates into interpretable aspects.
- **Control:** 10 facets with **descriptive anchor phrases** (not single words) to avoid lexical tricks. Anchors are embedded once with `text-embedding-3-large` and reused for all assignments.
- **Why descriptive phrases:** Single-word anchors encourage token matching; phrases force a semantic match and support the claim that gentags capture meaning, not surface form.

**Facets as diagnostic probes (not ontology):** The facets are **frozen diagnostic probes**—they are not part of the gentag representation. They are the axes against which we measure State-Gini, analogous to CheckList’s capability matrix. We do **not** claim these 10 axes are the “true” ontology of the domain; we claim they jointly cover the “semantic mass” we care about for this study.

| # | Facet key        | Anchor phrase (frozen) |
|---|------------------|------------------------|
| 1 | food_quality     | food quality, taste, freshness, delicious meals |
| 2 | coffee_drinks    | coffee, espresso, latte, beverages, drinks |
| 3 | service          | service quality, staff friendliness, speed, waiters |
| 4 | ambiance         | atmosphere, ambiance, vibe, decor, cozy environment |
| 5 | price_value      | price, value for money, affordable, expensive |
| 6 | crowding         | crowded, busy, wait times, lines, availability |
| 7 | seating          | seating, tables, outdoor patio, indoor space |
| 8 | dietary          | dietary options, vegan, vegetarian, gluten-free |
| 9 | portions         | portion size, generous servings, filling meals |
|10 | location         | location, parking, accessibility, neighborhood |

**Facet 10 “sink” risk:** Facet 10 (location / overall experience) must **not** act as a catch-all “OTHER”. In the code we **report the `other` bucket separately** (tags below τ). If >50% of tags fall in `other`, the facet list has failed to cover the domain and the localization claim is weakened. We also report per-facet counts so reviewers can see if one facet absorbs too much mass.

### 1.2 Similarity threshold τ = 0.35

- **Control:** τ = **0.35** for both gentags and baselines. Non-negotiable for proving **hard assignment**.
- **Why:** The methodology fix identified “embedding bleed” as the bug—e.g. “great coffee” influencing a “service” facet under soft assignment. With a **hard** τ=0.35, a tag is assigned to the argmax facet only if similarity ≥ τ; otherwise it goes to `other`. This proves gentags achieve a factorized representation while classical methods (TF-IDF, etc.) produce a “statistical spread.”
- **Sensitivity:** A reviewer may ask “why 0.35?” We must show that this threshold filters low-confidence noise without suppressing legitimate nuanced semantics. In the code, τ is a single frozen constant; sensitivity can be reported in an ablation (e.g. τ ∈ {0.30, 0.35, 0.40}) in the paper.

### 1.3 Hard assignment (argmax + τ)

- **Rule:** Each tag (gentag or baseline keyword) is assigned to **at most one** facet: the facet with highest cosine similarity to the tag embedding, and only if that similarity ≥ τ; otherwise the tag is counted in `other`.
- **State-Gini:** Gini is computed on the **integer counts** of the 10 facets only (excluding `other`). High Gini ⇒ mass concentrated in few facets (factorized). Low Gini ⇒ spread (diffuse).

### 1.4 Why τ = 0.35 (detailed rationale)

A strict reviewer will ask: *Why 0.35?* The threshold must:

1. **Filter noise:** Low-similarity tags (e.g. generic or off-domain phrases) should not be forced into a facet; they go to `other`.
2. **Preserve legitimate semantics:** Nuanced but valid tags (e.g. “friendly but slow service”) should still assign to the best-matching facet when that similarity is above the threshold.
3. **Be defensible:** The same τ is used for gentags and all baselines, so no method gets a softer or harder assignment rule.

**Empirical justification:** In embedding space, typical “clear match” tag–anchor similarities for our domain (venue reviews) often fall in the 0.4–0.8 range; random or very weak matches fall below 0.3. A value of 0.35 sits between “noise” and “weak but real” signal. **Sensitivity analysis (for the paper):** Run the same pipeline with τ ∈ {0.30, 0.35, 0.40} and report State-Gini and `other_rate` for each. If results are robust (gentags still clearly above baselines and `other_rate` not exploding), the choice is defended.

### 1.5 Orthogonality check (anchor embeddings)

We do **not** assume facets are uncorrelated in the real world (e.g. “cleanliness” and “food quality” may co-occur). We **do** require that the **anchor embeddings** are not too similar to each other; otherwise hard assignment can flip randomly between two nearby facets and stability metrics suffer.

**Procedure:**

1. After computing anchor embeddings (once per run or from cache), build the 10×10 pairwise cosine similarity matrix.
2. For each pair (i, j), i ≠ j, check cosine(anchor_i, anchor_j).
3. **Failure condition:** If any pair has cosine > 0.55–0.60, flag for review. Consider refining the anchor phrases for those facets so their embeddings separate in vector space.
4. Record the max pairwise cosine in the manifest (e.g. `anchor_max_pairwise_cosine`) for transparency.

**Code location:** This check is not yet in the scripts; it should be added as a pre-flight step (e.g. after `compute_anchor_embeddings`, before the main loop). Pseudocode: loop over `FACETS`, compute `cosine_similarity(anchor_embeddings[f1], anchor_embeddings[f2])` for f1 < f2, report max and warn if > 0.60.

---

## 2. Where This Is Implemented in Code

### 2.1 Frozen facets and anchors

**File:** `scripts/state_gini_full.py` (and mirrored in `scripts/phase3a_baselines.py`)

```python
# scripts/state_gini_full.py (excerpt)
FACETS = [
    "food_quality", "coffee_drinks", "service", "ambiance", "price_value",
    "crowding", "seating", "dietary", "portions", "location"
]

FACET_ANCHORS = {
    "food_quality": "food quality, taste, freshness, delicious meals",
    "coffee_drinks": "coffee, espresso, latte, beverages, drinks",
    "service": "service quality, staff friendliness, speed, waiters",
    # ... (all 10 as in table above)
}
```

**Purpose:** Single source of truth for facet names and anchor text. Both gentag and baseline pipelines use the same `FACETS` and `FACET_ANCHORS`.

### 2.2 Threshold τ and hard assignment

**File:** `scripts/state_gini_full.py`

```python
SEMANTIC_THRESHOLD = 0.35

def hard_assign_facet(tag, tag_embeddings, anchor_embeddings, threshold=SEMANTIC_THRESHOLD):
    """Assign tag to exactly ONE facet via argmax, or None if below threshold."""
    if tag not in tag_embeddings:
        return None, 0.0
    tag_emb = tag_embeddings[tag]
    best_facet, best_sim = None, -1.0
    for facet in FACETS:
        sim = cosine_similarity(tag_emb, anchor_embeddings[facet])
        if sim > best_sim:
            best_sim, best_facet = sim, facet
    if best_sim >= threshold:
        return best_facet, best_sim
    return None, best_sim  # → "other"
```

**Purpose:** Ensures no embedding bleed: each tag maps to one facet or to `other`. Same function shape is used in `phase3a_baselines.py` with keyword embeddings.

### 2.3 Facet counts and State-Gini

**File:** `scripts/state_gini_full.py`

```python
def compute_facet_counts(tags, tag_embeddings, anchor_embeddings, threshold=SEMANTIC_THRESHOLD):
    counts = {facet: 0 for facet in FACETS}
    other_count = 0
    for tag in tags:
        facet, sim = hard_assign_facet(tag, tag_embeddings, anchor_embeddings, threshold)
        if facet is not None:
            counts[facet] += 1
        else:
            other_count += 1
    return counts, other_count

def compute_state_gini(tags, tag_embeddings, anchor_embeddings, threshold=SEMANTIC_THRESHOLD):
    counts, other_count = compute_facet_counts(...)
    count_array = np.array([counts[f] for f in FACETS])
    state_gini = gini_coefficient(count_array)  # Gini on 10 facet counts only
    return state_gini, counts, other_count
```

**Purpose:** State-Gini is computed on the 10-dimensional count vector only; `other_count` is stored and reported separately for coverage transparency.

### 2.4 Anchor embeddings (once per run)

**File:** `scripts/state_gini_full.py`

```python
def compute_anchor_embeddings(client):
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    embeddings = embed_texts_batch(client, anchor_texts, batch_size=16)
    return {facet: emb for facet, emb in zip(FACETS, embeddings)}
```

**Purpose:** Anchors are embedded with the same model as tags (`text-embedding-3-large`). For full reproducibility, anchor embeddings can be cached and checksummed in the manifest.

### 2.5 Baselines use the same protocol

**File:** `scripts/phase3a_baselines.py`

- RAKE, TF-IDF, YAKE keywords are extracted with fixed `k=25` (or median gentag count) and max phrase length 4.
- Each keyword is **embedded**, then passed through the **same** `hard_assign_facet` and `compute_state_gini` with the **same** anchor embeddings and τ=0.35.
- So the only difference between gentags and baselines is the **source** of the phrases (LLM vs RAKE/TF-IDF/YAKE), not the assignment or Gini logic.

```python
# phase3a_baselines.py: same threshold and facets
SEMANTIC_THRESHOLD = 0.35
# ... same FACETS, FACET_ANCHORS ...
state_gini, counts, other_count = compute_state_gini(kw_embs, anchor_embeddings)
```

---

## 3. Other Phase 3 work (not this experiment)

**Scope of this document:** This doc is only for the **State-Gini** experiment. Phase 3 also includes other work (e.g. utility proof: Judge LLM, DIR/INV attribution tests). Those are **separate** experiments with their own protocols—see `docs/_archive/superseded/PHASE3_STATUS.md` and `docs/PHASE3_METHODOLOGY_FIX.md`. You do **not** need Judge or DIR/INV to run or interpret State-Gini.

**What follows in §3.1–§3.9** is **context only** (for when you work on the utility proof later). It does not define what you run for State-Gini. State-Gini = structural proof only (this doc). Utility proof = other Phase 3 work (other docs).

### 3.1 Judge LLM: evidence-conditioning

- **Control:** The Judge must reason **only** over the externalized gentag state. It must be **forbidden** from using internal knowledge (e.g. inventing “good vibe” when gentags only say “cheap food”).
- **Implementation:** Strict system prompt: “Use ONLY the provided gentags. Do NOT use external knowledge. Output JSON only: {\"score\": ..., \"justification\": ..., \"tags_used\": [...]}.” Parsing must validate JSON and optionally check that `tags_used` ⊆ provided gentags.

### 3.2 DIR (Directional Expectation)

- **Idea:** For a given user profile, a **pivotal** change to the gentag state should move the score in a predictable direction.
- **Example:** Profile “Disability-Access Seeker”; delete gentag “no ramp” → score should **increase** (monotonic).
- **Control:** Pre-define 3–5 user profiles and, for each, a small set of interventions (e.g. delete/flip specific gentags) with expected direction. Run Judge on original and revised state; require monotonicity.
- **Comparison:** CheckList reported ~34.6% failure rate on DIR for commercial models; we aim to show that gentags, by being discrete and addressable, reduce this failure rate when the Judge is evidence-conditioned.

### 3.3 INV (Invariance)

- **Idea:** Paraphrases that preserve meaning should not change the score beyond ε.
- **Source:** Use Phase 2’s **Semantic Gap (0.504)** finding: gentags are lexically unstable but semantically stable. So e.g. “fast service” → “rapid response” should yield |Δscore| ≤ ε.
- **Control:** Paraphrase pairs from Phase 2 or a fixed list; run Judge on A vs B; report pass rate for |Δscore| ≤ ε.

### 3.4 User profiles and sparse venues

- **Profiles:** At least 3–5 distinct personas (e.g. “Disability-Access Seeker,” “Budget Traveler,” “Coffee Quality Focus”) so DIR interventions are meaningful.
- **Sampling:** Focus causal (DIR/INV) experiments on **sparse venues** from Phase 2, e.g.:
  - `KzvuSntI35Z638fGoOJ4` (12 tokens)
  - `GVn2q90PoVQ5p6EcJb4W` (5 tokens)  
  In these, evidence is weakly constrained and removing or changing one gentag has the largest observable effect on the Judge.

### 3.5 Judge LLM: preventing hallucination

The Judge must **not** invent information. If the gentags only say “cheap food,” the Judge must not justify a score with “good vibe” or “great ambiance.” That would break the causal chain: we need the decision to be a function of the **externalized state** only.

**Controls:**

1. **Evidence-conditioning in the prompt:** Explicit instruction: “Use ONLY the provided gentags. Do NOT use external knowledge or infer facts not stated in the gentags.”
2. **JSON-only output:** Require a single JSON object (no prose before/after). Schema: `{"score": int 0-100, "justification": "one sentence", "tags_used": ["tag1", "tag2", ...]}`. This forces structured output and makes parsing deterministic.
3. **Validation of `tags_used`:** After parsing, check that every element of `tags_used` is in the provided gentag list (or in the revised list for DIR). If the Judge cites a tag that was not provided, count it as a hallucination and optionally reject or flag the run.
4. **Temperature:** Use temperature = 0 (or the minimum the API allows) for the Judge so responses are deterministic and reproducible.
5. **Primary and backup Judge models:** Use one high-capacity model (e.g. OpenAI) as primary and a second for cross-validation on a subset, to ensure the finding is not model-specific.

**Pilot check:** On 10 venues, inspect Judge outputs. If justifications ever reference concepts not present in the gentag list, tighten the system prompt and retest before the full run.

### 3.6 User profiles (full definitions for DIR)

User profiles must be **frozen** and **diverse** enough to trigger meaningful DIR interventions. Minimum 3–5 personas. Example set:

| Profile ID | Name | Short description | Example pivotal need |
|------------|------|-------------------|----------------------|
| P1 | Disability-Access Seeker | Prioritizes wheelchair access, ramps, accessible restrooms | “no ramp” is negative; removing it should ↑ score |
| P2 | Budget Traveler | Cares about price, value, affordable options | “expensive” or “overpriced” negative; “good value” positive |
| P3 | Coffee Quality Focus | Cares about coffee, espresso, drink quality | “great coffee” positive; “weak espresso” negative |
| P4 | Service-Oriented | Cares about speed, friendliness, wait times | “slow service” negative; “friendly staff” positive |
| P5 | Dietary-Restriction | Vegan, vegetarian, gluten-free options | “no vegan options” negative; “vegan options” positive |

For each profile, we pre-define **pivotal propositions**: specific gentags (or patterns) whose addition/removal/flip has a clear expected direction (e.g. delete “no ramp” → score must increase for P1).

### 3.7 DIR intervention rules and pivotal propositions

**Directional Expectation (DIR):** For a given user profile, we apply a **single intervention** to the gentag state (e.g. delete one tag, or flip one tag to its opposite). The Judge scores both the original and the revised state. We require **monotonicity**: the score must move in the expected direction (e.g. up when we remove a negative tag for a profile that cares about that dimension).

**Concrete example (from Phase 2 sparse venue):**

- **Venue:** `KzvuSntI35Z638fGoOJ4` (12 tokens). Sample gentags: `not wheelchair accessible`, `lack disability access`, `no ramp`.
- **Profile:** P1 (Disability-Access Seeker).
- **Intervention:** DELETE “no ramp” from the gentag list (simulate the venue adding a ramp).
- **Expected:** Revised state score **>** original state score (monotonic increase).
- **CheckList comparison:** CheckList found ~34.6% failure rate on DIR tests (e.g. adding negative phrases led to wrong direction for commercial models). Our goal: show that with **evidence-conditioned** Judge and **discrete, addressable** gentags, DIR failure rate is significantly lower, because the representation is factorized and editable.

**Intervention types to support:**

- **Delete:** Remove one gentag (e.g. “no ramp”). Expected direction depends on profile (for P1, delete negative → score up).
- **Flip (optional):** Replace a negative tag with a positive one (e.g. “no ramp” → “wheelchair accessible”). Again, expected direction is pre-defined.

### 3.8 INV protocol (paraphrase pairs and epsilon)

**Invariance (INV):** If two gentag sets are **paraphrases** (same meaning, different words), the Judge’s scores for the two sets should differ by at most ε.

**Source of paraphrases:** Phase 2 showed **Semantic Gap = 0.504** (cosine 0.977 vs Jaccard 0.471): gentags are lexically unstable but semantically stable across runs. So we can use:

- Pairs from Phase 2 run-to-run data where the same venue received different surface forms (e.g. “fast service” in run 1 vs “rapid response” in run 2), or
- A fixed list of paraphrase pairs (e.g. “fast service” ↔ “rapid response,” “great coffee” ↔ “excellent espresso”) chosen to match the domain.

**Protocol:**

1. For each (venue, profile), take gentag set A. Produce set B by replacing one or more tags with their paraphrases (same meaning).
2. Judge scores both A and B (same profile).
3. Compute Δscore = |score_A − score_B|. **Pass:** Δscore ≤ ε. **Fail:** Δscore > ε.
4. Define ε in advance (e.g. ε = 2 or 3 points on a 0–100 scale). Report INV pass rate.

**Rationale:** If the Judge is truly reasoning over semantic content, lexical variation that preserves meaning should not change the score. INV tests that the Judge is not overly sensitive to surface form.

### 3.9 CheckList comparison and target

- **CheckList (Ribeiro et al.):** Defined capability tests (e.g. negation, NER, taxonomy) and found **actionable bugs** in commercial NLU models (e.g. 34.6% failure on certain DIR-style tests when adding negative phrases).
- **Our adaptation:** We use DIR and INV as **capability tests** for “attribution-aware reasoning over externalized state.” Our facets are domain-specific probes (venue semantics) rather than generic linguistic capabilities.
- **Target:** Show that gentags, by being discrete and editable, **reduce** DIR failure rate compared to (a) the same Judge on dense embeddings (no discrete tags), and/or (b) baseline keyword representations. We report DIR pass rate and INV pass rate; comparison with CheckList’s 34.6% gives context.

---

## 4. Baseline War (Fair Comparison)

- **Control:** RAKE, TF-IDF, and YAKE must be evaluated with the **exact same** State-Gini methodology: same facets, same anchors, same τ, same hard assignment.
- **PhD critique:** If gentags got any “soft” advantage (e.g. different clustering or threshold), the comparison would be invalid. In our code, baselines and gentags share:
  - `FACETS`, `FACET_ANCHORS`, `SEMANTIC_THRESHOLD`
  - `hard_assign_facet`, `compute_state_gini`, `gini_coefficient`
- **Expected:** Gentags State-Gini in the 0.5–0.7 range; TF-IDF/RAKE/YAKE in the 0.1–0.3 range. That gap is the mathematical evidence of “semantic synthesis” vs “statistical spread.”

---

## 5. Prompt templates (for other Phase 3 experiments only)

The Judge LLM and DIR/INV experiments use frozen prompt templates (baseline decision, DIR intervention, INV paraphrase). Those are **not** used in the State-Gini experiment. Full templates and validation rules: see `docs/PHASE3_METHODOLOGY_FIX.md` (§4).

---

## 6. Reproducibility and Sampling

- **Seeds and temperature:** As in the Phase 1 manifest: lock extraction seeds and model temperatures so extractions can be re-run and audited. Document in the manifest: `extraction_seed`, `judge_temperature` (0), and any model version strings.
- **Caching:** Tag and anchor embeddings should be cached (e.g. in `results/phase2_cache/`) with stable filenames (e.g. `tag_embeddings_text_embedding_3_large_normeval.npz`) and optional hashes in the manifest so third parties can verify which embeddings were used.
- **Pilot (10 venues):** Before the full State-Gini run, run on 10 venues only. Check that facet counts and `other_count` look plausible and that no single facet dominates unreasonably. Then scale to the full set.

---

## 7. Execution Order (Do Not Invert)

1. **Preflight (do first):** Run orthogonality check and τ sensitivity sweep (`scripts/state_gini_preflight.py`). Confirm anchor max pairwise cosine is acceptable (e.g. < 0.60); choose τ (e.g. 0.35) from the sweep; lock parameters in code and record in manifest.

2. **Run State-Gini (this experiment):**  
   Run for **gentags** and for **RAKE, TF-IDF, YAKE** with the frozen protocol. Produce Table 1: State-Gini by method. If gentags are not clearly more localized than baselines, the “new representation primitive” claim needs re-evaluation.

3. **Other Phase 3 work:** Utility proof (Judge, DIR/INV) is a separate experiment; run after State-Gini if desired. See `docs/_archive/superseded/PHASE3_STATUS.md`.

---

## 8. Summary Table (Paper Appendix)

| Parameter | Value/Source | Purpose |
|-----------|--------------|---------|
| Facet assignment | Hard (argmax) | Eliminate embedding bleed |
| Threshold (τ) | 0.35 | Filter low-confidence noise; same for gentags and baselines |
| DIR target | Monotonicity (score ↑/↓) | Prove attribution-aware reasoning |
| INV target | Score epsilon ≈ 0 | Prove stability under lexical variation (paraphrase) |
| Baselines | RAKE, TF-IDF, YAKE | Same State-Gini protocol → fair comparison |
| Other bucket | Excluded from Gini; reported separately | Coverage and “sink” transparency |
| Probe framing | Facets = diagnostic probes, not ontology | Avoid reviewer confusion |

---

## 9. Facet Critique and Defenses

- **Probes vs ontology:** Facets are **frozen diagnostic probes**, not a claim about the “true” structure of the domain. We state this explicitly in the method.
- **Facet 10 sink:** We restrict Facet 10 to concepts like location, parking, accessibility, crowd/flow. We report **other** and per-facet counts so a hidden “OTHER” bucket cannot inflate Gini undetected.
- **τ = 0.35:** Justified as the shared threshold that removes embedding bleed while keeping meaningful assignments; sensitivity can be reported in an ablation.
- **Orthogonality:** We do not assume facets are orthogonal in the world (e.g. cleanliness vs food quality may correlate). We **do** require that anchor **embeddings** are not too similar pairwise (e.g. pairwise cosine < 0.55–0.60). Pre-flight check: compute pairwise cosine for anchor embeddings; if any pair exceeds the chosen bound, refine anchor phrases.

---

## 10. Comparison Table (Methodology Section)

| Feature | Topic models (e.g. LDA) | Dense embeddings | Gentag facet probes |
|--------|--------------------------|------------------|----------------------|
| Source of axes | Statistical (learned) | Opaque (latent) | Frozen (human-defined) |
| Assignment | Soft (probabilistic) | Diffuse (bleed) | Hard (argmax + τ) |
| Interpretability | Moderate | Low | High |
| Goal | Clustering | Retrieval | Diagnostic probing |

---

## 11. Pre-Flight Checklist

Before running the full State-Gini pipeline:

1. **Anchor overlap:** Pairwise cosine between anchor embeddings; flag if any pair > 0.55–0.60.
2. **Coverage:** After a pilot run, report `other_rate` (e.g. % of tags below τ). If >50%, facet list coverage is insufficient.
3. **Pilot:** 10 venues; validate outputs (counts, Gini, manifest). For Phase 4, validate Judge JSON and score logic on the same pilot.

---

## 12. Output Artifacts and Code

| Artifact | Path | Produced by |
|----------|------|-------------|
| State-Gini (gentags) | `results/phase3/tables/state_localization.csv` | `state_gini_full.py` |
| State-Gini (baselines) | `results/phase3a/tables/baseline_state_gini.csv` | `phase3a_baselines.py` |
| Summary | `results/phase3a/tables/state_gini_summary.csv` | `phase3a_baselines.py` |
| Manifest (Phase 3) | `results/phase3/phase3_v2_manifest.json` | `state_gini_full.py` |
| Manifest (Phase 3A) | `results/phase3a/phase3a_v2_manifest.json` | `phase3a_baselines.py` |

Manifests should record: threshold, facet list, run id, timestamp, and (optionally) hashes of cached embeddings.

---

## 13. Input and output schemas (CSV columns, manifest)

### state_localization.csv (gentags, Phase 3)

| Column | Type | Description |
|--------|------|-------------|
| exp_id | str | Extraction ID (venue × model × prompt × run) |
| venue_id | str | Venue identifier |
| model_key | str | claude / gemini / grok / openai |
| prompt_type | str | anti_hallucination / minimal / short_phrase |
| run_number | int | Run index (1, 2, ...) |
| n_gentags | int | Number of gentags for this extraction |
| gentag_state_gini | float | State-Gini (Gini on 10 facet counts) |
| gentag_other_count | int | Tags below τ (not in any facet) |
| gentag_assigned_count | int | n_gentags − gentag_other_count |
| gentag_count_{facet} | int | Count in each facet (10 columns) |

### baseline_state_gini.csv (Phase 3A)

| Column | Type | Description |
|--------|------|-------------|
| venue_id | str | Venue identifier |
| method | str | tfidf / rake / yake |
| n_keywords | int | Number of keywords extracted |
| state_gini | float | State-Gini for this venue/method |
| assigned_count | int | Keywords above τ |
| other_count | int | Keywords below τ |
| count_{facet} | int | Per-facet counts (10 columns) |
| retention_cosine | float | (Optional) Cosine to review embedding |

### Manifest JSON (phase3_v2_manifest.json)

Fields to include: `phase`, `run_id`, `timestamp_utc`, `methodology`, `threshold` (0.35), `facets` (list of 10 names), `counts` (n_extractions, n_venues, etc.), `results` (state_gini_mean, state_gini_std, ...), optionally `anchor_max_pairwise_cosine`, `embedding_model`, `cache_hashes`.

---

## 14. Success and failure criteria (quantitative)

**State-Gini (Phase 3):**

- **Success:** Gentags mean State-Gini in the **0.5–0.7** range and **clearly above** baselines (e.g. TF-IDF/RAKE/YAKE in 0.1–0.3). The gap is the evidence of factorized representation.
- **Failure:** Gentags overlap the baseline range or the advantage is marginal (e.g. < 0.1 difference). Then the “new representation primitive” claim is not supported; re-evaluate the structural claim before other Phase 3 work.

**Coverage:**

- **Success:** `other_rate` (fraction of tags below τ) **< 50%** overall. If > 50%, the facet list does not cover the domain’s semantic mass.
- **Failure:** other_rate > 50% → refine facets or report as limitation.

**Other Phase 3 (utility proof):**

- **(Other Phase 3)** DIR/INV pass rates apply to the utility proof, not State-Gini. See `docs/_archive/superseded/PHASE3_STATUS.md`. Original: DIR = score moves in expected direction (e.g. delete negative → score up). Report pass rate; target is significantly higher than CheckList’s ~34.6% failure rate (i.e. we want a high pass rate).
- DIR/INV (utility proof only): pass = |Δscore| ≤ ε (e.g. ε = 2 or 3). Report pass rate; high pass rate supports “semantic stability under paraphrase.”

---

## 15. Sparse venues (for other Phase 3 experiments, not State-Gini)

The following is **reference only** for when you run the utility proof (Judge, DIR/INV). State-Gini does **not** require sparse-venue sampling; it runs on all venues with extractions. From Phase 2, these venues have **weakly constrained** evidence; a single gentag change has the largest observable effect on the Judge.

| venue_id | total_tokens | mean_pairwise_distance | Why use for DIR/INV |
|----------|----------------|------------------------|---------------------|
| KzvuSntI35Z638fGoOJ4 | 12 | 0.307 | Very sparse; sample gentags include “no ramp,” “not wheelchair accessible” → ideal for P1 (Disability-Access) DIR. |
| GVn2q90PoVQ5p6EcJb4W | 5 | 0.129 | Extremely sparse; “rich food,” “fast service” → good for service/quality INV and DIR. |

Additional sparse venues can be taken from Phase 2 tables (e.g. token bucket < 200 or lowest token count). Prioritize venues where we have at least one extraction with 3+ gentags so interventions are meaningful.

---

## 16. Data flow (full pipeline steps)

1. **Load Phase 1 outputs:** `results/phase1_downloaded/{run_id}_extractions_*.csv`, `*_tags_*.csv` → extractions_df, tags_df. Columns: exp_id, venue_id, model_key, prompt_type, run_number; tags: exp_id, tag_norm_eval.
2. **Load venues:** `data/study1_venues_20250117.csv` (for baseline keyword extraction: review text per venue).
3. **Load or compute embeddings:** Tag embeddings from `results/phase2_cache/tag_embeddings_*_normeval.npz` (+ .map.json). Anchor embeddings: compute once via `compute_anchor_embeddings(client)` (or load from cache if cached).
4. **Pre-flight (optional but recommended):** Anchor pairwise cosine check; abort or warn if max > 0.60.
5. **Gentag State-Gini:** For each extraction row, get gentags → `compute_facet_counts` → `compute_state_gini` → write row to state_localization.csv.
6. **Baselines:** For each venue, get review text → RAKE/TF-IDF/YAKE (k=25, max 4 words) → embed keywords → same `compute_state_gini` → write to baseline_state_gini.csv (Phase 3A).
7. **Summarize:** state_gini_summary.csv (mean/std per method); optionally plots (state_gini_comparison.png).
8. **Manifest:** Write phase3_v2_manifest.json and phase3a_v2_manifest.json with threshold, facets, counts, results, timestamps.

---

## 17. Gini coefficient formula

For a vector of non-negative values \(x_1, \ldots, x_n\) (here: the 10 facet counts), sort them: \(x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}\). The Gini coefficient is:

\[
G = \frac{2 \sum_{i=1}^{n} i \cdot x_{(i)}}{n \sum_{i=1}^{n} x_{(i)}} - \frac{n+1}{n}
\]

- **G = 0:** perfectly equal counts across facets (maximally diffuse).
- **G → 1:** all mass in one facet (maximally concentrated).

**Code:** `scripts/state_gini_full.py`, function `gini_coefficient(values)` (lines 96–113): sorts values, applies the formula, returns `max(0, gini)`.

---

## 18. Baseline extraction parameters (code reference)

So that baselines are comparable to gentags:

| Parameter | Value | Code location |
|-----------|-------|---------------|
| k (number of keywords) | 25 or median gentag count per venue | `phase3a_baselines.py`: DEFAULT_K = 25; or `median_gentags` from retention data |
| Max phrase length | 4 words | MAX_PHRASE_WORDS = 4; in RAKE `max_length=4`, in TF-IDF `ngram_range=(1,4)`, in YAKE `n=4` |
| Embedding model | text-embedding-3-large | Same as gentags and anchors |
| Assignment | Hard, τ = 0.35 | Same `hard_assign_facet`, `compute_state_gini` as Phase 3 |

RAKE: `Rake(min_length=1, max_length=4, include_repeated_phrases=False)`, then `get_ranked_phrases()` filtered to ≤ 4 words, top k. TF-IDF: `TfidfVectorizer(ngram_range=(1,4), stop_words='english', max_features=k*5, min_df=1)`, then top-k by score. YAKE: `KeywordExtractor(lan="en", n=4, dedupLim=0.7, top=k*2)`, then filter to ≤ 4 words, top k.

---

## 19. Required inputs (full list with paths)

| Input | Path / source | Used in |
|-------|----------------|--------|
| Phase 1 extractions | `results/phase1_downloaded/{run_id}_extractions_*.csv` | Phase 3 gentag State-Gini |
| Phase 1 tags | `results/phase1_downloaded/{run_id}_tags_*.csv` | Phase 3 (tag_norm_eval) |
| Venue data | `data/study1_venues_20250117.csv` | Phase 3A (review text for baselines) |
| Tag embeddings | `results/phase2_cache/tag_embeddings_text_embedding_3_large_normeval.npz` + `.map.json` | Phase 3 |
| Facet list + anchors | Hard-coded in `state_gini_full.py` and `phase3a_baselines.py` | Phase 3 and 3A |
| (Phase 4) User profiles | `phase3_profiles.csv` or equivalent (frozen) | DIR/INV |
| (Phase 4) Paraphrase pairs | From Phase 2 or fixed list | INV |

---

## 20. Decisions log (locked parameters)

These are **locked** before the first full run; any change should be documented and justified.

| Decision | Value | Date / note |
|----------|-------|-------------|
| Facet count | 10 | Frozen |
| Facet names + anchor phrases | See §1.1 table | Frozen |
| Threshold τ | 0.35 | Frozen; sensitivity in ablation if needed |
| Assignment type | Hard (argmax + τ) | Frozen |
| Baselines | RAKE, TF-IDF, YAKE | Same methodology as gentags |
| k (keywords) | 25 or median gentags | Phase 3A |
| Max phrase length | 4 words | Phase 3A |
| Embedding model | text-embedding-3-large | All embeddings |
| Judge output | JSON only; schema with score, justification, tags_used | Phase 4 |
| Judge temperature | 0 | Phase 4 |
| DIR/INV epsilon | To be set (e.g. 2–3 points) | Phase 4 |
| Pilot size | 10 venues | Before full run |

---

## 21. Anticipated reviewer questions and answers

**Q: Why 10 facets? Why these?**  
A: The 10 facets are **diagnostic probes** chosen to cover the “semantic mass” of venue reviews (food, service, ambiance, price, etc.). We do not claim they are the true ontology; we claim they are a frozen set of axes for measuring concentration. Coverage is validated by reporting the `other` bucket; if > 50% of tags fall below τ, the set would be revised.

**Q: Why τ = 0.35?**  
A: Same threshold for gentags and baselines to avoid method advantage. It filters low-confidence noise while retaining meaningful assignments. We can report a sensitivity analysis (τ ∈ {0.30, 0.35, 0.40}) in the paper.

**Q: Isn’t State-Gini just measuring how concentrated your keyword list is?**  
A: No. We apply the **same** State-Gini protocol (same facets, same τ, same hard assignment) to gentags and to RAKE/TF-IDF/YAKE. So the comparison is fair. If gentags consistently show higher Gini, it means their semantic mass is more concentrated in the probe facets than statistical keyword methods—i.e. factorized representation vs statistical spread.

**Q: How do you prevent the Judge from hallucinating?**  
A: Evidence-conditioned prompt (“Use ONLY the provided gentags”), JSON-only output, validation that `tags_used` ⊆ provided gentags, and temperature 0. Pilot on 10 venues to catch any justification that references absent concepts.

**Q: Why sparse venues for DIR/INV?**  
A: When evidence is sparse, the state is weakly constrained and a single gentag change has a larger marginal effect. So the causal signal (score change when we delete “no ramp”) is easier to detect than in venues with many gentags where one change is diluted.

**Q: What if gentags and TF-IDF have similar State-Gini?**  
A: Then we do not have evidence that gentags are more factorized than classical keywords on this metric. We would report the result and either refine the claim (e.g. factorized only under certain conditions) or improve the representation/extraction before claiming a “new primitive.”

---

## 22. One-Page "How We Do It in Code" Summary “How We Do It in Code” Summary

1. **Load extractions** → `load_phase1_data()` in `state_gini_full.py`; columns include `venue_id`, `model_key`, `prompt_type`, `run_number`, `tag_norm_eval`.
2. **Load facets and anchors** → `FACETS` and `FACET_ANCHORS` in the same file (and in `phase3a_baselines.py`).
3. **Embed anchors once** → `compute_anchor_embeddings(client)`; same client/model as tag embeddings.
4. **Tag embeddings** → From Phase 2 cache (`tag_embeddings_*_normeval.npz` + `.map.json`); or embed on the fly and cache.
5. **Hard assignment** → For each tag, `hard_assign_facet(tag, tag_embeddings, anchor_embeddings, 0.35)` → (facet or None, sim).
6. **Facet counts** → `compute_facet_counts()` → `counts` (10 facets) + `other_count`.
7. **State-Gini** → `gini_coefficient(np.array([counts[f] for f in FACETS]))`.
8. **Baselines** → In `phase3a_baselines.py`: extract RAKE/TF-IDF/YAKE keywords per venue, embed them, then same `hard_assign_facet` and `compute_state_gini` with same anchors and τ.
9. **Write** → CSVs with per-facet counts and `other_count`; manifest with threshold, facets, run id.

This is the full chain from “frozen controls” to “Table 1” and sets up Phase 4 attribution experiments without changing the structural protocol.

---

## 23. End-to-end code flow (one extraction)

How a **single** extraction gets a State-Gini value in the current codebase:

**Script:** `scripts/state_gini_full.py`

1. **Inputs for one row:** `exp_id`, `venue_id`, `model_key`, `prompt_type`, `run_number` from `extractions_df`; gentags from `tags_df` where `exp_id` matches (column `tag_norm_eval`).
2. **Gentags list:** e.g. `["great coffee", "friendly staff", "no ramp", ...]` (lines 444–448).
3. **State-Gini call:** `compute_state_gini(gentags, tag_embeddings, anchor_embeddings)` (lines 455–456). Default `SEMANTIC_THRESHOLD = 0.35` is used.
4. **Inside `compute_state_gini` (lines 169–193):** `compute_facet_counts(...)` builds `counts` (10 facets) and `other_count`. For each tag, `hard_assign_facet(...)` returns (facet or None, sim); if facet is not None, that facet's count is incremented; else `other_count` is incremented. Then `count_array = [counts[f] for f in FACETS]` → 10 integers; `state_gini = gini_coefficient(count_array)`.
5. **Output row (lines 459–474):** `gentag_state_gini`, `gentag_other_count`, `gentag_assigned_count`, and `gentag_count_{facet}` for each facet, written to `state_localization.csv`.

**Baselines (phase3a_baselines.py):** For each venue, review text → RAKE/TF-IDF/YAKE keyword lists → embed keywords → same `compute_state_gini(kw_embs, anchor_embeddings)` → one State-Gini per (venue, method). Same τ and same facets throughout.
