# Phase 4 --- Behavioral Utility Proof (CheckList DIR/INV)

**Date:** 2026-02-17
**Status:** Planning (not yet run)
**Depends on:** Phase 2 (stability), Phase 3 (State-Gini + follow-ups)

---

## 0. Why Phase 4 Exists

### What Phases 2 and 3 Established

**Phase 2 --- Stability (the prerequisite for state):**
Gentags are **lexically unstable but semantically stable**. Across repeated runs, they achieve a median cosine similarity of 0.977 despite a Jaccard similarity of only 0.471. Different runs produce different paraphrases, but the underlying meaning remains identical. This is the prerequisite for any state object: if the representation isn't repeatable, it cannot serve as state.

**Phase 3 --- Semantic Synthesis (the Coverage-Concentration Tradeoff):**
Gentags have a lower State-Gini (0.600) than baselines like TF-IDF (0.715) or RAKE (0.701). But simply calling this "factorized" is insufficient --- what matters is *why* the numbers differ:

- **The Baseline Artifact:** TF-IDF and RAKE appear "more localized" (higher Gini) only because they have a catastrophic Other Rate of ~67%. They are "spiky" because they only assign ~6 keywords per venue. Distributing 6 items across 10 facets forces them into 1--2 slots, artificially inflating the Gini score. This is a **sample-size artifact**, not a structural advantage.
- **The Gentag Win:** Gentags achieve **Semantic Synthesis** --- they have a much lower Other Rate (~43%) and assign **twice the semantic mass** (~12.3 units vs ~6.1). Because they capture more of the review's intent, they spread across more facets, providing a balanced, interpretable, multi-facet structure rather than a spiky, incomplete snapshot.
- **The Bleed Defense:** Unlike raw embeddings, which "entangle" meaning and cause embedding bleed (where one concept erroneously influences all facets), gentags use hard assignment to ensure mass is localized into discrete, addressable slots.

**What remains unproven:** Stability and synthesis are necessary but not sufficient. We have not yet shown that gentags **drive better decisions** than baselines or that their factorized structure **enables localized updates** (changing one facet without bleeding into others).

### What Phase 4 Must Prove

Phase 4 must prove gentags are **actionable** --- that a downstream system makes better, more consistent decisions when it reasons over gentag state than when it uses keyword baselines or raw embeddings.

We adopt the **CheckList methodology** (Ribeiro et al., 2020): behavioral tests that expose whether a representation actually drives correct reasoning, not just benchmark accuracy. Like CheckList showed that high benchmark accuracy often hides "actionable bugs," we must show that baseline representations hide **semantic blindness** that gentags resolve.

**The claim we are testing:** Gentags are the "missing layer" between raw text and decision-making --- semantic enough to generalize, lexical enough to attribute.

**The mentor's hard question this must answer:** If a raw embedding is "verbose and opaque," how specifically do Phase 4 tests prove that a Judge LLM is more consistent when reasoning over 12 synthesized gentags than over a 3,072-dimensional vector? We must show that the factorized nature of gentags (the 0.60 Gini across 10 facets) allows for **localized updates** --- changing one facet of the state doesn't bleed into others. DIR tests prove this directly: a surgical tag edit produces a predictable, monotonic change in the Judge's output, which is impossible with entangled embeddings.

**Strategic focus:** Execute DIR/INV tests on **Sparse Venues** (S4 analysis) where the evidence is weakly constrained and the Synthesis advantage is most visible.

---

## 0.2 Research Integrity Constraint (No Post-Hoc Tuning)

Phase 4 must follow a strict separation between **evaluation** and **refinement** to avoid
p-hacking in representation form.

### The Rule

> If you modify gentags after seeing DIR/INV results, you must treat the new version as a
> new method, re-run everything from scratch, and report both versions side-by-side.

### Phase 4A --- Baseline Evaluation (As-Is)

Freeze:
- Current **minimal** extraction (as produced by Phase 1)
- Intervention catalog
- Judge protocol

Run the experiments. Report exactly what happens. No tweaking. Even if it's mediocre.

This becomes the **baseline controllability profile**.

### Phase 4B --- Representation Refinement (Separate Experiment, Only If Needed)

If Phase 4A reveals failures (too many BORDERLINE, weak directionality, high placebo movement,
low cross-model invariance), then introduce refinement as a **new method**:
- "Conflict-normalized gentags"
- "Specificity-preferred gentags"
- "CORE/DETAIL gentags"

Explicitly state: *"We test whether adding structured normalization improves behavioral
controllability."* Compare to Phase 4A results side-by-side.

### What This Determines About the Contribution

- If **minimal gentags already work** -> paper claim is simpler and stronger:
  *"Gentags as a concept enable controllability."*
- If **only normalized gentags work** -> normalization becomes part of the contribution:
  *"Thoughtfully normalized gentags enable controllability."*

Both are publishable. But they must not be blurred.

### What Strong Papers Do

1. Introduce simple version
2. Show limitations
3. Introduce structured refinement
4. Compare
5. Analyze improvement

### Strategic Order

1. **Run Phase 4A on current minimal gentags first. Do not redesign.**
2. Let the data show where the failure is.
3. If results are already strong, stop. No refinement needed.
4. If results are weak, Phase 4B becomes a response to evidence, not a guess.

---

## 1. Experiment A --- Directional Expectation (DIR) Tests

### 1.1 Purpose

Prove **causal attribution**: surgically perturbing the gentag state with a known intervention produces a predictable, monotonic change in a Judge LLM's output.

### 1.2 Protocol

1. **Baseline decision:** Present a Judge LLM with:
   - A **User Persona** (e.g., "Disabled Traveler requiring wheelchair accessibility")
   - A venue's **gentag state** (from Phase 1 extractions)
   - Judge returns: `{"score": 0-100, "justification": "...", "tags_used": [...]}`

2. **Intervention:** Surgically **add or remove** a causal gentag from the state.
   - Example: For sparse venue `KzvuSntI35Z638fGoOJ4`, add the tag `"no wheelchair ramp"`.
   - The intervention must target a tag with **clear primary facet** (bleed gap >= 0.10, from Phase 3 bleed check).

3. **Expected behavior:** The Judge's score must move **monotonically** in the expected direction.
   - Adding `"no wheelchair ramp"` for a Disabled Traveler persona -> score **decreases**.
   - Removing a negative tag -> score **increases**.

4. **Baseline comparison:** Run the same protocol with RAKE, TF-IDF, and YAKE keywords instead of gentags. Because baselines have ~67% Other rate (vs gentags ~43%), pivotal information is more likely to be missing from their representation, causing the Judge to **fail** the DIR test.

### 1.3 Metrics

| Metric | Definition |
|--------|------------|
| **DIR pass rate** | % of interventions where score moves in expected direction |
| **DIR delta (mean)** | Mean score change magnitude for passing cases |
| **DIR delta (median)** | Median score change magnitude (robust to Judge noise) |
| **Attribution precision** | % of passing cases where the intervened tag appears in `tags_used` |
| **Baseline failure rate** | % of cases where baselines fail DIR that gentags pass |
| **Coverage rate (baselines)** | % of DIR cases where the baseline state contains the pivotal concept |
| **Placebo DIR false-positive rate** | % of placebo interventions (irrelevant tag edits) that still move directionally |

### 1.4 What "Crushing the Baselines" Looks Like

RAKE/TF-IDF/YAKE are "semantically blind" to ~67% of the evidence. The DIR test exposes this:
- If the pivotal tag (e.g., "no wheelchair ramp") was **never extracted** by the baseline method, the Judge cannot respond to the intervention.
- Gentags, with ~57% facet coverage, are far more likely to contain the pivotal information.
- We report: for each DIR case, did the baseline method even **contain** the pivotal concept? If not, DIR failure is attributed to representation blindness.

---

## 2. Experiment B --- Invariance (INV) Tests

### 2.1 Purpose

Prove that gentags maintain a **stable semantic state** that leads to **identical decisions** despite lexical variation across runs (Jaccard 0.471 from Phase 2).

### 2.2 Protocol

1. **Setup:** Take the same venue, present the Judge LLM with:
   - **Set A:** Gentags from Run 1 (model X, prompt Y)
   - **Set B:** Gentags from Run 2 (same model X, same prompt Y, different run)
   - Same User Persona for both.

2. **Expected behavior:** The Judge's recommendation must be **label-preserving** (same tier or within epsilon of the same score) despite surface-level lexical differences.

3. **Epsilon threshold:** |score_A - score_B| <= epsilon. Epsilon TBD (candidate: 10 points on a 0-100 scale, or same recommendation tier if we discretize into tiers).

### 2.3 Metrics

| Metric | Definition |
|--------|------------|
| **INV pass rate** | % of venue pairs where |score_A - score_B| <= epsilon |
| **INV delta (mean)** | Mean |score_A - score_B| across all pairs |
| **INV delta (median)** | Median |score_A - score_B| across all pairs |
| **Label agreement** | % where both scores map to same recommendation tier |
| **Correlation** | Pearson r between score_A and score_B across venues |

### 2.4 Why This Matters

Phase 2 showed cosine 0.977 but Jaccard 0.471. INV proves this is not just a number --- it means a downstream system **doesn't care** about the lexical variation. The state is semantically robust. If the Judge's decision is invariant across Run 1 and Run 2, we have empirical proof that gentags function as stable state for decision-making.

---

## 3. Pre-Execution Constraints

These constraints prevent the "magic" or "shortcuts" that sink NLP papers at review.

### 3.1 Prioritize Sparse Venues (S4)

Focus interventions on venues with **low token counts** (< 200 tokens from Phase 2 S4 analysis). Examples:
- `GVn2q90PoVQ5p6EcJb4W` (sparse, weakly constrained)
- `KzvuSntI35Z638fGoOJ4` (sparse)

**Rationale:** In sparse settings, a single tag like `"fast service"` carries immense semantic weight, making DIR impact easier to identify and attribute. This also tests the cold-start claim.

### 3.2 Use "Clear Primary" Tags Only for DIR Interventions

Only use gentags that Phase 3 bleed check identified as having **clear primary facet** (gap >= 0.10, which is ~20.5% of tags).

**Rationale:** Tags with gap < 0.05 (57.4% of tags) sit between two facets in embedding space. Using them for DIR interventions introduces noise and "embedding bleed." We want clean causal signal.

### 3.3 Evidence-Conditioning (Anti-Hallucination)

The Judge LLM must be **strictly prompted** to use only the provided gentags/keywords. If the Judge "hallucinates" outside the provided state, the test is invalid.

- System prompt must include: `"Use ONLY the provided tags. Do NOT use external knowledge."`
- Validation: check `tags_used` in output --- all listed tags must be from the input set.
- If hallucination is detected, flag the case and exclude from pass-rate computation (report hallucination rate separately).

### 3.4 Reproducibility Controls

- **Judge model:** Fix one model (e.g., GPT-4o or Claude) and report the exact model version.
- **No Judge tuning:** Do not tune or optimize the Judge's decoding/sampling parameters. Use the
  provider defaults (or, if the API requires explicit values, keep them constant and document them).
- **Repetitions:** Repeat each DIR/INV test N >= 3 times to quantify stochasticity and report
  yield (pass rate) under inherent Judge variability.

### 3.5 Aggregation Rule

Per the execution spec (PHASE4_EXECUTION_SPEC.md): N=5 repetitions per condition, majority
vote, tie -> BORDERLINE, < 3 valid -> UNSCORABLE.

---

## 4. Capability Matrix (CheckList-Style)

Following CheckList (Ribeiro et al., 2020), map our 10 facets to specific linguistic capabilities.

### 4.1 Facet-to-Capability Mapping (To Define)

| Facet | Capability to Test | Example DIR Intervention |
|-------|--------------------|--------------------------|
| Service | Negation | Add `"not attentive staff"` --- score should decrease for service-sensitive persona |
| Food Quality | Taxonomy | Add `"stale bread"` (hyponym of poor food) --- score should decrease |
| Price/Value | Comparative | Change `"affordable"` to `"overpriced"` --- score should flip for budget persona |
| Accessibility | Presence/Absence | Add `"no wheelchair ramp"` --- score should decrease for disabled traveler |
| Ambiance | Sentiment flip | Change `"cozy atmosphere"` to `"noisy atmosphere"` --- directional change |
| Cleanliness | Negated negative | `"The food is not poor"` --- Judge must still recommend (harder test) |
| ... | ... | ... |

**Action needed:** Complete this matrix for all 10 facets before execution. Each facet needs at least 2 DIR test cases.

### 4.2 The "Negated Negative" Hard Test

Can the Judge handle `"The food is not poor"` and still recommend? This tests whether the representation + Judge system handles compositional semantics. Gentags should encode the **resolved** meaning (e.g., `"acceptable food"` or `"food not poor"`), while keyword baselines might extract `"poor"` and fail.

---

## 5. Handling the "Other" Bucket in DIR Tests

~43% of gentags fall in the "Other" bucket (below tau = 0.35 for all 10 facets). A pivotal tag for a DIR test might land in Other.

### 5.1 Strategy

- **Report separately:** For each DIR test, flag whether the pivotal tag is in-facet or in-Other.
- **Argue from strength:** Even when a pivotal tag is in Other, if the Judge correctly adjusts its score, that proves gentags capture meaning **beyond** the 10 diagnostic facets.
- **Compare to baselines:** Baselines have ~67% Other rate. If a pivotal concept falls in Other for baselines but in-facet for gentags, that's direct evidence of the coverage advantage.

### 5.2 Other-Rate as a Feature, Not a Bug

From Phase 3 Other-bucket probe: the 20,699 unique tags in Other are long-tail semantic propositions (e.g., "dim lighting", domain nuance), not noise. The Other bucket is evidence of **granularity** --- gentags capture what keywords miss.

---

## 6. The Embedding Comparison (Why Embeddings Fail as State)

We do NOT compare gentags to raw dense embeddings on State-Gini (that's "measuring the squareness of a circle"). Instead, we use DIR/INV to show embeddings fail the **state object** test.

### 6.1 Why Embeddings Are Not State

- **Entanglement:** A vector for "great coffee" has non-zero similarity to "service" and "ambiance". Every concept influences every facet. This is not attribution; it is a semantic smear.
- **Opacity:** You cannot inspect a 3,072-dim vector and "read" that it contains a complaint about a "dirty restroom".
- **Non-addressable:** You cannot surgically add or remove a concept from a dense embedding the way you can add/remove a gentag.

### 6.2 DIR with Embeddings (Expected Failure)

- To DIR-test embeddings, we would need to "add a concept" to a vector. The only way is to add a second embedding and average or concatenate --- but this changes the entire vector, not a single attribute.
- We predict embeddings will fail DIR because interventions are not **localizable** in the representation.
- This is the strongest argument: gentags are addressable state; embeddings are not.

---

## 7. Prompt Templates and Execution Protocol

**The exact Judge prompt, execution loop, aggregation rules, pass rules, and output metrics are defined in the strict execution spec:**

-> **`docs/phase4/PHASE4_EXECUTION_SPEC.md`**


---

## 8. Required Inputs

| Input | Source | Status |
|-------|--------|--------|
| Phase 1 extractions (gentags per venue per model per run) | `results/phase1_downloaded/` | Available |
| Phase 2 tag embeddings | `results/phase2_cache/` | Available |
| Phase 3 bleed check (clear primary tags, gap >= 0.10) | `results/phase3/bleed_check_summary.json` | Available |
| Phase 3 Other-bucket tags | `results/phase3/other_bucket_tags.csv` | Available |
| Venue data (including sparse venues) | `data/study1_venues_20250117.csv` | Available |
| Baseline keywords (RAKE/TF-IDF/YAKE per venue) | Phase 3A pipeline | Available (can regenerate) |
| Sample venue data (gentags + baselines, JSON) | `scripts/phase4_sample_venue.py` -> `results/phase4/sample_venue.json` | Script ready, run to generate |
| User Personas | **To create** | Not yet defined |
| Capability Matrix (facet -> test cases) | **To create** | Not yet defined |
| Judge LLM selection | **To decide** | Not yet decided |
| Intervention catalog (tag + direction + persona) | **To create** | Not yet defined |
| Epsilon for INV | **To decide** | Not yet decided |

---

## 9. Decisions Needed Before Running

| # | Decision | Options / Notes |
|---|----------|-----------------|
| 1 | **Judge LLM** | GPT-4o, Claude, or both? Single model preferred for consistency. Must document exact model version. |
| 2 | **User Personas** | How many? Minimum 3-5 covering different needs (accessibility, budget, food quality, ambiance, family). Must be diverse enough to exercise different facets. |
| 3 | **Venue sample** | How many venues per test? Prioritize sparse (S4) but include some dense for contrast. Candidate: 20-30 sparse + 10-15 dense = ~40 venues. |
| 4 | **Intervention catalog** | How many DIR cases per facet? Minimum 2 per facet x 10 facets = 20 interventions. Each paired with a persona. |
| 5 | **INV epsilon** | Score tolerance for "same decision". Candidates: |delta| <= 10 (on 0-100), or tier-based (discretize into 5 tiers of 20 points). |
| 6 | **INV pairs** | Which runs to compare? Run 1 vs Run 2, same model+prompt. How many venues x how many model-prompt combos? |
| 7 | **Repetitions (N)** | Frozen at N=5 per execution spec. Confirm or increase before running. |
| 8 | **Negated negative tests** | Include compositional semantics tests (e.g., "not poor")? How many? |
| 9 | **Embedding DIR protocol** | How to "intervene" on an embedding? Average with intervention embedding? Needed to show embeddings fail. |
| 10 | **Output schema validation** | How strict? Reject non-JSON? Reject hallucinated tags? Report rates separately? |
| 11 | **S4 stratification** | Report separate pass rates for sparse vs dense (token buckets), and which buckets define "Sparse". |

---

## 10. Execution Order

```
Step 0: Freeze decisions (all items in section 9)
  |
Step 0.5: Generate sample venue data
  |     -> poetry run python scripts/phase4_sample_venue.py [--venue-id X]
  |     -> results/phase4/sample_venue.json (gentags run1+run2, RAKE/TF-IDF/YAKE)
  |     -> Use --list-candidates to pick sparse venues
  |
Step 1: Create User Personas
  |     -> docs/phase4/user_personas.md or data/phase4/personas.csv
  |
Step 2: Build Intervention Catalog
  |     -> Map facets to DIR cases, select venues, select tags (clear primary only)
  |     -> data/phase4/intervention_catalog.csv
  |
Step 3: Build Capability Matrix
  |     -> Complete section 4.1 table for all 10 facets
  |     -> docs/phase4/capability_matrix.md
  |
Step 4: Implement Judge pipeline
  |     -> scripts/phase4_judge.py (baseline scoring)
  |     -> scripts/phase4_dir.py (DIR tests)
  |     -> scripts/phase4_inv.py (INV tests)
  |
Step 5: Preflight (small sample)
  |     -> Run 3-5 DIR cases, 3-5 INV cases
  |     -> Validate Judge output format, check for hallucination
  |     -> Calibrate epsilon if needed
  |     -> docs/phase4/preflight_runs.md
  |
Step 6: Full DIR run
  |     -> All interventions x all personas x N repetitions
  |     -> results/phase4/tables/dir_results.csv
  |
Step 7: Full INV run
  |     -> All venue pairs x all personas x N repetitions
  |     -> results/phase4/tables/inv_results.csv
  |
Step 8: Baseline comparison runs
  |     -> Same DIR/INV with RAKE/TF-IDF/YAKE keywords
  |     -> results/phase4/tables/dir_baselines.csv, inv_baselines.csv
  |
Step 9: (Optional) Embedding DIR
  |     -> Attempt DIR on dense embeddings to demonstrate failure
  |     -> results/phase4/tables/dir_embeddings.csv
  |
Step 10: Analysis and write-up
        -> Summary tables, pass rates, narrative
        -> docs/phase4/dir_inv_analysis.md
```

---

## 11. Success / Failure Criteria

Pass/fail per the execution spec (PHASE4_EXECUTION_SPEC.md). All rates reported with
**binomial 95% CI**.

### DIR Tests

| Outcome | Interpretation |
|---------|---------------|
| Gentag DIR pass rate > 80% | Gentags are causally actionable |
| Baseline DIR pass rate < 50% | Baselines are semantically blind to interventions |
| Gap (gentag - baseline) > 30pp | Strong evidence for "missing layer" |
| Placebo movement rate < 15% | Interventions are causal, not noise |
| Attribution precision > 70% | Judge correctly identifies the intervened tag |

### INV Tests

| Outcome | Interpretation |
|---------|---------------|
| INV pass rate > 85% | Semantic stability translates to decision stability |
| INV pass rate < 70% | Lexical variation leaks into decisions (problem) |

### If Tests Fail

- If DIR pass rate is low for gentags: investigate whether the Judge is hallucinating or ignoring tags. Check anti-hallucination prompt effectiveness.
- If INV pass rate is low: lexical variation may matter more than Phase 2 cosine suggests. Consider whether the Judge is surface-sensitive.
- If baselines pass DIR at similar rates to gentags: our coverage advantage doesn't translate to downstream utility. Re-examine whether the specific interventions test coverage-dependent reasoning.

---

## 12. Paper Narrative (How This Closes the Argument)

**Phase 2:** Gentags are stable (semantic) but varied (lexical).
**Phase 3:** Gentags are factorized (multi-facet coverage, lower Other rate than baselines).
**Phase 4:** Gentags are actionable (Judge succeeds with gentag state, fails with baseline state).

The paper's final argument:
> Gentags externalize LLM judgments into a **persistent, inspectable, actionable semantic state**. Unlike keywords, they capture sufficient semantic mass to drive correct decisions. Unlike embeddings, they are addressable --- you can add, remove, or inspect individual propositions. This makes them the **missing layer** that RAG systems need to track knowledge over time.
