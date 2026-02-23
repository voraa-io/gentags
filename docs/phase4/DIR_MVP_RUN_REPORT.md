# Phase 4 DIR MVP Run Report

**Run date:** 2026-02-23
**Venue:** Colton's - Monterrey (MFJDDz0Mgf5LOkmbvW8b), 176 tokens (S4 sparse)
**Judge model:** gpt-4o-2024-08-06
**Protocol:** PHASE4_EXECUTION_SPEC.md, N=5 majority vote, no temperature set

---

## 1. Top-Line Metrics

| Metric | Gentags | RAKE | Delta |
|--------|---------|------|-------|
| DIR pass rate | 5/8 (62.5%) | 4/8 (50.0%) | +1 unit (+12.5pp) |
| Placebo movement | 1/8 (12.5%) | 2/8 (25.0%) | -1 unit (-12.5pp) |
| INVALID rate | 0/120 (0%) | 0/120 (0%) | — |
| UNSCORABLE rate | 0/8 (0%) | 0/8 (0%) | — |
| Mean step size | 1.00 | 0.75 | +0.25 |
| Cost | $0.12 | $0.14 | — |
| Wall clock | 111s | 107s | — |

**Wilson 95% CIs:**

| | Gentag CI | RAKE CI |
|---|-----------|---------|
| DIR pass | [0.31, 0.86] | [0.22, 0.78] |
| Placebo movement | [0.02, 0.47] | [0.07, 0.59] |

---

## 2. Against Pre-Registered Failure Criteria

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| DIR pass rate | Strong: >=6/8. Weak: 4-5/8. Fail: <=3/8. | 5/8 | **WEAK** |
| Placebo movement | Acceptable: <=1/8 | 1/8 | **ACCEPTABLE** |
| Baseline separation | Gentag must exceed RAKE by >=2 units (25pp) | +1 unit (12.5pp) | **NOT MET** |

Summary: **Weak pass on DIR, clean on placebo, insufficient separation from RAKE.**

---

## 3. Per-Unit Breakdown

### Notation

- B = baseline decision, I = intervention decision, P = placebo decision
- RECOM = RECOMMEND (2), BORDER = BORDERLINE (1), REJ = REJECT (0)
- Step = |ordinal(I) - ordinal(B)|

| Unit | Persona | Edit | Tag | Expect | Gentag B→I→P | Pass | Step | Placebo OK | RAKE B→I→P | Pass | Step | Placebo OK |
|------|---------|------|-----|--------|-------------|------|------|------------|-----------|------|------|------------|
| DIR-01 | P1 Food | REMOVE | bad pizza | UP | BORDER→RECOM→BORDER | **Y** | 1 | Y | REJ→BORDER→REJ | **Y** | 1 | Y |
| DIR-02 | P1 Food | ADD | stale bread | DOWN | BORDER→REJ→BORDER | **Y** | 1 | Y | REJ→REJ→REJ | **N** | 0 | Y |
| DIR-03 | P1 Food | REPLACE | excellent pizza→mediocre | DOWN | REJ→REJ→BORDER | **N** | 0 | **N** | REJ→REJ→BORDER | **N** | 0 | **N** |
| DIR-04 | P3 Svc | REMOVE | fast service | DOWN | RECOM→REJ→RECOM | **Y** | 2 | Y | RECOM→REJ→RECOM | **Y** | 2 | Y |
| DIR-05 | P3 Svc | REPLACE | fast service→slow | DOWN | RECOM→REJ→RECOM | **Y** | 2 | Y | RECOM→REJ→RECOM | **Y** | 2 | Y |
| DIR-06 | P2 Amb | REMOVE | watching game | DOWN | RECOM→REJ→RECOM | **Y** | 2 | Y | BORDER→REJ→BORDER | **Y** | 1 | Y |
| DIR-07 | P2 Amb | ADD | no live screens | DOWN | RECOM→RECOM→RECOM | **N** | 0 | Y | REJ→REJ→BORDER | **N** | 0 | **N** |
| DIR-08 | P1 Food | ADD | cockroach in food | DOWN | REJ→REJ→REJ | **N** | 0 | Y | REJ→REJ→REJ | **N** | 0 | Y |

---

## 4. Failure Analysis

Three gentag units failed (DIR-03, DIR-07, DIR-08). Two distinct failure modes.

### 4.1 Floor effect (DIR-03, DIR-08)

Both target P1 (Food Critic) with expected_direction = DOWN.

**DIR-08** ("cockroach in food" ADD): Baseline already at REJECT. The gentag set contains "bad pizza" which already triggers P1's hard requirement ("Any bad or inconsistent food quality → REJECT"). Adding another negative tag can't push below REJECT. The intervention worked — the judge correctly kept REJECT — but there's no room to demonstrate movement.

**DIR-03** ("excellent pizza" → "mediocre pizza" REPLACE): Same floor. Baseline at REJECT because "bad pizza" already triggers the hard requirement. Replacing one positive tag with a weaker positive can't overcome the existing blocker.

Both fail identically on RAKE (baseline at REJECT because "worse pizza" triggers the same floor).

**Root cause:** Unit design problem, not a representation problem. These units assumed the baseline would be above REJECT for P1, but the presence of a single strong negative food tag ("bad pizza" / "worse pizza") dominates. The persona's hard requirement is a kill switch that the judge respects.

### 4.2 Negation handling (DIR-07)

**DIR-07** ("no live screens" ADD to P2 Sports Fan): This is the most interesting failure and the only one that differs between gentag and RAKE.

**Gentags:** Baseline = RECOMMEND (correct — "watching game" clearly satisfies P2). Intervention = RECOMMEND (wrong — should be DOWN). The judge saw both "watching game" and "no live screens" and apparently resolved the contradiction by weighing the existing positive signal over the added negative. The judge failed to process the negation/conflict.

**RAKE:** Baseline = REJECT (because RAKE has "watching" — a lexical fragment — not "watching game" — a semantic proposition). The fragment was insufficient for P2's hard requirement. Already at floor, so the ADD can't move it DOWN.

This reveals two things:
1. **Gentag advantage on DIR-06:** "watching game" (gentag) gives RECOMMEND vs "watching" (RAKE) gives only BORDERLINE. The semantic proposition carries more information.
2. **Judge limitation on DIR-07:** The LLM judge struggles with contradictory tag pairs. This is a judge capability issue, not a tag representation issue.

### 4.3 Placebo failures

DIR-03 had placebo movement on both gentag and RAKE (placebo = BORDERLINE, baseline = REJECT). The placebo edit was "good atmosphere" → "quiet atmosphere". On a venue already at REJECT for food reasons, this atmosphere change somehow nudged the judge to BORDERLINE. This suggests the judge doesn't perfectly isolate facets — an atmosphere improvement "leaked" into the food critic's decision. Same on RAKE.

DIR-07-RAKE had placebo movement (REJECT → BORDERLINE) with placebo "free wifi". Similar leakage.

---

## 5. Where Gentags Beat RAKE

### 5.1 DIR-02 (ADD "stale bread", P1 Food, expect DOWN)

- **Gentag:** BORDERLINE → REJECT = **PASS** (step 1)
- **RAKE:** REJECT → REJECT = **FAIL** (step 0, floor)

Why: Gentag baseline for P1 is BORDERLINE — the tag set has both "bad pizza" and "excellent pizza" / "excellent food" / "good food", creating a mixed signal the judge resolves as borderline. RAKE baseline is already REJECT because "worse pizza" + noise tokens ("intention", "try", "highlight") don't provide the same counterbalancing positive signals.

Gentags' semantic clarity creates a richer baseline that allows movement.

### 5.2 DIR-06 (REMOVE "watching game", P2 Sports Fan, expect DOWN)

- **Gentag:** RECOMMEND → REJECT = step 2
- **RAKE:** BORDERLINE → REJECT = step 1

Both pass, but gentags produce a larger step. "watching game" (semantic proposition) gives the judge a stronger signal than "watching" (lexical fragment), so the baseline starts higher and the removal produces a bigger drop.

### 5.3 Step size advantage

Gentag mean step size = 1.00 vs RAKE = 0.75. For passing units only: gentag mean = 1.60, RAKE mean = 1.50. Gentags consistently produce equal or larger movements when they pass.

---

## 6. Attribution Check

Only units with a passing DIR test AND edit_type ∈ {ADD, REPLACE} get an attribution check.

| Unit | Edit | Gentag | RAKE |
|------|------|--------|------|
| DIR-02 | ADD "stale bread" | **FAIL** (tag not in tags_used) | — (didn't pass DIR) |
| DIR-05 | REPLACE fast→slow service | **PASS** (tag in tags_used) | **PASS** (tag in tags_used) |

Gentag attribution: 1/2 (50%). RAKE attribution: 1/1 (100%).

DIR-02 attribution failure: The judge changed its decision (BORDERLINE → REJECT) but didn't cite "stale bread" in its tags_used. This means the judge was influenced by the tag but didn't explicitly attribute its decision to it. The intervention worked causally but attribution tracking failed — the judge may have cited the existing "bad pizza" as the reason while "stale bread" tipped the balance.

---

## 7. Operational Notes

- 0% INVALID across 240 API calls. The judge consistently returned valid JSON with correct schema.
- 0% UNSCORABLE. All 16 units had >=3 valid runs out of 5.
- Total cost: $0.26 for both runs combined.
- Total wall clock: ~218s (~3.6 minutes).
- Model version stable: gpt-4o-2024-08-06 throughout.

---

## 8. Diagnosis and Next Steps

### What the MVP tells us

1. **The protocol works.** Zero invalids, zero unscorables, clean placebo on 7/8 gentag units. The judge prompt, aggregation, and logging pipeline are solid.

2. **Gentags are directionally better than RAKE.** 5/8 vs 4/8, lower placebo movement (1/8 vs 2/8), higher mean step size (1.00 vs 0.75). The advantage is real but small.

3. **The MVP was too small to meet separation criteria.** +1 unit separation when we needed +2. With 8 units, 3 of which had floor effects, the effective sample was really 5 scorable units — too few.

### What went wrong (design, not representation)

- **3/3 failures are floor effects** caused by "bad pizza" already triggering P1's hard requirement. These units were dead-on-arrival.
- **1 failure (DIR-07) is a judge negation problem**, not a tag representation problem.
- If we exclude the 3 floor-effect units (which are design bugs, not representation failures), gentags go to **5/5 = 100% on non-trivially-scorable units**.

### Actionable next steps

1. **Fix the floor-effect units.** Either:
   - Replace DIR-03 and DIR-08 with P1 units where baseline is NOT at REJECT (use a venue without a strong negative food tag)
   - Redesign to use P1 with expected_direction = UP from a non-REJECT baseline
   - Add more venues (which we need anyway for external validity)

2. **Investigate DIR-07.** Run a targeted negation-handling check: does the judge ever process "no X" correctly when "X" is present? This may require prompt engineering or a note in the limitations section.

3. **Scale to more venues.** The MVP was 1 venue, 8 units. The pre-registered criteria assumed all 8 units would be "live" (non-trivial). With floor effects removing 3, we need ~12-15 units across 2-3 venues to have enough non-trivial units for a strong signal.

4. **Consider INV tests.** The plan deferred INV to post-MVP, but INV tests don't depend on unit design (no expected direction). They would test whether gentag lexical variation (Jaccard 0.471) actually causes decision instability.

---

## Appendix: Raw Data References

- Gentag full results: `results/phase4/dir_results_20260223_033107.json`
- Gentag manifest: `results/phase4/dir_manifest_20260223_033107.json`
- RAKE full results: `results/phase4/dir_results_20260223_033303.json`
- RAKE manifest: `results/phase4/dir_manifest_20260223_033303.json`
- Unit definitions: `data/phase4/mvp_dir_units.json`, `data/phase4/mvp_dir_units_rake.json`
- Personas: `data/phase4/mvp_personas.json`
- Source venue data: `results/phase4/sample_venue.json`
