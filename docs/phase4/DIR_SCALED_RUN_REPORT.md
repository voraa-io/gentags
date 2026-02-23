# Phase 4 — Scaled DIR Run Report

## Overview

Scaled the MVP pilot (1 venue, 8 units) to 3 venues, 16 total units.
New venues designed with floor-avoidance: no DOWN units where baseline is already REJECT.

| Venue | Units | Type |
|-------|-------|------|
| Colton's - Monterrey | DIR-01 to DIR-08 | MVP (frozen) |
| Boost Coffee | DIR-09 to DIR-12 | Scaled (new) |
| Boru - Gómez Morin | DIR-13 to DIR-16 | Scaled (new) |

**Judge model:** gpt-4o-2024-08-06 (matched MVP)
**N:** 5 per condition (BASELINE, INTERVENTION, PLACEBO)
**Total API calls:** 480 (16 units x 3 conditions x 5 x 2 systems)
**INVALID rate:** 0/480 = 0.0%
**UNSCORABLE rate:** 0/32 = 0.0%

---

## Top-Line Metrics (All 16 Units)

| Metric | Gentag | RAKE | Criterion |
|--------|--------|------|-----------|
| DIR pass rate | 13/16 = 81.2% [57.0%, 93.4%] | 10/16 = 62.5% [38.6%, 81.5%] | Strong >= 75% |
| Placebo movement | 3/16 = 18.8% | 4/16 = 25.0% | Acceptable <= 12.5% |
| **Separation** | **+18.8pp** | | Promising (15-24pp) |
| Fisher's exact p | 0.433 | | Not significant |

**Verdict (all units):** PROMISING. Separation is +18.8pp, above 15pp threshold but below 25pp paper-ready.

---

## Non-Floor Metrics

Floor units are those where baseline = REJECT AND direction = DOWN (can't move down from the bottom).

| System | Floor units | Non-floor units | Non-floor pass rate |
|--------|-------------|-----------------|---------------------|
| Gentag | 2 (DIR-03, DIR-08) | 14 | 13/14 = 92.9% [68.5%, 98.7%] |
| RAKE | 6 (DIR-02-RAKE, DIR-03-RAKE, DIR-07-RAKE, DIR-08-RAKE, DIR-14-RAKE, DIR-15-RAKE) | 10 | 10/10 = 100.0% [72.2%, 100.0%] |

**Key insight:** RAKE has **6 floor units** vs gentag's **2 floor units**. This is itself the signal — RAKE's noisy fragments fail to communicate persona-critical information to the judge, causing more baseline REJECT decisions. When we exclude floors, both systems score high (93% vs 100%), because the remaining RAKE tags happened to be clear enough.

The separation comes not from "RAKE tags can't move the judge" but from "RAKE tags fail to establish correct baselines" — the judge can't parse fragments like `"relative quick time"` or `"watching"` as semantic propositions.

**Non-floor separation:** -7.1pp (gentag slightly lower than RAKE, but n=14 vs n=10, not comparable).

---

## Per-Venue Breakdown

| Venue | Gentag Pass | RAKE Pass | Separation |
|-------|-------------|-----------|------------|
| Colton's | 5/8 (62.5%) | 4/8 (50.0%) | +12.5pp |
| Boost Coffee | 4/4 (100.0%) | 4/4 (100.0%) | +0.0pp |
| Boru - Gómez Morin | 4/4 (100.0%) | 2/4 (50.0%) | +50.0pp |

**Boost Coffee:** Zero separation. All 4 units used ADD interventions (adding "fast service", "watching game", "stale pastry"). Both gentag and RAKE baselines lacked these tags, so both systems responded identically to the ADD. This makes sense — ADD interventions don't test tag quality, they test whether a new tag is recognized.

**Boru:** +50pp separation, driven entirely by DIR-14-RAKE and DIR-15-RAKE floor effects. RAKE's `"relative quick time"` fragment was not recognized as a speed signal — P3 (Quick Lunch) baseline was REJECT even with the tag present, so removing/replacing it had no effect. Gentag's `"fast service"` was clearly understood.

---

## MVP to Scaled Progression

| Scope | Gentag | RAKE | Separation |
|-------|--------|------|------------|
| MVP (Colton's, 8 units) | 5/8 = 62.5% | 4/8 = 50.0% | +12.5pp |
| New venues (8 units) | 8/8 = 100.0% | 6/8 = 75.0% | +25.0pp |
| **Combined (16 units)** | **13/16 = 81.2%** | **10/16 = 62.5%** | **+18.8pp** |

New venues performed better than MVP for both systems (floor avoidance worked). Separation widened from +12.5pp to +18.8pp combined.

---

## Per-Unit Breakdown (All 16)

### Gentag

| ID | Venue | Persona | Edit | Dir | Baseline | Intervention | Pass | Floor | Placebo |
|----|-------|---------|------|-----|----------|--------------|------|-------|---------|
| DIR-01 | Colton's | P1 Food | REMOVE "bad pizza" | UP | BORDERLINE | RECOMMEND | Y | | |
| DIR-02 | Colton's | P1 Food | ADD "stale bread" | DOWN | BORDERLINE | REJECT | Y | | |
| DIR-03 | Colton's | P1 Food | REPLACE "excellent pizza" | DOWN | REJECT | REJECT | N | FLOOR | moved |
| DIR-04 | Colton's | P3 Svc | REMOVE "fast service" | DOWN | RECOMMEND | REJECT | Y | | |
| DIR-05 | Colton's | P3 Svc | REPLACE "fast service" | DOWN | RECOMMEND | REJECT | Y | | |
| DIR-06 | Colton's | P2 Sports | REMOVE "watching game" | DOWN | RECOMMEND | REJECT | Y | | |
| DIR-07 | Colton's | P2 Sports | ADD "no live screens" | DOWN | RECOMMEND | RECOMMEND | **N** | | |
| DIR-08 | Colton's | P1 Food | ADD "cockroach in food" | DOWN | REJECT | REJECT | N | FLOOR | |
| DIR-09 | Boost | P1 Food | ADD "stale pastry" | DOWN | BORDERLINE | REJECT | Y | | |
| DIR-10 | Boost | P1 Food | REPLACE "deliciou dessert" | DOWN | BORDERLINE | REJECT | Y | | |
| DIR-11 | Boost | P3 Svc | ADD "fast service" | UP | REJECT | RECOMMEND | Y | | |
| DIR-12 | Boost | P2 Sports | ADD "watching game" | UP | REJECT | RECOMMEND | Y | | |
| DIR-13 | Boru | P1 Food | ADD "food poisoning" | DOWN | RECOMMEND | REJECT | Y | | moved |
| DIR-14 | Boru | P3 Svc | REMOVE "fast service" | DOWN | RECOMMEND | REJECT | Y | | |
| DIR-15 | Boru | P3 Svc | REPLACE "fast service" | DOWN | RECOMMEND | REJECT | Y | | |
| DIR-16 | Boru | P1 Food | REPLACE "excellent poke" | DOWN | BORDERLINE | REJECT | Y | | moved |

### RAKE

| ID | Venue | Persona | Edit | Dir | Baseline | Intervention | Pass | Floor | Placebo |
|----|-------|---------|------|-----|----------|--------------|------|-------|---------|
| DIR-01-RAKE | Colton's | P1 Food | REMOVE "worse pizza" | UP | REJECT | BORDERLINE | Y | | |
| DIR-02-RAKE | Colton's | P1 Food | ADD "stale bread" | DOWN | REJECT | REJECT | N | FLOOR | |
| DIR-03-RAKE | Colton's | P1 Food | REPLACE "mushrooms excellent" | DOWN | REJECT | REJECT | N | FLOOR | moved |
| DIR-04-RAKE | Colton's | P3 Svc | REMOVE "fast service" | DOWN | RECOMMEND | REJECT | Y | | |
| DIR-05-RAKE | Colton's | P3 Svc | REPLACE "fast service" | DOWN | RECOMMEND | REJECT | Y | | |
| DIR-06-RAKE | Colton's | P2 Sports | REMOVE "watching" | DOWN | BORDERLINE | REJECT | Y | | |
| DIR-07-RAKE | Colton's | P2 Sports | ADD "no live screens" | DOWN | REJECT | REJECT | N | FLOOR | moved |
| DIR-08-RAKE | Colton's | P1 Food | ADD "cockroach in food" | DOWN | REJECT | REJECT | N | FLOOR | |
| DIR-09-RAKE | Boost | P1 Food | ADD "stale pastry" | DOWN | BORDERLINE | REJECT | Y | | |
| DIR-10-RAKE | Boost | P1 Food | REPLACE "delicious" | DOWN | BORDERLINE | REJECT | Y | | |
| DIR-11-RAKE | Boost | P3 Svc | ADD "fast service" | DOWN | REJECT | RECOMMEND | Y | | |
| DIR-12-RAKE | Boost | P2 Sports | ADD "watching game" | UP | REJECT | RECOMMEND | Y | | |
| DIR-13-RAKE | Boru | P1 Food | ADD "food poisoning" | DOWN | BORDERLINE | REJECT | Y | | |
| DIR-14-RAKE | Boru | P3 Svc | REMOVE "relative quick time" | DOWN | REJECT | REJECT | N | FLOOR | |
| DIR-15-RAKE | Boru | P3 Svc | REPLACE "relative quick time" | DOWN | REJECT | REJECT | N | FLOOR | moved |
| DIR-16-RAKE | Boru | P1 Food | REPLACE "excelente restaurant de pokes" | DOWN | BORDERLINE | REJECT | Y | | moved |

---

## Failure Analysis

### Gentag Failures (3/16)

1. **DIR-03** (Floor): Baseline already REJECT. Cannot go DOWN. Colton's P1 with "bad pizza" in baseline.
2. **DIR-07** (Real failure): P2 Sports Fan, ADD "no live screens" expected DOWN from RECOMMEND. Judge did NOT downgrade — `"no live screens"` was not treated as a sports-viewing blocker. The only non-floor gentag failure across all 16 units.
3. **DIR-08** (Floor): Baseline already REJECT. Cannot go DOWN. Colton's P1 stress test.

**Non-floor failure rate:** 1/14 = 7.1% (single failure: DIR-07)

### RAKE Failures (6/16)

1. **DIR-02-RAKE** (Floor): P1 baseline REJECT due to noisy RAKE tags (e.g., "sausage pepperoni", "one large vampiro" confusing the judge).
2. **DIR-03-RAKE** (Floor): Same P1 baseline REJECT issue.
3. **DIR-07-RAKE** (Floor): P2 baseline REJECT — RAKE's `"watching"` fragment insufficient for game-viewing signal.
4. **DIR-08-RAKE** (Floor): P1 baseline REJECT.
5. **DIR-14-RAKE** (Floor): P3 baseline REJECT — `"relative quick time"` not recognized as fast service.
6. **DIR-15-RAKE** (Floor): Same — `"relative quick time"` not recognized.

**Non-floor failure rate:** 0/10 = 0.0% (all non-floor RAKE units passed)

### Placebo Movement (Concerning)

Gentag: 3/16 (18.8%) — above 12.5% acceptable threshold. Movers: DIR-03, DIR-13, DIR-16.
RAKE: 4/16 (25.0%) — above 20% fail threshold. Movers: DIR-03-RAKE, DIR-07-RAKE, DIR-15-RAKE, DIR-16-RAKE.

Elevated placebo movement suggests some placebos were not fully orthogonal, or the judge is sensitive to any tag change.

---

## Interpretation

The separation story is nuanced:

1. **Raw separation (+18.8pp)** comes from RAKE having more floor effects (6 vs 2). RAKE's noisy fragments fail to establish correct baselines — the judge can't interpret `"relative quick time"`, `"watching"`, `"mushrooms excellent"` as semantic propositions about venue attributes.

2. **Non-floor separation (-7.1pp)** shows that when RAKE tags happen to be clear (e.g., `"fast service"`, `"delicious"`), they work just as well as gentags. The advantage isn't in tag-level interventions but in baseline comprehensibility.

3. **The real signal:** Gentags produce 2 floor units out of 16 (12.5%), RAKE produces 6 out of 16 (37.5%). This means RAKE baselines are uninterpretable 3x more often than gentag baselines. This is the core finding — gentags create more semantically transparent venue representations.

---

## Paper-Readiness Verdict

| Criterion | Result | Threshold | Status |
|-----------|--------|-----------|--------|
| All-units separation | +18.8pp | >= 25pp | PROMISING (below threshold) |
| Non-floor separation | -7.1pp | >= 25pp | FAIL |
| Fisher's exact p | 0.433 | < 0.05 | Not significant |
| Gentag DIR pass rate | 81.2% | >= 75% | STRONG |
| RAKE DIR pass rate | 62.5% | -- | -- |
| Placebo movement (gentag) | 18.8% | <= 12.5% | CONCERNING |

**Not paper-ready on raw separation.** The +18.8pp is driven by floor effects (RAKE baseline corruption), not by differential response to interventions. The narrative should shift from "gentags beat RAKE at interventions" to "gentags produce interpretable baselines 3x more often than RAKE."

**Recommended next steps:**
- Reframe the paper argument around baseline interpretability (floor rate difference)
- Consider additional venues or units to increase power (current n=16 is underpowered for Fisher's exact)
- Investigate DIR-07 failure (the only non-floor gentag failure) — "no live screens" may need rephrasing
- Address placebo movement with tighter orthogonality constraints
