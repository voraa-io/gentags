# Phase 4 — Sample Venue Test Design

**Venue:** Colton's - Monterrey (`MFJDDz0Mgf5LOkmbvW8b`)
**Sparsity:** 176 min input tokens, 5 reviews, 332 chars total
**Data:** `results/phase4/sample_venue.json`

---

## Source Reviews

1. "I have one large vampiro in a Large pizza with sausage pepperoni and mushrooms excellent very soft crust"
2. "Not bad but nothing special, service was the highlight."
3. "Worse pizza I have try not good at all"
4. "Nice place,"
5. "Excellent for watching your games, very good food that matches the intention of the place, good atmosphere and fast service."

---

## Gentag State (OpenAI, minimal, run1) — 12 tags

```json
["bad pizza", "excellent food", "excellent pizza", "fast service",
 "good atmosphere", "good food", "good service", "large pizza",
 "nice place", "nothing special", "soft crust", "watching game"]
```

**Facet coverage:**
- Food Quality: `bad pizza`, `excellent food`, `excellent pizza`, `good food`, `large pizza`, `soft crust`
- Service: `fast service`, `good service`
- Ambiance: `good atmosphere`, `nice place`, `watching game`
- Uncategorized: `nothing special`

---

## User Personas

| persona_id | Name | Description | Critical facets |
|------------|------|-------------|-----------------|
| P1 | Budget Family | Family of 4 looking for affordable, kid-friendly dining. Needs good food quality and reasonable portions. Sensitive to price and noise. | food_quality, price_value, portions |
| P2 | Disabled Traveler | Wheelchair user visiting the city. Requires physical accessibility. Values clear information about venue layout. | accessibility, seating |
| P3 | Sports Fan | Wants a place to watch live games with friends. Values atmosphere, screen visibility, and drinks. Food is secondary. | ambiance, service |
| P4 | Food Critic | Looking for culinary quality and consistency. Low tolerance for bad food. Ambiance and service matter but food is primary. | food_quality, service |
| P5 | Quick Lunch Worker | Office worker on a 30-minute lunch break. Needs fast service and decent food. Does not care about ambiance. | service, food_quality |

---

## DIR Intervention Catalog (Sample Venue)

Each row is one DIR unit. Baseline state S0 = the 12 gentags above.

| unit_id | persona_id | edit_type | edited_tag | replacement_tag | expected_direction | rationale |
|---------|------------|-----------|------------|-----------------|-------------------|-----------|
| DIR-01 | P2 | ADD | no wheelchair ramp | — | DOWN | Accessibility blocker for disabled traveler. Tag not in state = missing info. Adding it should trigger REJECT. |
| DIR-02 | P4 | REMOVE | bad pizza | — | UP | Removing a negative food signal should improve score for food critic. |
| DIR-03 | P4 | ADD | stale bread | — | DOWN | Adding a food quality blocker. Food critic should downgrade. |
| DIR-04 | P5 | REMOVE | fast service | — | DOWN | Removing the speed signal that matters most to quick lunch worker. |
| DIR-05 | P3 | REMOVE | watching game | — | DOWN | Removing the core value prop for sports fan. |
| DIR-06 | P1 | ADD | overpriced | — | DOWN | Budget family is price-sensitive. Adding price blocker. |
| DIR-07 | P1 | REPLACE | good food | poor food quality | DOWN | Flipping food quality for budget family. |
| DIR-08 | P4 | REPLACE | excellent pizza | mediocre pizza | DOWN | Downgrading quality signal for food critic. |
| DIR-09 | P3 | ADD | no live screens | — | DOWN | Sports fan needs screens. Blocker. |
| DIR-10 | P5 | REPLACE | fast service | slow service | DOWN | Flipping the speed signal for quick lunch worker. |

### Placebo edits (paired with each DIR unit)

| unit_id | placebo_edit_type | placebo_tag | placebo_replacement_tag | rationale |
|---------|-------------------|-------------|------------------------|-----------|
| DIR-01 | ADD | blue wall paint | — | Irrelevant to accessibility persona. Should not move decision. |
| DIR-02 | REMOVE | large pizza | — | Size is secondary for food critic. Should not flip decision. |
| DIR-03 | ADD | street parking | — | Irrelevant to food quality. |
| DIR-04 | ADD | red tablecloth | — | Irrelevant to speed. |
| DIR-05 | REMOVE | soft crust | — | Crust detail is irrelevant to sports fan. |
| DIR-06 | ADD | live music friday | — | Irrelevant to price. |
| DIR-07 | REMOVE | watching game | — | Entertainment is secondary for budget family food decision. |
| DIR-08 | ADD | free wifi | — | Irrelevant to food quality for critic. |
| DIR-09 | REMOVE | nothing special | — | Vague tag, irrelevant to screen availability. |
| DIR-10 | ADD | outdoor seating | — | Irrelevant to service speed. |

---

## INV Test Cases (Sample Venue)

Uses Run 1 vs Run 2 from each model. Same persona, different tag sets.

| inv_id | persona_id | model | tags_run_A source | tags_run_B source | expected |
|--------|------------|-------|-------------------|-------------------|----------|
| INV-01 | P4 | openai | openai run1 (12 tags) | openai run2 (11 tags) | decision_A == decision_B |
| INV-02 | P4 | gemini | gemini run1 (8 tags) | gemini run2 (10 tags) | decision_A == decision_B |
| INV-03 | P4 | claude | claude run1 (9 tags) | claude run2 (7 tags) | decision_A == decision_B |
| INV-04 | P4 | grok | grok run1 (10 tags) | grok run2 (10 tags) | decision_A == decision_B |
| INV-05 | P3 | openai | openai run1 | openai run2 | decision_A == decision_B |
| INV-06 | P3 | gemini | gemini run1 | gemini run2 | decision_A == decision_B |
| INV-07 | P3 | claude | claude run1 | claude run2 | decision_A == decision_B |
| INV-08 | P3 | grok | grok run1 | grok run2 | decision_A == decision_B |
| INV-09 | P5 | openai | openai run1 | openai run2 | decision_A == decision_B |
| INV-10 | P5 | claude | claude run1 | claude run2 | decision_A == decision_B |

---

## Baseline Comparison Notes

For each DIR unit, also run with RAKE keywords as S0 (same edit logic). Key observations:

- RAKE has 20 keywords but many are noise: `"intention"`, `"try"`, `"highlight"`, `"matches"`.
- RAKE **does** capture `fast service`, `good atmosphere`, `good food`, `nice place`, `soft crust` — some overlap with gentags.
- RAKE **does not** capture: `watching game` (only `"watching"`), `good service` (only `"service"`), `excellent food`/`excellent pizza` (only `"excellent"`).
- TF-IDF is worse: mostly n-gram fragments like `"good atmosphere fast service"`, `"large vampiro large pizza"`.

**Prediction:** DIR-05 (remove `watching game` for sports fan) should FAIL on RAKE because RAKE only has the fragment `"watching"` not the semantic proposition. DIR-01 (add `no wheelchair ramp`) should FAIL on all baselines because none capture accessibility information.

---

## What This File Is

This is the **test design worksheet** for the sample venue. It will be used to:
1. Validate the execution spec (PHASE4_EXECUTION_SPEC.md) during preflight
2. Inform the full intervention catalog when we scale to 40+ venues
3. Document the reasoning behind persona and intervention choices

**Next steps:**
- Freeze Judge LLM choice (Decision #1 in PHASE4_PLAN.md section 9)
- Run preflight: execute DIR-01 through DIR-03 and INV-01 through INV-02 to validate pipeline
- Scale to more venues using `scripts/phase4_sample_venue.py --list-candidates`
