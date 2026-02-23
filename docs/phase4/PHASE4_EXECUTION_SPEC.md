# PHASE 4 --- EXECUTION SPEC (STRICT VERSION)

This is not prose.
This is a runnable spec.

---

## GLOBAL FIXED CONSTANTS

| Constant | Value |
|----------|-------|
| Judge model | `<MODEL_NAME + VERSION>` (must be frozen before execution) |
| N (repetitions per condition) | 5 |
| Canonicalization | `sorted(set(tags))` |
| Decision ordinals | REJECT = 0, BORDERLINE = 1, RECOMMEND = 2 |
| RNG seed | 42 (applies to any preprocessing, sampling, or shuffling) |

**Majority rule:**

- At least 3 valid runs required.
- If tie -> BORDERLINE.
- If < 3 valid -> UNSCORABLE.

---

## PART 1 --- INPUT DEFINITIONS

Each DIR unit must contain:

```
unit_id
venue_id
persona_id
state_type          # gentag | baseline_rake | baseline_tfidf | baseline_yake

baseline_tags       # list of strings

edit_type           # ADD | REMOVE | REPLACE
edited_tag
replacement_tag     # if REPLACE

expected_direction  # UP | DOWN

placebo_edit_type
placebo_tag
placebo_replacement_tag  # if REPLACE
```

**No unit may be added after experiment begins.**

---

## PART 2 --- STATE CONSTRUCTION

For each unit:

### Step 1: Baseline State

```
S0 = sorted(unique(baseline_tags))
```

### Step 2: Intervention State

```
If ADD:
    S1 = sorted(unique(S0 + [edited_tag]))

If REMOVE:
    S1 = sorted(unique(S0 minus edited_tag))

If REPLACE:
    S1 = sorted(unique((S0 minus edited_tag) + [replacement_tag]))
```

### Step 3: Placebo State

Apply placebo edit using identical logic.

---

## PART 3 --- JUDGE PROMPT (EXACT)

**System:**

```
You are a strict Decision Judge.

Use ONLY the provided tags.
Do NOT use external knowledge.
Do NOT infer facts not present in the tags.

Decision rubric:
- REJECT: tags contain at least one clear blocker for the persona OR indicate high risk.
- RECOMMEND: tags contain clear supports AND no clear blockers.
- BORDERLINE: tags are mixed, ambiguous, or missing persona-critical information.

If persona-critical facets are not mentioned, choose BORDERLINE.

Output JSON ONLY:
{"decision":"REJECT|BORDERLINE|RECOMMEND",
 "justification":"one sentence",
 "tags_used":["..."]}

Rules:
- tags_used MUST be a subset of provided tags (exact string match).
- tags_used MUST list the minimal tags directly supporting the decision.
```

**User:**

```
Persona:
{persona_text}

Tags:
{JSON array of Sx}

Task:
Choose a decision.
```

**No deviation allowed.**

---

## PART 4 --- EXECUTION LOOP (PER UNIT)

```
For condition in [BASELINE, INTERVENTION, PLACEBO]:
    Repeat N=5 times:
        Send prompt
        Store raw response
        Attempt JSON parse
        Validate:
            - decision field valid (one of REJECT | BORDERLINE | RECOMMEND)
            - tags_used is a subset of input tags (exact string match)
        If invalid -> mark run INVALID
        Store:
            unit_id
            condition
            run_index
            valid_flag
            decision (if valid)
            tags_used (if valid)
```

---

## PART 5 --- AGGREGATION

For each condition:

```
Filter VALID runs
If valid_count < 3 -> condition UNSCORABLE
Otherwise:
    Count decisions
    Majority wins
    Tie -> BORDERLINE
Convert aggregated decision to ordinal (0, 1, 2)
```

---

## PART 6 --- DIR PASS RULE

```
Let:
    d0 = baseline decision ordinal
    d1 = intervention decision ordinal

If expected_direction == DOWN:
    PASS if d1 < d0

If expected_direction == UP:
    PASS if d1 > d0

Else FAIL.
```

---

## PART 7 --- PLACEBO CHECK

```
Let:
    dp = placebo decision ordinal

Placebo movement if:
    dp != d0

Log placebo_moved flag.
```

---

## PART 8 --- ATTRIBUTION CHECK

```
For ADD:
    attr_pass if edited_tag in tags_used (majority run)

For REPLACE:
    attr_pass if replacement_tag in tags_used

For REMOVE:
    Skip attribution requirement.
```

Log but do not hard-fail on attribution.

---

## PART 9 --- OUTPUT METRICS

After all units, report:

**Proportion metrics (binomial CI, Wilson):**

```
DIR_pass_rate
Baseline_DIR_pass_rate
Placebo_movement_rate
INVALID_rate
UNSCORABLE_rate
INV_pass_rate
```

**Descriptive metrics (no binomial CI):**

```
Step_size_distribution (|d1 - d0|)
Mean step size
Proportion step=1
Proportion step=2
```

---

## INV EXECUTION (STRICT)

For each INV unit:

### Inputs

```
venue_id
persona_id
tags_run_A
tags_run_B
```

### Procedure

```
Canonicalize both:
    A = sorted(unique(tags_run_A))
    B = sorted(unique(tags_run_B))

Run Judge N=5 on A -> aggregate -> decision_A
Run Judge N=5 on B -> aggregate -> decision_B
```

### INV Pass Rule

```
INV PASS if:
    decision_A == decision_B
```

INV separates extraction variability from judge variability.

### INV Output Metrics

```
INV_pass_rate
INVALID_rate
UNSCORABLE_rate
```
