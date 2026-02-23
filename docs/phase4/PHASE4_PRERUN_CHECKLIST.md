# PHASE 4 PRE-RUN CHECKLIST

You should not execute until every box is checked.
If even one of these is undefined, you are not ready.

---

## 1. Experimental Freeze

- [x] Judge model selected — `gpt-4o`. Log exact version string returned by API on first call. No switching mid-run. (`data/phase4/mvp_config.json`)
- [x] Judge temperature documented — NOT set. Provider default. No decoding parameters touched. (`data/phase4/mvp_config.json`)
- [x] Max tokens fixed — NOT set. If output truncates, mark unit INVALID. (`data/phase4/mvp_config.json`)
- [x] N (repetitions per condition) fixed — N=5 (`data/phase4/mvp_config.json`, `PHASE4_EXECUTION_SPEC.md`)
- [x] Canonicalization rule defined — `sorted(set(tags))` (`data/phase4/mvp_config.json`, `PHASE4_EXECUTION_SPEC.md`)
- [x] RNG seed fixed — 42 (`data/phase4/mvp_config.json`, `PHASE4_EXECUTION_SPEC.md`)

If any of these can change mid-run, stop.

---

## 2. Persona Set Locked

- [x] 3-5 personas written — 3 personas (`data/phase4/mvp_personas.json`)
- [x] Each persona contains at least one hard requirement — Food Critic: "Any bad or inconsistent food quality -> REJECT"; Sports Fan: "No game viewing indication -> REJECT"; Quick Lunch Worker: "No fast service indication -> REJECT"
- [x] Personas do not overlap in primary facet — food_quality, ambiance, service
- [x] Personas are frozen in a file (no editing after first run) — `data/phase4/mvp_personas.json`

---

## 3. Representation Freeze

- [x] One gentag extraction set chosen for DIR — openai minimal run1 (`data/phase4/mvp_config.json`)
- [x] Baseline keyword system chosen — RAKE (`data/phase4/mvp_config.json`)
- [x] No extraction changes allowed during Phase 4A — documented in `PHASE4_PLAN.md` section 0.2
- [x] Tags canonicalized before every Judge call — `PHASE4_EXECUTION_SPEC.md` Part 2

If you touch extraction after seeing results, you must treat it as Phase 4B.

---

## 4. Intervention Catalog Frozen

For each DIR unit:

- [x] venue_id — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] persona_id — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] baseline_tags stored — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] edit_type defined (ADD / REMOVE / REPLACE) — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] edited_tag defined — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] expected_direction defined (UP / DOWN) — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] placebo edit defined — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] placebo matches edit type — `data/phase4/mvp_dir_units.json` (all 8 units)
- [x] Intervention tag exists in baseline if REMOVE/REPLACE — validated programmatically, all passed
- [x] Intervention tag is persona-relevant — rationale field per unit
- [x] Placebo tag is persona-orthogonal — rationale field per unit

RAKE baseline catalog: `data/phase4/mvp_dir_units_rake.json` (same 8 units, same validations passed).

Catalog saved before first execution.

---

## 5. Judge Prompt Finalized

- [x] Decision rubric included — REJECT/BORDERLINE/RECOMMEND (`PHASE4_EXECUTION_SPEC.md` Part 3)
- [x] Grounding constraint included — "Use ONLY the provided tags. Do NOT use external knowledge." (`PHASE4_EXECUTION_SPEC.md` Part 3)
- [x] "tags_used must be minimal" rule included — "tags_used MUST list the minimal tags directly supporting the decision" (`PHASE4_EXECUTION_SPEC.md` Part 3)
- [x] JSON schema strictly defined — `{"decision":..., "justification":..., "tags_used":[...]}` (`PHASE4_EXECUTION_SPEC.md` Part 3)
- [x] No dynamic prompt edits allowed — "No deviation allowed" (`PHASE4_EXECUTION_SPEC.md` Part 3)

---

## 6. Aggregation Rules Locked

- [x] Majority threshold defined (>= ceil(N/2) valid runs) — >= 3 of 5 (`PHASE4_EXECUTION_SPEC.md` Part 5)
- [x] Tie -> BORDERLINE rule defined — (`PHASE4_EXECUTION_SPEC.md` Part 5)
- [x] < threshold valid -> UNSCORABLE — (`PHASE4_EXECUTION_SPEC.md` Part 5)
- [x] UNSCORABLE excluded from pass denominator — pass rule only applies to scored units (`PHASE4_EXECUTION_SPEC.md` Part 6)
- [x] INVALID logging defined — (`PHASE4_EXECUTION_SPEC.md` Part 4)

---

## 7. Metrics Pre-Defined

Before seeing results, you must commit to reporting:

- [x] Gentag_DIR_pass_rate — (`PHASE4_EXECUTION_SPEC.md` Part 9)
- [x] Baseline_DIR_pass_rate — (`PHASE4_EXECUTION_SPEC.md` Part 9)
- [x] Placebo_movement_rate — (`PHASE4_EXECUTION_SPEC.md` Part 9)
- [x] INVALID_rate — (`PHASE4_EXECUTION_SPEC.md` Part 9)
- [x] UNSCORABLE_rate — (`PHASE4_EXECUTION_SPEC.md` Part 9)
- [x] Step_size_distribution — (`PHASE4_EXECUTION_SPEC.md` Part 9)
- [x] INV_pass_rate (if running INV) — not running INV in MVP

No adding new metrics after results appear.

---

## 8. Data Logging Plan

For every run, log:

- [x] unit_id — (`PHASE4_EXECUTION_SPEC.md` Part 4)
- [x] condition (BASELINE / INTERVENTION / PLACEBO) — (`PHASE4_EXECUTION_SPEC.md` Part 4)
- [x] run_index — (`PHASE4_EXECUTION_SPEC.md` Part 4)
- [x] raw response — (`PHASE4_EXECUTION_SPEC.md` Part 4: "Store raw response")
- [x] parsed JSON — (`PHASE4_EXECUTION_SPEC.md` Part 4: "Attempt JSON parse")
- [x] valid_flag — (`PHASE4_EXECUTION_SPEC.md` Part 4)
- [x] decision — (`PHASE4_EXECUTION_SPEC.md` Part 4)
- [x] tags_used — (`PHASE4_EXECUTION_SPEC.md` Part 4)

If you cannot reproduce every decision later, your experiment is weak.

---

## 9. Sample Size Decision

- [x] How many venues? — 1 (Colton's - Monterrey, `MFJDDz0Mgf5LOkmbvW8b`)
- [x] How many DIR units total? — 8 gentag + 8 RAKE = 16 units
- [x] How many personas per venue? — 3 (Food Critic, Sports Fan, Quick Lunch Worker)

Stage: **MVP pilot.**

---

## 10. Failure Criteria Defined

- [x] DIR pass rate — Strong: >=6/8. Weak: 4-5/8. Fail: <=3/8. (`data/phase4/mvp_config.json`)
- [x] Placebo movement — Acceptable: <=1/8. Too high: >=2/8. (`data/phase4/mvp_config.json`)
- [x] Baseline separation — Gentag pass rate must exceed RAKE by >=2 units (25pp). (`data/phase4/mvp_config.json`)
- [x] INV — N/A for MVP.

---

## STATUS: ALL BOXES CHECKED. READY TO RUN.
