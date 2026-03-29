# Paper Source of Truth

> **Purpose:** Define which files are authoritative for paper claims, especially numeric results and experimental setup.
> **Scope:** Whole paper, with special attention to Section 4 / Phase 5.
> **Last verified:** 2026-03-28

---

## 1. Source Precedence

When sources disagree, use this order:

1. **Run artifacts in `results/`**
   - Manifests, summaries, generated analysis JSON, frozen outputs
   - These record what was actually run
2. **Frozen data/config in `data/`**
   - Sampled venue sets, persona definitions, frozen inputs used by runs
3. **Analysis and runner scripts in `scripts/`**
   - These define how metrics were computed
4. **Top-level paper docs in `docs/`**
   - These are the best paper-facing summaries, but must match `results/`
5. **Historical or archived docs**
   - Use only for background or old rationale, not for final claims

### Why code is not the sole authority

Code shows the implemented protocol, but not necessarily the exact version that produced the paper numbers. In this repo there are multiple experiment generations, archived scripts, and stale configs. For paper claims, the canonical question is:

> "What did the completed run actually do, and what numbers did it produce?"

That answer comes from manifests and generated analysis artifacts first, then from code.

---

## 2. Known Stale or Superseded Files

These files should **not** be used as primary sources for current paper claims without cross-checking:

| File | Issue | Status |
|------|-------|--------|
| `data/phase5/baseline_config.json` | Describes older 3-persona / 4-system design | **STALE** |
| `results/_archive/phase5/baseline_manifest_20260223_051347.json` | Older pre-v2 run | Superseded |
| `results/_archive/phase5/baseline_summary_20260223_051347.json` | Older pre-v2 run | Superseded |
| `results/_archive/phase5/baseline_summary_48venues.json` | Partial/older slice | Superseded |
| `docs/_archive/superseded/GENTAGS_FULL_ANALYSIS_REPORT.md` | Useful context, but Phase 3 framing is outdated | Secondary |
| `docs/_archive/**` | Historical planning and older narratives | Historical only |

For current Phase 5 claims, use the 2026-02-28 v2 artifacts.

---

## 3. Section-by-Section Map

### Section 1. Introduction

**Current status:** written in `docs/PAPER_complete.md`  
**Use as inputs:**
- `docs/_archive/superseded/paper_problem_question_report.md`
- `docs/_archive/superseded/MOTIVATION.md` as background only
- verified results sections below for finalized claims

**Do not use as sole authority:** old framing docs in `docs/_archive/`

### Section 2. Related Work

**Current status:** written in `docs/PAPER_complete.md`  
**Use as inputs:**
- literature notes to be gathered separately
- `docs/_archive/superseded/paper_problem_question_report.md` for framing boundaries only

### Section 3. Method - Extraction

**Primary paper doc:**
- `docs/EXTRACTION.md`

**Authority for protocol details:**
- `src/gentags/pipeline.py`
- `data/study1_venues_20250117.csv`
- `results/phase1_downloaded/*`

**Use for paper writing:**
- prose, definitions, prompt text, and taxonomy from `docs/EXTRACTION.md`
- if any number is challenged, verify against Phase 1 outputs and pipeline code

### Section 4. Method - Stability

**Primary paper input doc:**
- `docs/PHASE2_STABILITY.md`

**Authority for numbers:**
- `results/phase2/plots/*`
- `results/phase2/tables/*` if present
- `scripts/phase2_analysis.py`

**Use for paper writing:**
- narrative from `docs/PHASE2_STABILITY.md`
- final numbers from generated tables/plots and analysis code

### Section 5. Method - Structure (State-Gini)

**Primary design doc:**
- `docs/_archive/superseded/PHASE3_STATE_GINI_PLAN.md`

**Authority for executed results:**
- `results/phase3/*`
- `scripts/phase3a_baselines.py`
- `scripts/state_gini_venue_aggregate.py`

**Important note:**
- The plan doc is authoritative for the intended protocol.
- Final paper numbers must come from executed `results/phase3/*`, not from the plan alone.

### Section 6. Method - Decision Utility

**Primary paper input doc:**
- `docs/phase5/BASELINE_LEGIBILITY_REPORT.md`

**Authority for setup and results:**
- `results/phase5/baseline_manifest_openai_20260228_022320.json`
- `results/phase5/baseline_manifest_claude_20260228_032717.json`
- `results/phase5/baseline_summary_openai_20260228_022320.json`
- `results/phase5/baseline_summary_claude_20260228_032717.json`
- `results/phase5/baseline_legibility_analysis.json`
- `data/phase5/sampled_venues.json`
- `data/phase5/phase5_personas.json`
- `scripts/phase5_baseline_runner.py`
- `scripts/phase5_analyze.py`

**Rule:**
- For Section 6, numeric claims should come from `baseline_legibility_analysis.json` and manifests first.
- The report doc should be treated as a checked summary, not the canonical numeric source.

### Section 7. Results

**Best source pattern:**
- Pull from finalized per-phase artifacts, not from older omnibus reports

Use:
- Section 3: `docs/EXTRACTION.md` plus Phase 1 artifacts
- Section 4: Phase 2 artifacts
- Section 5: Phase 3 artifacts
- Section 6: Phase 5 artifacts

Avoid:
- relying on `docs/_archive/superseded/GENTAGS_FULL_ANALYSIS_REPORT.md` for final consolidated numbers

### Section 8. Discussion

**Use as inputs:**
- verified results from Sections 3-6
- `docs/_archive/superseded/paper_problem_question_report.md`
- `docs/phase4/DIR_SCALED_RUN_REPORT.md` only as supporting mechanism evidence

### Section 9. Limitations

**Use as inputs:**
- `docs/PAPER_STATUS.md`
- `docs/_archive/superseded/paper_problem_question_report.md`
- Phase-specific caveats from manifests and reports

### Section 10. Conclusion

**Use as inputs:**
- only finalized claims already supported above

---

## 4. Section 4 Experimental Setup: Primary Drafting Map

This is the immediate source map for writing the paper's experimental setup section around the decision study.

### 4.1 Data and Decision Context

Use these sources:

- `data/phase5/sampled_venues.json`
  - authoritative for sampled set size and stratification
  - current metadata: `n_venues = 50`, strata = `6 game`, `15 speed`, `29 neither`
- `data/phase5/phase5_personas.json`
  - authoritative for persona definitions and hard requirements
  - current design: `P1`, `P2`, `P3`, `P4`
- `results/phase5/baseline_manifest_openai_20260228_022320.json`
  - confirms `n_venues = 50`, `n_personas = 4`

Paper-safe facts currently verified:

- 50 venues
- 4 personas
- 3 hard-requirement personas (`P1`, `P2`, `P3`)
- 1 soft persona (`P4`)
- stratified venue sample: 6 game-tag, 15 speed-tag, 29 neither

### 4.2 Systems and Baselines

Use these sources:

- `results/phase5/baseline_manifest_openai_20260228_022320.json`
  - authoritative for systems actually run
- `scripts/phase5_baseline_runner.py`
  - authoritative for system definitions
- `data/phase5/sampled_venues.json`
  - authoritative for per-venue gentag / RAKE / YAKE / TF-IDF payloads

Paper-safe facts currently verified:

- systems: `gentag`, `rake`, `yake`, `tfidf`, `gentag_truncated`, `fer`
- `gentag_truncated` is the token-count / volume control
- `fer` is the full-evidence reference condition using raw reviews

### 4.3 Judges and Aggregation

Use these sources:

- `scripts/phase5_baseline_runner.py`
  - authoritative for `N = 5`, `MIN_VALID = 3`, majority-vote aggregation, tie -> `BORDERLINE`
- `results/phase5/baseline_manifest_openai_20260228_022320.json`
  - authoritative for primary judge run
- `results/phase5/baseline_manifest_claude_20260228_032717.json`
  - authoritative for cross-judge run
- `results/phase5/baseline_legibility_analysis.json`
  - authoritative for cross-judge agreement metrics

Paper-safe facts currently verified:

- primary judge: `gpt-4o-2024-08-06`
- cross-judge: `claude-sonnet-4-20250514`
- aggregation: `N = 5`, majority vote
- minimum valid responses per condition: `MIN_VALID = 3`
- cross-judge overall agreement: `81.35%`, `kappa = 0.7117`

### 4.4 Controlled Factors

Use these sources:

- `scripts/phase5_baseline_runner.py`
  - same decision prompt family and aggregation logic across systems
  - strict JSON validation
- `data/phase5/phase5_personas.json`
  - frozen indicator lexicons and requirement rules
- `results/phase5/baseline_legibility_analysis.json`
  - token-budget ablation and decision-distribution metrics
- `docs/phase5/BASELINE_LEGIBILITY_REPORT.md`
  - good prose summary after verification

Paper-safe facts currently verified:

- frozen persona lexicons
- exact-match indicator logic for hard personas
- strict structured output validation
- same judge / prompt family / aggregation across systems
- token-budget ablation via `gentag_truncated`
- same sampled venue set across systems

---

## 5. Verified Phase 5 Numbers for Paper Drafting

Use these numbers directly from `results/phase5/baseline_legibility_analysis.json` unless a later rerun supersedes them.

### Primary judge (OpenAI)

| Metric | gentag | RAKE | YAKE | TF-IDF |
|--------|--------|------|------|--------|
| FER agreement | 79.5% | 61.6% | 58.5% | 52.3% |
| Kappa vs FER | 0.6672 | 0.3877 | 0.3514 | 0.2579 |
| Combined compliance (P1+P2+P3) | 97.33% | 89.26% | 84.67% | 86.00% |

### Ablation

| Metric | gentag_truncated |
|--------|------------------|
| FER agreement | 74.87% |
| Kappa vs FER | 0.5956 |
| Combined compliance | 94.63% |

### Cross-judge

| Metric | Value |
|--------|-------|
| Overall agreement | 81.35% |
| Overall kappa | 0.7117 |

### Run manifests

| Run | Key facts |
|-----|-----------|
| OpenAI `20260228_022320` | 50 venues, 4 personas, 6 systems, 1200 conditions, 6000 calls, invalid rate 3.35% |
| Claude `20260228_032717` | 50 venues, 4 personas, 6 systems, 1199 conditions, 5995 calls, invalid rate 10.44% |

---

## 6. Drafting Rules

When writing paper text:

- Use `docs/` files for prose structure.
- Use `results/` JSON and manifests for all numeric claims.
- Use `data/` files for sample composition and persona definitions.
- Use `scripts/` to explain protocol details that are not obvious from artifacts.
- Do not cite stale configs or early-run summaries once superseded artifacts exist.

### One-line rule

If a sentence contains a number, the number should be traceable to `results/` or a frozen `data/` file.
