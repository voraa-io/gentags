# Codebase Evaluation: researchGentags

> **Date:** 2026-03-28
> **Scope:** Full repo audit — code, docs, data, results, tests, config
> **Role of this file:** cleanup source of truth for repo hygiene and reproducibility work. Update this file when cleanup decisions change.

---

## 1. Project Overview

**Gentags** is a research project backing an ACL 2026 paper. It studies whether discrete, evidence-grounded semantic tags (extracted by LLMs from venue reviews) improve constraint-sensitive decision reliability vs. lexical baselines (RAKE, YAKE, TF-IDF).

| Fact | Value |
|------|-------|
| Version | 1.2.0 |
| Python | 3.9+ (Poetry managed) |
| Dataset | 553 venues with Google reviews |
| Paper draft | `docs/PAPER_complete.md` — **COMPLETE** |
| Phases | 5 experimental phases, all executed |

---

## 2. Repository Structure

```
researchGentags/
├── src/gentags/          Core extraction pipeline (3 files, ~1,470 LOC)
├── scripts/              Phase runners + analysis (24 active, 13 archived)
├── tests/                Unit tests (4 files)
├── data/                 Frozen inputs (venues, personas, configs)
├── results/              Run artifacts (manifests, summaries, plots)
├── docs/                 Paper-facing docs + 19 active markdown docs + 41 archived markdown docs
├── notebooks/            3 Jupyter notebooks (exploratory)
├── examples/             1 file (Phase 3 facet matching example)
├── issues/               1 file (resolved perf issue)
├── pyproject.toml        Poetry config
├── AGENTS.md             Repo guidelines
├── README.md             Project overview
├── SETUP.md              Environment setup
├── CITATION.cff          Citation metadata
└── VM.md                 GCP VM notes (minimal)
```

---

## 3. Critical Issues

### 3.1 RESOLVED: tests now pass

```
tests/test_parsing.py    → imports from gentags.parsing   (DOES NOT EXIST)
tests/test_filter.py     → imports from gentags.normalize (DOES NOT EXIST)
tests/test_normalize.py  → imports from gentags.normalize (DOES NOT EXIST)
```

All three functions (`extract_json_list`, `filter_valid_tags`, `normalize_tag`) live in `gentags.pipeline`, not in separate modules. The tests were fixed by updating the test imports rather than changing pipeline code.

**Current state:** `poetry run pytest tests/` passes (`28 passed`).

**Cleanup rule preserved:** test harness fixes are allowed; pipeline behavior changes just to satisfy tests are not.

### 3.2 Duplicate figure locations

Figures exist in **three** places with no clear canonical source:

| Location | Contents |
|----------|----------|
| `docs/_archive/plots/` | 14 archived PNG copies (named `fig1_pipeline_overview.png`, `figA1_...`, etc.) |
| `results/figures/` | 1 file (`fig1_pipeline.png`) |
| `results/phase*/plots/` | Raw phase outputs (named `1_run_stability.png`, etc.) |

`PAPER_complete.md` references `results/figures/` (Fig 1 only) and `results/phase*/plots/` (all other figures). The duplicated doc-local copies were archived during cleanup because they were unused by the paper and had different filenames.

**Current state:** non-canonical duplicate plot copies live in `docs/_archive/plots/`.

### 3.3 Stale data config still present

`data/phase5/baseline_config.json` describes an older 3-persona / 4-system design. The actual experiment used 4 personas / 6 systems. `PAPER_SOURCE_OF_TRUTH.md` already flags it as STALE.

**Current state:** the file now includes an explicit in-file stale note and is retained only for historical reference.

### 3.4 RESOLVED: repo-facing docs now match the current test/validation path

This was previously a problem in:

- `README.md` says `poetry run pytest tests/` passes and references `scripts/smoke_test_minimal.py`
- `AGENTS.md` repeats the same guidance

It has now been corrected:

- `pytest` passes
- both docs now point to a verified `scripts/run_phase1.py --dry-run` validation path instead of the archived smoke test

### 3.5 Phase 1 extraction path is under-documented

The repo structure strongly suggests the large Phase 1 extraction was run on a Google Cloud VM and then copied back locally:

- `VM.md` contains a direct `gcloud compute ssh voraa-gentags --zone=us-central1-c` command
- `docs/_archive/historical/GCP_VM_SETUP.md` is a full VM setup and run guide for Phase 1
- `docs/_archive/historical/WEEK2_PHASE1_COMPLETE.md` says the extraction ran on Google Cloud Compute Engine
- `scripts/_archive/download_phase1_results.py` downloads Phase 1 outputs from a VM via `gcloud`

The current local canonical copy of those outputs is `results/phase1_downloaded/`, and downstream scripts depend on that path.

This was corrected in the current visible docs during cleanup; `results/phase1_downloaded/` is now the current documented local path.

---

## 4. Redundancy Analysis

### 4.1 Documentation redundancy (HIGH)

The `docs/` directory has **60 markdown files** (19 active + 41 archived). Many cover the same content at different stages of development. During cleanup, the most obvious paper-section drafts, superseded summaries, and secondary top-level docs were moved into `docs/_archive/superseded/`.

#### Files fully superseded by PAPER_complete.md

These docs' content is now entirely contained in the final paper draft:

| File | What it covered | Status |
|------|----------------|--------|
| `docs/_archive/superseded/SECTION3_GENTAGS.md` | Paper Section 3 draft | Archived |
| `docs/_archive/superseded/SECTION3_REPRESENTATION.md` | Alternative Section 3 framing | Archived |
| `docs/_archive/superseded/SECTION4_EXPERIMENTAL_SETUP.md` | Paper Section 4 draft | Archived |
| `docs/_archive/superseded/SECTION5_ANALYSIS_RESULTS.md` | Paper Section 5 draft | Archived |
| `docs/_archive/superseded/SECTION6_DISCUSSION.md` | Paper Section 6 draft | Archived |
| `docs/_archive/superseded/RESULTS.md` | Results narrative | Archived |
| `docs/_archive/superseded/APPENDIX_FAILURE_AUDIT.md` | Appendix B content | Archived |
| `docs/_archive/superseded/PAPER_FIGURES.md` | Figure inventory | Archived |
| `docs/_archive/superseded/MOTIVATION.md` | Problem framing | Archived |

#### Files that still serve a purpose

| File | Why keep it |
|------|------------|
| `PAPER_complete.md` | **THE** paper |
| `PAPER_SOURCE_OF_TRUTH.md` | Maps paper claims to source artifacts |
| `PAPER_STATUS.md` | Tracks open items |
| `REPRODUCE_PAPER.md` | Single operational entry point for installation, canonical artifacts, and reruns |
| `EXTRACTION.md` | Authoritative method doc (prompts, taxonomy detail beyond paper) |
| `PHASE2_STABILITY.md` | Detailed Phase 2 numbers backing paper claims |
| `PHASE3_METHODOLOGY_FIX.md` | Documents the anchor-fix decision |
| `phase3/state_gini_full_run_analysis.md` | Phase 3 results backing paper |
| `phase3/state_gini_preflight_runs.md` | Preflight decisions |
| `phase4/DIR_SCALED_RUN_REPORT.md` | Phase 4 results (discussion evidence) |
| `phase5/BASELINE_LEGIBILITY_REPORT.md` | Phase 5 detailed results |

#### Files that are pure historical archive

Already in `_archive/`: 41 markdown files (plans + historical + superseded visible docs). These are fine where they are.

Still visible and should be considered for archival next:

| File | Reason |
|------|--------|
| `phase3/STATE_GINI_FOLLOWUPS.md` | Follow-up plans (unclear if executed) |
| `phase4/PHASE4_PLAN.md` | Planning doc |
| `phase4/PHASE4_EXECUTION_SPEC.md` | Spec (useful for reproducibility) |
| `phase4/PHASE4_PRERUN_CHECKLIST.md` | Checklist |
| `phase4/DIR_MVP_RUN_REPORT.md` | Superseded by scaled run |
| `phase4/sample_venue_test_design.md` | Design worksheet |

Recently archived from visible `docs/` surface:

- `docs/_archive/superseded/GENTAGS_FULL_ANALYSIS_REPORT.md`
- `docs/_archive/superseded/PHASE3_STATUS.md`
- `docs/_archive/superseded/PHASE3_INFOGRAPHIC.md`
- `docs/_archive/superseded/RESEARCH.md`
- `docs/_archive/superseded/SECTION3_GENTAGS.md`
- `docs/_archive/superseded/SECTION3_REPRESENTATION.md`
- `docs/_archive/superseded/SECTION4_EXPERIMENTAL_SETUP.md`
- `docs/_archive/superseded/SECTION5_ANALYSIS_RESULTS.md`
- `docs/_archive/superseded/SECTION6_DISCUSSION.md`
- `docs/_archive/superseded/RESULTS.md`
- `docs/_archive/superseded/PAPER_FIGURES.md`
- `docs/_archive/superseded/APPENDIX_FAILURE_AUDIT.md`
- `docs/_archive/superseded/PHASE3_STATE_GINI_PLAN.md`
- `docs/_archive/superseded/STUDY1_LOCK.md`
- `docs/_archive/superseded/paper_problem_question_report.md`
- `docs/_archive/superseded/MOTIVATION.md`
- `docs/_archive/superseded/PHASE1_EXTRACTION.md`

### 4.2 Results redundancy

Phase 5 previously had superseded files mixed into the canonical output directory. These have now been archived to `results/_archive/phase5/`:

| File | Status |
|------|--------|
| `results/_archive/phase5/baseline_manifest_20260223_051347.json` | Superseded (pre-v2, 3 personas) |
| `results/_archive/phase5/baseline_summary_20260223_051347.json` | Superseded |
| `results/_archive/phase5/baseline_results_20260223_051347.json` | Superseded |
| `results/_archive/phase5/baseline_results_20260223_052658_partial.json` | Partial run |
| `results/_archive/phase5/baseline_results_20260223_062343_partial.json` | Partial run |
| `results/_archive/phase5/baseline_summary_48venues.json` | Old partial slice |
| `results/_archive/phase5/baseline_manifest_claude_20260228_015756.json` | Early Claude iteration |
| `results/_archive/phase5/baseline_summary_claude_20260228_015756.json` | Early Claude iteration |
| `results/_archive/phase5/baseline_results_claude_20260228_015756.json` | Early Claude iteration |
| `results/_archive/phase5/checkpoint.json` | Leftover checkpoint |

**Canonical Phase 5 artifacts (keep):**
- `*_openai_20260228_022320.*` (primary judge)
- `*_claude_20260228_032717.*` (cross-judge)
- `baseline_legibility_analysis.json` (computed metrics)
- `gentag_fer_disagreements.csv` (audit data)
- `plots/` (4 figures)

### 4.3 Phase 1 artifact location drift

For active downstream work, the canonical local Phase 1 artifacts are in:

- `results/phase1_downloaded/`

This is what active scripts use for:

- Phase 2/3 loading
- Phase 4 sample selection
- Phase 5 venue sampling

This drift has been corrected in the current visible docs. Historical docs in `_archive/` may still mention older paths.

### 4.4 Script redundancy

No obvious duplicates among the 24 active scripts. The 13 archived scripts in `scripts/_archive/` are clearly superseded (v1 versions replaced by current scripts). Clean.

### 4.5 Code redundancy

`src/gentags/pipeline.py` (1,287 LOC) is a monolith containing extraction, parsing, normalization, filtering, experiment running, cost tracking, and I/O.

There is also real duplication between `src/gentags/config.py` and `src/gentags/pipeline.py`: both define prompts, model metadata, hashes, and version info. Public package exports in `src/gentags/__init__.py` currently come from `pipeline.py`, while `test_hashes.py` imports from `config.py`. This split is workable, but it means the repo currently has two overlapping configuration sources rather than one clear authority.

---

## 5. Correctness Check

### 5.1 Paper numbers vs source artifacts

Headline paper numbers in `PAPER_complete.md` match the canonical run artifacts. The strongest checks are against `baseline_legibility_analysis.json`, the Phase 5 manifests, and the Phase 2/3 outputs:

| Claim in paper | Value | Verified against |
|---------------|-------|-----------------|
| FER agreement (gentag) | 79.5% | `baseline_legibility_analysis.json` |
| FER agreement (RAKE) | 61.6% | Same |
| FER agreement (YAKE) | 58.5% | Same |
| FER agreement (TF-IDF) | 52.3% | Same |
| Kappa vs FER (gentag) | 0.667 | Same (0.6672) |
| Combined compliance (gentag) | 97.3% | Same (97.33%) |
| Combined compliance (RAKE) | 89.3% | Same (89.26%) |
| Combined compliance (YAKE) | 84.7% | Same (84.67%) |
| Combined compliance (TF-IDF) | 86.0% | Same (86.00%) |
| Ablation FER agreement | 74.9% | Same (74.87%) |
| Cross-judge agreement | 81.3% | Same (81.35%) |
| Cross-judge kappa | 0.712 | Same (0.7117) |
| Run-to-run cosine | 0.977 | `results/phase2/tables/` |
| Run-to-run Jaccard | 0.471 | Same |
| State-Gini (gentag) | 0.600 | `results/phase3/tables/` |
| 50 venues | 50 | `data/phase5/sampled_venues.json` |
| 4 personas | 4 | `data/phase5/phase5_personas.json` |
| 6 systems | 6 | OpenAI manifest |
| N=5 judge calls | 5 | Script + manifest |
| Primary judge | gpt-4o-2024-08-06 | OpenAI manifest |
| Cross judge | claude-sonnet-4-20250514 | Claude manifest |
| OpenAI invalid rate | 3.35% | Manifest (201/6000) |
| Claude invalid rate | 10.44% | Manifest (626/5995) |

**Headline numeric claims spot-check correctly.**

`PAPER_SOURCE_OF_TRUTH.md` remains useful and is now aligned with the current draft status and Phase 1 local artifact path.

### 5.2 Figure references

All 14 figure references in `PAPER_complete.md` point to files that exist:

| Figure | Path | Exists? |
|--------|------|---------|
| Fig 1 | `results/figures/fig1_pipeline.png` | Yes |
| Fig 2 | `results/phase2/plots/6_surface_vs_semantic.png` | Yes |
| Fig 3 | `results/phase2/plots/1_run_stability.png` | Yes |
| Fig 4 | `results/phase2/plots/7_sparsity_analysis.png` | Yes |
| Fig 5 | `results/phase3/plots/1_gini_and_coverage.png` | Yes |
| Fig 6 | `results/phase5/plots/1_fer_agreement_and_compliance.png` | Yes |
| Fig 7 | `results/phase5/plots/2_decision_distribution.png` | Yes |
| Fig B1 | `results/phase5/plots/4_failure_audit.png` | Yes |
| Fig C1 | `results/phase2/plots/2_prompt_sensitivity.png` | Yes |
| Fig C2 | `results/phase2/plots/3_model_sensitivity.png` | Yes |
| Fig C3 | `results/phase2/plots/4_retention.png` | Yes |
| Fig C4 | `results/phase3/plots/2_threshold_sensitivity.png` | Yes |
| Fig C5 | `results/phase3/plots/3_facet_heatmap.png` | Yes |
| Fig C6 | `results/phase5/plots/3_cross_judge_kappa.png` | Yes |

**All figure paths resolve.**

---

## 6. Code Health

### 6.1 src/gentags/ (core package)

| File | LOC | Status |
|------|-----|--------|
| `__init__.py` | 37 | Clean exports |
| `config.py` | 146 | Frozen configs, but duplicated in `pipeline.py` |
| `pipeline.py` | 1,287 | Monolith but functional |

- 4 LLM providers (OpenAI, Gemini, Claude, Grok)
- 3 frozen prompts (minimal, anti_hallucination, short_phrase)
- JSON parsing with 3-strategy fallback
- Checkpoint/resume support
- Cost tracking per extraction

### 6.2 scripts/ (24 active)

| Phase | Scripts | Purpose |
|-------|---------|---------|
| Phase 1 | 1 | Full extraction run |
| Phase 2 | 2 | Stability analysis + plots |
| Phase 3 | 9 | State-Gini (preflight, full, bleed check, baselines, probes, plots) |
| Phase 4 | 3 | DIR experiment (sample, run, aggregate) |
| Phase 5 | 5 | Decision study (sample, add baselines, run, analyze, plots) |
| Utility | 2 | Shared plot style, pipeline diagram |
| Shell | 2 | Phase 2/3 runners |

No duplicates. Clear dependency chain: Phase 1 -> Phase 2 (cache) -> Phase 3 -> Phase 5.

### 6.3 tests/ (4 files, now passing)

| File | Tests | Status |
|------|-------|--------|
| `test_hashes.py` | 6 | **PASSES** |
| `test_parsing.py` | 8 | **PASSES** |
| `test_filter.py` | 6 | **PASSES** |
| `test_normalize.py` | 8 | **PASSES** |

### 6.4 Dependencies (pyproject.toml)

Core: pandas, openai, anthropic, google-genai, scipy, scikit-learn, matplotlib, seaborn, nltk, yake, rake-nltk
Dev: pytest, black, ruff, jupyter

All reasonable. Versions are constrained by ranges, not pinned exactly.

---

## 7. Empty / Unused Components

| Component | Status | Action |
|-----------|--------|--------|
| `data/external/` | gitignored placeholder | Fine |
| `data/sample/` | Empty placeholder | Fine |
| `notebooks/` | 3 exploratory notebooks | Fine (dev tools) |

Resolved in cleanup:

- deleted empty `CLAUDE.md`
- deleted unused empty `app/backend/` and `app/frontend/` placeholders

---

## 8. Summary

### What's solid

- **Paper draft is complete** and headline numbers match the canonical source artifacts
- **All 5 experimental phases** have executed runs with manifests
- **Source-of-truth approach** is strong: manifests/results are treated as the authority over planning docs
- **No code duplication** among active scripts
- **Clean dependency chain** between phases
- **Figure paths** all resolve correctly
- **Phase 1 outputs needed for later phases are present locally** in `results/phase1_downloaded/`
- **Top-level docs surface is much smaller** after archiving superseded section drafts

### What needs fixing

| Priority | Issue | Effort |
|----------|-------|--------|
| **MEDIUM** | Config metadata is duplicated across `src/gentags/config.py` and `src/gentags/pipeline.py` | 15 min |
| **MEDIUM** | Visible docs are now lean at the top level, but some secondary phase docs could still move into archive | 10 min |

### Open items for the paper

From `PAPER_STATUS.md`:
- [ ] Human gold labels (`data/phase5/gold_labels_manual.json`)
- [ ] Phase 4 DIR: paper section or appendix?
- [ ] LaTeX conversion
- [ ] Final figure numbering

---

## 9. Cleanup Working Plan

This is the current repo-cleanup execution order. Treat it as the working checklist.

### 9.1 Verified so far

- [x] Repository structure and file counts checked against the live tree
- [x] `pytest` failure mode verified directly
- [x] Phase 5 canonical manifests and analysis artifacts verified
- [x] Paper figure paths verified
- [x] `docs/plots/` duplication verified and archived to `docs/_archive/plots/`
- [x] Phase 1 extraction provenance traced to GCP VM docs + download script
- [x] Confirmed active downstream dependence on `results/phase1_downloaded/`

### 9.2 Next cleanup steps

1. Fix broken test imports so `poetry run pytest tests/` passes.
2. Update `README.md` and `AGENTS.md` so they match the actual current repo.
3. Make the canonical reproduction path explicit:
   - where Phase 1 came from
   - which local artifacts are canonical
   - which files back the paper
4. Clean obvious clutter:
   - stale Phase 5 config warning / archival
   - empty placeholders if unused
5. Reduce source-of-truth drift across docs:
   - fix `PHASE1_EXTRACTION.md`
   - fix stale status prose in `PAPER_SOURCE_OF_TRUTH.md`
   - add a simple reproduction doc

### 9.3 Canonical paths to preserve during cleanup

- Paper draft: `docs/PAPER_complete.md`
- Paper claim/artifact map: `docs/PAPER_SOURCE_OF_TRUTH.md`
- Paper status tracker: `docs/PAPER_STATUS.md`
- Local Phase 1 artifacts used by later phases: `results/phase1_downloaded/`
- Phase 2 outputs: `results/phase2/`
- Phase 3 outputs: `results/phase3/`
- Phase 5 canonical runs:
  - `results/phase5/baseline_manifest_openai_20260228_022320.json`
  - `results/phase5/baseline_manifest_claude_20260228_032717.json`
  - `results/phase5/baseline_legibility_analysis.json`

Archived non-canonical Phase 5 artifacts now live in:

- `results/_archive/phase5/`

### 9.4 Important cleanup rule

Do not rely on older docs when they conflict with executed artifacts or active scripts.

Priority order for truth during cleanup:

1. Executed artifacts in `results/`
2. Frozen inputs in `data/`
3. Active scripts in `scripts/`
4. Current paper docs in `docs/`
5. Archived / historical docs only for background

### 9.5 What not to change during cleanup

Do not change anything that could affect paper-backed behavior unless we explicitly decide to rerun experiments and bless new outputs.

Specifically, do **not**:

- edit `src/gentags/pipeline.py` or `src/gentags/config.py`
- change active experiment scripts in `scripts/` unless the change is documentation-only or strictly non-behavioral
- overwrite or regenerate canonical artifacts in:
  - `results/phase1_downloaded/`
  - `results/phase2/`
  - `results/phase3/`
  - `results/phase3a/`
  - `results/phase5/`
- delete stale files before checking whether any active script or current doc still points to them
- do broad refactors of config duplication or package structure before applications / arXiv
- change prompts, model names, thresholds, personas, or phase configs tied to executed runs
- rename artifact paths that downstream scripts currently depend on

Safe changes include:

- fixing broken test imports in test files
- README / docs / AGENTS updates
- adding reproduction docs
- archiving clearly superseded files after reference checks
- marking stale files as stale rather than rewriting history
- removing empty placeholders that are not referenced

Unsafe unless deliberately blessed:

- changing source code just to make tests pass
- changing experiment logic for cleanliness or consistency alone
- regenerating outputs and treating them as equivalent without explicit re-blessing

Rule of thumb:

- if a change could alter outputs, manifests, or rerun behavior, do not do it yet
- if it only improves truthfulness, navigation, testability, or reproducibility documentation, it is probably safe
