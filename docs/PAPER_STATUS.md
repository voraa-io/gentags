# Paper Writing Status

> **Target:** ACL 2026
> **Last updated:** 2026-03-06

---

## Paper Sections → Docs Mapping

Each paper section gets its own doc in `docs/`. We write one at a time, review it, then move to the next.

| # | Paper Section | Doc File | Phase(s) | Data Ready? | Doc Status |
|---|--------------|----------|----------|-------------|------------|
| 1 | Introduction | `docs/INTRO.md` | — | — | NOT STARTED |
| 2 | Related Work | `docs/RELATED_WORK.md` | — | — | NOT STARTED |
| 3 | Method — Extraction | `docs/EXTRACTION.md` | Phase 1 | YES | **WRITTEN** |
| 4 | Method — Stability | `docs/STABILITY.md` | Phase 2 | YES | NOT STARTED |
| 5 | Method — Structure (State-Gini) | `docs/STRUCTURE.md` | Phase 3 | YES | NOT STARTED |
| 6 | Method — Decision Utility | `docs/DECISION_UTILITY.md` | Phase 5 | YES | NOT STARTED |
| 7 | Results | `docs/RESULTS.md` | All | YES | NOT STARTED |
| 8 | Discussion | `docs/DISCUSSION.md` | — | — | NOT STARTED |
| 9 | Limitations | `docs/LIMITATIONS.md` | — | — | NOT STARTED |
| 10 | Conclusion | `docs/CONCLUSION.md` | — | — | NOT STARTED |

---

## What Each Doc Needs

### 1. Introduction (`docs/INTRO.md`)
- Problem framing: constraint-sensitive decisions, no inspectable semantic state
- Related work positioning (RAG, ReAct, Generative Agents, Reflexion)
- Gentag proposal: discrete, evidence-conditioned semantic state
- Three contributions
- **Depends on:** all other sections being finalized first (write last)
- **Status:** Paper intro exists in draft form (user provided it in conversation)

### 2. Related Work (`docs/RELATED_WORK.md`)
- Text representations (TF-IDF, topic models, embeddings)
- Keyword extraction (RAKE, YAKE, TF-IDF as extraction)
- LLM-based information extraction
- Externalized state in LLM systems (RAG, ReAct, Generative Agents, Reflexion)
- Constraint satisfaction in NLP
- **Status:** NOT STARTED

### 3. Extraction (`docs/EXTRACTION.md`) — **WRITTEN**
- Definition, taxonomy, word count distribution
- All 3 prompts verbatim, system prompts, models, parsing, filtering
- Contrast with keyword baselines (side-by-side examples)
- **Needs review:** check if taxonomy analysis needs more venues beyond 50

### 4. Stability (`docs/STABILITY.md`)
- S1: Run-to-run stability (cosine=0.977, Jaccard=0.471)
- S2: Prompt sensitivity (cross-prompt cosine >0.95)
- S3: Cross-model agreement (cross-model cosine >0.94)
- S4: Evidence-induced dispersion (r=-0.230)
- Source retention (+0.164 above random)
- Surface vs. semantic decoupling
- **Data:** `results/phase2/tables/`, `results/phase2/plots/`
- **Existing doc to reference:** `docs/PHASE2_STABILITY.md` (has all numbers + plots)
- **Status:** NOT STARTED (data and plots ready)

### 5. Structure — State-Gini (`docs/STRUCTURE.md`)
- Facet assignment method (10 facets, cosine threshold, anchor embeddings)
- State-Gini: gentags 0.600 vs baselines 0.70-0.74
- Other-rate: gentags 43% vs baselines 67-68%
- Threshold robustness (τ = 0.30, 0.35, 0.40)
- Bleed check results
- **Data:** `results/phase3/tables/`, `results/phase3/plots/`
- **Existing docs:** `docs/PHASE3_STATE_GINI_PLAN.md`, `docs/PHASE3_STATUS.md`
- **Status:** NOT STARTED (data and plots ready)

### 6. Decision Utility (`docs/DECISION_UTILITY.md`)
- Experimental design (50 venues, 4 personas, 6 systems, N=5)
- Frozen indicator lexicons, strict judge prompt
- FER agreement (79.5% vs 52-62%, all p<0.001)
- Hard requirement compliance (97.3% vs 85-89%, all p<0.006)
- Token-budget ablation (truncated still beats baselines)
- Cross-judge robustness (kappa=0.712)
- Decision entropy (L1 from FER)
- **Data:** `results/phase5/`, `data/phase5/`
- **Existing doc:** `docs/phase5/BASELINE_LEGIBILITY_REPORT.md`
- **Status:** NOT STARTED (data and report ready)

### 7. Results (`docs/RESULTS.md`)
- Consolidated tables across all phases
- Key figures for the paper
- Statistical tests summary
- **Status:** NOT STARTED

### 8. Discussion (`docs/DISCUSSION.md`)
- Why propositional state beats fragments
- Floor rate vs representation fidelity (Phase 4 → Phase 5 narrative)
- Implications for LLM pipeline design
- **Status:** NOT STARTED

### 9. Limitations (`docs/LIMITATIONS.md`)
- Single domain (venue reviews)
- Gentag taxonomy not uniformly propositional (60% descriptive, 11% entity)
- Claude judge invalid rate (10.4%)
- No human gold labels yet
- No temporal/update evaluation
- **Status:** NOT STARTED

### 10. Conclusion (`docs/CONCLUSION.md`)
- Summary of contributions
- Future work (belief, uncertainty, temporal, domain applications)
- **Status:** NOT STARTED

---

## Existing Docs to Consolidate

These older docs contain useful content that should be incorporated into the new section-by-section docs:

| Existing Doc | Content | Goes Into |
|-------------|---------|-----------|
| `docs/MOTIVATION.md` | Original framing (OUTDATED — needs rewrite) | Intro |
| `docs/PHASE1_EXTRACTION.md` | Phase 1 summary (brief) | Extraction |
| `docs/PHASE2_STABILITY.md` | Full Phase 2 results + plots | Stability |
| `docs/PHASE3_STATE_GINI_PLAN.md` | State-Gini design | Structure |
| `docs/PHASE3_STATUS.md` | Phase 3 execution status | Structure |
| `docs/PHASE3_METHODOLOGY_FIX.md` | Methodology corrections | Structure |
| `docs/SECTION3_REPRESENTATION.md` | Old Section 3 framing (OUTDATED) | Extraction / Stability |
| `docs/GENTAGS_FULL_ANALYSIS_REPORT.md` | Consolidated Phase 1-3 (OUTDATED) | All method sections |
| `docs/phase4/DIR_SCALED_RUN_REPORT.md` | Phase 4 DIR results | Discussion (floor rate narrative) |
| `docs/phase5/BASELINE_LEGIBILITY_REPORT.md` | Phase 5 v2 results | Decision Utility |
| `docs/paper_problem_question_report.md` | Problem framing | Intro / Related Work |

---

## Suggested Order of Writing

1. **Extraction** — DONE
2. **Stability** — next (Phase 2, all data ready)
3. **Structure** — Phase 3 State-Gini
4. **Decision Utility** — Phase 5 legibility
5. **Results** — consolidate key tables
6. **Discussion** — interpret findings
7. **Limitations** — honest assessment
8. **Related Work** — position against literature
9. **Introduction** — frame the full story
10. **Conclusion** — wrap up

---

## Open Items

- [ ] Human gold labels (15-20 venue-persona pairs) — `data/phase5/gold_labels_manual.json`
- [ ] Update `docs/MOTIVATION.md` to align with paper intro
- [ ] Decide whether Phase 4 DIR results go in the paper or appendix
- [ ] Finalize figure numbering across sections
