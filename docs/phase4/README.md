# Phase 4 docs

This folder holds **plans, protocols, and run logs** for Phase 4 experiments. Use it to see what we plan to run, what decisions are pending, and (later) what we got.

**Phase 4 goal:** Prove gentags are **actionable** as a state object via CheckList-style behavioral testing (DIR + INV).

- **PHASE4_PLAN.md** --- Rationale, constraints, capability matrix, inputs, decisions, execution order, success criteria, and paper narrative.
- **PHASE4_EXECUTION_SPEC.md** --- **Strict runnable spec.** Exact Judge prompt, state construction, execution loop, aggregation (majority vote N=5), DIR/INV pass rules, placebo check, attribution check, output metrics. No prose --- this is the machine-readable protocol.
- **PHASE4_PRERUN_CHECKLIST.md** --- Operational checklist. Every box must be checked before execution. Covers: experimental freeze, persona lock, representation freeze, intervention catalog, Judge prompt, aggregation rules, metrics, logging, sample size, failure criteria.
- **sample_venue_test_design.md** --- Test design worksheet for Colton's - Monterrey (sparse, 176 tokens). Full 5-persona / 10-DIR design (reference only; MVP uses a subset).
- *(run logs will be added here as experiments are executed)*

**MVP data artifacts (frozen before execution):**

- `data/phase4/mvp_config.json` --- Experimental constants: venue, gentag source (openai run1), baseline (RAKE), N=5, canonicalization, RNG seed. Judge model TBD.
- `data/phase4/mvp_personas.json` --- 3 personas: Food Critic (food_quality), Sports Fan (ambiance), Quick Lunch Worker (service). Each with hard requirement.
- `data/phase4/mvp_dir_units.json` --- 8 DIR units on gentags: 3 food, 2 service, 2 atmosphere, 1 stress test. Each with placebo.
- `data/phase4/mvp_dir_units_rake.json` --- Same 8 DIR units on RAKE baseline for comparison.
- `results/phase4/sample_venue.json` --- Source data: gentags (all 4 models, run1+run2), RAKE/TF-IDF/YAKE keywords, raw reviews.

**Runner script:**

- `scripts/phase4_dir_runner.py` --- Executes DIR experiment per `PHASE4_EXECUTION_SPEC.md`. Tracks time and cost per call. Saves all raw responses persistently.
  - `--dry-run` to validate states without API calls
  - `--units` to specify gentag or RAKE unit file
  - Outputs to `results/phase4/`:
    - `dir_results_<run_id>.json` — every run, every raw response, every parsed JSON
    - `dir_manifest_<run_id>.json` — metrics, cost, timing, config snapshot
    - `dir_summary_<run_id>.json` — one row per unit for quick inspection

**Depends on:**
- Phase 2 stability numbers (cosine 0.977, Jaccard 0.471)
- Phase 3 State-Gini + follow-ups (coverage gap, bleed check, Other-bucket probe)
- Phase 1 extractions and Phase 2 tag embeddings
