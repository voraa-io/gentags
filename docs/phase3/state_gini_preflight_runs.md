# State-Gini preflight run log

What we ran, what we got, and **whether we changed anything** (anchors, τ, etc.).

---

## Run 1 — 2025-02-08

**Command:**
```bash
poetry run python scripts/state_gini_preflight.py --orthogonality --tau-sweep --sample 10
```

**Artifacts:** `results/phase3/preflight_orthogonality.json`, `results/phase3/preflight_tau_sweep.json`

### Orthogonality

| Result | Value |
|--------|--------|
| Max pairwise cosine (anchors) | **0.5817** |
| Pair | food_quality / service |
| Pairs above 0.60 | **0** (none) |

**Decision:** No change. Max 0.5817 is below 0.60, so we did **not** refine anchor phrases. Anchors stay as in `../_archive/superseded/PHASE3_STATE_GINI_PLAN.md`.

### Tau sweep (10 venues, 240 extractions)

Preflight does **not** compute State-Gini — only how many tags fall below each τ (other_rate).

| τ   | n_extractions | mean_other_rate |
|-----|----------------|-----------------|
| 0.30 | 240 | 29.6% |
| 0.35 | 240 | 42.0% |
| 0.40 | 240 | 53.8% |

**Decision:** Preflight only checked other_rate (no State-Gini). We can run the **real experiment** at τ = 0.30, 0.35, and 0.40 (sensitivity). Default in code is 0.35; change SEMANTIC_THRESHOLD to run the full experiment at each τ for sensitivity. Higher τ gives more tags in the other bucket.

### Summary

- **Anchors:** unchanged  
- **τ:** Use 0.30, 0.35, 0.40 in the real experiment as needed (sensitivity); default 0.35 in code.  
- **Full State-Gini:** **NOT run yet.** Preflight confirmed orthogonality and τ range. When ready, run: `state_gini_full.py` then `phase3a_baselines.py` (or `./scripts/run_phase3.sh`). You can run the full experiment at each of τ ∈ {0.30, 0.35, 0.40} to report State-Gini by threshold.

---

## Later runs

Add new runs below in the same format: command, orthogonality result + decision, tau-sweep result + decision, and whether we changed anything.
