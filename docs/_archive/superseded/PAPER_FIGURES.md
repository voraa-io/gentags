# Paper Figures — ACL 2026

> **Last updated:** 2026-03-27
> **Principle:** 7 main figures. Tight narrative: what → stable → structured → useful.

---

## Main Paper Figures

### Fig 1: Pipeline Diagram (Intro)
- **File:** `results/figures/fig1_pipeline.png`
- **Claim:** What gentags are — LLM-extracted propositional state from reviews.
- **Why keep:** Readers need a fast mental model. Without this, "semantic state" stays abstract.

![Pipeline](../results/figures/fig1_pipeline.png)

---

### Fig 2: Surface vs Semantic Decoupling (Section 5.1.1)
- **File:** `results/phase2/plots/6_surface_vs_semantic.png`
- **Claim:** Gentags are semantically stable despite lexical variation.
- **Why keep:** Most visually compelling stability plot. Upper-left clustering = high cosine, low Jaccard. One figure communicates the entire decoupling story.

![Surface vs Semantic](../results/phase2/plots/6_surface_vs_semantic.png)

---

### Fig 3: Run-to-Run Stability (Section 5.1.1)
- **File:** `results/phase2/plots/1_run_stability.png`
- **Claim:** Repeated extraction recovers the same semantic state.
- **Why keep:** ECDF + boxplot showing median cosine 0.977 across 4 models. Supports interpretation of gentags as recoverable intermediate state.

![Run Stability](../results/phase2/plots/1_run_stability.png)

---

### Fig 4: Evidence-Induced Dispersion (Section 5.1.4)
- **File:** `results/phase2/plots/7_sparsity_analysis.png`
- **Claim:** Variation behaves systematically — sparse evidence produces more dispersed states.
- **Why keep:** Shows representation variation is meaningful, not random noise. r=-0.230 trend + token-bucket boxplots. Important nuance for reviewers.

![Sparsity Analysis](../results/phase2/plots/7_sparsity_analysis.png)

---

### Fig 5: State-Gini + Facet Coverage (Section 5.2)
- **File:** `results/phase3/plots/1_gini_and_coverage.png`
- **Claim:** Gentags yield broader, more balanced semantic state than lexical baselines.
- **Why keep:** Core structural evidence. Gini 0.600 vs 0.70–0.74, other-rate 43% vs 67–68%. Joint interpretation is the structural argument.

![State-Gini + Coverage](../results/phase3/plots/1_gini_and_coverage.png)

---

### Fig 6: FER Agreement + Constraint Compliance (Section 5.3.1–5.3.2)
- **File:** `results/phase5/plots/1_fer_agreement_and_compliance.png`
- **Claim:** Gentags preserve decisions and satisfy constraints better than baselines.
- **Why keep:** Most important figure. 79.5% vs 52–62% agreement (all p<0.001). 97.3% constraint compliance. P3 persona shows clearest separation.

![FER Agreement + Compliance](../results/phase5/plots/1_fer_agreement_and_compliance.png)

---

### Fig 7: Decision Distribution vs FER (Section 5.3.5)
- **File:** `results/phase5/plots/2_decision_distribution.png`
- **Claim:** Baselines don't just err more — they induce BORDERLINE bloat.
- **Why keep:** Shows mechanism. TF-IDF 36% BORDERLINE vs FER 12.5%. Without this, paper looks like pure accuracy comparison. L1 distances confirm gentags track FER closest.

![Decision Distribution](../results/phase5/plots/2_decision_distribution.png)

---

**Narrative flow:**
1. **Fig 1** — What gentags are
2. **Fig 2** — Gentags stable despite paraphrase
3. **Fig 3** — Gentags stable across runs
4. **Fig 4** — Variation behaves meaningfully with evidence
5. **Fig 5** — Gentags structured differently than lexical baselines
6. **Fig 6** — Gentags produce better decisions
7. **Fig 7** — Gentags preserve decision distribution structure

---

## Appendix Figures

| Fig | File | Description |
|-----|------|-------------|
| A1 | `results/phase2/plots/2_prompt_sensitivity.png` | Cross-prompt similarity heatmaps |
| A2 | `results/phase2/plots/3_model_sensitivity.png` | Cross-model similarity heatmaps |
| A3 | `results/phase2/plots/4_retention.png` | Source retention by model/prompt |
| A4 | `results/phase2/plots/5_cost_effectiveness.png` | Cost-effectiveness Pareto front |
| A5 | `results/phase3/plots/2_threshold_sensitivity.png` | Tau sweep {0.30, 0.35, 0.40} |
| A6 | `results/phase3/plots/3_facet_heatmap.png` | Per-facet distribution by method |
| A7 | `results/phase5/plots/3_cross_judge_kappa.png` | Cross-judge kappa by system |
| A8 | `results/phase5/plots/4_failure_audit.png` | Failure audit breakdown |
| A9 | `results/phase3b/plots/1_gini_comparison.png` | Drift Gini under evidence rewording |
| A10 | `results/phase3b/plots/3_paraphraser_consistency.png` | Paraphraser method consistency |

---

## Removed

| File | Reason |
|------|--------|
| `results/phase3b/plots/2_mmc_comparison.png` | Single-bar, one number, no comparison — text sufficient |
| `results/phase3/plots/` (old, deleted) | Wrong baselines ("Embeddings") |
| `results/phase3_archived_circular/` (deleted) | Circular methodology |

---

## Generation Scripts

| Script | Generates | Command |
|--------|-----------|---------|
| `scripts/phase2_plots.py` | Fig 2–4, A1–A4 | `poetry run python scripts/phase2_plots.py` |
| `scripts/phase3_plots.py` | Fig 5, A5–A6 | `poetry run python scripts/phase3_plots.py` |
| `scripts/phase5_plots.py` | Fig 6–7, A7–A8 | `poetry run python scripts/phase5_plots.py` |
| `scripts/pipeline_diagram.py` | Fig 1 | `poetry run python scripts/pipeline_diagram.py` |

---

## Remaining

- [x] Create pipeline diagram (Fig 1)
- [ ] Final figure numbering after all sections written
