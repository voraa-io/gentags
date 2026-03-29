# Results

> **Paper section:** Section 5
> **Status:** Draft
> **Last updated:** 2026-03-07
> **Source discipline:** Numeric claims in this draft were checked against phase-specific docs and verified artifacts.

---

## 5. Analysis / Results

The empirical results are organized around the three claims of the paper. First, if gentags are to function as an externalized semantic state, they must be recoverable across repeated extractions, prompts, and extractor models. Second, if they are to be more than a stylistic rewriting of reviews, they must exhibit a useful structural organization relative to lexical baselines. Third, if that structure matters in practice, it should improve downstream constrained decisions relative to fragment-level keyword representations.

The studies therefore proceed in three layers. **Phase 2** evaluates representation stability. **Phase 3** evaluates structural organization through facet coverage and State-Gini. **Phase 5** evaluates downstream decision utility under explicit hard constraints. Across all three layers, gentags are compared against text-derived alternatives under matched protocols rather than against unrelated systems or tasks.

For the decision study, each condition consists of a **venue**, a **persona**, and a **representation**. A judge model receives only the supplied representation and produces a decision. This allows the downstream evaluation to isolate the information carried by the representation itself.

## 5.1 Stability Analysis

The stability results ask whether gentags behave like a recoverable semantic state rather than a brittle surface artifact. The core empirical pattern is the same across all stability tests: wording changes substantially, but meaning remains highly stable.

### 5.1.1 Run-to-run Stability

The most basic question is whether repeated extractions under the same model and prompt recover the same semantic content. Across repeated runs, gentags show very high semantic consistency despite substantial lexical variation.

| Metric | Median | Q1 | Q3 |
|--------|--------|----|----|
| Semantic cosine | **0.977** | 0.968 | 0.986 |
| Surface Jaccard | **0.471** | 0.333 | 0.625 |
| Mean Max Cosine (semantic paraphrase consistency) | **0.887** | 0.839 | 0.927 |
| Semantic-surface gap | **0.504** | — | — |

Two things matter in this table. First, semantic cosine is extremely high: repeated runs recover nearly the same point in semantic space. Second, surface overlap is much lower. The gap between these two metrics shows that a large share of the variation lies in paraphrase, compression, and lexical choice rather than in semantic drift (Figure 2).

![Figure 2: Surface vs Semantic Decoupling](../results/phase2/plots/6_surface_vs_semantic.png)

This pattern is also visible by model. Claude and Grok produce the highest surface overlap, while Gemini and OpenAI show more paraphrastic variation. But all four models remain in the same basic regime: high semantic similarity and clearly lower lexical overlap.

| Model | Cosine | Jaccard | Mean Max Cosine |
|-------|--------|---------|-----|
| Claude | 0.982 | 0.574 | 0.913 |
| Gemini | 0.971 | 0.404 | 0.869 |
| Grok | 0.975 | 0.722 | 0.876 |
| OpenAI | 0.975 | 0.387 | 0.861 |

The implication is straightforward. Gentags do not require exact lexical reproducibility to be stable. They are stable in the stronger sense relevant for representation: repeated extraction recovers substantially the same meaning even when the tag strings are not identical (Figure 3).

![Figure 3: Run-to-Run Stability](../results/phase2/plots/1_run_stability.png)

### 5.1.2 Prompt Sensitivity

The next question is whether prompt wording changes the recovered semantic state or merely changes its resolution and style. Across all prompt pairs, semantic similarity remains high.

| Prompt Pair | Mean Cosine | Mean Jaccard |
|-------------|-------------|--------------|
| anti_hallucination ↔ minimal | **0.966** | 0.321 |
| anti_hallucination ↔ short_phrase | **0.962** | 0.282 |
| minimal ↔ short_phrase | **0.966** | 0.352 |

This result shows that prompt variation affects how the state is phrased and compressed, but not the core semantic content it recovers. The `anti_hallucination` prompt tends to produce more grounded and granular tags; `short_phrase` tends to compress them; `minimal` sits between the two. Yet the cross-prompt cosine remains above 0.95 throughout.

That matters methodologically. A representation that only exists under a single fragile prompt would be hard to defend as a reusable state abstraction. Gentags are more robust than that. Prompt changes alter surface form and granularity, but the same underlying venue semantics remain recoverable.

A second prompt-level pattern also emerges from the run-stability summaries. Across all four extractor models, the `anti_hallucination` prompt produces the **highest rerun Jaccard overlap**, and it also produces the highest cosine and Mean Max Cosine values. However, these should not be read as evidence that `anti_hallucination` yields a materially different semantic state. The absolute semantic gains are small: relative to `minimal`, cosine improves by only about **0.007-0.009** across models, and relative to `short_phrase` by about **0.007-0.013**. Mean Max Cosine increases somewhat more, by roughly **0.026-0.045**, but all prompts remain in the same high-semantic-similarity regime. By contrast, Jaccard increases much more visibly, by roughly **0.044-0.141**.

Paired Wilcoxon tests confirm this asymmetry. For all four models, `anti_hallucination` is significantly higher than the other prompts on rerun Jaccard, and in most cases also significantly higher on cosine and Mean Max Cosine. But because the number of venues is large, statistical significance alone would overstate the semantic effect. The defensible interpretation is therefore practical rather than purely statistical: stronger anti-hallucination instructions make the extracted **surface form** more repeatable, while the recovered **semantic content** is already highly stable across prompts.

### 5.1.3 Cross-model Agreement

The strongest recoverability test is cross-model agreement. If multiple extractor models recover closely aligned gentag states from the same evidence, the representation is harder to dismiss as a model-specific artifact.

Across model pairs, semantic agreement remains high:

| Model Pair | Mean Cosine | Mean Jaccard |
|------------|-------------|--------------|
| Claude ↔ Gemini | 0.951 | 0.253 |
| Claude ↔ Grok | 0.953 | 0.267 |
| Claude ↔ OpenAI | 0.951 | 0.236 |
| Gemini ↔ Grok | 0.969 | 0.323 |
| Gemini ↔ OpenAI | 0.958 | 0.248 |
| Grok ↔ OpenAI | 0.969 | 0.315 |

All pairwise semantic similarities exceed **0.94**, while lexical overlap remains much lower. This is exactly the pattern one would expect if the evidence constrains a shared semantic state but leaves freedom in phrasing and packaging.

The important conclusion here is representation-level stability. The extractor can change, and the wording can change, yet the recovered semantic object remains similar. That is the right kind of robustness for an externalized state representation.

### 5.1.4 Evidence-induced Dispersion

The final stability question is whether variation behaves meaningfully with respect to evidence. If gentag states are constrained by the source text, sparse evidence should produce less identifiable states and therefore greater dispersion.

That is what the data show. The correlation between evidence quantity and representation dispersion is **-0.230** by Pearson correlation, and this relationship is statistically significant (**p = 0.00045**). A rank-based Spearman check yields a similar result (**rho = -0.263, p = 5.4e-05**), indicating that the negative association is not an artifact of a particular linearity assumption. Lower-token venues show higher mean pairwise distance among recovered states, while better-evidenced venues show lower dispersion.

| Token Bucket | Mean Variability | N Venues |
|--------------|------------------|----------|
| <200 | **0.0568** | 104 |
| 200-400 | 0.0465 | 87 |
| 400-600 | 0.0454 | 29 |
| 600-1000 | 0.0462 | 9 |
| >1000 | **0.0424** | 1 |

The practical size of the effect is modest but meaningful. Venues under 200 tokens have mean variability **0.0568**, compared with **0.0465** for the 200-400 token bucket and **0.0455** for venues with at least 400 tokens. The gap between `<200` and `>=400` is therefore about **0.0113** in absolute terms, or roughly **25%** relative to the higher-evidence group, and this bucket comparison is also significant under a Mann-Whitney test (**p = 0.0339**). By contrast, the comparison against only the very highest-token venues is not stable enough to stand alone, because that bucket is small. The correct interpretation is therefore not a dramatic step change at a single threshold, but a statistically reliable tendency for sparse-evidence venues to exhibit more dispersed recovered states.

This result is conceptually important. Dispersion is not just model noise. It tracks how strongly the evidence constrains the semantic state. Under sparse evidence, multiple plausible gentag states can be recovered; under richer evidence, the state becomes more identifiable. That makes dispersion interpretable as an identifiability signal rather than as arbitrary stochastic failure (Figure 4).

![Figure 4: Evidence-Induced Dispersion](../results/phase2/plots/7_sparsity_analysis.png)

Taken together, the Phase 2 results support the first main claim of the paper: gentags are recoverable enough to function as an externalized semantic state. They are not lexically fixed, but they are semantically stable across reruns, prompts, and extractor models.

## 5.2 Structural Analysis

The structural analysis asks a different question: what kind of semantic state do gentags form? The goal is not merely to show that gentags are stable, but to determine whether they organize semantic content in a more useful way than lexical baselines.

The main probe projects tags or keywords into a shared 10-facet diagnostic space using frozen anchor embeddings and hard assignment with threshold `τ = 0.35`. Items that fail threshold are placed in an explicit `other` bucket. This detail is central: structural claims must be interpreted jointly through **facet coverage** and **State-Gini**, because a representation can appear highly concentrated simply by failing to place much of its semantic mass into the measured facet space.

### 5.2.1 Facet Coverage

Facet coverage is measured through the fraction of semantic units routed to the explicit `other` bucket. Lower `other_rate` means more semantic mass is captured by the diagnostic facet inventory.

In the executed Phase 3 run, gentags show substantially better facet coverage than lexical baselines:

| Method | Mean tags/keywords per unit | Assigned mean | Other mean | Other rate |
|--------|-----------------------------|---------------|------------|------------|
| Gentags | 21.9 | 12.3 | 9.6 | **~43%** |
| RAKE | 19.5 | 6.1 | 13.3 | **~68%** |
| TF-IDF | 19.8 | 6.6 | 13.2 | **~67%** |
| YAKE | 19.8 | 6.5 | 13.3 | **~67%** |

This is the cleanest structural difference between gentags and the lexical baselines. Gentags place a much larger share of their mass into the diagnostic facet space, while the keyword baselines leave most of their mass below threshold. In practical terms, gentags recover more units that map onto recognizable decision-relevant aspects of the venue, whereas lexical baselines generate many fragments that are too noisy, too local, or too semantically incomplete to survive thresholding.

This matters because downstream decisions do not operate over all possible text fragments equally. A representation is more useful when more of its mass is interpretable in the relevant semantic space. On that criterion, gentags outperform RAKE, YAKE, and TF-IDF.

### 5.2.2 State-Gini

State-Gini measures how concentrated the **assigned** semantic mass is across the 10 facets. On its own, high Gini means that assigned items pile into fewer facets; low Gini means that assigned items are spread more evenly across facets.

The raw State-Gini results are:

| Method | Mean State-Gini | Std. Dev. |
|--------|------------------|-----------|
| Gentags | **0.600** | 0.127 |
| RAKE | 0.701 | 0.140 |
| TF-IDF | 0.715 | 0.116 |
| YAKE | 0.738 | 0.150 |

At first glance, these numbers could be misread as favoring the baselines, since their Gini values are higher. But interpreted in isolation, that reading would be wrong. The baselines achieve their higher Gini while assigning far fewer units to the facet space at all. Because Gini is computed only on the assigned subset, a method with a very high `other_rate` can appear more concentrated simply because most of its mass has already been excluded.

This is exactly what happens here. RAKE, YAKE, and TF-IDF each assign only about 6 items on average, while gentags assign more than 12. With so few surviving items, baseline mass naturally looks spikier. Their higher Gini is therefore partly an artifact of low coverage.

The right interpretation is joint:

- **Gentags:** lower `other_rate`, lower Gini
- **Baselines:** higher `other_rate`, higher Gini

That combination indicates that gentags capture more semantic mass inside the facet inventory and distribute that captured mass across more decision-relevant dimensions. In contrast, the lexical baselines leave most of their mass outside the measured semantic space and concentrate what remains in a few surviving facets. This is better described as **spiky partial coverage** than as a superior structured state (Figure 5).

![Figure 5: State-Gini + Facet Coverage](../results/phase3/plots/1_gini_and_coverage.png)

The structural claim, then, is not that gentags are more single-facet concentrated than the baselines. It is that gentags produce a **broader, more balanced, and more semantically covered state** than fragment-level lexical baselines.

### 5.2.3 Threshold Sensitivity

To test whether the structural pattern is an artifact of a particular threshold, the facet-assignment procedure was rerun at `τ ∈ {0.30, 0.35, 0.40}`.

| τ | Method | State-Gini | Other rate (%) |
|---|--------|------------|----------------|
| 0.30 | Gentags | 0.575 | **26.6** |
| 0.30 | RAKE | 0.647 | 50.9 |
| 0.30 | TF-IDF | 0.689 | 50.5 |
| 0.30 | YAKE | 0.710 | 50.2 |
| 0.35 | Gentags | 0.600 | **42.6** |
| 0.35 | RAKE | 0.701 | 68.1 |
| 0.35 | TF-IDF | 0.715 | 66.7 |
| 0.35 | YAKE | 0.738 | 67.1 |
| 0.40 | Gentags | 0.630 | **55.1** |
| 0.40 | RAKE | 0.733 | 78.0 |
| 0.40 | TF-IDF | 0.774 | 80.2 |
| 0.40 | YAKE | 0.770 | 79.6 |

The important result is not the exact Gini movement but the robustness of the coverage gap. At every threshold, gentags retain more mass within the facet inventory than the lexical baselines. At `τ = 0.30`, gentags assign roughly three quarters of their mass, while the baselines assign only about half. At `τ = 0.40`, all methods lose coverage, but the baselines deteriorate much more severely.

This threshold sweep strengthens the structural interpretation. The gentag advantage in facet coverage is not a fragile byproduct of `τ = 0.35`; it persists across a reasonable range of thresholds (see Appendix Figures A5–A6). The same is true of the underlying structural asymmetry: gentags retain broader semantic coverage, while lexical baselines become increasingly sparse and spiky as the threshold tightens.

The Phase 3 results therefore support a structural claim narrower than the original preregistered hope but still substantive: gentags yield a better-covered and more balanced semantic state than lexical fragment baselines. That is the form of structure most relevant for downstream decision tasks.

### 5.2.4 Bleed Check

Because Phase 3 uses hard argmax assignment, a natural concern is whether many items are only weakly assigned to their winning facet. To test this, we examine the gap between the highest and second-highest facet similarities. If argmax assignments were generally clean, these gaps would be large. If many items sit near facet boundaries, the gaps would be small.

For gentags, the bleed-check results show that facet boundaries are often soft rather than sharply separated:

| Gentag metric | Value |
|---------------|-------|
| Mean primary-secondary gap | **0.065** |
| Median gap | **0.039** |
| Near-miss rate (`gap < 0.05`) | **57.4%** |
| Clear-primary rate (`gap >= 0.10`) | **20.5%** |
| Mean primary similarity | **0.343** |

These numbers mean that a majority of gentags have a relatively small margin between the best and second-best facet. In other words, many tags are not naturally one-facet objects. They sit near multiple diagnostic axes in embedding space. Only about one fifth have a clearly dominant facet under the current criterion.

The same diagnostic can also be computed for the lexical baselines:

| Method | Mean gap | Median gap | `gap < 0.05` | `gap >= 0.10` | Mean primary sim |
|--------|----------|------------|--------------|---------------|------------------|
| Gentags | **0.065** | **0.039** | **57.4%** | **20.5%** | **0.343** |
| RAKE | 0.056 | 0.030 | 64.0% | 18.0% | 0.319 |
| YAKE | 0.056 | 0.030 | 64.7% | 18.1% | 0.318 |
| TF-IDF | 0.055 | 0.025 | 66.9% | 17.4% | 0.321 |

This comparison is useful for two reasons. First, it shows that gentags are not uniquely ambiguous; in fact, the lexical baselines are slightly **more** boundary-ambiguous by this diagnostic. Second, it weakens a possible objection to the gentag structural story. The lower gentag State-Gini is not explained by unusually noisy argmax assignments relative to the baselines. If anything, the baselines exhibit smaller top-two margins and more near-miss assignments.

This does not turn the facet inventory into a clean ontology. All methods show soft boundaries. But it does clarify how to read Phase 3. The facets are best understood as a **diagnostic probe space**, and gentags perform at least as well as the baselines under that probe while also achieving much better facet coverage. That makes the joint reporting of `other_rate`, State-Gini, and gap-based ambiguity important for a faithful interpretation of the structural results.

## 5.3 Decision Evaluation

The decision study tests whether the representational differences observed in the structural analysis matter under explicit downstream constraints. This is the strongest practical test in the paper: if gentags are a better semantic state, they should better preserve full-evidence decisions and satisfy hard requirements more reliably than fragment-level lexical baselines.

The Phase 5 evaluation uses **50 venues**, **4 personas**, and **6 systems**, yielding **1,200 conditions** per judge. The systems are `gentag`, `rake`, `yake`, `tfidf`, `gentag_truncated`, and `fer`. Each condition is evaluated with **N = 5** repeated judge calls and majority-vote aggregation.

### 5.3.1 FER Agreement

The primary fidelity metric is agreement with **Full-Evidence Reference (FER)** decisions. FER uses the same judge, the same decision rubric, and the same aggregation procedure, but supplies raw reviews rather than a compressed representation. It therefore serves as a reference decision under full evidence, not as a separate task.

Gentags substantially outperform all lexical baselines on FER agreement:

| System | Matches | Total | Agreement | Kappa |
|--------|---------|-------|-----------|-------|
| Gentag | 159 | 200 | **79.5%** | **0.667** |
| RAKE | 122 | 198 | 61.6% | 0.388 |
| YAKE | 117 | 200 | 58.5% | 0.351 |
| TF-IDF | 104 | 199 | 52.3% | 0.258 |
| Gentag truncated | 149 | 199 | 74.9% | 0.596 |

The pairwise Fisher tests all favor gentags over the lexical baselines:

| Comparison | p-value |
|-----------|---------|
| gentag vs RAKE | **0.0001** |
| gentag vs YAKE | **0.000008** |
| gentag vs TF-IDF | **<0.0001** |

These are large gaps, not marginal ones. Gentags exceed RAKE by nearly 18 percentage points, YAKE by 21 points, and TF-IDF by more than 27 points. The kappa results tell the same story. Gentags show substantial agreement with FER, while the lexical baselines fall into the fair-agreement range.

This is the paper's clearest decision-fidelity result: gentags preserve far more of the decision-relevant content available under full evidence than keyword fragments do (Figure 6, left panel).

![Figure 6: FER Agreement + Constraint Compliance](../results/phase5/plots/1_fer_agreement_and_compliance.png)

A concrete example from the actual evaluation data helps make this metric intuitive. For **Coltons's Arcadia** under **P3 Quick Lunch Worker**, the full-evidence reference decision is `RECOMMEND`. The gentag representation also yields `RECOMMEND`, matching FER. The gentags for this venue include explicit service signals such as `"fast service"`, `"attentive service"`, `"amazing service"`, and `"excellent service"`. By contrast, the RAKE, YAKE, and TF-IDF representations all yield `REJECT` for the same venue-persona condition. In this case the representation-only decision diverges because the lexical baselines do not preserve the speed signal as clearly or directly as the gentag state does. This is exactly what FER agreement measures: whether the compressed representation leads to the same decision that the judge reaches when given the original reviews.

### 5.3.2 Constraint Compliance

The second main decision metric is hard-constraint compliance. This directly measures whether a representation enables correct decisions under explicit persona requirements.

Combined compliance across the three hard personas is:

| System | Correct | Total | Compliance |
|--------|---------|-------|------------|
| Gentag | 146 | 150 | **97.3%** |
| FER | 142 | 150 | 94.7% |
| Gentag truncated | 141 | 149 | 94.6% |
| RAKE | 133 | 149 | 89.3% |
| TF-IDF | 129 | 150 | 86.0% |
| YAKE | 127 | 150 | 84.7% |

The associated Fisher tests again favor gentags:

| Comparison | p-value |
|-----------|---------|
| gentag vs RAKE | **0.0054** |
| gentag vs YAKE | **0.0002** |
| gentag vs TF-IDF | **0.0006** |

Per-persona breakdown clarifies where the difference comes from. P1 is effectively a control and does not differentiate systems, because the sample contains no negative food indicators. The real separation comes from P2 and especially P3.

| Persona | Gentag | RAKE | YAKE | TF-IDF |
|---------|--------|------|------|--------|
| P2 Sports Fan | **96.0%** | 91.8% | 88.0% | 88.0% |
| P3 Quick Lunch Worker | **96.0%** | 76.0% | 66.0% | 70.0% |

P3 is the clearest case. Gentag phrases such as `"fast service"` and `"quick counter service"` communicate the relevant speed signal directly. Lexical fragments such as `"relative quick time"` or `"quick lunch"` are much less decision-legible. The representation does not merely lose nuance; it fails to carry the exact constraint signal the judge needs.

This is the downstream version of the structural story. Gentags better preserve semantically actionable facets, and as a result they better preserve hard-constraint decisions (Figure 6, right panel).

### 5.3.3 Token-budget Ablation

One possible objection is that gentags might simply work better because they provide more information. The token-budget ablation addresses this by truncating gentags to match the RAKE tag count for each venue.

Even under this matched-budget condition, gentags retain a large advantage:

| Metric | Gentag truncated | RAKE | YAKE | TF-IDF |
|--------|------------------|------|------|--------|
| FER agreement | **74.9%** | 61.6% | 58.5% | 52.3% |
| Combined compliance | **94.6%** | 89.3% | 84.7% | 86.0% |

At the level of floor rate, the ablation behaves as intended: once tag count is matched, floor rates are nearly identical. But the fidelity and compliance gaps remain. This is the crucial point. Matching the information budget does not erase the gentag advantage, which means the advantage cannot be reduced to simple verbosity.

The ablation therefore isolates the actual representational effect. Gentags work better because of **what they say**, not just because of **how much they say**.

### 5.3.4 Cross-judge Agreement

To test robustness to evaluator choice, the entire Phase 5 study was rerun with a second judge model. The cross-judge results show substantial overall agreement.

| System | Matches | Total | Agreement | Kappa |
|--------|---------|-------|-----------|-------|
| FER | 167 | 199 | 83.9% | 0.731 |
| Gentag | 147 | 176 | 83.5% | 0.746 |
| Gentag truncated | 144 | 173 | 83.2% | 0.744 |
| RAKE | 160 | 191 | 83.8% | 0.744 |
| TF-IDF | 135 | 178 | 75.8% | 0.643 |
| YAKE | 137 | 177 | 77.4% | 0.660 |
| Overall | 890 | 1094 | **81.3%** | **0.712** |

The overall kappa of **0.712** indicates substantial agreement between judges. The pattern across systems is also informative. Gentag, FER, and gentag_truncated remain in the most stable region. TF-IDF and YAKE show lower judge agreement, which is consistent with the broader interpretation that more fragmentary representations are harder to interpret consistently.

This matters because it rules out a narrow evaluator artifact. The main decision result is not specific to one judge model (see Appendix Figure A8).

<!-- ![Figure A8: Cross-Judge Kappa](../results/phase5/plots/3_cross_judge_kappa.png) -->

### 5.3.5 Decision Entropy

Decision entropy provides a complementary view of representational quality. A useful representation should not only be correct more often; it should also lead to more coherent and decisive downstream behavior.

The entropy analysis shows that gentag decisions track FER much more closely than the lexical baselines:

| System | H (bits) | P(REJECT) | P(BORDERLINE) | P(RECOMMEND) | L1 vs FER |
|--------|----------|-----------|---------------|--------------|-----------|
| FER | 1.393 | 52.5% | 12.5% | 35.0% | — |
| Gentag | 1.506 | 49.0% | 23.0% | 28.0% | **0.210** |
| Gentag truncated | 1.520 | 47.7% | 24.6% | 27.6% | 0.242 |
| RAKE | 1.500 | 49.5% | 28.3% | 22.2% | 0.316 |
| YAKE | 1.460 | 49.5% | 34.0% | 16.5% | 0.430 |
| TF-IDF | 1.430 | 49.8% | 36.2% | 14.1% | 0.474 |

The lexical baselines systematically shift probability mass away from `RECOMMEND` and into `BORDERLINE`. That pattern is exactly what one would expect from semantically incomplete or opaque representations: they do not only make more mistakes, they also induce more uncertainty. Gentags remain much closer to the full-evidence decision distribution.

This entropy result is important because it shows that gentags improve not just accuracy-like metrics but the overall decisional shape of downstream inference (Figure 7).

![Figure 7: Decision Distribution by System vs FER](../results/phase5/plots/2_decision_distribution.png)

### 5.3.6 Qualitative Failure Audit

The aggregate Phase 5 metrics establish that gentags preserve downstream decisions better than lexical baselines. A separate question is what remains in the disagreement set once that aggregate advantage is established. To answer that, we audited all conditions where the Gentag decision disagrees with the Full-Evidence Reference.

Across the 200 Gentag conditions, there are **41** Gentag-FER mismatches:

| FER -> Gentag | Count |
|--------|------:|
| RECOMMEND -> BORDERLINE | 18 |
| REJECT -> BORDERLINE | 9 |
| BORDERLINE -> RECOMMEND | 6 |
| RECOMMEND -> REJECT | 5 |
| REJECT -> RECOMMEND | 3 |

The dominant pattern is one-step drift rather than complete reversal. Of the 41 mismatches, **33** (`80.5%`) pass through `BORDERLINE`, while only **8** (`19.5%`) are direct `REJECT <-> RECOMMEND` flips. The disagreements are also concentrated rather than uniform: **24/41** occur for `P1`, **11/41** for `P4`, **5/41** for `P3`, and only **1/41** for `P2`.

This concentration already suggests that the remaining errors are structured. Most occur in mixed-evidence settings where a compressed propositional state must preserve both positive and negative cues, rather than in clean single-signal cases. The exact reversals form a much smaller subset.

The appendix audit further shows that these exact reversals are not dominated by unsupported or fabricated tags. In the four reversal cases where the Gentag judge cited non-empty `tags_used`, those cited tags are grounded in the source reviews on manual inspection. In the other four exact reversals, the Gentag judge cited no tags at all and rejected because no exact hard-indicator match was recognized. Appendix A provides the full failure-mode taxonomy, representative cases, and grounding notes.

## 5.4 Overall Interpretation

Across all three empirical layers, the same picture emerges.

Phase 2 shows that gentags are semantically stable across reruns, prompt variants, and extractor models. This establishes that they are recoverable enough to function as an externalized semantic state.

Phase 3 shows that gentags place more semantic mass into a shared diagnostic facet space than lexical baselines do. Although the lexical baselines show higher raw Gini, that higher concentration is paired with much worse facet coverage. The right structural interpretation is therefore joint: gentags yield a better-covered and more balanced semantic state, while lexical baselines are sparse and spiky.

Phase 5 shows that this representational advantage transfers downstream. Gentags agree more often with full-evidence decisions, satisfy hard constraints more reliably, retain their advantage under token-matched ablation, and produce decision distributions closer to the full-evidence reference. Cross-judge agreement further indicates that the effect is not an artifact of a single evaluator.

Taken together, the results support the central claim of the paper: **discrete, evidence-conditioned semantic state improves constraint-sensitive decision reliability relative to fragment-level lexical baselines.**

---

## File References

- Stability source: `docs/PHASE2_STABILITY.md`
- Structural source: `docs/phase3/state_gini_full_run_analysis.md`
- Decision source: `docs/phase5/BASELINE_LEGIBILITY_REPORT.md`
- Verified Phase 5 analysis: `results/phase5/baseline_legibility_analysis.json`
- Baseline bleed check: `results/phase3/baseline_bleed_check_summary.json`
- Source map: `docs/PAPER_SOURCE_OF_TRUTH.md`
