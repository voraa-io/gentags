# Section 4: Experimental Setup

> **Paper section:** Section 4
> **Status:** Draft
> **Last updated:** 2026-03-07
> **Source discipline:** All numeric claims in this draft were checked against `data/` and `results/` artifacts, with protocol details confirmed in `scripts/`.

---

## 4. Experimental Setup

The paper evaluates gentags through four empirical studies. These studies answer different questions and operate at different levels of analysis: extraction stability, structural organization, downstream decision utility, and intervention-based diagnostics. The shared design principle is to compare gentags against text-derived baselines while keeping the evaluation protocol fixed wherever possible.

The main paper should treat **Phase 2**, **Phase 3 (State-Gini)**, and **Phase 5** as the core empirical backbone. **Phase 4 DIR** is informative, but it is best used as supporting mechanism evidence rather than as the primary headline result, because its own scaled report is underpowered and not paper-ready as a standalone separation claim.

## 4.1 Data

All studies draw from the same underlying corpus of **553 venues**, each represented by **1-20 user reviews**, stored in `data/study1_venues_20250117.csv`. Gentags are extracted from this corpus as described in Section 3.

Different studies use different frozen analysis subsets:

| Study | Unit of analysis | Data subset | Purpose |
|-------|------------------|-------------|---------|
| **Phase 2: Stability** | extraction / venue | 230 venues with successful extractions from all 4 models | Run-to-run, prompt, and cross-model stability |
| **Phase 3: Structure** | extraction and venue | same 230 aligned venues | State-Gini and facet-coverage analysis |
| **Phase 5: Decision utility** | venue-persona-system condition | stratified 50-venue sample | Constraint-based decision evaluation |
| **Phase 4: DIR diagnostic** | intervention unit | 3 venues, 16 units | Mechanism-level intervention analysis |

The **230-venue aligned subset** is used for stability and structure because those studies require fair cross-model comparison. The **50-venue decision subset** is frozen in `data/phase5/sampled_venues.json` and stratified to ensure coverage of the two hard-requirement dimensions used in downstream evaluation: **6** venues with sports-viewing evidence, **15** with fast-service evidence, and **29** with neither. The **Phase 4 DIR** study uses a smaller, hand-constructed intervention set over **3 venues** and **16 total units**.

## 4.2 Representations and Baselines

The primary representation under study is the **gentag state**: a set of short, evidence-conditioned semantic units extracted from reviews. For downstream evaluation, the comparison is always representation-to-representation, not model-to-model. The baselines are derived from the same review text as the gentags.

The paper uses the following comparison objects:

| Representation | Construction | Used in |
|----------------|--------------|---------|
| **gentag** | LLM-extracted semantic tags | Phases 2, 3, 4, 5 |
| **RAKE** | lexical keyword extraction | Phases 3, 4, 5 |
| **YAKE** | lexical keyword extraction | Phases 3, 5 |
| **TF-IDF** | top weighted n-grams / phrases | Phases 3, 5 |
| **gentag_truncated** | gentags truncated to match RAKE count | Phase 5 |
| **FER** | raw review text given directly to judge | Phase 5 |

For the structural and decision studies, all baseline phrase lists are derived from the **same review evidence** as the gentags for each venue. In Phase 3, RAKE, YAKE, and TF-IDF are compared to gentags under the same facet-assignment procedure. In Phase 5, the same venue is evaluated under gentags, lexical baselines, a token-matched gentag ablation, and a full-evidence reference condition.

The `gentag_truncated` condition is important because it controls for information volume. If gentags outperform lexical baselines even after being truncated to the baseline tag count, the gain is attributable to semantic quality rather than simply to providing more text.

## 4.3 Study-Specific Setups

### 4.3.1 Phase 2: Stability Setup

Phase 2 asks whether gentags behave like a recoverable semantic state rather than a brittle surface artifact. The full extraction grid contains **13,272 extractions** from **553 venues**, crossing **4 extractor models**, **3 prompt variants**, and **2 runs per configuration**. Extraction outputs are validated as JSON lists under a fixed preprocessing rule; rows with `status != success` are excluded, and analyses requiring aligned comparison are then restricted to venues with successful extractions from all required model conditions. After applying this rule, the stability analysis is restricted to the **230 venues** with successful extractions from all four models, yielding **5,517 final extractions** and **118,832 tag rows**.

The four extractor models are OpenAI `gpt-5-nano`, Gemini `gemini-2.5-flash`, Claude `claude-sonnet-4-5`, and Grok `grok-4`. The three frozen prompts are `minimal`, `anti_hallucination`, and `short_phrase`. This design supports three separate stability checks: rerun consistency within the same model and prompt, prompt sensitivity within the same model, and cross-model agreement under matched prompts. A fourth analysis relates representation dispersion to evidence sparsity across venues.

We also inspected extraction-failure patterns to determine whether the aligned subset was likely to be selection-biased. In this run, failures are concentrated by extractor model rather than by prompt: Claude and Grok account for all extraction failures, while Gemini and OpenAI complete successfully throughout; prompt type shows no detectable failure difference. By simple evidence-size proxies from the source dataset (review count, review words, review characters), the retained and excluded venues do not differ detectably, which suggests that the aligned subset is shaped more by model-specific execution reliability than by a systematic preference for shorter or cleaner venues.

The primary measurements are semantic cosine similarity, surface Jaccard overlap, Mean Max Cosine as a semantic paraphrase-consistency measure, retention relative to source reviews, and evidence-induced variability. Because the same venue is observed under multiple models, prompts, and runs, Phase 2 functions as the paper's recoverability test for gentag state.

For clarity, these metrics operate at different levels. **Cosine similarity** is computed on the mean-pooled embedding of the full deduplicated tag set for each extraction, so it measures overall semantic similarity between two recovered states. **Jaccard similarity** is computed on the exact overlap of canonicalized tag strings after light normalization (`tag_norm_eval`), not on semantic matching. This normalization includes lowercasing, trimming punctuation and whitespace, and a small amount of evaluation-only cleanup such as removing common prefixes (e.g. `very`, `really`) and applying simple singularization. **Mean Max Cosine** is computed on individual tag embeddings: for each tag in one set, the method finds the best semantic match in the other set, averages those best-match cosines, and then symmetrizes across directions.

For the evidence-dispersion analysis, the unit of analysis is the **venue**. For each venue, we compute the mean pairwise distance `(1 - cosine)` among its available extracted states, then relate that venue-level dispersion score to evidence quantity. Statistical assessment of this relationship uses correlation tests at the venue level and nonparametric bucket comparisons when contrasting sparse-evidence and higher-evidence groups.

### 4.3.2 Phase 3: Structural Setup

Phase 3 asks whether gentags form a more usable semantic state than lexical fragments. The main structural probe is **State-Gini**, computed after hard-assigning tags or baseline keywords to a fixed set of **10 diagnostic facets** using cosine similarity to frozen anchor embeddings from `text-embedding-3-large`. Assignment uses an argmax rule with threshold **τ = 0.35**; items below threshold are routed to an explicit **`other`** bucket.

In the executed full run, gentag localization is computed over **10,373 extractions**. The lexical baseline comparison is computed over **230 aligned venues** for each of **RAKE**, **TF-IDF**, and **YAKE** under the same facet-assignment procedure. The structural comparison therefore holds the facet inventory, embedding model, assignment rule, and threshold fixed across all methods.

This `other` bucket is part of the main method, not a side detail. State-Gini is computed on the 10 facet counts, but **`other_rate` must be reported alongside it** because facet coverage affects interpretation. In the executed Phase 3 run, gentags show **lower `other_rate`** than RAKE/YAKE/TF-IDF, which means more semantic mass is captured by the diagnostic facets. Because high `other_rate` can inflate Gini by concentrating only a small assigned subset, the paper should present **State-Gini and `other_rate` together** rather than treating Gini alone as the structural result.

The facet-assignment procedure is the same for gentags and for all lexical baselines:

1. Embed each tag or keyword as a text vector using `text-embedding-3-large`.
2. Embed each of the 10 frozen facet anchors once using the same embedding model.
3. Compute cosine similarity between the tag/keyword vector and all 10 facet-anchor vectors.
4. Find the single best-matching facet by argmax.
5. If the best cosine is at least **τ = 0.35**, assign the item to that facet.
6. If the best cosine is below **τ**, assign the item to **`other`**.

This is a **hard assignment** procedure: each tag or keyword contributes to at most one facet, or to `other`. The purpose of the threshold is to avoid forcing weak matches into the facet inventory. After assignment, Phase 3 records both the 10 facet counts and the `other` count. **State-Gini** is computed on the 10 facet counts only, while **`other_rate`** is reported separately as the fraction of items that did not confidently map into the diagnostic facet space.

Because argmax assignment can hide ambiguity when the top two facet similarities are nearly tied, Phase 3 also includes a **bleed-check diagnostic**. For each gentag or baseline keyword, we record the highest facet similarity, the second-highest facet similarity, and their gap `(primary - secondary)`. Small gaps indicate that an item lies near a facet boundary and that the hard assignment should be interpreted as a pragmatic diagnostic choice rather than as proof of a uniquely correct ontology-level label. This diagnostic is reported for gentags and lexical baselines to show whether the facet system behaves like a clean partition or a softer semantic probe space.

### 4.3.3 Phase 5: Decision Utility Setup

Phase 5 asks whether the representation preserves decision-relevant information in a constraint-based setting. Each condition consists of a venue, a persona, and a representation type. A judge receives only the supplied representation and returns one of three decisions: `REJECT`, `BORDERLINE`, or `RECOMMEND`. For tag-based systems, the judge must also report whether persona-specific hard requirements are satisfied or violated using a strict structured format.

The evaluation uses **50 venues**, **4 personas**, and **6 systems** (`gentag`, `rake`, `yake`, `tfidf`, `gentag_truncated`, `fer`), yielding **1,200 conditions** per judge.

The `FER` condition is not a separate system; it is the **full-evidence reference** for the same decision problem. It uses the same persona, the same judge, the same rubric, and the same aggregation procedure as the representation-only systems, but the judge sees the raw reviews instead of tags. FER agreement therefore measures whether a compressed representation preserves the same decision that would be reached under full evidence.

### 4.3.4 Phase 4: DIR Diagnostic Setup

Phase 4 is a CheckList-style directional intervention study. It tests whether targeted edits to a representation produce predictable downstream decision changes. The scaled run covers **16 intervention units** across **3 venues**, using gentags and RAKE. This study produces useful qualitative and mechanistic evidence, especially about baseline comprehensibility and floor effects, but its own report concludes that the raw separation result is **not paper-ready as a primary claim**. The best use of Phase 4 is therefore as **supporting evidence**, discussion material, or appendix material.

## 4.4 Personas, Constraints, and Decision Context

The downstream decision study evaluates four personas. Three impose explicit hard requirements, and one serves as a soft-preference control:

| Persona | Type | Hard requirement |
|---------|------|------------------|
| **P1: Food Critic** | Hard | If a negative food-quality indicator is present, the venue must be rejected |
| **P2: Sports Fan** | Hard | If no sports-viewing indicator is present, the venue must be rejected |
| **P3: Quick Lunch Worker** | Hard | If no fast-service indicator is present, the venue must be rejected |
| **P4: Balanced Diner** | Soft | No single factor is a hard dealbreaker |

Persona definitions and indicator lexicons are frozen in `data/phase5/phase5_personas.json`. For the hard personas, requirement evaluation is based on **exact-match indicator sets** rather than post-hoc semantic reinterpretation. This keeps constraint specification fixed across all systems.

The same core insight also informs the Phase 4 DIR diagnostic. Several RAKE failures arise because fragments such as `"relative quick time"` or `"watching"` are not reliably interpretable as satisfying persona-critical constraints, whereas gentag phrases such as `"fast service"` or `"watching game"` are more decision-legible. That Phase 4 result should not be used as the paper's main statistical proof, but it is useful supporting evidence for why the Phase 5 decision gaps occur.

## 4.5 Judges and Aggregation

Judge-based evaluation is used in Phase 4 and Phase 5. The **primary judge** is `gpt-4o-2024-08-06`, and the **cross-judge validation model** in Phase 5 is `claude-sonnet-4-20250514`.

In Phase 5, each `(venue, persona, system)` condition is evaluated with **N = 5** repeated judge calls. Decisions are aggregated by majority vote. If fewer than **3** valid responses are returned, the condition is marked `UNSCORABLE`; if the vote ties, the aggregate decision is set to `BORDERLINE`. The OpenAI run contains **1,200 conditions** and **6,000 total judge calls**, with an invalid-call rate of **3.35%** and **4 unscorable conditions**. The Claude run uses the same condition grid and yields **1,199 scored conditions** and **5,995 total judge calls**, with an invalid-call rate of **10.44%** and **101 unscorable conditions**.

In Phase 4, the DIR diagnostic uses the same primary judge family (`gpt-4o-2024-08-06`) and the same repeated-call aggregation pattern with **N = 5**. The scaled run covers **480 total API calls** with **0 invalid responses** and **0 unscorable conditions**.

For tag-based systems, the judge must return structured JSON containing `decision`, `requirement_status`, `blockers`, `supports`, and `tags_used`. Strict validation enforces that `tags_used` must be an exact subset of the provided tags, and that `blockers` and `supports` must be subsets of `tags_used`. This prevents the judge from silently introducing external evidence or paraphrased support not present in the representation itself.

## 4.6 Controlled Factors

The experiments are designed to isolate representational effects rather than changes in evidence, prompts, or evaluators.

First, the **underlying evidence is held fixed** within each comparison. Gentags and all lexical baselines for a given venue are derived from the same reviews. FER uses those same reviews directly.

Second, the **evaluation protocol is held fixed** within each study. In Phase 3, gentags and lexical baselines use the same diagnostic facets, the same anchor embeddings, the same hard assignment rule, and the same threshold `τ = 0.35`. In Phase 5, all systems use the same persona text, the same decision rubric, the same repeated-judge aggregation scheme, and the same output validation.

Third, **constraint specification is frozen**. Persona lexicons are not tuned post hoc to favor gentags, and the structural probe reports an explicit `other` bucket so facet coverage failures cannot be hidden inside the localization metric.

Fourth, **representation size is controlled** where necessary. The `gentag_truncated` ablation in Phase 5 matches RAKE tag count per venue, directly testing whether semantic clarity matters beyond token volume.

Finally, **judge dependence is measured explicitly**. Phase 5 reruns the full decision study with a second judge model, and Phase 4 uses the same judge family and aggregation scheme as the main decision setting so that its diagnostic findings remain comparable.

Under this design, the central comparisons can be interpreted as differences in how well each representation preserves and exposes decision-relevant semantic content.

## 4.7 What Belongs in the Main Paper

For clarity, the paper should separate **core evidence** from **supporting diagnostics**:

- **Main paper:** Phase 2 stability, Phase 3 structure, Phase 5 decision utility
- **Main paper, brief support or discussion:** Phase 4 DIR floor-rate / baseline-comprehensibility insight
- **Appendix candidate:** full Phase 4 per-unit intervention tables and placebo analysis

The `other` bucket from Phase 3 should be in the **main paper**, not hidden in the appendix. It is necessary for interpreting State-Gini correctly. By contrast, the full Phase 4 intervention inventory is too detailed for the main narrative, but the qualitative takeaway is useful: lexical fragments often fail before intervention because they do not establish a semantically legible baseline state.

---

## File References

- Shared corpus: `data/study1_venues_20250117.csv`
- Source map: `docs/PAPER_SOURCE_OF_TRUTH.md`
- Stability input/status: `docs/PHASE2_STABILITY.md`
- State-Gini design: `docs/PHASE3_STATE_GINI_PLAN.md`
- State-Gini results: `docs/phase3/state_gini_full_run_analysis.md`
- DIR diagnostic report: `docs/phase4/DIR_SCALED_RUN_REPORT.md`
- Phase 5 report: `docs/phase5/BASELINE_LEGIBILITY_REPORT.md`
- Frozen Phase 5 sample: `data/phase5/sampled_venues.json`
- Frozen personas: `data/phase5/phase5_personas.json`
- Phase 5 OpenAI manifest: `results/phase5/baseline_manifest_openai_20260228_022320.json`
- Phase 5 Claude manifest: `results/phase5/baseline_manifest_claude_20260228_032717.json`
- Phase 5 aggregated analysis: `results/phase5/baseline_legibility_analysis.json`
