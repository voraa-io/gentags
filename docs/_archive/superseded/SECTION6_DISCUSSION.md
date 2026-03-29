# Discussion

> **Paper section:** Section 6
> **Status:** Draft
> **Last updated:** 2026-03-22
> **Related sections:** `docs/RESULTS.md`, `docs/APPENDIX_FAILURE_AUDIT.md`

## 6. Discussion

The results support a specific claim about representation design. Gentags are not merely more readable keyword lists. They function as an externalized semantic state that is stable across extraction conditions, better covered under the structural probe used in Phase 3, and more faithful to full-evidence downstream decisions than fragment-level lexical baselines. The practical value of this representation is that it makes semantic state explicit: individual units can be inspected, compared, and edited before a decision is made.

At the same time, the qualitative audit shows that the remaining Gentag errors are structured rather than random. Most Gentag-FER mismatches are one-step `BORDERLINE` drifts under mixed evidence, while a smaller subset of exact reversals arises from identifiable protocol or conflict-resolution failures. This makes the discussion section more than a generic limitations list. The error profile points directly to what the current representation does well, what it still leaves unresolved, and which extensions are most likely to matter.

## 6.1 Benefits

Gentags provide three benefits that matter for constrained downstream systems.

First, they provide an **interpretable semantic state**. Unlike dense embeddings or free-form summaries, Gentags expose individual semantic propositions as discrete units. This makes it possible to inspect the basis of a decision, compare state realizations across runs, and identify which parts of the representation are actually used downstream.

Second, they provide a **schema-free but still structured representation**. Gentags do not require a fixed ontology at extraction time, yet Phase 3 shows that they still occupy a more decision-relevant structural space than lexical baselines. Relative to RAKE, YAKE, and TF-IDF, gentags achieve broader facet coverage, lower mass in `other`, and less boundary ambiguity under the bleed-check diagnostic.

Third, they provide **better support for explicit constraint-sensitive decisions**. In Phase 5, Gentags preserve full-evidence decisions and hard-constraint compliance more reliably than lexical fragments, including under token-matched conditions. This suggests that the main advantage is not just readability or verbosity. It is that Gentags preserve semantically actionable units in a form the downstream judge can use.

Taken together, these benefits support the paper's central framing: semantic state structure is an architectural variable in LLM decision pipelines, and Gentags are one viable design for making that state explicit.

## 6.2 Limitations

The current study also has clear limits.

First, the evaluation is confined to a single restaurant-review decision domain. This domain is useful because it permits controlled hard constraints, repeated extraction, and manual inspection of disagreements, but it is still narrow. The paper therefore shows that Gentags help in this specific constraint-sensitive setting, not that they are universally preferable across all language-system tasks.

Second, the representation currently carries limited information about **uncertainty and evidence strength**. The dominant failure mode in the audit is not catastrophic reversal but mixed-evidence drift into `BORDERLINE` (`33/41` mismatches). This suggests that Gentags often preserve the relevant propositions but do not fully encode how strongly each proposition is supported, how broadly it is attested across reviews, or how decisive it should be under aggregation.

Third, the current pipeline does not fully resolve **contradictory propositions inside the state**. The `TenTen` case is the clearest example: the Gentag state contains both `efficient service` and `slow service`, but the downstream judge anchors on the exact positive indicator and ignores the contradiction. More generally, the representation exposes competing propositions but does not yet provide a principled mechanism for resolving them.

Fourth, part of the remaining error is introduced by the **evaluation protocol itself** rather than by the Gentag representation alone. The audit shows that `4/8` exact `REJECT <-> RECOMMEND` reversals arise from frozen exact-match hard-indicator lexicons. In these cases, the Gentag state contains semantically relevant support such as `fast delivery`, `game audio`, or `nfl game`, but the judge rejects because no exact indicator match is recognized. That means a significant portion of the exact reversals are protocol artifacts rather than clean evidence that Gentags failed to capture the underlying semantic signal.

Fifth, the system does not yet fully normalize **surface variation among semantically equivalent tags**. Phase 2 shows that semantic stability is high despite lower lexical overlap, which is a strength for recoverability, but it also means that canonicalization remains incomplete. Surface variants such as near-paraphrases, morphological variants, or differently scoped phrases may still be treated as distinct units unless a later stage merges them.

Finally, the extraction stage itself introduces **execution failures** that reduce the effective sample size for aligned analyses. In the Phase 2 extraction grid, these failures are strongly model-specific rather than prompt-specific, with Claude and Grok accounting for the observed unsuccessful runs. Although the retained and excluded venues do not differ detectably by simple review-length proxies, future work should improve schema-constrained extraction robustness so that alignment requirements do not unnecessarily discard otherwise usable cases. Differences in schema adherence across extractor models reduce the effective sample size for aligned multi-model comparisons and highlight the importance of structured output reliability in externalized semantic state pipelines.

More broadly, the current paper relies on LLM judges rather than human adjudication for the main downstream evaluation. The use of matched protocols, FER reference decisions, cross-judge replication, and the qualitative audit reduces this concern, but it does not remove it completely. The reported gains are therefore strongest as evidence of comparative representation utility under a controlled judge pipeline.

## 6.3 Future Work

The audit suggests three immediate technical directions.

First, future work should improve **hard-constraint matching** by moving beyond exact string indicators. Semantic indicator matching, lexicon expansion, or lightweight entailment checks would better align the decision protocol with the actual content of the Gentag state. This is the most direct response to FM2, where semantically relevant support is present but invisible to an exact-match rule.

Second, future Gentag variants should support **conflict-aware state resolution**. Cases like `TenTen` suggest that a proposition should not be represented only as a binary presence marker. Useful extensions would include semantic weights, evidence counts, polarity markers, or provenance counts so that a downstream judge can distinguish a lightly supported positive cue from a heavily attested contradictory cue.

Third, the extraction pipeline should retain richer **evidence provenance and uncertainty signals**. Storing source spans, review counts, or confidence-like summaries for each Gentag would make it easier to audit unsupported tags, reason about mixed evidence, and separate strongly grounded propositions from weak or isolated ones. This would also make future qualitative audits less manual.

Beyond these immediate extensions, the evaluation should move into broader and higher-stakes domains. Useful next settings include product reviews, support tickets, policy documents, and news or incident reports. These settings would test whether the advantages observed here transfer when the constraint structure is more open-ended, the evidence is more heterogeneous, or the cost of an incorrect decision is higher.

Another direction is **state canonicalization across time and runs**. The current work shows that Gentags are semantically stable despite surface variation, but longitudinal pipelines will need mechanisms for merging paraphrastic tags, updating stale propositions, and tracking how state changes over time. That would move Gentags closer to a reusable semantic substrate for iterative decision systems rather than a one-shot extraction layer.

Extraction robustness is also an immediate engineering target. Since aligned multi-model analyses lose sample size when schema-constrained outputs fail, future pipelines should reduce malformed-output rates through stronger format control, retry policy improvements, or post-parse repair so that representation studies are less exposed to model-specific execution noise.

One possible future direction is to characterize how individual semantic units differ in how strongly they constrain downstream behavior, potentially relating representation structure to information-theoretic notions of semantic density.

Overall, the current results suggest that Gentags are a useful starting point rather than a finished state model. Their main value is that they make semantic state explicit enough to measure, debug, and improve. The audit reinforces that conclusion: once the representation is externalized, its errors can be categorized, traced, and turned into concrete research directions.
