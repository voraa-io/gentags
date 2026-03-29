# Appendix: Qualitative Audit of Gentag Failures

This appendix audits the subset of Phase 5 conditions where the Gentag-based decision disagrees with the Full-Evidence Reference (FER) decision. The goal is not to relitigate aggregate performance, but to characterize the remaining error modes and determine whether they are predictable.

## A.1 Audit Scope

- Source decisions: `results/phase5/baseline_summary_openai_20260228_022320.json`
- Source run-level outputs: `results/phase5/baseline_results_openai_20260228_022320.json`
- Source representations and reviews: `data/phase5/sampled_venues.json`
- Derived disagreement table: `results/phase5/gentag_fer_disagreements.csv`

The audit uses the OpenAI judge run because it is the primary Phase 5 result used in the paper. We compare the aggregated Gentag decision against the aggregated FER decision for each `(venue, persona)` condition and then inspect the underlying run-level `tags_used`, `blockers`, `supports`, and FER `evidence_quotes`.

## A.2 Disagreement Summary

Out of 200 Gentag conditions (50 venues x 4 personas), Gentags disagree with FER in 41 cases.

| FER -> Gentag | Count |
| --- | ---: |
| RECOMMEND -> BORDERLINE | 18 |
| REJECT -> BORDERLINE | 9 |
| BORDERLINE -> RECOMMEND | 6 |
| RECOMMEND -> REJECT | 5 |
| REJECT -> RECOMMEND | 3 |

Most disagreements are one-step moves through `BORDERLINE` rather than direct reversals:

- `33/41` (`80.5%`) involve `BORDERLINE`
- `8/41` (`19.5%`) are exact reversals between `REJECT` and `RECOMMEND`

For the 8 exact reversals, we manually checked the underlying review text against the judge-cited Gentags.[^grounding]

Disagreements are also concentrated in specific personas:

| Persona | Count | Share |
| --- | ---: | ---: |
| `P1` Food Critic | 24 | 58.5% |
| `P4` Balanced Diner | 11 | 26.8% |
| `P3` Quick Lunch Worker | 5 | 12.2% |
| `P2` Sports Fan | 1 | 2.4% |

This concentration matters. The failures are not distributed uniformly across all conditions. They cluster in settings with mixed evidence (`P1`, `P4`) and in a smaller number of hard-indicator cases where lexical exact-match rules matter (`P2`, `P3`). Figure A9 summarizes the disagreement structure.

![Figure A9: Failure Audit Breakdown](../results/phase5/plots/4_failure_audit.png)

## A.3 Failure Mode Taxonomy

The disagreement set supports a four-part taxonomy that covers all 41 Gentag-vs-FER mismatches.

### FM1. Borderline Drift Under Mixed Evidence (`33/41`)

The dominant failure mode is not a full flip from correct to incorrect polarity. Instead, Gentags often move a case into `BORDERLINE` when the review evidence contains both strong positives and localized negatives. This is especially common for `P1` and `P4`.

Typical pattern:

- FER integrates the full review context and resolves the condition as `RECOMMEND` or `REJECT`
- the Gentag state preserves both positive and negative propositions
- the judge, when limited to Gentags, treats the condition as mixed and falls back to `BORDERLINE`

This is the core explanation for the 33 one-step disagreements:

- `RECOMMEND -> BORDERLINE`: 18
- `REJECT -> BORDERLINE`: 9
- `BORDERLINE -> RECOMMEND`: 6

Interpretation:

- these are mostly not random errors
- they reflect uncertainty induced by compressed propositional state under mixed evidence
- the representation usually stays near the FER decision, but loses enough contextual weighting to avoid a fully committed judgment

Representative cases:

| Venue | Persona | FER -> Gentag | What appears to happen |
| --- | --- | --- | --- |
| `Serenade American Brasserie` | `P1` | `REJECT -> BORDERLINE` | Gentags preserve both `tough ribeye` and `surprisingly good food`; FER treats the negative evidence as decisive, while the Gentag judge sees a mixed food state. |
| `Lulo Gelato - Arboleda` | `P1` | `REJECT -> BORDERLINE` | Reviews mention both `average` taste and positive flavor evidence; the Gentag state preserves mixed quality descriptors and does not force a clear rejection. |
| `El Raval` | `P4` | `RECOMMEND -> BORDERLINE` | Gentags preserve strong food positives but also `disappointing sangria`; the soft persona amplifies tradeoff sensitivity rather than a clean recommend. |

### FM2. Exact-Match Indicator Misses in Hard-Constraint Personas (`4/41`)

A smaller but important failure mode occurs when the Gentag state contains a semantically relevant support signal, but the hard persona indicator list is exact-match based and the judge does not treat the tag as satisfying the frozen rule.

This explains four exact reversals, which is half of all `REJECT <-> RECOMMEND` flips in the audit:

- `Colton's Apodaca`, `P2`: `RECOMMEND -> REJECT`
- `Barbacoa Don Rico`, `P3`: `RECOMMEND -> REJECT`
- `TEST - Sereno coffee bar`, `P3`: `RECOMMEND -> REJECT`
- `Au Pied de Cochon`, `P3`: `RECOMMEND -> REJECT`

Representative examples:

| Venue | Persona | FER -> Gentag | Gentag signal | FER evidence |
| --- | --- | --- | --- | --- |
| `Colton's Apodaca` | `P2` | `RECOMMEND -> REJECT` | Gentags include `game audio` and `nfl game`, but no exact sports-viewing indicator is cited in `tags_used`. | FER repeatedly cites “place to go watch NFL games.” |
| `Barbacoa Don Rico` | `P3` | `RECOMMEND -> REJECT` | Gentags include `fast delivery`, but the judge uses no tags and rejects for missing an exact speed indicator. | FER cites `delivered the order very quickly` and `attended to me super quickly`. |

Interpretation:

- this is partly a Gentag wording issue
- but it is also a protocol issue caused by frozen exact-match indicator lexicons
- the failure is predictable: semantically relevant support exists, but the evaluation rule does not recognize it
- in paper terms, a significant portion of the exact reversals are artifacts of the evaluation protocol's reliance on exact-match indicators rather than evidence that Gentags failed to capture the underlying semantic signal

### FM3. Positive-Cue Anchoring Under Contradictory Service Evidence (`2/41`)

Two `P3` cases flip in the opposite direction:

- `TenTen`: `REJECT -> RECOMMEND`
- `Cielo Tinto`: `REJECT -> RECOMMEND`

In both cases, the Gentag state contains the exact positive indicator `efficient service`, and the judge consistently cites it. FER, however, finds either explicit slow-service evidence (`TenTen`) or a weaker, noisier service picture (`Cielo Tinto`).

Representative example:

| Venue | Persona | FER -> Gentag | Gentag cue used | FER cue used |
| --- | --- | --- | --- | --- |
| `TenTen` | `P3` | `REJECT -> RECOMMEND` | `efficient service` is used in all 5 Gentag runs. | FER repeatedly cites `Service was pretty slow`. |

Interpretation:

- once an exact positive indicator is present, the hard-rule judge can anchor on it too strongly
- contradictory Gentags such as `slow service` may remain in the state but go unused
- this is a failure of conflict resolution inside the compressed state, not a purely random extraction error
- this suggests a concrete extension for future Gentag variants: attach semantic weights, evidence counts, or provenance counts so the downstream judge can resolve contradictions like `efficient service` versus `slow service` instead of treating the first exact indicator as decisive

### FM4. Missed Negative Cue Despite Negative Semantics (`2/41`)

The last failure mode consists of exact reversals where Gentags contain some negative semantics, but the downstream judge does not treat them as decisive blockers.

This occurs in:

- `Cedar Door Patio Bar & Grill`, `P1`: `RECOMMEND -> REJECT`
- `Catrinas Chilaquiles - Mezquital`, `P1`: `REJECT -> RECOMMEND`

These two cases move in opposite directions:

- `Cedar Door` shows localized negative evidence (`sour taco`) being treated as decisive against otherwise strong food positives.
- `Catrinas` shows the opposite: Gentags include `hygiene concern` and `unsanitary practice`, but the judge cites only `great food` and misses the negative semantics that drove FER rejection.

Interpretation:

- these are not generic Gentag failures
- they are high-value food-critic edge cases where negative evidence must be weighted correctly
- the representation preserves the relevant cues, but the downstream use of those cues is brittle

## A.4 Representative Case Notes

### Case 1: Localized blocker dominates otherwise positive state

- Venue: `Cedar Door Patio Bar & Grill`
- Persona: `P1`
- FER -> Gentag: `RECOMMEND -> REJECT`

Gentags used by the judge:

- blocker: `sour taco`
- supports: `great food`, `outstanding food`

FER evidence repeatedly cites both the localized complaint and broader positive evidence (`Food was great`, `absolutely outstanding`, `brisket tacos were to die for`). The Gentag judge treats the localized negative proposition as sufficient to reject.

### Case 2: Sports-viewing signal exists but is not counted as satisfying the rule

- Venue: `Colton's Apodaca`
- Persona: `P2`
- FER -> Gentag: `RECOMMEND -> REJECT`

Gentags include `game audio` and `nfl game`, but the judge uses no tags in the final decision. FER repeatedly cites “place to go watch NFL games.” This is the clearest exact-match indicator miss in the audit.

### Case 3: Fast-service support is present semantically but absent lexically

- Venue: `Barbacoa Don Rico`
- Persona: `P3`
- FER -> Gentag: `RECOMMEND -> REJECT`

Gentags include `fast delivery`, but no exact speed indicator is accepted by the rule-based prompt. FER repeatedly cites `very quickly` and `super quickly`. This case shows that hard constraints remain sensitive to lexicalization even when the semantic content is close.

### Case 4: Positive indicator overrides contradictory negative evidence

- Venue: `TenTen`
- Persona: `P3`
- FER -> Gentag: `REJECT -> RECOMMEND`

Gentags contain both `efficient service` and `slow service`, but only `efficient service` is used in the Gentag decision. FER repeatedly cites `Service was pretty slow`. This is a conflict-resolution failure inside the compressed state.

### Case 5: Negative semantics are preserved but not used

- Venue: `Catrinas Chilaquiles - Mezquital`
- Persona: `P1`
- FER -> Gentag: `REJECT -> RECOMMEND`

Gentags include `hygiene concern` and `unsanitary practice`, but the Gentag judge cites only `great food`. FER cites review text about unacceptable food handling and cancellation of the meal. This is the clearest case where a negative proposition is present in the state but not surfaced in the decision.

## A.5 Implications for the Paper

The qualitative audit sharpens the main claim in three ways.

First, the remaining Gentag errors are structured rather than arbitrary. Most mismatches are one-step `BORDERLINE` drifts under mixed evidence, not wholesale reversals.

Second, several exact reversals are tied to frozen exact-match constraint lexicons. This means some of the residual error is not purely representational; it is introduced by the evaluation protocol used to enforce hard requirements. In this audit, `4/8` exact reversals fall into this category.

Third, the failure cases point to concrete next improvements:

- store per-tag evidence provenance so unsupported or weakly supported Gentags can be audited directly
- replace exact string indicator matching with semantic indicator matching or lexicon expansion
- add explicit handling of contradictory propositions inside a Gentag state
- expose semantic weights or evidence counts so conflicting propositions can be resolved by more than binary presence

For the main paper, this appendix supports a stronger discussion claim:

> Gentag failures are concentrated in predictable edge cases, especially mixed-evidence conditions and hard-indicator lexical mismatches, rather than being randomly distributed across the decision space.

[^grounding]: Grounding check for the 8 exact reversals: in the 4 cases where the Gentag judge cited non-empty `tags_used` (`Cedar Door Patio Bar & Grill`, `TenTen`, `Cielo Tinto`, `Catrinas Chilaquiles - Mezquital`), the cited tags are factually supported by the source reviews on manual inspection. In the remaining 4 exact reversals (`Colton's Apodaca`, `Barbacoa Don Rico`, `TEST - Sereno coffee bar`, `Au Pied de Cochon`), the judge cited no Gentags at all and rejected solely because no exact indicator match was recognized. This weakens a simple "hallucinated Gentag" explanation for the reversal set.
