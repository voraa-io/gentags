# Gentags: Discrete Semantic Representations for Decision Fidelity in LLM Pipelines

## Abstract

LLM pipelines that act on textual evidence face two practical difficulties: (1) source text often exceeds the context window, requiring compression, and (2) even when meaning is present in context, it remains embedded in free-form prose, making it difficult to diagnose, reference, or attribute to specific decisions.

**Gentags** is a representation that compresses source text into short, evidence-grounded semantic units, each isolating one identifiable semantic condition. To evaluate whether this representation preserves decision-relevant meaning, we compare Gentags against lexical baselines (RAKE, YAKE, TF-IDF) in a constraint-sensitive decision setting that isolates representation structure while holding evidence, task, and evaluation fixed. Gentags improve agreement with full-evidence decisions to **79.5%** (vs. **52.3–61.6%**) and raise hard-constraint satisfaction to **97.3%** (vs. **84.7–89.3%**), while showing greater stability across runs, prompts, and extractor models. These results suggest that discrete semantic representations can effectively preserve decision-relevant meaning, and motivate further exploration of propositional compression as intermediate state in LLM pipelines.

This repository contains the full experimental pipeline and frozen artifacts supporting the Gentags paper.

## Paper

**Preprint:** [Gentags: Discrete Semantic State for Constraint-Sensitive Decision Pipelines](https://openreview.net/forum?id=Vm1P4G8RLb)
**Venue:** OpenReview Archive
**License:** CC BY 4.0
**Code license:** MIT

Canonical paper artifacts and reproduction instructions are listed in:

* `docs/REPRODUCE_PAPER.md`
* `docs/PAPER_SOURCE_OF_TRUTH.md`

---

## Motivation

Pipelines compress evidence through chunking, retrieval, summarization, or keyword extraction because full evidence often exceeds a single context window. Finer-grained units (including propositions) can outperform passage-level chunks, yet the intermediate signal passed downstream is still typically **free-form text**—passages, summaries, or fragments ordered by structure, limits, or scores, not by semantic content.

That leaves decision-relevant meaning hard to decompose: multiple conditions may be implicit, redundant, or split across sentences, so it is difficult to see which factors matter, reference them individually, or attribute outcomes to specific meaning. That is a **representational** issue, not only a retrieval or scaling issue.

Prior work shows that intermediate structure affects correctness (reasoning traces, decomposition, aggregation, executable intermediates) and that structured meaning can be derived from text or externalized into memory—but often as an evaluation endpoint, for verifying model outputs, or still over free-form text. **Gentags** tests a different axis: decomposing **source evidence** into discrete semantic units rather than text fragments. Each unit isolates one identifiable semantic condition supported by the evidence.

The central comparison is against lexical baselines (RAKE, YAKE, TF-IDF) that also yield discrete units from the same text: widely used, inexpensive, and directly usable in a context window. The design **isolates representation structure** by fixing source evidence, downstream task, judge model, and evaluation while varying only the intermediate representation.

**Evaluation setting:** constraint-sensitive decision-making, where explicit requirements are checked against evidence to yield binary, auditable outcomes. We report stability across runs, prompts, and extractor models; structural properties of the representation; and downstream decision fidelity (50 venues, four personas with explicit hard requirements, majority-vote aggregation over five repeated judge calls with two independent judge models).

---

## Contributions

1. We frame the structure of intermediate semantic representations as a **design variable** in LLM-based evidence pipelines.
2. We introduce **Gentags** as a discrete, evidence-grounded representation that compresses source text into individually addressable semantic units.
3. We provide **controlled empirical evidence** that propositional representations improve decision reliability relative to fragment-level lexical representations in this setting, including under **token-matched** conditions that control for information volume.

---

## Project

Study 1 evaluates whether discrete semantic state improves constraint-sensitive decisions relative to lexical baselines.

The executed Study 1 extraction set contains 553 venues drawn from `data/study1_venues_20250117.csv`, with 1-5 review objects per venue in the canonical Phase 1 subset. Venues were selected prior to evaluation and not filtered based on representation performance.

The experiments examine:

* stability across runs
* sensitivity to prompt variation
* cross-model agreement
* structural properties of extracted tags
* downstream decision fidelity compared to lexical baselines (`RAKE`, `YAKE`, `TF-IDF`)

Out of scope for this repo:

* recommendation training
* user interaction data
* UX evaluation

---

## Key Properties

Gentags are:

* extracted zero-shot from text
* short (1-4 words)
* schema-free
* interpretable by humans
* usable as structured semantic features
* designed for sparse text settings

They function as compact semantic hypotheses derived from evidence text and can be inspected, compared across runs, and consumed as structured intermediate state by downstream decision procedures.

---

## Main Result

* Gentags show high semantic stability across runs (`cosine = 0.977`).
* Gentags improve agreement with full-evidence reference decisions to **79.5%** versus **52.3–61.6%** for lexical baselines.
* Gentags improve hard-constraint compliance to **97.3%** versus **84.7–89.3%** for lexical baselines under matched conditions.
* Gentags show **greater stability across runs, prompts, and extractor models** than the lexical baselines in this study.

More detailed results are in `docs/PAPER_complete.md`.

---

## Repo Map

```text
src/gentags/   Core extraction pipeline
scripts/       Phase runners and analysis scripts
tests/         Unit tests
data/          Frozen inputs and sampled datasets
results/       Paper-backed outputs and archived runs
docs/          Paper-facing documentation
notebooks/     Exploratory analysis notebooks
```

Core paper-facing docs:

* `docs/PAPER_complete.md`
* `docs/PAPER_SOURCE_OF_TRUTH.md`
* `docs/PAPER_STATUS.md`
* `docs/REPRODUCE_PAPER.md`
* `data/DATASET_CARD.md`

Canonical paper-backed artifact locations:

* `results/phase1_downloaded/`
* `results/phase2/`
* `results/phase3/`
* `results/phase3a/`
* `results/phase4/`
* `results/phase5/`

---

## Quick Start

Install dependencies:

```bash
poetry install
```

Run unit tests:

```bash
poetry run pytest tests/
```

Validate the Phase 1 entry point without making API calls:

```bash
poetry run python scripts/run_phase1.py \
  --data data/study1_venues_20250117.csv \
  --sample-size 10 \
  --models openai \
  --prompts minimal \
  --runs 1 \
  --dry-run
```

Environment notes:

* Python and dependency management are handled through Poetry.
* API keys belong in `.env` when running extraction or other model-backed scripts.
* Full paper reproduction guidance is in `docs/REPRODUCE_PAPER.md`.

---

## Reproducing The Paper

Use `docs/REPRODUCE_PAPER.md` as the main operational guide.

That file covers:

* canonical artifact paths
* phase-by-phase commands
* frozen vs expensive stages
* which outputs back the paper

---

## Paper And Reproducibility

* Public preprint: https://openreview.net/forum?id=Vm1P4G8RLb
* Draft/source copy: `docs/PAPER_complete.md`
* Claim-to-artifact map: `docs/PAPER_SOURCE_OF_TRUTH.md`
* Submission status: `docs/PAPER_STATUS.md`
* Reproduction guide: `docs/REPRODUCE_PAPER.md`

---

## Status

Study 1 repo status:

* Phase 1: complete
* Phase 2: complete
* Phase 3: complete
* Phase 4: complete (supporting mechanism evidence)
* Phase 5: complete

---

## Data Availability

The repository provides paper-facing documentation, prompts, scripts, manifests, derived results, and archived outputs where permitted.

Raw third-party review text may be subject to redistribution, privacy, or platform-term restrictions and is not intended to be redistributed where those restrictions apply. The reproducibility materials are designed to support inspection of the experimental protocol and reproduction of reported metrics on an authorized copy of the data.

---

## License And Citation

The code in this repository is released under the MIT License.

The public preprint is available on OpenReview Archive under CC BY 4.0:

https://openreview.net/forum?id=Vm1P4G8RLb

Citation metadata is provided in `CITATION.cff`.
