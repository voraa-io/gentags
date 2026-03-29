# Gentags: Discrete Semantic State for Language-Based Systems

Gentags are short semantic attributes automatically extracted by large language models from text such as venue reviews.

This repository contains the full experimental pipeline and frozen artifacts supporting the Gentags paper. The project studies whether discrete semantic units extracted from text can improve consistency in constraint-sensitive decisions compared to lexical keyword representations.

Canonical paper artifacts and reproduction instructions are listed in:

- `docs/REPRODUCE_PAPER.md`
- `docs/PAPER_SOURCE_OF_TRUTH.md`

---

## Project

Study 1 evaluates whether representation structure affects consistency of constraint-sensitive decisions.

The executed Study 1 extraction set contains 553 venues drawn from `data/study1_venues_20250117.csv`, with 1-5 review objects per venue in the canonical Phase 1 subset.
Venues were selected prior to evaluation and not filtered based on representation performance.

The experiments examine:

- stability across runs
- sensitivity to prompt variation
- cross-model agreement
- structural properties of extracted tags
- downstream decision behavior compared to lexical baselines

Out of scope for this repo:

- recommendation training
- user interaction data
- UX evaluation

---

## Key Properties

Gentags are:

- extracted zero-shot from text
- short (1-4 words)
- schema-free
- interpretable by humans
- usable as structured semantic features
- designed for sparse text settings

They function as compact semantic hypotheses derived from evidence text.

---

## Key Findings

- Gentags show high semantic stability across runs (`cosine = 0.977`).
- Gentags improve agreement with full-evidence decisions compared to lexical baselines.
- Gentags improve hard-constraint compliance in controlled decision tasks.

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

- `docs/PAPER_complete.md`
- `docs/PAPER_SOURCE_OF_TRUTH.md`
- `docs/PAPER_STATUS.md`
- `docs/REPRODUCE_PAPER.md`
- `data/DATASET_CARD.md`

Canonical paper-backed artifact locations:

- `results/phase1_downloaded/`
- `results/phase2/`
- `results/phase3/`
- `results/phase3a/`
- `results/phase4/`
- `results/phase5/`

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

- Python and dependency management are handled through Poetry.
- API keys belong in `.env` when running extraction or other model-backed scripts.
- Full paper reproduction guidance is in `docs/REPRODUCE_PAPER.md`.

---

## Reproducing The Paper

Use `docs/REPRODUCE_PAPER.md` as the main operational guide.

That file covers:

- canonical artifact paths
- phase-by-phase commands
- frozen vs expensive stages
- which outputs back the paper

---

## Paper

- Draft: `docs/PAPER_complete.md`
- Claim-to-artifact map: `docs/PAPER_SOURCE_OF_TRUTH.md`
- Submission status: `docs/PAPER_STATUS.md`

---

## Status

Study 1 repo status:

- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete (supporting mechanism evidence)
- Phase 5: complete

---

## License And Citation

This repository is released under the MIT License.

Citation metadata is provided in `CITATION.cff`.
