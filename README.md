# Gentags: Discrete Semantic State for Language-Based Systems

Gentags (Generative Tags) are semantic attributes automatically extracted by large language models from sparse textual data (e.g., brief venue reviews).

Gentags form a **persistent, inspectable semantic state** composed of discrete propositions (1–4 words). They externalize model judgments into an addressable representation that can be stored, compared, and updated. While machine-generated rather than user-contributed, gentags resemble folksonomies in structure and utility: they are interpretable, compositional, and arise without predefined taxonomies.

---

## Key Ideas

- **Zero-shot semantic extraction** from review text
- **No predefined ontology** or schema
- **No ratings or sentiment labels**
- **Short, atomized tags** (1–4 words)
- **Interpretable and embed-friendly**
- **Designed for sparse and cold-start settings**

Gentags are treated as **semantic constraints**, not labels or summaries.

---

## Research Scope

This repository supports **Study 1** of the Gentags project.

**Study 1 focuses on:**

- Cross-model agreement (OpenAI, Gemini, Claude, Grok)
- Prompt sensitivity
- Stability across runs
- Behavior under sparse review conditions
- Reproducible, frozen extraction pipeline

**This repo does NOT include:**

- Recommendation model training
- User interaction logs
- Product or UX experiments

---

## Methodological Commitments (Study 1 Lock)

- Models are used **as provided** (no temperature, top-p, or decoding tuning)
- Gentags are extracted **zero-shot** (no examples, no few-shot prompting)
- No predefined semantic categories or ontology
- Ratings are explicitly excluded from extraction
- Output validity refers to **format correctness only**, not semantic truth

All definitions, prompts, and model identifiers are frozen and documented in: `docs/STUDY1_LOCK.md`

---

## Repository Structure

```
src/gentags/          # Extraction pipeline (importable)
notebooks/            # Reproducible experiments and analysis
scripts/              # CLI runners and utilities
docs/                 # Methodology and study documentation
data/                 # Datasets (see data/README.md)
  ├── sample/         # Small public-safe example data
  └── study1_venues_20250117.csv  # Main dataset
results/              # Experiment outputs (see results/README.md)
  ├── meta/           # Reproducibility manifests
  └── examples/       # Example outputs (small, non-sensitive)
tests/                # Unit tests (no API keys required)
```

---

## Quick Start

### Setup

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Set up API keys (optional - tests pass without them)
cp .env.example .env
# Edit .env and add your API keys for models you want to use
```

### Verify Installation

Run unit tests (no API keys required):

```bash
poetry run pytest tests/
```

Run smoke test (skips gracefully if no API keys):

```bash
poetry run python scripts/smoke_test_minimal.py
```

### Sample Run

```python
from gentags import (
    GentagExtractor,
    run_experiment,
    load_venue_data,
    summarize_cost,
    save_results
)

# Load data
df = load_venue_data("data/study1_venues_20250117.csv", sample_size=10)

# Initialize extractor
extractor = GentagExtractor()

# Run experiment
results = run_experiment(
    extractor=extractor,
    venues_df=df,
    models=["openai"],
    prompts=["minimal"],
    runs=1
)

# Save results
output_path = save_results(results, prefix="gentags")
print(f"Results saved to: {output_path}")

# Analyze costs
summary = summarize_cost(results)
print(f"Total cost: ${summary['total_cost_usd']:.6f}")
print(f"Avg cost per extraction: ${summary['avg_cost_per_extraction_usd']:.6f}")
```

### Output Files

Each experiment run produces:

- `results/gentags_<timestamp>.csv` - Full results (one row per tag)
- `results/meta/manifest_<timestamp>.json` - Reproducibility manifest

Use `summarize_cost()` to generate:

- Extraction-level CSV (one row per extraction)
- Cost breakdown by model/prompt

See `docs/STUDY1_LOCK.md` for the frozen Study 1 methodology.

---

## License & Citation

This repository is released under the MIT License.

If you use this work, please cite:

```
Gentags: Emergent Semantic Tags from Sparse Reviews
(CITATION.cff provided)
```

---

## Status

### Study 1 Progress

| Section | Description | Status |
|---------|-------------|--------|
| Motivation | Persistent, addressable semantic state | ✅ Complete |
| Phase 1 | Extraction pipeline & multi-model runs | ✅ Complete |
| Phase 2 | Semantic stability analysis | ✅ Complete |
| Phase 3 | Structural proof (State-Gini) + Utility proof (DIR/INV) | 🔜 Planned |
| Phase 4 | Downstream probes | 🔜 Planning |

### Core Thesis

> Gentags occupy an intermediate point between symbolic keywords and dense embeddings: **semantic enough to generalize, lexical enough to attribute.**

### Claims

| Claim (Phase 2 validated) | Evidence | Implication |
|--------------------------|----------|-------------|
| Gentags are semantically stable | Cosine 0.977 | Representation can serve as state |
| Gentags vary lexically | Jaccard 0.471 | Stability is semantic, not surface |
| Evidence-sensitive dispersion | r = -0.230 | Less evidence → less identifiability |
| Retention above random | +0.164 | Tags preserve source meaning |

### Planned Phase 3 Tests

- Structural proof: State-Gini factorization (gentags vs RAKE/TF-IDF/YAKE)
- Utility proof: CheckList DIR/INV interventions (gentags vs embeddings)

### Key Numbers

| Metric (Phase 2) | Value |
|------------------|-------|
| Semantic stability (cosine) | 0.977 |
| Surface variation (Jaccard) | 0.471 |
| Semantic gap (cosine - Jaccard) | 0.504 |
| Retention above random | +0.164 |
| Evidence-variability correlation | -0.230 |
| Localization Gini (embeddings) | 0.369 |
| **Localization advantage** | **1.50×** |
| Paraphrase MMC | 0.648 |

See `docs/SECTION3_REPRESENTATION.md` for the core representation claims.
See `docs/MOTIVATION.md` for why gentags are necessary.

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/MOTIVATION.md` | Why gentags exist (31.6% LLM instability) |
| `docs/PHASE1_EXTRACTION.md` | Multi-model extraction (13,272 extractions) |
| `docs/PHASE2_STABILITY.md` | Semantic stability analysis |
| `docs/SECTION3_REPRESENTATION.md` | **Core claims: localization, baselines, boundaries** |
| `docs/STUDY1_LOCK.md` | Frozen methodology |

---

## Out of Scope

- **Study 2** (applied recommendation & UX): Separate repository
- **pdensity**: Exploratory, not central contribution
- **User studies**: Not in this paper
