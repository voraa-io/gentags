# Gentags: Machine-Generated Folksonomy for Sparse Review Recommendation

Gentags (Generative Tags) are semantic attributes automatically extracted or synthesized by large language models from sparse textual data (e.g., brief venue reviews).

Gentags function as a **machine-generated folksonomy**: an emergent, human-readable semantic layer that represents venue atmosphere, suitability, and contextual use cases. They enable decision support and cold-start recommendation in low-data environments without relying on ratings, predefined taxonomies, or dense interaction logs.

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
data/sample/          # Small public-safe example data
results/examples/     # Example outputs (small, non-sensitive)
```

---

## Quick Start

### Setup

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys
```

Run scripts with Poetry:

```bash
poetry run python scripts/sanity_check.py
```

See `SETUP.md` for detailed setup instructions.

### Sample Run

```python
from gentags import GentagExtractor, run_experiment, load_venue_data

df = load_venue_data("data/sample/venues_sample.csv")
extractor = GentagExtractor()
results = run_experiment(extractor, df)
```

Or run the smoke test:

```bash
python scripts/sanity_check.py
```

Each extraction records:

- venue ID and review count
- model and prompt used
- extracted gentags
- token usage and cost (if available)
- run identifiers and hashes for reproducibility

Parse errors and raw outputs are logged separately for auditability.

---

## License & Citation

This repository is released under the MIT License.

If you use this work, please cite:

```
Gentags: Machine-Generated Folksonomy for Sparse Review Recommendation
(CITATION.cff provided)
```

---

## Status

- **Study 1:** In progress
- **Study 2** (applied recommendation & UX): Out of scope for this repository
