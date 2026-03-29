# Results Directory

## Overview

This directory contains frozen artifacts, phase outputs, plots, manifests, and smaller exploratory runs produced during development of the Gentags paper.

For the paper-backed reproduction path, the canonical result locations are:

- `results/phase1_downloaded/`
- `results/phase2/`
- `results/phase3/`
- `results/phase3a/`
- `results/phase4/`
- `results/phase5/`

Archived non-canonical outputs live in `results/_archive/`.

## Current Structure

```text
results/
├── README.md
├── _archive/                 Archived non-canonical results
├── figures/                  Paper-facing figures
├── meta/                     Early extraction manifests
├── phase1_downloaded/        Canonical local copy of Phase 1 outputs
├── phase2/                   Stability analysis outputs
├── phase2_cache/             Embedding cache for Phase 2/3 analyses
├── phase3/                   State-Gini analysis outputs
├── phase3a/                  Phase 3 baseline outputs
├── phase3b/                  Additional Phase 3 plots/tables
├── phase4/                   DIR supporting experiment outputs
├── phase5/                   Canonical decision-evaluation outputs
├── raw/                      Raw model responses and diagnostics
├── test_grok/                Small test run outputs
├── test_phase1/              Small Phase 1 test run outputs
└── week2_run_*.csv           Early extraction outputs retained for provenance
```

## Paper-Backed Outputs

### Phase 1

Canonical local extraction artifacts used by later phases:

- `results/phase1_downloaded/*`

These are treated as frozen local inputs for downstream analyses.

### Phase 2

Stability analysis outputs:

- `results/phase2/plots/`
- `results/phase2/tables/`
- `results/phase2/phase2_manifest.json`

### Phase 3

Structure analysis outputs:

- `results/phase3/plots/`
- `results/phase3/tables/`
- `results/phase3/*.json`

Baseline comparison outputs:

- `results/phase3a/plots/`
- `results/phase3a/tables/`

### Phase 4

Supporting mechanism evidence (DIR):

- `results/phase4/dir_manifest_*.json`
- `results/phase4/dir_results_*.json`
- `results/phase4/dir_summary_*.json`
- `results/phase4/scaled_aggregate.json`

### Phase 5

Canonical decision-evaluation outputs:

- `results/phase5/baseline_manifest_openai_20260228_022320.json`
- `results/phase5/baseline_manifest_claude_20260228_032717.json`
- `results/phase5/baseline_summary_openai_20260228_022320.json`
- `results/phase5/baseline_summary_claude_20260228_032717.json`
- `results/phase5/baseline_legibility_analysis.json`
- `results/phase5/gentag_fer_disagreements.csv`
- `results/phase5/plots/`

## Historical And Exploratory Outputs

The following are retained for provenance or development history, but are not the main paper-backed artifact path:

- `results/meta/`
- `results/raw/`
- `results/test_grok/`
- `results/test_phase1/`
- `results/smoke_test_*.csv`
- `results/week2_run_*.csv`

If a current doc conflicts with archived or exploratory results, prefer the canonical phase directories above.

## Analysis Notebooks

Exploratory notebooks live in `notebooks/` and are not treated as canonical paper artifacts.

Currently retained:

- `notebooks/02_phase1_analysis.ipynb`
- `notebooks/04_stability_exploration.ipynb`
- `notebooks/phase2_explore.ipynb`

## Reproduction Guidance

Use these docs for paper verification and reruns:

- `docs/REPRODUCE_PAPER.md`
- `docs/PAPER_SOURCE_OF_TRUTH.md`
- `docs/PAPER_complete.md`
