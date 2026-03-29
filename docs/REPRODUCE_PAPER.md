# Reproduce The Paper

This is the shortest path to understanding and reproducing the paper-backed outputs in this repository.

## 1. Canonical Docs

- Paper draft: `docs/PAPER_complete.md`
- Claim-to-artifact map: `docs/PAPER_SOURCE_OF_TRUTH.md`
- Paper status: `docs/PAPER_STATUS.md`
- Cleanup / repo hygiene tracker: `codebase.md`

## 2. Install And Verify

Install the project:

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

## 3. Canonical Local Artifacts

These are the local artifact directories that back the paper and current analysis scripts:

- Phase 1 local copy: `results/phase1_downloaded/`
- Phase 2 outputs: `results/phase2/`
- Phase 2 embedding cache: `results/phase2_cache/`
- Phase 3 outputs: `results/phase3/`
- Phase 3A baseline outputs: `results/phase3a/`
- Phase 5 outputs: `results/phase5/`

Canonical Phase 5 files:

- `results/phase5/baseline_manifest_openai_20260228_022320.json`
- `results/phase5/baseline_manifest_claude_20260228_032717.json`
- `results/phase5/baseline_legibility_analysis.json`

## 4. Important Provenance Note

The full Phase 1 extraction appears to have been executed on a Google Cloud VM and then copied back into this repository. The working local copy used by later phases is `results/phase1_downloaded/`.

For current paper reproduction, treat `results/phase1_downloaded/` as the canonical local source. You do not need to rerun the original full extraction to regenerate the paper figures and tables.

## 5. What Is Frozen vs What To Rerun

Frozen / paper-backed:

- Phase 1 extracted outputs in `results/phase1_downloaded/`
- Phase 2 tables and plots in `results/phase2/`
- Phase 3 tables and plots in `results/phase3/` and `results/phase3a/`
- Phase 5 manifests, summaries, analysis JSON, audit CSV, and plots in `results/phase5/`

Reasonable to rerun locally:

- Unit tests
- Small Phase 1 dry-run or sample run
- Plot generation from existing analysis outputs
- Phase 2/3 analysis if you already have the required local artifacts and API keys

Expensive / not required for paper verification:

- Repeating the full 553-venue Phase 1 extraction
- Repeating all judge calls for Phase 5

## 6. Phase Commands

### Phase 1

Small sample run:

```bash
poetry run python scripts/run_phase1.py \
  --data data/study1_venues_20250117.csv \
  --sample-size 10
```

### Phase 2

Run the stability analysis using the local Phase 1 artifacts:

```bash
bash scripts/run_phase2.sh
```

Generate Phase 2 plots:

```bash
poetry run python scripts/phase2_plots.py
```

Notes:

- First run computes embeddings and requires `OPENAI_API_KEY`.
- Later runs can reuse `results/phase2_cache/`.

### Phase 3

Run the State-Gini analysis and baseline comparison:

```bash
bash scripts/run_phase3.sh
```

Generate Phase 3 plots:

```bash
poetry run python scripts/phase3_plots.py
```

Notes:

- Requires local Phase 1 artifacts in `results/phase1_downloaded/`
- Requires Phase 2 embedding cache in `results/phase2_cache/`
- Requires `OPENAI_API_KEY` for anchor embeddings

### Phase 5

Recompute the Phase 5 aggregate analysis from canonical summaries:

```bash
poetry run python scripts/phase5_analyze.py \
  --summary results/phase5/baseline_summary_openai_20260228_022320.json \
  --summary2 results/phase5/baseline_summary_claude_20260228_032717.json
```

Generate Phase 5 plots:

```bash
poetry run python scripts/phase5_plots.py
```

Re-running the full judge study is possible via `scripts/phase5_baseline_runner.py`, but it is not necessary for paper verification.

## 7. Figure And Claim Traceability

Use these docs together:

- `docs/PAPER_complete.md` for the written paper
- `docs/PAPER_SOURCE_OF_TRUTH.md` for claim-to-artifact mapping
- `codebase.md` for current cleanup decisions and known repo drift

If docs conflict with executed artifacts or active scripts, prefer:

1. `results/`
2. `data/`
3. `scripts/`
4. current `docs/`
5. archived docs only for background
