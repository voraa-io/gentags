# Study 1 Dataset Card

## Overview

Study 1 uses venue review data from:

- `data/study1_venues_20250117.csv`

The canonical executed extraction set used by Phase 1 contains 553 venues, as reflected in:

- `results/phase1_downloaded/week2_run_20251223_191104_venues_gentags_summary.csv`

On that executed subset, venues contain 1-5 review objects each.

## Domain

- Domain: hospitality / food / cafe / bar / activity venues
- Unit of analysis: one venue
- Evidence source: review text stored in the `google_reviews` field
- Geographic scope: venue data used in the Gentags Study 1 pipeline

## Selection

- The working CSV contains 555 venue rows.
- The canonical executed Phase 1 extraction set contains 553 venues.
- Venue selection was fixed prior to representation evaluation.
- Venues were not filtered based on gentag quality, downstream performance, or representation outcomes.

## How It Is Used

- Phase 1: extraction of gentags from review text
- Phase 2: stability analysis
- Phase 3: structural analysis
- Phase 4: supporting DIR experiment inputs
- Phase 5: separate 50-venue stratified decision-evaluation subset derived for the paper's controlled downstream study

## Processing Notes

- The active extraction code keeps review text and explicitly ignores ratings during gentag extraction.
- The local canonical Phase 1 artifacts live in `results/phase1_downloaded/`.
- Downstream analyses use the frozen Phase 1 outputs rather than rerunning full extraction.

## Known Limitations

- The raw CSV and the executed Phase 1 subset are not the same size (555 rows in the CSV, 553 venues in the executed extraction set).
- Review counts are shallow on the executed subset; most venues have 5 review objects.
- Venue names are not unique identifiers; venue IDs should be used when tracing artifacts.
- This repository does not frame the dataset as representative of all venue domains or regions.

## Ethics And Privacy

- The repository uses review text as collected in the study dataset.
- Ratings are intentionally excluded from gentag extraction.
- This repo is intended for research on representation quality and decision behavior, not for profiling individuals.

## Canonical References

- `docs/EXTRACTION.md`
- `docs/REPRODUCE_PAPER.md`
- `docs/PAPER_SOURCE_OF_TRUTH.md`
