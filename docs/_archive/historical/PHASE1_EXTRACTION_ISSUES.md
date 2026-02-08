# Phase 1 Extraction Issues

**Created:** 2026-01-24
**Run ID:** week2_run_20251223_191104

---

## Summary

During Phase 1 extraction, two models (Claude and Grok) experienced significant API failures, resulting in incomplete data for a substantial portion of venues.

---

## Extraction Success Rates by Model

| Model | Total | Success | Error | Success Rate |
|-------|-------|---------|-------|--------------|
| gemini | 3,318 | 3,318 | 0 | **100.0%** |
| openai | 3,318 | 3,318 | 0 | **100.0%** |
| grok | 3,318 | 2,117 | 1,201 | **63.8%** |
| claude | 3,318 | 1,621 | 1,697 | **48.9%** |

**Total extractions:** 13,272
**Successful:** 10,374 (78.2%)
**Failed:** 2,898 (21.8%)

---

## Root Cause Analysis

### What Happened

The `status="error"` extractions have:
- `n_tags = 0`
- `raw_tags_json = NaN` (null)
- No output tokens recorded

This indicates the API call either:
1. **Timed out** — model took too long to respond
2. **Rate limited** — too many requests in short period
3. **Parse failure** — model returned non-JSON output that couldn't be parsed
4. **API error** — temporary service unavailability

### Why Claude and Grok Were Affected

Both Claude and Grok are newer/less stable APIs compared to OpenAI and Gemini:
- **Claude:** Anthropic's API may have had rate limiting or timeout issues during the extraction run
- **Grok:** xAI's API is newer and may have reliability issues

OpenAI and Gemini have mature, battle-tested APIs with better error handling.

### Venue Coverage

| Model | Venues with ≥1 Success | Venues with ALL Errors |
|-------|------------------------|------------------------|
| gemini | 553 (100%) | 0 |
| openai | 553 (100%) | 0 |
| grok | 357 (64.6%) | 196 (35.4%) |
| claude | 271 (49.0%) | 282 (51.0%) |

---

## Impact on Analysis

### Phase 2 Stability Analysis

When computing run stability (run1 vs run2):
- If **both runs failed** → 0 tags in both → cosine similarity = 0.0
- If **one run failed** → comparison is invalid

This caused:
- Claude median cosine = 0.0 (due to 51% failures)
- Grok also affected but less severely
- "77.1% show cosine > jaccard" metric was dragged down by zeros

### Correct Interpretation

The 0.0 cosine values are **not analysis bugs** — they correctly reflect missing data. The analysis code is working correctly; the input data is incomplete.

---

## Mitigation

### Option 1: Filter to Complete Venues Only (Implemented ✅)

Modify Phase 2 analysis to:
1. Load extraction data
2. Filter to `status == "success"` only
3. **Keep only venues with successful extractions from ALL 4 models**
4. Proceed with analysis on clean, comparable subset

This ensures fair model comparisons — we only analyze venues where all models succeeded.

### Option 2: Re-run Failed Extractions

Re-run the extraction pipeline for failed venues:
- Requires API access and budget
- May still fail if underlying issue persists
- Could add retry logic with exponential backoff

### Option 3: Report As-Is with Caveat

Report full results but note:
- Claude and Grok have incomplete data
- Metrics for these models are lower bounds
- Full comparison requires complete extraction

---

## Recommendations

1. **For this study:** Use Option 1 (filter to successful extractions)
2. **For future runs:**
   - Add more robust retry logic
   - Implement checkpointing to resume failed extractions
   - Consider running models sequentially to avoid rate limits
   - Log detailed error messages for debugging

---

## Data Files

**Raw extraction files with errors:**
```
results/phase1_downloaded/
├── week2_run_20251223_191104_extractions_claude.csv  # 51.1% errors
├── week2_run_20251223_191104_extractions_grok.csv    # 36.2% errors
├── week2_run_20251223_191104_extractions_gemini.csv  # 0% errors
└── week2_run_20251223_191104_extractions_openai.csv  # 0% errors
```

**To check error rates:**
```python
import pandas as pd
df = pd.read_csv('results/phase1_downloaded/week2_run_20251223_191104_extractions_claude.csv')
print(df['status'].value_counts())
```
