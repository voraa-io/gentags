# Week 2 Phase-1 Extraction: Complete ✅

**Date Completed:** December 25, 2024  
**Run ID:** `week2_run_20251223_191104`

---

## 🎯 Objective

Generate gentags for all venues across multiple models and prompts to create the core dataset for Study 1 stability analysis.

---

## ✅ What Was Accomplished

### Extraction Matrix

- **Venues:** 553 (all venues with valid reviews)
- **Models:** 4 (OpenAI, Gemini, Claude, Grok)
- **Prompts:** 3 (minimal, anti_hallucination, short_phrase)
- **Runs per combination:** 2
- **Total extractions:** 13,272 (553 venues × 4 models × 3 prompts × 2 runs)

### Execution Details

- **Execution Environment:** Google Cloud Compute Engine VM (`voraa-gentags`)
- **Parallelization:** 4 shards (one per model) running in parallel
- **Checkpointing:** Atomic checkpoint saves every 50 extractions
- **Resume Capability:** Full checkpoint/resume support (not needed - all completed successfully)

---

## 📊 Results Summary

### Model Completion Status

| Model      | Status          | Extractions       | Tags Extracted | Cost (USD)  |
| ---------- | --------------- | ----------------- | -------------- | ----------- |
| **Claude** | ✅ Complete     | 3,318/3,318       | ~68,530        | $0.62       |
| **Gemini** | ✅ Complete     | 3,318/3,318       | ~68,530        | $0.62       |
| **OpenAI** | ✅ Complete     | 3,318/3,318       | ~80,796        | $4.24       |
| **Grok**   | ✅ Complete     | 3,318/3,318       | ~47,986        | $7.13       |
| **TOTAL**  | ✅ **Complete** | **13,272/13,272** | **~265,842**   | **~$12.61** |

### Cost Breakdown

- **Total Cost:** ~$12.61 USD
- **Average Cost per Extraction:** ~$0.00095
- **Average Cost per Tag:** ~$0.000047

**Cost by Model:**

- Claude: $0.62 (most cost-effective)
- Gemini: $0.62 (most cost-effective)
- OpenAI: $4.24 (moderate)
- Grok: $7.13 (highest cost, but still reasonable)

---

## 📁 Output Files

All files are located in `/mnt/results/results/` on the VM (or can be downloaded).

### Per-Model Output Files

For each model (`openai`, `gemini`, `claude`, `grok`):

1. **Tags CSV** (`week2_run_20251223_191104_tags_{model}.csv`)

   - One row per tag
   - Columns: `run_id`, `exp_id`, `venue_id`, `venue_name`, `model`, `prompt_type`, `run_number`, `tag_raw`, `tag_norm`, `tag_norm_eval`, `timestamp`, `status`, `time_seconds`, `input_tokens`, `output_tokens`, `total_tokens`, `cost_usd`

2. **Extractions CSV** (`week2_run_20251223_191104_extractions_{model}.csv`)

   - One row per extraction (deduplicated)
   - Columns: extraction-level metrics, `n_tags`, `n_unique_tag_eval`, `cost_per_tag`, `cost_per_unique_tag`, `raw_tags_json`, `tags_filtered_count`

3. **Cost Breakdown CSV** (`week2_run_20251223_191104_cost_by_model_prompt_{model}.csv`)

   - Aggregated cost metrics by model/prompt combination
   - Columns: `model`, `prompt_type`, `n_extractions`, `cost_total`, `cost_mean`, `tokens_mean`, `tags_mean`, `unique_tags_mean`, `cost_per_tag_mean`

4. **Manifest JSON** (`meta/week2_run_20251223_191104_manifest_{model}.json`)
   - Reproducibility metadata: git commit, timestamp, dataset info, pipeline versions, model/prompt hashes, system/dependency details

### Checkpoint Files

- `week2_run_20251223_191104_checkpoint_{model}.csv` (final checkpoints, can be used for resume if needed)

### Log Files

- `/mnt/results/results/logs/{model}.log` - Full execution logs with progress bars and summaries

---

## 🔧 Technical Implementation

### Key Features Implemented

1. **Cost Guard**

   - Pilot run cost estimation
   - User confirmation for runs exceeding budget
   - Per-shard budget calculation for parallel runs

2. **Checkpointing & Resume**

   - Atomic checkpoint writes (temp file → rename)
   - Periodic saves every 50 extractions
   - Automatic resume by skipping completed `exp_id`s
   - Only resumes successful extractions (skips errors)

3. **Parallel Execution (Sharding)**

   - Model-based sharding (one process per model)
   - Shard-specific filenames to prevent collisions
   - Token-based lockfiles to prevent duplicate runs
   - Independent checkpoint files per shard

4. **Progress Tracking**

   - Real-time progress bars with `tqdm`
   - Status updates with emojis (✓ success, ⚠ parse_error, ✗ error)
   - Cost and time per extraction displayed
   - ETA calculations

5. **Robustness**

   - Retry logic with exponential backoff and jitter
   - 120s timeout for API calls
   - Graceful handling of `None` cost values
   - Atomic writes for all final output CSVs

6. **Cost Tracking**
   - Accurate pricing for all models (including Grok: $2/M input, $10/M output)
   - Per-extraction cost logging
   - Aggregated cost summaries by model/prompt

---

## 🐛 Issues Fixed During Execution

1. **Grok Cost Calculation Bug**

   - **Issue:** Grok pricing was set to `0.0`, causing cost to be `None` and crashing progress bar
   - **Fix:** Updated Grok pricing to actual values ($2/M input, $10/M output) and handled `None` cost in progress bar display

2. **Progress Bar Formatting**
   - **Issue:** `TypeError` when formatting `None` cost values
   - **Fix:** Added conditional formatting to display "N/A" when cost is `None`

---

## 📈 Data Quality

### Extraction Success Rate

- All models completed 100% of extractions (3,318/3,318 per model)
- No crashes or data loss
- All checkpoints saved successfully

### Tag Extraction

- Average tags per extraction varies by model:
  - Claude: ~20.7 tags/extraction
  - Gemini: ~20.7 tags/extraction
  - OpenAI: ~24.4 tags/extraction
  - Grok: ~14.5 tags/extraction

---

## 🚀 Next Steps

### Step 2: Stability Analysis (RQ1–RQ3)

**Research Questions:**

1. **RQ1:** Are gentags stable across runs? (Jaccard overlap between run 1 and run 2)
2. **RQ2:** Prompt sensitivity - How do tags differ across prompts (same model)?
3. **RQ3:** Model sensitivity - How do tags differ across models (same prompt)?

**Analysis Tasks:**

- [ ] Load and merge all 4 model outputs
- [ ] Compute Jaccard similarity between runs (same venue/model/prompt)
- [ ] Compare prompt sensitivity (same venue/model, different prompts)
- [ ] Compare model sensitivity (same venue/prompt, different models)
- [ ] Visualize stability metrics (heatmaps, distributions)
- [ ] Identify systematic vs. noisy drift patterns

### Step 3: Tag Behavior Diagnostics

- [ ] Tag count distributions per model/prompt
- [ ] Long-tail tag analysis
- [ ] Redundancy detection ("great food" vs "delicious food")
- [ ] Optional: Embedding-based clustering analysis

---

## 📝 Notes

- All extractions used frozen prompts and models (Study 1 lock)
- All outputs include reproducibility manifests
- Checkpoint files can be used to resume if needed (though all completed successfully)
- Logs are available for debugging and verification
- Total execution time: ~20-25 hours (parallel shards)

---

## ✅ Completion Checklist

- [x] All 4 models extracted successfully
- [x] All output files generated (tags, extractions, cost breakdowns, manifests)
- [x] Checkpoints saved
- [x] Logs captured
- [x] Cost tracking accurate
- [x] No data loss or corruption
- [x] Ready for analysis phase

---

**Status:** ✅ **Week 2 Phase-1 Extraction COMPLETE**

Ready to proceed to stability analysis and tag behavior diagnostics.
