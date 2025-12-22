# Results Directory

## Overview

This directory contains output from gentag extraction experiments. Most results are gitignored to avoid committing large files or sensitive data.

## Structure

```
results/
├── README.md          # This file
├── meta/              # Metadata files (public-safe)
│   └── [manifest files]
├── examples/           # Small example outputs (optional)
│   └── [example files]
├── raw/               # Raw model responses (gitignored)
│   └── [error responses]
└── [timestamped CSV files]  # Main results (gitignored)
```

## Output Format

### Main Results CSV

Results are saved as CSV files with naming convention:
```
gentags_YYYYMMDD_HHMMSS.csv
phase1_YYYYMMDD_HHMMSS.csv
```

**Columns:**
- `run_id`: Unique run identifier
- `venue_id`: Venue identifier
- `venue_name`: Venue name
- `model`: Model name used
- `prompt_type`: Prompt type used
- `run_number`: Run number (1, 2, ...)
- `exp_id`: Experiment identifier
- `timestamp`: Extraction timestamp
- `tag_raw`: Raw extracted tag
- `tag_norm`: Normalized tag (for matching)
- `tag_norm_eval`: Strictly normalized tag (for stability metrics)
- `word_count`: Number of words in tag
- `num_reviews`: Number of reviews for venue
- `reviews_total_chars`: Total characters in reviews
- `time_seconds`: Extraction time
- `input_tokens`: Input token count
- `output_tokens`: Output token count
- `total_tokens`: Total token count
- `cost_usd`: Estimated cost in USD
- `status`: Extraction status ("success", "parse_error", "error")
- `prompt_hash`: Hash of prompt version
- `system_prompt_hash`: Hash of system prompt
- `input_prompt_hash`: Hash of exact prompt sent
- `tags_filtered_count`: Number of tags filtered out
- `extraction_phase`: Phase identifier ("phase1")

### Raw Responses

Raw model responses are saved in `results/raw/` for:
- Debugging parse errors
- Auditing model outputs
- Error analysis

**Naming:** `{exp_id}_{run_id}.txt`

### Metadata Files

Metadata files in `results/meta/` contain:
- Environment information
- Data manifest
- Run configuration

## Usage

```python
from gentags import load_results

# Load results
df = load_results("results/gentags_20250117_120000.csv")

# Filter to successful extractions
df_success = df[df['status'] == 'success']
df_tags = df_success[df_success['tag_raw'].notna()]
```

## Analysis

See notebooks for analysis:
- `notebooks/02_phase1_analysis.ipynb`: Stability analysis
- `notebooks/03_reconstruction_eval.ipynb`: Reconstruction evaluation

## Gitignore

The following are gitignored:
- `*.csv` (main results files)
- `raw/` (raw responses)
- Large output files

The following are committed:
- `README.md` (this file)
- `meta/` (metadata, if public-safe)
- `examples/` (small examples, if included)

