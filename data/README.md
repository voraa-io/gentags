# Data Directory

## Overview

This directory contains datasets used for gentag extraction experiments.

## Structure

```
data/
├── README.md          # This file
├── sample/            # Small public-safe sample dataset
│   └── venues_sample.csv
└── external/          # Private data (gitignored)
    └── venues_data.csv
```

## Sample Data

The `sample/` directory contains a small, anonymized sample dataset suitable for:

- Testing the pipeline
- Demonstrating functionality
- Public distribution

**Format:** CSV with columns:

- `id`: Venue identifier
- `name`: Venue name
- `google_reviews`: JSON list of review objects (only `text` field is used, `rating` is ignored)

## External Data

The `external/` directory is gitignored and should contain:

- Full datasets (not suitable for public distribution)
- Private venue data
- Any sensitive information

**To use external data:**

1. Place your data file in `data/external/`
2. Update paths in scripts/notebooks to point to `data/external/your_file.csv`

## Data Format

### Input CSV Format

Required columns:

- `id`: Unique venue identifier
- `name`: Venue name
- `google_reviews`: JSON string containing list of review objects

Optional columns:

- `place_description`: Venue description
- `googleMapsTags`: Existing tags (for comparison)

### Review Object Format

Each review in `google_reviews` should be a JSON object with:

- `text`: Review text (required, used for extraction)
- `rating`: Rating value (present but ignored per Study 1 protocol)

Example:

```json
[
  { "text": "Great atmosphere, friendly staff", "rating": 5 },
  { "text": "Food was excellent", "rating": 4 }
]
```

## Usage

```python
from gentags import load_venue_data

# Load sample data
df = load_venue_data("data/sample/venues_sample.csv", sample_size=10)

# Load external data
df = load_venue_data("data/external/venues_data.csv", sample_size=500)
```

## Notes

- Ratings are explicitly excluded from extraction (Study 1 protocol)
- Only review text is used for tag extraction
- Data is filtered to venues with at least one review

