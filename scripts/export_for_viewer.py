#!/usr/bin/env python3
"""
Export for viewer: converts results to JSON for UI.

Usage:
    python scripts/export_for_viewer.py results/gentags_*.csv --output viewer_data.json
"""

import argparse
import json
import sys
from pathlib import Path
from glob import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import load_results


def main():
    parser = argparse.ArgumentParser(description="Export results to JSON for viewer")
    parser.add_argument(
        "file",
        type=str,
        help="Path to results CSV file (supports glob patterns)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="viewer_data.json",
        help="Output JSON file (default: viewer_data.json)"
    )
    
    args = parser.parse_args()
    
    # Handle glob patterns
    files = glob(args.file)
    if not files:
        print(f"ERROR: No files found matching {args.file}")
        return 1
    
    if len(files) > 1:
        print(f"WARNING: Multiple files found, using first: {files[0]}")
    
    # Load results
    print(f"Loading results from {files[0]}...")
    df = load_results(files[0])
    
    # Filter to successful extractions with tags
    df_tags = df[df['tag_raw'].notna()].copy()
    
    # Group by venue
    venues = {}
    for _, row in df_tags.iterrows():
        venue_id = row['venue_id']
        if venue_id not in venues:
            venues[venue_id] = {
                "venue_id": venue_id,
                "venue_name": row['venue_name'],
                "tags": [],
                "experiments": {}
            }
        
        exp_id = row['exp_id']
        if exp_id not in venues[venue_id]['experiments']:
            venues[venue_id]['experiments'][exp_id] = {
                "exp_id": exp_id,
                "model": row['model'],
                "model_key": row.get('model_key', 'unknown'),
                "prompt_type": row['prompt_type'],
                "run_number": row['run_number'],
                "tags": []
            }
        
        venues[venue_id]['experiments'][exp_id]['tags'].append({
            "tag_raw": row['tag_raw'],
            "tag_norm": row['tag_norm'],
            "word_count": row['word_count']
        })
        
        # Collect unique tags for venue
        if row['tag_raw'] not in venues[venue_id]['tags']:
            venues[venue_id]['tags'].append(row['tag_raw'])
    
    # Convert to list format
    output_data = {
        "metadata": {
            "total_venues": len(venues),
            "total_experiments": df['exp_id'].nunique(),
            "total_tags": len(df_tags)
        },
        "venues": list(venues.values())
    }
    
    # Save JSON
    print(f"Saving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported {len(venues)} venues to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

