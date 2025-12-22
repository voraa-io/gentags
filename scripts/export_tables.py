#!/usr/bin/env python3
"""
Export tables: generates paper tables from results.

Usage:
    python scripts/export_tables.py results/gentags_*.csv --output tables/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import load_results


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from results")
    parser.add_argument(
        "file",
        type=str,
        help="Path to results CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tables",
        help="Output directory for tables"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.file}...")
    df = load_results(args.file)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to successful extractions with tags
    df_tags = df[df['tag_raw'].notna()].copy()
    
    # Table 1: Summary statistics by model
    print("\nGenerating Table 1: Summary by model...")
    table1 = df_tags.groupby('model').agg({
        'exp_id': 'nunique',
        'tag_raw': 'count',
        'tag_raw': lambda x: x.nunique(),
        'word_count': 'mean',
        'cost_usd': 'sum'
    }).round(2)
    table1.columns = ['Experiments', 'Total Tags', 'Unique Tags', 'Avg Words', 'Total Cost (USD)']
    table1_path = output_dir / "table1_model_summary.csv"
    table1.to_csv(table1_path)
    print(f"  Saved to {table1_path}")
    
    # Table 2: Summary statistics by prompt
    print("\nGenerating Table 2: Summary by prompt...")
    table2 = df_tags.groupby('prompt_type').agg({
        'exp_id': 'nunique',
        'tag_raw': 'count',
        'tag_raw': lambda x: x.nunique(),
        'word_count': 'mean'
    }).round(2)
    table2.columns = ['Experiments', 'Total Tags', 'Unique Tags', 'Avg Words']
    table2_path = output_dir / "table2_prompt_summary.csv"
    table2.to_csv(table2_path)
    print(f"  Saved to {table2_path}")
    
    # Table 3: Tag frequency (top N)
    print("\nGenerating Table 3: Most frequent tags...")
    top_n = 50
    table3 = df_tags['tag_norm'].value_counts().head(top_n).reset_index()
    table3.columns = ['Tag', 'Frequency']
    table3_path = output_dir / "table3_top_tags.csv"
    table3.to_csv(table3_path, index=False)
    print(f"  Saved to {table3_path}")
    
    print(f"\nAll tables saved to {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

