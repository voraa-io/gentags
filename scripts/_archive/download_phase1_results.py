#!/usr/bin/env python3
"""
Download Phase 1 results from VM and create aggregated tables.

Downloads:
- Tag CSVs
- Extractions CSVs
- Cost summaries

Creates:
- venues_gentags_by_model_prompt_run.csv: Analysis-ready table with venue×model×prompt×run combinations
- venues_gentags_summary.csv: Browsing-friendly table with venues and all their tags (runs merged)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def ensure_model_key(df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
    """Ensure model_key column exists, inferring from model column or filename if needed."""
    if "model_key" in df.columns:
        return df
    
    # Try mapping from model column
    if "model" in df.columns:
        model_map = {
            "gpt-5-nano": "openai",
            "gemini-2.5-flash": "gemini",
            "claude-sonnet-4-5": "claude",
            "grok-4": "grok"
        }
        df["model_key"] = df["model"].map(model_map)
        if df["model_key"].notna().any():
            return df
    
    # Infer from filename
    filename_lower = filename.lower()
    if "openai" in filename_lower:
        df["model_key"] = "openai"
    elif "gemini" in filename_lower:
        df["model_key"] = "gemini"
    elif "claude" in filename_lower:
        df["model_key"] = "claude"
    elif "grok" in filename_lower:
        df["model_key"] = "grok"
    else:
        # Fallback: leave as None and let user fix
        df["model_key"] = None
    
    return df


def list_remote_files(vm_name: str, zone: str, remote_dir: str, pattern: str) -> list:
    """List files matching pattern on remote VM."""
    try:
        result = subprocess.run([
            "gcloud", "compute", "ssh", vm_name,
            f"--zone={zone}",
            f"--command=ls -1 {remote_dir}/{pattern} 2>/dev/null || echo ''"
        ], capture_output=True, text=True, check=True)
        
        files = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return files
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to list remote files: {e}")
        return []


def download_from_vm(run_id: str, vm_name: str, zone: str, vm_results_dir: str, local_output_dir: str):
    """Download Phase 1 result files from VM (handles wildcards reliably)."""
    local_output = Path(local_output_dir)
    local_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Phase 1 results for {run_id} from {vm_name}...")
    
    # Patterns to find
    patterns = [
        f"{run_id}_tags_*.csv",
        f"{run_id}_extractions_*.csv",
        f"{run_id}_cost_by_model_prompt_*.csv",
    ]
    
    all_files = []
    for pattern in patterns:
        print(f"  Finding files matching {pattern}...")
        files = list_remote_files(vm_name, zone, vm_results_dir, pattern)
        all_files.extend(files)
    
    if not all_files:
        print(f"⚠️  No files found matching patterns. Check run_id: {run_id}")
        return
    
    print(f"  Found {len(all_files)} files to download")
    
    # Download each file explicitly
    for remote_file in all_files:
        filename = Path(remote_file).name
        print(f"  Downloading {filename}...")
        try:
            subprocess.run([
                "gcloud", "compute", "scp",
                f"{vm_name}:{remote_file}",
                str(local_output / filename),
                f"--zone={zone}"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"    ⚠️  Failed to download {filename}: {e.stderr.decode() if e.stderr else str(e)}")
    
    print(f"✅ Files downloaded to {local_output}")


def create_analysis_table(run_id: str, local_dir: str) -> pd.DataFrame:
    """
    Create analysis-ready table: venue×model×prompt×run (one row per extraction).
    
    Columns:
    - venue_id, venue_name, model_key, prompt_type, run_number
    - tags_raw_json, tags_norm_eval_json
    - n_tags, n_unique_norm_eval
    """
    local_dir = Path(local_dir)
    
    # Load all tag files
    tag_files = list(local_dir.glob(f"{run_id}_tags_*.csv"))
    if not tag_files:
        raise FileNotFoundError(f"No tag files found for {run_id} in {local_dir}")
    
    print(f"Loading {len(tag_files)} tag files...")
    all_tags = []
    for file in tag_files:
        df = pd.read_csv(file)
        df = ensure_model_key(df, file.name)
        all_tags.append(df)
    
    tags_df = pd.concat(all_tags, ignore_index=True)
    print(f"Loaded {len(tags_df)} tag rows")
    
    # Group by extraction (venue×model×prompt×run) and aggregate tags
    results = []
    
    for (venue_id, venue_name, model_key, prompt_type, run_number), group in tags_df.groupby(
        ["venue_id", "venue_name", "model_key", "prompt_type", "run_number"]
    ):
        tags_raw = group["tag_raw"].dropna().unique().tolist()
        tags_norm_eval = group["tag_norm_eval"].dropna().unique().tolist()
        
        results.append({
            "venue_id": venue_id,
            "venue_name": venue_name,
            "model_key": model_key,
            "prompt_type": prompt_type,
            "run_number": run_number,
            "tags_raw_json": json.dumps(tags_raw) if tags_raw else None,
            "tags_norm_eval_json": json.dumps(tags_norm_eval) if tags_norm_eval else None,
            "n_tags": len(tags_raw),
            "n_unique_norm_eval": len(tags_norm_eval),
        })
    
    return pd.DataFrame(results)


def create_browsing_table(run_id: str, local_dir: str) -> pd.DataFrame:
    """
    Create browsing-friendly table: venues with all tags merged across runs.
    
    Columns:
    - venue_id, venue_name
    - For each model×prompt: tags_json, tags_count, tags_preview (first 5 tags)
    """
    local_dir = Path(local_dir)
    
    # Load all tag files
    tag_files = list(local_dir.glob(f"{run_id}_tags_*.csv"))
    if not tag_files:
        raise FileNotFoundError(f"No tag files found for {run_id} in {local_dir}")
    
    print(f"Loading {len(tag_files)} tag files...")
    all_tags = []
    for file in tag_files:
        df = pd.read_csv(file)
        df = ensure_model_key(df, file.name)
        all_tags.append(df)
    
    tags_df = pd.concat(all_tags, ignore_index=True)
    print(f"Loaded {len(tags_df)} tag rows")
    
    # Group by venue, then by model×prompt (merge runs)
    results = []
    
    for (venue_id, venue_name), venue_group in tags_df.groupby(["venue_id", "venue_name"]):
        row = {
            "venue_id": venue_id,
            "venue_name": venue_name,
        }
        
        # For each model×prompt, aggregate tags across all runs
        for (model, prompt), combo_group in venue_group.groupby(["model_key", "prompt_type"]):
            tags_raw = combo_group["tag_raw"].dropna().unique().tolist()
            tags_norm_eval = combo_group["tag_norm_eval"].dropna().unique().tolist()
            
            col_prefix = f"{model}_{prompt}"
            row[f"{col_prefix}_tags_json"] = json.dumps(tags_raw) if tags_raw else None
            row[f"{col_prefix}_tags_count"] = len(tags_raw)
            row[f"{col_prefix}_tags_unique_norm"] = len(tags_norm_eval)
            row[f"{col_prefix}_tags_preview"] = ", ".join(tags_raw[:5]) if tags_raw else ""  # First 5 only, for browsing
        
        results.append(row)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Download Phase 1 results from VM")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID (e.g., week2_run_20251223_191104)"
    )
    parser.add_argument(
        "--vm-name",
        type=str,
        default="voraa-gentags",
        help="VM name"
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="us-central1-c",
        help="VM zone"
    )
    parser.add_argument(
        "--vm-results-dir",
        type=str,
        default="/mnt/results/results",
        help="Results directory on VM"
    )
    parser.add_argument(
        "--local-output",
        type=str,
        default="results/phase1_downloaded",
        help="Local directory to save downloaded files"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download, use existing files in --local-output"
    )
    parser.add_argument(
        "--analysis-table",
        action="store_true",
        help="Create analysis-ready table (venue×model×prompt×run)"
    )
    parser.add_argument(
        "--browsing-table",
        action="store_true",
        help="Create browsing-friendly table (venues with tags merged across runs)"
    )
    
    args = parser.parse_args()
    
    # Default: create both tables if neither flag specified
    create_analysis = args.analysis_table
    create_browsing = args.browsing_table
    if not create_analysis and not create_browsing:
        create_analysis = True
        create_browsing = True
    
    local_output = Path(args.local_output)
    local_output.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_download:
        download_from_vm(
            args.run_id,
            args.vm_name,
            args.zone,
            args.vm_results_dir,
            args.local_output
        )
    
    if create_analysis:
        print("\nCreating analysis-ready table...")
        analysis_table = create_analysis_table(args.run_id, args.local_output)
        
        output_path = local_output / f"{args.run_id}_venues_gentags_by_model_prompt_run.csv"
        analysis_table.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
        print(f"   Table shape: {analysis_table.shape}")
        print(f"   Columns: {list(analysis_table.columns)}")
    
    if create_browsing:
        print("\nCreating browsing-friendly table...")
        browsing_table = create_browsing_table(args.run_id, args.local_output)
        
        output_path = local_output / f"{args.run_id}_venues_gentags_summary.csv"
        browsing_table.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")
        print(f"   Table shape: {browsing_table.shape}")
        print(f"   Columns: {len(browsing_table.columns)} (includes all model×prompt combinations)")


if __name__ == "__main__":
    main()
