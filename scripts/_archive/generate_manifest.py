#!/usr/bin/env python3
"""
Generate reproducibility manifest for experiment runs.

Usage:
    poetry run python scripts/generate_manifest.py --output results/meta/manifest_20250117.json
"""

import json
import sys
import subprocess
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import get_version_info


def get_git_info() -> Dict[str, Any]:
    """Get git commit hash and branch."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "is_dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip() != ""
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit_hash": None,
            "branch": None,
            "is_dirty": None
        }


def get_poetry_lock_hash() -> str:
    """Get hash of poetry.lock for dependency snapshot."""
    try:
        lock_path = Path(__file__).parent.parent / "poetry.lock"
        if lock_path.exists():
            import hashlib
            with open(lock_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        pass
    return None


def generate_manifest(
    dataset_name: str = None,
    dataset_path: str = None,
    row_count: int = None,
    sample_size: int = None,
    min_reviews: int = None,
    filters: Dict[str, Any] = None,
    output_path: str = None
) -> Dict[str, Any]:
    """
    Generate reproducibility manifest.
    
    Args:
        dataset_name: Name of dataset used
        dataset_path: Path to dataset file
        row_count: Number of rows/venues processed
        sample_size: Sample size if sampling was used
        min_reviews: Minimum reviews filter
        filters: Additional filter parameters
        output_path: Where to save manifest (optional)
    
    Returns:
        Manifest dict
    """
    version_info = get_version_info()
    git_info = get_git_info()
    
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git": git_info,
        "system": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "os": platform.system(),
            "os_version": platform.version(),
        },
        "pipeline": {
            "pipeline_version": version_info["pipeline_version"],
            "prompt_version": version_info["prompt_version"],
            "prompt_hash": version_info["prompt_hash"],
            "system_prompt_hash": version_info["system_prompt_hash"],
            "model_version": version_info["model_version"],
            "models": version_info["models"],
            "model_params": version_info["model_params"],
            "prompts": version_info["prompts"],
            "constraints": version_info["constraints"],
            "frozen_date": version_info["frozen_date"],
        },
        "dependencies": {
            "poetry_lock_hash": get_poetry_lock_hash(),
        },
        "dataset": {
            "name": dataset_name,
            "path": dataset_path,
            "row_count": row_count,
            "sample_size": sample_size,
            "min_reviews": min_reviews,
            "filters": filters or {},
        }
    }
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved to: {output_path}")
    
    return manifest


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reproducibility manifest")
    parser.add_argument("--output", type=str, default=None, help="Output path for manifest JSON")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name")
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset file path")
    parser.add_argument("--row-count", type=int, default=None, help="Number of rows processed")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size if sampling")
    parser.add_argument("--min-reviews", type=int, default=None, help="Minimum reviews filter")
    
    args = parser.parse_args()
    
    # Generate timestamped filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "results" / "meta"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(output_dir / f"manifest_{timestamp}.json")
    
    manifest = generate_manifest(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        row_count=args.row_count,
        sample_size=args.sample_size,
        min_reviews=args.min_reviews,
        output_path=args.output
    )
    
    print("\nManifest generated:")
    print(json.dumps(manifest, indent=2))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

