"""
I/O helpers: save_results(), save_raw_response(), load_results().
"""

from pathlib import Path
from datetime import datetime
import pandas as pd


def save_results(tags_df: pd.DataFrame, output_dir: str = "results", prefix: str = "gentags") -> str:
    """
    Save results to CSV with timestamp.
    
    Args:
        tags_df: DataFrame with results
        output_dir: Output directory
        prefix: Filename prefix
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{prefix}_{timestamp}.csv"
    
    tags_df.to_csv(filename, index=False)
    return str(filename)


def save_raw_response(raw_response: str, exp_id: str, run_id: str, output_dir: str = "results/raw") -> str:
    """
    Save raw model response to file (for debugging parse errors).
    
    Args:
        raw_response: Raw response text
        exp_id: Experiment ID
        run_id: Run ID
        output_dir: Output directory
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f"{exp_id}_{run_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(raw_response or "")
    return str(filename)


def load_results(filepath: str) -> pd.DataFrame:
    """
    Load results from CSV.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame with results
    """
    return pd.read_csv(filepath)

