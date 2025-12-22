"""
Data loading: load_venue_data().
"""

import ast
from typing import Optional
import pandas as pd


def load_venue_data(csv_path: str, sample_size: Optional[int] = None, random_seed: int = 42) -> pd.DataFrame:
    """
    Load venue data and prepare for extraction.
    
    Args:
        csv_path: Path to venues_data.csv
        sample_size: Optional number of venues to sample
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with columns: id, name, google_reviews (as list of texts)
    
    Note: Ratings are explicitly excluded from the review data.
    """
    df = pd.read_csv(csv_path)
    
    # Extract review texts from google_reviews column
    # IMPORTANT: Only extracts 'text' field, ratings are explicitly excluded
    def extract_review_texts(raw_val):
        if pd.isna(raw_val):
            return []
        reviews = raw_val
        if isinstance(raw_val, str):
            try:
                reviews = ast.literal_eval(raw_val)
            except Exception:
                return []
        if not isinstance(reviews, list):
            return []
        
        texts = []
        for review in reviews:
            if isinstance(review, dict):
                # Only extract text, explicitly ignore 'rating' field
                text = review.get('text', '')
                if text and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                # Note: review.get('rating') is present but intentionally ignored
        return texts
    
    df['google_reviews'] = df['google_reviews'].apply(extract_review_texts)
    
    # Filter to venues with reviews
    df = df[df['google_reviews'].apply(len) > 0].copy()
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed)
    
    # Select relevant columns
    cols = ['id', 'name', 'google_reviews']
    if 'place_description' in df.columns:
        cols.append('place_description')
    if 'googleMapsTags' in df.columns:
        cols.append('googleMapsTags')
    
    return df[cols].reset_index(drop=True)

