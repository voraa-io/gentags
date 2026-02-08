#!/usr/bin/env python3
"""
Phase 3: State Localization Analysis (CORRECTED VERSION)

This script computes STATE-GINI: Gini coefficient on facet counts.
This measures: "Is the representation localized/factorized?"

Key difference from v1:
- v1 computed DRIFT-GINI (change between runs) - wrong metric for "localized state"
- v2 computes STATE-GINI (counts per facet for single extraction) - correct metric

Methods:
- Hard semantic assignment: each tag → argmax facet (above threshold)
- Gini computed on integer counts, not normalized proportions
- Same method applied to gentags, RAKE, TF-IDF, YAKE for fair comparison

Output:
- state_localization.csv: State-Gini per venue/method
- drift_localization.csv: Drift-Gini (secondary, for completeness)
"""

import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

# Directories
PHASE2_DIR = Path("results/phase2")
PHASE2_TABLES = PHASE2_DIR / "tables"
PHASE2_CACHE = Path("results/phase2_cache")

OUTPUT_DIR = Path("results/phase3")
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Semantic threshold for hard assignment
SEMANTIC_THRESHOLD = 0.35

# =============================================================================
# FACET DEFINITIONS (10 semantic dimensions)
# =============================================================================

FACETS = [
    "food_quality",
    "coffee_drinks",
    "service",
    "ambiance",
    "price_value",
    "crowding",
    "seating",
    "dietary",
    "portions",
    "location"
]

# Fixed facet anchors (method-neutral, no circular reasoning)
FACET_ANCHORS = {
    "food_quality": "food quality, taste, freshness, delicious meals",
    "coffee_drinks": "coffee, espresso, latte, beverages, drinks",
    "service": "service quality, staff friendliness, speed, waiters",
    "ambiance": "atmosphere, ambiance, vibe, decor, cozy environment",
    "price_value": "price, value for money, affordable, expensive",
    "crowding": "crowded, busy, wait times, lines, availability",
    "seating": "seating, tables, outdoor patio, indoor space",
    "dietary": "dietary options, vegan, vegetarian, gluten-free",
    "portions": "portion size, generous servings, filling meals",
    "location": "location, parking, accessibility, neighborhood",
}

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of a distribution.

    - High Gini (→1): Values concentrated (LOCALIZED)
    - Low Gini (→0): Values spread evenly (DIFFUSE)
    """
    values = np.abs(values).astype(float)
    if values.sum() == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)

    # Gini formula
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return max(0.0, gini)


def hard_assign_facet(
    tag: str,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> Tuple[Optional[str], float]:
    """
    Hard assignment: assign tag to exactly ONE facet via argmax.

    Returns: (facet_name or None if below threshold, similarity score)
    """
    if tag not in tag_embeddings:
        return None, 0.0

    tag_emb = tag_embeddings[tag]
    best_facet = None
    best_sim = -1.0

    for facet in FACETS:
        sim = cosine_similarity(tag_emb, anchor_embeddings[facet])
        if sim > best_sim:
            best_sim = sim
            best_facet = facet

    if best_sim >= threshold:
        return best_facet, best_sim
    else:
        return None, best_sim  # Below threshold → "other"


def compute_facet_counts(
    tags: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> Tuple[Dict[str, int], int]:
    """
    Compute integer counts per facet using hard assignment.

    Returns: (counts_dict, other_count)
    """
    counts = {facet: 0 for facet in FACETS}
    other_count = 0

    for tag in tags:
        facet, sim = hard_assign_facet(tag, tag_embeddings, anchor_embeddings, threshold)
        if facet is not None:
            counts[facet] += 1
        else:
            other_count += 1

    return counts, other_count


def compute_state_gini(
    tags: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> Tuple[float, Dict[str, int], int]:
    """
    Compute STATE-GINI: Gini coefficient on facet counts.

    This is the CORRECT metric for "localized state":
    - Each tag is assigned to exactly one facet (hard assignment)
    - Gini is computed on integer counts
    - High Gini = representation concentrated in few facets
    - Low Gini = representation spread across many facets

    Returns: (state_gini, counts_dict, other_count)
    """
    counts, other_count = compute_facet_counts(tags, tag_embeddings, anchor_embeddings, threshold)

    # Gini on raw integer counts
    count_array = np.array([counts[f] for f in FACETS])
    state_gini = gini_coefficient(count_array)

    return state_gini, counts, other_count


def compute_drift_gini(
    tags1: List[str],
    tags2: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> float:
    """
    Compute DRIFT-GINI: Gini on change between two snapshots.

    This is a SECONDARY metric (not the main "localized state" claim).
    Measures: "When the representation changes, is the change concentrated?"
    """
    counts1, _ = compute_facet_counts(tags1, tag_embeddings, anchor_embeddings, threshold)
    counts2, _ = compute_facet_counts(tags2, tag_embeddings, anchor_embeddings, threshold)

    # Normalize to proportions
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())

    if total1 == 0 or total2 == 0:
        return 0.0

    profile1 = np.array([counts1[f] / total1 for f in FACETS])
    profile2 = np.array([counts2[f] / total2 for f in FACETS])

    drift = np.abs(profile1 - profile2)
    return gini_coefficient(drift)


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_embedding_client():
    """Initialize OpenAI embedding client."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv

    for path in [Path(__file__).parent.parent / ".env", Path.cwd() / ".env"]:
        if path.exists():
            load_dotenv(path)
            break

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


def embed_texts_batch(client, texts: List[str], batch_size: int = 128) -> List[np.ndarray]:
    """Embed texts in batches."""
    import time

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i:i + batch_size]

        for attempt in range(5):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                embeddings = [np.array(item.embedding) for item in response.data]
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise e

    return all_embeddings


def compute_anchor_embeddings(client) -> Dict[str, np.ndarray]:
    """Compute embeddings for fixed facet anchors."""
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    embeddings = embed_texts_batch(client, anchor_texts, batch_size=16)
    return {facet: emb for facet, emb in zip(FACETS, embeddings)}


def load_embeddings_cache() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load pre-computed tag and venue embeddings from Phase 2 cache."""
    embedding_model_slug = EMBEDDING_MODEL.replace("-", "_").replace(".", "_")

    # Tag embeddings
    tag_npz = PHASE2_CACHE / f"tag_embeddings_{embedding_model_slug}_normeval.npz"
    tag_map = PHASE2_CACHE / f"tag_embeddings_{embedding_model_slug}_normeval.map.json"

    if not tag_npz.exists():
        raise FileNotFoundError(f"Tag embeddings not found: {tag_npz}")

    with open(tag_map) as f:
        tag_idx_map = json.load(f)

    tag_data = np.load(tag_npz)
    tag_embeddings = {}
    for tag, idx in tag_idx_map.items():
        key = f"emb_{idx}"
        if key in tag_data:
            tag_embeddings[tag] = tag_data[key]

    # Venue embeddings
    venue_npz = PHASE2_CACHE / f"review_embeddings_{embedding_model_slug}.npz"
    venue_map = PHASE2_CACHE / f"review_embeddings_{embedding_model_slug}.map.json"

    venue_embeddings = {}
    if venue_npz.exists():
        with open(venue_map) as f:
            venue_idx_map = json.load(f)
        venue_data = np.load(venue_npz)
        for venue_id, indices in venue_idx_map.items():
            if isinstance(indices, list) and len(indices) > 0:
                review_embs = [venue_data[f"emb_{idx}"] for idx in indices if f"emb_{idx}" in venue_data]
                if review_embs:
                    venue_emb = np.mean(review_embs, axis=0)
                    norm = np.linalg.norm(venue_emb)
                    if norm > 0:
                        venue_emb = venue_emb / norm
                    venue_embeddings[venue_id] = venue_emb

    return tag_embeddings, venue_embeddings


# =============================================================================
# DATA LOADING
# =============================================================================

def load_phase1_data(run_id: str, results_dir: str = "results/phase1_downloaded") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1 extraction data."""
    results_path = Path(results_dir)

    # Find extraction and tag files
    extraction_files = list(results_path.glob(f"{run_id}_extractions_*.csv"))
    tag_files = list(results_path.glob(f"{run_id}_tags_*.csv"))

    if not extraction_files:
        raise FileNotFoundError(f"No extraction files found for {run_id} in {results_dir}")

    extractions = pd.concat([pd.read_csv(f) for f in extraction_files], ignore_index=True)
    tags = pd.concat([pd.read_csv(f) for f in tag_files], ignore_index=True)

    return extractions, tags


# =============================================================================
# CLASSICAL BASELINE EXTRACTION (utilities for Phase 3A comparison)
# These functions extract keywords using traditional methods.
# They are defined here as shared utilities but the actual comparison
# analysis is done in phase3a_baselines.py
# =============================================================================

def extract_rake_keywords(text: str, k: int = 25) -> List[str]:
    """Extract keywords using RAKE."""
    from rake_nltk import Rake

    if not text.strip():
        return []

    try:
        rake = Rake(min_length=1, max_length=4, include_repeated_phrases=False)
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()
        return [p for p in phrases if len(p.split()) <= 4][:k]
    except:
        return []


def extract_tfidf_keywords(text: str, k: int = 25) -> List[str]:
    """Extract keywords using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not text.strip():
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 4),
            stop_words='english',
            max_features=k * 5,
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        sorted_indices = np.argsort(scores)[::-1]
        keywords = []
        for idx in sorted_indices:
            if len(keywords) >= k:
                break
            phrase = feature_names[idx]
            if len(phrase.split()) <= 4:
                keywords.append(phrase)

        return keywords[:k]
    except:
        return []


def extract_yake_keywords(text: str, k: int = 25) -> List[str]:
    """Extract keywords using YAKE."""
    import yake

    if not text.strip():
        return []

    try:
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=4,
            dedupLim=0.7,
            top=k * 2,
            features=None
        )
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords if len(kw[0].split()) <= 4][:k]
    except:
        return []


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_state_localization_analysis(
    extractions_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compute STATE-GINI for gentags.

    This is the MAIN metric: Gini on facet counts per extraction.
    Classical baseline comparison is done separately in Phase 3A.
    """
    results = []

    print("   Computing State-Gini for each extraction...")

    for _, row in tqdm(extractions_df.iterrows(), total=len(extractions_df), desc="State-Gini"):
        exp_id = row['exp_id']
        venue_id = row['venue_id']
        model_key = row['model_key']
        prompt_type = row['prompt_type']
        run_number = row['run_number']

        # Get gentags for this extraction
        gentags = (
            tags_df.loc[tags_df['exp_id'] == exp_id, 'tag_norm_eval']
            .dropna().astype(str).map(str.strip).tolist()
        )
        gentags = [t for t in gentags if t]

        if not gentags:
            continue

        # Compute State-Gini for gentags
        gentag_state_gini, gentag_counts, gentag_other = compute_state_gini(
            gentags, tag_embeddings, anchor_embeddings
        )

        # Store result
        result = {
            'exp_id': exp_id,
            'venue_id': venue_id,
            'model_key': model_key,
            'prompt_type': prompt_type,
            'run_number': run_number,
            'n_gentags': len(gentags),
            'gentag_state_gini': gentag_state_gini,
            'gentag_other_count': gentag_other,
            'gentag_assigned_count': len(gentags) - gentag_other,
        }

        # Add per-facet counts for gentags
        for facet in FACETS:
            result[f'gentag_count_{facet}'] = gentag_counts.get(facet, 0)

        results.append(result)

    return pd.DataFrame(results)


def run_drift_localization_analysis(
    extractions_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Compute DRIFT-GINI for run pairs (secondary metric).

    This measures change localization, NOT state localization.
    """
    results = []

    # Group by venue/model/prompt to find run pairs
    groups = list(extractions_df.groupby(['venue_id', 'model_key', 'prompt_type']))

    print("   Computing Drift-Gini for run pairs...")

    for (venue_id, model_key, prompt_type), group in tqdm(groups, desc="Drift-Gini"):
        runs = group.sort_values('run_number')
        if len(runs) < 2:
            continue

        run1 = runs.iloc[0]
        run2 = runs.iloc[1]

        exp_id1 = run1['exp_id']
        exp_id2 = run2['exp_id']

        # Get tags for each run
        tags1 = (
            tags_df.loc[tags_df['exp_id'] == exp_id1, 'tag_norm_eval']
            .dropna().astype(str).map(str.strip).tolist()
        )
        tags2 = (
            tags_df.loc[tags_df['exp_id'] == exp_id2, 'tag_norm_eval']
            .dropna().astype(str).map(str.strip).tolist()
        )
        tags1 = [t for t in tags1 if t]
        tags2 = [t for t in tags2 if t]

        if not tags1 or not tags2:
            continue

        # Compute Drift-Gini
        drift_gini = compute_drift_gini(tags1, tags2, tag_embeddings, anchor_embeddings)

        results.append({
            'venue_id': venue_id,
            'model_key': model_key,
            'prompt_type': prompt_type,
            'exp_id_run1': exp_id1,
            'exp_id_run2': exp_id2,
            'n_tags_run1': len(tags1),
            'n_tags_run2': len(tags2),
            'drift_gini': drift_gini,
        })

    return pd.DataFrame(results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: State Localization Analysis (v2)")
    parser.add_argument("--run-id", type=str, default="week2_run_20251223_191104")
    parser.add_argument("--data", type=str, default="data/study1_venues_20250117.csv")
    parser.add_argument("--results-dir", type=str, default="results/phase1_downloaded")

    args = parser.parse_args()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3: STATE LOCALIZATION ANALYSIS (v2 - CORRECTED)")
    print("=" * 60)
    print("\nThis version computes STATE-GINI (Gini on counts)")
    print("NOT drift-Gini (Gini on change between runs)")
    print()

    # Load data
    print("1. Loading data...")
    extractions_df, tags_df = load_phase1_data(args.run_id, args.results_dir)
    print(f"   Loaded {len(extractions_df)} extractions, {len(tags_df)} tags")

    # Load venues
    venues_df = pd.read_csv(args.data)
    print(f"   Loaded {len(venues_df)} venues")

    # Filter to complete venues (all 4 models)
    all_models = {"claude", "gemini", "grok", "openai"}
    venue_model_counts = extractions_df.groupby("venue_id")["model_key"].apply(lambda x: set(x.unique()))
    complete_venues = set(venue_model_counts[venue_model_counts.apply(lambda x: x >= all_models)].index)

    extractions_df = extractions_df[extractions_df["venue_id"].isin(complete_venues)]
    valid_exp_ids = set(extractions_df["exp_id"].unique())
    tags_df = tags_df[tags_df["exp_id"].isin(valid_exp_ids)]

    print(f"   Filtered to {len(complete_venues)} complete venues")

    # Load embeddings
    print("\n2. Loading embeddings...")
    tag_embeddings, venue_embeddings = load_embeddings_cache()
    print(f"   Loaded {len(tag_embeddings)} tag embeddings")

    # Compute anchor embeddings
    print("\n3. Computing anchor embeddings...")
    client = get_embedding_client()
    anchor_embeddings = compute_anchor_embeddings(client)
    print(f"   Computed {len(anchor_embeddings)} facet anchors")

    # Run State-Gini analysis (MAIN METRIC)
    print("\n4. Computing STATE-GINI (main metric)...")
    state_df = run_state_localization_analysis(
        extractions_df, tags_df, tag_embeddings, anchor_embeddings
    )
    state_df.to_csv(TABLES_DIR / "state_localization.csv", index=False)
    print(f"   Saved {len(state_df)} rows to state_localization.csv")

    # Summary
    print("\n" + "=" * 60)
    print("STATE-GINI RESULTS (Main Metric)")
    print("=" * 60)
    print(f"\n   Gentag State-Gini:")
    print(f"   - Mean: {state_df['gentag_state_gini'].mean():.3f}")
    print(f"   - Std:  {state_df['gentag_state_gini'].std():.3f}")
    print(f"   - Median: {state_df['gentag_state_gini'].median():.3f}")

    # Run Drift-Gini analysis (SECONDARY METRIC)
    print("\n5. Computing DRIFT-GINI (secondary metric)...")
    drift_df = run_drift_localization_analysis(
        extractions_df, tags_df, tag_embeddings, anchor_embeddings
    )
    drift_df.to_csv(TABLES_DIR / "drift_localization.csv", index=False)
    print(f"   Saved {len(drift_df)} rows to drift_localization.csv")

    print("\n" + "=" * 60)
    print("DRIFT-GINI RESULTS (Secondary Metric)")
    print("=" * 60)
    print(f"\n   Gentag Drift-Gini:")
    print(f"   - Mean: {drift_df['drift_gini'].mean():.3f}")
    print(f"   - Std:  {drift_df['drift_gini'].std():.3f}")

    # Write manifest
    manifest = {
        "phase": "phase3_v2",
        "run_id": args.run_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "methodology": "STATE-GINI (Gini on counts, not drift)",
        "threshold": SEMANTIC_THRESHOLD,
        "counts": {
            "n_extractions": len(state_df),
            "n_venues": state_df['venue_id'].nunique(),
            "n_drift_pairs": len(drift_df),
        },
        "results": {
            "state_gini_mean": float(state_df['gentag_state_gini'].mean()),
            "state_gini_std": float(state_df['gentag_state_gini'].std()),
            "drift_gini_mean": float(drift_df['drift_gini'].mean()),
            "drift_gini_std": float(drift_df['drift_gini'].std()),
        },
        "facets": FACETS,
    }

    with open(OUTPUT_DIR / "phase3_v2_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 3 v2 Complete")
    print("=" * 60)

    return state_df, drift_df


if __name__ == "__main__":
    state_df, drift_df = main()
