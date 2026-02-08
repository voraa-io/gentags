#!/usr/bin/env python3
"""
Phase 3: Representation Comparison for Semantic State Observability

This script runs the complete Phase 3 analysis:
- Block G: Localization / Attribution (gentags vs embeddings)
- Block H: Cost Comparison
- Block I: Cold-Start Analysis

The model-in-the-loop baseline is run separately via phase3_model_in_loop.py.

Core Question:
> Which representation supports monitoring, attribution, and uncertainty-aware control?

Key Experiment (Block G):
Show that gentags enable LOCALIZED change attribution while embeddings produce DIFFUSE change.
- Gentags: High Gini coefficient (change concentrated in few facets)
- Embeddings: Low Gini coefficient (change spread across all facets)
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

from gentags import load_venue_data

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

# Keyword mapping for facet assignment
FACET_KEYWORDS = {
    "food_quality": ["food", "fresh", "tasty", "delicious", "bland", "meal", "breakfast", "lunch", "dinner", "dish", "cook", "chef", "menu", "eat"],
    "coffee_drinks": ["coffee", "espresso", "latte", "tea", "drink", "beverage", "cappuccino", "mocha", "brew", "roast"],
    "service": ["staff", "service", "friendly", "rude", "slow", "fast", "waiter", "barista", "server", "attentive", "helpful"],
    "ambiance": ["atmosphere", "vibe", "cozy", "noisy", "quiet", "decor", "music", "lighting", "ambiance", "aesthetic", "interior"],
    "price_value": ["price", "expensive", "cheap", "affordable", "value", "worth", "overpriced", "budget", "cost", "dollar"],
    "crowding": ["crowded", "busy", "wait", "line", "packed", "empty", "reservation", "queue"],
    "seating": ["seating", "outdoor", "patio", "table", "chair", "space", "indoor", "terrace", "booth"],
    "dietary": ["vegan", "vegetarian", "gluten", "allergy", "organic", "healthy", "dairy", "keto", "paleo"],
    "portions": ["portion", "size", "generous", "small", "large", "filling", "huge", "tiny"],
    "location": ["location", "parking", "accessible", "downtown", "corner", "find", "neighborhood", "walk", "drive"],
}

# =============================================================================
# FIXED FACET ANCHORS (Method-Neutral Evaluation)
# =============================================================================
# These are EXTERNAL descriptive phrases, NOT derived from tags.
# This ensures method-neutral evaluation and avoids circular reasoning.

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
# UTILITY FUNCTIONS
# =============================================================================

def assign_facet(tag: str) -> str:
    """Assign a tag to a facet based on keyword matching (LEXICAL - brittle)."""
    tag_lower = tag.lower()
    for facet, keywords in FACET_KEYWORDS.items():
        if any(kw in tag_lower for kw in keywords):
            return facet
    return "other"


def assign_facet_semantic(
    tag: str,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    """
    Assign a tag to a facet based on SEMANTIC similarity (robust).

    This is the correct approach for evaluating semantic representations:
    - Projects the tag embedding into the fixed anchor space
    - Assigns to the facet with highest cosine similarity

    Returns: (facet_name, similarity_score)
    """
    if tag not in tag_embeddings:
        return "other", 0.0

    tag_emb = tag_embeddings[tag]
    best_facet = "other"
    best_sim = -1.0

    for facet, anchor_emb in anchor_embeddings.items():
        sim = cosine_similarity(tag_emb, anchor_emb)
        if sim > best_sim:
            best_sim = sim
            best_facet = facet

    return best_facet, best_sim


def get_tag_facet_profile_semantic(
    tags: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute the facet profile for a tag set using semantic projection.

    Returns: array of mean similarity to each facet anchor
    """
    facet_sims = {facet: [] for facet in FACETS}

    for tag in tags:
        if tag in tag_embeddings:
            tag_emb = tag_embeddings[tag]
            for facet in FACETS:
                sim = cosine_similarity(tag_emb, anchor_embeddings[facet])
                facet_sims[facet].append(sim)

    # Mean similarity per facet (0 if no tags)
    profile = []
    for facet in FACETS:
        if facet_sims[facet]:
            profile.append(np.mean(facet_sims[facet]))
        else:
            profile.append(0.0)

    return np.array(profile)


# Similarity threshold for attributable state (τ = 0.35)
# Tags below this threshold are "semantically underdetermined"
SEMANTIC_THRESHOLD = 0.35


def get_tag_facet_profile_threshold(
    tags: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> Tuple[np.ndarray, int]:
    """
    Compute facet profile using hard semantic assignment WITH threshold.

    This is the GOLD STANDARD method:
    - Tags are assigned to their best-matching facet via cosine similarity
    - Tags below threshold are assigned to "other" (not counted in profile)
    - This balances semantic flexibility with state attribution

    Returns: (profile array, count of "other" tags)
    """
    counts = {facet: 0 for facet in FACETS}
    other_count = 0

    for tag in tags:
        if tag not in tag_embeddings:
            continue

        tag_emb = tag_embeddings[tag]
        best_facet = None
        best_sim = -1.0

        for facet in FACETS:
            sim = cosine_similarity(tag_emb, anchor_embeddings[facet])
            if sim > best_sim:
                best_sim = sim
                best_facet = facet

        if best_sim >= threshold:
            counts[best_facet] += 1
        else:
            other_count += 1

    total = sum(counts.values())
    profile = np.array([counts[f] / total if total > 0 else 0.0 for f in FACETS])

    return profile, other_count


def compute_gentag_facet_drift_threshold(
    tags1: List[str],
    tags2: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> np.ndarray:
    """
    Compute per-facet drift using THRESHOLD-BASED SEMANTIC PROJECTION.

    This is the GOLD STANDARD (τ=0.35):
    - Hard assignment based on cosine similarity to anchors
    - Tags below threshold go to "other" (excluded from drift)
    - Balances semantic flexibility with attributable state

    Returns: array of drift values per facet
    """
    profile1, _ = get_tag_facet_profile_threshold(tags1, tag_embeddings, anchor_embeddings, threshold)
    profile2, _ = get_tag_facet_profile_threshold(tags2, tag_embeddings, anchor_embeddings, threshold)

    drift = np.abs(profile1 - profile2)
    return drift


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of a distribution.

    - High Gini (→1): Change concentrated in few facets (LOCALIZED)
    - Low Gini (→0): Change spread evenly (DIFFUSE)
    """
    values = np.abs(values)
    if values.sum() == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)

    # Gini formula
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return max(0.0, gini)  # Ensure non-negative


def mean_pool(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean pooling of embeddings."""
    if not embeddings:
        return np.zeros(EMBEDDING_DIM)
    return np.mean(embeddings, axis=0)


# =============================================================================
# EMBEDDING FUNCTIONS (for Fixed Anchor Computation)
# =============================================================================

def get_embedding_client():
    """Initialize OpenAI embedding client."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv

    # Load .env
    for path in [Path(__file__).parent.parent / ".env", Path.cwd() / ".env"]:
        if path.exists():
            load_dotenv(path)
            break

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


def embed_texts_batch(client, texts: List[str], batch_size: int = 128) -> List[np.ndarray]:
    """Embed texts in batches using OpenAI API."""
    import time

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for attempt in range(5):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                embeddings = [np.array(e.embedding) for e in response.data]
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise e

    return all_embeddings


def compute_anchor_embeddings_fixed(client) -> Dict[str, np.ndarray]:
    """
    Compute anchor embeddings from FIXED descriptive phrases.

    This is the CORRECT approach for method-neutral evaluation:
    - Anchors are defined independently of any method's output
    - No circularity: we don't use tags to define facet anchors
    - No random fallback: every facet has a meaningful embedding

    All representations (Gentags, Embeddings, RAKE, TF-IDF) are projected
    into this same fixed semantic space.
    """
    print("   Embedding fixed facet anchor phrases...")
    anchor_embeddings = {}

    facet_texts = list(FACET_ANCHORS.values())
    facet_names = list(FACET_ANCHORS.keys())

    facet_embs = embed_texts_batch(client, facet_texts, batch_size=16)

    for name, emb in zip(facet_names, facet_embs):
        anchor_embeddings[name] = emb

    print(f"   Embedded {len(anchor_embeddings)} fixed facet anchors")
    return anchor_embeddings


# =============================================================================
# DATA LOADING
# =============================================================================

def load_phase1_data(run_id: str, results_dir: str = "results/phase1_downloaded") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1 extractions and tags."""
    results_path = Path(results_dir)

    # Load extractions
    extraction_files = list(results_path.glob(f"{run_id}_extractions_*.csv"))
    if not extraction_files:
        raise FileNotFoundError(f"No extraction files found for run_id: {run_id}")

    extractions_dfs = []
    for file in extraction_files:
        df = pd.read_csv(file)
        if "model_key" not in df.columns and "model" in df.columns:
            model_map = {
                "gpt-5-nano": "openai",
                "gemini-2.5-flash": "gemini",
                "claude-sonnet-4-5": "claude",
                "grok-4": "grok"
            }
            df["model_key"] = df["model"].map(model_map)
        extractions_dfs.append(df)
    extractions_df = pd.concat(extractions_dfs, ignore_index=True)

    # Filter to successful extractions
    if "status" in extractions_df.columns:
        extractions_df = extractions_df[extractions_df["status"] == "success"]

    # Load tags
    tag_files = list(results_path.glob(f"{run_id}_tags_*.csv"))
    if not tag_files:
        raise FileNotFoundError(f"No tag files found for run_id: {run_id}")

    tags_dfs = []
    for file in tag_files:
        df = pd.read_csv(file)
        if "model_key" not in df.columns and "model" in df.columns:
            model_map = {
                "gpt-5-nano": "openai",
                "gemini-2.5-flash": "gemini",
                "claude-sonnet-4-5": "claude",
                "grok-4": "grok"
            }
            df["model_key"] = df["model"].map(model_map)
        tags_dfs.append(df)
    tags_df = pd.concat(tags_dfs, ignore_index=True)

    # Filter to valid extractions
    valid_exp_ids = set(extractions_df["exp_id"].unique())
    tags_df = tags_df[tags_df["exp_id"].isin(valid_exp_ids)]

    return extractions_df, tags_df


def load_embeddings_cache() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load cached embeddings from Phase 2."""
    # Load tag embeddings
    tag_map_file = PHASE2_CACHE / f"tag_embeddings_text_embedding_3_large_normeval.map.json"
    tag_npz_file = PHASE2_CACHE / f"tag_embeddings_text_embedding_3_large_normeval.npz"

    if not tag_map_file.exists() or not tag_npz_file.exists():
        raise FileNotFoundError(
            f"Phase 2 cache not found. Run phase2_analysis.py first.\n"
            f"Expected: {tag_map_file}"
        )

    with open(tag_map_file, 'r') as f:
        tag_mapping = json.load(f)
    npz_data = np.load(tag_npz_file)
    tag_embeddings = {tag: npz_data[f"emb_{idx}"] for tag, idx in tag_mapping.items()}

    # Load review embeddings
    review_map_file = PHASE2_CACHE / f"review_embeddings_text_embedding_3_large.map.json"
    review_npz_file = PHASE2_CACHE / f"review_embeddings_text_embedding_3_large.npz"

    if review_map_file.exists() and review_npz_file.exists():
        with open(review_map_file, 'r') as f:
            review_mapping = json.load(f)
        npz_data = np.load(review_npz_file)
        venue_embeddings = {}
        for venue_id, indices in review_mapping.items():
            review_embs = [npz_data[f"emb_{idx}"] for idx in indices]
            venue_embeddings[venue_id] = mean_pool(review_embs)
    else:
        venue_embeddings = {}

    return tag_embeddings, venue_embeddings


# =============================================================================
# BLOCK G: LOCALIZATION / ATTRIBUTION
# =============================================================================

def compute_gentag_facet_drift(tags1: List[str], tags2: List[str]) -> np.ndarray:
    """
    Compute per-facet drift between two tag sets using KEYWORD MATCHING.
    This is the LEXICAL method - brittle but serves as lower bound.
    Returns: array of drift values per facet (0-1 scale)
    """
    drift = []

    for facet in FACETS:
        # Get tags belonging to this facet
        facet_tags1 = set(t for t in tags1 if assign_facet(t) == facet)
        facet_tags2 = set(t for t in tags2 if assign_facet(t) == facet)

        # Compute Jaccard distance (1 - similarity) for this facet
        if len(facet_tags1) == 0 and len(facet_tags2) == 0:
            facet_drift = 0.0  # No tags in this facet, no change
        else:
            intersection = len(facet_tags1 & facet_tags2)
            union = len(facet_tags1 | facet_tags2)
            jaccard = intersection / union if union > 0 else 0
            facet_drift = 1.0 - jaccard  # Convert to distance

        drift.append(facet_drift)

    return np.array(drift)


def compute_gentag_facet_drift_semantic(
    tags1: List[str],
    tags2: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute per-facet drift between two tag sets using SEMANTIC PROJECTION.
    This is the SEMANTIC method - robust and method-neutral.

    Projects each tag set into the fixed anchor space and measures
    the change in facet similarity profile.

    Returns: array of drift values per facet (absolute similarity change)
    """
    # Compute facet profiles for each tag set
    profile1 = get_tag_facet_profile_semantic(tags1, tag_embeddings, anchor_embeddings)
    profile2 = get_tag_facet_profile_semantic(tags2, tag_embeddings, anchor_embeddings)

    # Drift = absolute change in facet similarity
    drift = np.abs(profile1 - profile2)

    return drift


def compute_embedding_facet_drift(
    emb1: np.ndarray,
    emb2: np.ndarray,
    anchor_embeddings: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Compute per-facet drift between two embeddings using anchor similarity.
    """
    drift = []

    for facet in FACETS:
        anchor_emb = anchor_embeddings[facet]

        # Compute similarity to anchor for each embedding
        sim1 = cosine_similarity(emb1, anchor_emb)
        sim2 = cosine_similarity(emb2, anchor_emb)

        # Drift = absolute change in facet similarity
        facet_drift = abs(sim1 - sim2)
        drift.append(facet_drift)

    return np.array(drift)


# DEPRECATED: compute_anchor_embeddings (circular reasoning)
# This function was removed because it used tag embeddings to create anchors,
# then measured tag localization against those same anchors (self-fulfilling prophecy).
# It also used random vectors for facets with no coverage (noise injection).
#
# The correct approach is compute_anchor_embeddings_fixed() which uses
# fixed descriptive phrases embedded once, ensuring method-neutral evaluation.


def run_localization_analysis(
    extractions_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_embeddings: Dict[str, np.ndarray],
    embedding_client=None
) -> pd.DataFrame:
    """
    Block G: Compare localization between gentags and embeddings.

    For each run pair (run1 vs run2), compute:
    - Gentag Gini: How localized is the change in tag space?
    - Embedding Gini: How localized is the change in embedding space?

    Expected result:
    - Gentag Gini: HIGH (change concentrated in few facets)
    - Embedding Gini: LOW (change diffuse across all facets)

    NOTE: Anchor embeddings are computed from FIXED descriptive phrases,
    ensuring method-neutral evaluation (no circular reasoning).
    """
    # Initialize embedding client if not provided
    if embedding_client is None:
        embedding_client = get_embedding_client()

    # Use FIXED phrase anchors (method-neutral, no circular reasoning)
    anchor_embeddings = compute_anchor_embeddings_fixed(embedding_client)

    # Compute extraction-level pooled embeddings
    print("   Computing extraction embeddings...")
    extraction_embeddings = {}
    for exp_id, group in tags_df.groupby("exp_id"):
        unique_tags = (
            group["tag_norm_eval"]
            .dropna()
            .astype(str)
            .map(lambda s: s.strip())
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        if unique_tags:
            tag_embs = [tag_embeddings[tag] for tag in unique_tags if tag in tag_embeddings]
            if tag_embs:
                extraction_embeddings[exp_id] = mean_pool(tag_embs)

    results = []

    # Compare run1 vs run2 for same venue/model/prompt
    print("   Computing localization metrics...")
    groups = list(extractions_df.groupby(["venue_id", "model_key", "prompt_type"]))

    for (venue_id, model_key, prompt_type), group in tqdm(groups, desc="Localization", unit="combo"):
        runs = group.sort_values("run_number")
        if len(runs) < 2:
            continue

        run1 = runs.iloc[0]
        run2 = runs.iloc[1]

        exp_id1 = run1["exp_id"]
        exp_id2 = run2["exp_id"]

        # Get embeddings
        emb1 = extraction_embeddings.get(exp_id1)
        emb2 = extraction_embeddings.get(exp_id2)

        if emb1 is None or emb2 is None:
            continue

        # Get tags
        tags1 = (
            tags_df.loc[tags_df["exp_id"] == exp_id1, "tag_norm_eval"]
            .dropna().astype(str).map(str.strip).tolist()
        )
        tags2 = (
            tags_df.loc[tags_df["exp_id"] == exp_id2, "tag_norm_eval"]
            .dropna().astype(str).map(str.strip).tolist()
        )
        tags1 = [t for t in tags1 if t]
        tags2 = [t for t in tags2 if t]

        # Compute per-facet drift using THREE methods for sensitivity analysis
        # Method 1: KEYWORD-BASED (lexical, brittle - lower bound)
        gentag_drift_kw = compute_gentag_facet_drift(tags1, tags2)

        # Method 2: SEMANTIC MEAN (no threshold - diffuse, for comparison)
        gentag_drift_sem_mean = compute_gentag_facet_drift_semantic(
            tags1, tags2, tag_embeddings, anchor_embeddings
        )

        # Method 3: SEMANTIC THRESHOLD τ=0.35 (GOLD STANDARD)
        # Hard assignment with threshold - tags below τ go to "other"
        gentag_drift_sem_thr = compute_gentag_facet_drift_threshold(
            tags1, tags2, tag_embeddings, anchor_embeddings, threshold=SEMANTIC_THRESHOLD
        )

        # Embedding drift (already semantic)
        embedding_drift = compute_embedding_facet_drift(emb1, emb2, anchor_embeddings)

        # Compute Gini coefficients for ALL methods
        gentag_gini_kw = gini_coefficient(gentag_drift_kw)           # Keyword-based
        gentag_gini_sem_mean = gini_coefficient(gentag_drift_sem_mean)  # Semantic mean (diffuse)
        gentag_gini_sem_thr = gini_coefficient(gentag_drift_sem_thr)    # Semantic threshold (gold)
        embedding_gini = gini_coefficient(embedding_drift)

        # Store results with all three methods
        results.append({
            "venue_id": venue_id,
            "model_key": model_key,
            "prompt_type": prompt_type,
            # Keyword-based Gini (lexical, lower bound)
            "gentag_gini": gentag_gini_kw,
            "gentag_gini_kw": gentag_gini_kw,
            # Semantic mean Gini (no threshold, diffuse)
            "gentag_gini_sem": gentag_gini_sem_mean,
            "gentag_gini_sem_mean": gentag_gini_sem_mean,
            # Semantic threshold Gini (τ=0.35, GOLD STANDARD)
            "gentag_gini_sem_thr": gentag_gini_sem_thr,
            # Embedding Gini (baseline)
            "embedding_gini": embedding_gini,
            # Differences
            "gini_diff": gentag_gini_kw - embedding_gini,
            "gini_diff_kw": gentag_gini_kw - embedding_gini,
            "gini_diff_sem": gentag_gini_sem_mean - embedding_gini,
            "gini_diff_sem_thr": gentag_gini_sem_thr - embedding_gini,
            # Totals
            "gentag_total_drift_kw": gentag_drift_kw.sum(),
            "gentag_total_drift_sem": gentag_drift_sem_mean.sum(),
            "gentag_total_drift_sem_thr": gentag_drift_sem_thr.sum(),
            "embedding_total_drift": embedding_drift.sum(),
            # Per-facet drifts (keyword-based for backward compatibility)
            **{f"gentag_drift_{facet}": gentag_drift_kw[i] for i, facet in enumerate(FACETS)},
            # Per-facet drifts (semantic mean)
            **{f"gentag_drift_sem_{facet}": gentag_drift_sem_mean[i] for i, facet in enumerate(FACETS)},
            # Per-facet drifts (semantic threshold)
            **{f"gentag_drift_sem_thr_{facet}": gentag_drift_sem_thr[i] for i, facet in enumerate(FACETS)},
            **{f"embedding_drift_{facet}": embedding_drift[i] for i, facet in enumerate(FACETS)},
        })

    return pd.DataFrame(results)


def compute_facet_assignments(tags_df: pd.DataFrame) -> pd.DataFrame:
    """Assign each tag to a facet and return the mapping."""
    unique_tags = tags_df["tag_norm_eval"].dropna().unique()

    assignments = []
    for tag in unique_tags:
        tag_str = str(tag).strip()
        if tag_str:
            facet = assign_facet(tag_str)
            assignments.append({
                "tag": tag_str,
                "facet": facet,
            })

    return pd.DataFrame(assignments)


# =============================================================================
# BLOCK H: COST COMPARISON
# =============================================================================

def run_cost_comparison(
    extractions_df: pd.DataFrame,
    model_in_loop_cost: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Block H: Compare costs across representation methods.

    Compares:
    - Gentag extraction (one-time)
    - Model-in-the-loop (per-query)
    - Raw text storage
    - Embedding storage
    """
    results = []

    # Gentag extraction costs by model/prompt
    gentag_costs = extractions_df.groupby(["model_key", "prompt_type"]).agg(
        n_extractions=("exp_id", "count"),
        total_cost_usd=("cost_usd", "sum"),
        mean_cost_usd=("cost_usd", "mean"),
        total_tokens=("total_tokens", "sum"),
        mean_tokens=("total_tokens", "mean"),
    ).reset_index()

    for _, row in gentag_costs.iterrows():
        results.append({
            "method": "gentags",
            "model_key": row["model_key"],
            "prompt_type": row["prompt_type"],
            "n_extractions": row["n_extractions"],
            "total_cost_usd": row["total_cost_usd"],
            "mean_cost_per_venue_usd": row["mean_cost_usd"],
            "cost_type": "one_time",
            "notes": "Extract once, query unlimited times via tag lookup",
        })

    # Model-in-the-loop costs (if available)
    if model_in_loop_cost:
        results.append({
            "method": "model_in_loop",
            "model_key": "openai",
            "prompt_type": "facet_query",
            "n_extractions": model_in_loop_cost.get("n_queries", 0),
            "total_cost_usd": model_in_loop_cost.get("total_cost_usd", 0),
            "mean_cost_per_venue_usd": model_in_loop_cost.get("per_venue_cost_usd", 0),
            "cost_type": "per_query",
            "notes": "Cost scales linearly with number of queries",
        })

    # Embedding storage cost (estimated)
    n_venues = extractions_df["venue_id"].nunique()
    embedding_bytes_per_venue = EMBEDDING_DIM * 4  # float32
    total_embedding_mb = (n_venues * embedding_bytes_per_venue) / (1024 * 1024)

    results.append({
        "method": "embeddings",
        "model_key": "text-embedding-3-large",
        "prompt_type": "n/a",
        "n_extractions": n_venues,
        "total_cost_usd": 0.0,  # Storage is negligible
        "mean_cost_per_venue_usd": 0.0,
        "cost_type": "storage",
        "notes": f"Storage: {total_embedding_mb:.2f} MB for {n_venues} venues",
    })

    return pd.DataFrame(results)


# =============================================================================
# BLOCK I: COLD-START ANALYSIS
# =============================================================================

def run_cold_start_analysis(
    venues_csv: str,
    uncertainty_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Block I: Analyze behavior under sparse evidence (cold-start).

    Uses Phase 2 sparsity analysis results to show:
    - Variability increases with sparse data
    - This variability IS the uncertainty signal

    Control implication:
    - Low variance → high confidence → act
    - High variance → low confidence → seek more information
    """
    import ast

    venues_df = pd.read_csv(venues_csv)

    def count_tokens(reviews_str):
        try:
            reviews = ast.literal_eval(reviews_str)
            total_tokens = 0
            for review in reviews:
                if isinstance(review, dict) and 'text' in review:
                    total_tokens += len(review['text'].split())
            return total_tokens
        except:
            return 0

    def count_reviews(reviews_str):
        try:
            reviews = ast.literal_eval(reviews_str)
            return len(reviews)
        except:
            return 0

    venues_df['total_tokens'] = venues_df['google_reviews'].apply(count_tokens)
    venues_df['n_reviews'] = venues_df['google_reviews'].apply(count_reviews)

    # Merge with uncertainty data
    cold_start_df = uncertainty_df.merge(
        venues_df[['id', 'total_tokens', 'n_reviews']],
        left_on='venue_id',
        right_on='id',
        how='inner'
    )

    if 'id' in cold_start_df.columns:
        cold_start_df = cold_start_df.drop(columns=['id'])

    # Create sparsity buckets
    cold_start_df['evidence_level'] = pd.cut(
        cold_start_df['n_reviews'],
        bins=[0, 3, 5, 10, float('inf')],
        labels=['sparse (1-3)', 'low (4-5)', 'moderate (6-10)', 'rich (>10)']
    )

    cold_start_df['token_bucket'] = pd.cut(
        cold_start_df['total_tokens'],
        bins=[0, 200, 400, 600, float('inf')],
        labels=['<200', '200-400', '400-600', '>600']
    )

    return cold_start_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Representation Comparison Analysis")
    parser.add_argument(
        "--run-id",
        type=str,
        default="week2_run_20251223_191104",
        help="Run ID from Phase 1"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/study1_venues_20250117.csv",
        help="Path to venues CSV"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/phase1_downloaded",
        help="Directory containing Phase 1 results"
    )

    args = parser.parse_args()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3: REPRESENTATION COMPARISON ANALYSIS")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    extractions_df, tags_df = load_phase1_data(args.run_id, args.results_dir)
    print(f"   Loaded {len(extractions_df)} extractions, {len(tags_df)} tags")

    # Filter to complete venues (all 4 models)
    all_models = {"claude", "gemini", "grok", "openai"}
    venue_model_counts = extractions_df.groupby("venue_id")["model_key"].apply(lambda x: set(x.unique()))
    complete_venues = set(venue_model_counts[venue_model_counts.apply(lambda x: x >= all_models)].index)

    extractions_df = extractions_df[extractions_df["venue_id"].isin(complete_venues)]
    valid_exp_ids = set(extractions_df["exp_id"].unique())
    tags_df = tags_df[tags_df["exp_id"].isin(valid_exp_ids)]

    print(f"   Filtered to {len(complete_venues)} complete venues")

    # Clean tag_norm_eval
    tags_df["tag_norm_eval"] = tags_df["tag_norm_eval"].astype(str).map(str.strip)
    tags_df.loc[tags_df["tag_norm_eval"].isin(["", "nan", "None"]), "tag_norm_eval"] = np.nan

    # Load embeddings
    print("\n2. Loading embeddings cache...")
    tag_embeddings, venue_embeddings = load_embeddings_cache()
    print(f"   Loaded {len(tag_embeddings)} tag embeddings")

    # Block G: Facet assignments
    print("\n3. Block G: Computing facet assignments...")
    facet_assignments = compute_facet_assignments(tags_df)
    facet_assignments.to_csv(TABLES_DIR / "facet_assignments.csv", index=False)

    facet_counts = facet_assignments["facet"].value_counts()
    print(f"   Assigned {len(facet_assignments)} tags to facets:")
    for facet, count in facet_counts.head(5).items():
        print(f"      - {facet}: {count}")
    print(f"      - other: {facet_counts.get('other', 0)}")

    # Block G: Localization analysis
    print("\n4. Block G: Running localization analysis...")
    localization_df = run_localization_analysis(extractions_df, tags_df, tag_embeddings)
    localization_df.to_csv(TABLES_DIR / "localization.csv", index=False)

    # Localization summary - SENSITIVITY ANALYSIS (three methods)
    print(f"\n   Localization Results (n={len(localization_df)} pairs):")

    # Method 1: Keyword-based (lexical, lower bound)
    print(f"\n   === METHOD 1: KEYWORD-BASED (Lexical, ~50% filtered) ===")
    print(f"   - Gentag Gini (KW):      {localization_df['gentag_gini_kw'].mean():.3f}")
    print(f"   - Embedding Gini:        {localization_df['embedding_gini'].mean():.3f}")
    print(f"   - Advantage:             {localization_df['gentag_gini_kw'].mean() / localization_df['embedding_gini'].mean():.2f}x")
    print(f"   - % gentag > embedding:  {(localization_df['gini_diff_kw'] > 0).mean():.1%}")

    # Method 2: Semantic mean (no threshold, diffuse)
    print(f"\n   === METHOD 2: SEMANTIC MEAN (No threshold, diffuse) ===")
    print(f"   - Gentag Gini (SEM):     {localization_df['gentag_gini_sem_mean'].mean():.3f}")
    print(f"   - Embedding Gini:        {localization_df['embedding_gini'].mean():.3f}")
    print(f"   - Advantage:             {localization_df['gentag_gini_sem_mean'].mean() / localization_df['embedding_gini'].mean():.2f}x")
    print(f"   - % gentag > embedding:  {(localization_df['gini_diff_sem'] > 0).mean():.1%}")

    # Method 3: Semantic threshold (τ=0.35, GOLD STANDARD)
    print(f"\n   === METHOD 3: SEMANTIC THRESHOLD τ={SEMANTIC_THRESHOLD} (GOLD STANDARD) ===")
    print(f"   - Gentag Gini (THR):     {localization_df['gentag_gini_sem_thr'].mean():.3f}")
    print(f"   - Embedding Gini:        {localization_df['embedding_gini'].mean():.3f}")
    print(f"   - Advantage:             {localization_df['gentag_gini_sem_thr'].mean() / localization_df['embedding_gini'].mean():.2f}x")
    print(f"   - % gentag > embedding:  {(localization_df['gini_diff_sem_thr'] > 0).mean():.1%}")

    # Statistical tests for all methods
    from scipy.stats import mannwhitneyu, wilcoxon
    stat_kw, pvalue_kw = wilcoxon(localization_df['gentag_gini_kw'], localization_df['embedding_gini'])
    stat_sem, pvalue_sem = wilcoxon(localization_df['gentag_gini_sem_mean'], localization_df['embedding_gini'])
    stat_thr, pvalue_thr = wilcoxon(localization_df['gentag_gini_sem_thr'], localization_df['embedding_gini'])
    print(f"\n   === STATISTICAL SIGNIFICANCE ===")
    print(f"   - Wilcoxon p-value (KW):   {pvalue_kw:.2e}")
    print(f"   - Wilcoxon p-value (SEM):  {pvalue_sem:.2e}")
    print(f"   - Wilcoxon p-value (THR):  {pvalue_thr:.2e}")

    # Block H: Cost comparison
    print("\n5. Block H: Computing cost comparison...")

    # Try to load model-in-loop costs if available
    model_in_loop_cost = None
    mil_cost_file = OUTPUT_DIR / "model_in_loop_cost.json"
    if mil_cost_file.exists():
        with open(mil_cost_file, 'r') as f:
            model_in_loop_cost = json.load(f)
        print(f"   Loaded model-in-loop costs from {mil_cost_file}")

    cost_df = run_cost_comparison(extractions_df, model_in_loop_cost)
    cost_df.to_csv(TABLES_DIR / "cost_comparison.csv", index=False)
    print(f"   Saved cost comparison to {TABLES_DIR / 'cost_comparison.csv'}")

    # Block I: Cold-start analysis
    print("\n6. Block I: Running cold-start analysis...")

    # Load Phase 2 uncertainty data
    uncertainty_file = PHASE2_TABLES / "uncertainty_dispersion.csv"
    if uncertainty_file.exists():
        uncertainty_df = pd.read_csv(uncertainty_file)
        cold_start_df = run_cold_start_analysis(args.data, uncertainty_df)
        cold_start_df.to_csv(TABLES_DIR / "cold_start.csv", index=False)

        # Summary by evidence level
        print(f"\n   Cold-Start Results (variability by evidence level):")
        summary = cold_start_df.groupby('evidence_level', observed=True).agg({
            'mean_pairwise_distance': ['mean', 'std', 'count']
        }).round(4)
        print(summary)

        # Correlation
        corr = cold_start_df['n_reviews'].corr(cold_start_df['mean_pairwise_distance'])
        print(f"\n   Correlation (n_reviews vs variability): {corr:.3f}")
        print(f"   Expected: NEGATIVE (more evidence → less variability)")
    else:
        print(f"   ⚠ Phase 2 uncertainty data not found: {uncertainty_file}")
        print(f"   Run phase2_analysis.py first")

    # Write manifest
    print("\n7. Writing manifest...")
    manifest = {
        "phase": "phase3",
        "run_id": args.run_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "counts": {
            "n_extractions": int(len(extractions_df)),
            "n_tags": int(len(tags_df)),
            "n_venues": int(extractions_df["venue_id"].nunique()),
            "n_localization_pairs": int(len(localization_df)),
        },
        "localization_results": {
            # Method 1: Keyword-based (lexical, lower bound, ~50% filtered)
            "gentag_gini_kw_mean": float(localization_df['gentag_gini_kw'].mean()),
            "gini_diff_kw_mean": float(localization_df['gini_diff_kw'].mean()),
            "pct_gentag_more_localized_kw": float((localization_df['gini_diff_kw'] > 0).mean()),
            "wilcoxon_pvalue_kw": float(pvalue_kw),
            "advantage_kw": float(localization_df['gentag_gini_kw'].mean() / localization_df['embedding_gini'].mean()),
            # Method 2: Semantic mean (no threshold, diffuse)
            "gentag_gini_sem_mean_mean": float(localization_df['gentag_gini_sem_mean'].mean()),
            "gini_diff_sem_mean": float(localization_df['gini_diff_sem'].mean()),
            "pct_gentag_more_localized_sem": float((localization_df['gini_diff_sem'] > 0).mean()),
            "wilcoxon_pvalue_sem": float(pvalue_sem),
            "advantage_sem_mean": float(localization_df['gentag_gini_sem_mean'].mean() / localization_df['embedding_gini'].mean()),
            # Method 3: Semantic threshold τ=0.35 (GOLD STANDARD)
            "gentag_gini_sem_thr_mean": float(localization_df['gentag_gini_sem_thr'].mean()),
            "gini_diff_sem_thr_mean": float(localization_df['gini_diff_sem_thr'].mean()),
            "pct_gentag_more_localized_thr": float((localization_df['gini_diff_sem_thr'] > 0).mean()),
            "wilcoxon_pvalue_thr": float(pvalue_thr),
            "advantage_sem_thr": float(localization_df['gentag_gini_sem_thr'].mean() / localization_df['embedding_gini'].mean()),
            "semantic_threshold": SEMANTIC_THRESHOLD,
            # Common baseline
            "embedding_gini_mean": float(localization_df['embedding_gini'].mean()),
        },
        "facets": FACETS,
    }

    with open(OUTPUT_DIR / "phase3_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Phase 3 analysis complete!")
    print("=" * 60)
    print(f"\nKey Finding (Localization) - SENSITIVITY ANALYSIS:")
    print(f"\n  Method 1: KEYWORD-BASED (Lexical, ~50% filtered):")
    print(f"    Gentag Gini:    {localization_df['gentag_gini_kw'].mean():.3f}")
    print(f"    Embedding Gini: {localization_df['embedding_gini'].mean():.3f}")
    print(f"    Advantage:      {localization_df['gentag_gini_kw'].mean() / localization_df['embedding_gini'].mean():.2f}x")
    print(f"\n  Method 2: SEMANTIC MEAN (No threshold, diffuse):")
    print(f"    Gentag Gini:    {localization_df['gentag_gini_sem_mean'].mean():.3f}")
    print(f"    Embedding Gini: {localization_df['embedding_gini'].mean():.3f}")
    print(f"    Advantage:      {localization_df['gentag_gini_sem_mean'].mean() / localization_df['embedding_gini'].mean():.2f}x")
    print(f"\n  Method 3: SEMANTIC THRESHOLD τ={SEMANTIC_THRESHOLD} (GOLD STANDARD):")
    print(f"    Gentag Gini:    {localization_df['gentag_gini_sem_thr'].mean():.3f}")
    print(f"    Embedding Gini: {localization_df['embedding_gini'].mean():.3f}")
    print(f"    Advantage:      {localization_df['gentag_gini_sem_thr'].mean() / localization_df['embedding_gini'].mean():.2f}x")
    print(f"\n  INTERPRETATION:")
    print(f"    - Keyword-based shows highest Gini due to 'other' bucket filtering")
    print(f"    - Semantic threshold (τ=0.35) is the GOLD STANDARD")
    print(f"    - It balances semantic flexibility with attributable state")
    print(f"\nOutput files:")
    print(f"  - {TABLES_DIR / 'localization.csv'}")
    print(f"  - {TABLES_DIR / 'facet_assignments.csv'}")
    print(f"  - {TABLES_DIR / 'cost_comparison.csv'}")
    print(f"  - {TABLES_DIR / 'cold_start.csv'}")
    print(f"\nNext: Run phase3_plots.py to generate visualizations")


if __name__ == "__main__":
    main()
