#!/usr/bin/env python3
"""
Phase 2: Semantic Stability & Compression Analysis

Computes:
- Semantic similarity (cosine) between gentag sets
- Embedding drift analysis (runs, prompts, models)
- Compression/retention analysis (tokens → meaning, cost → semantic retention)
- Jaccard as surface diagnostic (not primary metric)

PRIMARY METRICS:
- Run stability: cosine(T_mean_unique(run1), T_mean_unique(run2))
- Prompt/model sensitivity: cosine between pooled representations
- Retention: cosine(R_mean(venue), T_mean_unique(extraction))
- Compression: delta_retention_per_dollar, Pareto front

ROBUSTNESS METRICS:
- MMC (Mean Max Cosine) for set-based semantic similarity
- Null baselines (random tags, shuffled venues, top-K tags)

Deliverables:
1. Run stability distribution (violin/ECDF per model)
2. Prompt sensitivity heatmap (3×3 per model)
3. Model sensitivity heatmap (4×4 per prompt)
4. Retention plot: cosine(review, gentags) by model/prompt
5. Cost-effectiveness plot: Pareto front (retention vs cost)
6. Surface vs semantic scatter: Jaccard vs cosine
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import load_venue_data


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI embedding model
EMBEDDING_DIM = 3072  # text-embedding-3-large dimension

# Canonical representations (LOCKED)
# - Review: R_mean = mean pooling of per-review embeddings
# - Tags: T_mean_unique = dedupe by tag_norm_eval, then mean pool

# Cache directories (using NPZ for efficiency)
CACHE_DIR = Path("results/phase2_cache")
REVIEW_EMBEDDINGS_NPZ = CACHE_DIR / "review_embeddings.npz"
REVIEW_EMBEDDINGS_MAP = CACHE_DIR / "review_embeddings_map.json"
TAG_EMBEDDINGS_NPZ = CACHE_DIR / "tag_embeddings.npz"
TAG_EMBEDDINGS_MAP = CACHE_DIR / "tag_embeddings_map.json"

# Output directories
OUTPUT_DIR = Path("results/phase2")
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_embedding_client():
    """Initialize OpenAI embedding client."""
    try:
        from openai import OpenAI
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package required for embeddings. Install with: poetry add openai")


def embed_texts_batch(client, texts: List[str], batch_size: int = 128) -> List[np.ndarray]:
    """
    Embed multiple texts in batches (efficient, avoids rate limits).
    
    Args:
        client: OpenAI client
        texts: List of texts to embed
        batch_size: Number of texts per batch
    
    Returns:
        List of embedding vectors (preserves order)
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        # Preserve order (response.data is sorted by index)
        batch_embeddings = sorted(response.data, key=lambda d: d.index)
        all_embeddings.extend([np.array(d.embedding) for d in batch_embeddings])
    
    return all_embeddings


def embed_text(client, text: str) -> np.ndarray:
    """Embed a single text string (wrapper for batch function)."""
    return embed_texts_batch(client, [text])[0]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def mean_pool(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean pooling of embeddings."""
    if not embeddings:
        return np.zeros(EMBEDDING_DIM)
    return np.mean(embeddings, axis=0)


def safe_load_tags_json(x) -> List[str]:
    """
    Safely parse raw_tags_json field (handles NaN, None, empty strings).
    
    Args:
        x: Value from DataFrame (can be str, float, None, NaN)
    
    Returns:
        List of tag strings (empty list if invalid)
    """
    if x is None:
        return []
    if isinstance(x, float) and np.isnan(x):
        return []
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() == "nan":
            return []
        try:
            out = json.loads(s)
            if isinstance(out, list):
                return [t for t in out if isinstance(t, str) and t.strip()]
            return []
        except Exception:
            return []
    return []


def jaccard_from_norm_eval(tags_df: pd.DataFrame, exp_id1: str, exp_id2: str) -> float:
    """
    Compute Jaccard similarity using tag_norm_eval (canonical for Phase 2).
    
    Args:
        tags_df: DataFrame with exp_id and tag_norm_eval columns
        exp_id1, exp_id2: Experiment IDs to compare
    
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    set1 = set(tags_df.loc[tags_df["exp_id"] == exp_id1, "tag_norm_eval"].dropna().astype(str))
    set2 = set(tags_df.loc[tags_df["exp_id"] == exp_id2, "tag_norm_eval"].dropna().astype(str))
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def make_derangement(ids: List[str], seed: int = 42, max_tries: int = 50) -> Dict[str, str]:
    """
    Create a derangement (permutation with no fixed points) of venue IDs.
    
    Args:
        ids: List of venue IDs
        seed: Random seed
        max_tries: Maximum attempts before allowing some fixed points
    
    Returns:
        Dict mapping venue_id -> shuffled_venue_id (no self-matches if possible)
    """
    rng = np.random.default_rng(seed)
    ids_array = np.array(ids)
    
    for _ in range(max_tries):
        perm = rng.permutation(ids_array)
        if np.all(perm != ids_array):
            return dict(zip(ids_array.tolist(), perm.tolist()))
    
    # Fallback: allow a few fixed points rather than infinite loop
    perm = rng.permutation(ids_array)
    return dict(zip(ids_array.tolist(), perm.tolist()))


def mean_max_cosine(set1_embs: List[np.ndarray], set2_embs: List[np.ndarray]) -> float:
    """
    Mean Max Cosine (MMC) between two sets of embeddings.
    
    For each embedding in set1, find max cosine to any embedding in set2.
    Average these max values, then symmetrize.
    
    This directly answers "same ideas but paraphrased."
    """
    if not set1_embs or not set2_embs:
        return 0.0
    
    # Forward: for each tag in set1, max cosine to set2
    forward_maxes = []
    for emb1 in set1_embs:
        max_cos = max(cosine_similarity(emb1, emb2) for emb2 in set2_embs)
        forward_maxes.append(max_cos)
    
    # Backward: for each tag in set2, max cosine to set1
    backward_maxes = []
    for emb2 in set2_embs:
        max_cos = max(cosine_similarity(emb2, emb1) for emb1 in set1_embs)
        backward_maxes.append(max_cos)
    
    # Symmetrize: average of forward and backward
    forward_mean = np.mean(forward_maxes)
    backward_mean = np.mean(backward_maxes)
    return (forward_mean + backward_mean) / 2.0


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_all_extractions(run_id: str, results_dir: str = "results") -> pd.DataFrame:
    """
    Load and merge all extraction CSVs from Phase 1.
    
    Returns DataFrame with one row per extraction.
    """
    results_path = Path(results_dir)
    extraction_files = list(results_path.glob(f"{run_id}_extractions_*.csv"))
    
    if not extraction_files:
        raise FileNotFoundError(f"No extraction files found for run_id: {run_id}")
    
    dfs = []
    for file in extraction_files:
        df = pd.read_csv(file)
        # Extract model_key from filename or model column
        if "model_key" not in df.columns:
            if "model" in df.columns:
                model_map = {
                    "gpt-5-nano": "openai",
                    "gemini-2.5-flash": "gemini",
                    "claude-sonnet-4-5": "claude",
                    "grok-4": "grok"
                }
                df["model_key"] = df["model"].map(model_map)
            else:
                # Try to extract from filename
                if "openai" in file.name:
                    df["model_key"] = "openai"
                elif "gemini" in file.name:
                    df["model_key"] = "gemini"
                elif "claude" in file.name:
                    df["model_key"] = "claude"
                elif "grok" in file.name:
                    df["model_key"] = "grok"
        dfs.append(df)
    
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def load_all_tags(run_id: str, results_dir: str = "results") -> pd.DataFrame:
    """Load and merge all tag CSVs from Phase 1."""
    results_path = Path(results_dir)
    tag_files = list(results_path.glob(f"{run_id}_tags_*.csv"))
    
    if not tag_files:
        raise FileNotFoundError(f"No tag files found for run_id: {run_id}")
    
    dfs = []
    for file in tag_files:
        df = pd.read_csv(file)
        # Extract model_key
        if "model_key" not in df.columns:
            if "model" in df.columns:
                model_map = {
                    "gpt-5-nano": "openai",
                    "gemini-2.5-flash": "gemini",
                    "claude-sonnet-4-5": "claude",
                    "grok-4": "grok"
                }
                df["model_key"] = df["model"].map(model_map)
            else:
                if "openai" in file.name:
                    df["model_key"] = "openai"
                elif "gemini" in file.name:
                    df["model_key"] = "gemini"
                elif "claude" in file.name:
                    df["model_key"] = "claude"
                elif "grok" in file.name:
                    df["model_key"] = "grok"
        dfs.append(df)
    
    merged = pd.concat(dfs, ignore_index=True)
    return merged


def load_venue_reviews(venues_csv: str) -> Dict[str, List[str]]:
    """Load venue reviews for embedding."""
    venues_df = load_venue_data(venues_csv)
    reviews_dict = {}
    for _, row in venues_df.iterrows():
        venue_id = row["id"]
        reviews = row["google_reviews"]
        reviews_dict[venue_id] = reviews
    return reviews_dict


# =============================================================================
# EMBEDDING COMPUTATION (with NPZ caching)
# =============================================================================

def compute_review_embeddings(
    client,
    reviews_dict: Dict[str, List[str]],
    cache_npz: Optional[Path] = None,
    cache_map: Optional[Path] = None
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, np.ndarray]]:
    """
    Compute embeddings for all reviews.
    
    Canonical representation: R_mean = mean pooling of per-review embeddings.
    
    Returns:
        (review_level_embeddings, venue_pooled_embeddings)
    """
    # Load cache if exists
    review_level = {}
    venue_pooled = {}
    
    if cache_npz and cache_npz.exists() and cache_map and cache_map.exists():
        # Load from NPZ cache
        with open(cache_map, 'r') as f:
            mapping = json.load(f)
        npz_data = np.load(cache_npz)
        
        for venue_id, review_indices in mapping.items():
            review_embs = [npz_data[f"emb_{idx}"] for idx in review_indices]
            review_level[venue_id] = review_embs
            venue_pooled[venue_id] = mean_pool(review_embs)
        
        return review_level, venue_pooled
    
    # Collect all reviews for batch embedding
    mapping = {}  # FIX: define mapping dict
    all_reviews = []
    venue_review_ranges = {}
    
    for venue_id, reviews in reviews_dict.items():
        # Guard against junk (ensure list of strings)
        if not isinstance(reviews, list):
            reviews = []
        reviews = [r for r in reviews if isinstance(r, str) and r.strip()]
        
        start_idx = len(all_reviews)
        all_reviews.extend(reviews)
        end_idx = len(all_reviews)
        venue_review_ranges[venue_id] = (start_idx, end_idx)
    
    # Embed all reviews in batches
    print(f"   Embedding {len(all_reviews)} reviews in batches...")
    all_review_embs = embed_texts_batch(client, all_reviews, batch_size=128)
    
    # Group embeddings by venue
    for venue_id, (start_idx, end_idx) in venue_review_ranges.items():
        review_embs = all_review_embs[start_idx:end_idx]
        review_level[venue_id] = review_embs
        venue_pooled[venue_id] = mean_pool(review_embs)
        mapping[venue_id] = list(range(start_idx, end_idx))
    
    # Save cache
    if cache_npz and cache_map:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_npz, **{f"emb_{i}": emb for i, emb in enumerate(all_review_embs)})
        with open(cache_map, 'w') as f:
            json.dump(mapping, f)
    
    return review_level, venue_pooled


def compute_tag_embeddings(
    client,
    tags_df: pd.DataFrame,
    cache_npz: Optional[Path] = None,
    cache_map: Optional[Path] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute embeddings for all tags.
    
    Canonical representation: T_mean_unique = dedupe by tag_norm_eval, then mean pool.
    
    Returns:
        (tag_level_embeddings, extraction_pooled_embeddings)
    """
    # Load cache if exists
    tag_embeddings = {}
    extraction_pooled = {}
    
    if cache_npz and cache_npz.exists() and cache_map and cache_map.exists():
        with open(cache_map, 'r') as f:
            tag_to_idx = json.load(f)
        npz_data = np.load(cache_npz)
        
        # Reconstruct tag embeddings dict
        for tag, idx in tag_to_idx.items():
            tag_embeddings[tag] = npz_data[f"emb_{idx}"]
        
        # Recompute extraction pooled embeddings
        for exp_id, group in tags_df.groupby("exp_id"):
            # Dedupe by tag_norm_eval and use canonical strings
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
                    extraction_pooled[exp_id] = mean_pool(tag_embs)
                else:
                    extraction_pooled[exp_id] = np.zeros(EMBEDDING_DIM)
            else:
                extraction_pooled[exp_id] = np.zeros(EMBEDDING_DIM)
        
        return tag_embeddings, extraction_pooled
    
    # Compute embeddings for unique tags (batch embedding)
    # Canonical: embed tag_norm_eval (not tag_raw)
    unique_tags = (
        tags_df["tag_norm_eval"]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    
    print(f"   Embedding {len(unique_tags)} unique tags (tag_norm_eval) in batches...")
    all_tag_embs = embed_texts_batch(client, unique_tags, batch_size=128)
    
    tag_to_idx = {}
    for idx, tag in enumerate(unique_tags):
        tag_embeddings[tag] = all_tag_embs[idx]
        tag_to_idx[tag] = idx
    
    # Save cache before computing extraction embeddings
    if cache_npz and cache_map:
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_npz, **{f"emb_{i}": emb for i, emb in enumerate(all_tag_embs)})
        with open(cache_map, 'w') as f:
            json.dump(tag_to_idx, f)
    
    # Compute extraction-level pooled embeddings (T_mean_unique)
    for exp_id, group in tags_df.groupby("exp_id"):
        # Dedupe by tag_norm_eval and embed those canonical strings
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
                extraction_pooled[exp_id] = mean_pool(tag_embs)
            else:
                extraction_pooled[exp_id] = np.zeros(EMBEDDING_DIM)
        else:
            extraction_pooled[exp_id] = np.zeros(EMBEDDING_DIM)
    
    return tag_embeddings, extraction_pooled


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_run_stability(
    extractions_df: pd.DataFrame,
    extraction_embeddings: Dict[str, np.ndarray],
    tags_df: pd.DataFrame,
    tag_embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Block A: Run semantic stability (RQ1-S)
    
    PRIMARY: cosine(T_mean_unique(run1), T_mean_unique(run2))
    ROBUSTNESS: MMC between tag sets
    """
    results = []
    
    for (venue_id, model_key, prompt_type), group in extractions_df.groupby(
        ["venue_id", "model_key", "prompt_type"]
    ):
        runs = group.sort_values("run_number")
        if len(runs) < 2:
            continue
        
        run1 = runs.iloc[0]
        run2 = runs.iloc[1]
        
        exp_id1 = run1["exp_id"]
        exp_id2 = run2["exp_id"]
        
        # Primary: pooled cosine similarity
        emb1 = extraction_embeddings.get(exp_id1)
        emb2 = extraction_embeddings.get(exp_id2)
        
        if emb1 is None or emb2 is None:
            continue
        
        cosine_sim = cosine_similarity(emb1, emb2)
        
        # Robustness: MMC computed on tag_norm_eval (canonical)
        set1 = tags_df.loc[tags_df["exp_id"] == exp_id1, "tag_norm_eval"].dropna().astype(str).tolist()
        set2 = tags_df.loc[tags_df["exp_id"] == exp_id2, "tag_norm_eval"].dropna().astype(str).tolist()
        # Dedupe
        set1 = sorted(set([s.strip() for s in set1 if s.strip()]))
        set2 = sorted(set([s.strip() for s in set2 if s.strip()]))
        
        # Embed canonical strings via tag_embeddings
        tags1_embs = [tag_embeddings[s] for s in set1 if s in tag_embeddings]
        tags2_embs = [tag_embeddings[s] for s in set2 if s in tag_embeddings]
        mmc = mean_max_cosine(tags1_embs, tags2_embs) if tags1_embs and tags2_embs else 0.0
        
        # Jaccard on tag_norm_eval (surface diagnostic)
        jaccard = jaccard_from_norm_eval(tags_df, exp_id1, exp_id2)
        
        results.append({
            "venue_id": venue_id,
            "model_key": model_key,
            "prompt_type": prompt_type,
            "cosine_similarity": cosine_sim,  # PRIMARY
            "mmc": mmc,  # ROBUSTNESS
            "jaccard_norm_eval": jaccard,  # SURFACE DIAGNOSTIC
            "n_unique_norm_eval_run1": len(set1),
            "n_unique_norm_eval_run2": len(set2),
            "n_tags_run1": run1.get("n_tags", 0),
            "n_tags_run2": run2.get("n_tags", 0),
        })
    
    return pd.DataFrame(results)


def compute_prompt_sensitivity(
    extractions_df: pd.DataFrame,
    extraction_embeddings: Dict[str, np.ndarray],
    tags_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Block B: Prompt semantic sensitivity (RQ2-S)
    
    PRIMARY: cosine between pooled representations
    """
    results = []
    
    for (venue_id, model_key, run_number), group in extractions_df.groupby(
        ["venue_id", "model_key", "run_number"]
    ):
        prompts = group.sort_values("prompt_type").to_dict("records")
        if len(prompts) < 2:
            continue
        
        # Compare all pairs (positional iteration to avoid index bugs)
        for a in range(len(prompts)):
            for b in range(a + 1, len(prompts)):
                row1 = prompts[a]
                row2 = prompts[b]
                
                exp_id1 = row1["exp_id"]
                exp_id2 = row2["exp_id"]
                
                emb1 = extraction_embeddings.get(exp_id1)
                emb2 = extraction_embeddings.get(exp_id2)
                
                if emb1 is None or emb2 is None:
                    continue
                
                cosine_sim = cosine_similarity(emb1, emb2)
                
                # Jaccard on tag_norm_eval
                jaccard = jaccard_from_norm_eval(tags_df, exp_id1, exp_id2)
                
                results.append({
                    "venue_id": venue_id,
                    "model_key": model_key,
                    "run_number": run_number,
                    "prompt1": row1["prompt_type"],
                    "prompt2": row2["prompt_type"],
                    "cosine_similarity": cosine_sim,
                    "jaccard_norm_eval": jaccard,
                })
    
    return pd.DataFrame(results)


def compute_model_sensitivity(
    extractions_df: pd.DataFrame,
    extraction_embeddings: Dict[str, np.ndarray],
    tags_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Block C: Model semantic sensitivity (RQ3-S)
    
    PRIMARY: cosine between pooled representations
    """
    results = []
    
    for (venue_id, prompt_type, run_number), group in extractions_df.groupby(
        ["venue_id", "prompt_type", "run_number"]
    ):
        models = group.sort_values("model_key").to_dict("records")
        if len(models) < 2:
            continue
        
        # Compare all pairs (positional iteration to avoid index bugs)
        for a in range(len(models)):
            for b in range(a + 1, len(models)):
                row1 = models[a]
                row2 = models[b]
                
                exp_id1 = row1["exp_id"]
                exp_id2 = row2["exp_id"]
                
                emb1 = extraction_embeddings.get(exp_id1)
                emb2 = extraction_embeddings.get(exp_id2)
                
                if emb1 is None or emb2 is None:
                    continue
                
                cosine_sim = cosine_similarity(emb1, emb2)
                
                # Jaccard on tag_norm_eval
                jaccard = jaccard_from_norm_eval(tags_df, exp_id1, exp_id2)
                
                results.append({
                    "venue_id": venue_id,
                    "prompt_type": prompt_type,
                    "run_number": run_number,
                    "model1": row1["model_key"],
                    "model2": row2["model_key"],
                    "cosine_similarity": cosine_sim,
                    "jaccard_norm_eval": jaccard,
                })
    
    return pd.DataFrame(results)


def compute_retention_with_baselines(
    extractions_df: pd.DataFrame,
    venue_embeddings: Dict[str, np.ndarray],
    extraction_embeddings: Dict[str, np.ndarray],
    tags_df: pd.DataFrame,
    tag_embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Block D: Semantic retention with null baselines
    
    PRIMARY: cosine(R_mean(venue), T_mean_unique(extraction))
    BASELINES:
    - Random tags (within-dataset)
    - Shuffled venue mapping
    - Top-K frequent tags (optional)
    """
    results = []
    rng = np.random.default_rng(42)  # FIX: deterministic baselines
    
    # Build global tag pool for random baseline (use tag_norm_eval for consistency)
    all_tags = tags_df["tag_norm_eval"].dropna().astype(str).map(lambda s: s.strip()).loc[lambda s: s != ""].unique().tolist()
    
    # Build top-K tags for frequency baseline (use tag_norm_eval for consistency)
    tag_counts = tags_df["tag_norm_eval"].dropna().astype(str).map(lambda s: s.strip()).loc[lambda s: s != ""].value_counts()
    top_k_tags = tag_counts.head(50).index.tolist()  # Top 50 most frequent
    
    # Shuffle venue IDs for shuffled baseline (derangement to avoid self-matches)
    venue_ids = list(venue_embeddings.keys())
    venue_shuffle_map = make_derangement(venue_ids, seed=42)
    
    for _, row in extractions_df.iterrows():
        venue_id = row["venue_id"]
        exp_id = row["exp_id"]
        
        venue_emb = venue_embeddings.get(venue_id)
        tag_emb = extraction_embeddings.get(exp_id)
        
        if venue_emb is None or tag_emb is None:
            continue
        
        # PRIMARY: actual retention
        retention_cosine = cosine_similarity(venue_emb, tag_emb)
        
        # BASELINE 1: Random tags (same count)
        n_tags = row.get("n_tags", 0)
        if n_tags > 0 and all_tags:
            random_tags = rng.choice(all_tags, size=min(n_tags, len(all_tags)), replace=False).tolist()
            random_tag_embs = [tag_embeddings[tag] for tag in random_tags if tag in tag_embeddings]
            if random_tag_embs:
                random_pooled = mean_pool(random_tag_embs)
                retention_random = cosine_similarity(venue_emb, random_pooled)
            else:
                retention_random = 0.0
        else:
            retention_random = 0.0
        
        # BASELINE 2: Shuffled venue mapping
        shuffled_venue_id = venue_shuffle_map[venue_id]
        shuffled_venue_emb = venue_embeddings.get(shuffled_venue_id)
        if shuffled_venue_emb is not None:
            retention_shuffled = cosine_similarity(shuffled_venue_emb, tag_emb)
        else:
            retention_shuffled = 0.0
        
        # BASELINE 3: Top-K frequent tags
        if n_tags > 0 and top_k_tags:
            top_k_sample = top_k_tags[:min(n_tags, len(top_k_tags))]
            top_k_embs = [tag_embeddings[tag] for tag in top_k_sample if tag in tag_embeddings]
            if top_k_embs:
                top_k_pooled = mean_pool(top_k_embs)
                retention_topk = cosine_similarity(venue_emb, top_k_pooled)
            else:
                retention_topk = 0.0
        else:
            retention_topk = 0.0
        
        # Delta retention (how much better than random baseline)
        delta_retention = retention_cosine - retention_random
        
        results.append({
            "venue_id": venue_id,
            "exp_id": exp_id,
            "model_key": row.get("model_key"),
            "prompt_type": row["prompt_type"],
            "run_number": row["run_number"],
            "retention_cosine": retention_cosine,  # PRIMARY
            "retention_random": retention_random,  # BASELINE 1
            "retention_shuffled": retention_shuffled,  # BASELINE 2
            "retention_topk": retention_topk,  # BASELINE 3
            "delta_retention": delta_retention,  # Improvement over random
            "cost_usd": row.get("cost_usd", 0),
            "total_tokens": row.get("total_tokens", 0),
            "n_tags": n_tags,
        })
    
    return pd.DataFrame(results)


def compute_uncertainty_dispersion(
    extractions_df: pd.DataFrame,
    extraction_embeddings: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Block F: Uncertainty signal as representation dispersion
    
    For each venue, compute:
    - Mean pairwise distance (1 - cosine) among all 24 representations
    - Within-run variance (run1 vs run2 comparisons)
    - Prompt variance (within model)
    - Model variance (within prompt)
    """
    results = []
    
    for venue_id, group in extractions_df.groupby("venue_id"):
        # Get all 24 representations (4 models × 3 prompts × 2 runs)
        venue_reps = []
        for _, row in group.iterrows():
            exp_id = row["exp_id"]
            emb = extraction_embeddings.get(exp_id)
            if emb is not None:
                venue_reps.append({
                    "emb": emb,
                    "model_key": row.get("model_key"),
                    "prompt_type": row["prompt_type"],
                    "run_number": row["run_number"],
                })
        
        if len(venue_reps) < 2:
            continue
        
        # Mean pairwise distance (1 - cosine)
        pairwise_distances = []
        for i, rep1 in enumerate(venue_reps):
            for j, rep2 in enumerate(venue_reps):
                if i < j:
                    cos_sim = cosine_similarity(rep1["emb"], rep2["emb"])
                    pairwise_distances.append(1.0 - cos_sim)
        
        mean_pairwise_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
        
        # Within-run variance (run1 vs run2)
        within_run_distances = []
        for model_key in ["openai", "gemini", "claude", "grok"]:
            for prompt_type in ["minimal", "anti_hallucination", "short_phrase"]:
                run1_reps = [r for r in venue_reps if r["model_key"] == model_key and 
                            r["prompt_type"] == prompt_type and r["run_number"] == 1]
                run2_reps = [r for r in venue_reps if r["model_key"] == model_key and 
                            r["prompt_type"] == prompt_type and r["run_number"] == 2]
                if run1_reps and run2_reps:
                    cos_sim = cosine_similarity(run1_reps[0]["emb"], run2_reps[0]["emb"])
                    within_run_distances.append(1.0 - cos_sim)
        
        within_run_mean_distance = np.mean(within_run_distances) if within_run_distances else 0.0
        
        # Prompt mean distance (within model, across prompts)
        prompt_mean_distances = []
        for model_key in ["openai", "gemini", "claude", "grok"]:
            model_reps = [r for r in venue_reps if r["model_key"] == model_key]
            if len(model_reps) >= 2:
                prompt_distances = []
                for i, rep1 in enumerate(model_reps):
                    for j, rep2 in enumerate(model_reps):
                        if i < j and rep1["prompt_type"] != rep2["prompt_type"]:
                            cos_sim = cosine_similarity(rep1["emb"], rep2["emb"])
                            prompt_distances.append(1.0 - cos_sim)
                if prompt_distances:
                    prompt_mean_distances.append(np.mean(prompt_distances))
        
        prompt_mean_distance = np.mean(prompt_mean_distances) if prompt_mean_distances else 0.0
        
        # Model mean distance (within prompt, across models)
        model_mean_distances = []
        for prompt_type in ["minimal", "anti_hallucination", "short_phrase"]:
            prompt_reps = [r for r in venue_reps if r["prompt_type"] == prompt_type]
            if len(prompt_reps) >= 2:
                model_distances = []
                for i, rep1 in enumerate(prompt_reps):
                    for j, rep2 in enumerate(prompt_reps):
                        if i < j and rep1["model_key"] != rep2["model_key"]:
                            cos_sim = cosine_similarity(rep1["emb"], rep2["emb"])
                            model_distances.append(1.0 - cos_sim)
                if model_distances:
                    model_mean_distances.append(np.mean(model_distances))
        
        model_mean_distance = np.mean(model_mean_distances) if model_mean_distances else 0.0
        
        results.append({
            "venue_id": venue_id,
            "mean_pairwise_distance": mean_pairwise_distance,  # PRIMARY
            "within_run_mean_distance": within_run_mean_distance,  # PRIMARY
            "prompt_mean_distance": prompt_mean_distance,  # PRIMARY
            "model_mean_distance": model_mean_distance,  # PRIMARY
            "n_representations": len(venue_reps),
        })
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def main():
    """Run Phase 2 analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Semantic Stability Analysis")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID prefix (e.g., week2_run_20251223_191104)"
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
        default="results",
        help="Directory containing Phase 1 results"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding computation (use cached)"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 2: SEMANTIC STABILITY ANALYSIS")
    print("=" * 60)
    print("\nCanonical representations:")
    print("  Reviews: R_mean = mean pooling of per-review embeddings")
    print("  Tags: T_mean_unique = dedupe by tag_norm_eval, then mean pool")
    
    # Load data
    print("\n1. Loading Phase 1 results...")
    extractions_df = load_all_extractions(args.run_id, args.results_dir)
    tags_df = load_all_tags(args.run_id, args.results_dir)
    reviews_dict = load_venue_reviews(args.data)
    
    print(f"   Loaded {len(extractions_df)} extractions")
    print(f"   Loaded {len(tags_df)} tag rows")
    print(f"   Loaded {len(reviews_dict)} venues with reviews")
    
    # Check cache if skipping embeddings
    if args.skip_embeddings:
        required_cache = [
            REVIEW_EMBEDDINGS_NPZ,
            REVIEW_EMBEDDINGS_MAP,
            TAG_EMBEDDINGS_NPZ,
            TAG_EMBEDDINGS_MAP
        ]
        missing = [p for p in required_cache if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing cache files: {missing}\n"
                f"Run without --skip-embeddings first to generate cache."
            )
    
    # Compute embeddings
    if not args.skip_embeddings:
        print("\n2. Computing embeddings...")
        client = get_embedding_client()
        
        print("   Computing review embeddings (R_mean)...")
        review_embs, venue_embs = compute_review_embeddings(
            client, reviews_dict, REVIEW_EMBEDDINGS_NPZ, REVIEW_EMBEDDINGS_MAP
        )
        print(f"   Computed embeddings for {len(venue_embs)} venues")
        
        print("   Computing tag embeddings (T_mean_unique)...")
        tag_embs, extraction_embs = compute_tag_embeddings(
            client, tags_df, TAG_EMBEDDINGS_NPZ, TAG_EMBEDDINGS_MAP
        )
        print(f"   Computed embeddings for {len(tag_embs)} unique tags")
        print(f"   Computed pooled embeddings for {len(extraction_embs)} extractions")
    else:
        print("\n2. Loading cached embeddings...")
        # Load from NPZ cache
        with open(REVIEW_EMBEDDINGS_MAP, 'r') as f:
            review_mapping = json.load(f)
        npz_data = np.load(REVIEW_EMBEDDINGS_NPZ)
        venue_embs = {}
        for venue_id, indices in review_mapping.items():
            review_embs_list = [npz_data[f"emb_{idx}"] for idx in indices]
            venue_embs[venue_id] = mean_pool(review_embs_list)
        
        with open(TAG_EMBEDDINGS_MAP, 'r') as f:
            tag_mapping = json.load(f)
        npz_data = np.load(TAG_EMBEDDINGS_NPZ)
        tag_embs = {tag: npz_data[f"emb_{idx}"] for tag, idx in tag_mapping.items()}
        
        # Reconstruct extraction embeddings
        extraction_embs = {}
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
                tag_embs_list = [tag_embs[tag] for tag in unique_tags if tag in tag_embs]
                if tag_embs_list:
                    extraction_embs[exp_id] = mean_pool(tag_embs_list)
                else:
                    extraction_embs[exp_id] = np.zeros(EMBEDDING_DIM)
            else:
                extraction_embs[exp_id] = np.zeros(EMBEDDING_DIM)
    
    # Compute metrics
    print("\n3. Computing metrics...")
    
    n_venues = extractions_df["venue_id"].nunique()
    n_models = extractions_df["model_key"].nunique()
    n_prompts = extractions_df["prompt_type"].nunique()
    
    print("   Block A: Run stability (cosine + MMC)...")
    run_stability = compute_run_stability(extractions_df, extraction_embs, tags_df, tag_embs)
    assert_approx("run_stability", len(run_stability), n_venues * n_models * n_prompts)
    run_stability.to_csv(TABLES_DIR / "run_stability.csv", index=False)
    print(f"   Computed {len(run_stability)} run comparisons")
    
    print("   Block B: Prompt sensitivity...")
    prompt_sensitivity = compute_prompt_sensitivity(extractions_df, extraction_embs, tags_df)
    prompt_sensitivity.to_csv(TABLES_DIR / "prompt_sensitivity.csv", index=False)
    print(f"   Computed {len(prompt_sensitivity)} prompt comparisons")
    
    print("   Block C: Model sensitivity...")
    model_sensitivity = compute_model_sensitivity(extractions_df, extraction_embs, tags_df)
    model_sensitivity.to_csv(TABLES_DIR / "model_sensitivity.csv", index=False)
    print(f"   Computed {len(model_sensitivity)} model comparisons")
    
    print("   Block D: Retention with baselines...")
    retention = compute_retention_with_baselines(
        extractions_df, venue_embs, extraction_embs, tags_df, tag_embs
    )
    assert_approx("retention", len(retention), len(extractions_df))
    
    # Add compression efficiency metrics
    retention["delta_retention_per_dollar"] = retention["delta_retention"] / retention["cost_usd"].clip(lower=0.0001)
    retention["delta_retention_per_token"] = retention["delta_retention"] / retention["total_tokens"].clip(lower=1)
    retention["delta_retention_per_tag"] = retention["delta_retention"] / retention["n_tags"].clip(lower=1)
    
    retention.to_csv(TABLES_DIR / "retention.csv", index=False)
    print(f"   Computed retention for {len(retention)} extractions")
    
    # Block E: Compression summary
    print("   Block E: Compression summary...")
    compression_summary = retention.groupby(["model_key", "prompt_type"]).agg(
        n_extractions=("exp_id", "count"),
        mean_retention=("retention_cosine", "mean"),
        mean_delta_retention=("delta_retention", "mean"),
        mean_cost=("cost_usd", "mean"),
        mean_tokens=("total_tokens", "mean"),
        mean_delta_per_dollar=("delta_retention_per_dollar", "mean"),
        median_delta_per_dollar=("delta_retention_per_dollar", "median"),
    ).reset_index()
    compression_summary.to_csv(TABLES_DIR / "compression_summary.csv", index=False)
    print(f"   Computed compression summary for {len(compression_summary)} model×prompt combinations")
    
    print("   Block F: Uncertainty dispersion...")
    uncertainty = compute_uncertainty_dispersion(extractions_df, extraction_embs)
    uncertainty.to_csv(TABLES_DIR / "uncertainty_dispersion.csv", index=False)
    print(f"   Computed uncertainty for {len(uncertainty)} venues")
    
    print("\n4. Summary statistics...")
    print(f"   Run stability - Mean cosine: {run_stability['cosine_similarity'].mean():.3f}")
    print(f"   Retention - Mean: {retention['retention_cosine'].mean():.3f}")
    print(f"   Retention - Mean delta (vs random): {retention['delta_retention'].mean():.3f}")
    print(f"   Uncertainty - Mean pairwise distance: {uncertainty['mean_pairwise_distance'].mean():.3f}")
    
    # Write manifest
    print("\n5. Writing manifest...")
    manifest = {
        "phase": "phase2",
        "run_id": args.run_id,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "dataset_csv": args.data,
        "results_dir": args.results_dir,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "counts": {
            "n_extractions": int(len(extractions_df)),
            "n_tag_rows": int(len(tags_df)),
            "n_venues_in_reviews": int(len(reviews_dict)),
            "n_venues_in_extractions": int(extractions_df["venue_id"].nunique()),
            "n_unique_tags_raw": int(tags_df["tag_raw"].dropna().nunique()) if "tag_raw" in tags_df.columns else None,
            "n_unique_tags_norm_eval": int(tags_df["tag_norm_eval"].dropna().astype(str).nunique()) if "tag_norm_eval" in tags_df.columns else None,
        },
        "seeds": {"baselines_rng": 42, "shuffled_venues": 42},
        "canonical_representations": {
            "reviews": "R_mean = mean pooling of per-review embeddings",
            "tags": "T_mean_unique = mean pooling of unique tag_norm_eval embeddings",
        }
    }
    with open(OUTPUT_DIR / "phase2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Manifest saved: {OUTPUT_DIR / 'phase2_manifest.json'}")
    
    print("\n✅ Phase 2 analysis complete!")
    print(f"   Tables: {TABLES_DIR}")
    print(f"   Plots: {PLOTS_DIR}")
    print("\nNext: Use notebooks/04_phase2_analysis.ipynb for exploration and plotting")


if __name__ == "__main__":
    main()
