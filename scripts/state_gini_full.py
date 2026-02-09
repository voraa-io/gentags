#!/usr/bin/env python3
"""
Full State-Gini experiment: compute State-Gini and Drift-Gini for gentags.

Uses the 10 facet anchors defined in docs/PHASE3_STATE_GINI_PLAN.md.
Hard assignment: each tag → argmax facet if sim >= τ, else "other".
Gini on facet counts (State-Gini) and on profile change (Drift-Gini).

Prerequisites: preflight done (orthogonality + τ sweep). Phase 1 extractions
and Phase 2 tag embeddings in place.

Usage:
  poetry run python scripts/state_gini_full.py
  poetry run python scripts/state_gini_full.py --tau 0.35
  poetry run python scripts/state_gini_full.py --tau 0.30 --output-suffix _tau030
"""

import sys
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

PHASE2_CACHE = Path("results/phase2_cache")
OUTPUT_DIR = Path("results/phase3")
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"

# =============================================================================
# FACET DEFINITIONS (frozen; see PHASE3_STATE_GINI_PLAN.md)
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
    "location",
]

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
    """Cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient. High = localized, low = diffuse."""
    values = np.abs(values).astype(float)
    if values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return max(0.0, gini)


def hard_assign_facet(
    tag: str,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[Optional[str], float]:
    """Assign tag to one facet (argmax) if sim >= threshold; else other. Returns (facet or None, sim)."""
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
    return None, best_sim


def compute_facet_counts(
    tags: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[Dict[str, int], int]:
    """Integer counts per facet + other count."""
    counts = {facet: 0 for facet in FACETS}
    other_count = 0
    for tag in tags:
        facet, _ = hard_assign_facet(tag, tag_embeddings, anchor_embeddings, threshold)
        if facet is not None:
            counts[facet] += 1
        else:
            other_count += 1
    return counts, other_count


def compute_state_gini(
    tags: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[float, Dict[str, int], int]:
    """State-Gini: Gini on facet counts. Returns (gini, counts, other_count)."""
    counts, other_count = compute_facet_counts(tags, tag_embeddings, anchor_embeddings, threshold)
    count_array = np.array([counts[f] for f in FACETS])
    state_gini = gini_coefficient(count_array)
    return state_gini, counts, other_count


def compute_drift_gini(
    tags1: List[str],
    tags2: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> float:
    """Drift-Gini: Gini on change between two facet profiles (secondary metric)."""
    counts1, _ = compute_facet_counts(tags1, tag_embeddings, anchor_embeddings, threshold)
    counts2, _ = compute_facet_counts(tags2, tag_embeddings, anchor_embeddings, threshold)
    total1 = sum(counts1.values())
    total2 = sum(counts2.values())
    if total1 == 0 or total2 == 0:
        return 0.0
    profile1 = np.array([counts1[f] / total1 for f in FACETS])
    profile2 = np.array([counts2[f] / total2 for f in FACETS])
    drift = np.abs(profile1 - profile2)
    return gini_coefficient(drift)


# =============================================================================
# EMBEDDING AND DATA LOADING
# =============================================================================


def get_embedding_client():
    """OpenAI embedding client."""
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
        batch = texts[i : i + batch_size]
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
    """Embed facet anchors."""
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    embeddings = embed_texts_batch(client, anchor_texts, batch_size=16)
    return {facet: emb for facet, emb in zip(FACETS, embeddings)}


def load_embeddings_cache() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load Phase 2 tag and venue embeddings."""
    slug = EMBEDDING_MODEL.replace("-", "_").replace(".", "_")
    tag_npz = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.npz"
    tag_map = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.map.json"
    if not tag_npz.exists():
        raise FileNotFoundError(f"Tag embeddings not found: {tag_npz}")
    with open(tag_map) as f:
        tag_idx_map = json.load(f)
    tag_data = np.load(tag_npz)
    tag_embeddings = {tag: tag_data[f"emb_{idx}"] for tag, idx in tag_idx_map.items() if f"emb_{idx}" in tag_data}

    venue_embeddings = {}
    venue_npz = PHASE2_CACHE / f"review_embeddings_{slug}.npz"
    venue_map = PHASE2_CACHE / f"review_embeddings_{slug}.map.json"
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


def load_phase1_data(run_id: str, results_dir: str = "results/phase1_downloaded") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1 extractions and tags."""
    results_path = Path(results_dir)
    extraction_files = list(results_path.glob(f"{run_id}_extractions_*.csv"))
    tag_files = list(results_path.glob(f"{run_id}_tags_*.csv"))
    if not extraction_files:
        raise FileNotFoundError(f"No extraction files for {run_id} in {results_dir}")
    extractions = pd.concat([pd.read_csv(f) for f in extraction_files], ignore_index=True)
    tags = pd.concat([pd.read_csv(f) for f in tag_files], ignore_index=True)
    return extractions, tags


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def run_state_localization(
    extractions_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    """Compute State-Gini per extraction."""
    results = []
    for _, row in tqdm(extractions_df.iterrows(), total=len(extractions_df), desc="State-Gini"):
        exp_id = row["exp_id"]
        venue_id = row["venue_id"]
        model_key = row["model_key"]
        prompt_type = row["prompt_type"]
        run_number = row["run_number"]

        gentags = (
            tags_df.loc[tags_df["exp_id"] == exp_id, "tag_norm_eval"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .tolist()
        )
        gentags = [t for t in gentags if t]
        if not gentags:
            continue

        state_gini, counts, other = compute_state_gini(gentags, tag_embeddings, anchor_embeddings, threshold)
        result = {
            "exp_id": exp_id,
            "venue_id": venue_id,
            "model_key": model_key,
            "prompt_type": prompt_type,
            "run_number": run_number,
            "n_gentags": len(gentags),
            "gentag_state_gini": state_gini,
            "gentag_other_count": other,
            "gentag_assigned_count": len(gentags) - other,
        }
        for facet in FACETS:
            result[f"gentag_count_{facet}"] = counts.get(facet, 0)
        results.append(result)
    return pd.DataFrame(results)


def run_drift_localization(
    extractions_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    """Compute Drift-Gini per run pair (secondary)."""
    results = []
    groups = list(extractions_df.groupby(["venue_id", "model_key", "prompt_type"]))
    for (venue_id, model_key, prompt_type), group in tqdm(groups, desc="Drift-Gini"):
        runs = group.sort_values("run_number")
        if len(runs) < 2:
            continue
        run1, run2 = runs.iloc[0], runs.iloc[1]
        exp_id1, exp_id2 = run1["exp_id"], run2["exp_id"]
        tags1 = (
            tags_df.loc[tags_df["exp_id"] == exp_id1, "tag_norm_eval"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .tolist()
        )
        tags2 = (
            tags_df.loc[tags_df["exp_id"] == exp_id2, "tag_norm_eval"]
            .dropna()
            .astype(str)
            .map(str.strip)
            .tolist()
        )
        tags1 = [t for t in tags1 if t]
        tags2 = [t for t in tags2 if t]
        if not tags1 or not tags2:
            continue
        drift_gini = compute_drift_gini(tags1, tags2, tag_embeddings, anchor_embeddings, threshold)
        results.append({
            "venue_id": venue_id,
            "model_key": model_key,
            "prompt_type": prompt_type,
            "exp_id_run1": exp_id1,
            "exp_id_run2": exp_id2,
            "n_tags_run1": len(tags1),
            "n_tags_run2": len(tags2),
            "drift_gini": drift_gini,
        })
    return pd.DataFrame(results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full State-Gini experiment (gentags)")
    parser.add_argument("--run-id", type=str, default="week2_run_20251223_191104")
    parser.add_argument("--data", type=str, default="data/study1_venues_20250117.csv")
    parser.add_argument("--results-dir", type=str, default="results/phase1_downloaded")
    parser.add_argument("--tau", type=float, default=0.35, help="Semantic threshold for hard assignment (e.g. 0.30, 0.35, 0.40)")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output files, e.g. _tau030 for sensitivity runs")
    args = parser.parse_args()

    threshold = args.tau
    suffix = args.output_suffix

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STATE-GINI FULL RUN")
    print("=" * 60)
    print(f"  τ (threshold): {threshold}")
    print(f"  Facets: {len(FACETS)} (from PHASE3_STATE_GINI_PLAN.md)")
    print()

    print("1. Loading data...")
    extractions_df, tags_df = load_phase1_data(args.run_id, args.results_dir)
    print(f"   {len(extractions_df)} extractions, {len(tags_df)} tags")

    venues_df = pd.read_csv(args.data)
    all_models = {"claude", "gemini", "grok", "openai"}
    venue_model_counts = extractions_df.groupby("venue_id")["model_key"].apply(lambda x: set(x.unique()))
    complete_venues = set(venue_model_counts[venue_model_counts.apply(lambda x: x >= all_models)].index)
    extractions_df = extractions_df[extractions_df["venue_id"].isin(complete_venues)]
    valid_exp_ids = set(extractions_df["exp_id"].unique())
    tags_df = tags_df[tags_df["exp_id"].isin(valid_exp_ids)]
    print(f"   {len(complete_venues)} complete venues")

    print("\n2. Loading embeddings...")
    tag_embeddings, venue_embeddings = load_embeddings_cache()
    print(f"   {len(tag_embeddings)} tag embeddings")

    print("\n3. Computing anchor embeddings...")
    client = get_embedding_client()
    anchor_embeddings = compute_anchor_embeddings(client)
    print(f"   {len(anchor_embeddings)} facet anchors")

    print("\n4. State-Gini...")
    state_df = run_state_localization(
        extractions_df, tags_df, tag_embeddings, anchor_embeddings, threshold
    )
    state_path = TABLES_DIR / f"state_localization{suffix}.csv"
    state_df.to_csv(state_path, index=False)
    print(f"   Saved {len(state_df)} rows → {state_path}")

    print("\n5. Drift-Gini...")
    drift_df = run_drift_localization(
        extractions_df, tags_df, tag_embeddings, anchor_embeddings, threshold
    )
    drift_path = TABLES_DIR / f"drift_localization{suffix}.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"   Saved {len(drift_df)} rows → {drift_path}")

    print("\n" + "=" * 60)
    print("STATE-GINI (main)")
    print("=" * 60)
    print(f"   Mean: {state_df['gentag_state_gini'].mean():.3f}  Std: {state_df['gentag_state_gini'].std():.3f}")

    print("\nDRIFT-GINI (secondary)")
    print(f"   Mean: {drift_df['drift_gini'].mean():.3f}  Std: {drift_df['drift_gini'].std():.3f}")

    manifest = {
        "script": "state_gini_full",
        "run_id": args.run_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "threshold": threshold,
        "output_suffix": suffix,
        "counts": {
            "n_extractions": len(state_df),
            "n_venues": state_df["venue_id"].nunique(),
            "n_drift_pairs": len(drift_df),
        },
        "results": {
            "state_gini_mean": float(state_df["gentag_state_gini"].mean()),
            "state_gini_std": float(state_df["gentag_state_gini"].std()),
            "drift_gini_mean": float(drift_df["drift_gini"].mean()),
            "drift_gini_std": float(drift_df["drift_gini"].std()),
        },
        "facets": FACETS,
    }
    manifest_path = OUTPUT_DIR / f"phase3_v2_manifest{suffix}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    return state_df, drift_df


if __name__ == "__main__":
    main()
