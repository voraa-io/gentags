#!/usr/bin/env python3
"""
State-Gini follow-up: Like-for-like unit alignment.

Compute Gentag State-Gini per VENUE (not per extraction): pool all gentags
for each venue across extractions, then assign to facets and compute Gini.
Output can be compared directly to phase3a baselines (also per venue).

Usage:
  poetry run python scripts/state_gini_venue_aggregate.py
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PHASE2_CACHE = Path("results/phase2_cache")
OUTPUT_DIR = Path("results/phase3")
TABLES_DIR = OUTPUT_DIR / "tables"
FACETS = [
    "food_quality", "coffee_drinks", "service", "ambiance", "price_value",
    "crowding", "seating", "dietary", "portions", "location",
]
TAU_DEFAULT = 0.35


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def gini_coefficient(values: np.ndarray) -> float:
    values = np.abs(values).astype(float)
    if values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return max(0.0, gini)


def hard_assign_facet(tag: str, tag_embeddings: Dict, anchor_embeddings: Dict, threshold: float) -> Tuple[Optional[str], float]:
    if tag not in tag_embeddings:
        return None, 0.0
    emb = tag_embeddings[tag]
    best_facet, best_sim = None, -1.0
    for facet in FACETS:
        sim = cosine_similarity(emb, anchor_embeddings[facet])
        if sim > best_sim:
            best_sim = sim
            best_facet = facet
    if best_sim >= threshold:
        return best_facet, best_sim
    return None, best_sim


def compute_facet_counts(tags: List[str], tag_embeddings: Dict, anchor_embeddings: Dict, threshold: float) -> Tuple[Dict[str, int], int]:
    counts = {f: 0 for f in FACETS}
    other = 0
    for tag in tags:
        facet, _ = hard_assign_facet(tag, tag_embeddings, anchor_embeddings, threshold)
        if facet:
            counts[facet] += 1
        else:
            other += 1
    return counts, other


def load_embeddings() -> Tuple[Dict[str, np.ndarray], Dict]:
    import json
    slug = "text_embedding_3_large"
    tag_npz = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.npz"
    tag_map = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.map.json"
    if not tag_npz.exists():
        raise FileNotFoundError(f"Tag embeddings not found: {tag_npz}")
    with open(tag_map) as f:
        tag_idx_map = json.load(f)
    tag_data = np.load(tag_npz)
    tag_embeddings = {tag: tag_data[f"emb_{idx}"] for tag, idx in tag_idx_map.items() if f"emb_{idx}" in tag_data}
    return tag_embeddings, {}


def load_phase1_data(run_id: str, results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_path = Path(results_dir)
    extraction_files = list(results_path.glob(f"{run_id}_extractions_*.csv"))
    tag_files = list(results_path.glob(f"{run_id}_tags_*.csv"))
    if not extraction_files:
        raise FileNotFoundError(f"No extraction files for {run_id}")
    extractions = pd.concat([pd.read_csv(f) for f in extraction_files], ignore_index=True)
    tags = pd.concat([pd.read_csv(f) for f in tag_files], ignore_index=True)
    return extractions, tags


def main():
    parser = argparse.ArgumentParser(description="Venue-aggregated Gentag State-Gini (like-for-like with baselines)")
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--run-id", type=str, default="week2_run_20251223_191104")
    parser.add_argument("--results-dir", type=str, default="results/phase1_downloaded")
    parser.add_argument("--data", type=str, default="data/study1_venues_20250117.csv")
    parser.add_argument("--quality-venues", type=str, default="results/phase2/tables/retention.csv", help="CSV with venue_id for filtering (phase3a uses retention venues)")
    args = parser.parse_args()

    # Use same venue set as phase3a (quality_venues from retention)
    quality_path = Path(args.quality_venues)
    if quality_path.exists():
        retention = pd.read_csv(quality_path)
        quality_venue_ids = set(retention["venue_id"].unique())
        print(f"Filtering to {len(quality_venue_ids)} quality venues (from retention)")
    else:
        quality_venue_ids = None

    print("Loading Phase 1 data...")
    extractions_df, tags_df = load_phase1_data(args.run_id, args.results_dir)
    venues_df = pd.read_csv(args.data)
    all_models = {"claude", "gemini", "grok", "openai"}
    venue_model_counts = extractions_df.groupby("venue_id")["model_key"].apply(lambda x: set(x.unique()))
    complete_venues = set(venue_model_counts[venue_model_counts.apply(lambda x: x >= all_models)].index)
    if quality_venue_ids is not None:
        complete_venues &= quality_venue_ids
    extractions_df = extractions_df[extractions_df["venue_id"].isin(complete_venues)]
    valid_exp_ids = set(extractions_df["exp_id"].unique())
    tags_df = tags_df[tags_df["exp_id"].isin(valid_exp_ids)]
    print(f"  {len(complete_venues)} venues, {len(extractions_df)} extractions")

    print("Loading embeddings...")
    tag_embeddings, _ = load_embeddings()
    print("Computing anchor embeddings...")
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    for p in [Path(__file__).parent.parent / ".env", Path.cwd() / ".env"]:
        if p.exists():
            load_dotenv(p)
            break
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    anchor_embs = []
    for i in range(0, len(anchor_texts), 16):
        r = client.embeddings.create(model="text-embedding-3-large", input=anchor_texts[i : i + 16])
        anchor_embs.extend([np.array(x.embedding) for x in r.data])
    anchor_embeddings = {f: anchor_embs[i] for i, f in enumerate(FACETS)}

    # Per venue: pool all gentags
    results = []
    for venue_id in tqdm(sorted(complete_venues), desc="Venue Gini"):
        exp_ids = extractions_df[extractions_df["venue_id"] == venue_id]["exp_id"].tolist()
        gentags = (
            tags_df.loc[tags_df["exp_id"].isin(exp_ids), "tag_norm_eval"]
            .dropna().astype(str).str.strip().tolist()
        )
        gentags = [t for t in gentags if t and t in tag_embeddings]
        if not gentags:
            continue
        counts, other = compute_facet_counts(gentags, tag_embeddings, anchor_embeddings, args.tau)
        count_array = np.array([counts[f] for f in FACETS])
        state_gini = gini_coefficient(count_array)
        results.append({
            "venue_id": venue_id,
            "n_tags": len(gentags),
            "assigned_count": len(gentags) - other,
            "other_count": other,
            "other_rate_pct": round(100 * other / len(gentags), 2),
            "state_gini": round(state_gini, 4),
        })
        for f in FACETS:
            results[-1][f"count_{f}"] = counts[f]

    df = pd.DataFrame(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "venue_gentag_state_gini.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"  State-Gini (venue-level) mean: {df['state_gini'].mean():.3f}  std: {df['state_gini'].std():.3f}")
    print(f"  Other rate (mean): {df['other_rate_pct'].mean():.1f}%")
    return df


if __name__ == "__main__":
    main()
