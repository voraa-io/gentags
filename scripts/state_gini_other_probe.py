#!/usr/bin/env python3
"""
State-Gini follow-up: Semantic identity of the "Other" bucket.

Exports all gentags that were assigned to Other (below τ) with their
best-matching facet and similarity, for qualitative probe: are they noise
or long-tail semantic content?

Usage:
  poetry run python scripts/state_gini_other_probe.py
  poetry run python scripts/state_gini_other_probe.py --sample 500 --out results/phase3/other_bucket_sample.csv
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Same config as state_gini_full
PHASE2_CACHE = Path("results/phase2_cache")
OUTPUT_DIR = Path("results/phase3")
FACETS = [
    "food_quality", "coffee_drinks", "service", "ambiance", "price_value",
    "crowding", "seating", "dietary", "portions", "location",
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
TAU_DEFAULT = 0.35


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def assign_with_best(
    tag: str,
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[Optional[str], float, str]:
    """Returns (assigned_facet or None, best_sim, best_facet_name)."""
    if tag not in tag_embeddings:
        return None, 0.0, ""
    tag_emb = tag_embeddings[tag]
    best_facet = ""
    best_sim = -1.0
    for facet in FACETS:
        sim = cosine_similarity(tag_emb, anchor_embeddings[facet])
        if sim > best_sim:
            best_sim = sim
            best_facet = facet
    if best_sim >= threshold:
        return best_facet, best_sim, best_facet
    return None, best_sim, best_facet


def load_embeddings():
    import json
    slug = "text_embedding_3_large"
    tag_npz = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.npz"
    tag_map = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.map.json"
    if not tag_npz.exists():
        raise FileNotFoundError(f"Tag embeddings not found: {tag_npz}")
    with open(tag_map) as f:
        tag_idx_map = json.load(f)
    tag_data = np.load(tag_npz)
    return {tag: tag_data[f"emb_{idx}"] for tag, idx in tag_idx_map.items() if f"emb_{idx}" in tag_data}


def embed_texts_batch(client, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
    import time
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                r = client.embeddings.create(model="text-embedding-3-large", input=batch)
                all_embs.extend([np.array(item.embedding) for item in r.data])
                break
            except Exception:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise
    return all_embs


def compute_anchor_embeddings(client):
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    embs = embed_texts_batch(client, anchor_texts, batch_size=16)
    return {facet: embs[i] for i, facet in enumerate(FACETS)}


def get_embedding_client():
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


def main():
    parser = argparse.ArgumentParser(description="Export gentags in Other bucket for qualitative probe")
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--run-id", type=str, default="week2_run_20251223_191104")
    parser.add_argument("--results-dir", type=str, default="results/phase1_downloaded")
    parser.add_argument("--sample", type=int, default=0, help="If >0, only save this many random Other tags (for coding)")
    parser.add_argument("--out", type=str, default="", help="Output CSV path (default: results/phase3/other_bucket_tags.csv)")
    args = parser.parse_args()

    # Load tags from Phase 1 (unique gentags)
    results_path = Path(args.results_dir)
    tag_files = list(results_path.glob(f"{args.run_id}_tags_*.csv"))
    if not tag_files:
        raise FileNotFoundError(f"No tag files for {args.run_id} in {args.results_dir}")
    tags_df = pd.concat([pd.read_csv(f) for f in tag_files], ignore_index=True)
    all_tags = tags_df["tag_norm_eval"].dropna().astype(str).str.strip().unique().tolist()
    all_tags = [t for t in all_tags if t]
    print(f"Loaded {len(all_tags)} unique gentags from Phase 1")

    # Load embeddings
    print("Loading tag embeddings...")
    tag_embeddings = load_embeddings()
    in_cache = [t for t in all_tags if t in tag_embeddings]
    print(f"  {len(in_cache)} tags have cached embeddings")

    # Anchor embeddings
    print("Computing anchor embeddings...")
    client = get_embedding_client()
    anchor_embeddings = compute_anchor_embeddings(client)

    # Classify each tag; collect Other
    other_rows = []
    for tag in tqdm(in_cache, desc="Assigning"):
        assigned, best_sim, best_facet = assign_with_best(tag, tag_embeddings, anchor_embeddings, args.tau)
        if assigned is None:
            other_rows.append({"tag": tag, "best_facet": best_facet, "best_sim": round(best_sim, 4)})

    print(f"Tags in Other (below τ={args.tau}): {len(other_rows)}")

    if args.sample > 0 and len(other_rows) > args.sample:
        import random
        random.seed(42)
        other_rows = random.sample(other_rows, args.sample)
        print(f"Sampled {len(other_rows)} for coding")

    out_path = Path(args.out) if args.out else OUTPUT_DIR / "other_bucket_tags.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(other_rows).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return other_rows


if __name__ == "__main__":
    main()
