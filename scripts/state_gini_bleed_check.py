#!/usr/bin/env python3
"""
State-Gini follow-up: Cross-facet similarity ("bleed" check).

For each gentag we compute similarity to all 10 facets; primary = max,
secondary = second max. We report the distribution of (primary - secondary)
to prove most gentags have a clear primary facet (no "embedding bleed").

Usage:
  poetry run python scripts/state_gini_bleed_check.py
  poetry run python scripts/state_gini_bleed_check.py --sample 5000
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PHASE2_CACHE = Path("results/phase2_cache")
OUTPUT_DIR = Path("results/phase3")
FACETS = [
    "food_quality", "coffee_drinks", "service", "ambiance", "price_value",
    "crowding", "seating", "dietary", "portions", "location",
]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (n1 * n2))


def load_embeddings() -> Dict[str, np.ndarray]:
    import json as _json
    slug = "text_embedding_3_large"
    tag_npz = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.npz"
    tag_map = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.map.json"
    if not tag_npz.exists():
        raise FileNotFoundError(f"Tag embeddings not found: {tag_npz}")
    with open(tag_map) as f:
        tag_idx_map = _json.load(f)
    tag_data = np.load(tag_npz)
    return {tag: tag_data[f"emb_{idx}"] for tag, idx in tag_idx_map.items() if f"emb_{idx}" in tag_data}


def main():
    parser = argparse.ArgumentParser(description="Bleed check: primary vs secondary facet similarity")
    parser.add_argument("--sample", type=int, default=0, help="If >0, use a random sample of tags (faster)")
    parser.add_argument("--out-json", type=str, default="", help="Summary JSON path")
    parser.add_argument("--out-csv", type=str, default="", help="Optional per-tag CSV (gap, primary_sim, ...)")
    args = parser.parse_args()

    print("Loading tag embeddings...")
    tag_embeddings = load_embeddings()
    tags = list(tag_embeddings.keys())
    if args.sample > 0 and len(tags) > args.sample:
        import random
        random.seed(42)
        tags = random.sample(tags, args.sample)
        print(f"Using sample of {len(tags)} tags")
    else:
        print(f"Using all {len(tags)} tags")

    # Load anchor embeddings from preflight or state_gini_full run (we need them)
    # Preflight writes results/phase3/preflight_orthogonality.json but not anchor vectors.
    # We must compute anchors (same as state_gini_full).
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
    import time
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    anchor_embs = []
    for i in range(0, len(anchor_texts), 16):
        batch = anchor_texts[i : i + 16]
        r = client.embeddings.create(model="text-embedding-3-large", input=batch)
        anchor_embs.extend([np.array(x.embedding) for x in r.data])
    anchor_embeddings = {f: anchor_embs[i] for i, f in enumerate(FACETS)}

    # For each tag: 10 similarities, sort descending, primary = sims[0], secondary = sims[1], gap = primary - secondary
    gaps = []
    primary_sims = []
    rows = []
    for tag in tqdm(tags, desc="Bleed check"):
        emb = tag_embeddings[tag]
        sims = [cosine_similarity(emb, anchor_embeddings[f]) for f in FACETS]
        sims_sorted = sorted(enumerate(sims), key=lambda x: -x[1])
        primary_idx, primary_sim = sims_sorted[0]
        secondary_sim = sims_sorted[1][1]
        gap = primary_sim - secondary_sim
        gaps.append(gap)
        primary_sims.append(primary_sim)
        if args.out_csv:
            rows.append({
                "tag": tag,
                "primary_facet": FACETS[primary_idx],
                "primary_sim": round(primary_sim, 4),
                "secondary_sim": round(secondary_sim, 4),
                "gap": round(gap, 4),
            })

    gaps = np.array(gaps)
    summary = {
        "n_tags": len(tags),
        "gap_mean": float(np.mean(gaps)),
        "gap_std": float(np.std(gaps)),
        "gap_median": float(np.median(gaps)),
        "pct_gap_lt_0.05": float(100 * np.mean(gaps < 0.05)),
        "pct_gap_lt_0.10": float(100 * np.mean(gaps < 0.10)),
        "pct_gap_ge_0.10": float(100 * np.mean(gaps >= 0.10)),
        "primary_sim_mean": float(np.mean(primary_sims)),
    }
    print("\nBleed check summary:")
    print(f"  gap (primary - secondary) mean: {summary['gap_mean']:.4f}  std: {summary['gap_std']:.4f}")
    print(f"  % near-miss (gap < 0.05): {summary['pct_gap_lt_0.05']:.1f}%")
    print(f"  % gap >= 0.10 (clear primary): {summary['pct_gap_ge_0.10']:.1f}%")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json) if args.out_json else OUTPUT_DIR / "bleed_check_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_json}")

    if args.out_csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"Saved per-tag CSV to {args.out_csv}")
    return summary


if __name__ == "__main__":
    main()
