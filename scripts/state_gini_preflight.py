#!/usr/bin/env python3
"""
State-Gini preflight: orthogonality check and tau sensitivity.

Run before the full State-Gini experiment (state_gini_full.py) to:
1. Check anchor embeddings pairwise cosine (max < 0.55–0.60).
2. Sweep tau in [0.30, 0.35, 0.40] and report mean other_rate only (no State-Gini in preflight).

Usage:
  poetry run python scripts/state_gini_preflight.py --orthogonality
  poetry run python scripts/state_gini_preflight.py --tau-sweep [--sample 10]
  poetry run python scripts/state_gini_preflight.py --orthogonality --tau-sweep
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path (same as phase2_analysis, phase3_analysis_v2)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# CONFIGURATION (must match phase3_analysis_v2 for same results)
# =============================================================================

PHASE2_CACHE = Path("results/phase2_cache")
OUTPUT_DIR = Path("results/phase3")

EMBEDDING_MODEL = "text-embedding-3-large"
SEMANTIC_THRESHOLD = 0.35

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

# =============================================================================
# CORE FUNCTIONS (same logic as phase3_analysis_v2)
# =============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a distribution (high = concentrated)."""
    values = np.abs(values).astype(float)
    if values.sum() == 0:
        return 0.0
    sorted_values = np.sort(values)
    n = len(values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return max(0.0, gini)


def hard_assign_facet(tag_emb: np.ndarray, anchor_embeddings: Dict[str, np.ndarray], threshold: float) -> Tuple[str, float]:
    """Argmax facet and sim; caller treats sim < threshold as 'other'."""
    best_facet = None
    best_sim = -1.0
    for facet in FACETS:
        sim = cosine_similarity(tag_emb, anchor_embeddings[facet])
        if sim > best_sim:
            best_sim = sim
            best_facet = facet
    return best_facet, best_sim


def compute_state_gini(
    tags: List[str],
    tag_embeddings: Dict[str, np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[float, Dict[str, int], int]:
    """State-Gini and facet counts (same as phase3_analysis_v2)."""
    counts = {facet: 0 for facet in FACETS}
    other_count = 0
    for tag in tags:
        if tag not in tag_embeddings:
            other_count += 1
            continue
        facet, sim = hard_assign_facet(tag_embeddings[tag], anchor_embeddings, threshold)
        if sim >= threshold:
            counts[facet] += 1
        else:
            other_count += 1
    count_array = np.array([counts[f] for f in FACETS])
    state_gini = gini_coefficient(count_array)
    return state_gini, counts, other_count


# =============================================================================
# EMBEDDING AND DATA LOADING
# =============================================================================

def get_embedding_client():
    """OpenAI client (same as phase3_analysis_v2)."""
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


def embed_texts_batch(client, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
    """Embed texts (same as phase3_analysis_v2)."""
    import time
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                all_embeddings.extend([np.array(item.embedding) for item in response.data])
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise e
    return all_embeddings


def compute_anchor_embeddings(client) -> Dict[str, np.ndarray]:
    """Anchor embeddings for the 10 facets."""
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    embeddings = embed_texts_batch(client, anchor_texts, batch_size=16)
    return {facet: emb for facet, emb in zip(FACETS, embeddings)}


def load_embeddings_cache() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load tag (and venue) embeddings from Phase 2 cache."""
    slug = EMBEDDING_MODEL.replace("-", "_").replace(".", "_")
    tag_npz = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.npz"
    tag_map = PHASE2_CACHE / f"tag_embeddings_{slug}_normeval.map.json"
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
    return tag_embeddings, {}


def load_phase1_data(run_id: str, results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
# PREFLIGHT RUNNERS
# =============================================================================

def run_orthogonality(save_json: bool = True) -> dict:
    """Compute anchor pairwise cosines; report max and matrix."""
    print("State-Gini preflight: anchor orthogonality")
    print("=" * 60)
    client = get_embedding_client()
    anchor_embeddings = compute_anchor_embeddings(client)
    n = len(FACETS)
    matrix = [[0.0] * n for _ in range(n)]
    max_cos = -1.0
    max_pair = (None, None)
    above_60 = []

    for i, f1 in enumerate(FACETS):
        for j, f2 in enumerate(FACETS):
            if i == j:
                matrix[i][j] = 1.0
                continue
            c = cosine_similarity(anchor_embeddings[f1], anchor_embeddings[f2])
            matrix[i][j] = round(c, 4)
            if i < j:
                if c > max_cos:
                    max_cos = c
                    max_pair = (f1, f2)
                if c > 0.60:
                    above_60.append((f1, f2, round(c, 4)))

    print("\nPairwise cosine (anchor embeddings):")
    print("       " + " ".join(f"{f[:6]:>6}" for f in FACETS))
    for i, f in enumerate(FACETS):
        row = " ".join(f"{matrix[i][j]:>6.3f}" for j in range(n))
        print(f"{f[:6]:>6} {row}")
    print(f"\nMax pairwise cosine (i<j): {max_cos:.4f}  pair: {max_pair[0]} / {max_pair[1]}")
    if above_60:
        print(f"WARNING: {len(above_60)} pair(s) above 0.60: {above_60}")
    else:
        print("OK: no pair above 0.60.")

    out = {
        "max_pairwise_cosine": round(max_cos, 4),
        "max_pair": list(max_pair),
        "pairs_above_0_60": [list(p) for p in above_60],
        "matrix": matrix,
        "facet_order": FACETS,
    }
    if save_json:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "preflight_orthogonality.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {path}")
    return out


def run_tau_sweep(
    run_id: str,
    results_dir: str,
    sample_venues: int = None,
    taus: List[float] = None,
    save_json: bool = True,
) -> dict:
    """
    Sweep tau; report assignment only (other_rate). No State-Gini in preflight.
    We only check how many tags fall below each threshold (go to 'other').
    """
    if taus is None:
        taus = [0.30, 0.35, 0.40]
    print("State-Gini preflight: tau sensitivity (assignment only, no Gini)")
    print("=" * 60)
    print(f"tau values: {taus}")
    if sample_venues:
        print(f"Sampling: first {sample_venues} venues")
    print()

    extractions_df, tags_df = load_phase1_data(run_id, results_dir)
    tag_embeddings, _ = load_embeddings_cache()
    client = get_embedding_client()
    anchor_embeddings = compute_anchor_embeddings(client)

    if sample_venues is not None:
        venue_ids = extractions_df["venue_id"].unique()[:sample_venues]
        extractions_df = extractions_df[extractions_df["venue_id"].isin(venue_ids)]
        tags_df = tags_df[tags_df["exp_id"].isin(extractions_df["exp_id"])]

    results_per_tau = {tau: [] for tau in taus}

    for _, row in tqdm(extractions_df.iterrows(), total=len(extractions_df), desc="tau sweep"):
        exp_id = row["exp_id"]
        gentags = (
            tags_df.loc[tags_df["exp_id"] == exp_id, "tag_norm_eval"]
            .dropna().astype(str).map(str.strip).tolist()
        )
        gentags = [t for t in gentags if t]
        if not gentags:
            continue
        for tau in taus:
            other_count = 0
            for tag in gentags:
                if tag not in tag_embeddings:
                    other_count += 1
                    continue
                _, sim = hard_assign_facet(tag_embeddings[tag], anchor_embeddings, tau)
                if sim < tau:
                    other_count += 1
            n = len(gentags)
            other_rate = other_count / n if n else 0.0
            results_per_tau[tau].append((other_rate, n))

    summary = []
    for tau in taus:
        items = results_per_tau[tau]
        if not items:
            summary.append({"tau": tau, "n_extractions": 0, "mean_other_rate": None})
            continue
        mean_other = np.mean([x[0] for x in items])
        summary.append({
            "tau": tau,
            "n_extractions": len(items),
            "mean_other_rate": round(mean_other, 4),
        })

    print("tau     n_extractions   mean_other_rate")
    print("-" * 45)
    for s in summary:
        o = s["mean_other_rate"] if s["mean_other_rate"] is not None else "—"
        print(f"{s['tau']:.2f}   {s['n_extractions']:>14}   {o!s:>16}")

    out = {"run_id": run_id, "taus": taus, "sample_venues": sample_venues, "summary": summary}
    if save_json:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / "preflight_tau_sweep.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {path}")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="State-Gini preflight: orthogonality and tau sweep")
    parser.add_argument("--orthogonality", action="store_true", help="Anchor pairwise cosine check")
    parser.add_argument("--tau-sweep", action="store_true", help="Sweep tau 0.30, 0.35, 0.40")
    parser.add_argument("--run-id", type=str, default="week2_run_20251223_191104")
    parser.add_argument("--results-dir", type=str, default="results/phase1_downloaded")
    parser.add_argument("--sample", type=int, default=None, metavar="N", help="Tau-sweep: use first N venues")
    parser.add_argument("--no-save", action="store_true", help="Do not write JSON")
    args = parser.parse_args()

    if not args.orthogonality and not args.tau_sweep:
        parser.print_help()
        print("\nUse --orthogonality and/or --tau-sweep.")
        sys.exit(1)

    save = not args.no_save
    if args.orthogonality:
        run_orthogonality(save_json=save)
    if args.tau_sweep:
        run_tau_sweep(
            run_id=args.run_id,
            results_dir=args.results_dir,
            sample_venues=args.sample,
            save_json=save,
        )


if __name__ == "__main__":
    main()
