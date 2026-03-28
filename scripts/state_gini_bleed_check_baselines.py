#!/usr/bin/env python3
"""
Phase 3 follow-up: bleed check for lexical baselines.

For each baseline keyword we compute similarity to all 10 facet anchors, then
record:
  - primary similarity (best facet)
  - secondary similarity (second-best facet)
  - gap = primary - secondary

This mirrors the gentag bleed check so we can compare argmax ambiguity across
RAKE, YAKE, and TF-IDF under the same facet space.

Usage:
  poetry run python scripts/state_gini_bleed_check_baselines.py
  poetry run python scripts/state_gini_bleed_check_baselines.py --sample-venues 50
  poetry run python scripts/state_gini_bleed_check_baselines.py --out-csv results/phase3/baseline_bleed_rows.csv
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase3a_baselines import (  # type: ignore
    FACETS,
    FACET_ANCHORS,
    DEFAULT_K,
    MAX_PHRASE_WORDS,
    concatenate_reviews,
    cosine_similarity,
    embed_texts_batch,
    extract_rake_keywords,
    extract_review_texts,
    extract_tfidf_keywords,
    extract_yake_keywords,
    get_embedding_client,
)


def summarize_gaps(rows: List[dict]) -> dict:
    gaps = np.array([row["gap"] for row in rows], dtype=float)
    primary_sims = np.array([row["primary_sim"] for row in rows], dtype=float)
    return {
        "n_keywords": int(len(rows)),
        "gap_mean": float(np.mean(gaps)) if len(gaps) else 0.0,
        "gap_std": float(np.std(gaps)) if len(gaps) else 0.0,
        "gap_median": float(np.median(gaps)) if len(gaps) else 0.0,
        "pct_gap_lt_0.05": float(100 * np.mean(gaps < 0.05)) if len(gaps) else 0.0,
        "pct_gap_lt_0.10": float(100 * np.mean(gaps < 0.10)) if len(gaps) else 0.0,
        "pct_gap_ge_0.10": float(100 * np.mean(gaps >= 0.10)) if len(gaps) else 0.0,
        "primary_sim_mean": float(np.mean(primary_sims)) if len(primary_sims) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bleed check for RAKE/YAKE/TF-IDF baselines")
    parser.add_argument("--sample-venues", type=int, default=0, help="Optional venue sample for faster runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when sampling venues")
    parser.add_argument("--out-json", type=str, default="results/phase3/baseline_bleed_check_summary.json")
    parser.add_argument("--out-csv", type=str, default="", help="Optional per-keyword output CSV")
    args = parser.parse_args()

    venue_df = pd.read_csv("data/study1_venues_20250117.csv")
    gentag_retention = pd.read_csv("results/phase2/tables/retention.csv")
    quality_venues = gentag_retention["venue_id"].unique().tolist()
    venue_df = venue_df[venue_df["id"].isin(quality_venues)].copy()

    if args.sample_venues and len(venue_df) > args.sample_venues:
        random.seed(args.seed)
        sampled_ids = set(random.sample(venue_df["id"].tolist(), args.sample_venues))
        venue_df = venue_df[venue_df["id"].isin(sampled_ids)].copy()

    median_gentags = int(gentag_retention.groupby("venue_id")["n_unique_norm_eval"].median().median())
    k = min(median_gentags, DEFAULT_K)

    venue_keywords: Dict[str, Dict[str, List[str]]] = {}
    for _, row in venue_df.iterrows():
        reviews = extract_review_texts(row.get("google_reviews", ""))
        if not reviews:
            continue
        full_text = concatenate_reviews(reviews)
        if not full_text.strip():
            continue
        venue_keywords[row["id"]] = {
            "tfidf": extract_tfidf_keywords(full_text, k=k, max_words=MAX_PHRASE_WORDS),
            "rake": extract_rake_keywords(full_text, k=k, max_words=MAX_PHRASE_WORDS),
            "yake": extract_yake_keywords(full_text, k=k, max_words=MAX_PHRASE_WORDS),
        }

    all_keywords = sorted(
        {
            kw
            for methods in venue_keywords.values()
            for keywords in methods.values()
            for kw in keywords
            if kw
        }
    )

    client = get_embedding_client()
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    anchor_embs = embed_texts_batch(client, anchor_texts, batch_size=16)
    anchor_embeddings = {facet: emb for facet, emb in zip(FACETS, anchor_embs)}

    keyword_embeddings_list = embed_texts_batch(client, all_keywords, batch_size=256)
    keyword_embeddings = {kw: emb for kw, emb in zip(all_keywords, keyword_embeddings_list)}

    rows = []
    for venue_id, methods in venue_keywords.items():
        for method, keywords in methods.items():
            for kw in keywords:
                emb = keyword_embeddings.get(kw)
                if emb is None:
                    continue
                sims = [cosine_similarity(emb, anchor_embeddings[f]) for f in FACETS]
                sims_sorted = sorted(enumerate(sims), key=lambda x: -x[1])
                primary_idx, primary_sim = sims_sorted[0]
                secondary_idx, secondary_sim = sims_sorted[1]
                rows.append(
                    {
                        "venue_id": venue_id,
                        "method": method,
                        "keyword": kw,
                        "primary_facet": FACETS[primary_idx],
                        "secondary_facet": FACETS[secondary_idx],
                        "primary_sim": float(primary_sim),
                        "secondary_sim": float(secondary_sim),
                        "gap": float(primary_sim - secondary_sim),
                    }
                )

    summary = {
        "n_venues": int(len(venue_keywords)),
        "methods": {},
    }
    rows_df = pd.DataFrame(rows)
    for method in ["rake", "yake", "tfidf"]:
        method_rows = rows_df[rows_df["method"] == method].to_dict("records")
        summary["methods"][method] = summarize_gaps(method_rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows_df.to_csv(out_csv, index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
