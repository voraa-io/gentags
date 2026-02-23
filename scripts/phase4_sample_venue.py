"""
Phase 4 — Extract sample venue data for DIR/INV test design.

Finds a sparse venue with 8-15 gentags, extracts:
  1. Gentag lists (Run 1 + Run 2) from Phase 1
  2. Baseline keywords (RAKE, TF-IDF, YAKE) for the same venue

Outputs JSON to results/phase4/sample_venue.json

Usage:
    poetry run python scripts/phase4_sample_venue.py
    poetry run python scripts/phase4_sample_venue.py --venue-id MFJDDz0Mgf5LOkmbvW8b
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_CSV = REPO / "data" / "study1_venues_20250117.csv"
TAGS_DIR = REPO / "results" / "phase1_downloaded"
TAGS_PATTERN = "week2_run_20251223_191104_tags_{model}.csv"
EXTRACTIONS_PATTERN = "week2_run_20251223_191104_extractions_{model}.csv"
OUTPUT_DIR = REPO / "results" / "phase4"
OUTPUT_FILE = OUTPUT_DIR / "sample_venue.json"

MODELS = ["openai", "gemini", "claude", "grok"]
DEFAULT_MODEL = "openai"
DEFAULT_PROMPT = "minimal"

# ---------------------------------------------------------------------------
# Baseline extraction (RAKE, TF-IDF, YAKE)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO / "scripts"))

DEFAULT_K = 20
MAX_PHRASE_WORDS = 4


def extract_tfidf_keywords(text, k=DEFAULT_K, max_words=MAX_PHRASE_WORDS):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vectorizer = TfidfVectorizer(
        ngram_range=(1, max_words),
        max_features=500,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform([text])
    names = vectorizer.get_feature_names_out()
    scores = tfidf.toarray().flatten()
    ranked = np.argsort(scores)[::-1]
    keywords = []
    for idx in ranked:
        phrase = names[idx]
        if len(phrase.split()) <= max_words:
            if len(keywords) >= k:
                break
            keywords.append(phrase)
    return keywords[:k]


def extract_rake_keywords(text, k=DEFAULT_K, max_words=MAX_PHRASE_WORDS):
    from rake_nltk import Rake

    rake = Rake(max_length=max_words)
    rake.extract_keywords_from_text(text)
    ranked = rake.get_ranked_phrases()
    return [p for p in ranked if len(p.split()) <= max_words][:k]


def extract_yake_keywords(text, k=DEFAULT_K, max_words=MAX_PHRASE_WORDS):
    import yake

    kw_extractor = yake.KeywordExtractor(
        lan="en", n=max_words, top=k * 2, dedupLim=0.7
    )
    keywords = kw_extractor.extract_keywords(text)
    filtered = [kw[0] for kw in keywords if len(kw[0].split()) <= max_words]
    return filtered[:k]


# ---------------------------------------------------------------------------
# Venue discovery
# ---------------------------------------------------------------------------

def load_venue_reviews(csv_path):
    """Load venue data, return dict of venue_id -> {name, review_text}.

    Review text comes from the google_reviews column (list of dicts with 'text' key),
    matching the Phase 3A baseline pipeline.
    """
    import ast
    import pandas as pd

    df = pd.read_csv(csv_path)
    venues = {}
    for _, row in df.iterrows():
        vid = row["id"]
        name = row.get("name", "")

        # Extract review text from google_reviews (same as phase3a_baselines.py)
        reviews_data = row.get("google_reviews", "")
        texts = []
        if pd.notna(reviews_data) and isinstance(reviews_data, str):
            try:
                reviews_list = ast.literal_eval(reviews_data)
                if isinstance(reviews_list, list):
                    for review in reviews_list:
                        if isinstance(review, dict) and "text" in review:
                            text = review.get("text", "")
                            if text and isinstance(text, str):
                                texts.append(text)
            except (ValueError, SyntaxError):
                pass

        review_text = " ".join(texts)
        venues[vid] = {"name": name, "review_text": review_text, "reviews": texts}
    return venues


def get_extraction_stats():
    """Return per-venue stats: total tags, avg tags, min input tokens."""
    stats = defaultdict(lambda: {"total_tags": 0, "extractions": 0, "min_tokens": 9999})
    for model in MODELS:
        path = TAGS_DIR / EXTRACTIONS_PATTERN.format(model=model)
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["status"] != "success":
                    continue
                vid = row["venue_id"]
                stats[vid]["total_tags"] += int(row["n_tags"])
                stats[vid]["extractions"] += 1
                try:
                    tok = int(float(row["input_tokens"]))
                    stats[vid]["min_tokens"] = min(stats[vid]["min_tokens"], tok)
                except (ValueError, KeyError):
                    pass
    return stats


def find_sparse_venue(min_avg_tags=8, max_avg_tags=20, max_tokens=250):
    """Find best sparse venue: low tokens, 8-15 avg tags."""
    stats = get_extraction_stats()
    candidates = []
    for vid, s in stats.items():
        if s["extractions"] == 0:
            continue
        avg = s["total_tags"] / s["extractions"]
        if min_avg_tags <= avg <= max_avg_tags and s["min_tokens"] <= max_tokens:
            candidates.append((vid, s["min_tokens"], avg))
    candidates.sort(key=lambda x: x[1])  # sort by sparsity (lowest tokens first)
    return candidates


def get_gentags(venue_id, model=DEFAULT_MODEL, prompt=DEFAULT_PROMPT):
    """Get gentags for a venue from Phase 1 tags CSV. Returns {run1: [...], run2: [...]}."""
    path = TAGS_DIR / TAGS_PATTERN.format(model=model)
    runs = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["venue_id"] == venue_id and row["prompt_type"] == prompt:
                runs[f"run{row['run_number']}"].append(row["tag_norm_eval"])
    return dict(runs)


def get_all_gentags(venue_id):
    """Get gentags for all models/prompts/runs."""
    all_tags = {}
    for model in MODELS:
        path = TAGS_DIR / TAGS_PATTERN.format(model=model)
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["venue_id"] != venue_id:
                    continue
                key = f"{model}_{row['prompt_type']}_run{row['run_number']}"
                all_tags.setdefault(key, []).append(row["tag_norm_eval"])
    return all_tags


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract sample venue data for Phase 4")
    parser.add_argument("--venue-id", default=None, help="Specific venue ID (default: auto-select sparse)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for primary gentag sample")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt type for primary gentag sample")
    parser.add_argument("--list-candidates", action="store_true", help="List sparse venue candidates and exit")
    args = parser.parse_args()

    # List candidates
    if args.list_candidates:
        candidates = find_sparse_venue()
        venues = load_venue_reviews(DATA_CSV)
        print(f"{'venue_id':>26s} | min_tok | avg_tags | name")
        for vid, tok, avg in candidates[:20]:
            name = venues.get(vid, {}).get("name", "?")
            print(f"{vid:>26s} | {tok:>7d} | {avg:>8.1f} | {name}")
        return

    # Select venue
    if args.venue_id:
        venue_id = args.venue_id
    else:
        candidates = find_sparse_venue()
        if not candidates:
            print("ERROR: No sparse venues found matching criteria")
            sys.exit(1)
        venue_id = candidates[0][0]
        print(f"Auto-selected sparse venue: {venue_id}")

    venues = load_venue_reviews(DATA_CSV)
    venue_info = venues.get(venue_id)
    if not venue_info:
        print(f"ERROR: Venue {venue_id} not found in data")
        sys.exit(1)

    print(f"Venue: {venue_info['name']} ({venue_id})")

    # --- Gentags (all 4 models, selected prompt, run1 + run2) ---
    gentags_by_model = {}
    for model in MODELS:
        gentags_by_model[model] = get_gentags(venue_id, model=model, prompt=args.prompt)

    print(f"\nGentags ({args.prompt} prompt, all models):")
    for model in MODELS:
        runs = gentags_by_model[model]
        for run_key in sorted(runs):
            print(f"  {model:7s} {run_key}: {len(runs[run_key])} tags")

    # --- Baselines ---
    review_text = venue_info["review_text"]
    print(f"\nReview text: {len(review_text)} chars")

    rake_kw = extract_rake_keywords(review_text)
    tfidf_kw = extract_tfidf_keywords(review_text)
    yake_kw = extract_yake_keywords(review_text)

    print(f"RAKE:  {len(rake_kw)} keywords")
    print(f"TF-IDF: {len(tfidf_kw)} keywords")
    print(f"YAKE:  {len(yake_kw)} keywords")

    # --- Stats ---
    stats = get_extraction_stats()
    venue_stats = stats.get(venue_id, {})

    # --- Build output ---
    gentags_section = {}
    for model in MODELS:
        runs = gentags_by_model[model]
        gentags_section[model] = {
            "run1": sorted(set(runs.get("run1", []))),
            "run2": sorted(set(runs.get("run2", []))),
        }

    output = {
        "venue_id": venue_id,
        "venue_name": venue_info["name"],
        "prompt": args.prompt,
        "reviews": venue_info["reviews"],
        "sparsity": {
            "min_input_tokens": venue_stats.get("min_tokens"),
            "total_extractions": venue_stats.get("extractions"),
            "avg_tags_per_extraction": round(
                venue_stats["total_tags"] / venue_stats["extractions"], 1
            ) if venue_stats.get("extractions") else None,
        },
        "gentags": gentags_section,
        "baselines": {
            "rake": sorted(set(rake_kw)),
            "tfidf": sorted(set(tfidf_kw)),
            "yake": sorted(set(yake_kw)),
        },
    }

    # --- Write ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {OUTPUT_FILE}")

    # --- Print summary for quick inspection ---
    print(f"\n{'='*60}")
    for model in MODELS:
        for run_key in ["run1", "run2"]:
            tags = gentags_section[model][run_key]
            print(f"\n{model.upper()} {run_key} ({len(tags)} tags):")
            print(json.dumps(tags, indent=2))
    print(f"\nRAKE ({len(output['baselines']['rake'])} keywords):")
    print(json.dumps(output["baselines"]["rake"], indent=2))
    print(f"\nTF-IDF ({len(output['baselines']['tfidf'])} keywords):")
    print(json.dumps(output["baselines"]["tfidf"], indent=2))
    print(f"\nYAKE ({len(output['baselines']['yake'])} keywords):")
    print(json.dumps(output["baselines"]["yake"], indent=2))


if __name__ == "__main__":
    main()
