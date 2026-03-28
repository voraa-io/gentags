"""
Phase 5 — Add YAKE + TF-IDF baselines to sampled_venues.json.

Reads existing sampled_venues.json, extracts YAKE and TF-IDF keywords
from each venue's review text, and writes them back.

Usage:
    poetry run python scripts/phase5_add_baselines.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from phase4_sample_venue import load_venue_reviews, extract_tfidf_keywords, extract_yake_keywords

VENUES_FILE = REPO / "data" / "phase5" / "sampled_venues.json"
DATA_CSV = REPO / "data" / "study1_venues_20250117.csv"


def main():
    print("Loading sampled venues...")
    with open(VENUES_FILE) as f:
        data = json.load(f)

    print("Loading venue reviews from CSV...")
    venues_csv = load_venue_reviews(DATA_CSV)

    total = len(data["venues"])
    for i, v in enumerate(data["venues"]):
        vid = v["venue_id"]
        review_text = venues_csv.get(vid, {}).get("review_text", "")

        yake_kw = extract_yake_keywords(review_text) if review_text else []
        tfidf_kw = extract_tfidf_keywords(review_text) if review_text else []

        v["yake_keywords"] = yake_kw
        v["yake_count"] = len(yake_kw)
        v["tfidf_keywords"] = tfidf_kw
        v["tfidf_count"] = len(tfidf_kw)

        print(f"  [{i+1}/{total}] {v['venue_name']}: "
              f"YAKE={len(yake_kw)}, TF-IDF={len(tfidf_kw)}")

    data["metadata"]["additional_baselines"] = ["yake", "tfidf"]

    with open(VENUES_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Stats
    yc = [v["yake_count"] for v in data["venues"]]
    tc = [v["tfidf_count"] for v in data["venues"]]
    print(f"\nDone. Saved {total} venues to {VENUES_FILE}")
    print(f"YAKE:   min={min(yc)}, max={max(yc)}, mean={sum(yc)/len(yc):.1f}")
    print(f"TF-IDF: min={min(tc)}, max={max(tc)}, mean={sum(tc)/len(tc):.1f}")


if __name__ == "__main__":
    main()
