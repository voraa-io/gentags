"""
Phase 5 — Sample 50 venues for Baseline Legibility Study.

Loads all 553 venues with openai minimal run1 gentags + review text.
Stratifies sample:
  - ~6 venues WITH game-viewing gentags (positive P2 examples)
  - ~15 venues WITH speed/service gentags (positive P3 examples)
  - ~29 venues WITHOUT either (negative examples for P2/P3 compliance)

For each venue extracts:
  - Gentags (openai minimal run1)
  - RAKE keywords
  - Review text (individual reviews)
  - Tag presence flags: has_speed_tag, has_game_tag
  - Truncated gentags (drop tags until count <= RAKE count)

Output: data/phase5/sampled_venues.json

Usage:
    poetry run python scripts/phase5_sample_venues.py
    poetry run python scripts/phase5_sample_venues.py --n 50 --seed 42
    poetry run python scripts/phase5_sample_venues.py --list-strata
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_CSV = REPO / "data" / "study1_venues_20250117.csv"
TAGS_DIR = REPO / "results" / "phase1_downloaded"
TAGS_FILE = TAGS_DIR / "week2_run_20251223_191104_tags_openai.csv"
OUTPUT_DIR = REPO / "data" / "phase5"
OUTPUT_FILE = OUTPUT_DIR / "sampled_venues.json"

# ---------------------------------------------------------------------------
# Reused from phase4_sample_venue.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(REPO / "scripts"))
from phase4_sample_venue import load_venue_reviews, extract_rake_keywords

# ---------------------------------------------------------------------------
# Tag presence detection
# ---------------------------------------------------------------------------
# Game/sports-viewing keywords (P2 hard requirement)
GAME_PATTERNS = [
    r"\bgame\b", r"\bgames\b", r"\bsports?\s*viewing\b",
    r"\bwatch(?:ing)?\s*(?:games?|sports?|matches?|live)\b",
    r"\blive\s*(?:sports?|games?|matches?|events?)\b",
    r"\bscreens?\b", r"\btv\s*screens?\b", r"\bprojector\b",
    r"\bsports?\s*bar\b", r"\bsports?\s*screen\b",
]

# Speed/service keywords (P3 hard requirement)
SPEED_PATTERNS = [
    r"\bfast\s*service\b", r"\bquick\s*service\b", r"\bspeedy\b",
    r"\brapid\s*service\b", r"\befficient\s*service\b",
    r"\bfast\s*food\b", r"\bquick\s*(?:lunch|meal|bite)\b",
    r"\bprompt\s*service\b", r"\bswift\s*service\b",
    r"\bshort\s*wait\b", r"\bno\s*wait\b", r"\bminimal\s*wait\b",
]


def has_pattern_match(tags: list, patterns: list) -> bool:
    """Check if any tag matches any pattern."""
    for tag in tags:
        tag_lower = tag.lower()
        for pattern in patterns:
            if re.search(pattern, tag_lower):
                return True
    return False


def get_all_venue_gentags() -> dict:
    """Load openai minimal run1 gentags for all venues."""
    venues = defaultdict(list)
    with open(TAGS_FILE) as f:
        for row in csv.DictReader(f):
            if row["prompt_type"] == "minimal" and row["run_number"] == "1":
                venues[row["venue_id"]].append(row["tag_norm_eval"])
    # Deduplicate and sort
    return {vid: sorted(set(tags)) for vid, tags in venues.items()}


def truncate_gentags(gentags: list, max_count: int) -> list:
    """Truncate gentags to max_count by dropping from end of sorted list."""
    if len(gentags) <= max_count:
        return gentags
    return gentags[:max_count]


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------
def classify_venue(gentags: list) -> str:
    """Classify venue into stratum based on tag presence."""
    has_game = has_pattern_match(gentags, GAME_PATTERNS)
    has_speed = has_pattern_match(gentags, SPEED_PATTERNS)
    if has_game:
        return "has_game"
    elif has_speed:
        return "has_speed"
    else:
        return "neither"


def stratified_sample(venue_strata: dict, n: int, seed: int) -> list:
    """
    Stratified sample targeting:
      - ~12% from has_game (6 venues for n=50)
      - ~30% from has_speed (15 venues for n=50)
      - ~58% from neither (29 venues for n=50)
    Falls back proportionally if strata are too small.
    """
    rng = random.Random(seed)

    targets = {
        "has_game": max(1, round(n * 0.12)),
        "has_speed": max(1, round(n * 0.30)),
        "neither": 0,  # Fill remainder
    }
    targets["neither"] = n - targets["has_game"] - targets["has_speed"]

    sampled = []
    for stratum, target in targets.items():
        pool = venue_strata.get(stratum, [])
        actual = min(target, len(pool))
        sampled.extend(rng.sample(pool, actual))

    # If under target due to small strata, fill from "neither"
    if len(sampled) < n:
        remaining_neither = [v for v in venue_strata.get("neither", [])
                             if v not in sampled]
        shortfall = n - len(sampled)
        sampled.extend(rng.sample(remaining_neither, min(shortfall, len(remaining_neither))))

    return sampled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Sample 50 venues")
    parser.add_argument("--n", type=int, default=50, help="Number of venues to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--list-strata", action="store_true",
                        help="List strata counts and exit")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    print("Loading gentags...")
    all_gentags = get_all_venue_gentags()
    print(f"  {len(all_gentags)} venues with openai minimal run1 tags")

    print("Loading venue reviews...")
    venue_reviews = load_venue_reviews(DATA_CSV)
    print(f"  {len(venue_reviews)} venues in CSV")

    # Only include venues that have BOTH gentags AND reviews
    valid_venues = [vid for vid in all_gentags
                    if vid in venue_reviews
                    and len(venue_reviews[vid].get("reviews", [])) > 0]
    print(f"  {len(valid_venues)} venues with both gentags and reviews")

    # Classify into strata
    venue_strata = defaultdict(list)
    for vid in valid_venues:
        stratum = classify_venue(all_gentags[vid])
        venue_strata[stratum].append(vid)

    print(f"\nStrata:")
    for stratum in ["has_game", "has_speed", "neither"]:
        print(f"  {stratum}: {len(venue_strata[stratum])} venues")

    if args.list_strata:
        # Show example tags for each stratum
        for stratum in ["has_game", "has_speed"]:
            print(f"\n--- {stratum} examples ---")
            for vid in venue_strata[stratum][:5]:
                name = venue_reviews.get(vid, {}).get("name", "?")
                tags = all_gentags[vid]
                matching = [t for t in tags
                            if has_pattern_match([t], GAME_PATTERNS if stratum == "has_game"
                                                 else SPEED_PATTERNS)]
                print(f"  {vid}: {name}")
                print(f"    Matching tags: {matching}")
        return

    # Sample
    sampled_ids = stratified_sample(venue_strata, args.n, args.seed)
    print(f"\nSampled {len(sampled_ids)} venues")

    # Build output
    venues_out = []
    for vid in sampled_ids:
        gentags = all_gentags[vid]
        vinfo = venue_reviews[vid]
        review_text = vinfo["review_text"]

        # RAKE keywords
        rake_kw = extract_rake_keywords(review_text) if review_text else []

        # Truncated gentags (match RAKE count)
        rake_count = len(rake_kw)
        gentags_truncated = truncate_gentags(gentags, rake_count)

        # Tag presence flags
        has_game = has_pattern_match(gentags, GAME_PATTERNS)
        has_speed = has_pattern_match(gentags, SPEED_PATTERNS)
        stratum = classify_venue(gentags)

        venues_out.append({
            "venue_id": vid,
            "venue_name": vinfo.get("name", ""),
            "stratum": stratum,
            "has_game_tag": has_game,
            "has_speed_tag": has_speed,
            "gentags": gentags,
            "gentag_count": len(gentags),
            "rake_keywords": rake_kw,
            "rake_count": len(rake_kw),
            "gentags_truncated": gentags_truncated,
            "gentags_truncated_count": len(gentags_truncated),
            "reviews": vinfo.get("reviews", []),
            "review_count": len(vinfo.get("reviews", [])),
        })

    # Summary stats
    strata_counts = defaultdict(int)
    for v in venues_out:
        strata_counts[v["stratum"]] += 1

    output = {
        "metadata": {
            "n_venues": len(venues_out),
            "seed": args.seed,
            "total_venues_available": len(valid_venues),
            "strata_counts": dict(strata_counts),
            "gentag_source": "openai_minimal_run1",
            "baseline_source": "rake",
        },
        "venues": venues_out,
    }

    # Write
    out_path = Path(args.output) if args.output else OUTPUT_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nOutput: {out_path}")
    print(f"\nStrata distribution:")
    for stratum in ["has_game", "has_speed", "neither"]:
        print(f"  {stratum}: {strata_counts.get(stratum, 0)} venues")

    print(f"\nTag count stats:")
    gc = [v["gentag_count"] for v in venues_out]
    rc = [v["rake_count"] for v in venues_out]
    tc = [v["gentags_truncated_count"] for v in venues_out]
    print(f"  Gentags: min={min(gc)}, max={max(gc)}, mean={sum(gc)/len(gc):.1f}")
    print(f"  RAKE:    min={min(rc)}, max={max(rc)}, mean={sum(rc)/len(rc):.1f}")
    print(f"  Trunc:   min={min(tc)}, max={max(tc)}, mean={sum(tc)/len(tc):.1f}")

    print(f"\nSample venues:")
    for v in venues_out[:5]:
        print(f"  {v['venue_id']}: {v['venue_name']} "
              f"({v['stratum']}, {v['gentag_count']}gt/{v['rake_count']}rk)")


if __name__ == "__main__":
    main()
