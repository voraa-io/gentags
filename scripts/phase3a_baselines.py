#!/usr/bin/env python3
"""
Phase 3A: Classical Baseline Comparison (v2 - State-Gini)

Compares gentags against classical keyword extraction methods:
- TF-IDF (statistical)
- RAKE (Rapid Automatic Keyword Extraction)
- YAKE (Yet Another Keyword Extractor)

Metrics:
- State-Gini: Gini on facet COUNTS (using hard assignment)
- Retention: cosine(embed(reviews), embed(representation_text))

Methodology (same as phase3_analysis_v2.py):
1. For each keyword, embed it
2. Compute cosine similarity to each facet anchor
3. Assign to argmax facet (if >= threshold)
4. Count keywords per facet
5. Gini on integer counts

This ensures fair comparison: same metric for gentags and baselines.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import ast
from tqdm import tqdm
import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
EMBEDDING_MODEL_SLUG = EMBEDDING_MODEL.replace("-", "_").replace(".", "_")

# Cache from Phase 2
CACHE_DIR = Path("results/phase2_cache")
REVIEW_EMBEDDINGS_NPZ = CACHE_DIR / f"review_embeddings_{EMBEDDING_MODEL_SLUG}.npz"
REVIEW_EMBEDDINGS_MAP = CACHE_DIR / f"review_embeddings_{EMBEDDING_MODEL_SLUG}.map.json"

# Output directories
OUTPUT_DIR = Path("results/phase3a")
PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"

# Baseline parameters
DEFAULT_K = 25  # Number of keywords/phrases to extract (match median gentags)
MAX_PHRASE_WORDS = 4  # Match gentag constraint

# Semantic threshold (same as Phase 3 v2)
SEMANTIC_THRESHOLD = 0.35

# =============================================================================
# FACET DEFINITIONS (10 semantic dimensions - same as Phase 3 v2)
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
    "location"
]

# Fixed facet anchors (method-neutral, no circular reasoning)
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
# CORE FUNCTIONS (same as Phase 3 v2)
# =============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of a distribution.

    - High Gini (→1): Values concentrated (LOCALIZED)
    - Low Gini (→0): Values spread evenly (DIFFUSE)
    """
    values = np.abs(values).astype(float)
    if values.sum() == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)

    # Gini formula
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return max(0.0, gini)


def hard_assign_facet(
    keyword_emb: np.ndarray,
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> Tuple[Optional[str], float]:
    """
    Hard assignment: assign keyword to exactly ONE facet via argmax.

    Returns: (facet_name or None if below threshold, similarity score)
    """
    best_facet = None
    best_sim = -1.0

    for facet in FACETS:
        sim = cosine_similarity(keyword_emb, anchor_embeddings[facet])
        if sim > best_sim:
            best_sim = sim
            best_facet = facet

    if best_sim >= threshold:
        return best_facet, best_sim
    else:
        return None, best_sim  # Below threshold → "other"


def compute_state_gini(
    keyword_embeddings: List[np.ndarray],
    anchor_embeddings: Dict[str, np.ndarray],
    threshold: float = SEMANTIC_THRESHOLD
) -> Tuple[float, Dict[str, int], int]:
    """
    Compute STATE-GINI: Gini coefficient on facet counts.

    This is the CORRECT metric for comparison with gentags.
    """
    counts = {facet: 0 for facet in FACETS}
    other_count = 0

    for kw_emb in keyword_embeddings:
        facet, sim = hard_assign_facet(kw_emb, anchor_embeddings, threshold)
        if facet is not None:
            counts[facet] += 1
        else:
            other_count += 1

    # Gini on raw integer counts
    count_array = np.array([counts[f] for f in FACETS])
    state_gini = gini_coefficient(count_array)

    return state_gini, counts, other_count


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_review_texts(reviews_data) -> List[str]:
    """Extract review text from the google_reviews column."""
    if pd.isna(reviews_data):
        return []

    if isinstance(reviews_data, str):
        try:
            reviews_list = ast.literal_eval(reviews_data)
        except:
            return []
    else:
        reviews_list = reviews_data

    texts = []
    if isinstance(reviews_list, list):
        for review in reviews_list:
            if isinstance(review, dict) and 'text' in review:
                text = review.get('text', '')
                if text and isinstance(text, str):
                    texts.append(text)
    return texts


def concatenate_reviews(reviews: List[str]) -> str:
    """Concatenate reviews into a single text."""
    return " ".join(reviews)


# =============================================================================
# KEYWORD EXTRACTION METHODS
# =============================================================================

def extract_tfidf_keywords(text: str, k: int = DEFAULT_K, max_words: int = MAX_PHRASE_WORDS) -> List[str]:
    """Extract top-k keywords using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not text.strip():
        return []

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, max_words),
            stop_words='english',
            max_features=k * 5,
            min_df=1
        )

        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        sorted_indices = np.argsort(scores)[::-1]
        keywords = []
        for idx in sorted_indices:
            if len(keywords) >= k:
                break
            phrase = feature_names[idx]
            if len(phrase.split()) <= max_words:
                keywords.append(phrase)

        return keywords[:k]
    except Exception:
        return []


def extract_rake_keywords(text: str, k: int = DEFAULT_K, max_words: int = MAX_PHRASE_WORDS) -> List[str]:
    """Extract top-k keywords using RAKE."""
    from rake_nltk import Rake

    if not text.strip():
        return []

    try:
        rake = Rake(
            min_length=1,
            max_length=max_words,
            include_repeated_phrases=False
        )
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()

        filtered = [p for p in phrases if len(p.split()) <= max_words]
        return filtered[:k]
    except Exception:
        return []


def extract_yake_keywords(text: str, k: int = DEFAULT_K, max_words: int = MAX_PHRASE_WORDS) -> List[str]:
    """Extract top-k keywords using YAKE."""
    import yake

    if not text.strip():
        return []

    try:
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=max_words,
            dedupLim=0.7,
            top=k * 2,
            features=None
        )
        keywords = kw_extractor.extract_keywords(text)

        filtered = [kw[0] for kw in keywords if len(kw[0].split()) <= max_words]
        return filtered[:k]
    except Exception:
        return []


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_embedding_client():
    """Initialize OpenAI embedding client."""
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
        batch = texts[i:i + batch_size]

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


def load_review_embeddings() -> Dict[str, np.ndarray]:
    """Load cached review embeddings from Phase 2."""
    if not REVIEW_EMBEDDINGS_NPZ.exists():
        raise FileNotFoundError(f"Review embeddings not found: {REVIEW_EMBEDDINGS_NPZ}")

    with open(REVIEW_EMBEDDINGS_MAP) as f:
        embedding_map = json.load(f)

    embeddings_data = np.load(REVIEW_EMBEDDINGS_NPZ)

    result = {}
    for venue_id, indices in embedding_map.items():
        if isinstance(indices, list) and len(indices) > 0:
            review_embs = []
            for idx in indices:
                key = f"emb_{idx}"
                if key in embeddings_data:
                    review_embs.append(embeddings_data[key])

            if review_embs:
                venue_emb = np.mean(review_embs, axis=0)
                norm = np.linalg.norm(venue_emb)
                if norm > 0:
                    venue_emb = venue_emb / norm
                result[venue_id] = venue_emb

    return result


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Run Phase 3A baseline comparison with State-Gini methodology."""
    import nltk

    # Download NLTK data for RAKE
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    print("=" * 60)
    print("Phase 3A: Classical Baseline Comparison (v2 - State-Gini)")
    print("=" * 60)
    print("\nThis version uses STATE-GINI (same methodology as gentags)")
    print("- Hard assignment: each keyword → argmax facet")
    print("- Gini computed on integer counts per facet")
    print()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    # Load data
    print("[1] Loading data...")
    venue_df = pd.read_csv("data/study1_venues_20250117.csv")
    print(f"    Loaded {len(venue_df)} venues")

    # Load gentag retention data
    gentag_retention = pd.read_csv("results/phase2/tables/retention.csv")
    quality_venues = gentag_retention['venue_id'].unique()
    print(f"    Quality-filtered venues: {len(quality_venues)}")

    venue_df = venue_df[venue_df['id'].isin(quality_venues)]
    print(f"    Venues after filtering: {len(venue_df)}")

    # Load cached review embeddings
    print("\n[2] Loading review embeddings...")
    review_embeddings = load_review_embeddings()
    print(f"    Loaded embeddings for {len(review_embeddings)} venues")

    # Get median gentag count to match budget
    median_gentags = int(gentag_retention.groupby('venue_id')['n_unique_norm_eval'].median().median())
    k = min(median_gentags, DEFAULT_K)
    print(f"\n[3] Using k={k} keywords (matching median gentags)")

    # Initialize embedding client
    print("\n[4] Initializing embedding client...")
    client = get_embedding_client()

    # Compute anchor embeddings
    print("\n[5] Computing anchor embeddings...")
    anchor_texts = [FACET_ANCHORS[f] for f in FACETS]
    anchor_embs = embed_texts_batch(client, anchor_texts, batch_size=16)
    anchor_embeddings = {facet: emb for facet, emb in zip(FACETS, anchor_embs)}
    print(f"    Computed {len(anchor_embeddings)} facet anchors")

    # Extract keywords for all venues
    print("\n[6] Extracting keywords...")
    venue_keywords = {}

    for _, row in tqdm(venue_df.iterrows(), total=len(venue_df), desc="Extracting keywords"):
        venue_id = row['id']

        if venue_id not in review_embeddings:
            continue

        reviews = extract_review_texts(row.get('google_reviews', ''))
        if not reviews:
            continue

        full_text = concatenate_reviews(reviews)
        if not full_text.strip():
            continue

        venue_keywords[venue_id] = {
            'tfidf': extract_tfidf_keywords(full_text, k=k),
            'rake': extract_rake_keywords(full_text, k=k),
            'yake': extract_yake_keywords(full_text, k=k),
        }

    print(f"    Extracted keywords for {len(venue_keywords)} venues")

    # Collect all unique keywords for embedding
    print("\n[7] Collecting unique keywords for embedding...")
    all_keywords = set()
    for venue_id, methods in venue_keywords.items():
        for method, keywords in methods.items():
            all_keywords.update(keywords)

    all_keywords = list(all_keywords)
    print(f"    Found {len(all_keywords)} unique keywords")

    # Embed all keywords
    print("\n[8] Embedding all keywords...")
    keyword_embeddings_list = embed_texts_batch(client, all_keywords, batch_size=256)
    keyword_embeddings = {kw: emb for kw, emb in zip(all_keywords, keyword_embeddings_list)}

    # Compute State-Gini for each venue/method
    print("\n[9] Computing State-Gini for each venue/method...")
    results = []

    for venue_id, methods in tqdm(venue_keywords.items(), desc="Computing State-Gini"):
        for method, keywords in methods.items():
            if not keywords:
                continue

            # Get embeddings for this venue's keywords
            kw_embs = [keyword_embeddings[kw] for kw in keywords if kw in keyword_embeddings]

            if not kw_embs:
                continue

            # Compute State-Gini
            state_gini, counts, other_count = compute_state_gini(kw_embs, anchor_embeddings)

            # Compute retention (for secondary analysis)
            kw_text = " ".join(keywords)
            if venue_id in review_embeddings:
                kw_agg_emb = embed_texts_batch(client, [kw_text], batch_size=1)[0]
                retention = cosine_similarity(review_embeddings[venue_id], kw_agg_emb)
            else:
                retention = None

            result = {
                'venue_id': venue_id,
                'method': method,
                'n_keywords': len(keywords),
                'state_gini': state_gini,
                'assigned_count': len(keywords) - other_count,
                'other_count': other_count,
                'retention_cosine': retention,
            }

            # Add per-facet counts
            for facet in FACETS:
                result[f'count_{facet}'] = counts.get(facet, 0)

            results.append(result)

    baseline_df = pd.DataFrame(results)

    # Load gentag State-Gini from Phase 3 v2
    print("\n[10] Loading gentag State-Gini from Phase 3...")
    phase3_state_path = Path("results/phase3/tables/state_localization.csv")
    if phase3_state_path.exists():
        phase3_df = pd.read_csv(phase3_state_path)
        gentag_state_gini_mean = phase3_df['gentag_state_gini'].mean()
        gentag_state_gini_std = phase3_df['gentag_state_gini'].std()
        print(f"    Gentag State-Gini: {gentag_state_gini_mean:.3f} +/- {gentag_state_gini_std:.3f}")
    else:
        print("    Phase 3 results not found. Run phase3_analysis_v2.py first.")
        gentag_state_gini_mean = None
        gentag_state_gini_std = None

    # Save results
    baseline_df.to_csv(TABLES_DIR / "baseline_state_gini.csv", index=False)
    print(f"\n    Saved {len(baseline_df)} rows to baseline_state_gini.csv")

    # Summary
    print("\n" + "=" * 60)
    print("STATE-GINI RESULTS")
    print("=" * 60)

    summary = baseline_df.groupby('method').agg({
        'state_gini': ['mean', 'std', 'median'],
        'n_keywords': 'mean',
        'assigned_count': 'mean',
        'other_count': 'mean',
    }).round(4)

    print("\n" + str(summary))

    # Add gentags to summary
    if gentag_state_gini_mean is not None:
        print(f"\n   Gentags State-Gini: {gentag_state_gini_mean:.3f} +/- {gentag_state_gini_std:.3f}")

        # Compute comparison
        print("\n" + "-" * 40)
        print("COMPARISON TO GENTAGS")
        print("-" * 40)

        for method in ['tfidf', 'rake', 'yake']:
            method_mean = baseline_df[baseline_df['method'] == method]['state_gini'].mean()
            diff = gentag_state_gini_mean - method_mean
            ratio = gentag_state_gini_mean / method_mean if method_mean > 0 else float('inf')
            print(f"    Gentags vs {method.upper()}: {diff:+.3f} ({ratio:.2f}x)")

    # Save summary
    summary_data = []
    for method in ['tfidf', 'rake', 'yake']:
        method_data = baseline_df[baseline_df['method'] == method]
        summary_data.append({
            'method': method,
            'state_gini_mean': method_data['state_gini'].mean(),
            'state_gini_std': method_data['state_gini'].std(),
            'n_keywords_mean': method_data['n_keywords'].mean(),
        })

    if gentag_state_gini_mean is not None:
        summary_data.append({
            'method': 'gentags',
            'state_gini_mean': gentag_state_gini_mean,
            'state_gini_std': gentag_state_gini_std,
            'n_keywords_mean': None,
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(TABLES_DIR / "state_gini_summary.csv", index=False)

    # Write manifest
    manifest = {
        "phase": "phase3a_v2",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "methodology": "STATE-GINI (Gini on counts, hard assignment)",
        "threshold": SEMANTIC_THRESHOLD,
        "k_keywords": k,
        "counts": {
            "n_venues": len(venue_keywords),
            "n_unique_keywords": len(all_keywords),
            "n_results": len(baseline_df),
        },
        "results": {
            "tfidf_state_gini_mean": float(baseline_df[baseline_df['method'] == 'tfidf']['state_gini'].mean()),
            "rake_state_gini_mean": float(baseline_df[baseline_df['method'] == 'rake']['state_gini'].mean()),
            "yake_state_gini_mean": float(baseline_df[baseline_df['method'] == 'yake']['state_gini'].mean()),
            "gentag_state_gini_mean": float(gentag_state_gini_mean) if gentag_state_gini_mean else None,
        },
        "facets": FACETS,
    }

    with open(OUTPUT_DIR / "phase3a_v2_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("Phase 3A v2 Complete")
    print("=" * 60)

    return baseline_df


if __name__ == "__main__":
    baseline_df = main()
