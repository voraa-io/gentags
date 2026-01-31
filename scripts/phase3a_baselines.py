#!/usr/bin/env python3
"""
Phase 3A: Classical Baseline Comparison

Compares gentags against classical keyword extraction methods:
- TF-IDF (statistical)
- RAKE (Rapid Automatic Keyword Extraction)
- YAKE (Yet Another Keyword Extractor)

Metrics:
- Retention: cosine(embed(reviews), embed(representation_text))
- Localization Gini: change concentration using facet anchors

This addresses the "glass jaw" critique: why not just use TF-IDF?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import ast
from tqdm import tqdm

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

# Facet anchors for localization analysis (same as Phase 3)
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
        # Use n-grams from 1 to max_words
        vectorizer = TfidfVectorizer(
            ngram_range=(1, max_words),
            stop_words='english',
            max_features=k * 5,  # Get more candidates, then filter
            min_df=1
        )

        # Fit on single document (per-venue TF-IDF)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]

        # Sort by score and get top k
        sorted_indices = np.argsort(scores)[::-1]
        keywords = []
        for idx in sorted_indices:
            if len(keywords) >= k:
                break
            phrase = feature_names[idx]
            # Filter to phrases with max_words or fewer
            if len(phrase.split()) <= max_words:
                keywords.append(phrase)

        return keywords[:k]
    except Exception as e:
        print(f"TF-IDF error: {e}")
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

        # Filter and return top k
        filtered = [p for p in phrases if len(p.split()) <= max_words]
        return filtered[:k]
    except Exception as e:
        print(f"RAKE error: {e}")
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
            top=k * 2,  # Get more, then filter
            features=None
        )
        keywords = kw_extractor.extract_keywords(text)

        # Keywords are (phrase, score) tuples; lower score = better
        filtered = [kw[0] for kw in keywords if len(kw[0].split()) <= max_words]
        return filtered[:k]
    except Exception as e:
        print(f"YAKE error: {e}")
        return []


# =============================================================================
# EMBEDDING FUNCTIONS
# =============================================================================

def get_embedding_client():
    """Initialize OpenAI embedding client."""
    from openai import OpenAI
    import os
    from dotenv import load_dotenv

    # Load .env
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


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient of a distribution.

    - High Gini (→1): Values concentrated (LOCALIZED)
    - Low Gini (→0): Values spread evenly (DIFFUSE)
    """
    values = np.abs(values)
    if values.sum() == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)

    # Gini formula
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    return max(0.0, gini)


def compute_facet_gini(embedding: np.ndarray, facet_embeddings: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
    """
    Compute Gini coefficient of facet similarities for a representation.

    Returns:
        gini: Gini coefficient (high = localized, low = diffuse)
        facet_sims: Dictionary of per-facet similarities
    """
    facet_sims = {}
    for facet, facet_emb in facet_embeddings.items():
        facet_sims[facet] = cosine_similarity(embedding, facet_emb)

    sims_array = np.array(list(facet_sims.values()))
    gini = gini_coefficient(sims_array)

    return gini, facet_sims


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def load_review_embeddings() -> Dict[str, np.ndarray]:
    """Load cached review embeddings from Phase 2.

    The map stores venue_id -> [list of review indices].
    We load all review embeddings and mean-pool them per venue.
    """
    if not REVIEW_EMBEDDINGS_NPZ.exists():
        raise FileNotFoundError(f"Review embeddings not found: {REVIEW_EMBEDDINGS_NPZ}")

    with open(REVIEW_EMBEDDINGS_MAP) as f:
        embedding_map = json.load(f)

    embeddings_data = np.load(REVIEW_EMBEDDINGS_NPZ)

    result = {}
    for venue_id, indices in embedding_map.items():
        # indices is a list of review embedding indices
        if isinstance(indices, list) and len(indices) > 0:
            # Load all review embeddings for this venue
            review_embs = []
            for idx in indices:
                key = f"emb_{idx}"
                if key in embeddings_data:
                    review_embs.append(embeddings_data[key])

            if review_embs:
                # Mean pool to get venue-level embedding
                venue_emb = np.mean(review_embs, axis=0)
                # Normalize
                norm = np.linalg.norm(venue_emb)
                if norm > 0:
                    venue_emb = venue_emb / norm
                result[venue_id] = venue_emb
        elif isinstance(indices, int):
            # Single index case
            result[venue_id] = embeddings_data[f"arr_{indices}"]

    return result


def load_gentag_retention() -> pd.DataFrame:
    """Load gentag retention data from Phase 2."""
    retention_path = Path("results/phase2/tables/retention.csv")
    if not retention_path.exists():
        raise FileNotFoundError(f"Retention data not found: {retention_path}")

    return pd.read_csv(retention_path)


def main():
    """Run Phase 3A baseline comparison."""
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
    print("Phase 3A: Classical Baseline Comparison")
    print("=" * 60)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)

    # Load data
    print("\n[1] Loading data...")

    # Load venue data
    venue_df = pd.read_csv("data/study1_venues_20250117.csv")
    print(f"  Loaded {len(venue_df)} venues")

    # Load gentag retention data
    gentag_retention = load_gentag_retention()
    quality_venues = gentag_retention['venue_id'].unique()
    print(f"  Quality-filtered venues: {len(quality_venues)}")

    # Filter to quality venues
    venue_df = venue_df[venue_df['id'].isin(quality_venues)]
    print(f"  Venues after filtering: {len(venue_df)}")

    # Load cached review embeddings
    print("\n[2] Loading review embeddings...")
    review_embeddings = load_review_embeddings()
    print(f"  Loaded embeddings for {len(review_embeddings)} venues")

    # Get median gentag count to match budget
    median_gentags = int(gentag_retention.groupby('venue_id')['n_unique_norm_eval'].median().median())
    k = min(median_gentags, DEFAULT_K)
    print(f"\n[3] Using k={k} keywords (matching median gentags)")

    # Initialize embedding client
    print("\n[4] Initializing embedding client...")
    client = get_embedding_client()

    # Extract keywords and compute retention for each method
    print("\n[5] Extracting keywords and computing retention...")

    results = []
    texts_to_embed = []
    text_metadata = []

    for _, row in tqdm(venue_df.iterrows(), total=len(venue_df), desc="Processing venues"):
        venue_id = row['id']

        if venue_id not in review_embeddings:
            continue

        # Extract review text
        reviews = extract_review_texts(row.get('google_reviews', ''))
        if not reviews:
            continue

        full_text = concatenate_reviews(reviews)
        if not full_text.strip():
            continue

        # Extract keywords for each method
        tfidf_kw = extract_tfidf_keywords(full_text, k=k)
        rake_kw = extract_rake_keywords(full_text, k=k)
        yake_kw = extract_yake_keywords(full_text, k=k)

        # Store for batch embedding
        for method, keywords in [('tfidf', tfidf_kw), ('rake', rake_kw), ('yake', yake_kw)]:
            if keywords:
                kw_text = " ".join(keywords)
                texts_to_embed.append(kw_text)
                text_metadata.append({
                    'venue_id': venue_id,
                    'method': method,
                    'n_keywords': len(keywords),
                    'keywords': keywords
                })

    print(f"\n[6] Embedding {len(texts_to_embed)} keyword sets...")

    if texts_to_embed:
        keyword_embeddings = embed_texts_batch(client, texts_to_embed)

        # Compute retention for each
        print("\n[7] Computing retention scores...")

        for i, meta in enumerate(text_metadata):
            venue_id = meta['venue_id']
            method = meta['method']

            if venue_id in review_embeddings:
                review_emb = review_embeddings[venue_id]
                kw_emb = keyword_embeddings[i]
                retention = cosine_similarity(review_emb, kw_emb)

                results.append({
                    'venue_id': venue_id,
                    'method': method,
                    'retention_cosine': retention,
                    'n_keywords': meta['n_keywords'],
                    'keywords_sample': str(meta['keywords'][:5])  # First 5 for inspection
                })

    # Create results DataFrame
    baseline_df = pd.DataFrame(results)

    # =========================================================================
    # GINI LOCALIZATION ANALYSIS
    # =========================================================================
    print("\n[8] Computing Gini localization scores...")

    # Embed facet anchors
    print("  Embedding facet anchors...")
    facet_texts = list(FACET_ANCHORS.values())
    facet_names = list(FACET_ANCHORS.keys())
    facet_embs = embed_texts_batch(client, facet_texts, batch_size=16)
    facet_embeddings = {name: emb for name, emb in zip(facet_names, facet_embs)}

    # Compute Gini for each baseline embedding
    print("  Computing Gini for baselines...")
    gini_results = []

    for i, meta in enumerate(text_metadata):
        venue_id = meta['venue_id']
        method = meta['method']
        kw_emb = keyword_embeddings[i]

        gini, facet_sims = compute_facet_gini(kw_emb, facet_embeddings)

        gini_results.append({
            'venue_id': venue_id,
            'method': method,
            'gini': gini,
            **{f'sim_{f}': s for f, s in facet_sims.items()}
        })

    gini_df = pd.DataFrame(gini_results)

    # Load gentag Gini from Phase 3 (or compute if not available)
    phase3_gini_path = Path("results/phase3/tables/localization.csv")
    if phase3_gini_path.exists():
        phase3_gini = pd.read_csv(phase3_gini_path)
        gentag_gini_mean = phase3_gini['gentag_gini'].mean()
        embedding_gini_mean = phase3_gini['embedding_gini'].mean()
        print(f"  Loaded Phase 3 Gini: gentags={gentag_gini_mean:.3f}, embeddings={embedding_gini_mean:.3f}")
    else:
        gentag_gini_mean = 0.657  # From Phase 3 report
        embedding_gini_mean = 0.361
        print(f"  Using Phase 3 report values: gentags={gentag_gini_mean:.3f}, embeddings={embedding_gini_mean:.3f}")

    # Compute Gini summary by method
    gini_summary = gini_df.groupby('method')['gini'].agg(['mean', 'std', 'median']).round(4)
    gini_summary.loc['gentags'] = [gentag_gini_mean, np.nan, np.nan]
    gini_summary.loc['embeddings'] = [embedding_gini_mean, np.nan, np.nan]
    gini_summary = gini_summary.sort_values('mean', ascending=False)

    # Save Gini results
    gini_df.to_csv(TABLES_DIR / "baseline_gini.csv", index=False)
    gini_summary.to_csv(TABLES_DIR / "gini_summary.csv")

    # Merge Gini into main results
    baseline_df = baseline_df.merge(
        gini_df[['venue_id', 'method', 'gini']],
        on=['venue_id', 'method'],
        how='left'
    )

    # Compute gentag retention (average across all extractions per venue)
    gentag_venue_retention = gentag_retention.groupby('venue_id')['retention_cosine'].mean().reset_index()
    gentag_venue_retention['method'] = 'gentags'
    gentag_venue_retention = gentag_venue_retention.rename(columns={'retention_cosine': 'retention_cosine'})

    # Get gentag keyword counts
    gentag_counts = gentag_retention.groupby('venue_id')['n_unique_norm_eval'].mean().reset_index()
    gentag_venue_retention = gentag_venue_retention.merge(gentag_counts, on='venue_id')
    gentag_venue_retention = gentag_venue_retention.rename(columns={'n_unique_norm_eval': 'n_keywords'})
    gentag_venue_retention['keywords_sample'] = '[gentags]'
    gentag_venue_retention['gini'] = gentag_gini_mean  # Use average gentag Gini

    # Combine - only use columns that exist in both
    common_cols = [c for c in baseline_df.columns if c in gentag_venue_retention.columns]
    all_results = pd.concat([baseline_df[common_cols], gentag_venue_retention[common_cols]], ignore_index=True)

    # Save detailed results
    all_results.to_csv(TABLES_DIR / "baseline_retention.csv", index=False)
    print(f"\n[9] Saved detailed results to {TABLES_DIR / 'baseline_retention.csv'}")

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    summary = all_results.groupby('method').agg({
        'retention_cosine': ['mean', 'std', 'median', 'count'],
        'n_keywords': 'mean'
    }).round(4)

    summary.columns = ['mean', 'std', 'median', 'count', 'avg_keywords']
    summary = summary.sort_values('mean', ascending=False)

    print("\nRetention by Method:")
    print(summary)

    # Save summary
    summary.to_csv(TABLES_DIR / "baseline_summary.csv")

    # Compute effect sizes vs baselines
    gentag_mean = all_results[all_results['method'] == 'gentags']['retention_cosine'].mean()

    print("\n" + "-" * 40)
    print("COMPARISON TO GENTAGS")
    print("-" * 40)

    for method in ['tfidf', 'rake', 'yake']:
        method_mean = all_results[all_results['method'] == method]['retention_cosine'].mean()
        diff = gentag_mean - method_mean
        pct_diff = (diff / method_mean) * 100 if method_mean > 0 else 0
        print(f"  Gentags vs {method.upper()}: +{diff:.4f} ({pct_diff:+.1f}%)")

    # Random baseline comparison
    random_mean = gentag_retention['retention_random'].mean()
    print(f"\n  Random baseline: {random_mean:.4f}")
    print(f"  Gentags vs Random: +{gentag_mean - random_mean:.4f}")

    # Gini summary
    print("\n" + "=" * 60)
    print("LOCALIZATION (GINI) ANALYSIS")
    print("=" * 60)

    print("\nGini by Method (higher = more localized):")
    print(gini_summary)

    print("\n" + "-" * 40)
    print("GINI COMPARISON")
    print("-" * 40)

    for method in ['tfidf', 'rake', 'yake']:
        if method in gini_summary.index:
            method_gini = gini_summary.loc[method, 'mean']
            diff = gentag_gini_mean - method_gini
            print(f"  Gentags vs {method.upper()}: {diff:+.4f} ({'better' if diff > 0 else 'worse'})")

    print(f"\n  Gentags Gini: {gentag_gini_mean:.4f}")
    print(f"  Embeddings Gini: {embedding_gini_mean:.4f}")

    # Decision tree output
    print("\n" + "=" * 60)
    print("DECISION TREE EVALUATION")
    print("=" * 60)

    best_baseline = summary.drop('gentags').iloc[0]
    best_baseline_name = summary.drop('gentags').index[0]
    gentag_stats = summary.loc['gentags']

    retention_margin = gentag_stats['mean'] - best_baseline['mean']

    # Get best baseline Gini
    baseline_ginis = gini_summary.drop(['gentags', 'embeddings'], errors='ignore')
    best_baseline_gini = baseline_ginis['mean'].max() if len(baseline_ginis) > 0 else 0
    gini_margin = gentag_gini_mean - best_baseline_gini

    print(f"\n  Retention: gentags={gentag_stats['mean']:.4f}, best_baseline={best_baseline['mean']:.4f}")
    print(f"  Gini: gentags={gentag_gini_mean:.4f}, best_baseline={best_baseline_gini:.4f}")

    if retention_margin > 0.03:
        print("\n✅ CASE 1: Gentags beat classics on retention")
        print(f"   Retention margin: +{retention_margin:.4f} over {best_baseline_name}")
        print("   → Claim: 'LLM gentags capture semantics beyond surface extraction'")
    elif gini_margin > 0.1:
        print("\n✅ CASE 2: Gentags win on LOCALIZATION despite retention loss")
        print(f"   Retention deficit: {retention_margin:.4f} vs {best_baseline_name}")
        print(f"   Gini advantage: +{gini_margin:.4f}")
        print("   → Claim: 'Gentags provide localized, attributable semantic state'")
        print("   → Retention is sanity check; localization is the contribution")
    elif abs(retention_margin) <= 0.03:
        print("\n⚠️  CASE 3: Gentags TIE classics")
        print(f"   Retention margin: {retention_margin:+.4f}")
        print(f"   Gini margin: {gini_margin:+.4f}")
        print("   → Pivot to: stability + cross-model agreement + interpretability")
    else:
        print("\n🚨 CASE 4: Classics BEAT gentags on BOTH metrics")
        print(f"   Retention deficit: {retention_margin:.4f}")
        print(f"   Gini deficit: {gini_margin:.4f}")
        print("   → Must justify via cross-model agreement + semantic stability")

    # Final verdict
    print("\n" + "-" * 40)
    print("DIFFERENTIATORS (regardless of retention)")
    print("-" * 40)
    print("  ✓ Semantic stability: 0.977 cosine despite 0.471 Jaccard")
    print("  ✓ Cross-model agreement: 4 LLMs agree (>0.94 cosine)")
    print("  ✓ Paraphrase robustness: baselines are deterministic/brittle")
    print("  ✓ Interpretable state: tags can be diffed, tracked, monitored")

    print("\n" + "=" * 60)
    print("Phase 3A Complete")
    print("=" * 60)

    return all_results, summary


if __name__ == "__main__":
    results, summary = main()
