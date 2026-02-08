#!/usr/bin/env python3
"""
Phase 3: Model-in-the-Loop Baseline

This script implements the model-in-the-loop baseline for comparison with gentags.
For each venue, we ask the LLM 10 facet-specific questions about the reviews.

Model-in-the-loop means:
- No persistent semantic state (nothing stored)
- Every decision requires a fresh LLM call over raw reviews
- No change detection possible (nothing to diff)

This baseline demonstrates the cost and limitations of not having
a persistent semantic representation (gentags).

Setup:
- 50 venues (randomly sampled)
- 10 facets × 2 runs = 20 LLM calls per venue
- Total: 50 × 10 × 2 = 1000 LLM calls
- Provider: OpenAI (gpt-5-nano) - cheapest, most reliable

Output:
- results/phase3/model_in_loop_responses.csv (raw responses)
- results/phase3/model_in_loop_stability.csv (run1 vs run2 comparison)
- results/phase3/model_in_loop_cost.csv (cost summary)
"""

import os
import sys
import json
import time
import random
import datetime
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gentags import load_venue_data

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration (using OpenAI for reliability and cost)
MODEL_NAME = "gpt-5-nano"
MODEL_PRICING = {"input_per_mtok": 0.05, "output_per_mtok": 0.40}

# Experiment configuration
N_VENUES = 50  # Sample size
N_RUNS = 2     # For stability analysis
SEED = 42      # Reproducibility

# Output directories
OUTPUT_DIR = Path("results/phase3")
CACHE_DIR = OUTPUT_DIR / "model_in_loop_cache"

# =============================================================================
# FACET DEFINITIONS
# =============================================================================

# 10 facets matching the gentag facet assignment in PHASE3_PLAN
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

# Facet-specific questions (direct questions about each semantic dimension)
FACET_QUESTIONS = {
    "food_quality": "How would you describe the food quality at this venue based on the reviews?",
    "coffee_drinks": "What do reviewers say about the coffee and drinks at this venue?",
    "service": "How do reviewers describe the service and staff at this venue?",
    "ambiance": "What is the atmosphere or vibe of this venue according to reviewers?",
    "price_value": "How do reviewers perceive the price and value for money at this venue?",
    "crowding": "What do reviewers say about how busy or crowded this venue is?",
    "seating": "What do reviewers mention about the seating options at this venue?",
    "dietary": "Do reviewers mention any dietary options (vegan, vegetarian, gluten-free, etc.)?",
    "portions": "What do reviewers say about portion sizes at this venue?",
    "location": "What do reviewers mention about the location or accessibility of this venue?",
}

# System prompt for model-in-the-loop queries
SYSTEM_PROMPT = """You analyze venue reviews and answer specific questions about the venue.
Be concise and factual. Only report what is directly stated or clearly implied in the reviews.
If there is no relevant information in the reviews, say "No information available."
"""

# User prompt template
USER_PROMPT_TEMPLATE = """Based on the following reviews for this venue, answer this question:

QUESTION: {question}

REVIEWS:
{reviews}

Provide a brief, factual answer (1-3 sentences) based only on what the reviews say.
If the reviews don't mention anything relevant, respond with "No information available."

Answer:"""

# =============================================================================
# API CLIENT
# =============================================================================

def get_openai_client():
    """Initialize OpenAI client."""
    try:
        from openai import OpenAI
        from pathlib import Path

        # Load .env file if available
        try:
            from dotenv import load_dotenv
            for path in [
                Path(__file__).parent.parent / ".env",
                Path.cwd() / ".env"
            ]:
                if path.exists():
                    load_dotenv(path)
                    break
        except ImportError:
            pass

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package required. Install with: poetry add openai")


def query_model(
    client,
    reviews: List[str],
    facet: str,
    max_retries: int = 3
) -> Tuple[str, int, int, float]:
    """
    Query the model about a specific facet for a venue's reviews.

    Args:
        client: OpenAI client
        reviews: List of review texts
        facet: Facet to query about
        max_retries: Number of retries on failure

    Returns:
        (response_text, input_tokens, output_tokens, cost_usd)
    """
    question = FACET_QUESTIONS[facet]
    reviews_text = "\n\n".join([f"- {r}" for r in reviews[:10]])  # Limit to first 10 reviews

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        reviews=reviews_text
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )

            # Extract response and token counts
            response_text = response.choices[0].message.content.strip()
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Calculate cost
            cost_usd = (
                input_tokens * MODEL_PRICING["input_per_mtok"] / 1_000_000 +
                output_tokens * MODEL_PRICING["output_per_mtok"] / 1_000_000
            )

            return response_text, input_tokens, output_tokens, cost_usd

        except Exception as e:
            if attempt == max_retries - 1:
                return f"ERROR: {str(e)}", 0, 0, 0.0
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)

    return "ERROR: max retries exceeded", 0, 0, 0.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_venues_sample(
    venues_csv: str,
    n_sample: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """Load a random sample of venues with their reviews."""
    venues_df = load_venue_data(venues_csv)

    # Filter to venues with sufficient reviews
    venues_df = venues_df[venues_df["google_reviews"].apply(len) >= 3]

    # Sample
    rng = np.random.default_rng(seed)
    if len(venues_df) > n_sample:
        sample_idx = rng.choice(len(venues_df), size=n_sample, replace=False)
        venues_df = venues_df.iloc[sample_idx]

    return venues_df.reset_index(drop=True)


def extract_review_texts(reviews: List[dict]) -> List[str]:
    """Extract text from review objects."""
    texts = []
    for review in reviews:
        if isinstance(review, dict) and 'text' in review:
            texts.append(review['text'])
        elif isinstance(review, str):
            texts.append(review)
    return texts


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_model_in_loop_experiment(
    venues_df: pd.DataFrame,
    n_runs: int = 2,
    max_workers: int = 10
) -> pd.DataFrame:
    """
    Run the model-in-the-loop baseline experiment with parallel processing.

    For each venue, query the model about each facet N times.
    This demonstrates the cost of not having persistent semantic state.

    Uses ThreadPoolExecutor for parallel API calls (~10x speedup).
    """
    client = get_openai_client()
    results = []

    # Build list of all tasks
    tasks = []
    for _, venue_row in venues_df.iterrows():
        venue_id = venue_row["id"]
        reviews = extract_review_texts(venue_row["google_reviews"])

        for run_number in range(1, n_runs + 1):
            for facet in FACETS:
                tasks.append({
                    "venue_id": venue_id,
                    "reviews": reviews,
                    "facet": facet,
                    "run_number": run_number,
                    "n_reviews": len(reviews),
                })

    total_calls = len(tasks)

    def process_task(task):
        """Process a single query task."""
        response, input_tokens, output_tokens, cost_usd = query_model(
            client, task["reviews"], task["facet"]
        )
        return {
            "venue_id": task["venue_id"],
            "facet": task["facet"],
            "run_number": task["run_number"],
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost_usd,
            "n_reviews": task["n_reviews"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # Run in parallel with ThreadPoolExecutor
    print(f"   Using {max_workers} parallel workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}

        with tqdm(total=total_calls, desc="Model-in-loop queries", unit="query") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    task = futures[future]
                    results.append({
                        "venue_id": task["venue_id"],
                        "facet": task["facet"],
                        "run_number": task["run_number"],
                        "response": f"ERROR: {str(e)}",
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                        "n_reviews": task["n_reviews"],
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })
                pbar.update(1)

    return pd.DataFrame(results)


def compute_stability_metrics(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stability metrics between run1 and run2 responses.

    For model-in-the-loop, we check:
    - Exact match rate (same response?)
    - Response length variation
    - "No information" agreement
    """
    results = []

    for (venue_id, facet), group in responses_df.groupby(["venue_id", "facet"]):
        runs = group.sort_values("run_number")
        if len(runs) < 2:
            continue

        run1 = runs.iloc[0]
        run2 = runs.iloc[1]

        resp1 = run1["response"]
        resp2 = run2["response"]

        # Exact match
        exact_match = resp1 == resp2

        # Length difference
        len1 = len(resp1)
        len2 = len(resp2)
        len_diff = abs(len1 - len2)
        len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0

        # "No information" agreement
        no_info_phrases = ["no information", "no relevant", "not mentioned", "don't mention"]
        is_no_info_1 = any(p in resp1.lower() for p in no_info_phrases)
        is_no_info_2 = any(p in resp2.lower() for p in no_info_phrases)
        no_info_agreement = is_no_info_1 == is_no_info_2

        results.append({
            "venue_id": venue_id,
            "facet": facet,
            "exact_match": exact_match,
            "len_diff": len_diff,
            "len_ratio": len_ratio,
            "no_info_agreement": no_info_agreement,
            "response_run1": resp1[:200],  # Truncate for CSV
            "response_run2": resp2[:200],
        })

    return pd.DataFrame(results)


def compute_cost_summary(responses_df: pd.DataFrame) -> Dict:
    """Compute cost summary statistics."""
    total_cost = responses_df["cost_usd"].sum()
    total_tokens = responses_df["total_tokens"].sum()
    n_queries = len(responses_df)
    n_venues = responses_df["venue_id"].nunique()
    n_facets = len(FACETS)
    n_runs = responses_df["run_number"].nunique()

    # Per-venue cost
    per_venue_cost = total_cost / n_venues

    # Per-query cost
    per_query_cost = total_cost / n_queries

    # What it would cost for full dataset (553 venues × 10 facets × 1 run)
    projected_full_one_run = per_query_cost * 553 * 10

    # What it would cost for repeated queries (e.g., 10 queries per venue)
    projected_10_queries = per_venue_cost * 553 * 10

    return {
        "n_venues": n_venues,
        "n_facets": n_facets,
        "n_runs": n_runs,
        "n_queries": n_queries,
        "total_tokens": int(total_tokens),
        "total_cost_usd": total_cost,
        "per_venue_cost_usd": per_venue_cost,
        "per_query_cost_usd": per_query_cost,
        "projected_full_dataset_one_run_usd": projected_full_one_run,
        "projected_10_queries_per_venue_usd": projected_10_queries,
        "comparison_note": "Gentag extraction is one-time; model-in-loop scales with queries",
    }


def main():
    """Run Phase 3 model-in-the-loop baseline experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Model-in-the-Loop Baseline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/study1_venues_20250117.csv",
        help="Path to venues CSV"
    )
    parser.add_argument(
        "--n-venues",
        type=int,
        default=N_VENUES,
        help=f"Number of venues to sample (default: {N_VENUES})"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help=f"Number of runs for stability (default: {N_RUNS})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
    )

    args = parser.parse_args()

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3: MODEL-IN-THE-LOOP BASELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Venues: {args.n_venues}")
    print(f"  Facets: {len(FACETS)}")
    print(f"  Runs: {args.n_runs}")
    print(f"  Total queries: {args.n_venues * len(FACETS) * args.n_runs}")
    print(f"  Parallel workers: {args.parallel}")
    print(f"  Seed: {args.seed}")

    # Load venue sample
    print("\n1. Loading venue sample...")
    venues_df = load_venues_sample(args.data, args.n_venues, args.seed)
    print(f"   Loaded {len(venues_df)} venues")

    # Run experiment
    print("\n2. Running model-in-the-loop queries...")
    responses_df = run_model_in_loop_experiment(venues_df, args.n_runs, args.parallel)

    # Save responses
    responses_path = OUTPUT_DIR / "model_in_loop_responses.csv"
    responses_df.to_csv(responses_path, index=False)
    print(f"   Saved {len(responses_df)} responses to {responses_path}")

    # Compute stability
    print("\n3. Computing stability metrics...")
    stability_df = compute_stability_metrics(responses_df)
    stability_path = OUTPUT_DIR / "model_in_loop_stability.csv"
    stability_df.to_csv(stability_path, index=False)
    print(f"   Saved stability metrics to {stability_path}")

    # Stability summary
    if len(stability_df) > 0:
        exact_match_rate = stability_df["exact_match"].mean()
        no_info_agreement_rate = stability_df["no_info_agreement"].mean()
        mean_len_ratio = stability_df["len_ratio"].mean()
        print(f"\n   Stability Summary:")
        print(f"   - Exact match rate: {exact_match_rate:.1%}")
        print(f"   - No-info agreement: {no_info_agreement_rate:.1%}")
        print(f"   - Mean length ratio: {mean_len_ratio:.3f}")
    else:
        exact_match_rate = 0.0
        no_info_agreement_rate = 0.0
        mean_len_ratio = 0.0
        print(f"\n   Stability Summary: N/A (need n_runs >= 2 for stability analysis)")

    # Compute cost summary
    print("\n4. Computing cost summary...")
    cost_summary = compute_cost_summary(responses_df)

    # Save cost summary
    cost_path = OUTPUT_DIR / "model_in_loop_cost.json"
    with open(cost_path, "w") as f:
        json.dump(cost_summary, f, indent=2)
    print(f"   Saved cost summary to {cost_path}")

    print(f"\n   Cost Summary:")
    print(f"   - Total cost: ${cost_summary['total_cost_usd']:.4f}")
    print(f"   - Per venue: ${cost_summary['per_venue_cost_usd']:.4f}")
    print(f"   - Per query: ${cost_summary['per_query_cost_usd']:.6f}")
    print(f"   - Projected full dataset (1 run): ${cost_summary['projected_full_dataset_one_run_usd']:.2f}")

    # Write manifest
    print("\n5. Writing manifest...")
    manifest = {
        "phase": "phase3_model_in_loop",
        "model": MODEL_NAME,
        "model_pricing": MODEL_PRICING,
        "n_venues": args.n_venues,
        "n_facets": len(FACETS),
        "n_runs": args.n_runs,
        "seed": args.seed,
        "facets": FACETS,
        "facet_questions": FACET_QUESTIONS,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": {
            "n_queries": len(responses_df),
            "exact_match_rate": float(exact_match_rate),
            "no_info_agreement_rate": float(no_info_agreement_rate),
            "total_cost_usd": cost_summary["total_cost_usd"],
        }
    }

    manifest_path = OUTPUT_DIR / "model_in_loop_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Saved manifest to {manifest_path}")

    print("\n" + "=" * 60)
    print("✅ Model-in-the-loop baseline complete!")
    print("=" * 60)
    print(f"\nKey insight:")
    print(f"  Model-in-the-loop requires {args.n_venues * len(FACETS) * args.n_runs} LLM calls")
    print(f"  for {args.n_venues} venues × {len(FACETS)} facets × {args.n_runs} runs.")
    print(f"  Gentags extract once and answer many queries via tag lookup.")
    print(f"\nNext steps:")
    print(f"  1. Compare with gentag extraction cost")
    print(f"  2. Run localization experiment (Block G)")


if __name__ == "__main__":
    main()
