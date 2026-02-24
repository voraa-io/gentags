"""
Phase 5 — Baseline Legibility Study Runner.

For each venue (50) x persona (3) x system (4), runs the judge N=5
on baseline tags only (no interventions). Aggregates to majority decision.

Systems:
  1. gentag — openai minimal run1 tags
  2. rake — RAKE keywords
  3. gentag_truncated — gentags truncated to match RAKE tag count
  4. fer — Full-Evidence Reference (judge on raw review text)

Reuses from Phase 4:
  - call_judge() pattern (single API call with timing/cost)
  - validate_response() (response validation)
  - aggregate_decisions() (majority vote over N=5)
  - SYSTEM_PROMPT (exact judge prompt for tag-based systems)

New:
  - SYSTEM_PROMPT_FER — variant for FER (uses "reviews" instead of "tags")
  - Checkpoint/resume support
  - --dry-run flag

Usage:
    poetry run python scripts/phase5_baseline_runner.py                    # full run
    poetry run python scripts/phase5_baseline_runner.py --dry-run          # print prompts only
    poetry run python scripts/phase5_baseline_runner.py --resume           # continue from checkpoint
    poetry run python scripts/phase5_baseline_runner.py --venue-limit 3    # test with 3 venues
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
VENUES_FILE = REPO / "data" / "phase5" / "sampled_venues.json"
CONFIG_FILE = REPO / "data" / "phase5" / "baseline_config.json"
PERSONAS_FILE = REPO / "data" / "phase4" / "mvp_personas.json"
OUTPUT_DIR = REPO / "results" / "phase5"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DECISION_ORDINALS = {"REJECT": 0, "BORDERLINE": 1, "RECOMMEND": 2}
VALID_DECISIONS = set(DECISION_ORDINALS.keys())
N = 5
MIN_VALID = 3

SYSTEMS = ["gentag", "rake", "gentag_truncated", "fer"]

# gpt-4o pricing (as of 2025)
PRICING = {"input_per_mtok": 2.50, "output_per_mtok": 10.00}

# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------
# Tag-based prompt (identical to Phase 4 — SYSTEM_PROMPT from phase4_dir_runner.py:58)
SYSTEM_PROMPT_TAGS = """You are a strict Decision Judge.

Use ONLY the provided tags.
Do NOT use external knowledge.
Do NOT infer facts not present in the tags.

Decision rubric:
- REJECT: tags contain at least one clear blocker for the persona OR indicate high risk.
- RECOMMEND: tags contain clear supports AND no clear blockers.
- BORDERLINE: tags are mixed, ambiguous, or missing persona-critical information.

If persona-critical facets are not mentioned, choose BORDERLINE.

Output JSON ONLY:
{"decision":"REJECT|BORDERLINE|RECOMMEND",
 "justification":"one sentence",
 "tags_used":["..."]}

Rules:
- tags_used MUST be a subset of provided tags (exact string match).
- tags_used MUST list the minimal tags directly supporting the decision."""

# FER prompt — identical rubric, but "reviews" instead of "tags"
SYSTEM_PROMPT_FER = """You are a strict Decision Judge.

Use ONLY the provided reviews.
Do NOT use external knowledge.
Do NOT infer facts not present in the reviews.

Decision rubric:
- REJECT: reviews contain at least one clear blocker for the persona OR indicate high risk.
- RECOMMEND: reviews contain clear supports AND no clear blockers.
- BORDERLINE: reviews are mixed, ambiguous, or missing persona-critical information.

If persona-critical facets are not mentioned, choose BORDERLINE.

Output JSON ONLY:
{"decision":"REJECT|BORDERLINE|RECOMMEND",
 "justification":"one sentence",
 "evidence_quotes":["..."]}

Rules:
- evidence_quotes MUST be short quotes from the provided reviews.
- evidence_quotes MUST list the minimal evidence directly supporting the decision."""


def build_user_prompt_tags(persona_text: str, tags: list) -> str:
    return f"""Persona:
{persona_text}

Tags:
{json.dumps(tags)}

Task:
Choose a decision."""


def build_user_prompt_fer(persona_text: str, reviews: list) -> str:
    reviews_text = "\n\n---\n\n".join(
        f"Review {i+1}: {r}" for i, r in enumerate(reviews)
    )
    return f"""Persona:
{persona_text}

Reviews:
{reviews_text}

Task:
Choose a decision."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_response_tags(parsed: dict, input_tags: list) -> bool:
    """Validate tag-based judge response."""
    if not isinstance(parsed, dict):
        return False
    decision = parsed.get("decision")
    if decision not in VALID_DECISIONS:
        return False
    tags_used = parsed.get("tags_used")
    if not isinstance(tags_used, list):
        return False
    input_set = set(input_tags)
    for t in tags_used:
        if t not in input_set:
            return False
    return True


def validate_response_fer(parsed: dict) -> bool:
    """Validate FER judge response."""
    if not isinstance(parsed, dict):
        return False
    decision = parsed.get("decision")
    if decision not in VALID_DECISIONS:
        return False
    # evidence_quotes is expected but we're lenient — just need a list
    quotes = parsed.get("evidence_quotes")
    if quotes is not None and not isinstance(quotes, list):
        return False
    return True


# ---------------------------------------------------------------------------
# Aggregation (same as phase4_dir_runner.py)
# ---------------------------------------------------------------------------
def aggregate_decisions(runs: list) -> dict:
    valid_runs = [r for r in runs if r["valid_flag"]]
    valid_count = len(valid_runs)

    if valid_count < MIN_VALID:
        return {"aggregated_decision": "UNSCORABLE", "ordinal": None,
                "valid_count": valid_count, "total_runs": len(runs)}

    counts = Counter(r["decision"] for r in valid_runs)
    max_count = max(counts.values())
    winners = [d for d, c in counts.items() if c == max_count]

    if len(winners) == 1:
        decision = winners[0]
    else:
        decision = "BORDERLINE"  # tie -> BORDERLINE

    return {
        "aggregated_decision": decision,
        "ordinal": DECISION_ORDINALS.get(decision),
        "valid_count": valid_count,
        "total_runs": len(runs),
        "decision_counts": dict(counts),
    }


# ---------------------------------------------------------------------------
# Wilson binomial CI
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, total: int, z: float = 1.96) -> dict:
    if total == 0:
        return {"point": 0, "lower": 0, "upper": 0, "n": 0, "k": 0}
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return {
        "point": round(p, 4),
        "lower": round(max(0, centre - spread), 4),
        "upper": round(min(1, centre + spread), 4),
        "n": total,
        "k": successes,
    }


# ---------------------------------------------------------------------------
# Single Judge call
# ---------------------------------------------------------------------------
def call_judge(client: OpenAI, model: str, system_prompt: str,
               user_prompt: str) -> dict:
    """Make a single judge API call. Returns parsed result with timing/cost."""
    t0 = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    elapsed = time.time() - t0
    raw = response.choices[0].message.content

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    cost = (input_tokens / 1_000_000) * PRICING["input_per_mtok"] + \
           (output_tokens / 1_000_000) * PRICING["output_per_mtok"]

    model_version = response.model if hasattr(response, "model") else model

    # Parse JSON
    parsed = None
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        parsed = json.loads(text)
    except (json.JSONDecodeError, Exception):
        parsed = None

    return {
        "raw_response": raw,
        "parsed_json": parsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "time_seconds": round(elapsed, 3),
        "model_version": model_version,
    }


# ---------------------------------------------------------------------------
# Run one condition (venue x persona x system, N=5)
# ---------------------------------------------------------------------------
def run_condition(client: OpenAI, model: str, venue: dict,
                  persona: dict, system: str) -> dict:
    """Run N=5 judge calls for one venue-persona-system combination."""

    persona_text = persona["text"]

    if system == "fer":
        system_prompt = SYSTEM_PROMPT_FER
        user_prompt = build_user_prompt_fer(persona_text, venue["reviews"])
    else:
        system_prompt = SYSTEM_PROMPT_TAGS
        if system == "gentag":
            tags = venue["gentags"]
        elif system == "rake":
            tags = venue["rake_keywords"]
        elif system == "gentag_truncated":
            tags = venue["gentags_truncated"]
        else:
            raise ValueError(f"Unknown system: {system}")
        user_prompt = build_user_prompt_tags(persona_text, tags)

    runs = []
    for i in range(N):
        result = call_judge(client, model, system_prompt, user_prompt)

        # Validate
        if system == "fer":
            valid = validate_response_fer(result["parsed_json"]) if result["parsed_json"] else False
        else:
            valid = validate_response_tags(result["parsed_json"], tags) if result["parsed_json"] else False

        result["valid_flag"] = valid
        result["decision"] = result["parsed_json"].get("decision") if valid else None
        result["justification"] = result["parsed_json"].get("justification") if valid else None
        result["run_index"] = i
        runs.append(result)

    # Aggregate
    agg = aggregate_decisions(runs)

    return {
        "venue_id": venue["venue_id"],
        "venue_name": venue["venue_name"],
        "persona_id": persona["persona_id"],
        "system": system,
        "aggregation": agg,
        "runs": runs,
        "total_cost_usd": round(sum(r["cost_usd"] for r in runs), 6),
        "total_time_seconds": round(sum(r["time_seconds"] for r in runs), 3),
    }


# ---------------------------------------------------------------------------
# Checkpoint support
# ---------------------------------------------------------------------------
def load_checkpoint() -> set:
    """Load set of completed (venue_id, persona_id, system) tuples."""
    if not CHECKPOINT_FILE.exists():
        return set()
    with open(CHECKPOINT_FILE) as f:
        data = json.load(f)
    return {(e["venue_id"], e["persona_id"], e["system"]) for e in data.get("completed", [])}


def save_checkpoint(completed: list, results: list, run_id: str):
    """Save checkpoint with completed keys and partial results."""
    data = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed": [{"venue_id": v, "persona_id": p, "system": s}
                       for v, p, s in completed],
        "n_completed": len(completed),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Also save partial results
    partial_file = OUTPUT_DIR / f"baseline_results_{run_id}_partial.json"
    with open(partial_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 5 Baseline Legibility Runner")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--venue-limit", type=int, default=None,
                        help="Limit number of venues (for testing)")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")

    # Load venues
    with open(VENUES_FILE) as f:
        venues_data = json.load(f)
    venues = venues_data["venues"]

    if args.venue_limit:
        venues = venues[:args.venue_limit]

    # Load config
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Load personas
    with open(PERSONAS_FILE) as f:
        personas_list = json.load(f)
    personas = {p["persona_id"]: p for p in personas_list}

    model = config["judge"]["model"]
    total_conditions = len(venues) * len(personas) * len(SYSTEMS)

    print(f"Phase 5 — Baseline Legibility Study")
    print(f"Judge model: {model}")
    print(f"N: {N}")
    print(f"Venues: {len(venues)}")
    print(f"Personas: {len(personas)}")
    print(f"Systems: {SYSTEMS}")
    print(f"Total conditions: {total_conditions}")
    print(f"Total API calls: {total_conditions * N}")

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        v = venues[0]
        for pid, persona in personas.items():
            for system in SYSTEMS:
                if system == "fer":
                    prompt = build_user_prompt_fer(persona["text"], v["reviews"])
                    sys_prompt = SYSTEM_PROMPT_FER
                else:
                    tags_key = {"gentag": "gentags", "rake": "rake_keywords",
                                "gentag_truncated": "gentags_truncated"}[system]
                    prompt = build_user_prompt_tags(persona["text"], v[tags_key])
                    sys_prompt = SYSTEM_PROMPT_TAGS

                print(f"--- {v['venue_name']} | {pid} | {system} ---")
                print(f"  System prompt: {sys_prompt[:80]}...")
                print(f"  User prompt ({len(prompt)} chars): {prompt[:200]}...")
                print()
        return

    # Initialize client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        return
    client = OpenAI(api_key=api_key)

    # Resume support
    completed_keys = set()
    results = []
    if args.resume:
        completed_keys = load_checkpoint()
        # Load partial results
        partial_files = sorted(OUTPUT_DIR.glob("baseline_results_*_partial.json"))
        if partial_files:
            with open(partial_files[-1]) as f:
                results = json.load(f)
            print(f"Resumed from checkpoint: {len(completed_keys)} conditions complete")
        else:
            print("No checkpoint found, starting fresh")

    run_start = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_version_logged = None
    condition_idx = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for vi, venue in enumerate(venues):
        for pid in ["P1", "P2", "P3"]:
            persona = personas[pid]
            for system in SYSTEMS:
                key = (venue["venue_id"], pid, system)
                if key in completed_keys:
                    condition_idx += 1
                    continue

                condition_idx += 1
                print(f"\n[{condition_idx}/{total_conditions}] "
                      f"{venue['venue_name']} | {pid} | {system} ...")

                result = run_condition(client, model, venue, persona, system)
                results.append(result)

                # Log model version from first call
                if model_version_logged is None:
                    first_run = result["runs"][0]
                    model_version_logged = first_run.get("model_version", model)
                    print(f"  Model version: {model_version_logged}")

                agg = result["aggregation"]
                print(f"  Decision: {agg['aggregated_decision']} "
                      f"({agg.get('decision_counts', {})})"
                      f"  Cost=${result['total_cost_usd']:.4f}"
                      f"  Time={result['total_time_seconds']:.1f}s")

                completed_keys.add(key)

        # Checkpoint after each venue
        save_checkpoint(list(completed_keys), results, run_id)
        print(f"  [Checkpoint: {len(completed_keys)}/{total_conditions} complete]")

    run_elapsed = time.time() - run_start

    # Build summary (one row per venue-persona-system)
    summary = []
    for r in results:
        summary.append({
            "venue_id": r["venue_id"],
            "venue_name": r["venue_name"],
            "persona_id": r["persona_id"],
            "system": r["system"],
            "decision": r["aggregation"]["aggregated_decision"],
            "ordinal": r["aggregation"].get("ordinal"),
            "valid_count": r["aggregation"]["valid_count"],
            "decision_counts": r["aggregation"].get("decision_counts", {}),
            "cost_usd": r["total_cost_usd"],
            "time_seconds": r["total_time_seconds"],
        })

    # Compute quick stats
    total_cost = sum(r["total_cost_usd"] for r in results)
    total_time = sum(r["total_time_seconds"] for r in results)
    all_runs = []
    for r in results:
        all_runs.extend(r["runs"])
    invalid_count = sum(1 for r in all_runs if not r["valid_flag"])
    unscorable_count = sum(1 for r in results
                           if r["aggregation"]["aggregated_decision"] == "UNSCORABLE")

    manifest = {
        "run_id": run_id,
        "experiment": "Phase 5 — Baseline Legibility Study",
        "model": model,
        "model_version": model_version_logged,
        "n_venues": len(venues),
        "n_personas": len(personas),
        "n_systems": len(SYSTEMS),
        "N": N,
        "total_conditions": len(results),
        "total_api_calls": len(all_runs),
        "invalid_calls": invalid_count,
        "invalid_rate": wilson_ci(invalid_count, len(all_runs)),
        "unscorable_conditions": unscorable_count,
        "wall_clock_seconds": round(run_elapsed, 1),
        "total_api_time_seconds": round(total_time, 1),
        "total_cost_usd": round(total_cost, 6),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Save outputs
    results_file = OUTPUT_DIR / f"baseline_results_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_file = OUTPUT_DIR / f"baseline_summary_{run_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_file = OUTPUT_DIR / f"baseline_manifest_{run_id}.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Clean up partial/checkpoint
    for pf in OUTPUT_DIR.glob("baseline_results_*_partial.json"):
        pf.unlink()
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    # Print report
    print(f"\n{'='*60}")
    print(f"PHASE 5 — BASELINE LEGIBILITY STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID:          {run_id}")
    print(f"Model:           {model_version_logged}")
    print(f"Venues:          {len(venues)}")
    print(f"Conditions:      {len(results)}")
    print(f"API calls:       {len(all_runs)}")
    print(f"Wall clock:      {run_elapsed:.1f}s")
    print(f"Total cost:      ${total_cost:.4f}")
    print(f"Invalid calls:   {invalid_count}/{len(all_runs)}")
    print(f"Unscorable:      {unscorable_count}/{len(results)}")

    # Quick decision distribution
    print(f"\nDecision distribution by system:")
    for system in SYSTEMS:
        sys_results = [r for r in results if r["system"] == system]
        decisions = Counter(r["aggregation"]["aggregated_decision"] for r in sys_results)
        total = len(sys_results)
        rej = decisions.get("REJECT", 0)
        bor = decisions.get("BORDERLINE", 0)
        rec = decisions.get("RECOMMEND", 0)
        print(f"  {system:>20s}: REJECT={rej:>3d} ({rej/total:.0%})"
              f" BORDERLINE={bor:>3d} ({bor/total:.0%})"
              f" RECOMMEND={rec:>3d} ({rec/total:.0%})")

    print(f"\nOutputs:")
    print(f"  Results:  {results_file}")
    print(f"  Summary:  {summary_file}")
    print(f"  Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
