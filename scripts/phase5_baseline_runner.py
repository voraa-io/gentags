"""
Phase 5 — Baseline Legibility Study Runner (v2).

For each venue (50) x persona (4) x system (6), runs the judge N=5
on baseline tags only (no interventions). Aggregates to majority decision.

Personas (4):
  P1: Food Critic (hard — negative food indicator → REJECT)
  P2: Sports Fan (hard — no sports indicator → REJECT)
  P3: Quick Lunch Worker (hard — no speed indicator → REJECT)
  P4: Balanced Diner (soft — no hard requirement)

Systems (6):
  1. gentag — openai minimal run1 tags
  2. rake — RAKE keywords
  3. yake — YAKE keywords
  4. tfidf — TF-IDF keywords
  5. gentag_truncated — gentags truncated to match RAKE tag count
  6. fer — Full-Evidence Reference (judge on raw review text)

Judges:
  --judge openai   → gpt-4o (default)
  --judge claude   → claude-sonnet-4-20250514 via Anthropic API

Features:
  - Strict judge prompt with requirement_status, blockers, supports
  - Frozen indicator lexicons per persona
  - Checkpoint/resume support
  - Progress tracking with ETA and running cost
  - Cross-judge support

Usage:
    poetry run python scripts/phase5_baseline_runner.py                         # full run (OpenAI)
    poetry run python scripts/phase5_baseline_runner.py --judge claude          # full run (Claude)
    poetry run python scripts/phase5_baseline_runner.py --dry-run               # print prompts only
    poetry run python scripts/phase5_baseline_runner.py --resume                # continue from checkpoint
    poetry run python scripts/phase5_baseline_runner.py --venue-limit 3         # test with 3 venues
"""

import argparse
import asyncio
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
VENUES_FILE = REPO / "data" / "phase5" / "sampled_venues.json"
CONFIG_FILE = REPO / "data" / "phase5" / "baseline_config.json"
PERSONAS_FILE = REPO / "data" / "phase5" / "phase5_personas.json"
OUTPUT_DIR = REPO / "results" / "phase5"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DECISION_ORDINALS = {"REJECT": 0, "BORDERLINE": 1, "RECOMMEND": 2}
VALID_DECISIONS = set(DECISION_ORDINALS.keys())
VALID_REQ_STATUS = {"SATISFIED", "VIOLATED", "NOT_APPLICABLE"}
N = 5
MIN_VALID = 3
MAX_CONCURRENT = 15  # max parallel API calls

SYSTEMS = ["gentag", "rake", "yake", "tfidf", "gentag_truncated", "fer"]

TAG_SYSTEM_KEYS = {
    "gentag": "gentags",
    "rake": "rake_keywords",
    "yake": "yake_keywords",
    "tfidf": "tfidf_keywords",
    "gentag_truncated": "gentags_truncated",
}

# Pricing per 1M tokens
PRICING = {
    "openai": {"input": 2.50, "output": 10.00},
    "claude": {"input": 3.00, "output": 15.00},
}

# ---------------------------------------------------------------------------
# Judge prompts (v2 — strict, paper-safe)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TAGS = """You are a strict Decision Judge.

Use ONLY the provided tags. Do NOT use external knowledge. Do NOT infer facts not present in the tags. Treat synonyms as NOT present unless they appear exactly in the tag list.

Decision rules:
- If the persona has a hard requirement and the requirement is VIOLATED based on the provided indicator set, output REJECT.
- If the persona has a hard requirement and the requirement is SATISFIED, do NOT reject unless other clear blockers exist.
- RECOMMEND if tags contain clear supports AND no clear blockers.
- BORDERLINE if tags are mixed or ambiguous.
- If the persona has no hard requirement, weigh all relevant tags.

Return ONE line of valid JSON and nothing else:
{"decision":"REJECT|BORDERLINE|RECOMMEND","requirement_status":"SATISFIED|VIOLATED|NOT_APPLICABLE","blockers":["..."],"supports":["..."],"tags_used":["..."],"justification":"one sentence"}

Strict rules:
- tags_used MUST be a subset of the provided tags (exact string match).
- blockers and supports MUST be subsets of tags_used.
- tags_used MUST list only the minimal tags directly supporting the decision.
- If you cite a tag not in the provided list, the response is INVALID."""

SYSTEM_PROMPT_FER = """You are a strict Decision Judge.

Use ONLY the provided reviews. Do NOT use external knowledge. Do NOT infer facts not present in the reviews.

Decision rules:
- If the persona has a hard requirement and the reviews show the requirement is VIOLATED, output REJECT.
- If the persona has a hard requirement and the reviews show the requirement is SATISFIED, do NOT reject unless other clear blockers exist.
- RECOMMEND if reviews contain clear supports AND no clear blockers.
- BORDERLINE if reviews are mixed or ambiguous.
- If the persona has no hard requirement, weigh all relevant evidence.

Return ONE line of valid JSON and nothing else:
{"decision":"REJECT|BORDERLINE|RECOMMEND","requirement_status":"SATISFIED|VIOLATED|NOT_APPLICABLE","blockers":["short quote"],"supports":["short quote"],"evidence_quotes":["..."],"justification":"one sentence"}

Strict rules:
- evidence_quotes MUST be short quotes from the provided reviews.
- evidence_quotes MUST list only the minimal evidence directly supporting the decision."""


def build_user_prompt_tags(persona: dict, tags: list) -> str:
    """Build user prompt for tag-based systems, including requirement + indicators."""
    parts = [f"Persona:\n{persona['text']}"]

    if persona.get("hard_requirement"):
        parts.append(f"\nHard requirement:\n{persona['hard_requirement']}")

    # Include indicator set for hard personas
    indicators = persona.get("indicators") or []
    indicators_neg = persona.get("indicators_negative") or []
    indicators_pos = persona.get("indicators_positive") or []

    if indicators:
        parts.append(f"\nRequirement indicators (exact tag match only):\n{json.dumps(indicators)}")
    if indicators_neg:
        parts.append(f"\nNegative indicators (if present → REJECT):\n{json.dumps(indicators_neg)}")
    if indicators_pos:
        parts.append(f"\nPositive indicators:\n{json.dumps(indicators_pos)}")

    parts.append(f"\nTags:\n{json.dumps(tags)}")
    parts.append("\nTask:\nDecide using ONLY the tags.")

    return "\n".join(parts)


def build_user_prompt_fer(persona: dict, reviews: list) -> str:
    """Build user prompt for FER system (reviews instead of tags)."""
    reviews_text = "\n\n---\n\n".join(
        f"Review {i+1}: {r}" for i, r in enumerate(reviews)
    )
    parts = [f"Persona:\n{persona['text']}"]

    if persona.get("hard_requirement"):
        parts.append(f"\nHard requirement:\n{persona['hard_requirement']}")

    parts.append(f"\nReviews:\n{reviews_text}")
    parts.append("\nTask:\nDecide using ONLY the reviews.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation (strict)
# ---------------------------------------------------------------------------
def validate_response_tags(parsed: dict, input_tags: list) -> bool:
    """Validate tag-based judge response (strict)."""
    if not isinstance(parsed, dict):
        return False
    if parsed.get("decision") not in VALID_DECISIONS:
        return False
    # requirement_status check
    req = parsed.get("requirement_status")
    if req is not None and req not in VALID_REQ_STATUS:
        return False
    # tags_used must be subset of input
    tags_used = parsed.get("tags_used")
    if not isinstance(tags_used, list):
        return False
    input_set = set(input_tags)
    for t in tags_used:
        if t not in input_set:
            return False
    # blockers/supports must be subset of tags_used
    tags_used_set = set(tags_used)
    for field in ["blockers", "supports"]:
        items = parsed.get(field, [])
        if isinstance(items, list):
            for item in items:
                if item not in tags_used_set:
                    return False
    return True


def validate_response_fer(parsed: dict) -> bool:
    """Validate FER judge response."""
    if not isinstance(parsed, dict):
        return False
    if parsed.get("decision") not in VALID_DECISIONS:
        return False
    req = parsed.get("requirement_status")
    if req is not None and req not in VALID_REQ_STATUS:
        return False
    quotes = parsed.get("evidence_quotes")
    if quotes is not None and not isinstance(quotes, list):
        return False
    return True


# ---------------------------------------------------------------------------
# Aggregation
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

    decision = winners[0] if len(winners) == 1 else "BORDERLINE"

    return {
        "aggregated_decision": decision,
        "ordinal": DECISION_ORDINALS.get(decision),
        "valid_count": valid_count,
        "total_runs": len(runs),
        "decision_counts": dict(counts),
    }


# ---------------------------------------------------------------------------
# Wilson CI
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
        "n": total, "k": successes,
    }


# ---------------------------------------------------------------------------
# Judge API calls
# ---------------------------------------------------------------------------
def call_judge_openai(client, model: str, system_prompt: str,
                      user_prompt: str) -> dict:
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
    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    cost = (in_tok / 1e6) * PRICING["openai"]["input"] + \
           (out_tok / 1e6) * PRICING["openai"]["output"]
    model_version = response.model if hasattr(response, "model") else model
    return _parse_result(raw, in_tok, out_tok, cost, elapsed, model_version)


def call_judge_claude(client, model: str, system_prompt: str,
                      user_prompt: str) -> dict:
    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    elapsed = time.time() - t0
    raw = response.content[0].text if response.content else ""
    in_tok = response.usage.input_tokens if response.usage else 0
    out_tok = response.usage.output_tokens if response.usage else 0
    cost = (in_tok / 1e6) * PRICING["claude"]["input"] + \
           (out_tok / 1e6) * PRICING["claude"]["output"]
    model_version = response.model if hasattr(response, "model") else model
    return _parse_result(raw, in_tok, out_tok, cost, elapsed, model_version)


def _parse_result(raw, in_tok, out_tok, cost, elapsed, model_version):
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
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": round(cost, 6),
        "time_seconds": round(elapsed, 3),
        "model_version": model_version,
    }


# ---------------------------------------------------------------------------
# Async judge API calls (for concurrent execution)
# ---------------------------------------------------------------------------
async def call_judge_openai_async(client, model: str, system_prompt: str,
                                  user_prompt: str) -> dict:
    t0 = time.time()
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
        except Exception as e:
            if ("rate_limit" in str(e) or "429" in str(e)) and attempt < 4:
                await asyncio.sleep(15 * (attempt + 1))
                continue
            raise
    elapsed = time.time() - t0
    raw = response.choices[0].message.content
    in_tok = response.usage.prompt_tokens if response.usage else 0
    out_tok = response.usage.completion_tokens if response.usage else 0
    cost = (in_tok / 1e6) * PRICING["openai"]["input"] + \
           (out_tok / 1e6) * PRICING["openai"]["output"]
    model_version = response.model if hasattr(response, "model") else model
    return _parse_result(raw, in_tok, out_tok, cost, elapsed, model_version)


async def call_judge_claude_async(client, model: str, system_prompt: str,
                                  user_prompt: str) -> dict:
    t0 = time.time()
    for attempt in range(5):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            break
        except Exception as e:
            if "rate_limit" in str(e) and attempt < 4:
                await asyncio.sleep(15 * (attempt + 1))
                continue
            raise
    elapsed = time.time() - t0
    raw = response.content[0].text if response.content else ""
    in_tok = response.usage.input_tokens if response.usage else 0
    out_tok = response.usage.output_tokens if response.usage else 0
    cost = (in_tok / 1e6) * PRICING["claude"]["input"] + \
           (out_tok / 1e6) * PRICING["claude"]["output"]
    model_version = response.model if hasattr(response, "model") else model
    return _parse_result(raw, in_tok, out_tok, cost, elapsed, model_version)


# ---------------------------------------------------------------------------
# Async run one condition (venue x persona x system, N=5 concurrent)
# ---------------------------------------------------------------------------
async def run_condition_async(semaphore, call_fn, model: str, venue: dict,
                              persona: dict, system: str) -> dict:
    if system == "fer":
        sys_prompt = SYSTEM_PROMPT_FER
        user_prompt = build_user_prompt_fer(persona, venue["reviews"])
        tags = None
    else:
        sys_prompt = SYSTEM_PROMPT_TAGS
        tags_key = TAG_SYSTEM_KEYS[system]
        tags = venue[tags_key]
        user_prompt = build_user_prompt_tags(persona, tags)

    async def single_call(i):
        async with semaphore:
            result = await call_fn(model, sys_prompt, user_prompt)
        if system == "fer":
            valid = validate_response_fer(result["parsed_json"]) if result["parsed_json"] else False
        else:
            valid = validate_response_tags(result["parsed_json"], tags) if result["parsed_json"] else False
        result["valid_flag"] = valid
        result["decision"] = result["parsed_json"].get("decision") if valid else None
        result["requirement_status"] = result["parsed_json"].get("requirement_status") if valid else None
        result["justification"] = result["parsed_json"].get("justification") if valid else None
        result["run_index"] = i
        return result

    runs = await asyncio.gather(*[single_call(i) for i in range(N)])
    runs = list(runs)
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
# Run one condition (venue x persona x system, N=5) — sync fallback
# ---------------------------------------------------------------------------
def run_condition(call_fn, model: str, venue: dict,
                  persona: dict, system: str) -> dict:
    if system == "fer":
        sys_prompt = SYSTEM_PROMPT_FER
        user_prompt = build_user_prompt_fer(persona, venue["reviews"])
        tags = None
    else:
        sys_prompt = SYSTEM_PROMPT_TAGS
        tags_key = TAG_SYSTEM_KEYS[system]
        tags = venue[tags_key]
        user_prompt = build_user_prompt_tags(persona, tags)

    runs = []
    for i in range(N):
        result = call_fn(model, sys_prompt, user_prompt)

        if system == "fer":
            valid = validate_response_fer(result["parsed_json"]) if result["parsed_json"] else False
        else:
            valid = validate_response_tags(result["parsed_json"], tags) if result["parsed_json"] else False

        result["valid_flag"] = valid
        result["decision"] = result["parsed_json"].get("decision") if valid else None
        result["requirement_status"] = result["parsed_json"].get("requirement_status") if valid else None
        result["justification"] = result["parsed_json"].get("justification") if valid else None
        result["run_index"] = i
        runs.append(result)

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
def get_checkpoint_file(judge_name: str) -> Path:
    return OUTPUT_DIR / f"checkpoint_{judge_name}.json"


def get_partial_pattern(judge_name: str) -> str:
    return f"baseline_results_{judge_name}_*_partial.json"


def load_checkpoint(judge_name: str) -> set:
    cp = get_checkpoint_file(judge_name)
    if not cp.exists():
        return set()
    with open(cp) as f:
        data = json.load(f)
    return {(e["venue_id"], e["persona_id"], e["system"])
            for e in data.get("completed", [])}


def save_checkpoint(judge_name: str, completed: list, results: list, run_id: str):
    cp = get_checkpoint_file(judge_name)
    data = {
        "run_id": run_id,
        "judge": judge_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed": [{"venue_id": v, "persona_id": p, "system": s}
                       for v, p, s in completed],
        "n_completed": len(completed),
    }
    with open(cp, "w") as f:
        json.dump(data, f, indent=2)

    partial_file = OUTPUT_DIR / f"baseline_results_{judge_name}_{run_id}_partial.json"
    with open(partial_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------
def format_eta(elapsed_seconds: float, done: int, total: int) -> str:
    if done == 0:
        return "calculating..."
    rate = elapsed_seconds / done
    remaining = (total - done) * rate
    if remaining < 60:
        return f"{remaining:.0f}s"
    elif remaining < 3600:
        return f"{remaining/60:.1f}m"
    else:
        return f"{remaining/3600:.1f}h"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Phase 5 Baseline Legibility Runner (v2)")
    parser.add_argument("--judge", default="openai", choices=["openai", "claude"],
                        help="Judge to use: openai (gpt-4o) or claude")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--venue-limit", type=int, default=None,
                        help="Limit number of venues (for testing)")
    parser.add_argument("--concurrency", type=int, default=MAX_CONCURRENT,
                        help=f"Max parallel API calls (default {MAX_CONCURRENT})")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")

    # Load venues
    with open(VENUES_FILE) as f:
        venues_data = json.load(f)
    venues = venues_data["venues"]

    if args.venue_limit:
        venues = venues[:args.venue_limit]

    # Load personas
    with open(PERSONAS_FILE) as f:
        personas_list = json.load(f)
    personas = {p["persona_id"]: p for p in personas_list}
    persona_ids = sorted(personas.keys())

    total_conditions = len(venues) * len(persona_ids) * len(SYSTEMS)

    # Judge setup
    if args.judge == "openai":
        model = "gpt-4o"
        judge_label = "openai"
    else:
        model = "claude-sonnet-4-20250514"
        judge_label = "claude"
        # Claude rate limit is 50 req/min — cap concurrency
        if args.concurrency > 8:
            args.concurrency = 8

    print(f"Phase 5 — Baseline Legibility Study (v2, async)")
    print(f"Judge:       {judge_label} ({model})")
    print(f"N:           {N}")
    print(f"Venues:      {len(venues)}")
    print(f"Personas:    {len(persona_ids)} ({', '.join(persona_ids)})")
    print(f"Systems:     {len(SYSTEMS)} ({', '.join(SYSTEMS)})")
    print(f"Conditions:  {total_conditions}")
    print(f"API calls:   {total_conditions * N}")
    print(f"Concurrency: {args.concurrency}")

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        v = venues[0]
        for pid in persona_ids:
            persona = personas[pid]
            for system in SYSTEMS:
                if system == "fer":
                    prompt = build_user_prompt_fer(persona, v["reviews"])
                    sys_prompt = SYSTEM_PROMPT_FER
                else:
                    tags_key = TAG_SYSTEM_KEYS[system]
                    prompt = build_user_prompt_tags(persona, v[tags_key])
                    sys_prompt = SYSTEM_PROMPT_TAGS

                print(f"--- {v['venue_name']} | {pid} | {system} ---")
                print(f"  System prompt ({len(sys_prompt)} chars):")
                print(f"    {sys_prompt[:120]}...")
                print(f"  User prompt ({len(prompt)} chars):")
                print(f"    {prompt[:250]}...")
                print()
        return

    # Initialize async client
    if args.judge == "openai":
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set in .env")
            return
        client = AsyncOpenAI(api_key=api_key)
        call_fn = lambda m, sp, up: call_judge_openai_async(client, m, sp, up)
    else:
        from anthropic import AsyncAnthropic
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            print("ERROR: CLAUDE_API_KEY not set in .env")
            return
        client = AsyncAnthropic(api_key=api_key)
        call_fn = lambda m, sp, up: call_judge_claude_async(client, m, sp, up)

    semaphore = asyncio.Semaphore(args.concurrency)

    # Resume support
    completed_keys = set()
    results = []
    if args.resume:
        completed_keys = load_checkpoint(judge_label)
        partial_files = sorted(OUTPUT_DIR.glob(f"baseline_results_{judge_label}_*_partial.json"))
        if partial_files:
            with open(partial_files[-1]) as f:
                results = json.load(f)
            print(f"Resumed: {len(completed_keys)} conditions complete")
        else:
            print("No checkpoint found, starting fresh")

    run_start = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_version_logged = None
    running_cost = sum(r["total_cost_usd"] for r in results)
    conditions_done = len(completed_keys)
    initial_done = conditions_done  # for ETA calculation

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for vi, venue in enumerate(venues):
        # Build list of conditions to run for this venue
        venue_tasks = []
        for pid in persona_ids:
            persona = personas[pid]
            for system in SYSTEMS:
                key = (venue["venue_id"], pid, system)
                if key in completed_keys:
                    continue
                venue_tasks.append((pid, persona, system, key))

        if not venue_tasks:
            conditions_done += len(persona_ids) * len(SYSTEMS)
            continue

        n_todo = len(venue_tasks)
        print(f"\n[Venue {vi+1}/{len(venues)}] {venue['venue_name']} "
              f"— {n_todo} conditions ({n_todo * N} calls, "
              f"max {args.concurrency} concurrent)")

        # Run all conditions for this venue concurrently
        coros = [
            run_condition_async(semaphore, call_fn, model, venue, persona, system)
            for pid, persona, system, key in venue_tasks
        ]
        venue_results = await asyncio.gather(*coros, return_exceptions=True)

        # Process results
        for (pid, persona, system, key), result in zip(venue_tasks, venue_results):
            if isinstance(result, Exception):
                print(f"  ERROR {pid} | {system}: {result}")
                continue

            results.append(result)
            running_cost += result["total_cost_usd"]
            completed_keys.add(key)
            conditions_done += 1

            if model_version_logged is None:
                model_version_logged = result["runs"][0].get("model_version", model)
                print(f"  Model version: {model_version_logged}")

            agg = result["aggregation"]
            invalid_in_run = sum(1 for r in result["runs"] if not r["valid_flag"])
            print(f"  {pid} | {system:18s} → {agg['aggregated_decision']:10s} "
                  f"({agg.get('decision_counts', {})})"
                  f"{'  !! INVALID=' + str(invalid_in_run) if invalid_in_run else ''}")

        # Checkpoint after each venue
        save_checkpoint(judge_label, list(completed_keys), results, run_id)
        elapsed = time.time() - run_start
        done_this_run = conditions_done - initial_done
        remaining_this_run = total_conditions - conditions_done
        eta = format_eta(elapsed, done_this_run, done_this_run + remaining_this_run)
        print(f"  -- checkpoint: {conditions_done}/{total_conditions} "
              f"({conditions_done/total_conditions:.0%}) | ${running_cost:.2f} "
              f"| {elapsed:.0f}s elapsed | ETA: {eta}")

    run_elapsed = time.time() - run_start

    # Build summary
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

    # Stats
    all_runs = []
    for r in results:
        all_runs.extend(r["runs"])
    invalid_count = sum(1 for r in all_runs if not r["valid_flag"])
    unscorable_count = sum(1 for r in results
                           if r["aggregation"]["aggregated_decision"] == "UNSCORABLE")

    manifest = {
        "run_id": run_id,
        "experiment": "Phase 5 — Baseline Legibility Study (v2, async)",
        "judge": judge_label,
        "model": model,
        "model_version": model_version_logged,
        "n_venues": len(venues),
        "n_personas": len(persona_ids),
        "n_systems": len(SYSTEMS),
        "systems": SYSTEMS,
        "persona_ids": persona_ids,
        "N": N,
        "total_conditions": len(results),
        "total_api_calls": len(all_runs),
        "invalid_calls": invalid_count,
        "invalid_rate": wilson_ci(invalid_count, len(all_runs)),
        "unscorable_conditions": unscorable_count,
        "wall_clock_seconds": round(run_elapsed, 1),
        "total_cost_usd": round(running_cost, 6),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "concurrency": args.concurrency,
    }

    # Save outputs (include judge name in filenames)
    results_file = OUTPUT_DIR / f"baseline_results_{judge_label}_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_file = OUTPUT_DIR / f"baseline_summary_{judge_label}_{run_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    manifest_file = OUTPUT_DIR / f"baseline_manifest_{judge_label}_{run_id}.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Clean up
    for pf in OUTPUT_DIR.glob(f"baseline_results_{judge_label}_*_partial.json"):
        pf.unlink()
    cp = get_checkpoint_file(judge_label)
    if cp.exists():
        cp.unlink()

    # Final report
    print(f"\n{'='*70}")
    print(f"PHASE 5 — BASELINE LEGIBILITY COMPLETE ({judge_label})")
    print(f"{'='*70}")
    print(f"Run ID:       {run_id}")
    print(f"Judge:        {judge_label} ({model_version_logged})")
    print(f"Venues:       {len(venues)}")
    print(f"Conditions:   {len(results)}")
    print(f"API calls:    {len(all_runs)}")
    print(f"Wall clock:   {run_elapsed:.1f}s ({run_elapsed/60:.1f}m)")
    print(f"Total cost:   ${running_cost:.4f}")
    print(f"Invalid:      {invalid_count}/{len(all_runs)} ({invalid_count/len(all_runs):.1%})")
    print(f"Unscorable:   {unscorable_count}/{len(results)}")
    print(f"Concurrency:  {args.concurrency}")

    print(f"\nDecision distribution by system:")
    for system in SYSTEMS:
        sys_results = [r for r in results if r["system"] == system]
        if not sys_results:
            continue
        decisions = Counter(r["aggregation"]["aggregated_decision"] for r in sys_results)
        total = len(sys_results)
        rej = decisions.get("REJECT", 0)
        bor = decisions.get("BORDERLINE", 0)
        rec = decisions.get("RECOMMEND", 0)
        print(f"  {system:>18s}: REJ={rej:>3d} ({rej/total:.0%})"
              f"  BOR={bor:>3d} ({bor/total:.0%})"
              f"  REC={rec:>3d} ({rec/total:.0%})")

    print(f"\nOutputs:")
    print(f"  Results:  {results_file}")
    print(f"  Summary:  {summary_file}")
    print(f"  Manifest: {manifest_file}")


if __name__ == "__main__":
    asyncio.run(main())
