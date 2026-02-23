"""
Phase 4 — DIR Experiment Runner (MVP)

Implements PHASE4_EXECUTION_SPEC.md exactly:
  - State construction (S0, S1, placebo)
  - Judge prompt (no deviation)
  - Execution loop (N=5 per condition)
  - Validation (decision field, tags_used subset)
  - Aggregation (majority vote, tie->BORDERLINE, <3 valid->UNSCORABLE)
  - DIR pass rule (ordinal comparison)
  - Placebo check
  - Attribution check
  - Output metrics with binomial 95% CI

Tracks time and cost per call and total. Saves everything persistently.

Usage:
    poetry run python scripts/phase4_dir_runner.py
    poetry run python scripts/phase4_dir_runner.py --units data/phase4/mvp_dir_units.json
    poetry run python scripts/phase4_dir_runner.py --dry-run
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from math import ceil, sqrt
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO / "data" / "phase4" / "mvp_config.json"
PERSONAS_FILE = REPO / "data" / "phase4" / "mvp_personas.json"
DEFAULT_UNITS_FILE = REPO / "data" / "phase4" / "mvp_dir_units.json"
OUTPUT_DIR = REPO / "results" / "phase4"

# ---------------------------------------------------------------------------
# Constants from execution spec
# ---------------------------------------------------------------------------
DECISION_ORDINALS = {"REJECT": 0, "BORDERLINE": 1, "RECOMMEND": 2}
VALID_DECISIONS = set(DECISION_ORDINALS.keys())
N = 5
MIN_VALID = 3

# gpt-4o pricing (as of 2025)
PRICING = {"input_per_mtok": 2.50, "output_per_mtok": 10.00}

# ---------------------------------------------------------------------------
# Judge prompt (EXACT — from execution spec Part 3)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a strict Decision Judge.

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


def build_user_prompt(persona_text: str, tags: list) -> str:
    return f"""Persona:
{persona_text}

Tags:
{json.dumps(tags)}

Task:
Choose a decision."""


# ---------------------------------------------------------------------------
# State construction (execution spec Part 2)
# ---------------------------------------------------------------------------
def canonicalize(tags: list) -> list:
    return sorted(set(tags))


def construct_state(baseline_tags: list, edit_type: str, edited_tag: str,
                    replacement_tag: str = None) -> list:
    s0 = set(baseline_tags)
    if edit_type == "ADD":
        s0.add(edited_tag)
    elif edit_type == "REMOVE":
        s0.discard(edited_tag)
    elif edit_type == "REPLACE":
        s0.discard(edited_tag)
        if replacement_tag:
            s0.add(replacement_tag)
    return sorted(s0)


# ---------------------------------------------------------------------------
# Validation (execution spec Part 4)
# ---------------------------------------------------------------------------
def validate_response(parsed: dict, input_tags: list) -> bool:
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


# ---------------------------------------------------------------------------
# Aggregation (execution spec Part 5)
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
        return {"point": 0, "lower": 0, "upper": 0, "n": 0}
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
def call_judge(client: OpenAI, model: str, persona_text: str,
               tags: list) -> dict:
    user_prompt = build_user_prompt(persona_text, tags)
    t0 = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    elapsed = time.time() - t0
    raw = response.choices[0].message.content

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    cost = (input_tokens / 1_000_000) * PRICING["input_per_mtok"] + \
           (output_tokens / 1_000_000) * PRICING["output_per_mtok"]

    # Log actual model version on first call
    model_version = response.model if hasattr(response, "model") else model

    # Parse JSON
    parsed = None
    try:
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        parsed = json.loads(text)
    except (json.JSONDecodeError, Exception):
        parsed = None

    valid = validate_response(parsed, tags) if parsed else False

    return {
        "raw_response": raw,
        "parsed_json": parsed,
        "valid_flag": valid,
        "decision": parsed.get("decision") if valid else None,
        "tags_used": parsed.get("tags_used") if valid else None,
        "justification": parsed.get("justification") if valid else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "time_seconds": round(elapsed, 3),
        "model_version": model_version,
    }


# ---------------------------------------------------------------------------
# Run one condition (N=5 calls)
# ---------------------------------------------------------------------------
def run_condition(client: OpenAI, model: str, unit_id: str, condition: str,
                  persona_text: str, tags: list) -> list:
    runs = []
    for i in range(N):
        result = call_judge(client, model, persona_text, tags)
        result["unit_id"] = unit_id
        result["condition"] = condition
        result["run_index"] = i
        result["input_tags"] = tags
        runs.append(result)
    return runs


# ---------------------------------------------------------------------------
# Process one DIR unit
# ---------------------------------------------------------------------------
def process_unit(client: OpenAI, model: str, unit: dict,
                 personas: dict) -> dict:
    unit_id = unit["unit_id"]
    persona = personas[unit["persona_id"]]
    baseline_tags = canonicalize(unit["baseline_tags"])

    # State construction (execution spec Part 2)
    s0 = baseline_tags
    s1 = construct_state(
        baseline_tags, unit["edit_type"],
        unit["edited_tag"], unit.get("replacement_tag"))
    sp = construct_state(
        baseline_tags, unit["placebo_edit_type"],
        unit["placebo_tag"], unit.get("placebo_replacement_tag"))

    # Execution loop (execution spec Part 4)
    baseline_runs = run_condition(client, model, unit_id, "BASELINE",
                                  persona["text"], s0)
    intervention_runs = run_condition(client, model, unit_id, "INTERVENTION",
                                      persona["text"], s1)
    placebo_runs = run_condition(client, model, unit_id, "PLACEBO",
                                 persona["text"], sp)

    all_runs = baseline_runs + intervention_runs + placebo_runs

    # Aggregation (execution spec Part 5)
    agg_baseline = aggregate_decisions(baseline_runs)
    agg_intervention = aggregate_decisions(intervention_runs)
    agg_placebo = aggregate_decisions(placebo_runs)

    d0 = agg_baseline["ordinal"]
    d1 = agg_intervention["ordinal"]
    dp = agg_placebo["ordinal"]

    # DIR pass rule (execution spec Part 6)
    dir_pass = None
    step_size = None
    if d0 is not None and d1 is not None:
        if unit["expected_direction"] == "DOWN":
            dir_pass = d1 < d0
        elif unit["expected_direction"] == "UP":
            dir_pass = d1 > d0
        step_size = abs(d1 - d0)

    # Placebo check (execution spec Part 7)
    placebo_moved = None
    if d0 is not None and dp is not None:
        placebo_moved = dp != d0

    # Attribution check (execution spec Part 8)
    attr_pass = None
    if dir_pass and agg_intervention["aggregated_decision"] != "UNSCORABLE":
        # Get tags_used from majority run
        valid_intervention = [r for r in intervention_runs if r["valid_flag"]]
        majority_tags_used = set()
        for r in valid_intervention:
            if r["decision"] == agg_intervention["aggregated_decision"]:
                majority_tags_used.update(r["tags_used"] or [])
                break  # first matching run

        if unit["edit_type"] == "ADD":
            attr_pass = unit["edited_tag"] in majority_tags_used
        elif unit["edit_type"] == "REPLACE":
            attr_pass = unit.get("replacement_tag") in majority_tags_used
        elif unit["edit_type"] == "REMOVE":
            attr_pass = None  # skip per spec

    # Cost/time totals for this unit
    total_cost = sum(r["cost_usd"] for r in all_runs)
    total_time = sum(r["time_seconds"] for r in all_runs)

    return {
        "unit_id": unit_id,
        "venue_id": unit["venue_id"],
        "persona_id": unit["persona_id"],
        "state_type": unit["state_type"],
        "edit_type": unit["edit_type"],
        "edited_tag": unit["edited_tag"],
        "replacement_tag": unit.get("replacement_tag"),
        "expected_direction": unit["expected_direction"],
        "category": unit.get("category"),
        "states": {"s0": s0, "s1": s1, "sp": sp},
        "aggregation": {
            "baseline": agg_baseline,
            "intervention": agg_intervention,
            "placebo": agg_placebo,
        },
        "dir_pass": dir_pass,
        "step_size": step_size,
        "placebo_moved": placebo_moved,
        "attribution_pass": attr_pass,
        "total_cost_usd": round(total_cost, 6),
        "total_time_seconds": round(total_time, 3),
        "runs": all_runs,
    }


# ---------------------------------------------------------------------------
# Compute output metrics (execution spec Part 9)
# ---------------------------------------------------------------------------
def compute_metrics(results: list) -> dict:
    scorable = [r for r in results if r["dir_pass"] is not None]
    all_runs = []
    for r in results:
        all_runs.extend(r["runs"])

    total_units = len(results)
    total_calls = len(all_runs)
    invalid_calls = sum(1 for r in all_runs if not r["valid_flag"])
    unscorable_units = sum(1 for r in results if r["dir_pass"] is None)

    dir_passes = sum(1 for r in scorable if r["dir_pass"])
    placebo_moves = sum(1 for r in results
                        if r["placebo_moved"] is True)
    placebo_scorable = sum(1 for r in results
                           if r["placebo_moved"] is not None)

    # Step size distribution
    steps = [r["step_size"] for r in scorable if r["step_size"] is not None]
    step_counts = Counter(steps)

    # Attribution
    attr_applicable = [r for r in results if r["attribution_pass"] is not None]
    attr_passes = sum(1 for r in attr_applicable if r["attribution_pass"])

    metrics = {
        "DIR_pass_rate": wilson_ci(dir_passes, len(scorable)),
        "Placebo_movement_rate": wilson_ci(placebo_moves, placebo_scorable),
        "INVALID_rate": wilson_ci(invalid_calls, total_calls),
        "UNSCORABLE_rate": wilson_ci(unscorable_units, total_units),
        "Attribution_rate": wilson_ci(attr_passes, len(attr_applicable))
                           if attr_applicable else None,
        "step_size_distribution": {
            "counts": dict(step_counts),
            "mean": round(sum(steps) / len(steps), 3) if steps else None,
            "proportion_step_1": round(
                step_counts.get(1, 0) / len(steps), 3) if steps else None,
            "proportion_step_2": round(
                step_counts.get(2, 0) / len(steps), 3) if steps else None,
        },
        "total_units": total_units,
        "total_api_calls": total_calls,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 4 DIR Runner (MVP)")
    parser.add_argument("--units", default=str(DEFAULT_UNITS_FILE),
                        help="Path to DIR units JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print states and prompts without calling API")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")

    # Load config
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Load personas
    with open(PERSONAS_FILE) as f:
        personas_list = json.load(f)
    personas = {p["persona_id"]: p for p in personas_list}

    # Load DIR units
    with open(args.units) as f:
        units = json.load(f)

    model = config["judge_model"]
    print(f"Judge model: {model}")
    print(f"N: {N}")
    print(f"Units: {len(units)}")
    print(f"Total API calls: {len(units) * 3 * N}")
    print(f"Units file: {args.units}")

    if args.dry_run:
        print("\n=== DRY RUN ===\n")
        for unit in units:
            persona = personas[unit["persona_id"]]
            s0 = canonicalize(unit["baseline_tags"])
            s1 = construct_state(s0, unit["edit_type"],
                                  unit["edited_tag"],
                                  unit.get("replacement_tag"))
            sp = construct_state(s0, unit["placebo_edit_type"],
                                  unit["placebo_tag"],
                                  unit.get("placebo_replacement_tag"))
            print(f"--- {unit['unit_id']} ({unit['persona_id']}) ---")
            print(f"  Edit: {unit['edit_type']} '{unit['edited_tag']}'"
                  f" -> expected {unit['expected_direction']}")
            print(f"  S0 ({len(s0)} tags): {s0}")
            print(f"  S1 ({len(s1)} tags): {s1}")
            print(f"  SP ({len(sp)} tags): {sp}")
            print(f"  Diff S0->S1: +{set(s1)-set(s0)} -{set(s0)-set(s1)}")
            print(f"  Diff S0->SP: +{set(sp)-set(s0)} -{set(s0)-set(sp)}")
            print()
        return

    # Initialize client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        return
    client = OpenAI(api_key=api_key)

    # Run experiment
    run_start = time.time()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_version_logged = None

    results = []
    for i, unit in enumerate(units):
        print(f"\n[{i+1}/{len(units)}] {unit['unit_id']} "
              f"({unit['persona_id']}, {unit['edit_type']} "
              f"'{unit['edited_tag']}') ...")

        result = process_unit(client, model, unit, personas)
        results.append(result)

        # Log model version from first call
        if model_version_logged is None:
            first_run = result["runs"][0]
            model_version_logged = first_run.get("model_version", model)
            print(f"  Model version: {model_version_logged}")

        d0 = result["aggregation"]["baseline"]["aggregated_decision"]
        d1 = result["aggregation"]["intervention"]["aggregated_decision"]
        dp = result["aggregation"]["placebo"]["aggregated_decision"]
        print(f"  Baseline={d0} -> Intervention={d1} "
              f"(expected {unit['expected_direction']})")
        print(f"  DIR={'PASS' if result['dir_pass'] else 'FAIL'}"
              f"  Placebo={dp} moved={result['placebo_moved']}"
              f"  Attr={result['attribution_pass']}")
        print(f"  Cost=${result['total_cost_usd']:.4f} "
              f"Time={result['total_time_seconds']:.1f}s")

    run_elapsed = time.time() - run_start

    # Compute metrics
    metrics = compute_metrics(results)

    # Build manifest
    total_cost = sum(r["total_cost_usd"] for r in results)
    total_time = sum(r["total_time_seconds"] for r in results)

    manifest = {
        "run_id": run_id,
        "model": model,
        "model_version": model_version_logged,
        "units_file": str(args.units),
        "n_units": len(units),
        "n_api_calls": len(units) * 3 * N,
        "N": N,
        "wall_clock_seconds": round(run_elapsed, 1),
        "total_api_time_seconds": round(total_time, 1),
        "total_cost_usd": round(total_cost, 6),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
    }

    # Save everything
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full results (every run, every response)
    results_file = OUTPUT_DIR / f"dir_results_{run_id}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Manifest (metrics + metadata)
    manifest_file = OUTPUT_DIR / f"dir_manifest_{run_id}.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Summary table (one row per unit, for quick inspection)
    summary = []
    for r in results:
        summary.append({
            "unit_id": r["unit_id"],
            "persona_id": r["persona_id"],
            "state_type": r["state_type"],
            "edit_type": r["edit_type"],
            "edited_tag": r["edited_tag"],
            "expected_direction": r["expected_direction"],
            "baseline_decision": r["aggregation"]["baseline"]["aggregated_decision"],
            "intervention_decision": r["aggregation"]["intervention"]["aggregated_decision"],
            "placebo_decision": r["aggregation"]["placebo"]["aggregated_decision"],
            "dir_pass": r["dir_pass"],
            "step_size": r["step_size"],
            "placebo_moved": r["placebo_moved"],
            "attribution_pass": r["attribution_pass"],
            "cost_usd": r["total_cost_usd"],
            "time_seconds": r["total_time_seconds"],
        })
    summary_file = OUTPUT_DIR / f"dir_summary_{run_id}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final report
    print(f"\n{'='*60}")
    print(f"PHASE 4 DIR — RUN COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID:          {run_id}")
    print(f"Model:           {model_version_logged}")
    print(f"Units:           {len(units)}")
    print(f"API calls:       {len(units) * 3 * N}")
    print(f"Wall clock:      {run_elapsed:.1f}s")
    print(f"Total API time:  {total_time:.1f}s")
    print(f"Total cost:      ${total_cost:.4f}")
    print()

    m = metrics
    print(f"DIR pass rate:       {m['DIR_pass_rate']['k']}/{m['DIR_pass_rate']['n']}"
          f" = {m['DIR_pass_rate']['point']:.1%}"
          f" [{m['DIR_pass_rate']['lower']:.1%}, {m['DIR_pass_rate']['upper']:.1%}]")
    print(f"Placebo movement:    {m['Placebo_movement_rate']['k']}/{m['Placebo_movement_rate']['n']}"
          f" = {m['Placebo_movement_rate']['point']:.1%}"
          f" [{m['Placebo_movement_rate']['lower']:.1%}, {m['Placebo_movement_rate']['upper']:.1%}]")
    print(f"INVALID rate:        {m['INVALID_rate']['k']}/{m['INVALID_rate']['n']}"
          f" = {m['INVALID_rate']['point']:.1%}")
    print(f"UNSCORABLE rate:     {m['UNSCORABLE_rate']['k']}/{m['UNSCORABLE_rate']['n']}"
          f" = {m['UNSCORABLE_rate']['point']:.1%}")
    if m.get("Attribution_rate"):
        print(f"Attribution rate:    {m['Attribution_rate']['k']}/{m['Attribution_rate']['n']}"
              f" = {m['Attribution_rate']['point']:.1%}")
    print(f"Mean step size:      {m['step_size_distribution']['mean']}")
    print()

    print(f"Results:  {results_file}")
    print(f"Manifest: {manifest_file}")
    print(f"Summary:  {summary_file}")


if __name__ == "__main__":
    main()
