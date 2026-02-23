"""
Phase 4 — Cross-venue DIR Aggregation.

Combines summary JSONs from all gentag and RAKE runs across venues.
Computes combined DIR pass rate, non-floor pass rate, placebo movement,
gentag vs RAKE separation with Fisher's exact test.

Usage:
    poetry run python scripts/phase4_aggregate.py \
      --gentag-summaries results/phase4/dir_summary_A.json results/phase4/dir_summary_B.json ... \
      --rake-summaries results/phase4/dir_summary_X.json results/phase4/dir_summary_Y.json ...
"""

import argparse
import json
from math import sqrt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "results" / "phase4"

# ---------------------------------------------------------------------------
# Wilson binomial CI (same as dir_runner)
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, total: int, z: float = 1.96) -> dict:
    if total == 0:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "n": 0, "k": 0}
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
# Fisher's exact test (2x2)
# ---------------------------------------------------------------------------
def fishers_exact_test(a, b, c, d):
    """2x2 contingency table Fisher's exact test.

    Table:
        | pass | fail |
    gt  |  a   |  b   |
    rake|  c   |  d   |

    Returns p-value (two-sided).
    """
    try:
        from scipy.stats import fisher_exact
        table = [[a, b], [c, d]]
        _, p = fisher_exact(table, alternative="two-sided")
        return round(p, 6)
    except ImportError:
        # Manual Fisher's exact for small tables
        from math import comb, factorial
        n = a + b + c + d
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d

        def hypergeom_pmf(k):
            return comb(col1, k) * comb(col2, row1 - k) / comb(n, row1)

        p_obs = hypergeom_pmf(a)
        p_value = 0.0
        for k in range(max(0, row1 - col2), min(row1, col1) + 1):
            p_k = hypergeom_pmf(k)
            if p_k <= p_obs + 1e-10:
                p_value += p_k
        return round(min(p_value, 1.0), 6)


# ---------------------------------------------------------------------------
# Floor detection
# ---------------------------------------------------------------------------
# Floor units: baseline=REJECT AND expected_direction=DOWN
# (can't go DOWN from REJECT — already at ordinal 0)
def is_floor(unit: dict) -> bool:
    return (unit.get("baseline_decision") == "REJECT" and
            unit.get("expected_direction") == "DOWN")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — Cross-venue DIR Aggregation")
    parser.add_argument("--gentag-summaries", nargs="+", required=True,
                        help="Gentag summary JSON files")
    parser.add_argument("--rake-summaries", nargs="+", required=True,
                        help="RAKE summary JSON files")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "scaled_aggregate.json"),
                        help="Output file path")
    args = parser.parse_args()

    # Load all units
    gentag_units = []
    for path in args.gentag_summaries:
        with open(path) as f:
            gentag_units.extend(json.load(f))

    rake_units = []
    for path in args.rake_summaries:
        with open(path) as f:
            rake_units.extend(json.load(f))

    # --- Per-venue breakdown ---
    def venue_from_unit_id(uid):
        num = int(uid.replace("DIR-", "").replace("-RAKE", ""))
        if num <= 8:
            return "coltons"
        elif num <= 12:
            return "boost"
        else:
            return "boru"

    venues_info = {
        "coltons": {"name": "Colton's - Monterrey", "venue_id": "MFJDDz0Mgf5LOkmbvW8b"},
        "boost": {"name": "Boost Coffee", "venue_id": "i81K8YdfE4SpNwiY1uPw"},
        "boru": {"name": "Boru - Gómez Morín", "venue_id": "52kAl7jt0PHLpWz3crjS"},
    }

    # --- Compute metrics ---
    def compute_group_metrics(units, label):
        scorable = [u for u in units if u["dir_pass"] is not None]
        passes = sum(1 for u in scorable if u["dir_pass"])
        placebo_scorable = [u for u in units if u["placebo_moved"] is not None]
        placebo_moves = sum(1 for u in placebo_scorable if u["placebo_moved"])

        non_floor = [u for u in units if not is_floor(u)]
        non_floor_scorable = [u for u in non_floor if u["dir_pass"] is not None]
        non_floor_passes = sum(1 for u in non_floor_scorable if u["dir_pass"])

        floor_units = [u for u in units if is_floor(u)]

        return {
            "label": label,
            "total_units": len(units),
            "scorable_units": len(scorable),
            "dir_passes": passes,
            "dir_pass_rate": wilson_ci(passes, len(scorable)),
            "non_floor_units": len(non_floor),
            "non_floor_scorable": len(non_floor_scorable),
            "non_floor_passes": non_floor_passes,
            "non_floor_pass_rate": wilson_ci(non_floor_passes, len(non_floor_scorable)),
            "floor_unit_ids": [u["unit_id"] for u in floor_units],
            "placebo_scorable": len(placebo_scorable),
            "placebo_moves": placebo_moves,
            "placebo_movement_rate": wilson_ci(placebo_moves, len(placebo_scorable)),
        }

    gentag_metrics = compute_group_metrics(gentag_units, "gentag")
    rake_metrics = compute_group_metrics(rake_units, "rake")

    # --- Per-venue breakdown ---
    per_venue = {}
    for venue_key in ["coltons", "boost", "boru"]:
        gt_venue = [u for u in gentag_units if venue_from_unit_id(u["unit_id"]) == venue_key]
        rk_venue = [u for u in rake_units if venue_from_unit_id(u["unit_id"]) == venue_key]
        per_venue[venue_key] = {
            "venue_name": venues_info[venue_key]["name"],
            "venue_id": venues_info[venue_key]["venue_id"],
            "gentag": compute_group_metrics(gt_venue, f"gentag_{venue_key}"),
            "rake": compute_group_metrics(rk_venue, f"rake_{venue_key}"),
        }

    # --- Separation ---
    gt_pass_rate = gentag_metrics["dir_pass_rate"]["point"]
    rk_pass_rate = rake_metrics["dir_pass_rate"]["point"]
    separation_pp = round((gt_pass_rate - rk_pass_rate) * 100, 1)

    gt_nf_rate = gentag_metrics["non_floor_pass_rate"]["point"]
    rk_nf_rate = rake_metrics["non_floor_pass_rate"]["point"]
    separation_non_floor_pp = round((gt_nf_rate - rk_nf_rate) * 100, 1)

    # Fisher's exact test for separation
    gt_pass = gentag_metrics["dir_passes"]
    gt_fail = gentag_metrics["scorable_units"] - gt_pass
    rk_pass = rake_metrics["dir_passes"]
    rk_fail = rake_metrics["scorable_units"] - rk_pass
    fisher_p = fishers_exact_test(gt_pass, gt_fail, rk_pass, rk_fail)

    # Non-floor Fisher's
    gt_nf_pass = gentag_metrics["non_floor_passes"]
    gt_nf_fail = gentag_metrics["non_floor_scorable"] - gt_nf_pass
    rk_nf_pass = rake_metrics["non_floor_passes"]
    rk_nf_fail = rake_metrics["non_floor_scorable"] - rk_nf_pass
    fisher_p_nf = fishers_exact_test(gt_nf_pass, gt_nf_fail, rk_nf_pass, rk_nf_fail)

    # --- MVP → Scaled progression ---
    mvp_gt = [u for u in gentag_units if venue_from_unit_id(u["unit_id"]) == "coltons"]
    mvp_rk = [u for u in rake_units if venue_from_unit_id(u["unit_id"]) == "coltons"]
    mvp_gt_scorable = [u for u in mvp_gt if u["dir_pass"] is not None]
    mvp_rk_scorable = [u for u in mvp_rk if u["dir_pass"] is not None]
    mvp_gt_pass = sum(1 for u in mvp_gt_scorable if u["dir_pass"])
    mvp_rk_pass = sum(1 for u in mvp_rk_scorable if u["dir_pass"])

    scaled_gt = [u for u in gentag_units if venue_from_unit_id(u["unit_id"]) != "coltons"]
    scaled_rk = [u for u in rake_units if venue_from_unit_id(u["unit_id"]) != "coltons"]
    scaled_gt_scorable = [u for u in scaled_gt if u["dir_pass"] is not None]
    scaled_rk_scorable = [u for u in scaled_rk if u["dir_pass"] is not None]
    scaled_gt_pass = sum(1 for u in scaled_gt_scorable if u["dir_pass"])
    scaled_rk_pass = sum(1 for u in scaled_rk_scorable if u["dir_pass"])

    progression = {
        "mvp": {
            "gentag": f"{mvp_gt_pass}/{len(mvp_gt_scorable)}",
            "gentag_rate": round(mvp_gt_pass / len(mvp_gt_scorable), 4) if mvp_gt_scorable else None,
            "rake": f"{mvp_rk_pass}/{len(mvp_rk_scorable)}",
            "rake_rate": round(mvp_rk_pass / len(mvp_rk_scorable), 4) if mvp_rk_scorable else None,
        },
        "new_venues": {
            "gentag": f"{scaled_gt_pass}/{len(scaled_gt_scorable)}",
            "gentag_rate": round(scaled_gt_pass / len(scaled_gt_scorable), 4) if scaled_gt_scorable else None,
            "rake": f"{scaled_rk_pass}/{len(scaled_rk_scorable)}",
            "rake_rate": round(scaled_rk_pass / len(scaled_rk_scorable), 4) if scaled_rk_scorable else None,
        },
        "combined": {
            "gentag": f"{gt_pass}/{gentag_metrics['scorable_units']}",
            "gentag_rate": gt_pass_rate,
            "rake": f"{rk_pass}/{rake_metrics['scorable_units']}",
            "rake_rate": rk_pass_rate,
        },
    }

    # --- Per-unit detail ---
    per_unit_gentag = []
    for u in gentag_units:
        per_unit_gentag.append({
            "unit_id": u["unit_id"],
            "venue": venue_from_unit_id(u["unit_id"]),
            "persona_id": u["persona_id"],
            "edit_type": u["edit_type"],
            "edited_tag": u["edited_tag"],
            "expected_direction": u["expected_direction"],
            "baseline": u["baseline_decision"],
            "intervention": u["intervention_decision"],
            "dir_pass": u["dir_pass"],
            "is_floor": is_floor(u),
            "placebo_moved": u["placebo_moved"],
        })

    per_unit_rake = []
    for u in rake_units:
        per_unit_rake.append({
            "unit_id": u["unit_id"],
            "venue": venue_from_unit_id(u["unit_id"]),
            "persona_id": u["persona_id"],
            "edit_type": u["edit_type"],
            "edited_tag": u["edited_tag"],
            "expected_direction": u["expected_direction"],
            "baseline": u["baseline_decision"],
            "intervention": u["intervention_decision"],
            "dir_pass": u["dir_pass"],
            "is_floor": is_floor(u),
            "placebo_moved": u["placebo_moved"],
        })

    # --- Build output ---
    output = {
        "experiment": "Phase 4 Scaled DIR",
        "total_gentag_units": len(gentag_units),
        "total_rake_units": len(rake_units),
        "gentag_metrics": gentag_metrics,
        "rake_metrics": rake_metrics,
        "separation": {
            "all_units_pp": separation_pp,
            "non_floor_pp": separation_non_floor_pp,
            "fisher_exact_p_all": fisher_p,
            "fisher_exact_p_non_floor": fisher_p_nf,
        },
        "per_venue": per_venue,
        "progression": progression,
        "per_unit_gentag": per_unit_gentag,
        "per_unit_rake": per_unit_rake,
    }

    # --- Write ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # --- Print report ---
    print("=" * 60)
    print("PHASE 4 SCALED DIR — AGGREGATED RESULTS")
    print("=" * 60)

    print(f"\n--- GENTAG ({gentag_metrics['total_units']} units) ---")
    m = gentag_metrics
    print(f"  DIR pass rate (all):       {m['dir_passes']}/{m['scorable_units']}"
          f" = {m['dir_pass_rate']['point']:.1%}"
          f" [{m['dir_pass_rate']['lower']:.1%}, {m['dir_pass_rate']['upper']:.1%}]")
    print(f"  DIR pass rate (non-floor): {m['non_floor_passes']}/{m['non_floor_scorable']}"
          f" = {m['non_floor_pass_rate']['point']:.1%}"
          f" [{m['non_floor_pass_rate']['lower']:.1%}, {m['non_floor_pass_rate']['upper']:.1%}]")
    print(f"  Floor units: {m['floor_unit_ids']}")
    print(f"  Placebo movement:          {m['placebo_moves']}/{m['placebo_scorable']}"
          f" = {m['placebo_movement_rate']['point']:.1%}")

    print(f"\n--- RAKE ({rake_metrics['total_units']} units) ---")
    m = rake_metrics
    print(f"  DIR pass rate (all):       {m['dir_passes']}/{m['scorable_units']}"
          f" = {m['dir_pass_rate']['point']:.1%}"
          f" [{m['dir_pass_rate']['lower']:.1%}, {m['dir_pass_rate']['upper']:.1%}]")
    print(f"  DIR pass rate (non-floor): {m['non_floor_passes']}/{m['non_floor_scorable']}"
          f" = {m['non_floor_pass_rate']['point']:.1%}"
          f" [{m['non_floor_pass_rate']['lower']:.1%}, {m['non_floor_pass_rate']['upper']:.1%}]")
    print(f"  Floor units: {m['floor_unit_ids']}")
    print(f"  Placebo movement:          {m['placebo_moves']}/{m['placebo_scorable']}"
          f" = {m['placebo_movement_rate']['point']:.1%}")

    print(f"\n--- SEPARATION ---")
    print(f"  All units:      {separation_pp:+.1f}pp (Fisher p={fisher_p})")
    print(f"  Non-floor:      {separation_non_floor_pp:+.1f}pp (Fisher p={fisher_p_nf})")

    print(f"\n--- PER-VENUE ---")
    for vk in ["coltons", "boost", "boru"]:
        v = per_venue[vk]
        gt = v["gentag"]
        rk = v["rake"]
        gt_rate = gt["dir_pass_rate"]["point"]
        rk_rate = rk["dir_pass_rate"]["point"]
        sep = round((gt_rate - rk_rate) * 100, 1)
        print(f"  {v['venue_name']}: "
              f"GT {gt['dir_passes']}/{gt['scorable_units']} ({gt_rate:.0%}) "
              f"RAKE {rk['dir_passes']}/{rk['scorable_units']} ({rk_rate:.0%}) "
              f"sep={sep:+.1f}pp")

    print(f"\n--- PROGRESSION ---")
    p = progression
    print(f"  MVP (Colton's):  GT {p['mvp']['gentag']} ({p['mvp']['gentag_rate']:.0%})"
          f"  RAKE {p['mvp']['rake']} ({p['mvp']['rake_rate']:.0%})")
    print(f"  New venues:      GT {p['new_venues']['gentag']} ({p['new_venues']['gentag_rate']:.0%})"
          f"  RAKE {p['new_venues']['rake']} ({p['new_venues']['rake_rate']:.0%})")
    print(f"  Combined:        GT {p['combined']['gentag']} ({p['combined']['gentag_rate']:.0%})"
          f"  RAKE {p['combined']['rake']} ({p['combined']['rake_rate']:.0%})")

    # --- Per-unit table ---
    print(f"\n--- PER-UNIT GENTAG ---")
    print(f"  {'ID':>8s} {'Venue':>8s} {'P':>3s} {'Edit':>8s} {'Dir':>5s} "
          f"{'Base':>12s} {'Intv':>12s} {'Pass':>5s} {'Floor':>5s} {'Plcb':>5s}")
    for u in per_unit_gentag:
        print(f"  {u['unit_id']:>8s} {u['venue']:>8s} {u['persona_id']:>3s} "
              f"{u['edit_type']:>8s} {u['expected_direction']:>5s} "
              f"{u['baseline']:>12s} {u['intervention']:>12s} "
              f"{'Y' if u['dir_pass'] else 'N':>5s} "
              f"{'Y' if u['is_floor'] else 'N':>5s} "
              f"{'Y' if u['placebo_moved'] else 'N':>5s}")

    print(f"\n--- PER-UNIT RAKE ---")
    print(f"  {'ID':>14s} {'Venue':>8s} {'P':>3s} {'Edit':>8s} {'Dir':>5s} "
          f"{'Base':>12s} {'Intv':>12s} {'Pass':>5s} {'Floor':>5s} {'Plcb':>5s}")
    for u in per_unit_rake:
        print(f"  {u['unit_id']:>14s} {u['venue']:>8s} {u['persona_id']:>3s} "
              f"{u['edit_type']:>8s} {u['expected_direction']:>5s} "
              f"{u['baseline']:>12s} {u['intervention']:>12s} "
              f"{'Y' if u['dir_pass'] else 'N':>5s} "
              f"{'Y' if u['is_floor'] else 'N':>5s} "
              f"{'Y' if u['placebo_moved'] else 'N':>5s}")

    # --- Verdict ---
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    if separation_pp >= 25:
        print(f"  PAPER-READY: Separation = {separation_pp:+.1f}pp >= 25pp threshold")
    elif separation_pp >= 15:
        print(f"  PROMISING: Separation = {separation_pp:+.1f}pp (15-24pp range)")
    else:
        print(f"  FAIL: Separation = {separation_pp:+.1f}pp < 15pp threshold")

    if separation_non_floor_pp >= 25:
        print(f"  NON-FLOOR PAPER-READY: {separation_non_floor_pp:+.1f}pp >= 25pp")
    elif separation_non_floor_pp >= 15:
        print(f"  NON-FLOOR PROMISING: {separation_non_floor_pp:+.1f}pp (15-24pp range)")
    else:
        print(f"  NON-FLOOR FAIL: {separation_non_floor_pp:+.1f}pp < 15pp")

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
