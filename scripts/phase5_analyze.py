"""
Phase 5 — Baseline Legibility Analysis.

Computes all metrics from the baseline legibility study:
  A) Baseline decision distribution (floor rate)
  B) Hard requirement compliance (P2/P3)
  C) FER agreement (exact match, Cohen's kappa, disagreement direction)
  D) Token-budget ablation (truncated gentag vs RAKE)
  E) Floor rate replication (Phase 4 finding)
  F) Decision entropy (distribution skew vs FER reference)

Reuses wilson_ci() and fishers_exact_test() from phase4_aggregate.py.

Usage:
    poetry run python scripts/phase5_analyze.py --summary results/phase5/baseline_summary_XXXXXX.json
    poetry run python scripts/phase5_analyze.py --summary results/phase5/baseline_summary_XXXXXX.json --venues data/phase5/sampled_venues.json
"""

import argparse
import json
from collections import Counter, defaultdict
from math import log2, sqrt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
VENUES_FILE = REPO / "data" / "phase5" / "sampled_venues.json"
OUTPUT_DIR = REPO / "results" / "phase5"

DECISION_ORDINALS = {"REJECT": 0, "BORDERLINE": 1, "RECOMMEND": 2}


# ---------------------------------------------------------------------------
# Reused from phase4_aggregate.py
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


def fishers_exact_test(a, b, c, d):
    """2x2 contingency table Fisher's exact test.
    Table:
        | success | fail |
    row1|    a    |  b   |
    row2|    c    |  d   |
    Returns p-value (two-sided).
    """
    try:
        from scipy.stats import fisher_exact
        table = [[a, b], [c, d]]
        _, p = fisher_exact(table, alternative="two-sided")
        return round(p, 6)
    except ImportError:
        from math import comb
        n = a + b + c + d
        row1 = a + b
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
# Cohen's Kappa
# ---------------------------------------------------------------------------
def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Compute Cohen's kappa for two lists of categorical labels."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    # Observed agreement
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        pa = sum(1 for a in labels_a if a == cat) / n
        pb = sum(1 for b in labels_b if b == cat) / n
        pe += pa * pb

    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


# ---------------------------------------------------------------------------
# Chi-squared test (2xK)
# ---------------------------------------------------------------------------
def chi_squared_test(observed: list, expected: list) -> dict:
    """Simple chi-squared goodness of fit."""
    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e > 0:
            chi2 += (o - e) ** 2 / e
    df = len(observed) - 1
    # Approximate p-value using chi2 CDF (fallback for no scipy)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = 1 - chi2_dist.cdf(chi2, df)
    except ImportError:
        p = None  # Can't compute without scipy
    return {"chi2": round(chi2, 4), "df": df, "p_value": round(p, 6) if p is not None else None}


# ---------------------------------------------------------------------------
# Decision entropy
# ---------------------------------------------------------------------------
def decision_entropy(decisions: list) -> float:
    """Shannon entropy over {REJECT, BORDERLINE, RECOMMEND} distribution.

    Max entropy = log2(3) ≈ 1.585 (uniform).
    Zero entropy = all decisions the same.
    """
    n = len(decisions)
    if n == 0:
        return 0.0
    counts = Counter(decisions)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * log2(p)
    return round(h, 4)


def normalized_entropy(decisions: list) -> float:
    """Entropy normalized to [0, 1] by dividing by log2(3)."""
    max_h = log2(3)
    return round(decision_entropy(decisions) / max_h, 4) if max_h > 0 else 0.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_summary(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def load_venues(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {v["venue_id"]: v for v in data["venues"]}


# ---------------------------------------------------------------------------
# A) Baseline Decision Distribution
# ---------------------------------------------------------------------------
def compute_decision_distribution(summary: list) -> dict:
    """Decision counts per system and per system-persona."""
    results = {}

    for system in ["gentag", "rake", "gentag_truncated", "fer"]:
        sys_rows = [r for r in summary if r["system"] == system]
        total = len(sys_rows)
        counts = Counter(r["decision"] for r in sys_rows)
        rej = counts.get("REJECT", 0)

        results[system] = {
            "total": total,
            "REJECT": rej,
            "BORDERLINE": counts.get("BORDERLINE", 0),
            "RECOMMEND": counts.get("RECOMMEND", 0),
            "UNSCORABLE": counts.get("UNSCORABLE", 0),
            "floor_rate": wilson_ci(rej, total),
        }

        # Per-persona breakdown
        per_persona = {}
        for pid in ["P1", "P2", "P3"]:
            p_rows = [r for r in sys_rows if r["persona_id"] == pid]
            p_total = len(p_rows)
            p_counts = Counter(r["decision"] for r in p_rows)
            p_rej = p_counts.get("REJECT", 0)
            per_persona[pid] = {
                "total": p_total,
                "REJECT": p_rej,
                "BORDERLINE": p_counts.get("BORDERLINE", 0),
                "RECOMMEND": p_counts.get("RECOMMEND", 0),
                "UNSCORABLE": p_counts.get("UNSCORABLE", 0),
                "floor_rate": wilson_ci(p_rej, p_total),
            }
        results[system]["per_persona"] = per_persona

    return results


# ---------------------------------------------------------------------------
# B) Hard Requirement Compliance
# ---------------------------------------------------------------------------
def compute_hard_requirement_compliance(summary: list, venues: dict) -> dict:
    """
    P2: has_game_tag=False → expected REJECT; True → expected NOT-REJECT
    P3: has_speed_tag=False → expected REJECT; True → expected NOT-REJECT
    P1: no hard binary requirement (control)
    """
    results = {}

    for system in ["gentag", "rake", "gentag_truncated", "fer"]:
        sys_rows = [r for r in summary if r["system"] == system]

        # P2 compliance
        p2_rows = [r for r in sys_rows if r["persona_id"] == "P2"]
        p2_correct = 0
        p2_total = 0
        p2_details = []
        for r in p2_rows:
            v = venues.get(r["venue_id"], {})
            has_game = v.get("has_game_tag", False)
            decision = r["decision"]
            if decision == "UNSCORABLE":
                continue
            p2_total += 1
            if not has_game:
                # Expected REJECT
                correct = (decision == "REJECT")
            else:
                # Expected NOT-REJECT
                correct = (decision != "REJECT")
            if correct:
                p2_correct += 1
            p2_details.append({
                "venue_id": r["venue_id"],
                "has_game_tag": has_game,
                "decision": decision,
                "expected": "REJECT" if not has_game else "NOT-REJECT",
                "correct": correct,
            })

        # P3 compliance
        p3_rows = [r for r in sys_rows if r["persona_id"] == "P3"]
        p3_correct = 0
        p3_total = 0
        p3_details = []
        for r in p3_rows:
            v = venues.get(r["venue_id"], {})
            has_speed = v.get("has_speed_tag", False)
            decision = r["decision"]
            if decision == "UNSCORABLE":
                continue
            p3_total += 1
            if not has_speed:
                correct = (decision == "REJECT")
            else:
                correct = (decision != "REJECT")
            if correct:
                p3_correct += 1
            p3_details.append({
                "venue_id": r["venue_id"],
                "has_speed_tag": has_speed,
                "decision": decision,
                "expected": "REJECT" if not has_speed else "NOT-REJECT",
                "correct": correct,
            })

        results[system] = {
            "P2": {
                "correct": p2_correct,
                "total": p2_total,
                "compliance": wilson_ci(p2_correct, p2_total),
                "details": p2_details,
            },
            "P3": {
                "correct": p3_correct,
                "total": p3_total,
                "compliance": wilson_ci(p3_correct, p3_total),
                "details": p3_details,
            },
            "combined": {
                "correct": p2_correct + p3_correct,
                "total": p2_total + p3_total,
                "compliance": wilson_ci(p2_correct + p3_correct, p2_total + p3_total),
            },
        }

    # Fisher's exact: gentag vs rake compliance
    gt_correct = results["gentag"]["combined"]["correct"]
    gt_wrong = results["gentag"]["combined"]["total"] - gt_correct
    rk_correct = results["rake"]["combined"]["correct"]
    rk_wrong = results["rake"]["combined"]["total"] - rk_correct
    results["fisher_gentag_vs_rake"] = fishers_exact_test(
        gt_correct, gt_wrong, rk_correct, rk_wrong)

    return results


# ---------------------------------------------------------------------------
# C) FER Agreement
# ---------------------------------------------------------------------------
def compute_fer_agreement(summary: list) -> dict:
    """Compare each system's decisions against FER decisions."""
    # Build FER lookup: (venue_id, persona_id) -> decision
    fer_decisions = {}
    for r in summary:
        if r["system"] == "fer":
            fer_decisions[(r["venue_id"], r["persona_id"])] = r["decision"]

    results = {}
    for system in ["gentag", "rake", "gentag_truncated"]:
        sys_rows = [r for r in summary if r["system"] == system]

        matches = 0
        total = 0
        sys_labels = []
        fer_labels = []
        upgrades = 0  # system decision > FER decision
        downgrades = 0  # system decision < FER decision
        details = []

        for r in sys_rows:
            key = (r["venue_id"], r["persona_id"])
            fer_dec = fer_decisions.get(key)
            sys_dec = r["decision"]

            if fer_dec is None or fer_dec == "UNSCORABLE" or sys_dec == "UNSCORABLE":
                continue

            total += 1
            sys_labels.append(sys_dec)
            fer_labels.append(fer_dec)

            if sys_dec == fer_dec:
                matches += 1
                direction = "match"
            else:
                sys_ord = DECISION_ORDINALS.get(sys_dec, -1)
                fer_ord = DECISION_ORDINALS.get(fer_dec, -1)
                if sys_ord > fer_ord:
                    upgrades += 1
                    direction = "upgrade"
                else:
                    downgrades += 1
                    direction = "downgrade"

            details.append({
                "venue_id": r["venue_id"],
                "persona_id": r["persona_id"],
                "system_decision": sys_dec,
                "fer_decision": fer_dec,
                "direction": direction,
            })

        kappa = cohens_kappa(sys_labels, fer_labels) if total > 0 else None

        results[system] = {
            "matches": matches,
            "total": total,
            "agreement_rate": wilson_ci(matches, total),
            "kappa": kappa,
            "upgrades": upgrades,
            "downgrades": downgrades,
            "upgrade_rate": wilson_ci(upgrades, total),
            "downgrade_rate": wilson_ci(downgrades, total),
            "details": details,
        }

    # Fisher's exact: gentag agreement vs rake agreement
    gt_match = results["gentag"]["matches"]
    gt_miss = results["gentag"]["total"] - gt_match
    rk_match = results["rake"]["matches"]
    rk_miss = results["rake"]["total"] - rk_match
    results["fisher_gentag_vs_rake"] = fishers_exact_test(
        gt_match, gt_miss, rk_match, rk_miss)

    return results


# ---------------------------------------------------------------------------
# D) Token-Budget Ablation
# ---------------------------------------------------------------------------
def compute_ablation(summary: list) -> dict:
    """Compare gentag_truncated vs RAKE floor rates."""
    trunc_rows = [r for r in summary if r["system"] == "gentag_truncated"]
    rake_rows = [r for r in summary if r["system"] == "rake"]

    trunc_rej = sum(1 for r in trunc_rows if r["decision"] == "REJECT")
    rake_rej = sum(1 for r in rake_rows if r["decision"] == "REJECT")
    trunc_total = len(trunc_rows)
    rake_total = len(rake_rows)

    trunc_rate = wilson_ci(trunc_rej, trunc_total)
    rake_rate = wilson_ci(rake_rej, rake_total)

    gap_pp = round((trunc_rate["point"] - rake_rate["point"]) * 100, 1)
    fisher_p = fishers_exact_test(trunc_rej, trunc_total - trunc_rej,
                                   rake_rej, rake_total - rake_rej)

    # Per-persona
    per_persona = {}
    for pid in ["P1", "P2", "P3"]:
        t_rows = [r for r in trunc_rows if r["persona_id"] == pid]
        r_rows = [r for r in rake_rows if r["persona_id"] == pid]
        t_rej = sum(1 for r in t_rows if r["decision"] == "REJECT")
        r_rej = sum(1 for r in r_rows if r["decision"] == "REJECT")
        per_persona[pid] = {
            "truncated_floor": wilson_ci(t_rej, len(t_rows)),
            "rake_floor": wilson_ci(r_rej, len(r_rows)),
            "gap_pp": round((t_rej/len(t_rows) - r_rej/len(r_rows)) * 100, 1)
                      if t_rows and r_rows else None,
        }

    return {
        "truncated_floor_rate": trunc_rate,
        "rake_floor_rate": rake_rate,
        "gap_pp": gap_pp,
        "fisher_p": fisher_p,
        "interpretation": (
            "truncated gentag still beats RAKE → advantage is semantics, not volume"
            if gap_pp < 0 else
            "truncated gentag ≈ RAKE → advantage may be volume, not semantics"
            if abs(gap_pp) < 5 else
            "truncated gentag worse than RAKE → unexpected"
        ),
        "per_persona": per_persona,
    }


# ---------------------------------------------------------------------------
# E) Floor Rate Replication
# ---------------------------------------------------------------------------
def compute_floor_replication(summary: list) -> dict:
    """Does Phase 4 finding replicate: gentag floor < RAKE floor?"""
    gentag_rows = [r for r in summary if r["system"] == "gentag"]
    rake_rows = [r for r in summary if r["system"] == "rake"]

    gt_rej = sum(1 for r in gentag_rows if r["decision"] == "REJECT")
    rk_rej = sum(1 for r in rake_rows if r["decision"] == "REJECT")
    gt_total = len(gentag_rows)
    rk_total = len(rake_rows)

    gt_rate = wilson_ci(gt_rej, gt_total)
    rk_rate = wilson_ci(rk_rej, rk_total)

    gap_pp = round((gt_rate["point"] - rk_rate["point"]) * 100, 1)
    fisher_p = fishers_exact_test(gt_rej, gt_total - gt_rej,
                                   rk_rej, rk_total - rk_rej)

    # Verdict
    abs_gap = abs(gap_pp)
    if gap_pp < -15:
        verdict = "PAPER-READY"
    elif gap_pp < -5:
        verdict = "PROMISING"
    elif gap_pp < 5:
        verdict = "FAIL (no significant gap)"
    else:
        verdict = "FAIL (reversed)"

    return {
        "gentag_floor_rate": gt_rate,
        "rake_floor_rate": rk_rate,
        "gap_pp": gap_pp,
        "fisher_p": fisher_p,
        "phase4_reference": {
            "gentag_floor": "12.5% (2/16)",
            "rake_floor": "37.5% (6/16)",
            "gap_pp": -25.0,
        },
        "replicates": gap_pp < -5,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# F) Decision Entropy
# ---------------------------------------------------------------------------
def compute_decision_entropy(summary: list) -> dict:
    """Decision entropy per system — measures distribution skew.

    If RAKE produces extreme skew (e.g., 60% REJECT) while gentags produce
    a balanced distribution similar to FER, that reinforces "semantic legibility."
    Closer to FER entropy = better representation fidelity.
    """
    results = {}

    for system in ["gentag", "rake", "gentag_truncated", "fer"]:
        sys_rows = [r for r in summary if r["system"] == system]
        decisions = [r["decision"] for r in sys_rows if r["decision"] != "UNSCORABLE"]
        counts = Counter(decisions)
        total = len(decisions)

        results[system] = {
            "entropy": decision_entropy(decisions),
            "normalized_entropy": normalized_entropy(decisions),
            "distribution": {
                "REJECT": round(counts.get("REJECT", 0) / total, 4) if total else 0,
                "BORDERLINE": round(counts.get("BORDERLINE", 0) / total, 4) if total else 0,
                "RECOMMEND": round(counts.get("RECOMMEND", 0) / total, 4) if total else 0,
            },
            "n": total,
        }

        # Per-persona entropy
        per_persona = {}
        for pid in ["P1", "P2", "P3"]:
            p_rows = [r for r in sys_rows if r["persona_id"] == pid]
            p_decisions = [r["decision"] for r in p_rows if r["decision"] != "UNSCORABLE"]
            per_persona[pid] = {
                "entropy": decision_entropy(p_decisions),
                "normalized_entropy": normalized_entropy(p_decisions),
            }
        results[system]["per_persona"] = per_persona

    # Distance from FER distribution (L1 norm)
    fer_dist = results["fer"]["distribution"]
    for system in ["gentag", "rake", "gentag_truncated"]:
        sys_dist = results[system]["distribution"]
        l1 = sum(abs(sys_dist[d] - fer_dist[d]) for d in ["REJECT", "BORDERLINE", "RECOMMEND"])
        results[system]["l1_distance_from_fer"] = round(l1, 4)
        results[system]["entropy_gap_from_fer"] = round(
            results[system]["entropy"] - results["fer"]["entropy"], 4)

    return results


# ---------------------------------------------------------------------------
# Per-venue breakdown
# ---------------------------------------------------------------------------
def compute_per_venue_breakdown(summary: list, venues: dict) -> list:
    """One row per venue with all system decisions by persona."""
    venue_data = defaultdict(dict)

    for r in summary:
        vid = r["venue_id"]
        key = f"{r['persona_id']}_{r['system']}"
        venue_data[vid][key] = r["decision"]
        venue_data[vid]["venue_name"] = r.get("venue_name", "")

    breakdown = []
    for vid, data in venue_data.items():
        v = venues.get(vid, {})
        row = {
            "venue_id": vid,
            "venue_name": data.get("venue_name", ""),
            "stratum": v.get("stratum", ""),
            "has_game_tag": v.get("has_game_tag", False),
            "has_speed_tag": v.get("has_speed_tag", False),
            "gentag_count": v.get("gentag_count", 0),
            "rake_count": v.get("rake_count", 0),
        }
        for pid in ["P1", "P2", "P3"]:
            for sys in ["gentag", "rake", "gentag_truncated", "fer"]:
                row[f"{pid}_{sys}"] = data.get(f"{pid}_{sys}", None)
        breakdown.append(row)

    return sorted(breakdown, key=lambda x: x["venue_id"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 5 Baseline Legibility Analysis")
    parser.add_argument("--summary", required=True,
                        help="Path to baseline_summary_XXXXX.json")
    parser.add_argument("--venues", default=str(VENUES_FILE),
                        help="Path to sampled_venues.json")
    parser.add_argument("--output", default=None,
                        help="Output path (default: results/phase5/baseline_legibility_analysis.json)")
    args = parser.parse_args()

    print("Loading data...")
    summary = load_summary(args.summary)
    venues = load_venues(args.venues)
    print(f"  {len(summary)} summary rows, {len(venues)} venues")

    # Compute all metrics
    print("\nA) Decision distribution...")
    decision_dist = compute_decision_distribution(summary)

    print("B) Hard requirement compliance...")
    compliance = compute_hard_requirement_compliance(summary, venues)

    print("C) FER agreement...")
    fer_agreement = compute_fer_agreement(summary)

    print("D) Token-budget ablation...")
    ablation = compute_ablation(summary)

    print("E) Floor rate replication...")
    floor_repl = compute_floor_replication(summary)

    print("F) Decision entropy...")
    entropy = compute_decision_entropy(summary)

    print("Per-venue breakdown...")
    per_venue = compute_per_venue_breakdown(summary, venues)

    # Build output (strip details for top-level, keep in separate section)
    compliance_summary = {}
    for sys in ["gentag", "rake", "gentag_truncated", "fer"]:
        compliance_summary[sys] = {
            "P2": {k: v for k, v in compliance[sys]["P2"].items() if k != "details"},
            "P3": {k: v for k, v in compliance[sys]["P3"].items() if k != "details"},
            "combined": compliance[sys]["combined"],
        }
    compliance_summary["fisher_gentag_vs_rake"] = compliance["fisher_gentag_vs_rake"]

    fer_summary = {}
    for sys in ["gentag", "rake", "gentag_truncated"]:
        fer_summary[sys] = {k: v for k, v in fer_agreement[sys].items() if k != "details"}
    fer_summary["fisher_gentag_vs_rake"] = fer_agreement["fisher_gentag_vs_rake"]

    output = {
        "experiment": "Phase 5 — Baseline Legibility Analysis",
        "summary_file": str(args.summary),
        "n_venues": len(venues),
        "n_summary_rows": len(summary),
        "A_decision_distribution": decision_dist,
        "B_hard_requirement_compliance": compliance_summary,
        "C_fer_agreement": fer_summary,
        "D_token_budget_ablation": ablation,
        "E_floor_rate_replication": floor_repl,
        "F_decision_entropy": entropy,
        "per_venue_breakdown": per_venue,
    }

    out_path = Path(args.output) if args.output else (OUTPUT_DIR / "baseline_legibility_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE 5 — BASELINE LEGIBILITY ANALYSIS")
    print(f"{'='*70}")

    # A) Decision Distribution
    print(f"\n--- A) DECISION DISTRIBUTION (Floor Rate = Baseline REJECT Rate) ---")
    print(f"  {'System':>20s} | {'REJECT':>8s} {'BORDER':>8s} {'RECOM':>8s} | {'Floor Rate':>12s} {'95% CI':>18s}")
    print(f"  {'-'*20}-+-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*12}-{'-'*18}")
    for sys in ["gentag", "rake", "gentag_truncated", "fer"]:
        d = decision_dist[sys]
        fr = d["floor_rate"]
        print(f"  {sys:>20s} | {d['REJECT']:>8d} {d['BORDERLINE']:>8d} {d['RECOMMEND']:>8d} | "
              f"{fr['point']:>11.1%} [{fr['lower']:.1%}, {fr['upper']:.1%}]")

    # B) Hard Requirement Compliance
    print(f"\n--- B) HARD REQUIREMENT COMPLIANCE ---")
    for pid in ["P2", "P3"]:
        print(f"\n  {pid}:")
        print(f"  {'System':>20s} | {'Correct':>8s} {'Total':>6s} | {'Compliance':>12s} {'95% CI':>18s}")
        print(f"  {'-'*20}-+-{'-'*8}-{'-'*6}-+-{'-'*12}-{'-'*18}")
        for sys in ["gentag", "rake", "gentag_truncated", "fer"]:
            c = compliance_summary[sys][pid]
            cr = c["compliance"]
            print(f"  {sys:>20s} | {c['correct']:>8d} {c['total']:>6d} | "
                  f"{cr['point']:>11.1%} [{cr['lower']:.1%}, {cr['upper']:.1%}]")
    print(f"\n  Fisher's exact (gentag vs rake combined): p={compliance_summary['fisher_gentag_vs_rake']}")

    # C) FER Agreement
    print(f"\n--- C) FER AGREEMENT ---")
    print(f"  {'System':>20s} | {'Match':>6s} {'Total':>6s} | {'Agreement':>10s} {'Kappa':>8s} | {'Upgrades':>9s} {'Downgrades':>11s}")
    print(f"  {'-'*20}-+-{'-'*6}-{'-'*6}-+-{'-'*10}-{'-'*8}-+-{'-'*9}-{'-'*11}")
    for sys in ["gentag", "rake", "gentag_truncated"]:
        f = fer_summary[sys]
        ar = f["agreement_rate"]
        print(f"  {sys:>20s} | {f['matches']:>6d} {f['total']:>6d} | "
              f"{ar['point']:>9.1%} {f['kappa']:>8.3f} | "
              f"{f['upgrades']:>9d} {f['downgrades']:>11d}")
    print(f"  Fisher's exact (gentag vs rake agreement): p={fer_summary['fisher_gentag_vs_rake']}")

    # D) Token-Budget Ablation
    print(f"\n--- D) TOKEN-BUDGET ABLATION ---")
    ab = ablation
    print(f"  Truncated gentag floor: {ab['truncated_floor_rate']['point']:.1%}")
    print(f"  RAKE floor:             {ab['rake_floor_rate']['point']:.1%}")
    print(f"  Gap:                    {ab['gap_pp']:+.1f}pp (Fisher p={ab['fisher_p']})")
    print(f"  Interpretation:         {ab['interpretation']}")

    # E) Floor Rate Replication
    print(f"\n--- E) FLOOR RATE REPLICATION ---")
    fl = floor_repl
    print(f"  Phase 4: gentag {fl['phase4_reference']['gentag_floor']} "
          f"vs RAKE {fl['phase4_reference']['rake_floor']} "
          f"(gap={fl['phase4_reference']['gap_pp']:+.1f}pp)")
    print(f"  Phase 5: gentag {fl['gentag_floor_rate']['point']:.1%} "
          f"({fl['gentag_floor_rate']['k']}/{fl['gentag_floor_rate']['n']}) "
          f"vs RAKE {fl['rake_floor_rate']['point']:.1%} "
          f"({fl['rake_floor_rate']['k']}/{fl['rake_floor_rate']['n']}) "
          f"(gap={fl['gap_pp']:+.1f}pp)")
    print(f"  Fisher p={fl['fisher_p']}")
    print(f"  Verdict: {fl['verdict']}")
    print(f"  Replicates: {'YES' if fl['replicates'] else 'NO'}")

    # F) Decision Entropy
    print(f"\n--- F) DECISION ENTROPY ---")
    print(f"  {'System':>20s} | {'H(bits)':>8s} {'H_norm':>7s} | {'P(REJ)':>7s} {'P(BOR)':>7s} {'P(REC)':>7s} | {'L1 vs FER':>10s}")
    print(f"  {'-'*20}-+-{'-'*8}-{'-'*7}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*10}")
    for sys in ["gentag", "rake", "gentag_truncated", "fer"]:
        e = entropy[sys]
        d = e["distribution"]
        l1 = e.get("l1_distance_from_fer", "—")
        l1_str = f"{l1:.3f}" if isinstance(l1, float) else l1
        print(f"  {sys:>20s} | {e['entropy']:>8.3f} {e['normalized_entropy']:>7.3f} | "
              f"{d['REJECT']:>7.1%} {d['BORDERLINE']:>7.1%} {d['RECOMMEND']:>7.1%} | "
              f"{l1_str:>10s}")
    print(f"\n  Interpretation: closer to FER entropy = better representation fidelity")
    print(f"  If RAKE has extreme REJECT skew while gentag matches FER distribution,")
    print(f"  this visually reinforces semantic legibility.")

    # Verdict summary
    print(f"\n{'='*70}")
    print("VERDICT SUMMARY")
    print(f"{'='*70}")

    # Floor rate
    fr_gap = fl["gap_pp"]
    if fr_gap <= -15:
        print(f"  Floor rate gap:        PAPER-READY ({fr_gap:+.1f}pp)")
    elif fr_gap <= -5:
        print(f"  Floor rate gap:        PROMISING ({fr_gap:+.1f}pp)")
    else:
        print(f"  Floor rate gap:        FAIL ({fr_gap:+.1f}pp)")

    # Compliance
    gt_comp = compliance_summary["gentag"]["combined"]["compliance"]["point"]
    rk_comp = compliance_summary["rake"]["combined"]["compliance"]["point"]
    print(f"  Compliance (gentag):   {gt_comp:.0%}")
    print(f"  Compliance (RAKE):     {rk_comp:.0%}")
    if gt_comp >= 0.85 and rk_comp < 0.70:
        print(f"  Compliance verdict:    PAPER-READY")
    elif gt_comp >= 0.75:
        print(f"  Compliance verdict:    PROMISING")
    else:
        print(f"  Compliance verdict:    FAIL")

    # FER agreement
    gt_agree = fer_summary["gentag"]["agreement_rate"]["point"]
    rk_agree = fer_summary["rake"]["agreement_rate"]["point"]
    agree_gap = round((gt_agree - rk_agree) * 100, 1)
    print(f"  FER agreement (gentag): {gt_agree:.0%}")
    print(f"  FER agreement (RAKE):   {rk_agree:.0%}")
    if agree_gap >= 10:
        print(f"  FER agreement verdict:  PAPER-READY ({agree_gap:+.1f}pp)")
    elif agree_gap >= 5:
        print(f"  FER agreement verdict:  PROMISING ({agree_gap:+.1f}pp)")
    else:
        print(f"  FER agreement verdict:  FAIL ({agree_gap:+.1f}pp)")

    # Ablation
    ab_gap = ablation["gap_pp"]
    if ab_gap < -5:
        print(f"  Ablation verdict:      PASS (truncated still > RAKE, gap={ab_gap:+.1f}pp)")
    elif abs(ab_gap) <= 5:
        print(f"  Ablation verdict:      MIXED (gap={ab_gap:+.1f}pp)")
    else:
        print(f"  Ablation verdict:      FAIL (gap={ab_gap:+.1f}pp)")

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
