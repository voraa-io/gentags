"""
Phase 5 — Baseline Legibility Analysis (v2).

Computes all metrics from the baseline legibility study:
  A) Baseline decision distribution (floor rate) — 6 systems x 4 personas
  B) Hard requirement compliance (P1/P2/P3 with frozen indicator lexicons)
  C) FER agreement (exact match, Cohen's kappa, disagreement direction)
  D) Token-budget ablation (truncated gentag vs all keyword baselines)
  E) Decision entropy (distribution skew vs FER reference)
  F) Cross-judge comparison (if two summary files provided)

Usage:
    poetry run python scripts/phase5_analyze.py --summary results/phase5/baseline_summary_openai_XXXXX.json
    poetry run python scripts/phase5_analyze.py --summary results/phase5/baseline_summary_openai_XXXXX.json --summary2 results/phase5/baseline_summary_claude_XXXXX.json
"""

import argparse
import json
from collections import Counter, defaultdict
from math import log2, sqrt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
VENUES_FILE = REPO / "data" / "phase5" / "sampled_venues.json"
PERSONAS_FILE = REPO / "data" / "phase5" / "phase5_personas.json"
OUTPUT_DIR = REPO / "results" / "phase5"

DECISION_ORDINALS = {"REJECT": 0, "BORDERLINE": 1, "RECOMMEND": 2}
TAG_SYSTEMS = ["gentag", "rake", "yake", "tfidf", "gentag_truncated"]
ALL_SYSTEMS = TAG_SYSTEMS + ["fer"]
HARD_PERSONAS = ["P1", "P2", "P3"]
ALL_PERSONAS = ["P1", "P2", "P3", "P4"]


# ---------------------------------------------------------------------------
# Stats functions
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
        "n": total, "k": successes,
    }


def fishers_exact_test(a, b, c, d):
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
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
        p_value = sum(hypergeom_pmf(k)
                      for k in range(max(0, row1 - col2), min(row1, col1) + 1)
                      if hypergeom_pmf(k) <= p_obs + 1e-10)
        return round(min(p_value, 1.0), 6)


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    categories = sorted(set(labels_a) | set(labels_b))
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    pe = sum((sum(1 for a in labels_a if a == c) / n) *
             (sum(1 for b in labels_b if b == c) / n) for c in categories)
    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def decision_entropy(decisions: list) -> float:
    n = len(decisions)
    if n == 0:
        return 0.0
    counts = Counter(decisions)
    return round(-sum((c/n) * log2(c/n) for c in counts.values() if c > 0), 4)


def normalized_entropy(decisions: list) -> float:
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

def load_personas(path: str) -> dict:
    with open(path) as f:
        return {p["persona_id"]: p for p in json.load(f)}


# ---------------------------------------------------------------------------
# A) Decision Distribution
# ---------------------------------------------------------------------------
def compute_decision_distribution(summary: list) -> dict:
    results = {}
    systems = sorted(set(r["system"] for r in summary))
    personas = sorted(set(r["persona_id"] for r in summary))

    for system in systems:
        sys_rows = [r for r in summary if r["system"] == system]
        total = len(sys_rows)
        counts = Counter(r["decision"] for r in sys_rows)
        rej = counts.get("REJECT", 0)

        per_persona = {}
        for pid in personas:
            p_rows = [r for r in sys_rows if r["persona_id"] == pid]
            p_total = len(p_rows)
            p_counts = Counter(r["decision"] for r in p_rows)
            p_rej = p_counts.get("REJECT", 0)
            per_persona[pid] = {
                "total": p_total, "REJECT": p_rej,
                "BORDERLINE": p_counts.get("BORDERLINE", 0),
                "RECOMMEND": p_counts.get("RECOMMEND", 0),
                "UNSCORABLE": p_counts.get("UNSCORABLE", 0),
                "floor_rate": wilson_ci(p_rej, p_total),
            }

        results[system] = {
            "total": total, "REJECT": rej,
            "BORDERLINE": counts.get("BORDERLINE", 0),
            "RECOMMEND": counts.get("RECOMMEND", 0),
            "UNSCORABLE": counts.get("UNSCORABLE", 0),
            "floor_rate": wilson_ci(rej, total),
            "per_persona": per_persona,
        }
    return results


# ---------------------------------------------------------------------------
# B) Hard Requirement Compliance
# ---------------------------------------------------------------------------
def compute_hard_requirement_compliance(summary: list, venues: dict, personas: dict) -> dict:
    results = {}
    systems = sorted(set(r["system"] for r in summary))

    for system in systems:
        sys_rows = [r for r in summary if r["system"] == system]
        system_results = {}

        for pid in HARD_PERSONAS:
            persona = personas.get(pid, {})
            rule = persona.get("requirement_rule", "")
            p_rows = [r for r in sys_rows if r["persona_id"] == pid]

            correct = 0
            total = 0

            for r in p_rows:
                v = venues.get(r["venue_id"], {})
                decision = r["decision"]
                if decision == "UNSCORABLE":
                    continue
                total += 1

                if pid == "P1":
                    has_neg = v.get("has_negative_food_tag", False)
                    if has_neg:
                        ok = (decision == "REJECT")
                    else:
                        ok = True  # no negative → any decision acceptable
                elif pid == "P2":
                    has_game = v.get("has_game_tag", False)
                    if has_game:
                        ok = (decision != "REJECT")
                    else:
                        ok = (decision == "REJECT")
                elif pid == "P3":
                    has_speed = v.get("has_speed_tag", False)
                    if has_speed:
                        ok = (decision != "REJECT")
                    else:
                        ok = (decision == "REJECT")
                else:
                    continue

                if ok:
                    correct += 1

            system_results[pid] = {
                "correct": correct,
                "total": total,
                "compliance": wilson_ci(correct, total),
            }

        # Combined P1+P2+P3
        combined_correct = sum(system_results[p]["correct"] for p in HARD_PERSONAS)
        combined_total = sum(system_results[p]["total"] for p in HARD_PERSONAS)
        system_results["combined"] = {
            "correct": combined_correct,
            "total": combined_total,
            "compliance": wilson_ci(combined_correct, combined_total),
        }

        results[system] = system_results

    # Fisher's: gentag vs each baseline
    gt = results.get("gentag", {}).get("combined", {})
    for baseline in ["rake", "yake", "tfidf"]:
        bl = results.get(baseline, {}).get("combined", {})
        if gt and bl:
            results[f"fisher_gentag_vs_{baseline}"] = fishers_exact_test(
                gt["correct"], gt["total"] - gt["correct"],
                bl["correct"], bl["total"] - bl["correct"])

    return results


# ---------------------------------------------------------------------------
# C) FER Agreement
# ---------------------------------------------------------------------------
def compute_fer_agreement(summary: list) -> dict:
    fer_decisions = {}
    for r in summary:
        if r["system"] == "fer":
            fer_decisions[(r["venue_id"], r["persona_id"])] = r["decision"]

    results = {}
    for system in TAG_SYSTEMS:
        sys_rows = [r for r in summary if r["system"] == system]
        matches = total = upgrades = downgrades = 0
        sys_labels = []
        fer_labels = []

        for r in sys_rows:
            fer_dec = fer_decisions.get((r["venue_id"], r["persona_id"]))
            sys_dec = r["decision"]
            if not fer_dec or fer_dec == "UNSCORABLE" or sys_dec == "UNSCORABLE":
                continue
            total += 1
            sys_labels.append(sys_dec)
            fer_labels.append(fer_dec)
            if sys_dec == fer_dec:
                matches += 1
            elif DECISION_ORDINALS.get(sys_dec, -1) > DECISION_ORDINALS.get(fer_dec, -1):
                upgrades += 1
            else:
                downgrades += 1

        kappa = cohens_kappa(sys_labels, fer_labels) if total > 0 else None
        results[system] = {
            "matches": matches, "total": total,
            "agreement_rate": wilson_ci(matches, total),
            "kappa": kappa,
            "upgrades": upgrades, "downgrades": downgrades,
            "upgrade_rate": wilson_ci(upgrades, total),
            "downgrade_rate": wilson_ci(downgrades, total),
        }

    # Fisher's: gentag vs each keyword baseline
    for baseline in ["rake", "yake", "tfidf"]:
        gt = results.get("gentag", {})
        bl = results.get(baseline, {})
        if gt and bl:
            results[f"fisher_gentag_vs_{baseline}"] = fishers_exact_test(
                gt["matches"], gt["total"] - gt["matches"],
                bl["matches"], bl["total"] - bl["matches"])

    return results


# ---------------------------------------------------------------------------
# D) Token-Budget Ablation
# ---------------------------------------------------------------------------
def compute_ablation(summary: list) -> dict:
    results = {}
    trunc_rows = [r for r in summary if r["system"] == "gentag_truncated"]
    trunc_rej = sum(1 for r in trunc_rows if r["decision"] == "REJECT")
    trunc_rate = wilson_ci(trunc_rej, len(trunc_rows))

    for baseline in ["rake", "yake", "tfidf"]:
        bl_rows = [r for r in summary if r["system"] == baseline]
        bl_rej = sum(1 for r in bl_rows if r["decision"] == "REJECT")
        bl_rate = wilson_ci(bl_rej, len(bl_rows))
        gap = round((trunc_rate["point"] - bl_rate["point"]) * 100, 1)
        fisher_p = fishers_exact_test(trunc_rej, len(trunc_rows) - trunc_rej,
                                       bl_rej, len(bl_rows) - bl_rej)
        results[baseline] = {
            "truncated_floor": trunc_rate,
            "baseline_floor": bl_rate,
            "gap_pp": gap,
            "fisher_p": fisher_p,
        }

    return results


# ---------------------------------------------------------------------------
# E) Decision Entropy
# ---------------------------------------------------------------------------
def compute_decision_entropy(summary: list) -> dict:
    results = {}
    systems = sorted(set(r["system"] for r in summary))

    for system in systems:
        sys_rows = [r for r in summary if r["system"] == system]
        decisions = [r["decision"] for r in sys_rows if r["decision"] != "UNSCORABLE"]
        counts = Counter(decisions)
        total = len(decisions)

        results[system] = {
            "entropy": decision_entropy(decisions),
            "normalized_entropy": normalized_entropy(decisions),
            "distribution": {d: round(counts.get(d, 0) / total, 4) if total else 0
                             for d in ["REJECT", "BORDERLINE", "RECOMMEND"]},
            "n": total,
        }

    # L1 distance from FER
    if "fer" in results:
        fer_dist = results["fer"]["distribution"]
        for system in TAG_SYSTEMS:
            if system in results:
                sys_dist = results[system]["distribution"]
                l1 = sum(abs(sys_dist[d] - fer_dist[d])
                         for d in ["REJECT", "BORDERLINE", "RECOMMEND"])
                results[system]["l1_distance_from_fer"] = round(l1, 4)

    return results


# ---------------------------------------------------------------------------
# F) Cross-Judge Comparison
# ---------------------------------------------------------------------------
def compute_cross_judge(summary1: list, summary2: list, label1: str, label2: str) -> dict:
    # Build lookups: (venue_id, persona_id, system) -> decision
    lookup1 = {(r["venue_id"], r["persona_id"], r["system"]): r["decision"]
               for r in summary1}
    lookup2 = {(r["venue_id"], r["persona_id"], r["system"]): r["decision"]
               for r in summary2}

    results = {}
    systems = sorted(set(r["system"] for r in summary1))

    for system in systems:
        matches = total = 0
        labels_1 = []
        labels_2 = []

        for key, dec1 in lookup1.items():
            if key[2] != system:
                continue
            dec2 = lookup2.get(key)
            if not dec2 or dec1 == "UNSCORABLE" or dec2 == "UNSCORABLE":
                continue
            total += 1
            labels_1.append(dec1)
            labels_2.append(dec2)
            if dec1 == dec2:
                matches += 1

        kappa = cohens_kappa(labels_1, labels_2) if total > 0 else None
        results[system] = {
            "matches": matches, "total": total,
            "agreement_rate": wilson_ci(matches, total),
            "kappa": kappa,
        }

    # Overall
    all_1 = []
    all_2 = []
    for key, dec1 in lookup1.items():
        dec2 = lookup2.get(key)
        if dec2 and dec1 != "UNSCORABLE" and dec2 != "UNSCORABLE":
            all_1.append(dec1)
            all_2.append(dec2)
    overall_matches = sum(1 for a, b in zip(all_1, all_2) if a == b)
    results["overall"] = {
        "matches": overall_matches, "total": len(all_1),
        "agreement_rate": wilson_ci(overall_matches, len(all_1)),
        "kappa": cohens_kappa(all_1, all_2) if all_1 else None,
    }

    return {
        "judges": [label1, label2],
        "per_system": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 5 Baseline Legibility Analysis (v2)")
    parser.add_argument("--summary", required=True, help="Primary summary JSON")
    parser.add_argument("--summary2", default=None,
                        help="Second judge summary JSON (for cross-judge comparison)")
    parser.add_argument("--venues", default=str(VENUES_FILE))
    parser.add_argument("--personas", default=str(PERSONAS_FILE))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("Loading data...")
    summary = load_summary(args.summary)
    venues = load_venues(args.venues)
    personas = load_personas(args.personas)
    print(f"  {len(summary)} rows, {len(venues)} venues, {len(personas)} personas")

    systems_in_data = sorted(set(r["system"] for r in summary))
    personas_in_data = sorted(set(r["persona_id"] for r in summary))
    print(f"  Systems: {systems_in_data}")
    print(f"  Personas: {personas_in_data}")

    # Compute metrics
    print("\nA) Decision distribution...")
    decision_dist = compute_decision_distribution(summary)

    print("B) Hard requirement compliance...")
    compliance = compute_hard_requirement_compliance(summary, venues, personas)

    print("C) FER agreement...")
    fer_agreement = compute_fer_agreement(summary)

    print("D) Token-budget ablation...")
    ablation = compute_ablation(summary)

    print("E) Decision entropy...")
    entropy = compute_decision_entropy(summary)

    cross_judge = None
    if args.summary2:
        print("F) Cross-judge comparison...")
        summary2 = load_summary(args.summary2)
        cross_judge = compute_cross_judge(summary, summary2, "judge1", "judge2")

    # Build output
    output = {
        "experiment": "Phase 5 — Baseline Legibility Analysis (v2)",
        "summary_file": str(args.summary),
        "n_venues": len(venues),
        "n_summary_rows": len(summary),
        "systems": systems_in_data,
        "personas": personas_in_data,
        "A_decision_distribution": decision_dist,
        "B_hard_requirement_compliance": compliance,
        "C_fer_agreement": fer_agreement,
        "D_token_budget_ablation": ablation,
        "E_decision_entropy": entropy,
    }
    if cross_judge:
        output["F_cross_judge"] = cross_judge

    out_path = Path(args.output) if args.output else (OUTPUT_DIR / "baseline_legibility_analysis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print(f"\n{'='*75}")
    print("PHASE 5 — BASELINE LEGIBILITY ANALYSIS (v2)")
    print(f"{'='*75}")

    # A) Decision Distribution
    print(f"\n--- A) DECISION DISTRIBUTION ---")
    print(f"  {'System':>18s} | {'REJ':>5s} {'BOR':>5s} {'REC':>5s} | {'Floor':>7s} {'95% CI':>18s}")
    print(f"  {'-'*18}-+-{'-'*5}-{'-'*5}-{'-'*5}-+-{'-'*7}-{'-'*18}")
    for sys in systems_in_data:
        d = decision_dist[sys]
        fr = d["floor_rate"]
        print(f"  {sys:>18s} | {d['REJECT']:>5d} {d['BORDERLINE']:>5d} {d['RECOMMEND']:>5d} | "
              f"{fr['point']:>6.1%} [{fr['lower']:.1%}, {fr['upper']:.1%}]")

    # B) Compliance
    print(f"\n--- B) HARD REQUIREMENT COMPLIANCE ---")
    for pid in HARD_PERSONAS:
        if pid not in personas_in_data:
            continue
        print(f"\n  {pid} ({personas.get(pid, {}).get('name', '')}):")
        print(f"  {'System':>18s} | {'OK':>4s} {'N':>4s} | {'Compliance':>11s}")
        print(f"  {'-'*18}-+-{'-'*4}-{'-'*4}-+-{'-'*11}")
        for sys in systems_in_data:
            c = compliance.get(sys, {}).get(pid, {})
            if not c:
                continue
            cr = c["compliance"]
            print(f"  {sys:>18s} | {c['correct']:>4d} {c['total']:>4d} | {cr['point']:>10.1%}")

    # Fisher's
    for baseline in ["rake", "yake", "tfidf"]:
        key = f"fisher_gentag_vs_{baseline}"
        if key in compliance:
            print(f"  Fisher gentag vs {baseline}: p={compliance[key]}")

    # C) FER Agreement
    print(f"\n--- C) FER AGREEMENT ---")
    print(f"  {'System':>18s} | {'Match':>5s} {'N':>5s} | {'Agree':>7s} {'Kappa':>7s} | {'Up':>4s} {'Down':>4s}")
    print(f"  {'-'*18}-+-{'-'*5}-{'-'*5}-+-{'-'*7}-{'-'*7}-+-{'-'*4}-{'-'*4}")
    for sys in TAG_SYSTEMS:
        if sys not in fer_agreement:
            continue
        f = fer_agreement[sys]
        ar = f["agreement_rate"]
        k = f["kappa"] if f["kappa"] is not None else 0
        print(f"  {sys:>18s} | {f['matches']:>5d} {f['total']:>5d} | "
              f"{ar['point']:>6.1%} {k:>7.3f} | {f['upgrades']:>4d} {f['downgrades']:>4d}")

    for baseline in ["rake", "yake", "tfidf"]:
        key = f"fisher_gentag_vs_{baseline}"
        if key in fer_agreement:
            print(f"  Fisher gentag vs {baseline}: p={fer_agreement[key]}")

    # D) Ablation
    print(f"\n--- D) TOKEN-BUDGET ABLATION (truncated gentag vs baselines) ---")
    for baseline in ["rake", "yake", "tfidf"]:
        if baseline not in ablation:
            continue
        ab = ablation[baseline]
        print(f"  vs {baseline:>6s}: trunc={ab['truncated_floor']['point']:.1%} "
              f"base={ab['baseline_floor']['point']:.1%} "
              f"gap={ab['gap_pp']:+.1f}pp (p={ab['fisher_p']})")

    # E) Entropy
    print(f"\n--- E) DECISION ENTROPY ---")
    print(f"  {'System':>18s} | {'H':>6s} {'H_n':>5s} | {'P(R)':>6s} {'P(B)':>6s} {'P(C)':>6s} | {'L1/FER':>7s}")
    print(f"  {'-'*18}-+-{'-'*6}-{'-'*5}-+-{'-'*6}-{'-'*6}-{'-'*6}-+-{'-'*7}")
    for sys in systems_in_data:
        e = entropy[sys]
        d = e["distribution"]
        l1 = e.get("l1_distance_from_fer", "—")
        l1s = f"{l1:.3f}" if isinstance(l1, float) else l1
        print(f"  {sys:>18s} | {e['entropy']:>6.3f} {e['normalized_entropy']:>5.3f} | "
              f"{d['REJECT']:>5.1%} {d['BORDERLINE']:>5.1%} {d['RECOMMEND']:>5.1%} | {l1s:>7s}")

    # F) Cross-judge
    if cross_judge:
        print(f"\n--- F) CROSS-JUDGE COMPARISON ---")
        cj = cross_judge["per_system"]
        print(f"  {'System':>18s} | {'Match':>5s} {'N':>5s} | {'Agree':>7s} {'Kappa':>7s}")
        print(f"  {'-'*18}-+-{'-'*5}-{'-'*5}-+-{'-'*7}-{'-'*7}")
        for sys in systems_in_data + ["overall"]:
            if sys not in cj:
                continue
            c = cj[sys]
            ar = c["agreement_rate"]
            k = c["kappa"] if c["kappa"] is not None else 0
            label = sys.upper() if sys == "overall" else sys
            print(f"  {label:>18s} | {c['matches']:>5d} {c['total']:>5d} | "
                  f"{ar['point']:>6.1%} {k:>7.3f}")

    # Verdict
    print(f"\n{'='*75}")
    print("VERDICT")
    print(f"{'='*75}")

    # FER agreement
    gt_agree = fer_agreement.get("gentag", {}).get("agreement_rate", {}).get("point", 0)
    for baseline in ["rake", "yake", "tfidf"]:
        bl_agree = fer_agreement.get(baseline, {}).get("agreement_rate", {}).get("point", 0)
        gap = round((gt_agree - bl_agree) * 100, 1)
        status = "PAPER-READY" if gap >= 10 else "PROMISING" if gap >= 5 else "FAIL"
        print(f"  FER agree gentag vs {baseline:>6s}: {gt_agree:.0%} vs {bl_agree:.0%} "
              f"({gap:+.1f}pp) → {status}")

    # Compliance
    gt_comp = compliance.get("gentag", {}).get("combined", {}).get("compliance", {}).get("point", 0)
    for baseline in ["rake", "yake", "tfidf"]:
        bl_comp = compliance.get(baseline, {}).get("combined", {}).get("compliance", {}).get("point", 0)
        gap = round((gt_comp - bl_comp) * 100, 1)
        print(f"  Compliance gentag vs {baseline:>6s}: {gt_comp:.0%} vs {bl_comp:.0%} ({gap:+.1f}pp)")

    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
