#!/usr/bin/env python3
"""
Phase 5: Plot Generation

Generates plots for Section 5.3 (Decision Evaluation) from the
baseline legibility analysis JSON.

Main paper (3 plots):
1. FER Agreement + Constraint Compliance (two-panel) — Section 5.3.1–5.3.2
2. Decision Distribution vs FER (stacked bars) — Section 5.3.5

Appendix (2 plots):
3. Cross-Judge Agreement (kappa bars) — Section 5.3.4
4. Failure Audit Breakdown — Section 5.3.6

Data source: results/phase5/baseline_legibility_analysis.json
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# Add scripts to path for shared style
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_style, SYS_COLORS, DEC_COLORS

# Configuration
DATA_PATH = Path("results/phase5/baseline_legibility_analysis.json")
PLOTS_DIR = Path("results/phase5/plots")

# Apply shared style
apply_style()

# System display config
SYSTEM_LABELS = {
    "gentag": "Gentag",
    "gentag_truncated": "Gentag (trunc.)",
    "rake": "RAKE",
    "yake": "YAKE",
    "tfidf": "TF-IDF",
    "fer": "FER",
}


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def plot_1_fer_and_compliance(data):
    """
    Plot 1 (main paper): Two-panel figure.
    Left: FER agreement by system.
    Right: Constraint compliance by persona.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left panel: FER Agreement ---
    ax = axes[0]
    fer = data["C_fer_agreement"]
    systems_order = ["gentag", "gentag_truncated", "rake", "yake", "tfidf"]
    labels = [SYSTEM_LABELS[s] for s in systems_order]
    agreements = [fer[s]["agreement_rate"]["point"] * 100 for s in systems_order]
    kappas = [fer[s]["kappa"] for s in systems_order]
    colors = [SYS_COLORS[SYSTEM_LABELS[s]] for s in systems_order]

    # Wilson CIs
    lowers = [fer[s]["agreement_rate"]["lower"] * 100 for s in systems_order]
    uppers = [fer[s]["agreement_rate"]["upper"] * 100 for s in systems_order]
    yerr_low = [a - l for a, l in zip(agreements, lowers)]
    yerr_high = [u - a for a, u in zip(agreements, uppers)]

    bars = ax.barh(labels, agreements, xerr=[yerr_low, yerr_high],
                   color=colors, edgecolor="white", linewidth=0.8, capsize=4, height=0.6)

    for bar, agr, kap in zip(bars, agreements, kappas):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(x + 1.5, y, f"{agr:.1f}%  (κ={kap:.2f})",
                va="center", fontsize=9.5, fontweight="bold")

    # Significance annotations
    fisher_map = {
        "rake": fer.get("fisher_gentag_vs_rake"),
        "yake": fer.get("fisher_gentag_vs_yake"),
        "tfidf": fer.get("fisher_gentag_vs_tfidf"),
    }
    for i, s in enumerate(systems_order):
        p = fisher_map.get(s)
        if p is not None:
            stars = sig_stars(p)
            if stars:
                bar = bars[i]
                ax.text(2, bar.get_y() + bar.get_height() / 2, stars,
                        va="center", fontsize=11, color="#c0392b", fontweight="bold")

    ax.set_xlabel("FER Agreement (%)", fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_title("(a) FER Agreement", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    # --- Right panel: Constraint Compliance by persona ---
    ax = axes[1]
    comp = data["B_hard_requirement_compliance"]
    personas = ["P2", "P3", "Combined"]
    persona_labels = ["P2\nSports Fan", "P3\nQuick Lunch", "Combined"]
    systems_comp = ["gentag", "gentag_truncated", "rake", "yake", "tfidf"]

    x = np.arange(len(personas))
    width = 0.15
    offsets = np.arange(len(systems_comp)) - (len(systems_comp) - 1) / 2

    for i, s in enumerate(systems_comp):
        label = SYSTEM_LABELS[s]
        vals = []
        for p in personas:
            key = p if p != "Combined" else "combined"
            vals.append(comp[s][key]["compliance"]["point"] * 100)
        bars = ax.bar(x + offsets[i] * width, vals, width * 0.9,
                      label=label, color=SYS_COLORS[label], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val < 100:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(persona_labels, fontsize=10)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    ax.set_ylim(50, 105)
    ax.set_title("(b) Hard-Constraint Compliance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left", ncol=2)

    fig.tight_layout()
    out = PLOTS_DIR / "1_fer_agreement_and_compliance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_2_decision_distribution(data):
    """
    Plot 2 (main paper): Decision distribution stacked horizontal bars.
    Shows P(REJECT), P(BORDERLINE), P(RECOMMEND) for each system,
    with FER as reference at top. L1 distance annotated.
    """
    ent = data["E_decision_entropy"]
    systems_order = ["fer", "gentag", "gentag_truncated", "rake", "yake", "tfidf"]
    labels = [SYSTEM_LABELS[s] for s in systems_order]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    y_pos = np.arange(len(systems_order))

    for i, s in enumerate(systems_order):
        dist = ent[s]["distribution"]
        rej = dist["REJECT"] * 100
        bor = dist["BORDERLINE"] * 100
        rec = dist["RECOMMEND"] * 100

        ax.barh(i, rej, color=DEC_COLORS["REJECT"], edgecolor="white", linewidth=0.5, height=0.65)
        ax.barh(i, bor, left=rej, color=DEC_COLORS["BORDERLINE"], edgecolor="white",
                linewidth=0.5, height=0.65)
        ax.barh(i, rec, left=rej + bor, color=DEC_COLORS["RECOMMEND"], edgecolor="white",
                linewidth=0.5, height=0.65)

        # Percentage labels inside bars (only if segment wide enough)
        if rej > 8:
            ax.text(rej / 2, i, f"{rej:.1f}%", ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        if bor > 8:
            ax.text(rej + bor / 2, i, f"{bor:.1f}%", ha="center", va="center", fontsize=8,
                    fontweight="bold")
        if rec > 8:
            ax.text(rej + bor + rec / 2, i, f"{rec:.1f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

        # L1 distance from FER (skip FER itself)
        l1 = ent[s].get("l1_distance_from_fer")
        if l1 is not None:
            ax.text(102, i, f"L1={l1:.3f}", va="center", fontsize=9, color="#555")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Decision Distribution (%)", fontsize=12)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()

    # Add a separator line below FER
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Legend
    patches = [mpatches.Patch(color=DEC_COLORS[d], label=d) for d in ["REJECT", "BORDERLINE", "RECOMMEND"]]
    ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title("Decision Distribution by System vs Full-Evidence Reference",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    out = PLOTS_DIR / "2_decision_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_3_cross_judge(data):
    """
    Plot 3 (appendix): Cross-judge Cohen's kappa per system.
    """
    cj = data["F_cross_judge"]["per_system"]
    systems_order = ["fer", "gentag", "gentag_truncated", "rake", "yake", "tfidf"]
    labels = [SYSTEM_LABELS[s] for s in systems_order]
    kappas = [cj[s]["kappa"] for s in systems_order]
    colors = [SYS_COLORS[SYSTEM_LABELS[s]] for s in systems_order]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(labels, kappas, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, k in zip(bars, kappas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{k:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Reference lines
    ax.axhline(y=0.6, color="#e67e22", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(5.6, 0.605, "Substantial", fontsize=8, color="#e67e22")
    ax.axhline(y=0.4, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=1)
    ax.text(5.6, 0.405, "Moderate", fontsize=8, color="#e74c3c")

    # Overall kappa
    overall_k = cj["overall"]["kappa"]
    ax.axhline(y=overall_k, color="#2c3e50", linestyle="-", alpha=0.4, linewidth=1.5)
    ax.text(0, overall_k + 0.01, f"Overall κ={overall_k:.3f}", fontsize=9, color="#2c3e50")

    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_ylim(0, 0.9)
    ax.set_title("Cross-Judge Agreement by System", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out = PLOTS_DIR / "3_cross_judge_kappa.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_4_failure_audit(data):
    """
    Plot 4 (appendix): Gentag-FER mismatch breakdown.
    Data from RESULTS.md Section 5.3.6 (hardcoded — not in JSON).
    """
    # From RESULTS.md: 41 Gentag-FER mismatches
    transitions = [
        ("REC→BORDER", 18),
        ("REJ→BORDER", 9),
        ("BORDER→REC", 6),
        ("REC→REJ", 5),
        ("REJ→REC", 3),
    ]
    labels = [t[0] for t in transitions]
    counts = [t[1] for t in transitions]

    # Color: one-step = amber, full reversal = red
    colors = ["#f39c12", "#f39c12", "#f39c12", "#e74c3c", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: transition types
    ax = axes[0]
    bars = ax.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.8, height=0.6)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(c), va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("(a) Mismatch Type", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 22)

    one_step = mpatches.Patch(color="#f39c12", label="One-step drift (80.5%)")
    full_rev = mpatches.Patch(color="#e74c3c", label="Full reversal (19.5%)")
    ax.legend(handles=[one_step, full_rev], fontsize=9, loc="lower right")

    # Right: by persona
    ax = axes[1]
    persona_counts = [("P1", 24), ("P4", 11), ("P3", 5), ("P2", 1)]
    p_labels = [p[0] for p in persona_counts]
    p_counts = [p[1] for p in persona_counts]
    p_colors = ["#3498db", "#9b59b6", "#e74c3c", "#2ecc71"]

    bars = ax.barh(p_labels, p_counts, color=p_colors, edgecolor="white", linewidth=0.8, height=0.6)
    for bar, c in zip(bars, p_counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{c}/41", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Count", fontsize=12)
    ax.set_title("(b) Mismatches by Persona", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 30)

    fig.suptitle("Gentag–FER Disagreement Audit (41 mismatches out of 200)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "4_failure_audit.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    print("=" * 60)
    print("Phase 5: Generating Plots")
    print("=" * 60)

    print("\n[1] FER Agreement + Constraint Compliance (main paper)...")
    plot_1_fer_and_compliance(data)

    print("\n[2] Decision Distribution (main paper)...")
    plot_2_decision_distribution(data)

    print("\n[3] Cross-Judge Kappa (appendix)...")
    plot_3_cross_judge(data)

    print("\n[4] Failure Audit (appendix)...")
    plot_4_failure_audit(data)

    print("\n" + "=" * 60)
    print("Done. Plots saved to:", PLOTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
