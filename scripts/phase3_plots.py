#!/usr/bin/env python3
"""
Phase 3: Plot Generation

Generates plots for Section 5.2 (Structural Analysis) comparing gentags
against RAKE, YAKE, TF-IDF baselines using State-Gini methodology.

Plots:
1. State-Gini + Other-Rate comparison (grouped bars) — Section 5.2.1–5.2.2
2. Threshold sensitivity sweep (tau=0.30, 0.35, 0.40) — Section 5.2.3
3. Facet coverage heatmap (per-facet counts by method) — Section 5.2

Data sources:
- Gentag: results/phase3/tables/state_localization[_tau030|_tau040].csv
- Baselines: results/phase3a/tables/baseline_state_gini[_tau030|_tau040].csv
- Bleed check: results/phase3/baseline_bleed_check_summary.json
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add scripts to path for shared style
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_style, METHOD_COLORS

# Configuration
PHASE3_DIR = Path("results/phase3")
PHASE3A_DIR = Path("results/phase3a")
PLOTS_DIR = PHASE3_DIR / "plots"

# Apply shared style
apply_style()
METHOD_ORDER = ["Gentags", "RAKE", "TF-IDF", "YAKE"]

FACETS = [
    "food_quality", "coffee_drinks", "service", "ambiance", "price_value",
    "crowding", "seating", "dietary", "portions", "location"
]


def load_gentag_stats(suffix=""):
    """Load gentag-level stats from state_localization CSV."""
    path = PHASE3_DIR / "tables" / f"state_localization{suffix}.csv"
    df = pd.read_csv(path)
    return {
        "state_gini_mean": df["gentag_state_gini"].mean(),
        "state_gini_std": df["gentag_state_gini"].std(),
        "n_tags_mean": df["n_gentags"].mean(),
        "assigned_mean": df["gentag_assigned_count"].mean(),
        "other_mean": df["gentag_other_count"].mean(),
        "other_rate_mean": (df["gentag_other_count"] / df["n_gentags"]).mean(),
        "facet_means": {f: df[f"gentag_count_{f}"].mean() for f in FACETS},
        "gini_values": df["gentag_state_gini"].values,
    }


def load_baseline_stats(suffix=""):
    """Load baseline-level stats from Phase 3a CSV."""
    path = PHASE3A_DIR / "tables" / f"baseline_state_gini{suffix}.csv"
    df = pd.read_csv(path)
    stats = {}
    for method in ["rake", "tfidf", "yake"]:
        m = df[df["method"] == method]
        stats[method] = {
            "state_gini_mean": m["state_gini"].mean(),
            "state_gini_std": m["state_gini"].std(),
            "n_keywords_mean": m["n_keywords"].mean(),
            "assigned_mean": m["assigned_count"].mean(),
            "other_mean": m["other_count"].mean(),
            "other_rate_mean": (m["other_count"] / m["n_keywords"]).mean(),
            "facet_means": {f: m[f"count_{f}"].mean() for f in FACETS},
            "gini_values": m["state_gini"].values,
        }
    return stats


def plot_1_gini_and_coverage():
    """
    Plot 1: State-Gini + Other-Rate comparison (Section 5.2.1–5.2.2).

    Two-panel grouped bar chart:
    - Left: Mean State-Gini by method (lower = more balanced)
    - Right: Mean Other-Rate by method (lower = better coverage)
    """
    g = load_gentag_stats()
    b = load_baseline_stats()

    methods = METHOD_ORDER
    gini_means = [
        g["state_gini_mean"],
        b["rake"]["state_gini_mean"],
        b["tfidf"]["state_gini_mean"],
        b["yake"]["state_gini_mean"],
    ]
    gini_stds = [
        g["state_gini_std"],
        b["rake"]["state_gini_std"],
        b["tfidf"]["state_gini_std"],
        b["yake"]["state_gini_std"],
    ]
    other_rates = [
        g["other_rate_mean"],
        b["rake"]["other_rate_mean"],
        b["tfidf"]["other_rate_mean"],
        b["yake"]["other_rate_mean"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = [METHOD_COLORS[m] for m in methods]
    x = np.arange(len(methods))

    # Left: State-Gini
    ax = axes[0]
    bars = ax.bar(x, gini_means, yerr=gini_stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, val in zip(bars, gini_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Mean State-Gini", fontsize=12)
    ax.set_title("State-Gini\n(lower = more balanced across facets)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Uniform baseline")
    ax.legend(fontsize=9)

    # Right: Other-Rate
    ax = axes[1]
    bars = ax.bar(x, [r * 100 for r in other_rates], color=colors,
                  edgecolor="white", linewidth=0.8, width=0.6)
    for bar, val in zip(bars, other_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Other-Rate (%)", fontsize=12)
    ax.set_title("Facet Coverage (Other-Rate)\n(lower = more mass in facet space)", fontsize=12)
    ax.set_ylim(0, 100)

    fig.suptitle("Phase 3: Gentags vs Lexical Baselines — Structural Analysis (τ = 0.35)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "1_gini_and_coverage.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_2_threshold_sensitivity():
    """
    Plot 2: Threshold sensitivity sweep (Section 5.2.3).

    Two-panel line plot across tau={0.30, 0.35, 0.40}:
    - Left: State-Gini by method across tau
    - Right: Other-Rate by method across tau
    """
    taus = [0.30, 0.35, 0.40]
    suffixes = ["_tau030", "", "_tau040"]

    data = {"tau": [], "method": [], "gini": [], "other_rate": []}

    for tau, suffix in zip(taus, suffixes):
        g = load_gentag_stats(suffix)
        b = load_baseline_stats(suffix)

        for label, stats in [("Gentags", g), ("RAKE", b["rake"]),
                              ("TF-IDF", b["tfidf"]), ("YAKE", b["yake"])]:
            data["tau"].append(tau)
            data["method"].append(label)
            data["gini"].append(stats["state_gini_mean"])
            data["other_rate"].append(stats["other_rate_mean"] * 100)

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: State-Gini across tau
    ax = axes[0]
    for method in METHOD_ORDER:
        mdf = df[df["method"] == method]
        ax.plot(mdf["tau"], mdf["gini"], "o-", color=METHOD_COLORS[method],
                label=method, linewidth=2, markersize=8)
        for _, row in mdf.iterrows():
            ax.annotate(f"{row['gini']:.3f}", (row["tau"], row["gini"]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha="center")
    ax.set_xlabel("Threshold (τ)", fontsize=12)
    ax.set_ylabel("Mean State-Gini", fontsize=12)
    ax.set_title("State-Gini across thresholds", fontsize=12)
    ax.set_xticks(taus)
    ax.set_ylim(0.4, 0.9)
    ax.legend(fontsize=10)

    # Right: Other-Rate across tau
    ax = axes[1]
    for method in METHOD_ORDER:
        mdf = df[df["method"] == method]
        ax.plot(mdf["tau"], mdf["other_rate"], "o-", color=METHOD_COLORS[method],
                label=method, linewidth=2, markersize=8)
        for _, row in mdf.iterrows():
            ax.annotate(f"{row['other_rate']:.1f}%", (row["tau"], row["other_rate"]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha="center")
    ax.set_xlabel("Threshold (τ)", fontsize=12)
    ax.set_ylabel("Other-Rate (%)", fontsize=12)
    ax.set_title("Other-Rate across thresholds", fontsize=12)
    ax.set_xticks(taus)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)

    fig.suptitle("Phase 3: Threshold Sensitivity — State-Gini and Facet Coverage",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = PLOTS_DIR / "2_threshold_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_3_facet_heatmap():
    """
    Plot 3: Per-facet mean tag/keyword counts by method (Section 5.2).

    Heatmap showing how each method distributes mass across the 10 facets,
    plus an 'other' column.
    """
    g = load_gentag_stats()
    b = load_baseline_stats()

    # Build matrix: rows = methods, cols = facets + other
    facet_labels = [f.replace("_", "\n") for f in FACETS] + ["other"]
    matrix = []

    for label, stats in [("Gentags", g), ("RAKE", b["rake"]),
                          ("TF-IDF", b["tfidf"]), ("YAKE", b["yake"])]:
        row = [stats["facet_means"][f] for f in FACETS] + [stats["other_mean"]]
        matrix.append(row)

    matrix = np.array(matrix)

    # Normalize each row to percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = (matrix / row_sums) * 100

    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = sns.heatmap(matrix_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                     xticklabels=facet_labels, yticklabels=METHOD_ORDER,
                     ax=ax, cbar_kws={"label": "% of total tags/keywords"},
                     linewidths=0.5)

    ax.set_title("Facet Distribution by Method (% of total semantic units, τ = 0.35)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    out = PLOTS_DIR / "3_facet_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3: Generating Plots")
    print("=" * 60)

    print("\n[1] State-Gini + Other-Rate comparison...")
    plot_1_gini_and_coverage()

    print("\n[2] Threshold sensitivity sweep...")
    plot_2_threshold_sensitivity()

    print("\n[3] Facet coverage heatmap...")
    plot_3_facet_heatmap()

    print("\n" + "=" * 60)
    print("Done. Plots saved to:", PLOTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
