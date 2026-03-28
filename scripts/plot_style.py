"""
Shared plot style for all paper figures.

Usage:
    from plot_style import apply_style, METHOD_COLORS, SYS_COLORS, DEC_COLORS

All scripts import this to ensure consistent styling across Phase 2/3/5 plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def apply_style():
    """Apply consistent style to all paper plots."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# Method colors (Phase 3 structure comparisons)
METHOD_COLORS = {
    "Gentags": "#2ecc71",   # green
    "RAKE":    "#e74c3c",   # red
    "TF-IDF":  "#3498db",   # blue
    "YAKE":    "#f39c12",   # orange
}

# System colors (Phase 5 decision evaluation — includes FER + truncated)
SYS_COLORS = {
    "Gentag":          "#2ecc71",
    "Gentag (trunc.)": "#82e0aa",
    "RAKE":            "#e74c3c",
    "TF-IDF":          "#3498db",
    "YAKE":            "#f39c12",
    "FER":             "#9b59b6",
}

# Decision colors
DEC_COLORS = {
    "REJECT":     "#e74c3c",
    "BORDERLINE": "#f4d03f",
    "RECOMMEND":  "#2ecc71",
}

# Model colors (Phase 2 stability — 4 extractor models)
MODEL_COLORS = {
    "claude": "#e74c3c",
    "gemini": "#f39c12",
    "grok":   "#2ecc71",
    "openai": "#9b59b6",
}
