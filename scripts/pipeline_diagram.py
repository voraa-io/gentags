#!/usr/bin/env python3
"""
Fig 1: Gentag Pipeline Diagram

Shows the extraction and evaluation pipeline for the ACL paper.
Two paths: Gentag (LLM extraction) vs Lexical baselines (RAKE/YAKE/TF-IDF).
Both feed into the same judge for downstream decision evaluation.

Output: results/figures/fig1_pipeline.png
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Add scripts to path for shared style
sys.path.insert(0, str(Path(__file__).parent))
from plot_style import apply_style

# Apply shared style (then override grid for diagram)
apply_style()

# Colors
C_SOURCE = "#ecf0f1"      # light gray
C_GENTAG = "#2ecc71"      # green
C_BASELINE = "#e74c3c"    # red
C_SHARED = "#3498db"      # blue
C_DECISION = "#f39c12"    # orange
C_EVAL = "#9b59b6"        # purple
C_TEXT = "#2c3e50"         # dark


def rounded_box(ax, xy, width, height, text, color, text_color=C_TEXT,
                fontsize=10, fontweight="normal", alpha=0.85, subtext=None):
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.15", facecolor=color, edgecolor="#7f8c8d",
        linewidth=1.2, alpha=alpha, zorder=2
    )
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.18, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=3)
        ax.text(x, y - 0.22, subtext, ha="center", va="center",
                fontsize=fontsize - 2, color="#2c3e50", alpha=0.7, style="italic", zorder=3)
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=3)


def arrow(ax, start, end, color="#7f8c8d", style="-|>", lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0"),
                zorder=1)


def curved_arrow(ax, start, end, color="#7f8c8d", rad=0.3, lw=1.5):
    """Draw a curved arrow."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(-1, 15.5)
    ax.set_ylim(-1.5, 6.5)
    ax.axis("off")

    # === Column positions ===
    x_source = 1.0
    x_extract = 4.3
    x_state = 8.0
    x_judge = 11.5
    x_decision = 14.0

    # === Row positions — more vertical spread ===
    y_gentag = 4.5
    y_baseline = 0.5
    y_shared = 2.5

    # ── SOURCE ──
    rounded_box(ax, (x_source, y_shared), 2.2, 2.8, "Reviews", C_SOURCE,
                fontsize=12, fontweight="bold",
                subtext="(venue evidence)")
    ax.text(x_source, y_shared - 0.65, '"Great food, slow service..."', ha="center",
            fontsize=7.5, color="#555", style="italic")

    # ── GENTAG PATH (top) ──
    # Fix #2: simplified extractor — no "4 models x 3 prompts" detail
    rounded_box(ax, (x_extract, y_gentag), 2.2, 1.2, "LLM Extractor", C_GENTAG,
                fontsize=11, fontweight="bold")

    # Fix #1 + #6: larger representation box, "Gentag Representation"
    rounded_box(ax, (x_state, y_gentag), 3.0, 1.8, "Gentag Representation", C_GENTAG,
                fontsize=12, fontweight="bold", alpha=0.9)
    # Fix #3: clean example tags, one per line style
    ax.text(x_state, y_gentag - 0.25, "fast service\noutdoor seating\nfriendly staff",
            ha="center", va="center", fontsize=8.5, color="#1a5e2f",
            linespacing=1.4, zorder=3)

    # ── BASELINE PATH (bottom) ──
    rounded_box(ax, (x_extract, y_baseline), 2.2, 1.2, "Keyword Extraction", C_BASELINE,
                fontsize=11, fontweight="bold",
                subtext="RAKE / YAKE / TF-IDF")

    # Fix #1: larger representation box for baseline too
    rounded_box(ax, (x_state, y_baseline), 3.0, 1.8, "Keyword Set", C_BASELINE,
                fontsize=12, fontweight="bold", alpha=0.9)
    ax.text(x_state, y_baseline - 0.25, "relative quick time\nlunch\nfashioned regional",
            ha="center", va="center", fontsize=8.5, color="#7b1a1a",
            linespacing=1.4, zorder=3)

    # ── SHARED JUDGE ──
    # Fix #5: "Decision rule" annotation inside judge
    rounded_box(ax, (x_judge, y_shared), 2.2, 2.2, "Judge LLM", C_SHARED,
                fontsize=12, fontweight="bold")
    ax.text(x_judge, y_shared - 0.35, "persona constraints\n+ decision rule",
            ha="center", va="center", fontsize=8.5, color="#1a3d6b",
            linespacing=1.3, zorder=3)

    # ── DECISION ──
    rounded_box(ax, (x_decision, y_shared), 1.6, 1.2, "Decision", C_DECISION,
                fontsize=11, fontweight="bold")
    ax.text(x_decision, y_shared - 0.25, "REC / BORDER / REJ",
            ha="center", fontsize=7.5, color="#6b3a00", zorder=3)

    # === ARROWS ===

    # Source → extractors
    arrow(ax, (x_source + 1.1, y_shared + 0.5), (x_extract - 1.1, y_gentag - 0.1),
          color=C_GENTAG, lw=1.8)
    arrow(ax, (x_source + 1.1, y_shared - 0.5), (x_extract - 1.1, y_baseline + 0.1),
          color=C_BASELINE, lw=1.8)

    # Extractors → representations
    arrow(ax, (x_extract + 1.1, y_gentag), (x_state - 1.5, y_gentag),
          color=C_GENTAG, lw=1.8)
    arrow(ax, (x_extract + 1.1, y_baseline), (x_state - 1.5, y_baseline),
          color=C_BASELINE, lw=1.8)

    # Representations → judge
    arrow(ax, (x_state + 1.5, y_gentag - 0.3), (x_judge - 1.1, y_shared + 0.5),
          color=C_GENTAG, lw=1.8)
    arrow(ax, (x_state + 1.5, y_baseline + 0.3), (x_judge - 1.1, y_shared - 0.5),
          color=C_BASELINE, lw=1.8)

    # Judge → decision
    arrow(ax, (x_judge + 1.1, y_shared), (x_decision - 0.8, y_shared),
          color=C_DECISION, lw=1.8)

    # Fix #4: FER dashed arrow — straight through the middle gap
    ax.annotate("", xy=(x_judge - 1.1, y_shared),
                xytext=(x_source + 1.1, y_shared),
                arrowprops=dict(arrowstyle="-|>", color="#95a5a6", lw=1.5,
                                linestyle="dashed",
                                connectionstyle="arc3,rad=0"),
                zorder=4)
    ax.text((x_source + x_judge) / 2, y_shared - 0.3,
            "Full-Evidence Reference (FER)",
            ha="center", fontsize=9, color="#7f8c8d", style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#d5d8dc", alpha=0.95), zorder=5)

    # === COLUMN HEADERS ===
    # Fix #1: emphasize "Intermediate Representation" column
    ax.text(x_extract, 6.0, "Extraction", ha="center", fontsize=10,
            fontweight="bold", color=C_EVAL)

    ax.text(x_state, 6.0, "INTERMEDIATE REPRESENTATION", ha="center", fontsize=11,
            fontweight="bold", color=C_EVAL)
    ax.text(x_state, 5.65, "(Stability + Structure evaluation)", ha="center",
            fontsize=8.5, color="#95a5a6")

    ax.text(x_judge, 6.0, "Decision", ha="center", fontsize=10,
            fontweight="bold", color=C_EVAL)
    ax.text(x_judge, 5.65, "(Utility evaluation)", ha="center",
            fontsize=8.5, color="#95a5a6")

    fig.tight_layout()

    # Save
    out = "results/figures/fig1_pipeline.png"
    import os
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
