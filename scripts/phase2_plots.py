#!/usr/bin/env python3
"""
Phase 2: Plot Generation

Generates the 6 required plots from Phase 2 analysis tables:
1. Run stability distribution (ECDF per model)
2. Prompt sensitivity heatmap (3×3 per model)
3. Model sensitivity heatmap (4×4 per prompt)
4. Retention plot (cosine by model/prompt)
5. Cost-effectiveness plot (Pareto front)
6. Surface vs semantic scatter (Jaccard vs cosine)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configuration
OUTPUT_DIR = Path("results/phase2")
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Plot style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


def load_tables():
    """Load all Phase 2 tables."""
    return {
        "run_stability": pd.read_csv(TABLES_DIR / "run_stability.csv"),
        "prompt_sensitivity": pd.read_csv(TABLES_DIR / "prompt_sensitivity.csv"),
        "model_sensitivity": pd.read_csv(TABLES_DIR / "model_sensitivity.csv"),
        "retention": pd.read_csv(TABLES_DIR / "retention.csv"),
        "uncertainty_dispersion": pd.read_csv(TABLES_DIR / "uncertainty_dispersion.csv"),
    }


def plot_1_run_stability(run_stability: pd.DataFrame):
    """Plot 1: Run stability distribution (ECDF per model)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = sorted(run_stability["model_key"].unique())
    colors = sns.color_palette("husl", len(models))
    
    # Left: ECDF
    ax = axes[0]
    for model, color in zip(models, colors):
        data = run_stability[run_stability["model_key"] == model]["cosine_similarity"].values
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, y, label=model, color=color, linewidth=2)
    
    ax.set_xlabel("Cosine Similarity (Run 1 vs Run 2)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Run Stability (ECDF)")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Right: Box plot
    ax = axes[1]
    sns.boxplot(data=run_stability, x="model_key", y="cosine_similarity", ax=ax, palette=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Run Stability (Distribution)")
    ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_run_stability.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / '1_run_stability.png'}")


def plot_2_prompt_sensitivity(prompt_sensitivity: pd.DataFrame):
    """Plot 2: Prompt sensitivity heatmap (3×3 per model)."""
    models = sorted(prompt_sensitivity["model_key"].unique())
    prompts = sorted(prompt_sensitivity["prompt1"].unique())
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Create pivot table for heatmap
        model_data = prompt_sensitivity[prompt_sensitivity["model_key"] == model]
        pivot = model_data.pivot_table(
            values="cosine_similarity",
            index="prompt1",
            columns="prompt2",
            aggfunc="mean"
        )
        
        # Ensure all prompts are present
        for p in prompts:
            if p not in pivot.index:
                pivot.loc[p] = np.nan
            if p not in pivot.columns:
                pivot[p] = np.nan
        pivot = pivot.loc[prompts, prompts]
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            cbar_kws={"label": "Mean Cosine Similarity"}
        )
        ax.set_title(f"Prompt Sensitivity\n{model}")
        ax.set_xlabel("Prompt 2")
        ax.set_ylabel("Prompt 1")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "2_prompt_sensitivity.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / '2_prompt_sensitivity.png'}")


def plot_3_model_sensitivity(model_sensitivity: pd.DataFrame):
    """Plot 3: Model sensitivity heatmap (4×4 per prompt)."""
    prompts = sorted(model_sensitivity["prompt_type"].unique())
    models = sorted(model_sensitivity["model1"].unique())
    
    n_prompts = len(prompts)
    fig, axes = plt.subplots(1, n_prompts, figsize=(6 * n_prompts, 5))
    if n_prompts == 1:
        axes = [axes]
    
    for idx, prompt in enumerate(prompts):
        ax = axes[idx]
        
        # Create pivot table for heatmap
        prompt_data = model_sensitivity[model_sensitivity["prompt_type"] == prompt]
        pivot = prompt_data.pivot_table(
            values="cosine_similarity",
            index="model1",
            columns="model2",
            aggfunc="mean"
        )
        
        # Ensure all models are present
        for m in models:
            if m not in pivot.index:
                pivot.loc[m] = np.nan
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot.loc[models, models]
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0.5,
            vmax=1.0,
            ax=ax,
            cbar_kws={"label": "Mean Cosine Similarity"}
        )
        ax.set_title(f"Model Sensitivity\n{prompt}")
        ax.set_xlabel("Model 2")
        ax.set_ylabel("Model 1")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "3_model_sensitivity.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / '3_model_sensitivity.png'}")


def plot_4_retention(retention: pd.DataFrame):
    """Plot 4: Retention plot (cosine by model/prompt)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Box plot by model
    ax = axes[0]
    models = sorted(retention["model_key"].unique())
    colors = sns.color_palette("husl", len(models))
    sns.boxplot(data=retention, x="model_key", y="retention_cosine", ax=ax, palette=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("Retention Cosine Similarity")
    ax.set_title("Retention by Model")
    ax.tick_params(axis="x", rotation=45)
    
    # Right: Box plot by prompt
    ax = axes[1]
    prompts = sorted(retention["prompt_type"].unique())
    colors = sns.color_palette("husl", len(prompts))
    sns.boxplot(data=retention, x="prompt_type", y="retention_cosine", ax=ax, palette=colors)
    ax.set_xlabel("Prompt Type")
    ax.set_ylabel("Retention Cosine Similarity")
    ax.set_title("Retention by Prompt")
    ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "4_retention.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / '4_retention.png'}")


def plot_5_cost_effectiveness(retention: pd.DataFrame):
    """Plot 5: Cost-effectiveness plot (Pareto front: retention vs cost)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = sorted(retention["model_key"].unique())
    prompts = sorted(retention["prompt_type"].unique())
    markers = ["o", "s", "D", "^", "v", "p", "*", "h"]
    
    for model in models:
        for prompt in prompts:
            data = retention[
                (retention["model_key"] == model) & 
                (retention["prompt_type"] == prompt)
            ]
            if len(data) > 0:
                mean_cost = data["cost_usd"].mean()
                mean_retention = data["retention_cosine"].mean()
                
                ax.scatter(
                    mean_cost,
                    mean_retention,
                    label=f"{model} / {prompt}",
                    s=200,
                    alpha=0.7,
                    marker=markers[len(ax.collections) % len(markers)]
                )
    
    ax.set_xlabel("Mean Cost (USD)")
    ax.set_ylabel("Mean Retention (Cosine Similarity)")
    ax.set_title("Cost-Effectiveness (Pareto Front)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "5_cost_effectiveness.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / '5_cost_effectiveness.png'}")


def plot_6_surface_vs_semantic(run_stability: pd.DataFrame):
    """Plot 6: Surface vs semantic scatter (Jaccard vs cosine)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Scatter plot
    models = sorted(run_stability["model_key"].unique())
    colors = sns.color_palette("husl", len(models))
    
    for model, color in zip(models, colors):
        data = run_stability[run_stability["model_key"] == model]
        ax.scatter(
            data["jaccard_norm_eval"],
            data["cosine_similarity"],
            label=model,
            alpha=0.5,
            color=color,
            s=30
        )
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect correlation")
    
    ax.set_xlabel("Jaccard Similarity (Surface)")
    ax.set_ylabel("Cosine Similarity (Semantic)")
    ax.set_title("Surface vs Semantic Similarity")
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "6_surface_vs_semantic.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved: {PLOTS_DIR / '6_surface_vs_semantic.png'}")


def main():
    """Generate all Phase 2 plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Generate plots")
    parser.add_argument(
        "--tables-dir",
        type=str,
        default=str(TABLES_DIR),
        help="Directory containing CSV tables"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    global TABLES_DIR, PLOTS_DIR
    TABLES_DIR = Path(args.tables_dir)
    PLOTS_DIR = Path(args.plots_dir)
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 2: PLOT GENERATION")
    print("=" * 60)
    
    # Load tables
    print("\nLoading tables...")
    tables = load_tables()
    
    for name, df in tables.items():
        print(f"  {name}: {len(df)} rows")
    
    # Generate plots
    print("\nGenerating plots...")
    
    print("\n1. Run stability...")
    plot_1_run_stability(tables["run_stability"])
    
    print("\n2. Prompt sensitivity...")
    plot_2_prompt_sensitivity(tables["prompt_sensitivity"])
    
    print("\n3. Model sensitivity...")
    plot_3_model_sensitivity(tables["model_sensitivity"])
    
    print("\n4. Retention...")
    plot_4_retention(tables["retention"])
    
    print("\n5. Cost-effectiveness...")
    plot_5_cost_effectiveness(tables["retention"])
    
    print("\n6. Surface vs semantic...")
    plot_6_surface_vs_semantic(tables["run_stability"])
    
    print("\n✅ All plots generated!")
    print(f"   Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()

