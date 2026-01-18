#!/usr/bin/env python3
"""
Generate animated GIF showing Monte Carlo robustness analysis evolution.

This script creates a 3-panel animation showing:
- Panel A: p-value distribution forming
- Panel B: Δ Mean Disagreement distribution forming  
- Panel C: Scatter cloud of random judges growing around the LLM point

=== NEW INTERFACE (using cleaned datasets) ===

Usage with cleaned CSV:
    python generate_monte_carlo_gif.py \
        --data path/to/cleaned_data.csv \
        --output path/to/output.gif \
        --llm-col "GPT_as_a_judge" \
        --min-score 0 \
        --max-score 6

For Newsroom (multi-metric dataset), specify the metric:
    python generate_monte_carlo_gif.py \
        --data benchmarks/newsroom/data/newsroom_judged_cleaned.csv \
        --output benchmarks/newsroom/figures/monte_carlo_informativeness.gif \
        --llm-col "GPT_informativeness_as_a_judge" \
        --metric informativeness \
        --min-score 1 --max-score 5

Examples:
    # MovieLens
    python scripts/generate_monte_carlo_gif.py \
        --data benchmarks/movielens/data/merged_cleaned.csv \
        --output benchmarks/movielens/figures/monte_carlo.gif \
        --llm-col "GPT-4o-mini" \
        --min-score 1 --max-score 5

    # WMT-Human
    python scripts/generate_monte_carlo_gif.py \
        --data benchmarks/wmt-human/data/wmt-human_en_de_judged_cleaned.csv \
        --output benchmarks/wmt-human/figures/monte_carlo.gif \
        --llm-col "GPT_as_a_judge" \
        --min-score 0 --max-score 6

    # Newsroom (informativeness)
    python scripts/generate_monte_carlo_gif.py \
        --data benchmarks/newsroom/data/newsroom_judged_cleaned.csv \
        --output benchmarks/newsroom/figures/monte_carlo_informativeness.gif \
        --llm-col "GPT_informativeness_as_a_judge" \
        --metric informativeness \
        --min-score 1 --max-score 5

    # Custom frame iterations (faster GIF with fewer frames)
    python scripts/generate_monte_carlo_gif.py \
        --data benchmarks/movielens/data/merged_cleaned.csv \
        --output benchmarks/movielens/figures/monte_carlo_quick.gif \
        --llm-col "GPT-4o-mini" \
        --min-score 1 --max-score 5 \
        --frame-iterations "100,500,1000,5000,10000,25000"

=== LEGACY INTERFACE (separate files) ===

    python generate_monte_carlo_gif.py \
        --llm-ratings path/to/llm_ratings.csv \
        --human-judgments path/to/human_judgments.csv \
        --output path/to/output.gif \
        --llm-col "GPT-4o mini" \
        --min-score 0 --max-score 2
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from PIL import Image
import io

from core.llm_good_enough import LLMGoodEnough


def load_cleaned_data(data_path: str, llm_col: str, metric: str = None) -> tuple:
    """
    Load a cleaned dataset and auto-detect human/LLM columns.
    
    Parameters
    ----------
    data_path : str
        Path to the cleaned CSV file
    llm_col : str
        Name of the LLM column to use
    metric : str, optional
        For multi-metric datasets like newsroom, filter columns by metric
        (e.g., 'informativeness', 'relevance', 'fluency', 'coherence')
    
    Returns
    -------
    tuple
        (DataFrame, human_cols list, llm_cols list)
    """
    df = pd.read_csv(data_path)
    
    # Auto-detect human columns
    # Pattern 1: human_#1, human_#2, etc. (movielens, wmt-human)
    # Pattern 2: metric_human_#1 (newsroom)
    if metric:
        # Newsroom-style: filter by metric
        human_cols = [c for c in df.columns if f"{metric}_human_#" in c]
        llm_cols = [c for c in df.columns if f"{metric}_as_a_judge" in c]
        if not human_cols:
            raise ValueError(f"No human columns found for metric '{metric}'. "
                           f"Available columns: {list(df.columns)}")
    else:
        # Standard style: human_#1, human_#2, etc.
        human_cols = [c for c in df.columns if c.startswith("human_#")]
        
        # If no standard human cols, try to detect newsroom-style
        if not human_cols:
            human_cols = [c for c in df.columns if "_human_#" in c]
        
        # Auto-detect LLM columns (ending with _as_a_judge or exact match)
        llm_cols = [c for c in df.columns if c.endswith("_as_a_judge")]
    
    # Verify the requested LLM column exists
    if llm_col not in df.columns:
        available_llm = [c for c in df.columns if "as_a_judge" in c.lower() or "gpt" in c.lower() or "llama" in c.lower() or "mistral" in c.lower()]
        raise ValueError(f"LLM column '{llm_col}' not found. "
                        f"Available LLM-like columns: {available_llm}")
    
    return df, human_cols, llm_cols


def prepare_data_legacy(llm_ratings_path: str, human_judgments_path: str, llm_col_name: str) -> tuple:
    """
    [LEGACY] Load and merge LLM ratings with human judgments from separate files.
    
    This is the original interface for PolitiFact-style data.
    """
    
    # Load data
    llm_ratings = pd.read_csv(llm_ratings_path)
    human_judgments = pd.read_csv(human_judgments_path)
    
    # Rename rating column to LLM name
    if "rating" in llm_ratings.columns:
        llm_ratings = llm_ratings.rename(columns={"rating": llm_col_name})
    
    # Rename human columns to human_#1, human_#2, etc.
    first_col = human_judgments.columns[0]  # Usually 'statement' or 'id'
    human_judgments.columns = [first_col] + [f"human_#{i}" for i in range(1, len(human_judgments.columns))]
    
    # Merge on first column (statement/id)
    merged_df = pd.merge(llm_ratings, human_judgments, on=first_col, how="inner")
    
    # Identify columns
    human_cols = [c for c in merged_df.columns if c.startswith("human_#")]
    llm_cols = [llm_col_name]
    
    return merged_df, human_cols, llm_cols


def render_frame(
    df_mc_subset: pd.DataFrame,
    human_dis: np.ndarray,
    llm_dis: np.ndarray,
    delta_llm: float,
    p_val_llm: float,
    current_n: int,
    max_n: int,
    figsize: tuple = (20, 6),
) -> Image.Image:
    """Render a single frame of the animation."""
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Color scheme
    color_random = "orangered"
    color_llm = "#2ecc71"
    color_kde1 = "royalblue"
    color_kde2 = "#333333"
    
    p_values = df_mc_subset["p_value"].values
    delta_values = df_mc_subset["delta_mean"].values
    
    # ----- Panel A: p-value Distribution -----
    if len(p_values) >= 10:
        half = len(p_values) // 2
        pA, pB = p_values[:half], p_values[half:]
        
        if len(pA) >= 2 and len(pB) >= 2:
            label_pA = f"First half\n(μ={np.mean(pA):.3f}, σ={np.std(pA):.3f})"
            label_pB = f"Second half\n(μ={np.mean(pB):.3f}, σ={np.std(pB):.3f})"
            sns.kdeplot(pA, fill=True, ax=axes[0], label=label_pA, alpha=0.5, color=color_kde1)
            sns.kdeplot(pB, fill=True, ax=axes[0], label=label_pB, alpha=0.5, color=color_kde2)
    
    axes[0].axvline(0.05, linestyle="--", color="black", linewidth=2, label="α = 0.05")
    axes[0].set_title("Panel A: p-Value Distribution", fontweight="bold", fontsize=18)
    axes[0].set_xlabel("p-Value", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Density", fontsize=14, fontweight="bold")
    axes[0].set_xlim(-0.05, 1.05)
    axes[0].legend(fontsize=11, loc="upper right", edgecolor="black", facecolor="white", framealpha=0.95)
    axes[0].tick_params(labelsize=12)
    
    # ----- Panel B: Δ Mean Distribution -----
    if len(delta_values) >= 10:
        half = len(delta_values) // 2
        dA, dB = delta_values[:half], delta_values[half:]
        
        if len(dA) >= 2 and len(dB) >= 2:
            label_dA = f"First half\n(μ={np.mean(dA):.4f}, σ={np.std(dA):.4f})"
            label_dB = f"Second half\n(μ={np.mean(dB):.4f}, σ={np.std(dB):.4f})"
            sns.kdeplot(dA, fill=True, ax=axes[1], label=label_dA, alpha=0.5, color=color_kde1)
            sns.kdeplot(dB, fill=True, ax=axes[1], label=label_dB, alpha=0.5, color=color_kde2)
    
    axes[1].axvline(0, linestyle="-.", color="black", linewidth=1.5)
    axes[1].set_title("Panel B: Δ Mean Disagreement", fontweight="bold", fontsize=18)
    axes[1].set_xlabel("Δ Mean Disagreement", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Density", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11, loc="upper right", edgecolor="black", facecolor="white", framealpha=0.95)
    axes[1].tick_params(labelsize=12)
    
    # ----- Panel C: Scatter Plot -----
    if len(df_mc_subset) > 0:
        axes[2].scatter(
            df_mc_subset["delta_mean"],
            df_mc_subset["p_value"],
            alpha=0.4,
            s=60,
            color=color_random,
            label=f"Random judges (n={len(df_mc_subset):,})",
            edgecolor="none",
        )
    
    # LLM point (always visible)
    axes[2].scatter(
        delta_llm, p_val_llm,
        color=color_llm,
        edgecolor="black",
        s=200,
        marker="X",
        linewidth=2,
        label="LLM",
        zorder=10,
    )
    
    axes[2].axhline(0.05, linestyle="--", color="black", linewidth=2)
    axes[2].axvline(0, linestyle="-.", color="gray", linewidth=1.5)
    axes[2].set_title("Panel C: Random Cloud vs LLM", fontweight="bold", fontsize=18)
    axes[2].set_xlabel("Δ Mean Disagreement", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("p-Value", fontsize=14, fontweight="bold")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=12, loc="upper right", edgecolor="black", facecolor="white", framealpha=0.95)
    axes[2].tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # ----- Iteration Counter (top center) -----
    fig.suptitle(
        f"Monte Carlo Robustness Analysis  •  n = {current_n:,} / {max_n:,}",
        fontsize=22,
        fontweight="bold",
        y=1.044,
    )
    
    # Progress bar effect below title
    progress = current_n / max_n
    progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
    fig.text(0.5, 0.96, f"[{progress_bar}] {progress*100:.0f}%", 
             ha="center", fontsize=14, family="monospace", color="dimgray")
    
    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def generate_gif(
    judge: LLMGoodEnough,
    llm_col: str,
    output_path: str,
    max_iterations: int = 25000,
    frame_iterations: list = None,
    duration: int = 400,
    final_pause: int = 2000,
):
    """Generate the animated GIF."""
    
    if frame_iterations is None:
        # Logarithmic spacing for nice visual progression
        frame_iterations = [
            50, 100, 200, 400, 700, 1000, 1500, 2000, 3000, 4000,
            5000, 7000, 10000, 13000, 16000, 20000, 25000
        ]
        # Filter to max_iterations
        frame_iterations = [n for n in frame_iterations if n <= max_iterations]
        if max_iterations not in frame_iterations:
            frame_iterations.append(max_iterations)
    
    print(f"🎬 Generating GIF with {len(frame_iterations)} frames...")
    print(f"   Frame iterations: {frame_iterations}")
    
    # Compute human and LLM disagreements
    human_dis = judge.compute_human_disagreements()
    llm_dis = judge.compute_llm_human_disagreements(llm_col)
    
    # LLM statistics (constant across frames)
    delta_llm = np.mean(llm_dis) - np.mean(human_dis)
    p_val_llm = mannwhitneyu(llm_dis, human_dis, alternative="greater").pvalue
    
    print(f"   LLM Δ Mean: {delta_llm:.4f}, p-value: {p_val_llm:.4f}")
    
    # Run full Monte Carlo simulation
    print(f"   Running Monte Carlo simulation ({max_iterations:,} iterations)...")
    df_mc = judge._monte_carlo_random_judges(
        iterations=max_iterations,
        min_score=judge.min_score,
        max_score=judge.max_score,
        df_items=judge.df,
        human_cols=judge.human_cols,
    )
    
    # Generate frames
    frames = []
    for i, n in enumerate(frame_iterations):
        print(f"   Rendering frame {i+1}/{len(frame_iterations)}: n={n:,}")
        
        # Take first n samples (cumulative growth)
        df_subset = df_mc.iloc[:n]
        
        frame = render_frame(
            df_mc_subset=df_subset,
            human_dis=human_dis,
            llm_dis=llm_dis,
            delta_llm=delta_llm,
            p_val_llm=p_val_llm,
            current_n=n,
            max_n=max_iterations,
        )
        frames.append(frame)
    
    # Set durations (longer pause on final frame)
    durations = [duration] * len(frames)
    durations[-1] = final_pause
    
    # Save GIF
    print(f"   Saving GIF to: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,  # Infinite loop
    )
    
    # Report file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ GIF saved! Size: {file_size:.2f} MB")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate animated GIF of Monte Carlo robustness analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # New interface (cleaned data)
    parser.add_argument("--data", help="Path to cleaned CSV file (new interface)")
    parser.add_argument("--metric", help="Metric name for multi-metric datasets like newsroom "
                       "(e.g., informativeness, relevance, fluency, coherence)")
    
    # Legacy interface (separate files)
    parser.add_argument("--llm-ratings", help="[Legacy] Path to LLM ratings CSV")
    parser.add_argument("--human-judgments", help="[Legacy] Path to human judgments CSV")
    
    # Common arguments
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--llm-col", required=True, help="LLM column name to analyze")
    parser.add_argument("--min-score", type=int, required=True, help="Minimum score value")
    parser.add_argument("--max-score", type=int, required=True, help="Maximum score value")
    parser.add_argument("--max-iterations", type=int, default=25000, help="Max Monte Carlo iterations")
    parser.add_argument("--frame-iterations", type=str, default=None, 
                       help="Comma-separated list of iteration counts for frames "
                            "(e.g., '100,500,1000,5000,10000,25000'). "
                            "If not provided, uses logarithmic spacing up to max-iterations.")
    parser.add_argument("--duration", type=int, default=400, help="Frame duration in ms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate arguments
    use_new_interface = args.data is not None
    use_legacy_interface = args.llm_ratings is not None and args.human_judgments is not None
    
    if not use_new_interface and not use_legacy_interface:
        parser.error("Either --data (new interface) or both --llm-ratings and --human-judgments (legacy) required")
    
    if use_new_interface and use_legacy_interface:
        parser.error("Cannot use both --data and --llm-ratings/--human-judgments. Choose one interface.")
    
    print("=" * 60)
    print("Monte Carlo Robustness GIF Generator")
    print("=" * 60)
    
    if use_new_interface:
        print(f"Data file:       {args.data}")
        if args.metric:
            print(f"Metric:          {args.metric}")
    else:
        print(f"LLM Ratings:     {args.llm_ratings}")
        print(f"Human Judgments: {args.human_judgments}")
    
    print(f"LLM column:      {args.llm_col}")
    print(f"Output:          {args.output}")
    print(f"Score range:     [{args.min_score}, {args.max_score}]")
    print(f"Max iterations:  {args.max_iterations:,}")
    if args.frame_iterations:
        print(f"Frame iterations: {args.frame_iterations}")
    print("=" * 60)
    
    # Parse frame iterations if provided
    frame_iterations = None
    if args.frame_iterations:
        try:
            frame_iterations = [int(x.strip()) for x in args.frame_iterations.split(",")]
            # Sort and ensure max_iterations is respected
            frame_iterations = sorted([n for n in frame_iterations if n <= args.max_iterations])
            if not frame_iterations:
                raise ValueError("No valid frame iterations after filtering")
        except ValueError as e:
            parser.error(f"Invalid --frame-iterations format: {e}. Use comma-separated integers.")
    
    # Load data
    print("\n📊 Loading and preparing data...")
    
    if use_new_interface:
        merged_df, human_cols, llm_cols = load_cleaned_data(
            args.data,
            args.llm_col,
            args.metric,
        )
    else:
        merged_df, human_cols, llm_cols = prepare_data_legacy(
            args.llm_ratings,
            args.human_judgments,
            args.llm_col,
        )
    
    print(f"   Data shape:    {merged_df.shape}")
    print(f"   Human columns: {human_cols}")
    print(f"   LLM column:    {args.llm_col}")
    
    # Create evaluator
    print("\n🔧 Initializing LLMGoodEnough evaluator...")
    judge = LLMGoodEnough(
        df=merged_df,
        human_cols=human_cols,
        llm_cols=[args.llm_col],
        min_score=args.min_score,
        max_score=args.max_score,
        verbosity=0,
        seed=args.seed,
    )
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate GIF
    print("\n🎬 Generating animation...")
    generate_gif(
        judge=judge,
        llm_col=args.llm_col,
        output_path=args.output,
        max_iterations=args.max_iterations,
        frame_iterations=frame_iterations,
        duration=args.duration,
    )
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
