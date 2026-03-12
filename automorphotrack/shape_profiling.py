# ============================================================
# AutoMorphoTrack – Shape Profiling (Combined Organelle Metrics)
# ============================================================
# Author: Armin Bayati, Ph.D.
# Description:
#     Comprehensive shape profiling module for mitochondrial and
#     lysosomal morphometric analysis with statistical comparisons.
#
# Functions:
#     - profile_shape_data(): Combined violin plots + statistics
#     - compare_shape_distributions(): Statistical comparison
#     - plot_individual_metrics(): Detailed metric distributions
#     - export_shape_summary(): Summary statistics CSV
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
from automorphotrack.utils import ensure_dir, save_high_dpi
import warnings

# Colorblind-safe palette
CB_MITO = "#0173B2"
CB_LYSO = "#DE8F05"


def profile_shape_data(
    mito_shape_path="Shape_Feature_Outputs/Mito_ShapeMetrics.csv",
    lyso_shape_path="Shape_Feature_Outputs/Lyso_ShapeMetrics.csv",
    out_dir="Shape_Profiling_Outputs"):
    """
    Generate combined violin plots for mitochondrial and lysosomal shape metrics.

    Parameters
    ----------
    mito_shape_path : str
        Path to mitochondrial shape metrics CSV.
    lyso_shape_path : str
        Path to lysosomal shape metrics CSV.
    out_dir : str, default "Shape_Profiling_Outputs"
        Output directory for profiling results.

    Returns
    -------
    pd.DataFrame
        Combined shape data DataFrame.
    """
    ensure_dir(out_dir)

    try:
        # ---------- Load and label datasets ----------
        mito_df = pd.read_csv(mito_shape_path).assign(Type="Mitochondria")
        lyso_df = pd.read_csv(lyso_shape_path).assign(Type="Lysosomes")
        combined_df = pd.concat([mito_df, lyso_df], ignore_index=True)
    except FileNotFoundError as e:
        warnings.warn(f"Could not load shape data: {e}")
        return pd.DataFrame()

    # ---------- Save combined CSV ----------
    combined_csv_path = Path(out_dir) / "Combined_ShapeData.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined shape data CSV → {combined_csv_path}")

    # ---------- Violin plots ----------
    metrics = ["Circularity", "Solidity", "Aspect_Ratio"]

    # Filter to existing metrics
    available_metrics = [m for m in metrics if m in combined_df.columns]

    if not available_metrics:
        warnings.warn(f"No shape metrics found in data. Available columns: {combined_df.columns.tolist()}")
        return combined_df

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(10*len(available_metrics), 8))

    # Handle single metric case
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        sns.violinplot(
            data=combined_df,
            x="Type",
            y=metric,
            palette={"Mitochondria": CB_MITO, "Lysosomes": CB_LYSO},
            cut=0,
            inner="quartile",
            ax=ax
        )
        ax.set_title(f"{metric}", fontsize=12, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = Path(out_dir) / "Shape_ViolinPlots.png"
    save_high_dpi(fig, out_path)

    print(f"Shape profiling complete → {Path(out_dir).resolve()}")
    return combined_df


def compare_shape_distributions(combined_df, out_dir="Shape_Profiling_Outputs", alpha=0.05):
    """
    Perform statistical comparison of shape metrics between organelles.

    Uses Mann-Whitney U test (Spearman rank-based) for distribution comparison.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined shape data with 'Type' column.
    out_dir : str, default "Shape_Profiling_Outputs"
        Output directory for statistics.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Statistical comparison results.
    """
    ensure_dir(out_dir)

    metrics = [col for col in combined_df.columns if col not in ['Type', 'ID', 'Frame']]

    stats_list = []
    for metric in metrics:
        if metric not in combined_df.columns:
            continue

        mito_data = combined_df[combined_df['Type'] == 'Mitochondria'][metric].dropna()
        lyso_data = combined_df[combined_df['Type'] == 'Lysosomes'][metric].dropna()

        if len(mito_data) == 0 or len(lyso_data) == 0:
            continue

        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(mito_data, lyso_data)

        # Effect size (rank-biserial correlation)
        n1, n2 = len(mito_data), len(lyso_data)
        r = 1 - (2*u_stat) / (n1 * n2)

        stats_list.append({
            'Metric': metric,
            'Mito_Mean': mito_data.mean(),
            'Lyso_Mean': lyso_data.mean(),
            'Mito_Std': mito_data.std(),
            'Lyso_Std': lyso_data.std(),
            'U_Statistic': u_stat,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < alpha else 'No',
            'Effect_Size': r
        })

    stats_df = pd.DataFrame(stats_list)
    stats_path = Path(out_dir) / "Shape_Statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved statistical comparison → {stats_path}")

    return stats_df


def plot_individual_metrics(combined_df, out_dir="Shape_Profiling_Outputs"):
    """
    Create detailed distribution plots for each shape metric.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined shape data with 'Type' column.
    out_dir : str
        Output directory for plots.

    Returns
    -------
    None
    """
    ensure_dir(out_dir)

    metrics = [col for col in combined_df.columns if col not in ['Type', 'ID', 'Frame']]

    for metric in metrics:
        if metric not in combined_df.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Distribution plot
        for organelle_type, ax in zip(['Mitochondria', 'Lysosomes'], axes):
            data = combined_df[combined_df['Type'] == organelle_type][metric].dropna()
            color = CB_MITO if organelle_type == 'Mitochondria' else CB_LYSO

            ax.hist(data, bins=30, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            ax.set_title(f"{organelle_type} - {metric}", fontsize=12, fontweight='bold')
            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3, axis='y')

        plt.tight_layout()
        out_path = Path(out_dir) / f"Distribution_{metric}.png"
        save_high_dpi(fig, out_path)


def export_shape_summary(combined_df, out_dir="Shape_Profiling_Outputs"):
    """
    Export summary statistics for all shape metrics.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Combined shape data with 'Type' column.
    out_dir : str
        Output directory for summary.

    Returns
    -------
    pd.DataFrame
        Summary statistics.
    """
    ensure_dir(out_dir)

    metrics = [col for col in combined_df.columns if col not in ['Type', 'ID', 'Frame']]

    summary_list = []
    for organelle_type in ['Mitochondria', 'Lysosomes']:
        subset = combined_df[combined_df['Type'] == organelle_type]

        for metric in metrics:
            if metric not in subset.columns:
                continue

            data = subset[metric].dropna()
            summary_list.append({
                'Type': organelle_type,
                'Metric': metric,
                'Mean': data.mean(),
                'Std': data.std(),
                'Median': data.median(),
                'Min': data.min(),
                'Max': data.max(),
                'Q1': data.quantile(0.25),
                'Q3': data.quantile(0.75),
                'N': len(data)
            })

    summary_df = pd.DataFrame(summary_list)
    summary_path = Path(out_dir) / "Shape_Summary_Statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary statistics → {summary_path}")

    return summary_df
