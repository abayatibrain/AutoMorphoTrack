# ============================================================
# AutoMorphoTrack – Motility Analysis
# ============================================================

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from automorphotrack.utils import ensure_dir, save_high_dpi

# Colorblind-friendly palette (Okabe-Ito inspired)
CB_MITO = "#0173B2"   # blue
CB_LYSO = "#DE8F05"   # orange

def analyze_motility(
    mito_tracks_path="Tracking_Outputs/Mito_Tracks.csv",
    lyso_tracks_path="Tracking_Outputs/Lyso_Tracks.csv",
    out_dir="Motility_Outputs",
    fps=5):

    ensure_dir(out_dir)

    # ---------- Load data ----------
    mito_df = pd.read_csv(mito_tracks_path)
    lyso_df = pd.read_csv(lyso_tracks_path)
    print(f"Loaded {len(mito_df)} mitochondrial and {len(lyso_df)} lysosomal coordinates")

    # ---------- Compute displacement & velocity ----------
    def compute_motility(df, label):
        df = df.sort_values(["Organelle", "Frame"])
        df["DX"] = df.groupby("Organelle")["X"].diff()
        df["DY"] = df.groupby("Organelle")["Y"].diff()
        df["Displacement"] = np.sqrt(df["DX"]**2 + df["DY"]**2)
        df["Velocity"] = df["Displacement"]

        # Per-organelle summary
        summary = (
            df.groupby("Organelle")
            .agg({
                "Displacement": ["mean", "sum"],
                "Velocity": "mean"
            })
            .reset_index()
        )
        summary.columns = ["Organelle", "Mean_Displacement",
                           "Total_Displacement", "Mean_Velocity"]
        summary["Organelle_Type"] = label

        # Per-frame mean values
        frame_summary = (
            df.groupby("Frame")[["Displacement", "Velocity"]]
            .mean()
            .reset_index()
            .rename(columns={
                "Displacement": "Mean_Displacement",
                "Velocity": "Mean_Velocity"
            })
        )
        frame_summary["Organelle_Type"] = label
        return df, summary, frame_summary

    mito_full, mito_summary, mito_frame = compute_motility(mito_df, "Mitochondria")
    lyso_full, lyso_summary, lyso_frame = compute_motility(lyso_df, "Lysosomes")

    # ---------- Combine summaries ----------
    combined_summary = pd.concat([mito_summary, lyso_summary], ignore_index=True)
    combined_frame = pd.concat([mito_frame, lyso_frame], ignore_index=True)

    # ---------- Save CSVs ----------
    summary_csv = Path(out_dir) / "Motility_Summary.csv"
    frame_csv = Path(out_dir) / "Motility_PerFrame.csv"
    combined_summary.to_csv(summary_csv, index=False)
    combined_frame.to_csv(frame_csv, index=False)

    print(f"Saved organelle summary → {summary_csv}")
    print(f"Saved per-frame summary → {frame_csv}")

    # ---------- Distribution Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # clip=(0, None) prevents KDE from extending into physically invalid negative values
    sns.kdeplot(mito_summary["Mean_Velocity"], color=CB_MITO, fill=True, ax=axes[0], label="Mitochondria", clip=(0, None))
    sns.kdeplot(lyso_summary["Mean_Velocity"], color=CB_LYSO, fill=True, alpha=0.4, ax=axes[0], label="Lysosomes", clip=(0, None))
    axes[0].set_title("Mean Velocity Distribution")
    axes[0].set_xlabel("Velocity (px/frame)")
    axes[0].set_xlim(left=0)
    axes[0].legend()

    sns.kdeplot(mito_summary["Total_Displacement"], color=CB_MITO, fill=True, ax=axes[1], label="Mitochondria", clip=(0, None))
    sns.kdeplot(lyso_summary["Total_Displacement"], color=CB_LYSO, fill=True, alpha=0.4, ax=axes[1], label="Lysosomes", clip=(0, None))
    axes[1].set_title("Total Displacement Distribution")
    axes[1].set_xlabel("Displacement (px)")
    axes[1].set_xlim(left=0)
    axes[1].legend()

    plt.tight_layout()
    save_high_dpi(fig, Path(out_dir) / "Motility_Distributions.png")

    # ---------- Scatter Plot ----------
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(data=mito_summary, x="Total_Displacement", y="Mean_Velocity",
                    color=CB_MITO, s=30, label="Mitochondria")
    sns.scatterplot(data=lyso_summary, x="Total_Displacement", y="Mean_Velocity",
                    color=CB_LYSO, s=30, alpha=0.6, label="Lysosomes")
    ax.set_xlabel("Total Displacement (px)")
    ax.set_ylabel("Mean Velocity (px/frame)")
    ax.set_title("Motility Scatter Plot")
    ax.legend()
    plt.tight_layout()
    save_high_dpi(fig, Path(out_dir) / "Motility_Scatter.png")

    print(f"Motility analysis complete — outputs saved in {Path(out_dir).resolve()}")
