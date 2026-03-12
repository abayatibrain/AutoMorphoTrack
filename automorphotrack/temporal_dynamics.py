# ============================================================
# AutoMorphoTrack – Temporal Dynamics Analysis
# ============================================================
# Analyzes time-series behavior of organelle metrics including
# autocorrelation, change-point detection, and temporal clustering.

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import zscore
from automorphotrack.utils import ensure_dir, save_high_dpi

# Colorblind-friendly palette
CB_MITO = "#0173B2"
CB_LYSO = "#DE8F05"
CB_ACCENT = "#CC79A7"


def autocorrelation(series, max_lag=None):
    """Compute normalized autocorrelation of a 1-D signal."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 2
    var = np.sum(x ** 2)
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.correlate(x, x, mode='full')[n - 1:]
    acf = acf[:max_lag + 1] / var
    return acf


def detect_change_points(series, threshold=2.0):
    """Detect abrupt changes via first-difference z-score thresholding."""
    x = np.asarray(series, dtype=float)
    diff = np.diff(x)
    if np.std(diff) == 0:
        return np.array([], dtype=int)
    z = np.abs(zscore(diff))
    cps, _ = find_peaks(z, height=threshold)
    return cps + 1  # offset by 1 since diff shortens array


def temporal_stability_index(series):
    """Coefficient of variation as a measure of temporal stability."""
    x = np.asarray(series, dtype=float)
    mu = x.mean()
    if mu == 0:
        return 0.0
    return float(np.std(x) / mu)


def analyze_temporal_dynamics(
    tif_path="Composite.tif",
    out_dir="Temporal_Dynamics_Outputs",
    morphology_csv=None,
    lyso_count_csv=None,
    mito_tracks_csv=None,
    lyso_tracks_csv=None,
    coloc_csv=None):
    """
    Analyze temporal dynamics from pre-computed pipeline outputs.

    Reads per-frame CSVs from earlier pipeline steps and computes:
      - Autocorrelation of key metrics
      - Change-point detection
      - Temporal stability indices
      - Rolling-window summaries
    """
    ensure_dir(out_dir)
    out = Path(out_dir)
    parent = Path(tif_path).parent

    # Auto-discover CSVs if not provided
    def find_csv(name, subdir_candidates):
        for sd in subdir_candidates:
            p = parent / sd / name
            if p.exists():
                return p
        return None

    if morphology_csv is None:
        morphology_csv = find_csv("Morphology_Summary.csv",
                                  ["V2_Morphology", "Morphology_Outputs"])
    if lyso_count_csv is None:
        lyso_count_csv = find_csv("Lysosome_Counts.csv",
                                  ["V2_LysoCount", "Lyso_Count_Outputs"])
    if coloc_csv is None:
        coloc_csv = find_csv("Colocalization.csv",
                             ["V2_Colocalization", "Colocalization_Outputs"])
    if mito_tracks_csv is None:
        mito_tracks_csv = find_csv("Mito_Tracks.csv",
                                   ["V2_Tracking", "Tracking_Outputs"])
    if lyso_tracks_csv is None:
        lyso_tracks_csv = find_csv("Lyso_Tracks.csv",
                                   ["V2_Tracking", "Tracking_Outputs"])

    records = {}

    # ---- Morphology time series ----
    if morphology_csv and Path(morphology_csv).exists():
        mdf = pd.read_csv(morphology_csv)
        if "Elongated" in mdf.columns and "Punctate" in mdf.columns:
            total = mdf["Elongated"] + mdf["Punctate"]
            ratio = mdf["Elongated"] / total.replace(0, np.nan)
            records["Elongation_Ratio"] = ratio.values

    # ---- Lysosome counts ----
    if lyso_count_csv and Path(lyso_count_csv).exists():
        ldf = pd.read_csv(lyso_count_csv)
        count_col = [c for c in ldf.columns if "count" in c.lower() or "lyso" in c.lower()]
        if count_col:
            records["Lysosome_Count"] = ldf[count_col[0]].values

    # ---- Colocalization ----
    if coloc_csv and Path(coloc_csv).exists():
        cdf = pd.read_csv(coloc_csv)
        if "Manders_M1" in cdf.columns:
            records["Manders_M1"] = cdf["Manders_M1"].values
        if "Pearson_r" in cdf.columns:
            records["Pearson_r"] = cdf["Pearson_r"].values

    # ---- Motility (per-frame mean velocity) ----
    for label, csv_path in [("Mito_MeanVel", mito_tracks_csv),
                             ("Lyso_MeanVel", lyso_tracks_csv)]:
        if csv_path and Path(csv_path).exists():
            tdf = pd.read_csv(csv_path)
            if "Frame" in tdf.columns and "Velocity" in tdf.columns:
                per_frame = tdf.groupby("Frame")["Velocity"].mean()
                records[label] = per_frame.values

    if not records:
        print("No temporal data found — skipping temporal dynamics analysis.")
        return

    n_metrics = len(records)
    print(f"Analyzing temporal dynamics for {n_metrics} metrics")

    # ---- Compute autocorrelation ----
    max_lag = min(20, min(len(v) for v in records.values()) // 2)
    acf_data = {}
    for name, vals in records.items():
        acf_data[name] = autocorrelation(vals, max_lag)

    # ---- Change-point detection ----
    cp_data = {}
    for name, vals in records.items():
        cp_data[name] = detect_change_points(vals, threshold=1.8)

    # ---- Stability indices ----
    stability = {name: temporal_stability_index(vals)
                 for name, vals in records.items()}

    # ---- Save CSV ----
    stab_df = pd.DataFrame([
        {"Metric": k, "CV": v, "N_ChangePoints": len(cp_data[k])}
        for k, v in stability.items()
    ])
    stab_df.to_csv(out / "Temporal_Stability.csv", index=False)

    # ---- Plot 1: Autocorrelation ----
    colors = [CB_MITO, CB_LYSO, CB_ACCENT, "#56B4E9", "#009E73", "#D55E00"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, acf) in enumerate(acf_data.items()):
        ax.plot(acf, label=name, color=colors[i % len(colors)], linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Temporal Autocorrelation of Organelle Metrics")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save_high_dpi(fig, out / "Temporal_Autocorrelation.png")

    # ---- Plot 2: Time series with change points ----
    n_plots = min(n_metrics, 4)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    for i, (name, vals) in enumerate(list(records.items())[:n_plots]):
        ax = axes[i]
        frames = np.arange(len(vals))
        ax.plot(frames, vals, color=colors[i % len(colors)], linewidth=1.2)
        # Mark change points
        cps = cp_data[name]
        for cp in cps:
            if cp < len(vals):
                ax.axvline(cp, color='red', alpha=0.5, linestyle=':', linewidth=1)
        ax.set_ylabel(name.replace("_", " "), fontsize=9)
        ax.tick_params(labelsize=8)
        cv = stability[name]
        ax.text(0.98, 0.92, f"CV={cv:.3f}", transform=ax.transAxes,
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Temporal Profiles with Change Points", fontsize=12, y=1.01)
    plt.tight_layout()
    save_high_dpi(fig, out / "Temporal_Profiles.png")

    # ---- Plot 3: Stability bar chart ----
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(stability.keys())
    cvs = [stability[n] for n in names]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    ax.barh(names, cvs, color=bar_colors, edgecolor='white')
    ax.set_xlabel("Coefficient of Variation")
    ax.set_title("Temporal Stability of Organelle Metrics")
    plt.tight_layout()
    save_high_dpi(fig, out / "Temporal_Stability_BarChart.png")

    print(f"Temporal dynamics analysis complete — results saved in {out.resolve()}")
