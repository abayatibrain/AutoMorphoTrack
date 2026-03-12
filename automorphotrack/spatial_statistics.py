# ============================================================
# AutoMorphoTrack – Spatial Statistics Analysis
# ============================================================
# Analyzes spatial distributions of organelles including
# nearest-neighbor distances, Ripley's K function, and
# spatial density mapping.

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from scipy.spatial import cKDTree
from automorphotrack.utils import ensure_dir, save_high_dpi

# Colorblind-friendly palette
CB_MITO = "#0173B2"
CB_LYSO = "#DE8F05"
CB_ACCENT = "#CC79A7"


def nearest_neighbor_distances(coords):
    """Compute nearest-neighbor distance for each point in a 2-D array."""
    if len(coords) < 2:
        return np.array([])
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)   # k=2 because self is k=1
    return dists[:, 1]


def ripleys_k(coords, radii, area):
    """Compute Ripley's K function for a set of 2-D points.

    K(r) = area / n^2  *  sum_{i!=j} I(d_ij < r)

    Returns K(r) for each radius in *radii*.
    """
    n = len(coords)
    if n < 2:
        return np.zeros_like(radii, dtype=float)
    tree = cKDTree(coords)
    K = np.zeros(len(radii))
    for i, r in enumerate(radii):
        pairs = tree.count_neighbors(tree, r)
        K[i] = area / (n ** 2) * (pairs - n)  # subtract self-pairs
    return K


def cross_nearest_neighbor(coords_a, coords_b):
    """Nearest-neighbor distance from each point in *a* to the closest in *b*."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.array([])
    tree_b = cKDTree(coords_b)
    dists, _ = tree_b.query(coords_a, k=1)
    return dists


def analyze_spatial_statistics(
    tif_path="Composite.tif",
    out_dir="Spatial_Statistics_Outputs",
    mito_tracks_csv=None,
    lyso_tracks_csv=None,
    morphology_csv=None):
    """
    Compute spatial statistics from tracking outputs.

    Reads per-frame organelle coordinates and computes:
      - Nearest-neighbor distance distributions (within and cross-organelle)
      - Ripley's K function
      - Spatial density heatmaps
      - Clark-Evans aggregation index
    """
    ensure_dir(out_dir)
    out = Path(out_dir)
    parent = Path(tif_path).parent

    # Auto-discover CSVs
    def find_csv(name, subdirs):
        for sd in subdirs:
            p = parent / sd / name
            if p.exists():
                return p
        return None

    if mito_tracks_csv is None:
        mito_tracks_csv = find_csv("Mito_Tracks.csv",
                                    ["V2_Tracking", "Tracking_Outputs"])
    if lyso_tracks_csv is None:
        lyso_tracks_csv = find_csv("Lyso_Tracks.csv",
                                    ["V2_Tracking", "Tracking_Outputs"])

    if not (mito_tracks_csv and Path(mito_tracks_csv).exists()):
        print("Mito tracks not found — skipping spatial statistics.")
        return
    if not (lyso_tracks_csv and Path(lyso_tracks_csv).exists()):
        print("Lyso tracks not found — skipping spatial statistics.")
        return

    mito_df = pd.read_csv(mito_tracks_csv)
    lyso_df = pd.read_csv(lyso_tracks_csv)
    print(f"Loaded {len(mito_df)} mito and {len(lyso_df)} lyso track points")

    # Use frame 0 for field-of-view area estimate
    all_x = np.concatenate([mito_df["X"].values, lyso_df["X"].values])
    all_y = np.concatenate([mito_df["Y"].values, lyso_df["Y"].values])
    fov_area = (all_x.max() - all_x.min()) * (all_y.max() - all_y.min())

    frames = sorted(set(mito_df["Frame"]) & set(lyso_df["Frame"]))
    if not frames:
        print("No overlapping frames — skipping spatial statistics.")
        return

    # ---- Per-frame metrics ----
    nn_mito_all, nn_lyso_all, cross_ml_all, cross_lm_all = [], [], [], []
    ce_mito, ce_lyso = [], []
    frame_ids = []

    for f in frames:
        mc = mito_df[mito_df["Frame"] == f][["X", "Y"]].values
        lc = lyso_df[lyso_df["Frame"] == f][["X", "Y"]].values

        # Within-organelle NND
        nn_m = nearest_neighbor_distances(mc)
        nn_l = nearest_neighbor_distances(lc)
        nn_mito_all.extend(nn_m)
        nn_lyso_all.extend(nn_l)

        # Cross-organelle NND
        cross_ml = cross_nearest_neighbor(mc, lc)
        cross_lm = cross_nearest_neighbor(lc, mc)
        cross_ml_all.extend(cross_ml)
        cross_lm_all.extend(cross_lm)

        # Clark-Evans index: R = mean(NND) / (0.5 * sqrt(area/n))
        n_m, n_l = len(mc), len(lc)
        if n_m > 1 and nn_m.size > 0:
            expected_m = 0.5 * np.sqrt(fov_area / n_m)
            ce_mito.append(nn_m.mean() / (expected_m + 1e-12))
        if n_l > 1 and nn_l.size > 0:
            expected_l = 0.5 * np.sqrt(fov_area / n_l)
            ce_lyso.append(nn_l.mean() / (expected_l + 1e-12))

        frame_ids.append(f)

    nn_mito_all = np.array(nn_mito_all)
    nn_lyso_all = np.array(nn_lyso_all)
    cross_ml_all = np.array(cross_ml_all)
    cross_lm_all = np.array(cross_lm_all)

    # ---- Ripley's K on representative frame (median frame) ----
    mid_f = frames[len(frames) // 2]
    mc_mid = mito_df[mito_df["Frame"] == mid_f][["X", "Y"]].values
    lc_mid = lyso_df[lyso_df["Frame"] == mid_f][["X", "Y"]].values
    max_r = np.sqrt(fov_area) / 4
    radii = np.linspace(0, max_r, 50)
    K_mito = ripleys_k(mc_mid, radii, fov_area)
    K_lyso = ripleys_k(lc_mid, radii, fov_area)
    K_csr = np.pi * radii ** 2  # complete spatial randomness

    # ---- Save CSV ----
    summary_records = []
    summary_records.append({
        "Metric": "Mean_NND_Mito", "Value": nn_mito_all.mean() if len(nn_mito_all) else 0})
    summary_records.append({
        "Metric": "Mean_NND_Lyso", "Value": nn_lyso_all.mean() if len(nn_lyso_all) else 0})
    summary_records.append({
        "Metric": "Mean_CrossNND_Mito_to_Lyso", "Value": cross_ml_all.mean() if len(cross_ml_all) else 0})
    summary_records.append({
        "Metric": "Mean_CrossNND_Lyso_to_Mito", "Value": cross_lm_all.mean() if len(cross_lm_all) else 0})
    summary_records.append({
        "Metric": "ClarkEvans_Mito", "Value": np.mean(ce_mito) if ce_mito else 0})
    summary_records.append({
        "Metric": "ClarkEvans_Lyso", "Value": np.mean(ce_lyso) if ce_lyso else 0})
    pd.DataFrame(summary_records).to_csv(out / "Spatial_Summary.csv", index=False)

    # Full NND distributions
    nnd_df = pd.DataFrame({
        "NND": np.concatenate([nn_mito_all, nn_lyso_all]),
        "Type": (["Mitochondria"] * len(nn_mito_all) +
                 ["Lysosomes"] * len(nn_lyso_all))
    })
    nnd_df.to_csv(out / "NND_Distributions.csv", index=False)
    print(f"Saved spatial statistics CSVs")

    # ---- Plot 1: NND distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if len(nn_mito_all) > 1:
        sns.kdeplot(nn_mito_all, ax=axes[0], color=CB_MITO, fill=True,
                    alpha=0.5, label="Mito", clip=(0, None))
    if len(nn_lyso_all) > 1:
        sns.kdeplot(nn_lyso_all, ax=axes[0], color=CB_LYSO, fill=True,
                    alpha=0.4, label="Lyso", clip=(0, None))
    axes[0].set_xlabel("Nearest-Neighbor Distance (px)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Within-Organelle NND")
    axes[0].set_xlim(left=0)
    axes[0].legend(frameon=False)

    if len(cross_ml_all) > 1:
        sns.kdeplot(cross_ml_all, ax=axes[1], color=CB_MITO, fill=True,
                    alpha=0.5, label="Mito→Lyso", clip=(0, None))
    if len(cross_lm_all) > 1:
        sns.kdeplot(cross_lm_all, ax=axes[1], color=CB_LYSO, fill=True,
                    alpha=0.4, label="Lyso→Mito", clip=(0, None))
    axes[1].set_xlabel("Cross-Organelle NND (px)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Cross-Organelle NND")
    axes[1].set_xlim(left=0)
    axes[1].legend(frameon=False)

    plt.tight_layout()
    save_high_dpi(fig, out / "NND_Distributions.png")

    # ---- Plot 2: Ripley's K ----
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(radii, K_mito, color=CB_MITO, linewidth=1.5, label="Mito K(r)")
    ax.plot(radii, K_lyso, color=CB_LYSO, linewidth=1.5, label="Lyso K(r)")
    ax.plot(radii, K_csr, 'k--', linewidth=1, alpha=0.6, label="CSR (πr²)")
    ax.set_xlabel("Radius r (px)")
    ax.set_ylabel("K(r)")
    ax.set_title("Ripley's K Function")
    ax.legend(frameon=False)
    plt.tight_layout()
    save_high_dpi(fig, out / "Ripleys_K.png")

    # ---- Plot 3: Spatial density heatmap (representative frame) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, coords, color, title in [
        (axes[0], mc_mid, CB_MITO, "Mito Density"),
        (axes[1], lc_mid, CB_LYSO, "Lyso Density"),
    ]:
        if len(coords) > 1:
            ax.hexbin(coords[:, 0], coords[:, 1], gridsize=20,
                      cmap="Blues" if color == CB_MITO else "Oranges",
                      mincnt=1)
            ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        ax.invert_yaxis()
    plt.tight_layout()
    save_high_dpi(fig, out / "Spatial_Density.png")

    # ---- Plot 4: Clark-Evans time series ----
    if ce_mito and ce_lyso:
        min_len = min(len(ce_mito), len(ce_lyso), len(frame_ids))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frame_ids[:min_len], ce_mito[:min_len],
                color=CB_MITO, linewidth=1.2, label="Mito R-index")
        ax.plot(frame_ids[:min_len], ce_lyso[:min_len],
                color=CB_LYSO, linewidth=1.2, label="Lyso R-index")
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8,
                    label="Random (R=1)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Clark-Evans R")
        ax.set_title("Spatial Aggregation Over Time")
        ax.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        save_high_dpi(fig, out / "ClarkEvans_TimeSeries.png")

    print(f"Spatial statistics analysis complete — results saved in {out.resolve()}")
