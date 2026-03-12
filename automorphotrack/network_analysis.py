# ============================================================
# AutoMorphoTrack – Mitochondrial Network Topology Analysis
# ============================================================
# Quantifies mitochondrial network structure including
# branch points, endpoints, fragmentation index, and
# network connectivity via skeletonization.

import numpy as np, pandas as pd, matplotlib.pyplot as plt, tifffile, cv2
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize, remove_small_objects, binary_opening, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from automorphotrack.utils import ensure_dir, save_high_dpi, upscale_frame

# Colorblind-friendly palette
CB_MITO = "#0173B2"
CB_LYSO = "#DE8F05"
CB_ACCENT = "#CC79A7"


def count_skeleton_features(skel):
    """Count branch points and endpoints in a binary skeleton.

    A pixel is a branch point if it has >2 neighbors,
    an endpoint if it has exactly 1 neighbor.
    """
    pad = np.pad(skel.astype(np.uint8), 1, mode='constant')
    neighbors = np.zeros_like(pad, dtype=int)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            neighbors += np.roll(np.roll(pad, dy, axis=0), dx, axis=1)
    neighbors = neighbors[1:-1, 1:-1]
    branch_pts = np.sum((skel > 0) & (neighbors > 2))
    endpoints = np.sum((skel > 0) & (neighbors == 1))
    skel_length = np.sum(skel > 0)
    return int(branch_pts), int(endpoints), int(skel_length)


def fragmentation_index(n_objects, total_area):
    """Fragmentation index: ratio of object count to total area.

    Higher values indicate more fragmented networks.
    """
    if total_area == 0:
        return 0.0
    return float(n_objects / total_area) * 1000  # per 1000 px²


def network_connectivity(branch_pts, endpoints, skel_length):
    """Simple connectivity metric: branch_pts / (endpoints + 1).

    Higher values indicate more interconnected networks.
    """
    return float(branch_pts / (endpoints + 1))


def analyze_network_topology(
    tif_path="Composite.tif",
    mito_channel=0,
    out_dir="Network_Analysis_Outputs",
    upscale=4,
    min_size=5,
    thr_factor=0.85):
    """
    Analyze mitochondrial network topology from fluorescence stacks.

    For each frame, segments mito channel, skeletonizes, and computes:
      - Branch points, endpoints, skeleton length
      - Fragmentation index
      - Network connectivity
      - Number and mean size of connected components
    """
    ensure_dir(out_dir)
    out = Path(out_dir)

    stack = tifffile.imread(tif_path)
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for network topology analysis")

    records = []
    skel_frames = []

    for f in range(n_frames):
        mito = stack[f][..., mito_channel].astype(float)
        mito = (mito - mito.min()) / (np.ptp(mito) + 1e-12)

        thr = threshold_otsu(mito) * thr_factor
        mask = binary_opening(mito > thr, footprint=disk(1))
        mask = clear_border(mask)
        mask = remove_small_objects(mask, min_size=min_size)

        # Connected components
        lbl = label(mask)
        props = regionprops(lbl)
        n_objects = len(props)
        total_area = sum(p.area for p in props)
        mean_area = total_area / n_objects if n_objects else 0

        # Skeletonize
        skel = skeletonize(mask > 0)
        bp, ep, skel_len = count_skeleton_features(skel)
        frag = fragmentation_index(n_objects, total_area)
        conn = network_connectivity(bp, ep, skel_len)

        records.append({
            "Frame": f,
            "N_Components": n_objects,
            "Total_Area": total_area,
            "Mean_Area": mean_area,
            "Skeleton_Length": skel_len,
            "Branch_Points": bp,
            "Endpoints": ep,
            "Fragmentation_Index": frag,
            "Connectivity": conn,
        })

        # Visualization: overlay skeleton on mito
        mito_norm = (mito / (mito.max() + 1e-12) * 255).astype(np.uint8)
        rgb = np.zeros((*mito.shape, 3), np.uint8)
        # Blue channel for mito
        rgb[..., 0] = (mito_norm * (1 / 255)).astype(np.uint8)
        rgb[..., 1] = (mito_norm * (115 / 255)).astype(np.uint8)
        rgb[..., 2] = (mito_norm * (178 / 255)).astype(np.uint8)
        # Skeleton in white
        rgb[skel > 0] = (255, 255, 255)

        skel_frames.append(upscale_frame(rgb, scale=upscale))

    # ---- Save CSV ----
    df = pd.DataFrame(records)
    df.to_csv(out / "Network_Topology.csv", index=False)
    print(f"Saved topology metrics for {n_frames} frames")

    # ---- Save frame 0 still ----
    if skel_frames:
        frame0_path = out / "Network_Frame0.png"
        cv2.imwrite(str(frame0_path),
                    cv2.cvtColor(skel_frames[0], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # ---- Plot 1: Network metrics time series ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(df["Frame"], df["N_Components"], color=CB_MITO, linewidth=1.2)
    axes[0].set_ylabel("# Components")
    axes[0].set_title("Mitochondrial Network Topology Over Time")

    axes[1].plot(df["Frame"], df["Branch_Points"], color=CB_ACCENT,
                 linewidth=1.2, label="Branch points")
    axes[1].plot(df["Frame"], df["Endpoints"], color=CB_LYSO,
                 linewidth=1.2, label="Endpoints")
    axes[1].set_ylabel("Count")
    axes[1].legend(frameon=False, fontsize=9)

    axes[2].plot(df["Frame"], df["Fragmentation_Index"], color=CB_MITO,
                 linewidth=1.2, label="Fragmentation")
    ax2 = axes[2].twinx()
    ax2.plot(df["Frame"], df["Connectivity"], color=CB_ACCENT,
             linewidth=1.2, linestyle='--', label="Connectivity")
    axes[2].set_ylabel("Fragmentation Index")
    ax2.set_ylabel("Connectivity")
    axes[2].set_xlabel("Frame")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)

    plt.tight_layout()
    save_high_dpi(fig, out / "Network_Topology_TimeSeries.png")

    # ---- Plot 2: Scatter — components vs fragmentation ----
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(df["N_Components"], df["Fragmentation_Index"],
                    c=df["Frame"], cmap="viridis", s=40, edgecolors='white',
                    linewidths=0.5)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Fragmentation Index")
    ax.set_title("Network Fragmentation vs Component Count")
    plt.colorbar(sc, ax=ax, label="Frame")
    plt.tight_layout()
    save_high_dpi(fig, out / "Fragmentation_Scatter.png")

    # ---- Plot 3: Bar chart summary ----
    means = df[["Branch_Points", "Endpoints", "N_Components"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [CB_ACCENT, CB_LYSO, CB_MITO]
    ax.barh(means.index, means.values, color=colors, edgecolor='white')
    ax.set_xlabel("Mean Count per Frame")
    ax.set_title("Average Network Features")
    plt.tight_layout()
    save_high_dpi(fig, out / "Network_Summary_Bar.png")

    print(f"Network topology analysis complete — results saved in {out.resolve()}")
