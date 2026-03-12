# ============================================================
# AutoMorphoTrack – Organelle Shape Feature Extraction
# ============================================================

import tifffile, numpy as np, pandas as pd, cv2, seaborn as sns, matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import clear_border
from pathlib import Path
from automorphotrack.utils import ensure_dir, save_high_dpi, upscale_frame

# Colorblind-friendly palette (Okabe-Ito inspired)
CB_MITO = "#0173B2"   # blue
CB_LYSO = "#DE8F05"   # orange

def detect_mask(frame, ch, min_size=3, thr_factor=0.8):
    """Threshold and clean a single channel."""
    img = frame[..., ch].astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)
    thr = threshold_otsu(img) * thr_factor
    mask = clear_border(binary_opening(img > thr, footprint=disk(1)))
    mask = remove_small_objects(mask, min_size)
    return (mask > 0).astype(np.uint8), img

def analyze_shape_features(
    tif_path="Composite.tif",
    out_dir="Shape_Feature_Outputs",
    mito_channel=0,
    lyso_channel=1):

    ensure_dir(out_dir)
    stack = tifffile.imread(tif_path)
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for shape feature extraction")

    mito_records, lyso_records = [], []

    for f in range(n_frames):
        # ---------- Mitochondria ----------
        mito_mask, mito_img = detect_mask(stack[f], mito_channel, min_size=5, thr_factor=0.85)
        lbl_m = label(mito_mask)
        for p in regionprops(lbl_m):
            if p.area < 5:
                continue
            # Use perimeter_crofton for more accurate continuous perimeter estimate
            # to avoid circularity > 1 due to discrete pixel-boundary underestimation
            perim = getattr(p, 'perimeter_crofton', p.perimeter)
            circ = (4 * np.pi * p.area) / (perim ** 2 + 1e-9)
            circ = min(circ, 1.0)  # Clamp to [0, 1] for physical validity
            aspect = p.major_axis_length / (p.minor_axis_length + 1e-9)
            mito_records.append({
                "Frame": f,
                "Area": p.area,
                "Eccentricity": p.eccentricity,
                "Solidity": p.solidity,
                "Circularity": circ,
                "Aspect_Ratio": aspect,
                "Orientation": p.orientation,
            })

        # ---------- Lysosomes ----------
        lyso_mask, lyso_img = detect_mask(stack[f], lyso_channel, min_size=2, thr_factor=0.85)
        lbl_l = label(lyso_mask)
        for p in regionprops(lbl_l):
            if p.area < 5:
                continue
            perim = getattr(p, 'perimeter_crofton', p.perimeter)
            circ = (4 * np.pi * p.area) / (perim ** 2 + 1e-9)
            circ = min(circ, 1.0)  # Clamp to [0, 1] for physical validity
            aspect = p.major_axis_length / (p.minor_axis_length + 1e-9)
            lyso_records.append({
                "Frame": f,
                "Area": p.area,
                "Eccentricity": p.eccentricity,
                "Solidity": p.solidity,
                "Circularity": circ,
                "Aspect_Ratio": aspect,
                "Orientation": p.orientation,
            })

    mito_df = pd.DataFrame(mito_records)
    lyso_df = pd.DataFrame(lyso_records)

    mito_csv = Path(out_dir) / "Mito_ShapeMetrics.csv"
    lyso_csv = Path(out_dir) / "Lyso_ShapeMetrics.csv"
    mito_df.to_csv(mito_csv, index=False)
    lyso_df.to_csv(lyso_csv, index=False)
    print(f"Saved shape metrics for {len(mito_df)} mitochondria and {len(lyso_df)} lysosomes")

    # ---------- KDE Distribution Plots ----------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.kdeplot(mito_df["Circularity"], ax=axes[0], color=CB_MITO, fill=True, alpha=0.5, label="Mito", clip=(0, 1))
    sns.kdeplot(lyso_df["Circularity"], ax=axes[0], color=CB_LYSO, fill=True, alpha=0.4, label="Lyso", clip=(0, 1))
    axes[0].set_title("Circularity Distribution")
    axes[0].set_xlabel("Circularity (4\u03C0A/P\u00B2)")
    axes[0].set_xlim(0, 1)
    axes[0].legend(frameon=False)

    sns.kdeplot(mito_df["Solidity"], ax=axes[1], color=CB_MITO, fill=True, alpha=0.5, label="Mito", clip=(0, 1))
    sns.kdeplot(lyso_df["Solidity"], ax=axes[1], color=CB_LYSO, fill=True, alpha=0.4, label="Lyso", clip=(0, 1))
    axes[1].set_title("Solidity Distribution")
    axes[1].set_xlabel("Solidity")
    axes[1].legend(frameon=False)

    plt.tight_layout()
    save_high_dpi(fig, Path(out_dir) / "Shape_Distributions.png")

    print(f"Shape feature analysis complete — results saved in {Path(out_dir).resolve()}")
