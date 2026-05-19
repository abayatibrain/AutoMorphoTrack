# ============================================================
# AutoMorphoTrack – Colocalization Analysis (Manders + Pearson)
# ============================================================

import tifffile, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from pathlib import Path
from automorphotrack.utils import ensure_dir, save_high_dpi

# Colorblind-friendly palette
CB_MITO = "#0173B2"   # blue
CB_LYSO = "#DE8F05"   # orange

def analyze_colocalization(
    tif_path="Composite.tif",
    mito_channel=0,
    lyso_channel=1,
    out_dir="Colocalization_Outputs",
    fps=5,
    upscale=4.0,
    overlay_alpha=0.6):

    ensure_dir(out_dir)
    stack = tifffile.imread(tif_path)
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for colocalization analysis")

    # ---------- Helpers ----------
    def upscale_frame(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)

    def write_video(frames, out_path, fps):
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            out.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        out.release()

    frames_out = []
    overlap_percent, m1_list, m2_list = [], [], []
    pearson_r, overlap_R = [], []
    mito_sum, lyso_sum = [], []

    for f in range(n_frames):
        mito = stack[f][..., mito_channel].astype(float)
        lyso = stack[f][..., lyso_channel].astype(float)

        # Background correction
        mito -= np.percentile(mito, 5)
        lyso -= np.percentile(lyso, 5)
        mito, lyso = np.clip(mito, 0, None), np.clip(lyso, 0, None)

        # Binary masks
        thr_m, thr_l = threshold_otsu(mito), threshold_otsu(lyso)
        mito_mask, lyso_mask = mito > thr_m, lyso > thr_l
        overlap_mask, union_mask = mito_mask & lyso_mask, mito_mask | lyso_mask

        # Metrics
        overlap_ratio = (overlap_mask.sum() / union_mask.sum() * 100) if union_mask.sum() else 0
        overlap_percent.append(overlap_ratio)
        M1 = mito[lyso_mask].sum() / mito.sum() if mito.sum() > 0 else 0
        M2 = lyso[mito_mask].sum() / lyso.sum() if lyso.sum() > 0 else 0
        m1_list.append(M1)
        m2_list.append(M2)

        flat_m, flat_l = mito.flatten(), lyso.flatten()
        if np.std(flat_m) > 0 and np.std(flat_l) > 0:
            pearson_r.append(np.corrcoef(flat_m, flat_l)[0, 1])
        else:
            pearson_r.append(0)
        denom = np.sqrt((mito**2).sum() * (lyso**2).sum())
        overlap_R.append((mito * lyso).sum() / denom if denom > 0 else 0)

        mito_sum.append(mito.sum())
        lyso_sum.append(lyso.sum())

        # Visualization with CB-safe blue (mito) / orange (lyso) overlay
        mito_norm = (mito / (mito.max() + 1e-12) * 255).astype(np.uint8)
        lyso_norm = (lyso / (lyso.max() + 1e-12) * 255).astype(np.uint8)
        rgb = np.zeros((*mito.shape, 3), np.uint8)
        # Mito → blue (1, 115, 178)
        rgb[..., 0] = (mito_norm * (1 / 255)).astype(np.uint8)
        rgb[..., 1] = (mito_norm * (115 / 255)).astype(np.uint8)
        rgb[..., 2] = (mito_norm * (178 / 255)).astype(np.uint8)
        # Lyso → orange (222, 143, 5) added on top
        rgb[..., 0] = np.clip(rgb[..., 0].astype(int) + (lyso_norm * (222 / 255)).astype(int), 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(rgb[..., 1].astype(int) + (lyso_norm * (143 / 255)).astype(int), 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(rgb[..., 2].astype(int) + (lyso_norm * (5 / 255)).astype(int), 0, 255).astype(np.uint8)

        mask = overlap_mask.astype(bool)
        white_overlay = np.full_like(rgb[mask], (255, 255, 255), dtype=np.uint8)
        rgb[mask] = cv2.addWeighted(rgb[mask], 1 - overlay_alpha, white_overlay, overlay_alpha, 0)

        frames_out.append(upscale_frame(rgb))

    # ---------- Save metrics ----------
    # Column naming reflects R2.5(iii): each metric explicitly documents whether
    # it is computed on binary masks or on raw intensity, and the unit/scale.
    csv_path = Path(out_dir) / "Colocalization.csv"
    pd.DataFrame({
        "Frame": np.arange(n_frames),
        "Jaccard_overlap_mask": np.array(overlap_percent) / 100.0,   # binary masks, 0–1
        "Jaccard_percent_mask": overlap_percent,                     # same, 0–100
        "Manders_M1_intensity": m1_list,                             # raw intensity, 0–1
        "Manders_M2_intensity": m2_list,                             # raw intensity, 0–1
        "Pearson_r_intensity": pearson_r,                            # raw intensity, -1..1
        "Cosine_similarity_intensity": overlap_R,                    # raw intensity, 0–1
        "Mito_total_intensity": mito_sum,
        "Lyso_total_intensity": lyso_sum,
    }).to_csv(csv_path, index=False)
    # Companion definitions file so legends are unambiguous in publications.
    defs_path = Path(out_dir) / "Colocalization_metric_definitions.csv"
    pd.DataFrame([
        {"Metric": "Jaccard_overlap_mask",
         "Basis": "binary mask",
         "Definition": "|A ∩ B| / |A ∪ B|, range 0–1, 1 = perfect overlap"},
        {"Metric": "Manders_M1_intensity",
         "Basis": "raw intensity",
         "Definition": "Σ Mito intensity inside Lyso mask / Σ Mito intensity, 0–1"},
        {"Metric": "Manders_M2_intensity",
         "Basis": "raw intensity",
         "Definition": "Σ Lyso intensity inside Mito mask / Σ Lyso intensity, 0–1"},
        {"Metric": "Pearson_r_intensity",
         "Basis": "raw intensity",
         "Definition": "Pearson correlation of paired Mito/Lyso pixel intensities, -1..1"},
        {"Metric": "Cosine_similarity_intensity",
         "Basis": "raw intensity",
         "Definition": "Σ(Mito·Lyso) / sqrt(Σ Mito² · Σ Lyso²), 0–1"},
    ]).to_csv(defs_path, index=False)
    print(f"Saved metrics → {csv_path}")
    print(f"Saved metric definitions → {defs_path}")

    # ---------- Save video ----------
    video_path = Path(out_dir) / "Colocalization_BrightBlueOverlay.mp4"
    write_video(frames_out, video_path, fps=fps)

    # ---------- Save frame 0 still ----------
    frame0_path = Path(out_dir) / "Colocalization_Frame0.png"
    cv2.imwrite(str(frame0_path),
                cv2.cvtColor(frames_out[0], cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 0])

    # ---------- Plot metrics ----------
    plot_path = Path(out_dir) / "Colocalization_MetricsPlot.png"
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(m1_list, color=CB_MITO, linestyle='-', label='Manders M1 \u2014 intensity (mito\u2192lyso mask)')
    ax.plot(m2_list, color=CB_LYSO, linestyle='-', label='Manders M2 \u2014 intensity (lyso\u2192mito mask)')
    ax.plot(np.array(overlap_percent) / 100, color='#009E73', linestyle='--',
            label='Jaccard \u2014 binary masks')
    ax.plot(pearson_r, color='#000000', linestyle='-.', label='Pearson r \u2014 raw intensity')
    ax.plot(overlap_R, color='#CC79A7', linestyle=':', label='Cosine similarity \u2014 raw intensity')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Coefficient / Fraction (0–1)")
    ax.set_ylim(0, 1)
    ax.legend(loc='best', fontsize=9, frameon=True)
    ax.set_title("Colocalization Metrics Over Time\n(intensity- vs mask-based metrics distinguished)")
    plt.tight_layout()
    save_high_dpi(fig, plot_path)

    print(f"Colocalization analysis complete — results saved in {Path(out_dir).resolve()}")
