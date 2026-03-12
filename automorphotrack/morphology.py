# ============================================================
# AutoMorphoTrack – Mitochondrial Morphology Classification
# ============================================================

import tifffile, numpy as np, cv2, matplotlib.pyplot as plt, pandas as pd
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import clear_border
from pathlib import Path
from automorphotrack.utils import ensure_dir, upscale_frame, save_high_dpi

def detect_mask(frame, ch, thr_factor=0.8, min_size=10):
    """Threshold a channel to obtain clean mitochondrial mask."""
    img = frame[..., ch].astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)
    thr = threshold_otsu(img) * thr_factor
    mask = clear_border(binary_opening(img > thr, footprint=disk(1)))
    mask = remove_small_objects(mask, min_size)
    return (mask * 255).astype(np.uint8), img

def draw_text_with_outline(img, text, pos, font_scale, color, thickness, outline_color):
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, outline_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def classify_morphology(
    tif_path="Composite.tif",
    out_dir="Morphology_Outputs",
    mito_channel=0,
    area_thresh_elongated=0.12,
    ecc_thresh_elongated=0.80,
    area_thresh_punctate=0.03,
    ecc_thresh_punctate=0.60,
    upscale_factor=4,
    fps=5):

    ensure_dir(out_dir)
    stack = tifffile.imread(tif_path)
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    if stack.shape[-1] == 3:
        stack = stack[..., :2]
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for morphology classification")

    thr_stack, gray_stack = [], []
    for f in range(n_frames):
        m, g = detect_mask(stack[f], mito_channel)
        thr_stack.append(m)
        gray_stack.append(g)

    results, summary_counts = [], []

    for f in range(n_frames):
        lbl = label(thr_stack[f])
        props = regionprops(lbl, intensity_image=gray_stack[f])
        frame_res = []
        for p in props:
            ecc, area = p.eccentricity, p.area
            if area >= area_thresh_elongated and ecc >= ecc_thresh_elongated:
                morph = "Elongated"
            elif area >= area_thresh_punctate and ecc <= ecc_thresh_punctate:
                morph = "Punctate"
            else:
                morph = "Intermediate"
            frame_res.append({
                "Frame": f,
                "Label": p.label,
                "Area": area,
                "Eccentricity": ecc,
                "Morphology": morph
            })
        results += frame_res
        summary_counts.append({
            "Frame": f,
            "Elongated": sum(r["Morphology"] == "Elongated" for r in frame_res),
            "Punctate": sum(r["Morphology"] == "Punctate" for r in frame_res)
        })

    # Save CSV outputs
    pd.DataFrame(results).to_csv(Path(out_dir) / "Morphology_AllFrames.csv", index=False)
    pd.DataFrame(summary_counts).to_csv(Path(out_dir) / "Morphology_Summary.csv", index=False)
    print("Saved morphology classification CSVs")

    # Visualization: E/P-labeled video and frame 0
    video_frames = []
    font_scale = 0.45
    font_thickness = 1
    font_color_E = (0, 255, 0)
    font_color_P = (255, 255, 0)
    outline_color = (0, 0, 0)

    for f in range(n_frames):
        mito_gray = gray_stack[f]
        lbl = label(thr_stack[f])
        props = regionprops(lbl, intensity_image=mito_gray)
        base = (mito_gray * 255).astype(np.uint8)
        base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        base_up = upscale_frame(base_rgb, upscale_factor)

        elong_cnts, punct_cnts = [], []
        for p in props:
            cnt_mask = (lbl == p.label).astype(np.uint8)
            cnts, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                c[:] *= upscale_factor
                if p.eccentricity >= ecc_thresh_elongated:
                    elong_cnts.append(c)
                else:
                    punct_cnts.append(c)

        overlay = base_up.copy()
        cv2.drawContours(overlay, elong_cnts, -1, font_color_E, 2, cv2.LINE_AA)
        cv2.drawContours(overlay, punct_cnts, -1, font_color_P, 2, cv2.LINE_AA)

        for p in props:
            y, x = p.centroid
            x, y = int(x * upscale_factor), int(y * upscale_factor)
            morph = "E" if p.eccentricity >= ecc_thresh_elongated else "P"
            col = font_color_E if morph == "E" else font_color_P
            draw_text_with_outline(overlay, morph, (x, y),
                                   font_scale, col, font_thickness, outline_color)

        video_frames.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Write labeled video
    video_path = Path(out_dir) / "Morphology_Labeled.mp4"
    h, w = video_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fr in video_frames:
        writer.write(fr)
    writer.release()

    # Save labeled frame 0 still
    out_path = Path(out_dir) / "Morphology_Frame0_Labeled.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(video_frames[0], cv2.COLOR_RGB2BGR))
    print(f"Morphology visualization saved → {out_path}")
