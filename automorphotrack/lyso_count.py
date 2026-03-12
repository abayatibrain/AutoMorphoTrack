# ============================================================
# AutoMorphoTrack – Lysosomal Counting per Frame
# ============================================================

import tifffile, numpy as np, cv2, pandas as pd, matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import clear_border
from pathlib import Path
from automorphotrack.utils import ensure_dir, write_video, upscale_frame

def detect_mask(frame, ch, thr_factor=0.8, min_size=2):
    """Detect lysosomes in a single channel frame."""
    img = frame[..., ch].astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)
    thr = threshold_otsu(img) * thr_factor
    mask = clear_border(binary_opening(img > thr, footprint=disk(1)))
    mask = remove_small_objects(mask, min_size)
    return (mask * 255).astype(np.uint8), img

def count_lysosomes_per_frame(
    tif_path="Composite.tif",
    out_dir="Lyso_Count_Outputs",
    lyso_channel=1,
    upscale_factor=4,
    fps=5):

    ensure_dir(out_dir)
    stack = tifffile.imread(str(tif_path))
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for lysosomal counting")

    lyso_counts, lyso_frames = [], []

    for f in range(n_frames):
        lyso_mask, lyso_img = detect_mask(stack[f], lyso_channel, thr_factor=0.8, min_size=2)
        cnts, _ = cv2.findContours(lyso_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lyso_count = len(cnts)
        lyso_counts.append({"Frame": f, "Lysosome_Count": lyso_count})

        lyso_gray = (lyso_img * 255).astype(np.uint8)
        lyso_rgb = cv2.cvtColor(lyso_gray, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(lyso_rgb, cnts, -1, (222, 143, 5), 1, cv2.LINE_AA)  # CB orange

        # Add small orange numbers at each lysosome position
        for i, c in enumerate(cnts):
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.putText(lyso_rgb, str(i + 1), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (222, 143, 5), 1, cv2.LINE_AA)

        # Add total count label in white (small font)
        cv2.putText(lyso_rgb, f"Count: {lyso_count}", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        lyso_frames.append(upscale_frame(lyso_rgb, upscale_factor))

    # Save representative frame
    cv2.imwrite(str(Path(out_dir) / "Lyso_Frame0_Count.png"),
                cv2.cvtColor(lyso_frames[0], cv2.COLOR_RGB2BGR))

    # Save video
    write_video(lyso_frames, Path(out_dir) / "Lyso_Count_Video.mp4", fps=fps)

    # Save CSV
    df = pd.DataFrame(lyso_counts)
    csv_path = Path(out_dir) / "Lysosome_Counts.csv"
    df.to_csv(csv_path, index=False)

    # Plot counts over frames
    plt.figure(figsize=(6, 4))
    plt.plot(df["Frame"], df["Lysosome_Count"], "-o", color="#DE8F05")
    plt.xlabel("Frame")
    plt.ylabel("Lysosome Count")
    plt.title("Lysosome Count per Frame")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "Lyso_Count_Plot.png", dpi=200)
    plt.close()

    print(f"Lysosomal counting complete — outputs saved in {Path(out_dir).resolve()}")
