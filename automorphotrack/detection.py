# ============================================================
# AutoMorphoTrack – Organelle Detection
# ============================================================

import tifffile, numpy as np, cv2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.segmentation import clear_border
from pathlib import Path
from automorphotrack.utils import ensure_dir, write_video, upscale_frame

def detect_mask(frame, ch, thr_factor=0.8, min_size=3):
    """Threshold and clean a single channel."""
    img = frame[..., ch].astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)
    thr = threshold_otsu(img) * thr_factor
    mask = clear_border(binary_opening(img > thr, footprint=disk(1)))
    mask = remove_small_objects(mask, min_size)
    return (mask * 255).astype(np.uint8), img

def detect_organelles(
    tif_path="Composite.tif",
    out_dir="Detection_Outputs",
    mito_channel=0,
    lyso_channel=1,
    upscale_factor=4,
    fps=5):

    ensure_dir(out_dir)
    stack = tifffile.imread(str(tif_path))
    if stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    n_frames = stack.shape[0]
    print(f"Loaded {n_frames} frames for detection")

    mito_frames, lyso_frames = [], []

    for f in range(n_frames):
        # ----- Mitochondria (relaxed detection) -----
        mito_mask, mito_img = detect_mask(stack[f], mito_channel, thr_factor=0.6, min_size=0.5)
        mito_gray = (mito_img * 255).astype(np.uint8)
        mito_rgb = cv2.cvtColor(mito_gray, cv2.COLOR_GRAY2RGB)
        cnts, _ = cv2.findContours(mito_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mito_rgb, cnts, -1, (1, 115, 178), 1, cv2.LINE_AA)  # CB blue
        mito_frames.append(upscale_frame(mito_rgb, upscale_factor))

        # ----- Lysosomes (standard detection) -----
        lyso_mask, lyso_img = detect_mask(stack[f], lyso_channel, thr_factor=0.8, min_size=2)
        lyso_gray = (lyso_img * 255).astype(np.uint8)
        lyso_rgb = cv2.cvtColor(lyso_gray, cv2.COLOR_GRAY2RGB)
        cnts, _ = cv2.findContours(lyso_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(lyso_rgb, cnts, -1, (222, 143, 5), 1, cv2.LINE_AA)  # CB orange
        lyso_frames.append(upscale_frame(lyso_rgb, upscale_factor))

    # Save representative frames
    cv2.imwrite(str(Path(out_dir) / "Mito_Frame0.png"),
                cv2.cvtColor(mito_frames[0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(out_dir) / "Lyso_Frame0.png"),
                cv2.cvtColor(lyso_frames[0], cv2.COLOR_RGB2BGR))

    # Save videos
    write_video(mito_frames, Path(out_dir) / "Mitochondria_Detection.mp4", fps=fps)
    write_video(lyso_frames, Path(out_dir) / "Lysosomes_Detection.mp4", fps=fps)

    print(f"Detection complete — outputs saved in {Path(out_dir).resolve()}")
