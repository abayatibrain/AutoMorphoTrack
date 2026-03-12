# ============================================================
# AutoMorphoTrack Utilities
# ============================================================

import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path):
    """Create directory if missing."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_high_dpi(fig, path, dpi=600):
    """Save Matplotlib figure with high resolution."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")

def upscale_frame(img, scale=4):
    """Upscale an image for high-quality visualization."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def write_video(frames, path, fps=5):
    """Write RGB frames into an MP4 video."""
    if not frames:
        print("No frames provided for video.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for fr in frames:
        out.write(cv2.cvtColor(fr.astype(np.uint8), cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Saved video: {path}")
