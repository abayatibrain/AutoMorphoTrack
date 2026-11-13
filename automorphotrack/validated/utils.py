"""Optional image utilities; video playback never supplies acquisition timing."""

from pathlib import Path
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_high_dpi(fig, path, dpi=300):
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def upscale_frame(image, scale=2):
    import cv2
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    h, w = image.shape[:2]
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_CUBIC)


def write_video(frames, path, fps=5):
    import cv2
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video requires frames and positive playback fps")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise OSError("Video encoder could not open output")
    try:
        for frame in frames:
            if frame.shape != (h, w, 3):
                raise ValueError("All video frames must be equally sized RGB arrays")
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
