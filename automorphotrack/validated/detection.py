"""Compatibility entry point using the shared revision pipeline.

The revision API accepts PipelineConfig rather than unrecorded stage-specific
parameters. Each entry point runs the same analysis and returns all outputs.
"""

from .pipeline import run_pipeline


def detect_organelles(tif_path="Composite.tif", out_dir="Detection_Outputs", config=None):
    return run_pipeline(tif_path, out_dir, config)


def detect_mask(frame, ch, min_size=3, thr_factor=0.8):
    """Compatibility mask helper; main analyses use PipelineConfig."""
    import numpy as np
    from .config import SegmentationConfig
    from .core import segment_frame
    image = np.asarray(frame)[..., ch].astype(float)
    mask, _ = segment_frame(image, SegmentationConfig(thr_factor, min_size))
    normalized = (image - image.min()) / np.ptp(image) if np.ptp(image) else np.zeros_like(image)
    return mask.astype(np.uint8), normalized
