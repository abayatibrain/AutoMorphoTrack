"""Noise-scaled spot detection with explicit, bounded instance masks."""

from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops


@dataclass(frozen=True)
class SpotConfig:
    sigma_px: float = 1.5
    background_sigma_px: float = 6.0
    threshold_snr: float = 3.0
    min_distance_px: int = 3
    support_fraction: float = 0.2
    max_radius_px: float = 4.5
    min_area_px: int = 3

    def __post_init__(self):
        for name in ("sigma_px", "background_sigma_px", "threshold_snr", "max_radius_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.background_sigma_px <= self.sigma_px:
            raise ValueError("background_sigma_px must exceed sigma_px")
        if not 0 < self.support_fraction < 1:
            raise ValueError("support_fraction must lie between zero and one")
        for name in ("min_distance_px", "min_area_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def segment_spots(image, config=None):
    """Return instance labels, intensity-weighted positions, and detector metadata.

    Positions use positive background-subtracted intensity inside each mask.
    These intensity support regions are not validated physical organelle borders.
    No temporal information or reference annotations enter segmentation.
    """
    config = config or SpotConfig()
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("Expected a finite, nonnegative, nonempty YX image")
    empty = np.zeros(image.shape, np.uint32)
    if np.ptp(image) == 0:
        return empty, np.empty((0, 2)), {"noise_response": 0.0, "threshold": 0.0, "detections": 0}
    background = gaussian_filter(image, config.background_sigma_px)
    foreground = image - background
    response = -gaussian_laplace(foreground, config.sigma_px) * config.sigma_px ** 2
    # A robust response-scale estimate includes correlated acquisition noise.
    median = float(np.median(response))
    noise = float(np.median(np.abs(response - median)) / 0.6744897501960817)
    noise = max(noise, np.finfo(float).eps * max(float(image.max()), 1.0))
    threshold = config.threshold_snr * noise
    peaks = peak_local_max(response, min_distance=config.min_distance_px,
                           threshold_abs=threshold, exclude_border=False)
    if not len(peaks):
        return empty, np.empty((0, 2)), {"noise_response": noise, "threshold": threshold, "detections": 0}
    markers = np.zeros(image.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    smooth = gaussian_filter(foreground, max(config.sigma_px / 2, 0.5))
    labels = watershed(-smooth, markers, mask=smooth > 0)
    coords = []
    result = empty.copy()
    for region in regionprops(labels):
        index = region.label - 1
        y, x = peaks[index]
        pixels = region.coords
        py, px = pixels.T
        keep = ((py - y) ** 2 + (px - x) ** 2 <= config.max_radius_px ** 2)
        keep &= smooth[py, px] >= max(smooth[y, x] * config.support_fraction, 0)
        py, px = py[keep], px[keep]
        if len(py) < config.min_area_px:
            continue
        weights = np.maximum(smooth[py, px], 0)
        if weights.sum() <= 0:
            continue
        coords.append([float(np.average(py, weights=weights)), float(np.average(px, weights=weights))])
        result[py, px] = len(coords)
    return result, np.asarray(coords).reshape(-1, 2), {
        "noise_response": noise, "threshold": threshold, "detections": len(coords)}
