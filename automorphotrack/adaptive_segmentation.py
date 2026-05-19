# ============================================================
# AutoMorphoTrack – Adaptive Segmentation Backends
# ============================================================
# Author: Armin Bayati, Ph.D.
#
# Reviewer #2 (R2.4) noted that pure Otsu thresholding plus morphological
# opening is sensitive to SNR, background heterogeneity, bleaching, and
# organelle density. This module provides drop-in alternative backends so
# users can swap the segmentation step without modifying the rest of the
# pipeline:
#
#     - segment_otsu          : the default global-threshold path
#     - segment_sauvola       : Sauvola adaptive thresholding for uneven background
#     - segment_niblack       : Niblack adaptive thresholding
#     - segment_local_otsu    : Otsu inside a sliding window for very large fields
#     - segment_subtracted    : rolling-ball / top-hat background subtraction
#                               followed by Otsu (handles bleaching/heterogeneity)
#
# All backends expose the same signature::
#
#     mask = segment_<name>(image, **kwargs)
#
# where ``image`` is a 2D float array in [0, 1] and ``mask`` is a 2D bool
# array. This lets the validation.sensitivity_analysis sweep run against any
# of them.
# ============================================================

from __future__ import annotations

import numpy as np
from skimage.filters import (
    threshold_otsu, threshold_sauvola, threshold_niblack, threshold_local
)
from skimage.morphology import (
    remove_small_objects, binary_opening, disk, white_tophat
)
from skimage.segmentation import clear_border


def _clean(mask: np.ndarray, min_size: int = 3,
           opening_radius: int = 1,
           clear_borders: bool = True) -> np.ndarray:
    if opening_radius > 0:
        mask = binary_opening(mask, footprint=disk(opening_radius))
    if clear_borders:
        mask = clear_border(mask)
    if min_size > 0:
        mask = remove_small_objects(mask, max(1, int(min_size)))
    return mask.astype(bool)


def segment_otsu(image: np.ndarray, thr_factor: float = 0.8,
                 min_size: int = 3, **clean_kwargs) -> np.ndarray:
    """Global Otsu (the default path), exposed for completeness."""
    thr = threshold_otsu(image) * thr_factor
    return _clean(image > thr, min_size=min_size, **clean_kwargs)


def segment_sauvola(image: np.ndarray, window_size: int = 25,
                    k: float = 0.2, min_size: int = 3,
                    **clean_kwargs) -> np.ndarray:
    """Sauvola adaptive thresholding (handles uneven illumination)."""
    if window_size % 2 == 0:
        window_size += 1
    thr_map = threshold_sauvola(image, window_size=window_size, k=k)
    return _clean(image > thr_map, min_size=min_size, **clean_kwargs)


def segment_niblack(image: np.ndarray, window_size: int = 25,
                    k: float = 0.2, min_size: int = 3,
                    **clean_kwargs) -> np.ndarray:
    """Niblack adaptive thresholding."""
    if window_size % 2 == 0:
        window_size += 1
    thr_map = threshold_niblack(image, window_size=window_size, k=k)
    return _clean(image > thr_map, min_size=min_size, **clean_kwargs)


def segment_local_otsu(image: np.ndarray, block_size: int = 51,
                       offset: float = 0.0, min_size: int = 3,
                       **clean_kwargs) -> np.ndarray:
    """Tile-wise local Otsu via skimage.filters.threshold_local."""
    if block_size % 2 == 0:
        block_size += 1
    thr_map = threshold_local(image, block_size=block_size, offset=offset)
    return _clean(image > thr_map, min_size=min_size, **clean_kwargs)


def segment_subtracted(image: np.ndarray, background_radius: int = 25,
                       thr_factor: float = 0.8, min_size: int = 3,
                       **clean_kwargs) -> np.ndarray:
    """Top-hat background subtraction + Otsu.

    Equivalent to ImageJ/FIJI's rolling-ball background subtraction step. Use
    when the field of view has slow intensity gradients from out-of-focus
    haze, photobleaching, or uneven illumination.
    """
    bg_subtracted = white_tophat(image, footprint=disk(background_radius))
    if bg_subtracted.max() <= 0:
        return np.zeros_like(image, dtype=bool)
    bg_subtracted = bg_subtracted / bg_subtracted.max()
    thr = threshold_otsu(bg_subtracted) * thr_factor
    return _clean(bg_subtracted > thr, min_size=min_size, **clean_kwargs)


BACKENDS = {
    "otsu": segment_otsu,
    "sauvola": segment_sauvola,
    "niblack": segment_niblack,
    "local_otsu": segment_local_otsu,
    "subtracted": segment_subtracted,
}


def segment(image: np.ndarray, backend: str = "otsu", **kwargs) -> np.ndarray:
    """Dispatch to a named backend. ``backend`` is one of ``BACKENDS``."""
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. Choices: {sorted(BACKENDS)}"
        )
    return BACKENDS[backend](image, **kwargs)
