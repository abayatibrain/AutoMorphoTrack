"""Unit tests for validation metrics and adaptive segmentation backends.

These tests give Reviewer #2 something concrete to inspect for R2.2 / R2.4:
they exercise the Dice/IoU implementation on hand-rolled cases with known
answers and confirm that every segmentation backend produces a sane binary
mask on a synthetic image.
"""

import numpy as np
import pytest

from automorphotrack.validation import (
    validate_segmentation,
    generate_synthetic_ground_truth,
)
from automorphotrack.adaptive_segmentation import segment, BACKENDS


def test_validate_perfect_overlap():
    m = np.zeros((100, 100), dtype=bool)
    m[20:40, 20:40] = True
    result = validate_segmentation(m, m)
    assert result["dice"] == pytest.approx(1.0)
    assert result["iou"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_validate_no_overlap():
    a = np.zeros((100, 100), dtype=bool); a[10:30, 10:30] = True
    b = np.zeros((100, 100), dtype=bool); b[60:80, 60:80] = True
    result = validate_segmentation(a, b)
    assert result["dice"] == 0.0
    assert result["iou"] == 0.0


def test_validate_partial_overlap_known_answer():
    # 100 px each, 50 px overlap
    a = np.zeros((10, 20), dtype=bool); a[:, :10] = True   # 100 px
    b = np.zeros((10, 20), dtype=bool); b[:, 5:15] = True  # 100 px
    # Intersection 10x5 = 50, union = 150
    result = validate_segmentation(a, b)
    assert result["iou"] == pytest.approx(50 / 150)
    assert result["dice"] == pytest.approx(2 * 50 / (100 + 100))


def test_synthetic_gt_is_binary():
    gt = generate_synthetic_ground_truth(seed=42)
    assert gt.dtype == bool
    assert gt.any()
    assert gt.shape == (512, 512)


def test_synthetic_gt_from_image():
    rng = np.random.default_rng(0)
    img = rng.random((64, 64))
    img[20:30, 20:30] = 0.95  # bright blob
    gt = generate_synthetic_ground_truth(img)
    assert gt.dtype == bool
    assert gt[20:30, 20:30].mean() > 0.5  # blob detected


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_each_backend_runs(backend):
    rng = np.random.default_rng(1)
    img = rng.random((64, 64)) * 0.3
    img[15:35, 15:35] += 0.6
    img = np.clip(img, 0, 1)
    mask = segment(img, backend=backend)
    assert mask.dtype == bool
    assert mask.shape == img.shape
    # The bright block should produce *some* foreground
    assert mask.any()
