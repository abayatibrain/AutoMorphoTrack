"""Software regression fixtures are synthetic, not biological ground truth."""

from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tifffile
from automorphotrack.validated import PipelineConfig, SegmentationConfig, run_pipeline
from automorphotrack.validated.io import normalize_axes, load_stack
from automorphotrack.validated.core import (
    segment_frame, contour_geometry, link_centroids, motility_from_tracks,
    colocalization_metrics, audit_drift,
)
from automorphotrack.validated.pipeline import frame_summaries
from automorphotrack.validated.summary import aggregate_biological_replicates
from automorphotrack.validated.motility import analyze_motility


def write_sample(path, blank=False):
    image = np.zeros((4, 40, 40, 2), np.uint16)
    if not blank:
        for frame in range(4):
            image[frame, 8:15, 8 + frame:15 + frame, 0] = 1000
            image[frame, 23:30, 23 - frame:30 - frame, 1] = 800
    tifffile.imwrite(path, image, metadata={"axes": "TYXC"}, photometric="minisblack")
    return image


@pytest.mark.parametrize("channels", [2, 3])
def test_axis_permutation_preserves_every_value(channels):
    original = np.arange(4 * 7 * 8 * channels).reshape(4, 7, 8, channels)
    result = normalize_axes(original.transpose(0, 3, 1, 2), "TCYX")
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(normalize_axes(original[0], "YXC"), original[:1])
    np.testing.assert_array_equal(normalize_axes(original[:, None], "TZYXC"), original)


@pytest.mark.parametrize("array,axes", [(np.zeros((4, 2, 8, 8)), "ZCYX"),
                                        (np.zeros((4, 8, 8)), "QYX"),
                                        (np.zeros((4, 2, 8, 8)), "TCYY")])
def test_ambiguous_or_spatial_axes_rejected(array, axes):
    with pytest.raises(ValueError):
        normalize_axes(array, axes)


def test_axes_override_requires_reason(tmp_path):
    path = tmp_path / "native.tif"
    tifffile.imwrite(path, np.zeros((4, 2, 8, 8), np.uint8), metadata={"axes": "ZCYX"})
    with pytest.raises(ValueError, match="override"):
        load_stack(path, PipelineConfig(axes="TCYX"))
    stack, meta = load_stack(path, PipelineConfig(axes="TCYX", axes_override_reason="Synthetic mislabeled-axis fixture"))
    assert stack.shape == (4, 8, 8, 2) and meta["native_axes"] == "ZCYX"


@pytest.mark.parametrize("kwargs", [{"frame_interval_s": 0}, {"pixel_size_um": -1},
                                   {"mito_channel": 1}, {"min_elongated_area_um2": 0.12},
                                   {"elongated_eccentricity": 0.5}])
def test_invalid_calibration_or_classifier_rejected(kwargs):
    with pytest.raises(ValueError):
        PipelineConfig(**kwargs)


def test_blank_frame_and_integer_size_filter():
    cfg = SegmentationConfig(min_size_px=5, opening_radius_px=0, clear_border=False)
    blank, _ = segment_frame(np.full((20, 20), 5), cfg)
    assert not blank.any()
    image = np.zeros((20, 20)); image[2:4, 2:4] = 30; image[10:13, 10:13] = 30
    mask, _ = segment_frame(image, cfg)
    assert mask.sum() == 9


def test_local_and_background_options_have_real_effect():
    yy, xx = np.indices((64, 64))
    foreground = (yy - 32) ** 2 + (xx - 20) ** 2 <= 16
    image = xx * 2.0 + foreground * 40
    base = SegmentationConfig(threshold_factor=1, min_size_px=0, opening_radius_px=0, clear_border=False)
    global_mask, _ = segment_frame(image, base)
    local_mask, _ = segment_frame(image, replace(base, method="local", local_block_size_px=21, local_offset=-8))
    top_mask, _ = segment_frame(image, replace(base, background_radius_px=8))
    assert not np.array_equal(global_mask, local_mask)
    assert not np.array_equal(global_mask, top_mask)
    assert np.logical_and(local_mask, foreground).sum() / foreground.sum() > 0.8


@pytest.mark.parametrize("mask", [np.ones((2, 2), bool), np.ones((7, 9), bool),
                                  np.eye(6, dtype=bool), np.pad(np.ones((4, 4), bool), 2)])
def test_polygon_circularity_uses_consistent_geometry_without_clipping(mask):
    area, perimeter, circularity = contour_geometry(mask)
    assert area > 0 and perimeter > 0
    assert circularity == pytest.approx(4 * np.pi * area / perimeter ** 2)
    assert 0 < circularity <= 1 + 1e-12


def test_hole_contributes_to_area_and_perimeter():
    full = np.ones((15, 15), bool); hole = full.copy(); hole[5:10, 5:10] = False
    area_full, perimeter_full, c_full = contour_geometry(full)
    area_hole, perimeter_hole, c_hole = contour_geometry(hole)
    assert area_hole < area_full and perimeter_hole > perimeter_full and c_hole < c_full


def test_order_reversal_tracks_identity_instead_of_label_number():
    points = [np.array([[5, 5], [30, 30]]), np.array([[30, 31], [5, 6]])]
    tracks = link_centroids(points, max_distance=3)
    first = tracks[tracks.Frame.eq(0)].set_index("Track_ID")
    last = tracks[tracks.Frame.eq(1)].set_index("Track_ID")
    assert last.loc[0, "X"] == 6 and last.loc[1, "X"] == 31
    assert last.loc[0, "Label"] != first.loc[0, "Label"]


def test_one_to_one_merge_ambiguity_and_gate():
    tracks = link_centroids([np.array([[10, 10], [10, 12]]), np.array([[10, 11]])], 3)
    second = tracks[tracks.Frame.eq(1)]
    assert len(second) == 1 and second.Link_Status.item() == "ambiguous"
    assert not tracks.duplicated(["Track_ID", "Frame"]).any()
    steps, summary, _ = motility_from_tracks(tracks, PipelineConfig())
    assert summary.Accepted_Steps.sum() == 0
    tracks_far = link_centroids([np.array([[0, 0]]), np.array([[0, 20]])], 3)
    assert tracks_far.Track_ID.nunique() == 2


def test_disappearance_then_return_is_new_track():
    tracks = link_centroids([np.array([[2, 2]]), np.empty((0, 2)), np.array([[2, 2]])])
    assert tracks.Track_ID.tolist() == [0, 1]


def test_calibrated_speed_and_net_displacement_are_distinct():
    tracks = link_centroids([np.array([[0, 0]]), np.array([[0, 3]]), np.array([[4, 3]])], 5)
    _, summary, _ = motility_from_tracks(tracks, PipelineConfig(pixel_size_um=0.5, frame_interval_s=2))
    assert summary.Path_Length.item() == pytest.approx(3.5)
    assert summary.Net_Displacement.item() == pytest.approx(2.5)
    assert summary.Mean_Speed.item() == pytest.approx(0.875)
    assert summary.Speed_Unit.item() == "um/s"
    _, uncal, _ = motility_from_tracks(tracks, PipelineConfig())
    assert uncal.Mean_Speed.item() == pytest.approx(3.5) and uncal.Speed_Unit.item() == "px/frame"


def test_duplicate_track_frame_rejected():
    tracks = link_centroids([np.array([[1, 1]])])
    with pytest.raises(ValueError):
        motility_from_tracks(pd.concat([tracks, tracks]), PipelineConfig())


def test_colocalization_known_manders_and_jaccard():
    m = np.array([[2, 4], [1, 0.]])
    l = np.array([[1, 3], [5, 0.]])
    mm = np.array([[True, True], [False, False]])
    lm = np.array([[False, True], [True, False]])
    values = colocalization_metrics(m, l, mm, lm)
    assert values["Jaccard_Overlap"] == pytest.approx(1 / 3)
    assert values["Thresholded_Manders_M1"] == pytest.approx(2 / 3)
    assert values["Thresholded_Manders_M2"] == pytest.approx(3 / 8)
    anti = colocalization_metrics(np.array([[0, 1, 2]]), np.array([[2, 1, 0]]), np.ones((1, 3),bool), np.ones((1,3),bool))
    assert anti["Intensity_Pearson_r"] == pytest.approx(-1)


def test_constant_or_empty_colocalization_is_undefined():
    zeros = np.zeros((3, 3)); masks = zeros.astype(bool)
    result = colocalization_metrics(zeros, zeros, masks, masks)
    assert np.isnan(result["Intensity_Pearson_r"])
    assert np.isnan(result["Thresholded_Manders_M1"])
    assert np.isnan(result["Jaccard_Overlap"])
    with pytest.raises(ValueError):
        colocalization_metrics(zeros, zeros, masks, masks, roi=masks)


def test_drift_audit_reports_inverse_alignment_shift():
    rng = np.random.default_rng(19)
    frame = rng.normal(20, 2, (32, 32))
    translated = np.roll(np.roll(frame, 2, axis=0), -3, axis=1)
    stack = np.stack([frame, translated])[..., None]
    audit = audit_drift(stack, (0,))
    assert audit.loc[1, "Align_Shift_Y_px"] == pytest.approx(-2, abs=0.1)
    assert audit.loc[1, "Align_Shift_X_px"] == pytest.approx(3, abs=0.1)


def test_pipeline_reproducibility_schema_and_no_pseudoreplication(tmp_path):
    path = tmp_path / "sample.tif"; write_sample(path)
    cfg = PipelineConfig(render=False, max_link_distance_px=3)
    first = run_pipeline(path, tmp_path / "first", cfg)
    second = run_pipeline(path, tmp_path / "second", cfg)
    assert first["manifest"]["run_id"] == second["manifest"]["run_id"]
    assert len(first["frame_summary"]) == 4
    assert first["frame_summary"].Frame.is_unique
    assert "Frame" not in first["correlation"] and "Label" not in first["correlation"]
    assert first["mito_shapes"].groupby("Frame").size().tolist() == [1] * 4
    assert first["mito_masks"].shape == (4, 40, 40)
    with pytest.raises(FileExistsError):
        run_pipeline(path, tmp_path / "first", cfg)
    other = run_pipeline(path, tmp_path / "other", replace(cfg, max_link_distance_px=4))
    assert first["manifest"]["run_id"] != other["manifest"]["run_id"]
    assert PipelineConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        aggregate_biological_replicates([first["sample_summary"]])


def test_pipeline_blank_data_has_readable_tables(tmp_path):
    path = tmp_path / "sample.tif"; write_sample(path, blank=True)
    result = run_pipeline(path, tmp_path / "out", PipelineConfig(render=False))
    assert len(result["mito_shapes"]) == 0
    assert len(pd.read_csv(tmp_path / "out/Mito_ShapeMetrics.csv")) == 0
    assert result["frame_summary"].Mito_Count.sum() == 0
    assert len(result["frame_summary"]) == 4


def test_external_instance_backend_preserves_touching_labels(tmp_path):
    path = tmp_path / "sample.tif"; image = write_sample(path)
    labels = np.zeros(image.shape[:3], np.uint16)
    labels[:, 8:15, 8:12] = 6
    labels[:, 8:15, 12:16] = 20
    cfg = PipelineConfig(render=False, mito_segmentation=SegmentationConfig(method="external"))
    result = run_pipeline(path, tmp_path / "external", cfg, external_masks={"mito": labels})
    assert result["mito_shapes"].groupby("Frame").size().tolist() == [2] * 4
    assert set(result["mito_tracks"].Label.unique()) == {6, 20}
    assert result["manifest"]["external_labels_sha256"]["mito"]


def test_legacy_frame_labels_refused_for_motility(tmp_path):
    path = tmp_path / "legacy.csv"
    pd.DataFrame({"Organelle": [0, 0], "Frame": [0, 1], "X": [2, 3], "Y": [2, 3]}).to_csv(path,index=False)
    with pytest.raises(ValueError, match="Legacy"):
        analyze_motility(path, path, tmp_path / "out")
