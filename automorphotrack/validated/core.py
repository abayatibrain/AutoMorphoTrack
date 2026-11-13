"""Deterministic segmentation, measurements, linking, and colocalization."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import opening, white_tophat, disk
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, find_contours


def segment_frame(image, config):
    """Otsu threshold on native intensity, then one documented cleanup sequence."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("Segmentation requires a finite nonnegative 2D image")
    if config.method == "external":
        raise ValueError("External segmentation requires supplied labeled masks")
    if config.background_radius_px:
        image = white_tophat(image, footprint=disk(config.background_radius_px))
    if np.ptp(image) == 0:
        return np.zeros(image.shape, dtype=bool), float(image.flat[0])
    if config.method == "otsu":
        threshold = float(threshold_otsu(image)) * config.threshold_factor
        mask = image > threshold
    else:
        local_threshold = threshold_local(image, block_size=config.local_block_size_px,
                                          method="gaussian", offset=config.local_offset)
        local_threshold *= config.threshold_factor
        mask = image > local_threshold
        threshold = float(np.median(local_threshold))
    if config.opening_radius_px:
        mask = opening(mask, footprint=disk(config.opening_radius_px), mode="ignore")
    if config.clear_border:
        mask = clear_border(mask)
    if config.min_size_px:
        labels = label(mask, connectivity=2)
        sizes = np.bincount(labels.ravel())
        mask = (labels > 0) & (sizes[labels] >= config.min_size_px)
    return mask.astype(bool), threshold


def contour_geometry(mask):
    """Use the same interpolated contours for polygon area and perimeter."""
    contours = find_contours(np.pad(mask.astype(float), 1), 0.5, fully_connected="high")
    signed_area, perimeter = 0.0, 0.0
    for contour in contours:
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        y, x = contour.T
        signed_area += 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        perimeter += np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()
    area = abs(signed_area)
    circularity = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else np.nan
    return float(area), float(perimeter), float(circularity)


SHAPE_COLUMNS = ["Frame", "Label", "X", "Y", "Area_px2", "Area_um2", "Contour_Area_px2",
                 "Contour_Perimeter_px", "Circularity", "Eccentricity", "Solidity", "Aspect_Ratio",
                 "Orientation_rad", "Morphology"]


def measure_shapes(masks, config, labeled=False):
    records, centroids = [], []
    for frame, mask in enumerate(masks):
        frame_centroids = []
        frame_labels = mask.astype(np.int64) if labeled else label(mask, connectivity=2)
        for region in regionprops(frame_labels):
            y, x = region.centroid
            frame_centroids.append((y, x))
            contour_area, perimeter, circularity = contour_geometry(region.image)
            area_um2 = region.area * config.pixel_size_um ** 2 if config.pixel_size_um else np.nan
            elongated_area = config.min_elongated_area_um2 is None or area_um2 >= config.min_elongated_area_um2
            punctate_area = config.min_punctate_area_um2 is None or area_um2 >= config.min_punctate_area_um2
            morphology = "Intermediate"
            if region.eccentricity >= config.elongated_eccentricity and elongated_area:
                morphology = "Elongated"
            elif region.eccentricity <= config.punctate_eccentricity and punctate_area:
                morphology = "Punctate"
            records.append([frame, region.label, x, y, region.area, area_um2, contour_area, perimeter,
                            circularity, region.eccentricity, region.solidity,
                            region.axis_major_length / region.axis_minor_length if region.axis_minor_length else np.nan,
                            region.orientation, morphology])
        centroids.append(np.asarray(frame_centroids, dtype=float).reshape(-1, 2))
    return pd.DataFrame(records, columns=SHAPE_COLUMNS), centroids


def gated_assignment(previous, current, max_distance):
    """Maximize the number of gated one-to-one links, then minimize distance."""
    if not len(previous) or not len(current):
        return {}, np.zeros(len(previous), int), np.zeros(len(current), int)
    distances = cdist(previous, current)
    allowed = distances <= max_distance
    n_previous, n_current = distances.shape
    unmatched = (n_previous + n_current + 1) * (max_distance + 1)
    costs = np.full((n_previous + n_current, n_current + n_previous), 4 * unmatched)
    costs[:n_previous, :n_current] = np.where(allowed, distances, 4 * unmatched)
    costs[:n_previous, n_current:] = unmatched
    costs[n_previous:, :n_current] = unmatched
    costs[n_previous:, n_current:] = 0
    rows, cols = linear_sum_assignment(costs)
    links = {int(i): int(j) for i, j in zip(rows, cols)
             if i < n_previous and j < n_current and allowed[i, j]}
    return links, allowed.sum(axis=1), allowed.sum(axis=0)


TRACK_COLUMNS = ["Frame", "Track_ID", "Label", "X", "Y", "Link_Status", "Link_Distance_px",
                 "Previous_Candidate_Count", "Current_Candidate_Count"]


def link_centroids(centroids, max_distance=15, labels_per_frame=None):
    """Adjacent-frame links only. Births get new IDs; gaps and fission are unresolved."""
    rows, previous_ids, next_id = [], {}, 0
    for frame, current in enumerate(centroids):
        current = np.asarray(current, dtype=float).reshape(-1, 2)
        previous = np.asarray(centroids[frame - 1], dtype=float).reshape(-1, 2) if frame else np.empty((0, 2))
        links, previous_counts, current_counts = gated_assignment(previous, current, max_distance)
        reverse = {j: i for i, j in links.items()}
        current_ids = {}
        for j, (y, x) in enumerate(current):
            if j in reverse:
                i = reverse[j]
                current_ids[j] = previous_ids[i]
                status = "ambiguous" if previous_counts[i] > 1 or current_counts[j] > 1 else "matched"
                distance = np.linalg.norm(previous[i] - current[j])
                previous_count = int(previous_counts[i])
            else:
                current_ids[j] = next_id
                next_id += 1
                status = "start" if frame == 0 else "birth_or_unmatched"
                distance, previous_count = np.nan, 0
            object_label = labels_per_frame[frame][j] if labels_per_frame is not None else j + 1
            rows.append([frame, current_ids[j], object_label, x, y, status, distance,
                         previous_count, int(current_counts[j])])
        previous_ids = current_ids
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


def motility_from_tracks(tracks, config):
    """Path length and time-weighted mean speed use only accepted observed links."""
    if tracks.duplicated(["Track_ID", "Frame"]).any():
        raise ValueError("A track cannot have multiple observations in one frame")
    full = tracks.sort_values(["Track_ID", "Frame"]).copy()
    grouped = full.groupby("Track_ID", sort=True)
    full["Delta_Frame"] = grouped["Frame"].diff()
    full["DX_px"] = grouped["X"].diff()
    full["DY_px"] = grouped["Y"].diff()
    scale, interval = config.pixel_size_um or 1.0, config.frame_interval_s or 1.0
    full["Step_Distance"] = np.hypot(full["DX_px"], full["DY_px"]) * scale
    full["Step_Duration"] = full["Delta_Frame"] * interval
    accepted = full["Link_Status"].eq("matched")
    if not config.exclude_ambiguous_links:
        accepted |= full["Link_Status"].eq("ambiguous")
    accepted &= full["Delta_Frame"].eq(1)
    full["Accepted_Step"] = accepted
    full["Speed"] = (full["Step_Distance"] / full["Step_Duration"]).where(accepted)
    length_unit = "um" if config.pixel_size_um else "px"
    time_unit = "s" if config.frame_interval_s else "frame"
    summaries = []
    for track_id, sub in full.groupby("Track_ID", sort=True):
        valid = sub[sub["Accepted_Step"]]
        path = valid["Step_Distance"].sum() if len(valid) else np.nan
        duration = valid["Step_Duration"].sum() if len(valid) else np.nan
        all_links = len(valid) == len(sub) - 1 and len(sub) > 1
        net = np.hypot(sub["X"].iloc[-1] - sub["X"].iloc[0],
                       sub["Y"].iloc[-1] - sub["Y"].iloc[0]) * scale if all_links else np.nan
        summaries.append(dict(Track_ID=track_id, Observations=len(sub), Accepted_Steps=len(valid),
                              Ambiguous_Steps=int(sub["Link_Status"].eq("ambiguous").sum()),
                              Path_Length=path, Net_Displacement=net, Accepted_Duration=duration,
                              Mean_Speed=path / duration if duration > 0 else np.nan,
                              Complete_Accepted_Trajectory=all_links,
                              Length_Unit=length_unit, Speed_Unit=f"{length_unit}/{time_unit}"))
    columns = ["Track_ID", "Observations", "Accepted_Steps", "Ambiguous_Steps", "Path_Length",
               "Net_Displacement", "Accepted_Duration", "Mean_Speed", "Complete_Accepted_Trajectory",
               "Length_Unit", "Speed_Unit"]
    summary = pd.DataFrame(summaries, columns=columns)
    frame_summary = full.groupby("Frame").agg(Mean_Speed=("Speed", "mean"),
                                              Accepted_Steps=("Accepted_Step", "sum")).reset_index()
    return full, summary, frame_summary


def colocalization_metrics(mito, lyso, mito_mask, lyso_mask, roi=None, background_percentile=None):
    """Intensity Pearson and thresholded Manders within a shared explicit ROI."""
    mito, lyso = np.asarray(mito, float), np.asarray(lyso, float)
    if mito.shape != lyso.shape or mito.shape != mito_mask.shape or mito.shape != lyso_mask.shape:
        raise ValueError("Images and masks must have the same shape")
    if roi is None:
        roi = np.ones(mito.shape, dtype=bool)
    roi = np.asarray(roi, dtype=bool)
    if roi.shape != mito.shape or not roi.any():
        raise ValueError("ROI must have matching shape and at least one pixel")
    if not np.isfinite(mito).all() or not np.isfinite(lyso).all() or np.any(mito < 0) or np.any(lyso < 0):
        raise ValueError("Colocalization requires finite nonnegative intensities")
    if background_percentile is not None:
        mito = np.maximum(mito - np.percentile(mito[roi], background_percentile), 0)
        lyso = np.maximum(lyso - np.percentile(lyso[roi], background_percentile), 0)
    m, l = mito[roi], lyso[roi]
    mm, lm = mito_mask[roi].astype(bool), lyso_mask[roi].astype(bool)
    intersection, union = mm & lm, mm | lm
    mden, lden = m[mm].sum(), l[lm].sum()
    pearson = float(np.corrcoef(m, l)[0, 1]) if np.ptp(m) > 0 and np.ptp(l) > 0 and len(m) > 1 else np.nan
    cosine_den = np.sqrt(np.dot(m, m) * np.dot(l, l))
    return dict(Jaccard_Overlap=intersection.sum() / union.sum() if union.any() else np.nan,
                Thresholded_Manders_M1=m[intersection].sum() / mden if mden > 0 else np.nan,
                Thresholded_Manders_M2=l[intersection].sum() / lden if lden > 0 else np.nan,
                Intensity_Pearson_r=pearson,
                Intensity_Cosine_Overlap=np.dot(m, l) / cosine_den if cosine_den > 0 else np.nan,
                ROI_Pixels=int(roi.sum()), Overlap_Pixels=int(intersection.sum()),
                Mito_Positive_Pixels=int(mm.sum()), Lyso_Positive_Pixels=int(lm.sum()))


def audit_drift(stack, channels):
    """Estimate frame translations by phase correlation, without modifying data."""
    from skimage.registration import phase_cross_correlation
    images = np.asarray(stack[..., list(channels)], dtype=float).sum(axis=-1)
    rows = [dict(Frame=0, Align_Shift_Y_px=0.0, Align_Shift_X_px=0.0, Registration_Error=np.nan)]
    for frame in range(1, len(images)):
        if np.ptp(images[frame - 1]) == 0 or np.ptp(images[frame]) == 0:
            shift, error = (np.nan, np.nan), np.nan
        else:
            shift, error, _ = phase_cross_correlation(images[frame - 1], images[frame], upsample_factor=10)
        rows.append(dict(Frame=frame, Align_Shift_Y_px=float(shift[0]), Align_Shift_X_px=float(shift[1]),
                         Registration_Error=float(error)))
    return pd.DataFrame(rows)
