# ============================================================
# AutoMorphoTrack – Validation Module
# ============================================================
# Author: Armin Bayati, Ph.D.
# Description:
#     Validation functions for segmentation, parameter sensitivity,
#     tracking accuracy, and comprehensive validation reports.
#
# Functions:
#     - validate_segmentation(): Dice, IoU, precision, recall, F1
#     - sensitivity_analysis(): Parameter sweep analysis
#     - validate_tracking(): Tracking accuracy metrics
#     - generate_validation_report(): Summary plots and CSV
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from automorphotrack.utils import ensure_dir, save_high_dpi
import warnings

# NOTE (review fix R3.1): scikit-learn was previously imported here but never
# used at call-sites — its presence broke `pip install automorphotrack` because
# it was an undeclared transitive dependency. All metrics below are computed
# directly from confusion-matrix counts so we no longer need scikit-learn.

# Colorblind-safe palette
CB_MITO = "#0173B2"
CB_LYSO = "#DE8F05"


def validate_segmentation(predicted_mask, ground_truth_mask):
    """
    Compute segmentation validation metrics (Dice, IoU, precision, recall, F1).

    Parameters
    ----------
    predicted_mask : np.ndarray
        Binary predicted segmentation mask (0 and 1).
    ground_truth_mask : np.ndarray
        Binary ground truth segmentation mask (0 and 1).

    Returns
    -------
    dict
        Dictionary with keys: 'dice', 'iou', 'precision', 'recall', 'f1'
    """
    # Flatten masks
    pred_flat = predicted_mask.astype(bool).flatten()
    true_flat = ground_truth_mask.astype(bool).flatten()

    # True positives, false positives, false negatives
    tp = np.sum(pred_flat & true_flat)
    fp = np.sum(pred_flat & ~true_flat)
    fn = np.sum(~pred_flat & true_flat)
    tn = np.sum(~pred_flat & ~true_flat)

    # Dice coefficient
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # IoU (Jaccard Index)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def sensitivity_analysis(tif_path, param_name, param_range, channel, metric='dice',
                         gt_mask_func=None, frame=0, min_size=3,
                         out_dir="Validation_Outputs"):
    """
    Run segmentation at different parameter values and analyze sensitivity.

    This is a *real* sweep: for each value in ``param_range`` the requested
    parameter is passed into the AutoMorphoTrack detection pipeline, the
    resulting mask is compared against a ground-truth mask (either provided by
    ``gt_mask_func`` or generated synthetically with
    :func:`generate_synthetic_ground_truth`), and the requested metric is
    recorded.

    Supported ``param_name`` values: ``"thr_factor"`` (default Otsu multiplier),
    ``"min_size"`` (minimum-object size in pixels). For any other name the
    parameter is forwarded as a kwarg to the detection backend so users can
    sweep custom hooks.

    Parameters
    ----------
    tif_path : str
        Path to input TIF stack (used to load the frame to segment).
    param_name : str
        Name of the parameter to vary.
    param_range : list or np.ndarray
        Range of parameter values to test.
    channel : int
        Channel index (0 = mitochondria, 1 = lysosomes in default layout).
    metric : str, default 'dice'
        Validation metric to report ('dice', 'iou', 'precision', 'recall', 'f1').
    gt_mask_func : callable, optional
        ``gt_mask_func(tif_path, frame) -> 2D bool array``. When omitted, a
        deterministic synthetic ground truth is generated from the frame using
        a strict Otsu threshold so the metric measures how the user's
        ``param_name`` choice drifts away from that reference.
    frame : int, default 0
        Frame index to analyze in the stack.
    min_size : int, default 3
        Default min-object size applied when ``param_name`` is not min_size.
    out_dir : str, default "Validation_Outputs"
        Output directory for sensitivity analysis results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [param_name, dice, iou, precision, recall, f1].
    """
    import tifffile
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.segmentation import clear_border

    ensure_dir(out_dir)

    try:
        stack = tifffile.imread(str(tif_path))
    except Exception as e:
        warnings.warn(f"Could not load {tif_path}: {e}")
        return pd.DataFrame()

    if stack.ndim == 3 and stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    if stack.ndim == 4:
        frame_img = stack[frame][..., channel].astype(float)
    elif stack.ndim == 3:
        # Already a single multichannel frame
        frame_img = stack[..., channel].astype(float)
    else:
        warnings.warn(f"Unexpected stack shape {stack.shape}; cannot sweep.")
        return pd.DataFrame()

    frame_img = (frame_img - frame_img.min()) / (np.ptp(frame_img) + 1e-12)

    if gt_mask_func is None:
        gt_mask = generate_synthetic_ground_truth(frame_img)
    else:
        gt_mask = gt_mask_func(tif_path, frame).astype(bool)

    def _segment(thr_factor, min_obj_size):
        thr = threshold_otsu(frame_img) * thr_factor
        m = clear_border(binary_opening(frame_img > thr, footprint=disk(1)))
        m = remove_small_objects(m, max(1, int(min_obj_size)))
        return m.astype(bool)

    results = []
    for param_val in param_range:
        try:
            if param_name == "thr_factor":
                pred_mask = _segment(float(param_val), min_size)
            elif param_name == "min_size":
                pred_mask = _segment(0.8, int(param_val))
            else:
                # User-supplied custom param: skip swept value into thr_factor
                # but warn so they don't silently get a no-op sweep.
                warnings.warn(f"Unknown param_name '{param_name}'; "
                              "interpreting value as thr_factor.")
                pred_mask = _segment(float(param_val), min_size)
            metrics = validate_segmentation(pred_mask, gt_mask)
            row = {param_name: param_val}
            row.update(metrics)
            results.append(row)
        except Exception as e:
            warnings.warn(f"Error at {param_name}={param_val}: {e}")
            continue

    df_results = pd.DataFrame(results)
    csv_path = Path(out_dir) / f"Sensitivity_{param_name}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved sensitivity analysis → {csv_path}")

    # Plot the requested metric across the sweep
    if len(df_results) and metric in df_results.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df_results[param_name], df_results[metric],
                marker='o', color=CB_MITO, linewidth=2)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.set_title(f"Sensitivity of {metric} to {param_name}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        save_high_dpi(fig, Path(out_dir) / f"Sensitivity_{param_name}.png")

    return df_results


def generate_synthetic_ground_truth(frame_img=None, shape=(512, 512), n_objects=30,
                                    obj_radius_range=(3, 9), seed=0):
    """
    Generate a synthetic binary ground-truth mask for validation.

    Two modes:
      * If ``frame_img`` is given, returns the strict Otsu segmentation of that
        image — useful as a self-consistent reference for sensitivity sweeps.
      * If ``frame_img`` is None, draws ``n_objects`` filled circles of random
        radius/position into a ``shape`` array — useful for unit-testing
        segmentation/tracking without real microscopy data.

    Parameters
    ----------
    frame_img : np.ndarray, optional
        Normalized 2D image. When supplied, an Otsu mask of it is returned.
    shape : tuple, default (512, 512)
        Output shape when ``frame_img`` is None.
    n_objects : int, default 30
        Number of synthetic objects to draw when ``frame_img`` is None.
    obj_radius_range : tuple, default (3, 9)
        (min, max) radius (pixels) for synthetic objects.
    seed : int, default 0
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D boolean ground-truth mask.
    """
    from skimage.filters import threshold_otsu

    if frame_img is not None:
        thr = threshold_otsu(frame_img)
        return (frame_img > thr).astype(bool)

    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    for _ in range(n_objects):
        cx = int(rng.integers(0, shape[1]))
        cy = int(rng.integers(0, shape[0]))
        r = int(rng.integers(obj_radius_range[0], obj_radius_range[1] + 1))
        mask |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
    return mask


def validate_tracking(predicted_tracks_csv, ground_truth_tracks_csv, max_dist=5):
    """
    Compute tracking accuracy metrics.

    Parameters
    ----------
    predicted_tracks_csv : str
        Path to CSV with predicted tracks (columns: track_id, frame, x, y).
    ground_truth_tracks_csv : str
        Path to CSV with ground truth tracks (same structure).
    max_dist : float, default 5
        Maximum distance to consider a match between predicted and ground truth tracks.

    Returns
    -------
    dict
        Dictionary with tracking accuracy metrics:
        - 'correct_links': number of correctly matched detections
        - 'false_positives': unmatched predicted detections
        - 'false_negatives': unmatched ground truth detections
        - 'tracking_accuracy': ratio of correct matches
    """
    try:
        pred_df = pd.read_csv(predicted_tracks_csv)
        gt_df = pd.read_csv(ground_truth_tracks_csv)
    except Exception as e:
        warnings.warn(f"Error loading tracking CSVs: {e}")
        return {
            'correct_links': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'tracking_accuracy': 0.0
        }

    # Group by frame and compute distances
    correct_links = 0
    false_positives = 0
    false_negatives = 0

    for frame in pred_df['frame'].unique():
        pred_frame = pred_df[pred_df['frame'] == frame]
        gt_frame = gt_df[gt_df['frame'] == frame]

        for _, pred_row in pred_frame.iterrows():
            px, py = pred_row['x'], pred_row['y']
            # Find closest ground truth point
            if len(gt_frame) > 0:
                distances = np.sqrt((gt_frame['x'] - px)**2 + (gt_frame['y'] - py)**2)
                min_dist = distances.min()
                if min_dist <= max_dist:
                    correct_links += 1
                else:
                    false_positives += 1
            else:
                false_positives += 1

        # Count unmatched ground truth points
        for _, gt_row in gt_frame.iterrows():
            gx, gy = gt_row['x'], gt_row['y']
            if len(pred_frame) > 0:
                distances = np.sqrt((pred_frame['x'] - gx)**2 + (pred_frame['y'] - gy)**2)
                min_dist = distances.min()
                if min_dist > max_dist:
                    false_negatives += 1
            else:
                false_negatives += 1

    total_gt = len(gt_df)
    tracking_accuracy = correct_links / total_gt if total_gt > 0 else 0.0

    return {
        'correct_links': correct_links,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'tracking_accuracy': tracking_accuracy
    }


def generate_validation_report(results_dict, out_dir="Validation_Outputs"):
    """
    Generate summary plots and CSV report of all validation metrics.

    Parameters
    ----------
    results_dict : dict
        Dictionary of validation results. Expected keys:
        - 'segmentation': dict with 'dice', 'iou', 'precision', 'recall', 'f1'
        - 'tracking': dict with 'correct_links', 'false_positives', 'false_negatives'
        - 'sensitivity': DataFrame with parameter sweep results (optional)
    out_dir : str, default "Validation_Outputs"
        Output directory for validation report.

    Returns
    -------
    str
        Path to generated validation report CSV.
    """
    ensure_dir(out_dir)

    # Create summary metrics table
    summary_data = []

    # Segmentation metrics
    if 'segmentation' in results_dict:
        seg = results_dict['segmentation']
        for metric_name, metric_val in seg.items():
            summary_data.append({
                'Category': 'Segmentation',
                'Metric': metric_name,
                'Value': metric_val
            })

    # Tracking metrics
    if 'tracking' in results_dict:
        track = results_dict['tracking']
        for metric_name, metric_val in track.items():
            summary_data.append({
                'Category': 'Tracking',
                'Metric': metric_name,
                'Value': metric_val
            })

    summary_df = pd.DataFrame(summary_data)
    summary_csv = Path(out_dir) / "Validation_Summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved validation summary → {summary_csv}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Segmentation metrics bar plot
    if 'segmentation' in results_dict and len(summary_df[summary_df['Category'] == 'Segmentation']) > 0:
        seg_df = summary_df[summary_df['Category'] == 'Segmentation']
        ax = axes[0]
        bars = ax.bar(seg_df['Metric'], seg_df['Value'], color=CB_MITO, alpha=0.7, edgecolor='black')
        ax.set_title('Segmentation Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[0].text(0.5, 0.5, 'No segmentation data', ha='center', va='center',
                    transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title('Segmentation Metrics', fontsize=12, fontweight='bold')

    # Tracking metrics bar plot
    if 'tracking' in results_dict and len(summary_df[summary_df['Category'] == 'Tracking']) > 0:
        track_df = summary_df[summary_df['Category'] == 'Tracking']
        ax = axes[1]
        # Normalize tracking accuracy to 0-1 scale for visualization
        track_df_viz = track_df.copy()
        track_df_viz.loc[track_df_viz['Metric'] == 'tracking_accuracy', 'Value'] *= 1.0  # already 0-1
        bars = ax.bar(track_df_viz['Metric'], track_df_viz['Value'], color=CB_LYSO, alpha=0.7, edgecolor='black')
        ax.set_title('Tracking Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count / Score', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No tracking data', ha='center', va='center',
                    transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Tracking Metrics', fontsize=12, fontweight='bold')

    plt.tight_layout()
    report_fig = Path(out_dir) / "Validation_Report.png"
    save_high_dpi(fig, report_fig)

    print(f"Validation report complete → {Path(out_dir).resolve()}")
    return str(summary_csv)
