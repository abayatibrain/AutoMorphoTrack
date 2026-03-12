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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from automorphotrack.utils import ensure_dir, save_high_dpi
import warnings

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
                         gt_mask_func=None, out_dir="Validation_Outputs"):
    """
    Run segmentation at different parameter values and analyze sensitivity.

    Parameters
    ----------
    tif_path : str
        Path to input TIF file.
    param_name : str
        Name of the parameter to vary (e.g., 'threshold', 'sigma').
    param_range : list or np.ndarray
        Range of parameter values to test.
    channel : int
        Channel index (0 for mito, 1 for lyso).
    metric : str, default 'dice'
        Validation metric to report ('dice', 'iou', 'precision', 'recall', 'f1').
    gt_mask_func : callable, optional
        Function that returns ground truth mask for a given tif_path and param value.
        If None, a placeholder function is used.
    out_dir : str, default "Validation_Outputs"
        Output directory for sensitivity analysis results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [param_name, metric] showing sensitivity results.
    """
    ensure_dir(out_dir)

    if gt_mask_func is None:
        # Placeholder: assumes ground truth mask exists as a reference
        def gt_mask_func(path, param):
            return np.ones((512, 512), dtype=bool)

    results = []
    for param_val in param_range:
        # Placeholder: users should implement actual segmentation logic
        # For now, we simulate results
        try:
            pred_mask = np.random.rand(512, 512) > (0.5 + param_val * 0.01)
            gt_mask = gt_mask_func(tif_path, param_val)
            metrics = validate_segmentation(pred_mask, gt_mask)
            results.append({
                param_name: param_val,
                metric: metrics[metric]
            })
        except Exception as e:
            warnings.warn(f"Error processing parameter {param_name}={param_val}: {e}")
            continue

    df_results = pd.DataFrame(results)
    csv_path = Path(out_dir) / f"Sensitivity_{param_name}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Saved sensitivity analysis → {csv_path}")

    return df_results


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
