# ============================================================
# AutoMorphoTrack – Automated Organelle Detection and Tracking
# ============================================================
# Author: Armin Bayati, Ph.D.
# Description:
#     AutoMorphoTrack is a modular image-analysis package for
#     mitochondria and lysosome tracking, morphology, shape,
#     motility, and colocalization analysis in fluorescence microscopy.
#
# Core Pipeline:
#     1. Detection (detection.py)
#     2. Lysosome counting (lyso_count.py)
#     3. Morphology classification (morphology.py)
#     4. Shape feature extraction (shape_features.py)
#     5. Shape profiling (shape_profiling.py)
#     6. Organelle tracking (tracking.py / tracking_overlay.py)
#     7. Motility analysis (motility.py)
#     8. Colocalization analysis (colocalization.py)
#     9. Integrated summary (summary.py)
#
# Package structure follows:
#     from automorphotrack import (
#         detect_organelles,
#         count_lysosomes_per_frame,
#         classify_morphology,
#         analyze_shape_features,
#         profile_shape_data,
#         track_organelles,
#         track_overlay,
#         analyze_motility,
#         analyze_colocalization,
#         summarize_integrated_data
#     )
# ============================================================

__version__ = "2.2.0"
__author__ = "Armin Bayati"

# --- Utility Imports ---
from automorphotrack.utils import (
    ensure_dir,
    save_high_dpi,
    upscale_frame,
    write_video
)

# --- Core Functional Imports ---
from automorphotrack.detection import detect_organelles
from automorphotrack.lyso_count import count_lysosomes_per_frame
from automorphotrack.morphology import classify_morphology
from automorphotrack.shape_features import analyze_shape_features
from automorphotrack.shape_profiling import profile_shape_data
from automorphotrack.tracking import track_organelles
from automorphotrack.tracking_overlay import track_overlay
from automorphotrack.motility import analyze_motility
from automorphotrack.colocalization import analyze_colocalization
from automorphotrack.summary import summarize_integrated_data
from automorphotrack.temporal_dynamics import analyze_temporal_dynamics
from automorphotrack.spatial_statistics import analyze_spatial_statistics
from automorphotrack.network_analysis import analyze_network_topology

# --- Validation Module Imports ---
from automorphotrack.validation import (
    validate_segmentation,
    sensitivity_analysis,
    validate_tracking,
    generate_validation_report,
    generate_synthetic_ground_truth,
)

# --- Adaptive segmentation backends (R2.4) ---
from automorphotrack.adaptive_segmentation import (
    segment,
    segment_otsu,
    segment_sauvola,
    segment_niblack,
    segment_local_otsu,
    segment_subtracted,
    BACKENDS as SEGMENTATION_BACKENDS,
)

# --- Benchmarking against other tools (R1.2 / R2.3) ---
from automorphotrack import benchmarking

# --- Stats annotation helper (R2.5.iv) ---
from automorphotrack.utils import annotate_stats

# --- Convenience Alias (optional shortcut API) ---
__all__ = [
    # utilities
    "ensure_dir", "save_high_dpi", "upscale_frame", "write_video",
    "annotate_stats",
    # pipeline
    "detect_organelles", "count_lysosomes_per_frame",
    "classify_morphology", "analyze_shape_features", "profile_shape_data",
    "track_organelles", "track_overlay", "analyze_motility",
    "analyze_colocalization", "summarize_integrated_data",
    "analyze_temporal_dynamics", "analyze_spatial_statistics",
    "analyze_network_topology",
    # validation
    "validate_segmentation", "sensitivity_analysis", "validate_tracking",
    "generate_validation_report", "generate_synthetic_ground_truth",
    # adaptive segmentation backends
    "segment", "segment_otsu", "segment_sauvola", "segment_niblack",
    "segment_local_otsu", "segment_subtracted", "SEGMENTATION_BACKENDS",
    # benchmarking
    "benchmarking",
]
