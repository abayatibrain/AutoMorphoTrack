# ============================================================
# AutoMorphoTrack – Napari Plugin GUI
# ============================================================
# Author: Armin Bayati, Ph.D.
#
# A drag-and-drop GUI for the AutoMorphoTrack pipeline, registered as a
# napari plugin so it appears in the napari Plugins menu after install.
#
# Install:
#     pip install "automorphotrack[napari]"
#
# Launch:
#     napari       # then  Plugins → AutoMorphoTrack
#     # or directly:
#     automorphotrack gui
#
# Widgets exposed:
#     - "Detect organelles"       – runs detection, adds masks as layers
#     - "Shape features"          – per-organelle morphometrics + plots
#     - "Tracking + motility"     – reconstruct tracks, scatter & KDE
#     - "Colocalization"          – Manders / Jaccard / Pearson / cosine
#     - "Validation"              – Dice / IoU / sensitivity sweeps
#     - "Run full pipeline"       – everything at once into one out-dir
#     - "Segmentation backend"    – switch Otsu / Sauvola / Niblack / …
#
# This module imports cleanly even without napari/magicgui installed; the
# plugin entry points are only activated if those packages are present.
# That keeps `pip install automorphotrack` cheap for headless users.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from magicgui import magic_factory
    from magicgui.widgets import Container, FileEdit
    _NAPARI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NAPARI_AVAILABLE = False
    magic_factory = None  # type: ignore
    Container = None  # type: ignore


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _load_stack(path: str) -> np.ndarray:
    import tifffile
    stack = tifffile.imread(str(path))
    if stack.ndim == 3 and stack.shape[1] == 3 and stack.shape[-1] != 3:
        stack = np.moveaxis(stack, 1, -1)
    return stack


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    return (img - img.min()) / (np.ptp(img) + 1e-12)


# ----------------------------------------------------------------------
# Widgets (each is gated behind napari/magicgui availability)
# ----------------------------------------------------------------------
if _NAPARI_AVAILABLE:

    @magic_factory(
        call_button="Detect",
        tif_path={"label": "TIF stack", "mode": "r", "filter": "*.tif *.tiff"},
        mito_channel={"label": "Mito channel", "min": 0, "max": 5},
        lyso_channel={"label": "Lyso channel", "min": 0, "max": 5},
        backend={
            "label": "Backend",
            "choices": ["otsu", "sauvola", "niblack", "local_otsu", "subtracted"],
        },
        thr_factor={"label": "thr_factor", "min": 0.1, "max": 2.0, "step": 0.05},
        min_size={"label": "min_size (px)", "min": 0, "max": 200},
    )
    def detect_widget(
        tif_path: Path,
        mito_channel: int = 0,
        lyso_channel: int = 1,
        backend: str = "otsu",
        thr_factor: float = 0.8,
        min_size: int = 3,
    ) -> "list":  # napari LayerDataTuple list
        """Segment mitochondria and lysosomes; add masks as napari Labels layers."""
        from automorphotrack.adaptive_segmentation import segment

        stack = _load_stack(str(tif_path))
        # Use frame 0 for the interactive preview
        frame = stack[0] if stack.ndim == 4 else stack
        mito_img = _normalize(frame[..., mito_channel])
        lyso_img = _normalize(frame[..., lyso_channel])

        mito_mask = segment(mito_img, backend=backend, thr_factor=thr_factor,
                            min_size=min_size).astype(int)
        lyso_mask = segment(lyso_img, backend=backend, thr_factor=thr_factor,
                            min_size=min_size).astype(int) * 2  # color #2

        return [
            (mito_img, {"name": "Mito (raw)", "colormap": "cyan"}, "image"),
            (lyso_img, {"name": "Lyso (raw)", "colormap": "magenta"}, "image"),
            (mito_mask, {"name": f"Mito mask ({backend})"}, "labels"),
            (lyso_mask, {"name": f"Lyso mask ({backend})"}, "labels"),
        ]

    @magic_factory(
        call_button="Run shape features",
        tif_path={"label": "TIF stack", "mode": "r", "filter": "*.tif *.tiff"},
        out_dir={"label": "Output dir", "mode": "d"},
    )
    def shape_widget(
        tif_path: Path,
        out_dir: Path = Path("./Shape_Feature_Outputs"),
        mito_channel: int = 0,
        lyso_channel: int = 1,
    ) -> None:
        from automorphotrack import analyze_shape_features
        analyze_shape_features(
            str(tif_path), out_dir=str(out_dir),
            mito_channel=mito_channel, lyso_channel=lyso_channel,
        )

    @magic_factory(
        call_button="Run tracking + motility",
        tif_path={"label": "TIF stack", "mode": "r", "filter": "*.tif *.tiff"},
        out_dir={"label": "Output dir", "mode": "d"},
        min_detectable_displacement={
            "label": "Min detectable Δ (px)",
            "min": 0.0, "max": 10.0, "step": 0.1,
        },
    )
    def motility_widget(
        tif_path: Path,
        out_dir: Path = Path("./Motility_Outputs"),
        mito_channel: int = 0,
        lyso_channel: int = 1,
        fps: int = 5,
        min_detectable_displacement: float = 0.0,
    ) -> None:
        from automorphotrack import (
            track_organelles, analyze_motility,
        )
        track_out = Path(out_dir).parent / "Tracking_Outputs"
        track_organelles(str(tif_path), out_dir=str(track_out),
                         mito_channel=mito_channel, lyso_channel=lyso_channel)
        analyze_motility(
            mito_tracks_path=str(track_out / "Mito_Tracks.csv"),
            lyso_tracks_path=str(track_out / "Lyso_Tracks.csv"),
            out_dir=str(out_dir), fps=fps,
            min_detectable_displacement=min_detectable_displacement,
        )

    @magic_factory(
        call_button="Run colocalization",
        tif_path={"label": "TIF stack", "mode": "r", "filter": "*.tif *.tiff"},
        out_dir={"label": "Output dir", "mode": "d"},
    )
    def colocalization_widget(
        tif_path: Path,
        out_dir: Path = Path("./Colocalization_Outputs"),
        mito_channel: int = 0,
        lyso_channel: int = 1,
    ) -> None:
        from automorphotrack import analyze_colocalization
        analyze_colocalization(str(tif_path), out_dir=str(out_dir),
                               mito_channel=mito_channel, lyso_channel=lyso_channel)

    @magic_factory(
        call_button="Run validation sweep",
        tif_path={"label": "TIF stack", "mode": "r", "filter": "*.tif *.tiff"},
        out_dir={"label": "Output dir", "mode": "d"},
        param={"label": "Sweep param", "choices": ["thr_factor", "min_size"]},
        values_csv={
            "label": "Values (comma-separated)",
            "tooltip": "e.g. 0.4,0.6,0.8,1.0,1.2",
        },
        metric={"label": "Metric", "choices": ["dice", "iou", "precision", "recall", "f1"]},
    )
    def validation_widget(
        tif_path: Path,
        out_dir: Path = Path("./Validation_Outputs"),
        channel: int = 0,
        frame: int = 0,
        param: str = "thr_factor",
        values_csv: str = "0.4,0.6,0.8,1.0,1.2",
        metric: str = "dice",
    ) -> None:
        from automorphotrack.validation import sensitivity_analysis
        values = [float(v) for v in values_csv.split(",") if v.strip()]
        sensitivity_analysis(
            str(tif_path), param, values, channel,
            metric=metric, frame=frame, out_dir=str(out_dir),
        )

    @magic_factory(
        call_button="Run full pipeline",
        tif_path={"label": "TIF stack", "mode": "r", "filter": "*.tif *.tiff"},
        out_dir={"label": "Output root", "mode": "d"},
    )
    def full_pipeline_widget(
        tif_path: Path,
        out_dir: Path = Path("./AMT_Outputs"),
        mito_channel: int = 0,
        lyso_channel: int = 1,
        fps: int = 5,
        min_detectable_displacement: float = 0.0,
    ) -> None:
        """Run every pipeline stage and write all outputs under out_dir."""
        from automorphotrack.cli import _cmd_run
        import argparse
        ns = argparse.Namespace(
            tif=str(tif_path), out=str(out_dir),
            mito_channel=mito_channel, lyso_channel=lyso_channel,
            fps=fps, upscale=4, backend="otsu",
            min_detectable_displacement=min_detectable_displacement,
        )
        _cmd_run(ns)


# ----------------------------------------------------------------------
# Plugin entry points consumed by napari.yaml
# ----------------------------------------------------------------------
def make_detect_widget():
    if not _NAPARI_AVAILABLE:
        raise ImportError("Install napari extras: pip install 'automorphotrack[napari]'")
    return detect_widget()

def make_shape_widget():
    return shape_widget()

def make_motility_widget():
    return motility_widget()

def make_colocalization_widget():
    return colocalization_widget()

def make_validation_widget():
    return validation_widget()

def make_full_pipeline_widget():
    return full_pipeline_widget()


# ----------------------------------------------------------------------
# Launcher used by `automorphotrack gui`
# ----------------------------------------------------------------------
def launch_gui() -> int:
    """Open napari with all AMT widgets docked into the main window.

    Returns the napari event-loop exit code.
    """
    try:
        import napari
    except ImportError as e:
        raise ImportError(
            "napari is required for the AMT GUI.\n"
            "Install with:  pip install 'automorphotrack[napari]'\n"
            f"  ({e})"
        )
    viewer = napari.Viewer(title="AutoMorphoTrack")
    viewer.window.add_dock_widget(detect_widget(), name="Detect", area="right")
    viewer.window.add_dock_widget(shape_widget(), name="Shape", area="right")
    viewer.window.add_dock_widget(motility_widget(), name="Motility", area="right")
    viewer.window.add_dock_widget(colocalization_widget(), name="Coloc", area="right")
    viewer.window.add_dock_widget(validation_widget(), name="Validation", area="right")
    viewer.window.add_dock_widget(full_pipeline_widget(), name="Full pipeline", area="right")
    napari.run()
    return 0
