"""Compatibility entry point using the shared revision pipeline.

The revision API accepts PipelineConfig rather than unrecorded stage-specific
parameters. Each entry point runs the same analysis and returns all outputs.
"""

from .pipeline import run_pipeline


def track_organelles(tif_path="Composite.tif", out_dir="Tracking_Outputs", config=None):
    return run_pipeline(tif_path, out_dir, config)
