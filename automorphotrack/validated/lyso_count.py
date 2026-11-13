"""Compatibility entry point using the shared revision pipeline.

The revision API accepts PipelineConfig rather than unrecorded stage-specific
parameters. Each entry point runs the same analysis and returns all outputs.
"""

from .pipeline import run_pipeline


def count_lysosomes_per_frame(tif_path="Composite.tif", out_dir="Lyso_Count_Outputs", config=None):
    return run_pipeline(tif_path, out_dir, config)
