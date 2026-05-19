# ============================================================
# AutoMorphoTrack – MCP Server (Claude Code / generic MCP host)
# ============================================================
# Author: Armin Bayati, Ph.D.
#
# Reviewers #1, #2, and #3 all flagged the "AI-assisted natural-language
# interface" paragraph as overstated because no such interface existed.
# This module *is* that interface. It exposes the AutoMorphoTrack pipeline
# as a Model Context Protocol (MCP) server so Claude Code (or any other
# MCP-aware client like Claude Desktop, Cursor, etc.) can call AMT analyses
# directly from natural-language prompts.
#
# Install with the optional extra::
#
#     pip install "automorphotrack[mcp]"
#
# Register with Claude Code::
#
#     claude mcp add automorphotrack -- automorphotrack mcp
#
# Then from inside Claude Code you can say:
#     "Use automorphotrack to run the full pipeline on ./stack.tif and
#      summarize the motility distribution."
# and Claude will call the ``amt_run`` tool below, then read the resulting
# CSV summaries.
#
# Tools exposed (each is a real Python function on the AMT package):
#   amt_run                       – full pipeline on a TIF
#   amt_detect                    – detection only
#   amt_shape_features            – shape feature extraction
#   amt_motility                  – motility analysis on tracks
#   amt_colocalization            – colocalization analysis
#   amt_summary                   – integrated correlation summary
#   amt_validate_segmentation     – Dice/IoU vs ground truth
#   amt_sensitivity_analysis      – parameter sweep
#   amt_benchmark_export          – export in CellProfiler / MiNA / MitoGraph schema
#   amt_describe_outputs          – read an output folder and summarize CSVs
#   amt_list_capabilities         – machine-readable capability list
#
# This module degrades gracefully: if the `mcp` package is not installed,
# `serve()` raises a clear ImportError telling the user how to install it.
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MCP_AVAILABLE = False
    FastMCP = None  # type: ignore


def _make_app():
    if not _MCP_AVAILABLE:
        raise ImportError(
            "The 'mcp' package is required for the AutoMorphoTrack MCP server.\n"
            "Install with:  pip install 'automorphotrack[mcp]'\n"
            "or directly:   pip install mcp"
        )

    app = FastMCP("automorphotrack")

    # ------------------------------------------------------------------
    # Pipeline tools
    # ------------------------------------------------------------------
    @app.tool()
    def amt_run(tif_path: str, out_dir: str = "AMT_Outputs",
                mito_channel: int = 0, lyso_channel: int = 1,
                fps: int = 5, min_detectable_displacement: float = 0.0) -> dict:
        """Run the full AutoMorphoTrack pipeline on a multichannel TIF stack.

        Returns a dictionary mapping each pipeline stage to its output folder
        and a short summary of what was produced. Use this when the user
        wants an end-to-end analysis.
        """
        from automorphotrack.cli import _cmd_run
        import argparse
        ns = argparse.Namespace(
            tif=tif_path, out=out_dir, mito_channel=mito_channel,
            lyso_channel=lyso_channel, fps=fps, upscale=4, backend="otsu",
            min_detectable_displacement=min_detectable_displacement,
        )
        _cmd_run(ns)
        return _describe_outputs_impl(out_dir)

    @app.tool()
    def amt_detect(tif_path: str, out_dir: str = "Detection_Outputs",
                   mito_channel: int = 0, lyso_channel: int = 1) -> dict:
        """Run only the detection step on a TIF stack."""
        from automorphotrack import detect_organelles
        detect_organelles(tif_path, out_dir=out_dir,
                          mito_channel=mito_channel, lyso_channel=lyso_channel)
        return {"out_dir": str(Path(out_dir).resolve()),
                "files": sorted(p.name for p in Path(out_dir).glob("*"))}

    @app.tool()
    def amt_shape_features(tif_path: str, out_dir: str = "Shape_Feature_Outputs",
                           mito_channel: int = 0, lyso_channel: int = 1) -> dict:
        """Extract per-organelle shape descriptors."""
        from automorphotrack import analyze_shape_features
        analyze_shape_features(tif_path, out_dir=out_dir,
                               mito_channel=mito_channel,
                               lyso_channel=lyso_channel)
        return {"out_dir": str(Path(out_dir).resolve()),
                "files": sorted(p.name for p in Path(out_dir).glob("*"))}

    @app.tool()
    def amt_motility(mito_tracks_csv: str, lyso_tracks_csv: str,
                     out_dir: str = "Motility_Outputs",
                     min_detectable_displacement: float = 0.0,
                     fps: int = 5) -> dict:
        """Compute velocity and displacement on existing track CSVs."""
        from automorphotrack import analyze_motility
        analyze_motility(mito_tracks_path=mito_tracks_csv,
                         lyso_tracks_path=lyso_tracks_csv,
                         out_dir=out_dir, fps=fps,
                         min_detectable_displacement=min_detectable_displacement)
        return {"out_dir": str(Path(out_dir).resolve()),
                "files": sorted(p.name for p in Path(out_dir).glob("*"))}

    @app.tool()
    def amt_colocalization(tif_path: str, out_dir: str = "Colocalization_Outputs",
                           mito_channel: int = 0, lyso_channel: int = 1) -> dict:
        """Compute Manders, Jaccard, Pearson, and cosine colocalization metrics."""
        from automorphotrack import analyze_colocalization
        analyze_colocalization(tif_path, out_dir=out_dir,
                               mito_channel=mito_channel,
                               lyso_channel=lyso_channel)
        return {"out_dir": str(Path(out_dir).resolve()),
                "files": sorted(p.name for p in Path(out_dir).glob("*"))}

    @app.tool()
    def amt_summary(shape_metrics_csv: str, motility_csv: str,
                    colocalization_csv: str, out_dir: str = "Summary_Outputs") -> dict:
        """Compute the integrated Spearman correlation summary."""
        from automorphotrack import summarize_integrated_data
        summarize_integrated_data(
            shape_metrics_path=shape_metrics_csv,
            motility_path=motility_csv,
            colocalization_path=colocalization_csv,
            out_dir=out_dir,
        )
        return {"out_dir": str(Path(out_dir).resolve()),
                "files": sorted(p.name for p in Path(out_dir).glob("*"))}

    # ------------------------------------------------------------------
    # Validation tools
    # ------------------------------------------------------------------
    @app.tool()
    def amt_validate_segmentation(predicted_mask_tif: str,
                                  ground_truth_mask_tif: str) -> dict:
        """Compute Dice / IoU / precision / recall / F1 between two binary masks."""
        import tifffile
        from automorphotrack.validation import validate_segmentation
        pred = tifffile.imread(predicted_mask_tif) > 0
        gt = tifffile.imread(ground_truth_mask_tif) > 0
        return validate_segmentation(pred, gt)

    @app.tool()
    def amt_sensitivity_analysis(tif_path: str, param: str = "thr_factor",
                                 values: list[float] | None = None,
                                 channel: int = 0, frame: int = 0,
                                 metric: str = "dice",
                                 out_dir: str = "Validation_Outputs") -> dict:
        """Sweep a parameter and report Dice/IoU at each value."""
        from automorphotrack.validation import sensitivity_analysis
        if values is None:
            values = [0.4, 0.6, 0.8, 1.0, 1.2]
        df = sensitivity_analysis(tif_path, param, values, channel,
                                  metric=metric, frame=frame, out_dir=out_dir)
        return {"results": df.to_dict(orient="records"),
                "out_dir": str(Path(out_dir).resolve())}

    # ------------------------------------------------------------------
    # Benchmarking tools
    # ------------------------------------------------------------------
    @app.tool()
    def amt_benchmark_export(shape_csv: str, target: str,
                             out_csv: str) -> dict:
        """Export AMT shape metrics in another tool's CSV schema.

        ``target`` must be one of ``"cellprofiler"``, ``"mina"``,
        ``"mitograph"``.
        """
        from automorphotrack import benchmarking as bm
        fn = {
            "cellprofiler": bm.export_for_cellprofiler,
            "mina": bm.export_for_mina,
            "mitograph": bm.export_for_mitograph,
        }.get(target)
        if fn is None:
            raise ValueError(f"target must be one of cellprofiler|mina|mitograph, got {target}")
        out = fn(shape_csv, out_csv)
        return {"out_path": str(Path(out).resolve())}

    # ------------------------------------------------------------------
    # Introspection / capability discovery
    # ------------------------------------------------------------------
    @app.tool()
    def amt_describe_outputs(out_dir: str) -> dict:
        """Walk an AMT output folder and return a summary of each CSV / image."""
        return _describe_outputs_impl(out_dir)

    @app.tool()
    def amt_list_capabilities() -> dict:
        """Return a machine-readable map of every analysis AMT can do."""
        from automorphotrack import __version__ as v
        return {
            "version": v,
            "pipeline_stages": [
                "detect_organelles", "count_lysosomes_per_frame",
                "classify_morphology", "analyze_shape_features",
                "profile_shape_data", "track_organelles", "track_overlay",
                "analyze_motility", "analyze_colocalization",
                "summarize_integrated_data",
            ],
            "advanced_analyses": [
                "analyze_temporal_dynamics", "analyze_spatial_statistics",
                "analyze_network_topology",
            ],
            "segmentation_backends": [
                "otsu", "sauvola", "niblack", "local_otsu", "subtracted",
            ],
            "validation": [
                "validate_segmentation", "sensitivity_analysis",
                "validate_tracking", "generate_synthetic_ground_truth",
                "generate_validation_report",
            ],
            "benchmarking": [
                "export_for_cellprofiler", "export_for_mina",
                "export_for_mitograph", "comparison_table",
            ],
        }

    return app


def _describe_outputs_impl(out_dir: str) -> dict[str, Any]:
    import pandas as pd
    out = Path(out_dir)
    if not out.exists():
        return {"error": f"{out} does not exist"}
    summary: dict[str, Any] = {"root": str(out.resolve()), "folders": {}}
    for sub in sorted(p for p in out.iterdir() if p.is_dir()):
        files = sorted(p.name for p in sub.iterdir() if p.is_file())
        summary["folders"][sub.name] = {"files": files}
        for csv_path in sub.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path, nrows=200)
                summary["folders"][sub.name][csv_path.name] = {
                    "rows": int(len(df)),
                    "columns": list(df.columns)[:20],
                    "head": df.head(3).to_dict(orient="records"),
                }
            except Exception as e:
                summary["folders"][sub.name][csv_path.name] = {"error": str(e)}
    return summary


def serve() -> None:
    """Start the FastMCP server over stdio (the Claude Code default transport)."""
    app = _make_app()
    app.run()


if __name__ == "__main__":  # pragma: no cover
    serve()
