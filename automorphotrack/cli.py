# ============================================================
# AutoMorphoTrack – Command-Line Interface
# ============================================================
# Author: Armin Bayati, Ph.D.
#
# Reviewer #2 (R2.6) asked for a value-added demonstration with reduced
# setup burden compared to assembling FIJI/ImageJ scripts. This CLI exposes
# the entire pipeline as one command::
#
#     automorphotrack run path/to/stack.tif --out ./results
#
# Sub-commands:
#   run         – full pipeline on a TIF stack
#   validate    – run validation against synthetic or user ground truth
#   sweep       – parameter sensitivity sweep over thr_factor / min_size
#   benchmark   – export AMT output in CellProfiler / MiNA / MitoGraph schema
#   mcp         – start the AMT MCP server (for Claude Code integration)
#
# Installed as console_scripts entry point ``automorphotrack``.
# ============================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_run(sp):
    p = sp.add_parser("run", help="Run the full AMT pipeline on a TIF stack")
    p.add_argument("tif", help="Path to multichannel TIF stack")
    p.add_argument("--out", default="AMT_Outputs", help="Output directory")
    p.add_argument("--mito-channel", type=int, default=0)
    p.add_argument("--lyso-channel", type=int, default=1)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--upscale", type=int, default=4)
    p.add_argument("--backend", default="otsu",
                   choices=["otsu", "sauvola", "niblack", "local_otsu", "subtracted"],
                   help="Segmentation backend (R2.4)")
    p.add_argument("--min-detectable-displacement", type=float, default=0.0,
                   help="Per-frame displacement floor below which motion is treated as 0 (R3.4)")
    return p


def _add_validate(sp):
    p = sp.add_parser("validate", help="Run validation against ground truth")
    p.add_argument("tif", help="Path to TIF stack")
    p.add_argument("--gt", default=None,
                   help="Optional path to ground-truth label TIF (CellPose/StarDist style). "
                        "If omitted, synthetic GT is generated.")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="Validation_Outputs")
    return p


def _add_sweep(sp):
    p = sp.add_parser("sweep", help="Parameter sensitivity sweep")
    p.add_argument("tif", help="Path to TIF stack")
    p.add_argument("--param", default="thr_factor",
                   choices=["thr_factor", "min_size"])
    p.add_argument("--values", default="0.4,0.6,0.8,1.0,1.2",
                   help="Comma-separated list of values")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--metric", default="dice",
                   choices=["dice", "iou", "precision", "recall", "f1"])
    p.add_argument("--out", default="Validation_Outputs")
    return p


def _add_benchmark(sp):
    p = sp.add_parser("benchmark", help="Export AMT outputs in other tools' schemas")
    p.add_argument("shape_csv", help="Path to AMT shape-metrics CSV "
                                     "(e.g. Shape_Feature_Outputs/Mito_ShapeMetrics.csv)")
    p.add_argument("--for", dest="target", required=True,
                   choices=["cellprofiler", "mina", "mitograph"])
    p.add_argument("--out", required=True, help="Output CSV path")
    return p


def _add_mcp(sp):
    p = sp.add_parser("mcp", help="Start the AMT MCP server for Claude Code")
    p.add_argument("--transport", default="stdio", choices=["stdio"])
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="automorphotrack",
        description="AutoMorphoTrack — modular organelle morphology/motility analysis."
    )
    sp = parser.add_subparsers(dest="cmd", required=True)
    _add_run(sp)
    _add_validate(sp)
    _add_sweep(sp)
    _add_benchmark(sp)
    _add_mcp(sp)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "validate":
        return _cmd_validate(args)
    if args.cmd == "sweep":
        return _cmd_sweep(args)
    if args.cmd == "benchmark":
        return _cmd_benchmark(args)
    if args.cmd == "mcp":
        return _cmd_mcp(args)
    parser.error(f"Unknown command {args.cmd}")


def _cmd_run(args):
    from automorphotrack import (
        detect_organelles, count_lysosomes_per_frame, classify_morphology,
        analyze_shape_features, profile_shape_data, track_organelles,
        track_overlay, analyze_motility, analyze_colocalization,
        summarize_integrated_data,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[AMT] Running full pipeline on {args.tif} → {out}")
    detect_organelles(args.tif, out_dir=str(out / "Detection_Outputs"),
                      mito_channel=args.mito_channel, lyso_channel=args.lyso_channel,
                      upscale_factor=args.upscale, fps=args.fps)
    count_lysosomes_per_frame(args.tif, out_dir=str(out / "LysoCount_Outputs"),
                              lyso_channel=args.lyso_channel)
    classify_morphology(args.tif, out_dir=str(out / "Morphology_Outputs"),
                        mito_channel=args.mito_channel)
    analyze_shape_features(args.tif, out_dir=str(out / "Shape_Feature_Outputs"),
                           mito_channel=args.mito_channel,
                           lyso_channel=args.lyso_channel)
    profile_shape_data(
        mito_shape_path=str(out / "Shape_Feature_Outputs/Mito_ShapeMetrics.csv"),
        lyso_shape_path=str(out / "Shape_Feature_Outputs/Lyso_ShapeMetrics.csv"),
        out_dir=str(out / "Shape_Profiling_Outputs"),
    )
    track_organelles(args.tif, out_dir=str(out / "Tracking_Outputs"),
                     mito_channel=args.mito_channel, lyso_channel=args.lyso_channel)
    track_overlay(args.tif, tracks_dir=str(out / "Tracking_Outputs"),
                  out_dir=str(out / "Tracking_Overlays"),
                  mito_channel=args.mito_channel, lyso_channel=args.lyso_channel,
                  fps=args.fps)
    analyze_motility(
        mito_tracks_path=str(out / "Tracking_Outputs/Mito_Tracks.csv"),
        lyso_tracks_path=str(out / "Tracking_Outputs/Lyso_Tracks.csv"),
        out_dir=str(out / "Motility_Outputs"), fps=args.fps,
        min_detectable_displacement=args.min_detectable_displacement,
    )
    analyze_colocalization(args.tif, out_dir=str(out / "Colocalization_Outputs"),
                           mito_channel=args.mito_channel,
                           lyso_channel=args.lyso_channel, fps=args.fps)
    summarize_integrated_data(
        shape_metrics_path=str(out / "Shape_Feature_Outputs/Mito_ShapeMetrics.csv"),
        motility_path=str(out / "Motility_Outputs/Motility_PerFrame.csv"),
        colocalization_path=str(out / "Colocalization_Outputs/Colocalization.csv"),
        out_dir=str(out / "Summary_Outputs"),
    )
    print(f"[AMT] Done → {out.resolve()}")
    return 0


def _cmd_validate(args):
    from automorphotrack.validation import (
        validate_segmentation, generate_synthetic_ground_truth,
    )
    from automorphotrack.adaptive_segmentation import segment_otsu
    import tifffile

    stack = tifffile.imread(args.tif)
    if stack.ndim == 3 and stack.shape[1] == 3 and stack.shape[-1] != 3:
        import numpy as np
        stack = np.moveaxis(stack, 1, -1)
    frame_img = stack[args.frame][..., args.channel].astype(float)
    import numpy as np
    frame_img = (frame_img - frame_img.min()) / (np.ptp(frame_img) + 1e-12)

    if args.gt is None:
        gt = generate_synthetic_ground_truth(frame_img)
    else:
        gt = (tifffile.imread(args.gt) > 0).astype(bool)
    pred = segment_otsu(frame_img)
    metrics = validate_segmentation(pred, gt)
    print(f"[AMT] Validation result: {metrics}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([metrics]).to_csv(out / "Validation_Single.csv", index=False)
    return 0


def _cmd_sweep(args):
    from automorphotrack.validation import sensitivity_analysis
    values = [float(v) for v in args.values.split(",")]
    df = sensitivity_analysis(
        args.tif, args.param, values, args.channel,
        metric=args.metric, frame=args.frame, out_dir=args.out,
    )
    print(df.to_string(index=False))
    return 0


def _cmd_benchmark(args):
    from automorphotrack import benchmarking as bm
    func = {
        "cellprofiler": bm.export_for_cellprofiler,
        "mina": bm.export_for_mina,
        "mitograph": bm.export_for_mitograph,
    }[args.target]
    p = func(args.shape_csv, args.out)
    print(f"[AMT] Wrote {args.target} CSV → {p}")
    return 0


def _cmd_mcp(args):
    try:
        from automorphotrack.mcp_server import serve
    except ImportError as e:
        print(f"[AMT] MCP server requires the optional 'mcp' extra: "
              f"pip install automorphotrack[mcp]\n  ({e})", file=sys.stderr)
        return 1
    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
