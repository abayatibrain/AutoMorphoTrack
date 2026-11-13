"""A single parameterized pipeline with retained masks and run provenance."""

from pathlib import Path
from importlib import metadata as import_metadata
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import tifffile

from .config import PipelineConfig
from .io import load_stack, sha256_file
from .core import segment_frame, measure_shapes, link_centroids, motility_from_tracks, colocalization_metrics, audit_drift


VERSION = "1.1.0rc2"


def frame_summaries(mito_shapes, lyso_shapes, motility, colocalization, n_frames):
    merged = pd.DataFrame({"Frame": np.arange(n_frames)})
    metrics = ["Area_px2", "Circularity", "Eccentricity", "Solidity", "Aspect_Ratio"]
    for name, table in (("Mito", mito_shapes), ("Lyso", lyso_shapes)):
        summary = table.groupby("Frame")[metrics].mean().add_prefix(name + "_").reset_index()
        counts = table.groupby("Frame").size().rename(name + "_Count").reset_index()
        merged = merged.merge(summary, on="Frame", how="left", validate="one_to_one")
        merged = merged.merge(counts, on="Frame", how="left", validate="one_to_one")
        merged[name + "_Count"] = merged[name + "_Count"].fillna(0).astype(int)
        if len(motility):
            sub = motility[motility["Organelle_Type"].eq(name)][["Frame", "Mean_Speed"]]
            merged = merged.merge(sub.rename(columns={"Mean_Speed": name + "_Mean_Speed"}),
                                  on="Frame", how="left", validate="one_to_one")
        else:
            merged[name + "_Mean_Speed"] = np.nan
    merged = merged.merge(colocalization, on="Frame", how="left", validate="one_to_one")
    excluded = {"Frame", "ROI_Pixels", "Overlap_Pixels", "Mito_Positive_Pixels", "Lyso_Positive_Pixels"}
    numeric = merged.select_dtypes(include=[np.number]).drop(columns=list(excluded), errors="ignore")
    corr = numeric.corr(method="spearman", min_periods=3)
    valid = numeric.notna().astype(int)
    pair_n = valid.T.dot(valid)
    return merged, corr, pair_n


def _source_identity():
    root = Path(__file__).parent
    files = {path.name: sha256_file(path) for path in sorted(root.glob("*.py"))}
    aggregate = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return files, aggregate


def run_pipeline(tif_path, out_dir="AutoMorphoTrack_Outputs", config=None, roi=None, external_masks=None):
    """Analyze one 2D multichannel sequence. ROI applies to colocalization only.

    Existing run outputs are never overwritten. Unknown temporal or spatial
    calibration leaves distances in pixels and durations in frames.
    """
    config = config or PipelineConfig()
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be PipelineConfig")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    stack, metadata = load_stack(tif_path, config)
    if roi is not None:
        roi = np.asarray(roi, dtype=bool)
        if roi.shape not in (stack.shape[1:3], stack.shape[:3]):
            raise ValueError("ROI must be YX or TYX with matching dimensions")
        if roi.ndim == 2:
            roi = np.broadcast_to(roi, stack.shape[:3])
        if not np.all(roi.any(axis=(1, 2))):
            raise ValueError("Every ROI frame must contain at least one pixel")
    results = {"stack": stack, "config": config, "metadata": metadata}
    external_masks = external_masks or {}
    if set(external_masks).difference({"mito", "lyso"}):
        raise ValueError("External masks must be keyed mito and/or lyso")
    external_hashes = {}
    thresholds = []
    motility_frames = []
    for name, channel, segmentation in (("mito", config.mito_channel, config.mito_segmentation),
                                         ("lyso", config.lyso_channel, config.lyso_segmentation)):
        if segmentation.method == "external":
            if name not in external_masks:
                raise ValueError(f"External {name} segmentation requires a TYX labeled array")
            labels = np.asarray(external_masks[name])
            if (labels.shape != stack.shape[:3] or labels.dtype.kind not in "biu" or np.any(labels < 0)
                    or np.any(labels > np.iinfo(np.uint32).max)):
                raise ValueError("External labels must be nonnegative integer TYX arrays matching the image")
            if labels.dtype.kind == "b":
                from skimage.measure import label
                labels = np.array([label(mask, connectivity=2) for mask in labels])
            labels = labels.astype(np.int64)
            external_hashes[name] = hashlib.sha256(labels.tobytes()).hexdigest()
            masks = labels > 0
            thresholds.extend(dict(Frame=frame, Organelle_Type=name.title(), Threshold=np.nan)
                              for frame in range(len(stack)))
            shapes, centroids = measure_shapes(labels, config, labeled=True)
        else:
            if name in external_masks:
                raise ValueError("Supplied external masks require method='external' in segmentation config")
            masks = []
            for frame in range(len(stack)):
                mask, threshold = segment_frame(stack[frame, ..., channel], segmentation)
                masks.append(mask)
                thresholds.append(dict(Frame=frame, Organelle_Type=name.title(), Threshold=threshold))
            masks = np.asarray(masks)
            from skimage.measure import label
            labels = np.array([label(mask, connectivity=2) for mask in masks])
            shapes, centroids = measure_shapes(labels, config, labeled=True)
        labels_per_frame = [shapes.loc[shapes.Frame.eq(frame), "Label"].tolist() for frame in range(len(stack))]
        tracks = link_centroids(centroids, config.max_link_distance_px, labels_per_frame)
        steps, motility, per_frame = motility_from_tracks(tracks, config)
        per_frame["Organelle_Type"] = name.title()
        motility_frames.append(per_frame)
        results.update({f"{name}_masks": masks, f"{name}_shapes": shapes, f"{name}_tracks": tracks,
                        f"{name}_steps": steps, f"{name}_motility": motility, f"{name}_labels": labels})
    coloc = []
    for frame in range(len(stack)):
        row = colocalization_metrics(stack[frame, ..., config.mito_channel], stack[frame, ..., config.lyso_channel],
                                     results["mito_masks"][frame], results["lyso_masks"][frame],
                                     roi[frame] if roi is not None else None, config.background_percentile)
        coloc.append(dict(Frame=frame, **row))
    results["colocalization"] = pd.DataFrame(coloc)
    results["motility_per_frame"] = pd.concat(motility_frames, ignore_index=True)
    results["thresholds"] = pd.DataFrame(thresholds)
    frames, corr, corr_n = frame_summaries(results["mito_shapes"], results["lyso_shapes"],
                                          results["motility_per_frame"], results["colocalization"], len(stack))
    results.update(frame_summary=frames, correlation=corr, correlation_n=corr_n)
    sample = frames.drop(columns="Frame").mean(numeric_only=True).to_dict()
    sample.update(Sample_ID=config.sample_id, Biological_Replicate_ID=config.biological_replicate_id,
                  Frames=len(stack), Analysis_Unit="one image stack; independence unverified")
    results["sample_summary"] = pd.DataFrame([sample])
    if config.audit_drift:
        results["drift_audit"] = audit_drift(stack, (config.mito_channel, config.lyso_channel))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mito_shapes": "Mito_ShapeMetrics.csv", "lyso_shapes": "Lyso_ShapeMetrics.csv",
        "mito_tracks": "Mito_Tracks.csv", "lyso_tracks": "Lyso_Tracks.csv",
        "mito_steps": "Mito_MotilitySteps.csv", "lyso_steps": "Lyso_MotilitySteps.csv",
        "mito_motility": "Mito_MotilitySummary.csv", "lyso_motility": "Lyso_MotilitySummary.csv",
        "motility_per_frame": "Motility_PerFrame.csv", "colocalization": "Colocalization.csv",
        "frame_summary": "Frame_Summary.csv", "sample_summary": "Sample_Summary.csv",
        "thresholds": "Thresholds.csv",
    }
    for key, filename in paths.items():
        table = results[key].copy()
        if "Sample_ID" not in table:
            table.insert(0, "Sample_ID", config.sample_id)
        table.to_csv(out_dir / filename, index=False)
    if config.audit_drift:
        results["drift_audit"].to_csv(out_dir / "Drift_Audit.csv", index=False)
    corr.to_csv(out_dir / "Frame_Descriptive_Spearman.csv")
    corr_n.to_csv(out_dir / "Frame_Correlation_N.csv")
    tifffile.imwrite(out_dir / "Mito_Masks.tif", results["mito_masks"].astype(np.uint8), photometric="minisblack", metadata={"axes": "TYX"})
    tifffile.imwrite(out_dir / "Lyso_Masks.tif", results["lyso_masks"].astype(np.uint8), photometric="minisblack", metadata={"axes": "TYX"})
    tifffile.imwrite(out_dir / "Mito_InstanceLabels.tif", results["mito_labels"].astype(np.uint32), photometric="minisblack", metadata={"axes": "TYX"})
    tifffile.imwrite(out_dir / "Lyso_InstanceLabels.tif", results["lyso_labels"].astype(np.uint32), photometric="minisblack", metadata={"axes": "TYX"})
    if roi is not None:
        tifffile.imwrite(out_dir / "Colocalization_ROI.tif", roi.astype(np.uint8), photometric="minisblack", metadata={"axes": "TYX"})
    source_files, source_hash = _source_identity()
    input_hash = sha256_file(tif_path)
    roi_hash = hashlib.sha256(roi.tobytes()).hexdigest() if roi is not None else None
    dependencies = {}
    for package in ("numpy", "pandas", "scipy", "scikit-image", "tifffile", "matplotlib"):
        try:
            dependencies[package] = import_metadata.version(package)
        except import_metadata.PackageNotFoundError:
            dependencies[package] = None
    identity = dict(input_sha256=input_hash, configuration=config.to_dict(), source_sha256=source_hash,
                    roi_sha256=roi_hash, external_labels_sha256=external_hashes,
                    environment={"dependencies": dependencies, "python": platform.python_version(),
                                 "system": platform.system(), "machine": platform.machine()})
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = dict(version=VERSION, run_id=run_id, created_utc=datetime.now(timezone.utc).isoformat(),
                    input_path=str(Path(tif_path).resolve()), **identity, source_files=source_files,
                    input_metadata=metadata, python=sys.version, executable=sys.executable,
                    platform=platform.platform(), dependencies=dependencies,
                    units={"coordinate": "px", "area": "px2", "distance": "um" if config.pixel_size_um else "px",
                           "time": "s" if config.frame_interval_s else "frame"},
                    warnings=["Centroid links are estimates; crossings, merges, splits, and gaps are unresolved",
                              "Ambiguous links are excluded from speed by default",
                              "Frame correlations are descriptive and temporally dependent",
                              "Pixel overlap does not establish contact or organelle interaction",
                              "No biological ground truth or replicate independence is inferred"],
                    observations={name: len(results[name + "_tracks"]) for name in ("mito", "lyso")},
                    ambiguous_links={name: int(results[name + "_tracks"]["Link_Status"].eq("ambiguous").sum())
                                     for name in ("mito", "lyso")})
    if config.render:
        from .plotting import render_outputs
        render_outputs(results, out_dir)
    manifest["output_files"] = {path.name: sha256_file(path) for path in sorted(out_dir.iterdir()) if path.is_file()}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    results["manifest"] = manifest
    return results
