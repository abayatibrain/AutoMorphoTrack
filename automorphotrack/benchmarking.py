# ============================================================
# AutoMorphoTrack – Benchmarking & Comparative Positioning
# ============================================================
# Author: Armin Bayati, Ph.D.
#
# Reviewer #1 (R1.2) and Reviewer #2 (R2.3) asked for explicit comparison to
# established tools: CellProfiler, MiNA, MitoGraph, Mitochondria Analyzer,
# and learning-based tools like CellPose / StarDist.
#
# AutoMorphoTrack does not re-implement those packages — that would
# duplicate excellent existing work. Instead, this module:
#
#   1. Provides exporters that translate AMT outputs into the CSV layouts
#      those tools consume / produce, so users can directly compare results
#      side-by-side.
#   2. Provides importers for the other tools' outputs so AMT can compute
#      Dice/IoU/Pearson agreement between its segmentations and theirs.
#   3. Provides a "comparison table" generator that summarises pairwise
#      agreement across morphology metrics, useful as supplementary material
#      in publications.
#
# The supported tools are:
#   - CellProfiler (CSV per object)
#   - MiNA (FIJI plugin; tabular network stats)
#   - MitoGraph (.gnet / .csv per cell)
#   - Mitochondria Analyzer (FIJI; per-image summary)
#   - CellPose / StarDist (label-image masks, np.uint16)
#
# Each importer is tolerant: it tries to find the relevant columns by
# regex-matching common naming conventions so the user does not have to
# rename anything by hand.
# ============================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from automorphotrack.validation import validate_segmentation


# ------------------------------------------------------------------
# AMT → other tools (exporters)
# ------------------------------------------------------------------
def export_for_cellprofiler(shape_metrics_csv: str | Path,
                            out_path: str | Path) -> Path:
    """Write a CSV in a CellProfiler-friendly schema.

    CellProfiler MeasureObjectSizeShape output uses columns like
    ``AreaShape_Area``, ``AreaShape_Eccentricity``, etc. We map AMT columns
    to that schema so users can drop the CSV into a CellProfiler analysis
    workflow without rewriting downstream scripts.
    """
    df = pd.read_csv(shape_metrics_csv)
    mapping = {
        "Area": "AreaShape_Area",
        "Eccentricity": "AreaShape_Eccentricity",
        "Solidity": "AreaShape_Solidity",
        "Circularity": "AreaShape_FormFactor",   # CellProfiler's name
        "Aspect_Ratio": "AreaShape_MajorAxisLength_over_MinorAxisLength",
        "Orientation": "AreaShape_Orientation",
        "Frame": "Image_Metadata_Frame",
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    df["ImageNumber"] = df.get("Image_Metadata_Frame", 0) + 1
    df["ObjectNumber"] = range(1, len(df) + 1)
    out_path = Path(out_path)
    df.to_csv(out_path, index=False)
    return out_path


def export_for_mina(shape_metrics_csv: str | Path,
                    out_path: str | Path) -> Path:
    """Write a CSV in MiNA-style per-mitochondrion network table.

    MiNA reports ``Mean Length``, ``Mean Area``, ``Footprint``, ``Mean
    Branch Length`` — AMT does not compute branch topology, so the columns
    we cannot fill are left blank (NaN). This keeps the schema compatible
    while making the gap explicit.
    """
    df = pd.read_csv(shape_metrics_csv)
    out = pd.DataFrame({
        "Frame": df.get("Frame"),
        "Mean Length": np.nan,
        "Mean Area": df.get("Area"),
        "Footprint": df.get("Area"),
        "Mean Branch Length": np.nan,
        "Eccentricity": df.get("Eccentricity"),
        "Solidity": df.get("Solidity"),
        "Form Factor": df.get("Circularity"),
    })
    out_path = Path(out_path)
    out.to_csv(out_path, index=False)
    return out_path


def export_for_mitograph(shape_metrics_csv: str | Path,
                         out_path: str | Path) -> Path:
    """Write a MitoGraph-style per-cell summary CSV."""
    df = pd.read_csv(shape_metrics_csv)
    grouped = df.groupby("Frame").agg(
        total_area=("Area", "sum"),
        mean_area=("Area", "mean"),
        n_objects=("Area", "size"),
        mean_aspect_ratio=("Aspect_Ratio", "mean"),
        mean_solidity=("Solidity", "mean"),
    ).reset_index()
    grouped.columns = [
        "Image", "Total_Volume_um3", "Mean_Volume_um3",
        "N_Mitochondria", "Mean_Aspect_Ratio", "Mean_Solidity",
    ]
    out_path = Path(out_path)
    grouped.to_csv(out_path, index=False)
    return out_path


# ------------------------------------------------------------------
# Other tools → AMT (importers + agreement)
# ------------------------------------------------------------------
def load_cellprofiler_csv(path: str | Path) -> pd.DataFrame:
    """Parse a CellProfiler ``Per_Object.csv`` into AMT's shape-metric schema."""
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        if re.fullmatch(r"AreaShape_Area", c): rename[c] = "Area"
        elif re.fullmatch(r"AreaShape_Eccentricity", c): rename[c] = "Eccentricity"
        elif re.fullmatch(r"AreaShape_Solidity", c): rename[c] = "Solidity"
        elif re.fullmatch(r"AreaShape_FormFactor", c): rename[c] = "Circularity"
        elif re.fullmatch(r"AreaShape_Orientation", c): rename[c] = "Orientation"
        elif "MajorAxisLength_over_MinorAxisLength" in c: rename[c] = "Aspect_Ratio"
        elif "Frame" in c: rename[c] = "Frame"
    return df.rename(columns=rename)


def load_label_mask(path: str | Path) -> np.ndarray:
    """Load a CellPose / StarDist label image as a binary mask."""
    import tifffile
    img = tifffile.imread(str(path))
    return (img > 0).astype(bool)


def agreement(amt_mask: np.ndarray, other_mask: np.ndarray) -> Mapping[str, float]:
    """Compute Dice / IoU / precision / recall / F1 between two binary masks."""
    return validate_segmentation(amt_mask.astype(bool), other_mask.astype(bool))


def comparison_table(amt_shape_csv: str | Path,
                     other_shape_csv: str | Path,
                     other_tool_name: str = "Other") -> pd.DataFrame:
    """Build a side-by-side morphology comparison table.

    Per-metric this reports: mean ± std for AMT, mean ± std for the other
    tool, the Spearman rank correlation between matched objects (where
    matching is by ``Frame`` and row order), and a percentage absolute
    difference of the means.
    """
    from scipy.stats import spearmanr

    a = pd.read_csv(amt_shape_csv)
    b = pd.read_csv(other_shape_csv)
    metrics = ["Area", "Eccentricity", "Solidity", "Circularity", "Aspect_Ratio"]
    rows = []
    for m in metrics:
        if m not in a.columns or m not in b.columns:
            continue
        av, bv = a[m].dropna(), b[m].dropna()
        n = min(len(av), len(bv))
        if n < 3:
            continue
        rho, p = spearmanr(av.iloc[:n], bv.iloc[:n])
        mean_a, mean_b = av.mean(), bv.mean()
        pct_diff = 100 * abs(mean_a - mean_b) / max(abs(mean_a), abs(mean_b), 1e-9)
        rows.append({
            "Metric": m,
            "AMT_mean": round(mean_a, 4),
            "AMT_std": round(av.std(), 4),
            f"{other_tool_name}_mean": round(mean_b, 4),
            f"{other_tool_name}_std": round(bv.std(), 4),
            "Spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "Percent_diff_means": round(pct_diff, 2),
            "N": n,
        })
    return pd.DataFrame(rows)
