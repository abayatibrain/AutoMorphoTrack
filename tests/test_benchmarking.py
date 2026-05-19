"""Tests for the benchmarking exporters (Reviewer #1.2 / Reviewer #2.3)."""

import pandas as pd

from automorphotrack import benchmarking as bm


def _fake_shape_csv(tmp_path):
    df = pd.DataFrame({
        "Frame": [0, 0, 0, 1, 1],
        "Area": [10.0, 22.5, 8.0, 30.0, 15.0],
        "Eccentricity": [0.2, 0.7, 0.4, 0.5, 0.3],
        "Solidity": [0.95, 0.7, 0.88, 0.91, 0.9],
        "Circularity": [0.85, 0.6, 0.92, 0.7, 0.95],
        "Aspect_Ratio": [1.1, 2.5, 1.4, 1.7, 1.05],
        "Orientation": [0.1, -0.5, 0.3, 0.2, -0.1],
    })
    p = tmp_path / "shape.csv"
    df.to_csv(p, index=False)
    return p


def test_cellprofiler_export_columns(tmp_path):
    src = _fake_shape_csv(tmp_path)
    out = bm.export_for_cellprofiler(src, tmp_path / "cp.csv")
    df = pd.read_csv(out)
    for c in ["AreaShape_Area", "AreaShape_Eccentricity",
              "AreaShape_Solidity", "AreaShape_FormFactor",
              "ImageNumber", "ObjectNumber"]:
        assert c in df.columns, f"Missing CellProfiler column {c}"


def test_mina_export_columns(tmp_path):
    src = _fake_shape_csv(tmp_path)
    out = bm.export_for_mina(src, tmp_path / "mina.csv")
    df = pd.read_csv(out)
    assert "Mean Area" in df.columns
    assert "Form Factor" in df.columns
    assert df["Mean Area"].notna().any()


def test_mitograph_export_grouping(tmp_path):
    src = _fake_shape_csv(tmp_path)
    out = bm.export_for_mitograph(src, tmp_path / "mito.csv")
    df = pd.read_csv(out)
    # One row per frame in source data
    assert len(df) == 2
    assert {"Image", "Total_Volume_um3", "N_Mitochondria"}.issubset(df.columns)


def test_comparison_table_returns_one_row_per_metric(tmp_path):
    a = _fake_shape_csv(tmp_path)
    b_df = pd.read_csv(a)
    b_df["Area"] = b_df["Area"] * 1.05  # 5% systematic offset
    b = tmp_path / "shape_other.csv"
    b_df.to_csv(b, index=False)
    table = bm.comparison_table(a, b, other_tool_name="Other")
    assert "Metric" in table.columns
    assert "Spearman_rho" in table.columns
    assert "Percent_diff_means" in table.columns
    assert len(table) >= 1
