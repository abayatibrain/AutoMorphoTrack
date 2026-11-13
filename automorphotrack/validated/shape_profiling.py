"""Combine shape tables without treating object rows as independent replicates."""

from pathlib import Path
import pandas as pd


def profile_shape_data(mito_shape_path, lyso_shape_path, out_dir="Shape_Profiling_Outputs"):
    mito = pd.read_csv(mito_shape_path).assign(Organelle_Type="Mito")
    lyso = pd.read_csv(lyso_shape_path).assign(Organelle_Type="Lyso")
    table = pd.concat([mito, lyso], ignore_index=True)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "Combined_ShapeData.csv", index=False)
    return table
