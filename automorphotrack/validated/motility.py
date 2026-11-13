"""Motility from persistent revision tracks and explicit acquisition units."""

from pathlib import Path
import pandas as pd
from .config import PipelineConfig
from .core import motility_from_tracks


def analyze_motility(mito_tracks_path, lyso_tracks_path, out_dir="Motility_Outputs", config=None):
    config = config or PipelineConfig()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be empty")
    tables = {}
    for name, path in (("Mito", mito_tracks_path), ("Lyso", lyso_tracks_path)):
        table = pd.read_csv(path)
        if "Track_ID" not in table:
            raise ValueError("Legacy per-frame Organelle labels are not persistent tracks; rerun tracking")
        tables[name] = motility_from_tracks(table, config)
    out.mkdir(parents=True, exist_ok=True)
    for name, (steps, summary, per_frame) in tables.items():
        steps.to_csv(out / f"{name}_MotilitySteps.csv", index=False)
        summary.to_csv(out / f"{name}_MotilitySummary.csv", index=False)
        per_frame.to_csv(out / f"{name}_MotilityPerFrame.csv", index=False)
    return tables
