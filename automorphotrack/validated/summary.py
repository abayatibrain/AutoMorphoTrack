"""Frame descriptions and explicitly grouped independent-sample summaries."""

from pathlib import Path
import pandas as pd


def summarize_integrated_data(frame_summary_path, out_dir="Summary_Outputs"):
    table = pd.read_csv(frame_summary_path)
    excluded = ["Frame", "Label", "Track_ID", "Sample_ID", "Biological_Replicate_ID",
                "ROI_Pixels", "Overlap_Pixels", "Mito_Positive_Pixels", "Lyso_Positive_Pixels"]
    numeric = table.drop(columns=excluded, errors="ignore").select_dtypes("number")
    corr = numeric.corr(method="spearman", min_periods=3)
    valid = numeric.notna().astype(int)
    counts = valid.T.dot(valid)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    corr.to_csv(out / "Frame_Descriptive_Spearman.csv")
    counts.to_csv(out / "Frame_Correlation_N.csv")
    return corr, counts


def aggregate_biological_replicates(sample_tables):
    """Require user-supplied biological IDs; collapse technical stacks first.

    This produces descriptive replicate means without p-values or a claim that
    labels establish experimental independence.
    """
    combined = pd.concat(sample_tables, ignore_index=True)
    if "Biological_Replicate_ID" not in combined or combined["Biological_Replicate_ID"].isna().any():
        raise ValueError("Every stack requires a user-supplied Biological_Replicate_ID")
    if combined["Sample_ID"].duplicated().any():
        raise ValueError("Duplicate Sample_ID values require explicit reconciliation")
    numeric = combined.select_dtypes("number").columns.difference(["Frames"])
    return combined.groupby("Biological_Replicate_ID")[numeric].mean().reset_index()
