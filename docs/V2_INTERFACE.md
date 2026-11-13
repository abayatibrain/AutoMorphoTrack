# AutoMorphoTrack

**AutoMorphoTrack** is a comprehensive, modular image-analysis pipeline for automated detection, morphology classification, shape profiling, motility tracking, and colocalization analysis of mitochondria and lysosomes in multichannel fluorescence microscopy data.

Developed by **Armin Bayati, Ph.D.**

---

## Overview

AutoMorphoTrack processes time-lapse `.tif` stacks (typically two-channel: mitochondria + lysosomes) and generates **publication-ready visual and quantitative outputs** at every analysis step. The pipeline integrates seamlessly into scientific workflows and prioritizes **transparency, reproducibility, and ease of use**.

---

## Key Features

- **Automated organelle detection** with morphological filtering
- **Pluggable segmentation backends** — Otsu, Sauvola, Niblack, local-Otsu,
  rolling-ball/top-hat — to handle uneven illumination, bleaching, and
  variable SNR (v2.2.0)
- **Integrated tracking** of mitochondria and lysosomes across time
- **Shape profiling** with publication-ready violin plots
- **Motility analysis** with velocity and displacement quantification,
  optional per-frame displacement floor to suppress jitter
- **Colocalization metrics** — Manders M1/M2 (intensity), Jaccard (mask),
  Pearson r (intensity), cosine similarity — with an accompanying
  metric-definitions CSV so reviewers and readers never have to guess what a
  column means
- **Comprehensive validation tools**: Dice / IoU / precision / recall / F1
  against expert annotations or synthetic ground truth, plus real parameter
  sensitivity sweeps
- **Benchmarking exporters** for CellProfiler, MiNA, and MitoGraph schemas
  so AMT results can be compared side-by-side with those tools (v2.2.0)
- **Command-line interface**: `automorphotrack run stack.tif --out ./results`
  (v2.2.0)
- **MCP connector for Claude Code / Claude Desktop**: drive the pipeline by
  natural language without leaving your AI assistant (v2.2.0,
  see [`docs/MCP.md`](docs/MCP.md))
- **Colorblind-safe visualizations** throughout
- **High-resolution outputs** (600 DPI publication-ready figures)

---

## Installation

### Via PyPI (Recommended)
```bash
pip install automorphotrack            # core install
pip install "automorphotrack[mcp]"     # + the Claude Code MCP connector
```

### From Source
```bash
git clone https://github.com/abayatibrain/AutoMorphoTrack.git
cd AutoMorphoTrack
pip install -e .
```

### Development Install
```bash
git clone https://github.com/abayatibrain/AutoMorphoTrack.git
cd AutoMorphoTrack
pip install -e ".[dev,mcp]"
pytest -q tests/
```

---

## Quick Start

### One-command CLI (v2.2.0+)

```bash
# Whole pipeline, one TIF in, one folder out
automorphotrack run path/to/Composite.tif --out ./results
amt run path/to/Composite.tif --out ./results          # short alias

# Just validate the segmentation against ground truth or synthetic GT
automorphotrack validate path/to/Composite.tif --channel 0

# Sensitivity sweep over thr_factor
automorphotrack sweep path/to/Composite.tif --param thr_factor \
    --values 0.4,0.6,0.8,1.0,1.2 --channel 0

# Re-export AMT outputs in another tool's CSV schema
automorphotrack benchmark results/Shape_Feature_Outputs/Mito_ShapeMetrics.csv \
    --for cellprofiler --out cp_export.csv
```

### Use from Claude Code (MCP)

```bash
pip install "automorphotrack[mcp]"
claude mcp add automorphotrack -- automorphotrack mcp
```

Then in Claude Code: *"Use automorphotrack to run the full pipeline on
./stack.tif and summarize the motility distribution."* See
[`docs/MCP.md`](docs/MCP.md) for the full tool list.

### Point-and-click GUI (napari plugin, v2.3.0+)

```bash
pip install "automorphotrack[napari]"
automorphotrack gui                # launches napari with widgets pre-docked
# or:  napari   →   Plugins → AutoMorphoTrack
```

Six widgets: Detect, Shape features, Tracking + motility, Colocalization,
Validation sweep, Run full pipeline. See [`docs/GUI.md`](docs/GUI.md).

### Python API

```python
from automorphotrack import (
    detect_organelles,
    count_lysosomes_per_frame,
    classify_morphology,
    analyze_shape_features,
    profile_shape_data,
    track_organelles,
    track_overlay,
    analyze_motility,
    analyze_colocalization,
    summarize_integrated_data,
    segment_sauvola,         # pluggable adaptive segmentation backend
    validate_segmentation,   # Dice / IoU / F1
    sensitivity_analysis,    # real parameter sweep
)

tif_path = "path/to/Composite.tif"
detect_organelles(tif_path)
count_lysosomes_per_frame(tif_path)
classify_morphology(tif_path)
analyze_shape_features(tif_path)
profile_shape_data()
track_organelles(tif_path)
track_overlay(tif_path)
analyze_motility(min_detectable_displacement=0.5)   # suppress sub-pixel jitter
analyze_colocalization(tif_path)
summarize_integrated_data()
```

---

## Module Overview

| Module | Function | Inputs | Outputs | Purpose |
|--------|----------|--------|---------|---------|
| **detection** | `detect_organelles()` | TIF file | PNG, MP4, masks | Segment mitochondria and lysosomes |
| **lyso_count** | `count_lysosomes_per_frame()` | TIF file | PNG, CSV, MP4 | Quantify lysosomes per frame |
| **morphology** | `classify_morphology()` | TIF file | PNG, CSV, MP4 | Classify mitochondria as elongated vs. punctate |
| **shape_features** | `analyze_shape_features()` | TIF file | PNG, CSV | Extract circularity, solidity, aspect ratio |
| **shape_profiling** | `profile_shape_data()` | CSVs | PNG, CSV | Generate combined violin plots |
| **tracking** | `track_organelles()` | TIF file | PNG, CSV, MP4 | Track organelle trajectories |
| **tracking_overlay** | `track_overlay()` | TIF file | PNG, CSV, MP4 | Overlay tracks on intensity images |
| **motility** | `analyze_motility()` | CSVs | PNG, CSV | Compute velocity and displacement |
| **colocalization** | `analyze_colocalization()` | TIF file | PNG, CSV, MP4 | Quantify mitochondrial-lysosomal overlap |
| **summary** | `summarize_integrated_data()` | CSVs | PNG, CSV | Correlation matrix across all metrics |
| **validation** | `validate_segmentation()` | Masks | dict | Compute Dice, IoU, F1, precision, recall |
| **validation** | `validate_tracking()` | CSVs | dict | Track accuracy metrics |
| **validation** | `sensitivity_analysis()` | TIF + params | CSV, plots | Parameter sweep analysis |
| **validation** | `generate_validation_report()` | Results | PNG, CSV | Comprehensive validation summary |

---

## Output Files

| Step | Output Type | Example Files |
|------|--------------|---------------|
| **Detection** | PNG + MP4 | `Mito_Frame0.png`, `Mitochondria_Detection.mp4` |
| **Lysosome Count** | PNG + CSV + MP4 | `Lyso_Count_Plot.png`, `Lysosome_Counts.csv` |
| **Morphology** | PNG + MP4 + CSV | `Morphology_Frame0_Labeled.png`, `Morphology_Labeled.mp4` |
| **Shape Features** | PNG + CSV | `Shape_Distributions.png`, `Mito_ShapeMetrics.csv` |
| **Shape Profiling** | PNG + CSV | `Shape_ViolinPlots.png`, `Combined_ShapeData.csv` |
| **Tracking** | PNG + MP4 + CSV | `Cumulative_Mito.png`, `Mito_Tracks.csv` |
| **Tracking Overlay** | PNG + MP4 | `Cumulative_Composite.png`, `Composite_CumulativeTracks.mp4` |
| **Motility** | PNG + CSV | `Motility_Distributions.png`, `Motility_Scatter.png` |
| **Colocalization** | PNG + MP4 + CSV | `Colocalization_Frame0.png`, `Colocalization.csv` |
| **Summary** | PNG + CSV | `Integrated_CorrelationMatrix.png`, `Integrated_Merged_Data.csv` |
| **Validation** | PNG + CSV | `Validation_Report.png`, `Validation_Summary.csv` |

---

## Dependencies

**Core Requirements:**
- Python ≥ 3.9
- numpy ≥ 1.23.0
- pandas ≥ 1.5.0
- matplotlib ≥ 3.6.0
- seaborn ≥ 0.12.0
- opencv-python ≥ 4.6.0
- scikit-image ≥ 0.19.0
- scipy ≥ 1.9.0
- tifffile ≥ 2022.8.12

**Optional extras:**
- `automorphotrack[mcp]` adds `mcp >= 1.0.0` for the Claude Code MCP connector
- `automorphotrack[dev]` adds `pytest`, `build`, `twine` for development

---

## Visualization Design

**Colorblind-Safe Palette:**
- Mitochondria: `#0173B2` (blue)
- Lysosomes: `#DE8F05` (orange)

All visualizations use colorblind-safe colors to ensure accessibility and publication readiness.

---

## Statistical Notes

**Correlation Analysis (Summary Module):**
The integrated summary module uses **Spearman rank correlation** to compute relationships between extracted metrics, providing a distribution-free assessment of feature associations.

---

## Comparison with Existing Tools

| Feature | AutoMorphoTrack | CellProfiler | MiNA | MitoGraph | Imaris |
|---------|-----------------|--------------|------|-----------|--------|
| **Ease of Use** | High (Python API) | Medium (GUI-heavy) | Medium | Medium | Low (proprietary) |
| **Transparency** | Full (open source) | Partial | Partial | Partial | Closed |
| **Lightweight** | Yes | No | No | No | No |
| **Publication-Ready Outputs** | Yes | Yes | Partial | Partial | Yes |
| **Tracking** | Yes | Optional | Yes | Yes | Yes |
| **Shape Profiling** | Yes | Limited | Yes | Yes | Yes |
| **Colocalization** | Yes | Yes | Limited | Limited | Yes |
| **Customizable** | Yes (modular) | Yes | Limited | Limited | No |
| **Cost** | Free | Free | Free | Free | Expensive |

**What Makes AutoMorphoTrack Different:**
- **Integrated workflow**: All analyses in one package without external dependencies
- **Full transparency**: Complete source code and methodology documentation
- **Lightweight**: Minimal dependencies, fast execution
- **Publication-ready**: High-DPI outputs and colorblind-safe visualizations by default
- **Modular design**: Use only the modules you need

---

## AI Assistance Note

AutoMorphoTrack **does** ship a real natural-language interface as of
v2.2.0: the [Model Context Protocol](https://modelcontextprotocol.io)
server (`automorphotrack mcp`) lets Claude Code, Claude Desktop, Cursor,
and any other MCP-aware client call AMT analyses directly from prompts.
See [`docs/MCP.md`](docs/MCP.md). All underlying algorithms remain
explicit, deterministic, and interpretable — no LLM is involved in the
image-analysis path itself; the MCP layer only routes natural-language
requests to the existing Python API.

---

## License

MIT License – See [LICENSE.md](LICENSE.md) for details.

---

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@article{bayati2026automorphotrack,
  title={AutoMorphoTrack: A modular framework for quantitative analysis of organelle morphology, motility, and interactions at single-cell resolution},
  author={Bayati, Armin and Schumacher, Jackson G. and Chen, Xiqun},
  journal={eLife},
  year={2026},
  doi={10.7554/eLife.109936.1},
  url={https://elifesciences.org/reviewed-preprints/109936}
}
```

Or in text:

> **Bayati A, Schumacher JG, Chen X.** AutoMorphoTrack: A modular
> framework for quantitative analysis of organelle morphology, motility,
> and interactions at single-cell resolution. *eLife* 2026; reviewed
> preprint RP109936. doi:10.7554/eLife.109936.1.

---

## Documentation

For detailed documentation, examples, and tutorials, visit:
- **GitHub**: [github.com/abayatibrain/AutoMorphoTrack](https://github.com/abayatibrain/AutoMorphoTrack)
- **Issues & Support**: [github.com/abayatibrain/AutoMorphoTrack/issues](https://github.com/abayatibrain/AutoMorphoTrack/issues)

---

## Author

**Armin Bayati, Ph.D.**
Email: [a.bayati.brain@gmail.com](mailto:a.bayati.brain@gmail.com)
