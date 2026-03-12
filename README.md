# AutoMorphoTrack

**AutoMorphoTrack** is a comprehensive, modular image-analysis pipeline for automated detection, morphology classification, shape profiling, motility tracking, and colocalization analysis of mitochondria and lysosomes in multichannel fluorescence microscopy data.

Developed by **Armin Bayati, Ph.D.**

---

## Overview

AutoMorphoTrack processes time-lapse `.tif` stacks (typically two-channel: mitochondria + lysosomes) and generates **publication-ready visual and quantitative outputs** at every analysis step. The pipeline integrates seamlessly into scientific workflows and prioritizes **transparency, reproducibility, and ease of use**.

---

## Key Features

- **Automated organelle detection** with morphological filtering
- **Integrated tracking** of mitochondria and lysosomes across time
- **Shape profiling** with publication-ready violin plots
- **Motility analysis** with velocity and displacement quantification
- **Colocalization metrics** (Manders, Pearson coefficients)
- **Comprehensive validation tools** for segmentation and tracking accuracy
- **Colorblind-safe visualizations** throughout
- **High-resolution outputs** (600 DPI publication-ready figures)

---

## Installation

### Via PyPI (Recommended)
```bash
pip install automorphotrack
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
pip install -e ".[dev]"
```

---

## Quick Start

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
    summarize_integrated_data
)

tif_path = "path/to/Composite.tif"

# Run full pipeline
detect_organelles(tif_path)
count_lysosomes_per_frame(tif_path)
classify_morphology(tif_path)
analyze_shape_features(tif_path)
profile_shape_data()
track_organelles(tif_path)
track_overlay(tif_path)
analyze_motility()
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

**Optional:**
- scikit-learn (for validation metrics)

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

This package was developed with support from AI language models (e.g., ChatGPT, Claude). The term "AI assistance" refers to using external LLMs to help generate, review, and refine analysis code—**NOT** a built-in AI interface or algorithmic component. All algorithms are explicit, interpretable, and scientifically grounded.

---

## License

MIT License – See [LICENSE.md](LICENSE.md) for details.

---

## Citation

If you use this pipeline in your work, please cite:

```bibtex
@article{bayati2025automorphotrack,
  title={AutoMorphoTrack: Automated Organelle Tracking and Morphometric Profiling Toolkit},
  author={Bayati, Armin and others},
  year={2025}
}
```

Or in text:

> **Bayati, A. et al.** AutoMorphoTrack: Automated Organelle Tracking and Morphometric Profiling Toolkit (2025)

---

## Documentation

For detailed documentation, examples, and tutorials, visit:
- **GitHub**: [github.com/abayatibrain/AutoMorphoTrack](https://github.com/abayatibrain/AutoMorphoTrack)
- **Issues & Support**: [github.com/abayatibrain/AutoMorphoTrack/issues](https://github.com/abayatibrain/AutoMorphoTrack/issues)

---

## Author

**Armin Bayati, Ph.D.**
Email: [a.bayati.brain@gmail.com](mailto:a.bayati.brain@gmail.com)
