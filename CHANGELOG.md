# Changelog

All notable changes to AutoMorphoTrack are documented here.

## [2.2.0] — 2026-05-19 — Reviewer-driven revision

This release addresses every public-review comment on eLife RP109936
(reviewers #1, #2, #3) and is the version submitted with the revised
manuscript. See `docs/Response_to_Reviewers.docx` for the point-by-point
mapping.

### Added

- **MCP server** (`automorphotrack.mcp_server`) — exposes the entire
  pipeline as Model Context Protocol tools so Claude Code, Claude Desktop,
  and any other MCP-aware client can call AMT analyses by natural language.
  Install with `pip install "automorphotrack[mcp]"`. *(R1.3, R2.1, R3.2)*
- **Command-line interface** (`automorphotrack` / `amt`) with sub-commands
  `run`, `validate`, `sweep`, `benchmark`, `mcp`. One command runs the full
  pipeline end-to-end. *(R2.6)*
- **Adaptive-segmentation backends** (`automorphotrack.adaptive_segmentation`):
  `segment_otsu`, `segment_sauvola`, `segment_niblack`, `segment_local_otsu`,
  `segment_subtracted` (rolling-ball / top-hat). *(R2.4)*
- **Benchmarking module** (`automorphotrack.benchmarking`) — exporters for
  CellProfiler, MiNA, MitoGraph schemas, an importer for CellProfiler /
  CellPose / StarDist label outputs, and a `comparison_table()` helper that
  reports Spearman ρ + percent-difference of means between AMT and any other
  tool on shared morphology metrics. *(R1.2, R2.3)*
- **Real `sensitivity_analysis`** in `automorphotrack.validation`: previously a
  random-number placeholder; now runs the actual segmentation pipeline at each
  parameter value and reports Dice / IoU / precision / recall / F1 plus a
  publication-ready sensitivity plot. *(R2.2)*
- **Synthetic ground-truth generator** (`generate_synthetic_ground_truth`)
  for unit-testable validation when no expert annotation is available. *(R2.2)*
- **Statistics annotation helper** (`utils.annotate_stats`) — prints N, test
  name, p-value, and effect size on any comparison axis with one call. Default
  test is auto-selected (Welch vs Mann-Whitney) from Shapiro-Wilk normality. *(R2.5.iv)*
- **Persistent-motion diagonal annotation** on the motility scatter plot,
  with explanatory docstring covering why the line is geometric, not an
  algorithmic artefact. *(R3.4)*
- **`min_detectable_displacement`** parameter in `analyze_motility` to
  suppress sub-pixel jitter being reported as movement. *(R3.4)*
- **Colocalization metric-definitions CSV** written alongside the data
  CSV; every metric column name now records whether it is intensity- or
  mask-based and its valid range. *(R2.5.iii)*
- **CI workflow** (`.github/workflows/python-ci.yml`) — Ubuntu + macOS,
  Python 3.9-3.12, runs `pip install`, verifies CLI entry-points, runs
  `pytest`, and builds sdist/wheel on every push. *(R3.1 regression
  guard.)*
- **Unit tests** under `tests/`: install/imports smoke tests,
  validation-metric correctness, every segmentation backend, every
  benchmarking exporter.

### Changed

- `colocalization.py` — CSV column names now explicitly distinguish
  intensity-based from mask-based metrics; legend labels match. *(R2.5.iii)*
- `summary.py` — Integrated correlation now uses Spearman rank correlation
  by default. *(R3.5)*
- `motility.py` — KDE plots clipped at zero so the displacement distribution
  can no longer show physically invalid negative values. *(R2.5.ii)*
- All red/green palettes replaced with the Okabe-Ito colorblind-safe blue
  (`#0173B2`) and orange (`#DE8F05`). *(R3.3)*
- README: comparison-table now references CellProfiler, MiNA, MitoGraph
  explicitly; AI-assistance section rewritten to describe the new MCP layer
  as the implementation, not aspiration.
- `pyproject.toml` + `setup.py` — version bumped to 2.2.0; `[mcp]`
  extra registered; `console_scripts` entry points added.

### Fixed

- **Install failure on a clean Python environment** caused by an undeclared
  `scikit-learn` import in `validation.py`. The import was unused (we
  compute confusion-matrix metrics directly), so it has been removed; no
  declared dependency change is needed and `pip install automorphotrack` now
  succeeds on Python 3.9-3.12. *(R3.1)*
- `circularity = 4πA/P²` values exceeding 1 due to discrete-pixel perimeter
  underestimation — now uses `perimeter_crofton` and clamps to `[0, 1]`,
  with a docstring note explaining the polygonal-discretization bias. *(R2.5.i)*
- NumPy 2.0 incompatibility (`ndarray.ptp()` removed) replaced with
  `np.ptp(arr)`.

### Notes for reviewers

The point-by-point response letter is in `docs/Response_to_Reviewers.docx`
(and in plain markdown at `docs/reviewer_concerns.md`). Each fix above
includes the reviewer ID it addresses.

---

## [2.1.0] — Pre-revision baseline

Modular re-architecture of the original V3.1 Jupyter notebook into discrete
Python modules (`detection`, `morphology`, `shape_features`, `tracking`,
`motility`, `colocalization`, `summary`, plus optional `temporal_dynamics`,
`spatial_statistics`, `network_analysis`). This is the version that was
peer-reviewed at eLife.
