# AutoMorphoTrack 2.5.0

AutoMorphoTrack 2.5.0 requires Python 3.11 or later and has two explicitly versioned analysis routes. The automorphotrack.validated namespace conserves the 18 modules of core 1.1.0rc2. The automorphotrack.refined namespace includes five additional modules that provide particle detection and association, described below, and make use of the unchanged baseline measurement modules. NumPy, pandas, SciPy, scikit-image, and tifffile are the main dependencies. The two implementations are distinguishable by their source, and examples show which route produced a given result.

## Installation

```sh
python -m pip install "automorphotrack==2.5.0"
```

Minimal Debian and Ubuntu systems may need libxcb1, libgl1, and libglib2.0-0 for the OpenCV import. The GitHub CI matrix completed wheel and source installations on Windows, Linux, and macOS with Python 3.11 and 3.12, and retained test reports, dependency versions, source checks, and known-motion outputs.

## Particle pipeline

```python
from automorphotrack.refined import RefinedConfig, run_pipeline
config = RefinedConfig(channels=(0, 1), sample_id="recording")
result = run_pipeline("recording.ome.tif", "new_results", config)
```

```sh
amt-refined recording.ome.tif --out new_results --config configuration.json
```

The particle example uses a two-channel synthetic movie with known motion. Run the guarded adapter from the release directory after installing the 2.5.0 wheel. It checks the 18 baseline and five particle modules before importing the numerical implementation. Use a new output directory for every run. For a static image, set min_observations to 1. Particle support masks have not been validated to be mitochondrial membrane boundaries. The default setting requires three observations, thus short-lived events will not be retained and retained masks and motion summaries will not reflect these. Raw outputs preserve these events.

The input must identify time, channel, and spatial axes. Calibration requires a recorded source; otherwise motion remains in pixels per frame. Label masks and track IDs are linked through each frame's Label column. A different configuration requires its own evaluation.

## Recorded validation

Six reserved movies contain 145 annotated trajectories and 2,764 adjacent reference links. Mean accepted-link recall for the frozen default and fixed top-hat configurations was 0.014 and 0.212, respectively. With the particle pipeline, mean accepted-link recall was 0.696, and mean localization recall was 0.768. The new pipeline recovered 2,024 accepted reference links, and four additional accepted links related to different annotated identities, giving conditional precision of 2,024/2,028 among the links with two reference-matched endpoints. Remaining predictions were left unverified. Thirteen trajectories were completely missed, and 33 were fully recovered with accepted links, compared to 41 missed and two fully recovered with the top-hat configuration. Accepted-link recall was 0.739 with a 5-pixel tolerance and 0.587 with a 2-pixel tolerance. Individual reserved movies yielded accepted-link recall from 0.408 to 0.911. Supplementary Fig. 10 and Table 9 show the paired comparisons and their shortcomings.

The original 50 synthetic sequences, when rerun, yielded mean Dice scores of 0.877, 0.752 and 0.532, for the new particle masks, the fixed top-hat configuration and the frozen default, respectively. Mean instance precision and accepted-link recall increased from 0.407 to 0.966 and 0.178 to 0.965, respectively. The two baselines reproduced their earlier results to within an absolute error of 10^-12. Transfer to the external mitochondrial masks was limited. Without the length filter, the selected detector yielded mean Dice scores of 0.369 and 0.060 for the seven MoDL SIM patches and the 13 Myofibril patches, respectively. The MoDL score was still below the earlier top-hat result of 0.421. The particle pipeline thus does not replace target-specific segmentation validation for elongated or densely packed mitochondria.

AutoMorphoTrack 2.5.0 passed 72 tests in each of 12 GitHub CI installations. It built wheels and source distributions for Windows, Linux, and macOS, with Python 3.11 and 3.12. All installations kept the 18 baseline and five particle source modules. Each test compared 19 baseline and 19 particle CSV or TIFFs with recorded references. Integer masks were exact, and for baseline and particle outputs, numerical tolerances were 10^-12 and 10^-9, respectively. The test failed initially on Windows due to Git line-ending conversion which changed the source hashes. Enforcing LF line endings solved this, and the numerical code remained the same. The OpenCV import failure and its system-library remedy documented for minimal Linux remain documented.

Supplementary figures and tables are in the manuscript files attached to [release v2.5.0](https://github.com/abayatibrain/AutoMorphoTrack/releases/tag/v2.5.0). The [reproduction archive](https://github.com/abayatibrain/AutoMorphoTrack/releases/download/v2.5.0/AMT_Revision_Reproducibility_v2.5.0.zip) includes the development protocol, every candidate result, reserved-movie outputs, controls, and prior analyses. [The 12-job CI run](https://github.com/abayatibrain/AutoMorphoTrack/actions/runs/43602083124) records Windows, Linux, and macOS installations.

Biological source: [Richens et al., PLOS Biology 2025](https://doi.org/10.1371/journal.pbio.3003099), [data version 1](https://doi.org/10.17863/CAM.114338), CC BY 4.0. All 12 movies had been assessed with the baseline before six were reserved from the new parameter selection. This is a within-study split. Manual annotations cover a subset of vesicles; unmatched predictions remain unverified.

## Reproducible integration

For the 2.5.0 route, Gemini wrote a new adapter and Codex ran the captured source against the installed wheel. The first adapter returned a file-hash mapping where the contract required a digest string. A focused change resolved that metadata error. The corrected adapter, the Codex adapter, a standalone command line, and an executed Jupyter notebook reproduced the 19 reference particle files exactly on the local host. Altered manifests and source files were rejected before analysis. The Codex guard verifies all 23 modules across both numerical namespaces. Prior to the requests, Argon High was selected in Gemini, but the interface changed its model label after the responses; the underlying model identity was not verified. These records test software integration, and numerical reproducibility, without constituting independent biological validation by either assistant.

The runnable adapter is `examples/refined_verified.py`. The notebook is `examples/Refined_Pipeline.ipynb`. Keep both source manifests under version control. Run a fixture after changing an adapter, configuration, preprocessing step, environment, or downstream calculation. Source integrity alone does not validate those changes.

```sh
python -I examples/refined_verified.py --input validation/refined_motion.ome.tif --out new_fixture_results --config validation/refined_motion_config.json --validation-dir validation
```

## Baseline route

Install the pinned 2.5.0 wheel in a virtual environment. Use amt-refined, or import RefinedConfig and run_pipeline from automorphotrack.refined, for the particle pipeline. The amt-validated command retains the baseline implementation for Figures 1-17 and Supplementary Figures 1-9. Supplementary Figure 10 reports the new particle evaluation. The automorphotrack and amt commands remain unchanged. The compressed-tiff extra supplies a TIFF decoder.

```sh
amt-validated recording.ome.tif --out baseline_results --config baseline_configuration.json
```

| Implementation | Modules | Source digest |
| --- | ---: | --- |
| automorphotrack.validated, core 1.1.0rc2 | 18 | 8a4125a5ebe2b49e019f79b783503f31403a24ea3134d2d10b1bd4610ac3bce8 |
| automorphotrack.refined, package 2.5.0 | 5 | dcacbe6eede1cfef2cfa165418f8e0c9428aafebf3632a27a6a7655e3605f3a6 |

## License and citation

Software: MIT. External datasets retain their individual licenses, recorded in the reproduction archive. See `CITATION.cff` for this release.

Bayati A, Schumacher JG, Chen X. AutoMorphoTrack. [eLife reviewed preprint 109936, version 1](https://doi.org/10.7554/eLife.109936.1). This is the older preprint, preceding the v2.5.0 particle revision. The revised manuscript is deposited with the current GitHub release.
