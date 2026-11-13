# AutoMorphoTrack

AutoMorphoTrack 2.4.0 includes the frozen quantitative core described in the revised eLife manuscript. It can be launched through the automorphotrack.validated namespace or via the amt-validated command. The current 2.x APIs, GUI, and MCP interface are also available and are not covered in this manuscript.

## Installation

```sh
python -m pip install "automorphotrack==2.4.0"
```

Python >=3.11

## Manuscript analysis

```python
from automorphotrack.validated import PipelineConfig, run_pipeline
config = PipelineConfig(axes="TCYX", mito_channel=0, lyso_channel=1)
result = run_pipeline("recording.tif", "results", config)
```

```sh
amt-validated recording.tif --out results --config configuration.json
```

Provide true axes and channel mappings. Physical calibration is not required; uncalibrated motion is in pixels per frame. The pipeline retains masks, objects, accepted links, tables, configuration, input identity, and source hashes. Adapters should invoke the validated namespace and leave the source unaltered. An unaltered core does not guarantee correctness on a new image domain.

## Validation scope

The revision assesses segmentation on a set of 7 public microscopy studies, of which 6 are confocal. It describes the study-specific limitations and provides failed cases. Representative example image crops are regions from a single recording. The benchmark does not prove membrane contact, or functional interaction, nor does it prove identity tracking in biological data in general.

The public release candidate passed 51 software tests on the local macOS arm64 host. An installed-wheel check verified 18 modules of the core and compared 19 CSV and TIFF files to the known-motion fixture. Both channels contained a tracked object with an expected mean speed of 1 pixel per frame. The separate command-line route was also successfully completed. Numerical and execution consistency for the known-motion fixture were evaluated; these tests do not prove correctness for new biological images.

## Release identities

| Item | Identifier |
| --- | --- |
| Public distribution | 2.4.0 |
| Manuscript namespace | automorphotrack.validated |
| Frozen core version | 1.1.0rc2 |
| Core digest | 8a4125a5ebe2b49e019f79b783503f31403a24ea3134d2d10b1bd4610ac3bce8 |
| Source manifest | validation/core_manifest.json |
| Earlier interfaces | docs/V2_INTERFACE.md |
| Release records | docs/VALIDATED_CORE.md |

## License and citation

MIT license

Bayati A, Schumacher JG, Chen X. AutoMorphoTrack. eLife reviewed preprint 109936. https://doi.org/10.7554/eLife.109936.1
