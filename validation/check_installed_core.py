"""Check installed module bytes and the fixed known-motion fixture."""
from pathlib import Path
from importlib import metadata
import hashlib
import json
import tempfile
import numpy as np
import pandas as pd
import tifffile

HERE = Path(__file__).resolve().parent
pins = json.loads((HERE / "core_manifest.json").read_text())
distribution = metadata.distribution("automorphotrack")
assert distribution.version == pins["package_version"]
root = Path(distribution.locate_file("automorphotrack/validated")).resolve()
files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")}
assert files == pins["files"]
from automorphotrack.validated import PipelineConfig, run_pipeline
import automorphotrack.validated as core
assert Path(core.__file__).resolve() == root / "__init__.py"
with tempfile.TemporaryDirectory() as temp:
    result = run_pipeline(HERE / "known_motion.ome.tif", temp,
                          PipelineConfig.from_dict(json.loads((HERE / "known_motion_config.json").read_text())))
    assert result["manifest"]["source_sha256"] == pins["source_sha256"]
    differences = []
    expected = HERE / "known_motion_expected"
    for path in sorted(expected.iterdir()):
        actual = Path(temp) / path.name
        if path.suffix == ".tif":
            a, b = tifffile.imread(path), tifffile.imread(actual)
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a,b)
        else:
            a, b = pd.read_csv(path), pd.read_csv(actual)
            assert a.columns.tolist() == b.columns.tolist()
            assert a.shape == b.shape
            for column in a:
                if pd.api.types.is_numeric_dtype(a[column]):
                    np.testing.assert_allclose(a[column], b[column], atol=1e-12, rtol=0, equal_nan=True)
                else:
                    assert a[column].fillna("").tolist() == b[column].fillna("").tolist()
        differences.append(path.name)
    assert len(differences) == 19
    for prefix in ["Mito", "Lyso"]:
        motion = pd.read_csv(Path(temp) / f"{prefix}_MotilitySummary.csv")
        assert len(motion) == 1
        np.testing.assert_allclose(motion.Mean_Speed, 1, atol=1e-12, rtol=0)
        assert motion.Speed_Unit.tolist() == ["px/frame"]
print(json.dumps({"status":"passed","distribution_version":distribution.version,
                  "core_version":core.__version__,"source_sha256":pins["source_sha256"],
                  "module_count":len(files),"output_files_compared":len(differences),
                  "expected_speed_px_per_frame":1,"numeric_atol":1e-12,"numeric_rtol":0},indent=2))
