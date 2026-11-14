"""Validate an installed distribution without importing a source checkout."""
from pathlib import Path
from importlib import import_module, metadata
import argparse
import hashlib
import json
import platform
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", default="automorphotrack.validated")
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import tifffile
    args.output.mkdir(parents=True, exist_ok=True)
    pins = json.loads((args.validation_dir / "core_manifest.json").read_text())
    dist = metadata.distribution("automorphotrack")
    core = import_module(args.namespace)
    root = Path(dist.locate_file(args.namespace.replace(".", "/"))).resolve()
    assert Path(core.__file__).resolve() == root / "__init__.py"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.py")}
    assert hashes == pins["files"], "Installed quantitative core differs from the frozen source"
    config = core.PipelineConfig.from_dict(json.loads((args.validation_dir / "known_motion_config.json").read_text()))
    result = core.run_pipeline(args.validation_dir / "known_motion.ome.tif", args.output / "known_motion", config)
    assert result["manifest"]["source_sha256"] == pins["source_sha256"]
    compared = []
    for expected in sorted((args.validation_dir / "known_motion_expected").iterdir()):
        actual = args.output / "known_motion" / expected.name
        if expected.suffix == ".tif":
            a, b = tifffile.imread(expected), tifffile.imread(actual)
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
        else:
            a, b = pd.read_csv(expected), pd.read_csv(actual)
            assert a.columns.tolist() == b.columns.tolist()
            assert a.shape == b.shape
            for column in a:
                if pd.api.types.is_numeric_dtype(a[column]):
                    np.testing.assert_allclose(a[column], b[column], atol=1e-12, rtol=0, equal_nan=True)
                else:
                    assert a[column].fillna("").tolist() == b[column].fillna("").tolist()
        compared.append(expected.name)
    assert len(compared) == 19
    for name in ["Mito", "Lyso"]:
        table = pd.read_csv(args.output / "known_motion" / f"{name}_MotilitySummary.csv")
        assert len(table) == 1
        np.testing.assert_allclose(table.Mean_Speed, 1, atol=1e-12, rtol=0)
    cli = "amt-validated" if args.namespace.endswith("validated") else "automorphotrack"
    entry = Path(sys.executable).parent / (cli + (".exe" if platform.system() == "Windows" else ""))
    if not entry.exists() and platform.system() == "Windows":
        entry = Path(sys.executable).parent / "Scripts" / (cli + ".exe")
    help_result = subprocess.run([str(entry), "--help"], capture_output=True, text=True)
    if help_result.returncode:
        raise RuntimeError(help_result.stderr)
    (args.output / "cli_help.txt").write_text(help_result.stdout)
    report = dict(status="passed", distribution_version=dist.version, core_version=core.__version__,
                  core_sha256=pins["source_sha256"], installed_module_count=len(hashes),
                  installed_source_path=str(root), output_files_compared=compared,
                  expected_speed_px_per_frame=1, absolute_tolerance=1e-12, relative_tolerance=0,
                  system=platform.system(), machine=platform.machine(), platform=platform.platform(),
                  python=sys.version, executable=sys.executable,
                  dependencies={name: metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-image", "tifffile"]})
    (args.output / "installed_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if dist.version == "2.5.0":
        refined_check(args.validation_dir, args.output, dist)


def refined_check(validation, output, dist):
    import numpy as np
    import pandas as pd
    import tifffile
    from automorphotrack.refined import RefinedConfig, run_pipeline
    pins=json.loads((validation/'refined_manifest.json').read_text())
    root=Path(dist.locate_file('automorphotrack/refined')).resolve()
    actual={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
    assert actual==pins['source_sha256']
    config=RefinedConfig.from_dict(json.loads((validation/'refined_motion_config.json').read_text()))
    run_pipeline(validation/'refined_motion.ome.tif',output/'refined_motion',config)
    compared=[]
    for reference in sorted((validation/'refined_motion_expected').iterdir()):
        if reference.name=='run_manifest.json':continue
        observed=output/'refined_motion'/reference.name
        if reference.suffix=='.tif':
            np.testing.assert_array_equal(tifffile.imread(reference),tifffile.imread(observed))
        else:
            a,b=pd.read_csv(reference),pd.read_csv(observed)
            assert a.columns.tolist()==b.columns.tolist() and a.shape==b.shape
            for column in a:
                if pd.api.types.is_numeric_dtype(a[column]):
                    np.testing.assert_allclose(a[column],b[column],atol=1e-9,rtol=0,equal_nan=True)
                else:assert a[column].fillna('').tolist()==b[column].fillna('').tolist()
        compared.append(reference.name)
    report=dict(status='passed',distribution_version=dist.version,source_digest=pins['source_digest'],
        module_count=len(actual),files_compared=compared,numeric_atol=1e-9,numeric_rtol=0,
        tolerance_basis='Numerical reproducibility tolerance for floating-point filtered positions and measurements; exact integer masks required.')
    (output/'refined_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
