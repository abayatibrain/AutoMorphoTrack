"""Run the pinned particle pipeline after checking both numerical namespaces."""
from pathlib import Path
from importlib import metadata
import argparse
import hashlib
import json


def verify(validation_dir):
    """Check all 23 source modules before importing the quantitative package."""
    dist = metadata.distribution("automorphotrack")
    if dist.version != "2.5.0":
        raise RuntimeError("This example requires automorphotrack==2.5.0")
    roots = {}
    for namespace, filename, files_key, digest_key in [
        ("validated", "core_manifest.json", "files", "source_sha256"),
        ("refined", "refined_manifest.json", "source_sha256", "source_digest"),
    ]:
        pins = json.loads((Path(validation_dir) / filename).read_text())
        root = Path(dist.locate_file("automorphotrack/" + namespace)).resolve()
        observed = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.glob("*.py"))}
        expected = pins[files_key]
        digest = hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()
        if observed != expected or digest != pins[digest_key]:
            raise RuntimeError("Frozen source mismatch: " + namespace)
        roots[namespace] = (root, digest)
    return roots


def run_verified(input_path, output_path, config_path, validation_dir):
    roots = verify(validation_dir)
    import automorphotrack.refined as refined
    import automorphotrack.validated as validated
    for name, module in [("refined", refined), ("validated", validated)]:
        if Path(module.__file__).resolve().parent != roots[name][0]:
            raise RuntimeError("Import path differs from verified source: " + name)
    config = refined.RefinedConfig.from_dict(json.loads(Path(config_path).read_text()))
    result = refined.run_pipeline(input_path, output_path, config)
    if result["manifest"]["source_sha256"] != roots["refined"][1]:
        raise RuntimeError("Returned source digest differs from the verified pipeline")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-dir", required=True)
    args = parser.parse_args()
    run_verified(args.input, args.out, args.config, args.validation_dir)
