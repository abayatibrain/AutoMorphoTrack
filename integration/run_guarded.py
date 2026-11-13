"""Fresh-process CLI around the unchanged, installed AutoMorphoTrack RC2 core."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

# Explicitly load the adjacent integration helper when Python uses isolated mode.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from amt_guard import (IntegrityError, canonical_sha256, environment_snapshot,
                       import_verified_core, sha256_file, verify_core, wrapper_identity)


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_preprocessing_record(path, input_path):
    if path is None:
        return {"wrapper_operations": [], "input_path": str(input_path),
                "earlier_input_history": "not inferred; caller-supplied input bytes are fingerprinted"}
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if not isinstance(record.get("operations"), list) or not isinstance(record.get("source_inputs"), list):
        raise IntegrityError("preprocessing_record_schema", "A preprocessing record requires operations and source_inputs lists.")
    sources = []
    for item in record["source_inputs"]:
        source = Path(item["path"])
        source = source.resolve() if source.is_absolute() else (path.parent / source).resolve()
        actual = sha256_file(source)
        if actual != item["sha256"]:
            raise IntegrityError("preprocessing_source_changed", "A declared preprocessing source does not match its checksum.", {"path": str(source)})
        sources.append({"path": str(source), "sha256": actual})
    return {"manifest_path": str(path), "manifest_sha256": sha256_file(path),
            "declared_record": record, "checked_source_inputs": sources,
            "scope": "Caller-declared lineage; the wrapper does not independently verify the operations or their scientific validity.",
            "wrapper_operations": []}


def execute(args):
    input_path, output_path = Path(args.input).resolve(), Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    receipt_path = Path(args.receipt).resolve() if args.receipt else output_path.with_name(output_path.name + ".integration.json")
    if receipt_path.exists():
        raise FileExistsError(f"Receipt already exists: {receipt_path}")
    receipt = {"schema_version": "1.0", "started_utc": utc(), "status": "started",
               "analysis_started": False, "validation_scope": "core_integrity_only",
               "new_data_accuracy_established": False, "input_path": str(input_path),
               "output_path": str(output_path), "config_path": str(config_path),
               "environment": environment_snapshot(),
               "scope_note": "An unchanged core does not validate new data, settings, preprocessing, wrapper logic, or downstream interpretation."}
    try:
        context = verify_core(args.wheel, allow_unvalidated_environment=args.allow_unvalidated_environment,
                              expected_wrapper_sha256=args.expected_wrapper_sha256)
        receipt.update(wheel_path=context["wheel_path"], wheel_sha256=context["wheel_sha256"],
                       installed_package_root=str(context["package_root"]),
                       installed_source_files=context["source_files"],
                       installed_source_sha256=context["pins"]["source_sha256"],
                       pin_record_sha256=sha256_file(ROOT / "pins/rc2.json"),
                       wrapper_identity=context["wrapper"],
                       environment_status=context["environment_status"],
                       environment_differences=context["environment_differences"],
                       allow_unvalidated_environment=context["allow_unvalidated_environment"])
        input_sha = sha256_file(input_path)
        if args.expected_input_sha256 and input_sha != args.expected_input_sha256:
            raise IntegrityError("input_sha256_mismatch", "Input bytes differ from the requested reproduction fingerprint.", {"expected": args.expected_input_sha256, "actual": input_sha})
        receipt["input_sha256"] = input_sha
        receipt["raw_config_file_sha256"] = sha256_file(config_path)
        raw_config = json.loads(config_path.read_text())
        receipt["preprocessing"] = load_preprocessing_record(args.preprocessing_manifest, input_path)
        if args.adapter:
            adapter = Path(args.adapter).resolve()
            adapter_sha = sha256_file(adapter)
            receipt["adapter"] = {"path": str(adapter), "sha256": adapter_sha,
                                  "scope": "Identity supplied by caller; the isolated runner does not execute this adapter."}
            if args.expected_adapter_sha256 and adapter_sha != args.expected_adapter_sha256:
                raise IntegrityError("adapter_sha256_mismatch", "Adapter bytes differ from the requested reproduction fingerprint.")
        elif args.expected_adapter_sha256:
            raise IntegrityError("adapter_path_required", "An expected adapter checksum requires an adapter path.")
        with import_verified_core(context) as amt:
            config = amt.PipelineConfig.from_dict(raw_config)
            normalized = config.to_dict()
            config_sha = canonical_sha256(normalized)
            receipt.update(configuration=normalized, configuration_sha256=config_sha,
                           configuration_hash_convention="UTF-8 JSON, sorted keys, compact separators, finite values")
            if args.expected_config_sha256 and config_sha != args.expected_config_sha256:
                raise IntegrityError("configuration_sha256_mismatch", "Settings differ from the requested reproduction fingerprint.", {"expected": args.expected_config_sha256, "actual": config_sha})
            if output_path.exists() and any(output_path.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {output_path}")
            receipt["analysis_started"] = True
            result = amt.run_pipeline(input_path, output_path, config)
            receipt["core_run_id"] = result["manifest"]["run_id"]
            receipt["core_version"] = result["manifest"]["version"]
            receipt["units"] = result["manifest"]["units"]
        if sha256_file(input_path) != input_sha:
            raise IntegrityError("input_changed_during_run", "Input bytes changed during execution.")
        if sha256_file(config_path) != receipt["raw_config_file_sha256"]:
            raise IntegrityError("config_file_changed_during_run", "The supplied configuration file changed during execution.")
        if wrapper_identity() != context["wrapper"]:
            raise IntegrityError("wrapper_changed_during_run", "Integration helper files changed during execution.")
        after = environment_snapshot()
        if after["dependencies"] != context["environment"]["dependencies"]:
            raise IntegrityError("dependencies_changed_during_run", "Recorded dependency versions changed during execution.")
        receipt["outputs"] = {str(path.relative_to(output_path)): sha256_file(path)
                              for path in sorted(output_path.rglob("*")) if path.is_file()}
        receipt["status"] = "completed"
        return_code = 0
    except Exception as error:
        receipt["status"] = "rejected" if not receipt["analysis_started"] else "failed"
        receipt["error"] = {"type": type(error).__name__, "code": getattr(error, "code", "execution_error"),
                            "message": str(error), "details": getattr(error, "details", {})}
        receipt["traceback"] = traceback.format_exc()
        return_code = 2
    receipt["completed_utc"] = utc()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "analysis_started": receipt["analysis_started"],
                      "receipt": str(receipt_path), "validation_scope": receipt["validation_scope"]}))
    return return_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "config", "wheel"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--receipt")
    parser.add_argument("--adapter")
    parser.add_argument("--preprocessing-manifest")
    parser.add_argument("--allow-unvalidated-environment", action="store_true")
    for name in ("input", "config", "wrapper", "adapter"):
        parser.add_argument("--expected-" + name + "-sha256")
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
