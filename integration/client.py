"""Provider-independent launcher for a fresh, isolated guarded analysis process."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_guarded(input_path, output_path, config_path, wheel_path, *, python_executable=None,
                receipt_path=None, adapter_path=None, preprocessing_manifest=None,
                expected_input_sha256=None, expected_config_sha256=None,
                expected_wrapper_sha256=None, expected_adapter_sha256=None,
                allow_unvalidated_environment=False):
    """Run the pinned core; numerical outputs retain their own validation limits."""
    output = Path(output_path).resolve()
    receipt = Path(receipt_path).resolve() if receipt_path else output.with_name(output.name + ".integration.json")
    command = [str(python_executable or sys.executable), "-I", "-B", str(ROOT / "run_guarded.py"),
               "--input", str(Path(input_path).resolve()), "--output", str(output),
               "--config", str(Path(config_path).resolve()), "--wheel", str(Path(wheel_path).resolve()),
               "--receipt", str(receipt)]
    optional = {"adapter": adapter_path, "preprocessing-manifest": preprocessing_manifest,
                "expected-input-sha256": expected_input_sha256,
                "expected-config-sha256": expected_config_sha256,
                "expected-wrapper-sha256": expected_wrapper_sha256,
                "expected-adapter-sha256": expected_adapter_sha256}
    for flag, value in optional.items():
        if value is not None:
            command.extend(["--" + flag, str(value)])
    if allow_unvalidated_environment:
        command.append("--allow-unvalidated-environment")
    completed = subprocess.run(command, text=True, capture_output=True)
    record = json.loads(receipt.read_text()) if receipt.exists() else None
    if completed.returncode:
        raise RuntimeError(json.dumps({"command_exit_code": completed.returncode, "receipt": str(receipt),
                                       "error": record.get("error") if record else completed.stderr}))
    if record is None or record["status"] != "completed":
        raise RuntimeError("The guarded process did not produce a completed receipt")
    return record
