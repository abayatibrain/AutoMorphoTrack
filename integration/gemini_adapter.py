#!/usr/bin/env python3
"""Standalone stdlib-only guarded core adapter for AutoMorphoTrack integration tests.

This module invokes client.run_guarded from a supplied integration directory,
preserves all integrity checks, leaves input/config/wheel paths and core outputs
unchanged, verifies the fixed synthetic test invariants, and writes a separate
sibling JSON report recording this adapter file identity without claiming
accuracy on new datasets.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


EXPECTED_FRAME_COUNT = 4
EXPECTED_COUNT_PER_CHANNEL_PER_FRAME = 1
EXPECTED_MEAN_SPEED = 1.0
EXPECTED_SPEED_UNIT = "px/frame"
SPEED_ABS_TOLERANCE = 1e-6


def compute_file_sha256(path: Path) -> str:
    """Compute the hexadecimal SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_within_directory(candidate: Path, directory: Path) -> bool:
    """Return True if candidate resolves inside directory or equals directory."""
    resolved_candidate = str(candidate.resolve())
    resolved_directory = str(directory.resolve())
    try:
        common = os.path.commonpath([resolved_candidate, resolved_directory])
    except ValueError:
        return False
    return common == resolved_directory


def load_run_guarded(integration_root_arg: str) -> Callable[..., Dict[str, Any]]:
    """Import only client.run_guarded from the supplied integration directory."""
    integration_root = Path(integration_root_arg).resolve()
    if not integration_root.is_dir():
        raise RuntimeError(
            f"Supplied --integration-root is not an existing directory: {integration_root}"
        )

    client_py = integration_root / "client.py"
    client_pkg = integration_root / "client" / "__init__.py"
    if not client_py.is_file() and not client_pkg.is_file():
        raise RuntimeError(
            f"Could not find client module inside --integration-root: {integration_root}"
        )

    root_str = str(integration_root)
    if sys.path[:1] != [root_str]:
        sys.path.insert(0, root_str)

    from client import run_guarded

    if not callable(run_guarded):
        raise RuntimeError("Imported client.run_guarded is not callable.")

    code_obj = getattr(run_guarded, "__code__", None)
    if code_obj is not None:
        origin_path = Path(code_obj.co_filename).resolve()
        if not is_within_directory(origin_path, integration_root):
            raise RuntimeError(
                f"client.run_guarded resolved outside --integration-root: {origin_path}"
            )

    return run_guarded


def prepare_fresh_output_paths(
    input_arg: str,
    output_arg: Optional[str],
    report_arg: Optional[str],
) -> Tuple[Path, Path, Path]:
    """Validate or allocate a fresh output directory and sibling artifact paths."""
    input_resolved = Path(input_arg).resolve()

    if output_arg is None:
        temp_parent = Path(tempfile.mkdtemp(prefix="amt_gemini_adapter_"))
        output_dir = temp_parent / "output"
    else:
        output_dir = Path(output_arg)
        if output_dir.resolve() == input_resolved:
            raise RuntimeError("Output path must be distinct from input path.")
        if output_dir.exists():
            if not output_dir.is_dir():
                raise RuntimeError(
                    f"Output path exists and is not a directory: {output_dir}"
                )
            if any(output_dir.iterdir()):
                raise RuntimeError(
                    f"Output directory is not fresh because it is non-empty: {output_dir}"
                )
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    default_receipt_path = output_dir.parent / f"{output_dir.name}.integration.json"
    if default_receipt_path.exists():
        raise RuntimeError(
            f"Sibling receipt already exists for output location: {default_receipt_path}"
        )

    if report_arg is None:
        report_path = output_dir.parent / f"{output_dir.name}.gemini_report.json"
    else:
        report_path = Path(report_arg)

    if is_within_directory(report_path, output_dir):
        raise RuntimeError(
            "Separate JSON report path must not be placed inside the core output directory."
        )
    if report_path.resolve() == default_receipt_path.resolve():
        raise RuntimeError(
            "Separate JSON report path must not overwrite the integration receipt file."
        )
    if report_path.exists():
        raise RuntimeError(
            f"Separate JSON report path already exists: {report_path}"
        )

    return output_dir, default_receipt_path, report_path


def locate_core_csv(output_dir: Path, filename: str) -> Path:
    """Locate a single required CSV file under output_dir without modifying core outputs."""
    direct_path = output_dir / filename
    matches = sorted(path for path in output_dir.rglob(filename) if path.is_file())
    if not matches:
        raise RuntimeError(f"Required core CSV {filename} was not found under {output_dir}")
    if len(matches) > 1:
        if direct_path in matches and len(matches) == 1:
            return direct_path
        raise RuntimeError(
            f"Expected one {filename} under {output_dir}, found {len(matches)}: {matches}"
        )
    return matches[0]


def parse_exact_nonnegative_int(raw_value: Optional[str], field_name: str, row_index: int) -> int:
    """Parse an integer count field while rejecting missing, fractional, negative, or NaN values."""
    if raw_value is None or not raw_value.strip():
        raise RuntimeError(f"Missing {field_name} at row {row_index} in Frame_Summary.csv")
    numeric = float(raw_value.strip())
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise RuntimeError(
            f"Invalid integer count for {field_name} at row {row_index}: {raw_value!r}"
        )
    return int(numeric)


def read_frame_summary(csv_path: Path) -> Tuple[int, List[int], List[int]]:
    """Read per-frame organelle detection counts from Frame_Summary.csv."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Frame_Summary.csv has no header row: {csv_path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        reader.fieldnames = fieldnames
        for required_col in ("Mito_Count", "Lyso_Count"):
            if required_col not in fieldnames:
                raise RuntimeError(
                    f"Frame_Summary.csv is missing required column {required_col}: {csv_path}"
                )

        mito_counts: List[int] = []
        lyso_counts: List[int] = []
        for idx, row in enumerate(reader):
            if not any((val or "").strip() for val in row.values()):
                continue
            mito_counts.append(parse_exact_nonnegative_int(row.get("Mito_Count"), "Mito_Count", idx))
            lyso_counts.append(parse_exact_nonnegative_int(row.get("Lyso_Count"), "Lyso_Count", idx))

    return len(mito_counts), mito_counts, lyso_counts


def read_motility_summary(csv_path: Path, channel_label: str) -> Tuple[float, str]:
    """Read finite Mean_Speed and Speed_Unit values from a channel motility summary CSV."""
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"{channel_label} motility CSV has no header row: {csv_path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        reader.fieldnames = fieldnames
        for required_col in ("Mean_Speed", "Speed_Unit"):
            if required_col not in fieldnames:
                raise RuntimeError(
                    f"{channel_label} motility CSV is missing column {required_col}: {csv_path}"
                )

        speeds: List[float] = []
        units: List[str] = []
        for idx, row in enumerate(reader):
            if not any((val or "").strip() for val in row.values()):
                continue
            raw_speed = row.get("Mean_Speed")
            raw_unit = row.get("Speed_Unit")
            if raw_speed is None or not raw_speed.strip():
                raise RuntimeError(
                    f"{channel_label} motility CSV has empty Mean_Speed at row {idx}: {csv_path}"
                )
            if raw_unit is None or not raw_unit.strip():
                raise RuntimeError(
                    f"{channel_label} motility CSV has empty Speed_Unit at row {idx}: {csv_path}"
                )
            speed_val = float(raw_speed.strip())
            if not math.isfinite(speed_val):
                raise RuntimeError(
                    f"{channel_label} motility CSV has non-finite Mean_Speed at row {idx}: {raw_speed!r}"
                )
            speeds.append(speed_val)
            units.append(raw_unit.strip())

    if not speeds:
        raise RuntimeError(f"{channel_label} motility CSV contains zero data rows: {csv_path}")

    distinct_units = sorted(set(units))
    if len(distinct_units) != 1:
        raise RuntimeError(
            f"{channel_label} motility CSV contains inconsistent Speed_Unit values: {distinct_units}"
        )

    mean_speed = sum(speeds) / len(speeds)
    if not math.isfinite(mean_speed):
        raise RuntimeError(f"{channel_label} computed mean speed is non-finite.")

    return mean_speed, distinct_units[0]


def validate_receipt(
    receipt: Any,
    default_receipt_path: Path,
    adapter_sha256: str,
) -> Tuple[Path, str]:
    """Verify the completed receipt dictionary and sibling receipt file on disk."""
    if not isinstance(receipt, dict):
        raise RuntimeError("client.run_guarded did not return a receipt dictionary.")

    validation_scope = receipt.get("validation_scope")
    if not isinstance(validation_scope, str) or not validation_scope.strip():
        raise RuntimeError("Completed receipt is missing a non-empty validation_scope.")

    reported_path = receipt.get("receipt_path")
    if isinstance(reported_path, str) and reported_path.strip():
        receipt_path = Path(reported_path)
    else:
        receipt_path = default_receipt_path

    if not receipt_path.is_file():
        raise RuntimeError(f"Expected receipt file was not found on disk: {receipt_path}")

    with receipt_path.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)

    if not isinstance(persisted, dict):
        raise RuntimeError(f"Persisted receipt JSON is not an object: {receipt_path}")
    if persisted.get("validation_scope") != validation_scope:
        raise RuntimeError(
            "Mismatch between returned receipt validation_scope and persisted receipt file."
        )

    recorded_adapter_hash = receipt.get("adapter_sha256")
    if isinstance(recorded_adapter_hash, str) and recorded_adapter_hash.strip():
        if recorded_adapter_hash.lower() != adapter_sha256.lower():
            raise RuntimeError(
                "Receipt adapter_sha256 does not match this adapter source file identity."
            )

    nested_adapter = receipt.get("adapter")
    if isinstance(nested_adapter, dict):
        nested_hash = nested_adapter.get("sha256")
        if isinstance(nested_hash, str) and nested_hash.strip():
            if nested_hash.lower() != adapter_sha256.lower():
                raise RuntimeError(
                    "Receipt nested adapter sha256 does not match this adapter source file identity."
                )

    return receipt_path, validation_scope


def evaluate_synthetic_checks(
    frame_count: int,
    mito_counts: Sequence[int],
    lyso_counts: Sequence[int],
    mito_speed: float,
    mito_unit: str,
    lyso_speed: float,
    lyso_unit: str,
) -> List[str]:
    """Check fixed synthetic test expectations and return a list of any failure messages."""
    failures: List[str] = []

    if frame_count != EXPECTED_FRAME_COUNT:
        failures.append(
            f"Expected {EXPECTED_FRAME_COUNT} frames, observed {frame_count}."
        )

    if any(count != EXPECTED_COUNT_PER_CHANNEL_PER_FRAME for count in mito_counts):
        failures.append(
            f"Expected Mito_Count == {EXPECTED_COUNT_PER_CHANNEL_PER_FRAME} in every frame, "
            f"observed {list(mito_counts)}."
        )

    if any(count != EXPECTED_COUNT_PER_CHANNEL_PER_FRAME for count in lyso_counts):
        failures.append(
            f"Expected Lyso_Count == {EXPECTED_COUNT_PER_CHANNEL_PER_FRAME} in every frame, "
            f"observed {list(lyso_counts)}."
        )

    if not math.isclose(mito_speed, EXPECTED_MEAN_SPEED, rel_tol=0.0, abs_tol=SPEED_ABS_TOLERANCE):
        failures.append(
            f"Expected Mito Mean_Speed == {EXPECTED_MEAN_SPEED}, observed {mito_speed}."
        )

    if mito_unit != EXPECTED_SPEED_UNIT:
        failures.append(
            f"Expected Mito Speed_Unit == {EXPECTED_SPEED_UNIT!r}, observed {mito_unit!r}."
        )

    if not math.isclose(lyso_speed, EXPECTED_MEAN_SPEED, rel_tol=0.0, abs_tol=SPEED_ABS_TOLERANCE):
        failures.append(
            f"Expected Lyso Mean_Speed == {EXPECTED_MEAN_SPEED}, observed {lyso_speed}."
        )

    if lyso_unit != EXPECTED_SPEED_UNIT:
        failures.append(
            f"Expected Lyso Speed_Unit == {EXPECTED_SPEED_UNIT!r}, observed {lyso_unit!r}."
        )

    return failures


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run AutoMorphoTrack through client.run_guarded on a fresh output directory, "
            "verify fixed synthetic test outputs, and write a separate sibling JSON report."
        )
    )
    parser.add_argument(
        "--integration-root",
        required=True,
        help="Path to helper integration directory containing client.run_guarded.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input path passed unchanged to client.run_guarded.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Configuration path passed unchanged to client.run_guarded.",
    )
    parser.add_argument(
        "--wheel",
        required=True,
        help="Pinned wheel path passed unchanged to client.run_guarded.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable path passed to client.run_guarded.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Fresh output directory path for core analysis outputs.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional sibling path for the separate adapter JSON report.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute guarded core analysis, validate synthetic invariants, and write report."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        adapter_file = Path(__file__).resolve()
        adapter_sha256 = compute_file_sha256(adapter_file)

        output_dir, default_receipt_path, report_path = prepare_fresh_output_paths(
            input_arg=args.input,
            output_arg=args.output,
            report_arg=args.report,
        )

        run_guarded = load_run_guarded(args.integration_root)

        receipt = run_guarded(
            args.input,
            str(output_dir),
            args.config,
            args.wheel,
            python_executable=args.python,
            adapter_path=__file__,
        )

        receipt_path, validation_scope = validate_receipt(
            receipt=receipt,
            default_receipt_path=default_receipt_path,
            adapter_sha256=adapter_sha256,
        )

        frame_csv = locate_core_csv(output_dir, "Frame_Summary.csv")
        mito_csv = locate_core_csv(output_dir, "Mito_MotilitySummary.csv")
        lyso_csv = locate_core_csv(output_dir, "Lyso_MotilitySummary.csv")

        frame_count, mito_counts, lyso_counts = read_frame_summary(frame_csv)
        mito_speed, mito_unit = read_motility_summary(mito_csv, "Mito")
        lyso_speed, lyso_unit = read_motility_summary(lyso_csv, "Lyso")

        mito_summed_detections = sum(mito_counts)
        lyso_summed_detections = sum(lyso_counts)

        failures = evaluate_synthetic_checks(
            frame_count=frame_count,
            mito_counts=mito_counts,
            lyso_counts=lyso_counts,
            mito_speed=mito_speed,
            mito_unit=mito_unit,
            lyso_speed=lyso_speed,
            lyso_unit=lyso_unit,
        )

        report: Dict[str, Any] = {
            "adapter_path": str(adapter_file),
            "adapter_sha256": adapter_sha256,
            "output_path": str(output_dir.resolve()),
            "receipt_path": str(receipt_path.resolve()),
            "validation_scope": validation_scope,
            "frame_count": frame_count,
            "summed_repeated_detections": {
                "Mito": mito_summed_detections,
                "Lyso": lyso_summed_detections,
            },
            "per_frame_counts": {
                "Mito": mito_counts,
                "Lyso": lyso_counts,
            },
            "motility": {
                "Mito": {
                    "mean_speed": mito_speed,
                    "speed_unit": mito_unit,
                    "is_finite": True,
                },
                "Lyso": {
                    "mean_speed": lyso_speed,
                    "speed_unit": lyso_unit,
                    "is_finite": True,
                },
            },
            "synthetic_checks_passed": len(failures) == 0,
            "synthetic_check_failures": failures,
            "scope_statement": (
                "This report verifies guarded pinned-core execution and fixed synthetic "
                "test invariants without calibration or preprocessing. It does not claim "
                "accuracy on new datasets."
            ),
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_report = json.dumps(report, indent=2, allow_nan=False) + "\n"
        report_path.write_text(serialized_report, encoding="utf-8")

        if failures:
            for message in failures:
                print(f"SYNTHETIC CHECK FAILED: {message}", file=sys.stderr)
            return 1

        stdout_payload = dict(report)
        stdout_payload["report_path"] = str(report_path.resolve())
        print(json.dumps(stdout_payload, indent=2, allow_nan=False))
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
