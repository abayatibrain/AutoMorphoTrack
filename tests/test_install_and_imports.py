"""Smoke tests — guards against the install failure Reviewer #3 hit."""

import importlib


def test_top_level_import():
    a = importlib.import_module("automorphotrack")
    assert a.__version__
    assert isinstance(a.__all__, list)
    # Critical exports the README and manuscript reference must exist
    for name in [
        "detect_organelles",
        "analyze_shape_features",
        "analyze_motility",
        "analyze_colocalization",
        "summarize_integrated_data",
        "validate_segmentation",
        "sensitivity_analysis",
        "segment", "segment_otsu", "segment_sauvola", "segment_subtracted",
        "annotate_stats",
        "benchmarking",
    ]:
        assert hasattr(a, name), f"Missing public export: {name}"


def test_no_sklearn_dep():
    """Reviewer #3 install failure was caused by an undeclared sklearn import."""
    import automorphotrack.validation as v
    import sys
    # sklearn must NOT be required to import validation
    assert "sklearn" not in v.__dict__


def test_cli_entry_point_callable():
    from automorphotrack.cli import main
    # --help exits zero — wrap in try/except since argparse SystemExit
    import sys
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
