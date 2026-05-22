"""Smoke test for the napari plugin module.

We can't open a display in CI, but we can verify that:
  - the module imports without napari installed (graceful degradation)
  - the manifest YAML is included in the package data
  - the CLI exposes the `gui` sub-command
"""

import importlib
from pathlib import Path


def test_napari_plugin_imports_without_napari():
    mod = importlib.import_module("automorphotrack.napari_plugin")
    assert hasattr(mod, "_NAPARI_AVAILABLE")
    assert hasattr(mod, "launch_gui")
    assert hasattr(mod, "make_detect_widget")


def test_napari_yaml_ships_with_package():
    import automorphotrack
    pkg_dir = Path(automorphotrack.__file__).parent
    manifest = pkg_dir / "napari.yaml"
    assert manifest.exists(), f"napari.yaml not packaged at {manifest}"
    text = manifest.read_text()
    assert "AutoMorphoTrack" in text
    # Every widget command must be declared
    for cmd in ["detect", "shape", "motility", "colocalization",
                "validation", "full_pipeline"]:
        assert f"automorphotrack.{cmd}" in text, f"Missing command {cmd} in manifest"


def test_cli_has_gui_subcommand():
    from automorphotrack.cli import main
    # --help on the gui subcommand exits zero
    try:
        main(["gui", "--help"])
    except SystemExit as e:
        assert e.code == 0


def test_launch_gui_raises_clear_error_without_napari():
    from automorphotrack.napari_plugin import launch_gui
    # Without napari installed in the test env, calling launch_gui should
    # raise ImportError with a helpful message — not a generic ModuleNotFoundError
    try:
        import napari  # noqa: F401
    except ImportError:
        try:
            launch_gui()
        except ImportError as e:
            assert "napari" in str(e)
            assert "automorphotrack[napari]" in str(e)
        else:
            raise AssertionError("Expected ImportError when napari is missing")
