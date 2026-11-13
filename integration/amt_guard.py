"""Consistency checks around the frozen RC2 wheel; this is not a security sandbox."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.abc
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import sysconfig
import zipfile

ROOT = Path(__file__).resolve().parent
PIN_RECORD_SHA256 = "5232f2efe72da8f1006795c5560a4e39eb6b5e7b51ee1eab55d1c1048b3d70ae"
WHEEL_SHA256 = "732cf7c0b503c328a19625c6ade95bebcddc7a1f5b60f5b1e512e52d551358ea"
CORE_DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-image", "tifffile")
WRAPPER_FILES = ("amt_guard.py", "client.py", "run_guarded.py")


class IntegrityError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def wrapper_identity():
    files = {name: sha256_file(ROOT / name) for name in WRAPPER_FILES}
    return {"files": files, "sha256": canonical_sha256(files)}


def environment_snapshot():
    dependencies = {}
    for name in CORE_DEPENDENCIES:
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    packages = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return {"python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_full": sys.version, "executable": sys.executable,
            "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "system": platform.system(), "machine": platform.machine(),
            "platform": platform.platform(), "dependencies": dependencies,
            "installed_distributions": dict(sorted(packages.items()))}


def environment_differences(observed, expected):
    differences = {}
    for key in ("python_version", "python_implementation", "system", "machine", "platform"):
        if observed[key] != expected[key]:
            differences[key] = {"expected": expected[key], "observed": observed[key]}
    for name, value in expected["dependencies"].items():
        if observed["dependencies"].get(name) != value:
            differences["dependency:" + name] = {"expected": value, "observed": observed["dependencies"].get(name)}
    return differences


def load_pins():
    path = ROOT / "pins/rc2.json"
    actual = sha256_file(path)
    if actual != PIN_RECORD_SHA256:
        raise IntegrityError("pin_record_changed", "The recorded RC2 pin manifest has changed.", {"expected": PIN_RECORD_SHA256, "actual": actual})
    pins = json.loads(path.read_text())
    if pins["wheel_sha256"] != WHEEL_SHA256:
        raise IntegrityError("wheel_pin_changed", "The RC2 wheel pin is inconsistent.")
    source_aggregate = hashlib.sha256(json.dumps(pins["source_files"], sort_keys=True).encode()).hexdigest()
    if source_aggregate != pins["source_sha256"]:
        raise IntegrityError("source_manifest_inconsistent", "The source manifest aggregate does not match its files.")
    return pins


def verify_package_files(package_root, expected):
    package_root = Path(package_root).resolve()
    found = {str(path.relative_to(package_root)) for path in package_root.rglob("*.py")}
    if found != set(expected):
        raise IntegrityError("package_file_inventory_changed", "Installed Python module inventory differs from RC2.",
                             {"missing": sorted(set(expected) - found), "extra": sorted(found - set(expected))})
    extensions = [str(path.relative_to(package_root)) for path in package_root.rglob("*")
                  if path.is_file() and path.suffix in (".so", ".pyd", ".dll")]
    if extensions:
        raise IntegrityError("unexpected_package_extension", "The RC2 package contains unexpected compiled code.", {"files": extensions})
    actual = {}
    for relative, wanted in expected.items():
        path = package_root / relative
        if not path.resolve().is_relative_to(package_root):
            raise IntegrityError("package_symlink_outside_root", "A package module resolves outside its installed package.")
        actual[relative] = sha256_file(path)
        if actual[relative] != wanted:
            raise IntegrityError("installed_module_sha256_mismatch", "An installed AMT module differs from RC2. Use a new version and revalidate changed quantitative code.",
                                 {"module": relative, "expected": wanted, "actual": actual[relative]})
    return actual


def verify_core(wheel_path, *, allow_unvalidated_environment=False, expected_wrapper_sha256=None):
    pins = load_pins()
    wheel_path = Path(wheel_path).resolve()
    actual_wheel = sha256_file(wheel_path)
    if actual_wheel != WHEEL_SHA256:
        raise IntegrityError("wheel_sha256_mismatch", "The supplied wheel is not the pinned RC2 artifact.",
                             {"expected": WHEEL_SHA256, "actual": actual_wheel})
    with zipfile.ZipFile(wheel_path) as archive:
        archive_python = {name.removeprefix("automorphotrack/validated/"): hashlib.sha256(archive.read(name)).hexdigest()
                          for name in archive.namelist() if name.startswith("automorphotrack/validated/") and name.endswith(".py")}
    if archive_python != pins["source_files"]:
        raise IntegrityError("wheel_source_manifest_mismatch", "Wheel module bytes do not match the frozen source manifest.")
    loaded = sorted(name for name in sys.modules if name == "automorphotrack" or name.startswith("automorphotrack."))
    if loaded:
        raise IntegrityError("amt_already_imported", "Use the isolated runner before importing AutoMorphoTrack; existing live modules are not accepted.", {"modules": loaded})
    try:
        distribution = metadata.distribution(pins["distribution"])
    except metadata.PackageNotFoundError as error:
        raise IntegrityError("wheel_not_installed", "Install the pinned wheel into this interpreter first.") from error
    if distribution.version != pins["version"]:
        raise IntegrityError("installed_version_mismatch", "The installed distribution version differs from RC2.", {"actual": distribution.version})
    direct_url = distribution.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        raise IntegrityError("editable_install_rejected", "An editable source checkout is not the pinned wheel installation.")
    package_root = Path(distribution.locate_file("automorphotrack/validated")).resolve()
    site_roots = {Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")}
    if not any(package_root.is_relative_to(path) for path in site_roots):
        raise IntegrityError("installed_location_mismatch", "AMT is not located in this interpreter's installed package directories.", {"package_root": str(package_root), "allowed_roots": sorted(map(str, site_roots))})
    spec = importlib.util.find_spec("automorphotrack")
    expected_origin = package_root.parent / "__init__.py"
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected_origin:
        raise IntegrityError("source_shadow_rejected", "Python would import an unexpected AutoMorphoTrack location.",
                             {"expected": str(expected_origin), "observed": None if spec is None else spec.origin})
    actual_files = verify_package_files(package_root, pins["source_files"])
    wrapper = wrapper_identity()
    if expected_wrapper_sha256 and wrapper["sha256"] != expected_wrapper_sha256:
        raise IntegrityError("wrapper_sha256_mismatch", "The integration wrapper differs from the requested reproduction fingerprint.",
                             {"expected": expected_wrapper_sha256, "actual": wrapper["sha256"]})
    observed = environment_snapshot()
    differences = environment_differences(observed, pins["observed_environment_profile"])
    if differences and not allow_unvalidated_environment:
        raise IntegrityError("environment_differs", "This environment differs from the recorded wheel-smoke profile. Explicit opt-in records an unvalidated environment; it does not waive core integrity checks.", {"differences": differences})
    return {"pins": pins, "package_root": package_root, "source_files": actual_files,
            "wheel_path": str(wheel_path), "wheel_sha256": actual_wheel, "wrapper": wrapper,
            "environment": observed, "environment_differences": differences,
            "environment_status": "unvalidated_environment_opt_in" if differences else "matches_recorded_profile",
            "allow_unvalidated_environment": bool(allow_unvalidated_environment),
            "distribution_direct_url": json.loads(direct_url) if direct_url else None}


class VerifiedSourceLoader(importlib.abc.Loader):
    """Compile the checked source bytes directly, without reading bytecode caches."""
    def __init__(self, fullname, path, expected_sha256):
        self.fullname, self.path, self.expected_sha256 = fullname, Path(path), expected_sha256

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        body = self.path.read_bytes()
        actual = hashlib.sha256(body).hexdigest()
        if actual != self.expected_sha256:
            raise IntegrityError("module_changed_before_import", "An AMT module changed after the integrity check.", {"module": self.fullname})
        exec(compile(body, str(self.path), "exec", dont_inherit=True), module.__dict__)


class VerifiedPackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, files):
        self.root, self.files = Path(root), files

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "automorphotrack" and not fullname.startswith("automorphotrack."):
            return None
        relative = "__init__.py" if fullname == "automorphotrack" else fullname.removeprefix("automorphotrack.").replace(".", "/") + ".py"
        if relative not in self.files:
            raise IntegrityError("unlisted_amt_module", "An unlisted module was requested inside the pinned package.", {"module": fullname})
        source = self.root / relative
        loader = VerifiedSourceLoader(fullname, source, self.files[relative])
        return importlib.util.spec_from_file_location(fullname, source, loader=loader,
                                                      submodule_search_locations=[str(self.root)] if fullname == "automorphotrack" else None)


@contextmanager
def import_verified_core(context):
    finder = VerifiedPackageFinder(context["package_root"], context["source_files"])
    sys.meta_path.insert(0, finder)
    try:
        package = importlib.import_module("automorphotrack")
        if Path(package.__file__).resolve() != context["package_root"] / "__init__.py":
            raise IntegrityError("loaded_path_mismatch", "The loaded AMT package origin is inconsistent.")
        yield package
        verify_package_files(context["package_root"], context["pins"]["source_files"])
    finally:
        sys.meta_path.remove(finder)
