"""Metadata-aware TIFF input and reproducible run identities."""

from pathlib import Path
import hashlib
import numpy as np
import tifffile


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_axes(array, axes):
    """Return TYXC without reinterpreting unknown dimensions or projecting Z."""
    axes = axes.upper().replace("S", "C")
    if len(axes) != array.ndim or len(set(axes)) != len(axes):
        raise ValueError("Axes must uniquely describe every input dimension")
    if any(axis not in "TZYXC" for axis in axes) or "Y" not in axes or "X" not in axes:
        raise ValueError(f"Ambiguous or unsupported axes {axes!r}; supply explicit axes")
    if "Z" in axes:
        zi = axes.index("Z")
        if array.shape[zi] != 1:
            raise ValueError("Non-singleton Z is unsupported. Do not reinterpret Z as time without acquisition evidence")
        array = np.take(array, 0, axis=zi)
        axes = axes.replace("Z", "")
    for axis in "TC":
        if axis not in axes:
            array = np.expand_dims(array, -1)
            axes += axis
    array = np.transpose(array, [axes.index(axis) for axis in "TYXC"])
    if any(size == 0 for size in array.shape):
        raise ValueError("Empty image dimensions are unsupported")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Image intensities must be finite numeric values")
    if np.any(array < 0):
        raise ValueError("Negative intensities require explicit preprocessing")
    return array


def load_stack(path, config):
    with tifffile.TiffFile(path) as handle:
        if len(handle.series) != 1:
            raise ValueError("Multiple TIFF series require explicit extraction before analysis")
        series = handle.series[0]
        array = series.asarray()
        native_axes = series.axes
        ij = handle.imagej_metadata or {}
        metadata = {
            "native_axes": native_axes,
            "native_shape": list(array.shape),
            "dtype": str(array.dtype),
            "imagej_unit": ij.get("unit"),
            "imagej_finterval": ij.get("finterval"),
        }
    axes = config.axes or native_axes
    if config.axes and axes != native_axes and not config.axes_override_reason:
        raise ValueError("An axes override requires axes_override_reason in the run configuration")
    stack = normalize_axes(array, axes)
    if max(config.mito_channel, config.lyso_channel) >= stack.shape[-1]:
        raise ValueError("Channel index is outside the input channel dimension")
    metadata.update(effective_axes=axes, canonical_axes="TYXC", canonical_shape=list(stack.shape))
    return stack, metadata
