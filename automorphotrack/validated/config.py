"""Explicit analysis parameters. Acquisition calibration is never guessed."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import math


@dataclass(frozen=True)
class SegmentationConfig:
    threshold_factor: float = 0.8
    min_size_px: int = 3
    opening_radius_px: int = 1
    clear_border: bool = True
    method: str = "otsu"
    local_block_size_px: int = 31
    local_offset: float = 0.0
    background_radius_px: int = 0

    def __post_init__(self):
        if not math.isfinite(self.threshold_factor) or self.threshold_factor <= 0:
            raise ValueError("threshold_factor must be positive and finite")
        for name in ("min_size_px", "opening_radius_px", "background_radius_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.method not in {"otsu", "local", "external"}:
            raise ValueError("method must be otsu, local, or external")
        if self.local_block_size_px < 3 or self.local_block_size_px % 2 != 1:
            raise ValueError("local_block_size_px must be an odd integer >= 3")
        if not math.isfinite(self.local_offset):
            raise ValueError("local_offset must be finite")


@dataclass(frozen=True)
class PipelineConfig:
    axes: Optional[str] = None
    mito_channel: int = 0
    lyso_channel: int = 1
    pixel_size_um: Optional[float] = None
    frame_interval_s: Optional[float] = None
    calibration_source: Optional[str] = None
    axes_override_reason: Optional[str] = None
    sample_id: str = "sample"
    biological_replicate_id: Optional[str] = None
    mito_segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    lyso_segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    max_link_distance_px: float = 15.0
    exclude_ambiguous_links: bool = True
    elongated_eccentricity: float = 0.8
    punctate_eccentricity: float = 0.6
    min_elongated_area_um2: Optional[float] = None
    min_punctate_area_um2: Optional[float] = None
    background_percentile: Optional[float] = None
    render: bool = False
    audit_drift: bool = False

    def __post_init__(self):
        for name in ("pixel_size_um", "frame_interval_s", "max_link_distance_px"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if self.mito_channel == self.lyso_channel:
            raise ValueError("The two organelle channels must differ")
        for channel in (self.mito_channel, self.lyso_channel):
            if isinstance(channel, bool) or not isinstance(channel, int) or channel < 0:
                raise ValueError("Channel indices must be nonnegative integers")
        if not 0 <= self.punctate_eccentricity < self.elongated_eccentricity <= 1:
            raise ValueError("Eccentricity thresholds must satisfy 0 <= punctate < elongated <= 1")
        for name in ("min_elongated_area_um2", "min_punctate_area_um2"):
            value = getattr(self, name)
            if value is not None:
                if self.pixel_size_um is None or not math.isfinite(value) or value < 0:
                    raise ValueError("Physical area gates require pixel_size_um and nonnegative areas")
        if self.background_percentile is not None and not 0 <= self.background_percentile <= 100:
            raise ValueError("background_percentile must be between 0 and 100")
        if not self.sample_id:
            raise ValueError("sample_id must be nonempty")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for name in ("mito_segmentation", "lyso_segmentation"):
            if name in values and isinstance(values[name], dict):
                values[name] = SegmentationConfig(**values[name])
        return cls(**values)
