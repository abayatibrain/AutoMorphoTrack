"""Recorded single- or multichannel particle analysis, separate from the frozen core."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
from importlib import metadata
import numpy as np
import pandas as pd
import tifffile
from .segmentation import SpotConfig, segment_spots
from .tracking import TrackingConfig, link_spots
from ..validated.io import normalize_axes, sha256_file
from ..validated.config import PipelineConfig as MeasurementConfig
from ..validated.core import measure_shapes, motility_from_tracks, colocalization_metrics


@dataclass(frozen=True)
class RefinedConfig:
    axes: str | None = None
    axes_override_reason: str | None = None
    channels: tuple[int, ...] = (0,)
    spot: SpotConfig = field(default_factory=SpotConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    min_observations: int = 3
    pixel_size_um: float | None = None
    frame_interval_s: float | None = None
    calibration_source: str | None = None
    sample_id: str = "sample"

    def __post_init__(self):
        if not isinstance(self.spot, SpotConfig) or not isinstance(self.tracking, TrackingConfig):
            raise TypeError("Expected SpotConfig and TrackingConfig")
        if (not self.channels or len(set(self.channels)) != len(self.channels)
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.channels)):
            raise ValueError("channels must contain distinct nonnegative integers")
        if isinstance(self.min_observations, bool) or not isinstance(self.min_observations, int) or self.min_observations < 1:
            raise ValueError("min_observations must be a positive integer")
        if not self.sample_id:
            raise ValueError("sample_id must be nonempty")
        MeasurementConfig(pixel_size_um=self.pixel_size_um, frame_interval_s=self.frame_interval_s)
        if (self.pixel_size_um is not None or self.frame_interval_s is not None) and not self.calibration_source:
            raise ValueError("Physical calibration requires calibration_source")

    @classmethod
    def from_dict(cls, values):
        values=dict(values)
        if 'spot' in values: values['spot']=SpotConfig(**values['spot'])
        if 'tracking' in values: values['tracking']=TrackingConfig(**values['tracking'])
        if 'channels' in values: values['channels']=tuple(values['channels'])
        return cls(**values)


def analyze_movie(images, config=None):
    """Analyze a nonnegative TYX array; label IDs remain tied to recorded detections."""
    config=config or RefinedConfig()
    images=np.asarray(images)
    if images.ndim != 3 or not all(images.shape):
        raise ValueError("Expected a nonempty TYX array")
    if len(images) < config.min_observations:
        raise ValueError("Movie is shorter than min_observations; set that value explicitly for brief or static inputs")
    raw_labels=[];points=[];thresholds=[]
    for frame, im in enumerate(images):
        lab,xy,meta=segment_spots(im,config.spot)
        raw_labels.append(lab);points.append(xy);thresholds.append(dict(Frame=frame,**meta))
    raw_labels=np.asarray(raw_labels)
    raw_tracks=link_spots(points,config.tracking)
    counts=raw_tracks.groupby('Track_ID').size()
    keep=counts[counts>=config.min_observations].index
    tracks=raw_tracks[raw_tracks.Track_ID.isin(keep)].copy()
    labels=np.zeros_like(raw_labels)
    for frame in range(len(images)):
        allowed=tracks.loc[tracks.Frame.eq(frame),'Label'].to_numpy()
        labels[frame]=np.where(np.isin(raw_labels[frame],allowed),raw_labels[frame],0)
    measurement=MeasurementConfig(pixel_size_um=config.pixel_size_um,frame_interval_s=config.frame_interval_s)
    shapes,_=measure_shapes(labels,measurement,labeled=True)
    shapes=shapes.rename(columns={'X':'Mask_Centroid_X','Y':'Mask_Centroid_Y'})
    shapes=shapes.merge(tracks[['Frame','Label','X','Y','Track_ID']],on=['Frame','Label'],how='left',validate='one_to_one')
    # The frozen measurement routine independently computes actual frame intervals.
    steps,motility,per_frame=motility_from_tracks(tracks,measurement)
    return dict(labels=labels,raw_labels=raw_labels,tracks=tracks,raw_tracks=raw_tracks,
                shapes=shapes,steps=steps,motility=motility,motility_per_frame=per_frame,
                thresholds=pd.DataFrame(thresholds))


def run_pipeline(input_path, out_dir, config=None):
    """Write all inputs, settings, source identities, and quantitative outputs.

    For exactly two selected channels, retained foreground masks also define the
    full-field colocalization measurements. This does not establish contact.
    """
    config=config or RefinedConfig()
    if not isinstance(config,RefinedConfig): raise TypeError("config must be RefinedConfig")
    out=Path(out_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Output directory is not empty: {out}")
    with tifffile.TiffFile(input_path) as tf:
        if len(tf.series)!=1: raise ValueError("Extract one TIFF series before analysis")
        series=tf.series[0];native_axes=series.axes;array=series.asarray()
    axes=config.axes or native_axes
    if axes!=native_axes and not config.axes_override_reason:
        raise ValueError("An axes override requires axes_override_reason")
    stack=normalize_axes(array,axes)
    if max(config.channels)>=stack.shape[-1]: raise ValueError("Selected channel is outside the image")
    channels={channel:analyze_movie(stack[...,channel],config) for channel in config.channels}
    out.mkdir(parents=True,exist_ok=True)
    for channel,result in channels.items():
        for name,value in result.items():
            path=out/f'channel_{channel}_{name}'
            if isinstance(value,pd.DataFrame): value.to_csv(path.with_suffix('.csv'),index=False)
            else: tifffile.imwrite(path.with_suffix('.tif'),value,compression='deflate',metadata={'axes':'TYX'},photometric='minisblack')
    if len(config.channels)==2:
        a,b=config.channels
        values=[dict(Frame=f,**colocalization_metrics(stack[f,...,a],stack[f,...,b],channels[a]['labels'][f]>0,channels[b]['labels'][f]>0)) for f in range(len(stack))]
        pd.DataFrame(values).to_csv(out/'colocalization.csv',index=False)
    source={p.name:sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    frozen={p.name:sha256_file(p) for p in sorted((Path(__file__).parent.parent/'validated').glob('*.py'))}
    manifest=dict(schema_version=1,algorithm_version='2.5.0',namespace='automorphotrack.refined',
        created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=sha256_file(input_path),
        input_name=Path(input_path).name,native_axes=native_axes,effective_axes=axes,
        canonical_shape=list(stack.shape),configuration=asdict(config),
        source_files_sha256=source,source_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
        frozen_measurement_sha256=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest(),
        python=platform.python_version(),platform=platform.platform(),
        dependencies={name:metadata.version(name) for name in ['numpy','scipy','pandas','scikit-image','tifffile']},
        observations={str(k):dict(raw=len(v['raw_tracks']),retained=len(v['tracks'])) for k,v in channels.items()},
        limitations=['Intensity-support masks are not validated organelle membrane boundaries.',
                     'Minimum track length can exclude real brief events.',
                     'Ambiguous and gap links are excluded from adjacent-step speeds.',
                     'No inferred fission, fusion, or membrane contact.'],
        output_files={p.name:sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return dict(channels=channels,manifest=manifest)
