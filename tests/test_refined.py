"""Scientific behavior and integration checks for the versioned particle pipeline."""
from dataclasses import replace
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import tifffile
from automorphotrack.refined import (SpotConfig,TrackingConfig,RefinedConfig,
    segment_spots,link_spots,analyze_movie,run_pipeline)
from automorphotrack.validated.core import motility_from_tracks
from automorphotrack.validated import PipelineConfig


def movie(frames=5):
    yy,xx=np.indices((64,64))
    return np.asarray([100+300*np.exp(-((yy-20)**2+(xx-20-f)**2)/4.5)
                       +200*np.exp(-((yy-44)**2+(xx-44)**2)/4.5)
                       for f in range(frames)],np.float32)


def test_intensity_offset_and_scale_preserve_spots():
    image=movie()[0]
    a,pa,_=segment_spots(image)
    b,pb,_=segment_spots(image*3+500)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_allclose(pa,pb,atol=1e-5,rtol=0)
    assert len(pa)==2


def test_close_spots_get_distinct_bounded_instance_masks():
    yy,xx=np.indices((64,64))
    image=sum(200*np.exp(-((yy-30)**2+(xx-x)**2)/3) for x in [26,33])+100
    labels,points,_=segment_spots(image)
    assert len(points)==2
    assert set(np.unique(labels))=={0,1,2}
    assert np.linalg.norm(points[0]-points[1])>6


def test_flat_movie_has_no_objects_or_motion():
    result=analyze_movie(np.full((5,32,32),100.))
    assert not result['labels'].any()
    assert result['tracks'].empty and result['motility'].empty


@pytest.mark.parametrize('image',[np.zeros((2,3,4)),np.full((10,10),np.nan),np.full((10,10),-1)])
def test_bad_images_rejected(image):
    with pytest.raises(ValueError): segment_spots(image)


@pytest.mark.parametrize('kwargs',[{'threshold_snr':0},{'sigma_px':float('nan')},
    {'min_distance_px':True},{'support_fraction':1},{'background_sigma_px':1}])
def test_bad_detector_config_rejected(kwargs):
    with pytest.raises(ValueError): SpotConfig(**kwargs)


def test_short_inputs_require_explicit_length_setting():
    with pytest.raises(ValueError,match='min_observations'): analyze_movie(movie(1))
    result=analyze_movie(movie(1),RefinedConfig(min_observations=1))
    assert len(result['tracks'])==2 and result['steps'].Speed.isna().all()


def test_short_noise_track_removed_from_every_retained_mask():
    im=movie();im[2:]=100
    result=analyze_movie(im)
    assert len(result['raw_tracks'])==4
    assert result['tracks'].empty and not result['labels'].any()


def test_track_order_identity_and_speed():
    points=[np.array([[10,10+f],[30,30-f]])[::-1] if f%2 else np.array([[10,10+f],[30,30-f]]) for f in range(5)]
    tracks=link_spots(points)
    assert not tracks.duplicated(['Track_ID','Frame']).any()
    assert tracks.Track_ID.nunique()==2
    _,motion,_=motility_from_tracks(tracks,PipelineConfig())
    np.testing.assert_allclose(motion.Mean_Speed,1,rtol=0,atol=1e-12)


def test_tied_merge_remains_ambiguous_and_excluded():
    tracks=link_spots([np.array([[10,10],[10,12]]),np.array([[10,11]])])
    assert tracks.iloc[-1].Link_Status=='ambiguous'
    _,motion,_=motility_from_tracks(tracks,PipelineConfig())
    assert motion.Accepted_Steps.sum()==0


def test_nearby_alternative_does_not_make_clear_link_ambiguous():
    tracks=link_spots([np.array([[10,10],[10,15]]),np.array([[10,10.2],[10,15.2]])])
    assert tracks.loc[tracks.Frame.eq(1),'Link_Status'].tolist()==['matched','matched']
    assert tracks.loc[tracks.Frame.eq(1),'Previous_Candidate_Count'].min()==2


def test_short_gap_retained_but_not_used_as_an_observed_speed():
    points=[np.array([[10,10]]),np.empty((0,2)),np.array([[10,12]])]
    default=link_spots(points)
    assert default.Track_ID.nunique()==2
    optional=link_spots(points,TrackingConfig(max_gap_frames=1))
    assert optional.Track_ID.nunique()==1 and optional.iloc[-1].Link_Status=='gap'
    _,motion,_=motility_from_tracks(optional,PipelineConfig())
    assert motion.Accepted_Steps.sum()==0 and motion.Mean_Speed.isna().all()


def test_motion_prediction_is_opt_in():
    points=[np.array([[10,5],[10,15]]),np.array([[10,8],[10,12]]),np.array([[10,11],[10,9]])]
    predicted=link_spots(points,TrackingConfig(prediction_weight=1))
    assert predicted[(predicted.Track_ID==0)&(predicted.Frame==2)].X.item()==11
    assert TrackingConfig().prediction_weight==0


def test_mask_label_and_track_positions_remain_joinable():
    result=analyze_movie(movie())
    for frame in range(5):
        mask_ids=set(np.unique(result['labels'][frame]))-{0}
        track_ids=set(result['tracks'].loc[result['tracks'].Frame.eq(frame),'Label'])
        assert mask_ids==track_ids
    assert len(result['shapes'])==len(result['tracks'])==10
    assert result['shapes'].Track_ID.notna().all()


def test_axes_calibration_and_output_guard(tmp_path):
    source=tmp_path/'movie.tif'
    tifffile.imwrite(source,movie(),metadata={'axes':'ZYX'},photometric='minisblack')
    with pytest.raises(ValueError):run_pipeline(source,tmp_path/'fail',RefinedConfig(axes='TYX'))
    config=RefinedConfig(axes='TYX',axes_override_reason='Synthetic test explicitly encodes time as Z')
    result=run_pipeline(source,tmp_path/'out',config)
    assert result['manifest']['effective_axes']=='TYX'
    assert result['manifest']['frozen_measurement_sha256']=='8a4125a5ebe2b49e019f79b783503f31403a24ea3134d2d10b1bd4610ac3bce8'
    with pytest.raises(FileExistsError):run_pipeline(source,tmp_path/'out',config)
    with pytest.raises(ValueError,match='calibration_source'):RefinedConfig(frame_interval_s=5)


def test_cli_and_api_produce_identical_tables(tmp_path):
    source=tmp_path/'movie.tif';config=tmp_path/'config.json'
    images=np.stack([movie(),movie()],axis=-1)
    tifffile.imwrite(source,images,metadata={'axes':'TYXC'},photometric='minisblack')
    config.write_text(json.dumps({'channels':[0,1]}))
    api=run_pipeline(source,tmp_path/'api',RefinedConfig(channels=(0,1)))
    subprocess.run([sys.executable,'-m','automorphotrack.refined',str(source),'--out',str(tmp_path/'cli'),'--config',str(config)],check=True,capture_output=True,text=True)
    for file in (tmp_path/'api').glob('*.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(file),pd.read_csv(tmp_path/'cli'/file.name),check_exact=True)
    assert (tmp_path/'api/colocalization.csv').is_file()
