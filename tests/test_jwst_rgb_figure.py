"""Photometry must remain attached to the right image when catalogs change."""
import pytest
import h5py
import numpy as np
import pandas as pd
from kappaccino.plotting.jwst_rgb import marker_photometry
from kappaccino.io import store


def test_marker_matches_survive_reordering_but_reject_moved_images(tmp_path):
    path=tmp_path/'phot.csv'
    pd.DataFrame(dict(image_label=['a','b'],ra=[64.,64.],dec=[-24.,-24.],
        phot_id=[1,2],match_separation_arcsec=[.1,.2],m_F200W_AB=[24.,28.],
        valid_photometry=[True,True])).to_csv(path,index=False)
    catalog=pd.DataFrame(dict(image_label=['b','a','new'],x_obs_arcsec=[0.,0.,0.],y_obs_arcsec=[0.,0.,0.]))
    result=marker_photometry(catalog,(3,64.,-24.),path)
    np.testing.assert_allclose(result.m_F200W_AB[:2],[28,24])
    assert result.marker_area_pt2.iloc[1]>result.marker_area_pt2.iloc[0]
    assert result.marker_area_pt2.iloc[2]==9
    catalog.loc[1,'y_obs_arcsec']=1
    result=marker_photometry(catalog,(3,64.,-24.),path)
    assert not result.valid_photometry.iloc[1]
    assert result.marker_area_pt2.iloc[1]==9


def test_missing_photometry_keeps_all_image_markers(tmp_path):
    catalog=pd.DataFrame(dict(image_label=['a','b'],x_obs_arcsec=[0.,1.],y_obs_arcsec=[0.,1.]))
    result=marker_photometry(catalog,(3,64.,-24.),tmp_path/'missing.csv')
    assert len(result)==2
    assert not result.valid_photometry.any()
    assert (result.marker_area_pt2==9).all()


def test_critical_band_for_translating_line_and_identical_draws():
    from kappaccino.plotting.jwst_rgb import critical_quantile_band
    x=np.linspace(-2,2,401)
    shifts=np.linspace(-1,1,101)
    lo,hi,band=critical_quantile_band(x[None,:]-shifts[:,None])
    np.testing.assert_allclose(lo,x-.68,atol=1e-14)
    np.testing.assert_allclose(hi,x+.68,atol=1e-14)
    assert np.all(np.abs(x[band])<=.68+1e-12)
    assert not critical_quantile_band(np.tile(x,(32,1)))[2].any()


def test_median_determinant_contour_and_band_are_matched():
    from kappaccino.plotting.jwst_rgb import critical_quantile_summary
    import contourpy
    x=np.linspace(-2,2,501);y=np.linspace(-1,1,11)
    xx,yy=np.meshgrid(x,y)
    shifts=np.linspace(-1,1,101)**3+.2
    draws=xx[None]-shifts[:,None,None]+.1*yy[None]
    lo,median,hi=critical_quantile_summary(draws)
    curves=contourpy.contour_generator(x=x,y=y,z=median).lines(0.)
    for curve in curves:
        np.testing.assert_allclose(curve[:,0]+.1*curve[:,1],np.median(shifts),atol=1e-14)
        assert np.all(curve[:,0]+.1*curve[:,1]>=np.quantile(shifts,.16))
        assert np.all(curve[:,0]+.1*curve[:,1]<=np.quantile(shifts,.84))
    assert np.all(lo<=median) and np.all(median<=hi)


def test_critical_band_shading_preserves_subpixel_interval():
    import matplotlib.pyplot as plt
    from kappaccino.plotting.jwst_rgb import draw_critical_interval
    # Old min(-lower,upper) is negative at every node, losing the entire band.
    x=np.array([[-1.,1.],[-1.,1.]])
    low,high=x-.02,x+.02
    assert np.max(np.minimum(-low,high))<0
    fig,ax=plt.subplots()
    fill=draw_critical_interval(ax,low,high,'tab:blue',(-2,2,-1,1))
    assert fill is not None and fill.get_clip_path() is not None
    assert fill.get_clip_box() is not None
    np.testing.assert_allclose(fill.get_clip_box().bounds,ax.bbox.bounds)
    assert fill.get_paths()[0].contains_point((0.,0.))
    clip=fill.get_clip_path().get_fully_transformed_path()
    assert clip.contains_point(ax.transData.transform((0.,0.)))
    assert not clip.contains_point(ax.transData.transform((-.5,0.)))
    assert not fill.get_paths()[0].contains_point((.5,0.))
    plt.close(fig)


def _fake_overview(monkeypatch):
    """Small stand-ins for the imaging, the overview geometry (1 arcsec = 10 pixels about pixel 16) and the RGB
    cutout of the RGB critical overview; returns the reporting-draw stand-in class."""
    from types import SimpleNamespace as NS
    from astropy.wcs import WCS
    from kappaccino.plotting import mass_figure
    from kappaccino.imaging import cutouts
    wcs=WCS(naxis=2)
    wcs.wcs.ctype=['RA---TAN','DEC--TAN'];wcs.wcs.crval=[64.,-24.]
    wcs.wcs.crpix=[16.,16.];wcs.wcs.cdelt=[-.1/3600,.1/3600]
    gx,gy=np.meshgrid(np.linspace(-1.55,1.55,32),np.linspace(-1.55,1.55,32))
    geometry=dict(wcs=wcs,resolution=32,center=None,size=3.2,xlim=(-1.6,1.6),ylim=(-1.6,1.6),x=gx,y=gy)
    monkeypatch.setattr(mass_figure,'overview_geometry',lambda *a,**k:geometry)
    monkeypatch.setattr(mass_figure,'aligned_rgb',lambda image,*a:image)
    monkeypatch.setattr(mass_figure,'grid_pixels',lambda geometry,x,y:(16+np.asarray(x)*10,16+np.asarray(y)*10))
    monkeypatch.setattr(cutouts,'rgb_cutout',lambda *a:np.zeros((32,32,3)))
    monkeypatch.setattr(cutouts,'rgb_reference_band',lambda bands:bands[0])
    class Draws:
        indices=np.arange(3); key='test'; evaluator=NS(gp_knee=False)
        def __len__(self):return 3
        def grid_maps(self,name,x,y,z,log=print,**kwargs):
            assert name=='overview' and z==9.
            x,y=np.meshgrid(np.linspace(-1,1,32),np.linspace(-1,1,32))
            return {'detA':np.array([x*x+y*y-r for r in [.3,.4,.5]])}
    return Draws


def _overview_catalog():
    return pd.DataFrame(dict(image_label=['1.a','2.b'],family_id=['1','2'],x_obs_arcsec=[-.3,.3],y_obs_arcsec=[-.3,.3],
                             x_model_arcsec=[-.31,.29],y_model_arcsec=[-.3,.31],image_recovery_status=['recovered','not_recovered']))


@pytest.mark.parametrize('cluster',['hera','ares','as1063','newcluster','m0416'])
def test_simulated_hst_overview_has_no_m0416_overlays(tmp_path, monkeypatch, cluster):
    from types import SimpleNamespace as NS
    from kappaccino.plotting import jwst_rgb as rgb, mass_figure
    from kappaccino.diagnostics import position_contours as image_position_contours
    Draws=_fake_overview(monkeypatch)
    position_calls=[]
    def positions(*args, **kwargs):
        position_calls.append(True)
        return {'curves': []}
    monkeypatch.setattr(image_position_contours, 'load_or_compute_contours', positions)
    monkeypatch.setattr(image_position_contours, 'load_position_contours', lambda tables: {'curves': []})
    catalog=_overview_catalog()
    extra=pd.DataFrame(dict(family_id=['1'],x_model_arcsec=[.1],y_model_arcsec=[-.2]))
    config=NS(paths=NS(run_name=cluster),image_catalog=NS(candidate_catalog_path='must-not-read.json'))
    if cluster == 'm0416':
        import json
        candidate_path = tmp_path/'candidates.json'
        candidate_path.write_text(json.dumps({'candidates':[
            dict(id='K66.c1',system='K66',kind='counterimage',ra=64.,dec=-24.),
            dict(id='A.1',system='A',kind='new_family',ra=64.,dec=-24.)]}))
        config.image_catalog.candidate_catalog_path = candidate_path
        catalog['family_id'] = ['K66','K67']
    state=NS(reference=(3,64.,-24.))
    output=tmp_path/f'{cluster}_hst_rgb_critical_z9.pdf'
    assert rgb.render_rgb_critical(config,state,Draws(),catalog,extra,
        ({'F814W':object()},('F814W',),None,cluster),output,tmp_path,log=lambda _:None) == output
    import json
    info=store.read_figure_record(output)
    assert not output.with_suffix('.json').exists()
    assert output.exists() and info['candidate_overlays']==[]
    assert len(position_calls) == 1
    assert info['not_recovered_images'] == 1 and info['extra_images'] == 1
    # The overview draws no stretch sticks and saves no stretch directions.
    assert 'stretch_sticks' not in info and 'stretch_direction' not in info and info['redrawn_from_tables'] is False
    assert info['recovered_position_probability'] == .682689492 and info['recovered_position_contour_scale'] == 4.0
    assert info['candidate_catalog'] is None
    assert (info['marker_photometry'] is None) == (cluster != 'm0416')
    assert info['insets'] is None
    assert info['magnitude_band'] == []
    assert store.has_arrays(tmp_path,'hst_rgb_critical_z9') and store.has_record(tmp_path,'hst_rgb_critical_z9_band')
    assert not store.has_arrays(tmp_path,'jwst_rgb_critical_z9')
    assert sorted(p.name for p in tmp_path.iterdir() if p.suffix in ('.csv','.json','.h5') and p.name!='candidates.json')==[store.RESULTS_FILE]
    assert not store.has_table(tmp_path,'hst_rgb_critical_z9_stretch_directions')
    markers=store.read_table(tmp_path,'hst_rgb_markers')
    assert len(markers)==2 and (markers.marker_area_pt2==9).all()
    assert list(markers.color) == ['#FF9500', '#FF3030']        # the second image is not recovered
    candidate_output = output.with_name(output.stem+'_candidates.pdf')
    assert candidate_output.exists() == (cluster == 'm0416')
    if cluster == 'm0416':
        variant = store.read_figure_record(candidate_output)
        assert variant['candidate_marker'] == 'diamond'
        assert variant['candidate_overlays'] == ['K66.c1','A.1']
        colored = store.read_table(tmp_path,'hst_rgb_markers_candidates')
        assert colored.color.tolist() == ['#40DFFF','#FF3030']


def test_recovered_contours_are_enlarged_about_their_own_mean_position():
    from kappaccino.plotting.jwst_rgb import (enlarged_contour_segments, recovered_contour_segments,
                                              RECOVERED_CONTOUR_PROBABILITY, RECOVERED_CONTOUR_SCALE)
    assert RECOVERED_CONTOUR_PROBABILITY == .682689492 and RECOVERED_CONTOUR_SCALE == 4.
    positions=np.array([[1.,2.],[3.,2.],[np.nan,np.nan],[2.,5.]])         # finite mean (2, 3)
    segments=[np.array([[2.1,3.],[2.,3.2]]),np.array([[1.9,2.9]])]
    enlarged=enlarged_contour_segments(segments,positions,scale=10.)
    np.testing.assert_allclose(enlarged[0],[[3.,3.],[2.,5.]])
    np.testing.assert_allclose(enlarged[1],[[1.,2.]])                    # both segments share the centre
    np.testing.assert_allclose(enlarged_contour_segments(segments,positions)[0],[[2.4,3.],[2.,3.8]])   # the figure's factor
    with pytest.raises(ValueError):
        enlarged_contour_segments(segments,np.full((3,2),np.nan))
    # Only the 1 sigma contours of the observed images, each about its own image's positions.
    data=dict(image_labels=np.array(['1.a','2.b']),recovered=np.stack([positions,positions+10.],1))
    curves=[dict(kind='recovered',identity='2.b',probability=.682689492,segments=[np.array([[12.,13.5]])]),
            dict(kind='recovered',identity='2.b',probability=.954499736,segments=[np.array([[0.,0.]])]),
            dict(kind='extra',identity='1 extra 1',probability=.682689492,segments=[np.array([[0.,0.]])])]
    selected=recovered_contour_segments(dict(curves=curves,position_data=data))
    assert [identity for identity,_ in selected]==['2.b']
    np.testing.assert_allclose(selected[0][1][0],[[12.,13.+.5*RECOVERED_CONTOUR_SCALE]])   # 0.5 above the mean (12, 13)
    assert recovered_contour_segments(dict(curves=[]))==[]


def test_saved_tables_keep_string_ids(tmp_path):
    tables=tmp_path/'final'/'tables'
    from kappaccino.plotting.jwst_rgb import read_saved_extra_images
    frame=pd.DataFrame(dict(family_id=['1','NA'],image_label=['1.a','NA.b'],x_obs_arcsec=[1.,np.nan],
                            image_recovery_status=['recovered','not_recovered']))
    store.write_table(tables,'image_fit_quality',frame)
    read=store.read_table(tables,'image_fit_quality')
    assert read.family_id.tolist()==['1','NA'] and read.image_label.tolist()==['1.a','NA.b']
    assert read.x_obs_arcsec.dtype==float and np.isnan(read.x_obs_arcsec.iloc[1])
    pd.testing.assert_frame_equal(read,frame)
    empty=read_saved_extra_images(tables)                             # missing: the pipeline's empty table
    assert empty.empty and {'family_id','x_model_arcsec','y_model_arcsec'} <= set(empty.columns)
    store.write_table(tables,'image_recovery_extra_images',pd.DataFrame(columns=['family_id','x_model_arcsec','y_model_arcsec']))
    assert read_saved_extra_images(tables).empty
    store.write_table(tables,'image_recovery_extra_images',pd.DataFrame(dict(family_id=['3'],x_model_arcsec=[.1],y_model_arcsec=[.2])))
    assert read_saved_extra_images(tables).family_id.tolist()==['3']


# The likelihood sections of a run's saved config.json: a knee mass field shared by every image, and none.
KNEE_LIKELIHOOD=dict(likelihood_mode='gp_image_plane',gp_kernel='knee',sigma_grf_prior={'family':'TruncatedNormal'},
                     gp_knee_spectrum='knee',gp_blocks='all')
NO_FIELD_LIKELIHOOD=dict(likelihood_mode='smoothed',gp_kernel='image_plane',sigma_grf_prior=None,
                         gp_knee_spectrum='knee',gp_blocks='all')
REDRAW_INDICES=np.arange(100,112)


def _write_run_inputs(run, likelihood):
    """The run's command and its saved configuration (only the likelihood is read once rebuild_run_config is faked)."""
    import json
    (run/'inputs').mkdir(parents=True,exist_ok=True); (run/'final'/'artifacts').mkdir(parents=True,exist_ok=True)
    (run/'inputs'/'command.json').write_text(json.dumps(dict(cluster='catalogs/hera')))
    (run/'final'/'artifacts'/'config.json').write_text(json.dumps(dict(likelihood=likelihood)))


def _write_recovery(tables, catalog, indices=REDRAW_INDICES):
    """The per-draw positions of the image recovery (12 draws, enough for a KDE) and the common extra image."""
    store.write_table(tables,'image_recovery_extra_images',pd.DataFrame(dict(family_id=['1'],x_model_arcsec=[.1],y_model_arcsec=[-.2])))
    rng=np.random.default_rng(1)
    rows=[]
    for draw,index in enumerate(indices):
        for label,family,(x,y) in zip(catalog.image_label,catalog.family_id,[(-.3,-.3),(.3,.3)]):
            dx,dy=rng.normal(0,.01,2)*[1.,3.]
            rows.append(dict(draw_index=draw,posterior_index=int(index),family_id=family,image_label=label,recovered=True,
                             x_obs_arcsec=x,y_obs_arcsec=y,x_model_arcsec=x+dx,y_model_arcsec=y+dy,x_true_arcsec=x+dx,
                             y_true_arcsec=y+dy,x_predicted_arcsec=x+dx,y_predicted_arcsec=y+dy))
    store.write_table(tables,'posterior_image_residual_draws',pd.DataFrame(rows))


def _saved_run(tmp_path, *, shape=32, likelihood=KNEE_LIKELIHOOD):
    """A finished run's saved results in ``final/results.h5``: the band, the catalog, the extra images, the per-draw
    positions of the image recovery."""
    run=tmp_path/'results'/'hera'/'run'
    tables=run/'final'/'tables'; tables.mkdir(parents=True)
    (run/'final'/'figures').mkdir()
    _write_run_inputs(run, likelihood)
    catalog=_overview_catalog()
    store.write_table(tables,'image_fit_quality',catalog)
    _write_recovery(tables, catalog)
    x,y=np.meshgrid(np.linspace(-1,1,shape),np.linspace(-1,1,shape))
    determinant=x*x+y*y-.4
    band=dict(determinant=determinant,lower=determinant-.1,upper=determinant+.1,
              band=np.abs(determinant)<.1,fingerprint='saved-key',z_source=9.,draw_indices=REDRAW_INDICES,
              central_statistic='pointwise posterior median determinant')
    store.write_arrays(tables,'hst_rgb_critical_z9',band)
    store.write_record(tables,'hst_rgb_critical_z9_band',dict(resolution=shape))
    return run


def _redraw_config(run):
    from types import SimpleNamespace as NS
    image_config=NS(bands=('F814W',),image_dir='fake',figure_suffix='',candidate_catalog_path=None)
    return NS(paths=NS(run_name=run.name,figure_prefix='hera'),image_catalog=image_config,
              model=NS(reference=NS(reference=3,ra0_deg=64.,dec0_deg=-24.),image_constraints=None))


FAKE_IMAGING=({'F814W':object()},('F814W',),None,'hera')


def _patch_redraw(monkeypatch, run, config=None):
    from types import SimpleNamespace as NS
    from kappaccino.plotting import jwst_rgb as rgb
    from kappaccino.imaging import cutouts
    _fake_overview(monkeypatch)
    config=_redraw_config(run) if config is None else config
    monkeypatch.setattr(rgb,'rebuild_run_config',lambda run_dir:config)
    monkeypatch.setattr(cutouts,'imaging_variants',lambda config:[config.image_catalog])
    monkeypatch.setattr(cutouts,'load_cluster_imaging',lambda image_config,model,run_name:FAKE_IMAGING)
    import dataclasses
    monkeypatch.setattr(dataclasses,'replace',lambda obj,**changes:NS(**{**vars(obj),**changes}))
    return config


def _snapshot(run):
    return {path.relative_to(run):path.read_bytes() for path in sorted(run.rglob('*')) if path.is_file()}


def test_redraw_from_saved_tables(tmp_path, monkeypatch):
    from kappaccino.plotting import jwst_rgb as rgb
    run=_saved_run(tmp_path)
    _patch_redraw(monkeypatch, run)
    tables=run/'final'/'tables'
    def band():
        return store.read_arrays(tables,'hst_rgb_critical_z9'), store.read_record(tables,'hst_rgb_critical_z9_band')
    before=band()
    drawn=[]
    select=rgb.recovered_contour_segments
    monkeypatch.setattr(rgb,'recovered_contour_segments',lambda contours:drawn.extend(select(contours)) or drawn)
    written=rgb.redraw_rgb_critical(run,log=lambda _:None)
    output=run/'final'/'figures'/'hera_hst_rgb_critical_z9.pdf'
    assert written==[output] and output.exists()
    info=store.read_figure_record(output)
    assert not output.with_suffix('.json').exists()
    assert info['redrawn_from_tables'] is True and info['cache_fingerprint']=='saved-key' and info['reporting_draws']==12
    assert info['recovered_position_probability']==.682689492 and info['recovered_position_contour_scale']==4.
    assert 'stretch_sticks' not in info and info['not_recovered_images']==1 and info['extra_images']==1
    assert info['grf_field']=='shared field of the run' and info['determinant_resolution']==32
    assert info['model'].startswith('parametric + conditional GRF') and info['recovered_position_contours']==2
    # The band is read, never rewritten.
    after=band()
    assert after[1]==before[1] and set(after[0])==set(before[0])
    for key in before[0]:
        np.testing.assert_array_equal(after[0][key],before[0][key])
    # One enlarged 1 sigma contour per image, centred on the mean of its draws.
    positions=store.read_arrays(tables,'posterior_overview_positions')
    assert sorted(identity for identity,_ in drawn)==['1.a','2.b']
    for identity,segments in drawn:
        points=positions['recovered'][:,list(positions['image_labels']).index(identity)]
        centre=points.mean(0)
        assert np.hypot(*(segments[0]-centre).T).max()>5*np.hypot(*(points-centre).T).std()
    markers=store.read_table(tables,'hst_rgb_markers')
    assert markers.color.tolist()==['#FF9500','#FF3030']


def test_redraw_model_description_comes_from_the_saved_config(tmp_path, monkeypatch):
    import json
    from kappaccino.plotting import jwst_rgb as rgb
    from kappaccino.likelihood.grf_fields import shared_grf_label
    from types import SimpleNamespace as NS
    run=_saved_run(tmp_path, likelihood=NO_FIELD_LIKELIHOOD)
    config=_patch_redraw(monkeypatch, run)
    config.likelihood=NS(**KNEE_LIKELIHOOD)                 # the current run file, which must not be read
    rgb.redraw_rgb_critical(run,log=lambda _:None)
    info=store.read_figure_record(run/'final'/'figures'/'hera_hst_rgb_critical_z9.pdf')
    assert info['grf_field']==shared_grf_label(NS(gp_knee=False)) and info['model'].startswith('parametric only')
    saved={'likelihood':dict(KNEE_LIKELIHOOD,gp_blocks='systems')}
    assert rgb.saved_grf_field(saved)==(True,'shared field rebuilt from a system-block run (all-block twin, projected)')
    assert rgb.saved_grf_field({'likelihood':dict(KNEE_LIKELIHOOD,sigma_grf_prior=None)})[0] is False
    assert rgb.saved_grf_field({'likelihood':dict(KNEE_LIKELIHOOD,gp_knee_spectrum='knee_foreground')})[0] is False
    with pytest.raises(ValueError, match='likelihood.likelihood_mode'):
        rgb.saved_grf_field({'likelihood':{}}, 'run/final/artifacts/config.json')


def _compare(a, b, path='inputs'):
    """Assert two drawing inputs are equal: arrays, tables, mappings and sequences element by element."""
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), obj=path)
    elif isinstance(a, dict):
        assert set(a)==set(b), path
        for key in a:
            _compare(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)) and not (len(a) and isinstance(a[0], (int, float, str))):
        assert len(a)==len(b), path
        for k,(x,y) in enumerate(zip(a,b)):
            _compare(x, y, f'{path}[{k}]')
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a,b=np.asarray(a),np.asarray(b)
        if a.dtype.kind in 'fc':
            np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=path)
        else:
            np.testing.assert_array_equal(a.astype(str) if a.dtype.kind in 'OUS' else a,
                                          b.astype(str) if b.dtype.kind in 'OUS' else b, err_msg=path)
    elif isinstance(a, float) and isinstance(b, float):
        assert a==b or (np.isnan(a) and np.isnan(b)), path
    elif hasattr(a, '__dict__') and not isinstance(a, type):
        assert a is b or vars(a)==vars(b), path
    else:
        assert a==b, path


@pytest.mark.parametrize('field',[False,True])
def test_pipeline_and_redraw_draw_the_same_picture(tmp_path, monkeypatch, field):
    import json
    from types import SimpleNamespace as NS
    from kappaccino.plotting import jwst_rgb as rgb, mass_figure
    run=_saved_run(tmp_path, likelihood=KNEE_LIKELIHOOD if field else NO_FIELD_LIKELIHOOD)
    tables,figures=run/'final'/'tables',run/'final'/'figures'
    with h5py.File(tables.parent/store.RESULTS_FILE,'a') as file:     # the band is written by the pipeline below
        del file['arrays/hst_rgb_critical_z9'], file['records/hst_rgb_critical_z9_band']
    config=_patch_redraw(monkeypatch, run)
    Draws=_fake_overview(monkeypatch)
    class RunDraws(Draws):
        indices=REDRAW_INDICES
        evaluator=(NS(gp_knee=True,sigma_grf_index=0,gp_knee_spectrum='knee',state=NS(gp_blocks='all'))
                   if field else NS(gp_knee=False))
        def __len__(self):return len(self.indices)
    # Rows in the reverse of the catalog order, as the redraw test has them.
    monkeypatch.setattr(mass_figure,'arc_direction_table',lambda draws,catalog:pd.DataFrame(dict(
        image_label=['2.b','1.a'],e_arc_x=[np.nan,0.],e_arc_y=[np.nan,1.])))
    calls=[]
    draw=rgb.draw_rgb_critical
    def recording(*args,**kwargs):
        calls.append((args,kwargs))
        written=draw(*args,**kwargs)
        records={path.name:store.read_figure_record(path) for path in written}
        return written if not records else (calls[-1][1].update(_records=records) or written)
    monkeypatch.setattr(rgb,'draw_rgb_critical',recording)
    catalog=store.read_table(tables,'image_fit_quality')
    extra=rgb.read_saved_extra_images(tables)
    output=figures/'hera_hst_rgb_critical_z9.pdf'
    rgb.render_rgb_critical(config,NS(reference=(3,64.,-24.)),RunDraws(),catalog,extra,FAKE_IMAGING,output,tables,log=lambda _:None)
    pipeline_record=store.read_figure_record(output)
    pipeline_markers=store.read_table(tables,'hst_rgb_markers')
    rgb.redraw_rgb_critical(run,log=lambda _:None)
    redraw_record=store.read_figure_record(output)
    assert len(calls)==2
    (pipeline_args,pipeline_kwargs),(redraw_args,redraw_kwargs)=calls
    for kwargs in (pipeline_kwargs,redraw_kwargs):
        kwargs.pop('_records'); kwargs.pop('log')
    assert pipeline_kwargs.pop('redrawn',False) is False and redraw_kwargs.pop('redrawn') is True
    assert pipeline_kwargs['shared_grf'] is field
    _compare(pipeline_args, redraw_args, 'args')
    _compare(pipeline_kwargs, redraw_kwargs, 'kwargs')
    assert pipeline_record.pop('redrawn_from_tables') is False and redraw_record.pop('redrawn_from_tables') is True
    assert pipeline_record==redraw_record and pipeline_record['recovered_position_contours']==2
    assert pipeline_record['grf_field']==('shared field of the run' if field else 'none: no knee mass field')
    assert pipeline_record['model'].startswith('parametric + conditional GRF' if field else 'parametric only')
    pd.testing.assert_frame_equal(pipeline_markers,store.read_table(tables,'hst_rgb_markers'))


def test_redraw_refuses_draws_missing_from_the_image_recovery(tmp_path, monkeypatch):
    from kappaccino.plotting import jwst_rgb as rgb
    run=_saved_run(tmp_path)
    _patch_redraw(monkeypatch, run)
    rgb.redraw_rgb_critical(run,log=lambda _:None)                    # a first redraw writes every output
    tables=run/'final'/'tables'
    saved=store.read_arrays(tables,'hst_rgb_critical_z9')
    store.write_arrays(tables,'hst_rgb_critical_z9',{**saved,'draw_indices':REDRAW_INDICES+1})
    before=_snapshot(run)
    with pytest.raises(ValueError, match=r'1 of the 12 draw indices of hst_rgb_critical_z9 are not a posterior_index .*\(112\)'):
        rgb.redraw_rgb_critical(run,log=lambda _:None)
    assert _snapshot(run)==before                                       # nothing rewritten, nothing added


def test_redraw_logs_a_figure_without_contours(tmp_path, monkeypatch):
    from kappaccino.plotting import jwst_rgb as rgb
    from kappaccino.diagnostics import position_contours
    run=_saved_run(tmp_path)
    _patch_redraw(monkeypatch, run)
    monkeypatch.setattr(position_contours,'load_or_compute_contours',lambda *a,**k:{'curves':[]})
    lines=[]
    rgb.redraw_rgb_critical(run,log=lines.append)
    assert any('no recovered-position contour drawn although 1 images are recovered' in line for line in lines)


def test_redraw_falls_back_and_refuses(tmp_path, monkeypatch):
    from kappaccino.plotting import jwst_rgb as rgb
    run=_saved_run(tmp_path)
    _patch_redraw(monkeypatch, run)
    tables=run/'final'/'tables'
    figure=run/'final'/'figures'/'hera_hst_rgb_critical_z9.pdf'
    # The mass figure's stretch directions, when saved, are not read: the overview draws no sticks.
    store.write_table(tables,'image_catalog_arc_directions',pd.DataFrame(dict(image_label=['1.a','1.a'],e_arc_x=[1.,0.],e_arc_y=[0.,1.])))
    lines=[]
    rgb.redraw_rgb_critical(run,log=lines.append)
    assert not any('stretch' in line for line in lines) and 'stretch_sticks' not in store.read_figure_record(figure)
    with h5py.File(tables.parent/store.RESULTS_FILE,'a') as file:
        del file['arrays/hst_rgb_critical_z9']
    assert rgb.redraw_rgb_critical(run,log=lambda _:None)==[]          # no band: skipped
    other=_saved_run(tmp_path/'other', shape=16)
    _patch_redraw(monkeypatch, other)
    with pytest.raises(ValueError, match='determinant'):
        rgb.redraw_rgb_critical(other,log=lambda _:None)


@pytest.mark.parametrize('name',['posterior_image_residual_draws','image_fit_quality','final/artifacts/config.json'])
def test_redraw_names_the_run_of_a_missing_table(tmp_path, monkeypatch, name):
    from kappaccino.plotting import jwst_rgb as rgb
    run=_saved_run(tmp_path)
    _patch_redraw(monkeypatch, run)
    if '/' in name:
        (run/name).unlink()
        message=f'no {name}'
    else:
        message=f'no final table {name}'
        with h5py.File(run/'final'/store.RESULTS_FILE,'a') as file:
            del file[f'tables/{name}']
    before=_snapshot(run)
    with pytest.raises(FileNotFoundError, match=rf'Run {run.resolve()}: {message}; .*completed image recovery'):
        rgb.redraw_rgb_critical(run,log=lambda _:None)
    assert _snapshot(run)==before


def test_redraw_entry_point_refuses_a_missing_folder(tmp_path, capsys):
    from kappaccino.plotting import jwst_rgb as rgb
    missing=tmp_path/'results'/'hera'/'no-such-run'
    with pytest.raises(SystemExit) as stop:
        rgb.main([str(missing)])
    assert stop.value.code==2 and f'not a run folder: {missing}' in capsys.readouterr().err
    assert not missing.exists()


def test_rebuilt_run_config_must_match_the_saved_picture_inputs(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from kappaccino import cli
    from kappaccino.plotting.jwst_rgb import rebuild_run_config, saved_grf_field
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    run=tmp_path/'results'/'hera'/'myrun'
    (run/'inputs').mkdir(parents=True); (run/'final'/'artifacts').mkdir(parents=True)
    (run/'inputs'/'command.json').write_text(json.dumps(dict(cluster='catalogs/hera',profiles=['fast'],overrides=[],
                                                             images=None,member_profile=None,member_slope=None)))
    built=cli.build_config('catalogs/hera',profiles=['fast'],run_label='myrun',results=tmp_path/'results')
    saved=built.to_json_dict()
    (run/'final'/'artifacts'/'config.json').write_text(json.dumps(saved))
    config=rebuild_run_config(run)
    assert (Path(config.paths.output_dir)/config.paths.run_name).resolve()==run.resolve()
    # The saved likelihood gives the label the run's evaluator gives.
    shared,label=saved_grf_field(json.loads(json.dumps(saved)))
    assert isinstance(shared,bool) and label in {'shared field of the run','none: no knee mass field',
        'shared field rebuilt from a system-block run (all-block twin, projected)'}
    assert shared==(label!='none: no knee mass field')
    altered=json.loads(json.dumps(saved))
    altered['image_catalog']['bands']=['F105W','F125W','F160W']
    (run/'final'/'artifacts'/'config.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='image_catalog'):
        rebuild_run_config(run)
    altered=json.loads(json.dumps(saved))
    del altered['model']['image_constraints']
    (run/'final'/'artifacts'/'config.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError, match=r'config\.json has no model\.image_constraints'):
        rebuild_run_config(run)


def test_overview_catalog_magnitude_scaling_and_id_alignment(tmp_path):
    from types import SimpleNamespace as NS
    from kappaccino.plotting.jwst_rgb import overview_marker_photometry, rgb_overview_stem
    path = tmp_path / 'images.yaml'
    path.write_text('''format: lenscluster-images/1
systems:
  '1':
    z: 2.0
    images:
      a: {ra: 64, dec: -24, bands: {hst_f814w: [24, 0.1]}}
      b: {ra: 64, dec: -24, bands: {hst_f814w: [28, 0.1]}}
''')
    config = NS(model=NS(image_constraints=NS(catalog_path=path)))
    catalog = pd.DataFrame(dict(image_label=['1.b', '1.a', 'new'], x_obs_arcsec=[0.,0.,0.], y_obs_arcsec=[0.,0.,0.]))
    result = overview_marker_photometry(config, catalog, (3,64.,-24.), None, 'hst_rgb_critical_z9')
    np.testing.assert_allclose(result.magnitude_AB[:2], [28.,24.])
    np.testing.assert_allclose(result.marker_area_pt2, [9.,16*10**.4,9.])
    assert result.valid_photometry.tolist() == [True, True, False]
    assert rgb_overview_stem(NS(), ['F435W','F606W','F814W']) == 'hst_rgb_critical_z9'
    assert rgb_overview_stem(NS(), ['F150W','F200W','F444W']) == 'jwst_rgb_critical_z9'
