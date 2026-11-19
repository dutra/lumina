"""Photometry must remain attached to the right image when catalogs change."""
import pytest
import numpy as np
import pandas as pd
from kappaccino.plotting.jwst_rgb import marker_photometry
from kappaccino.io.hdf5 import save_arrays


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
    # One defined and one undefined stretch direction: a stick is drawn for the first only.
    monkeypatch.setattr(mass_figure,'arc_direction_table',lambda draws,catalog:pd.DataFrame(dict(
        image_label=catalog.image_label,e_arc_x=[1.,np.nan],e_arc_y=[0.,np.nan])))
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
    info=json.loads(output.with_suffix('.json').read_text())
    assert output.exists() and info['candidate_overlays']==[]
    assert len(position_calls) == 1
    assert 'own redshift' in info['stretch_direction'] and info['not_recovered_images'] == 1 and info['extra_images'] == 1
    assert info['stretch_sticks'] == 1
    assert info['recovered_position_probability'] == .682689492 and info['recovered_position_contour_scale'] == 4.0
    assert info['candidate_catalog'] is None
    assert (info['marker_photometry'] is None) == (cluster != 'm0416')
    assert info['insets'] is None
    assert info['magnitude_band'] == []
    assert (tmp_path/'hst_rgb_critical_z9.h5').exists() and (tmp_path/'hst_rgb_critical_z9_band.json').exists()
    assert not (tmp_path/'jwst_rgb_critical_z9.h5').exists()
    # The stretch directions are saved beside the band.
    saved=pd.read_csv(tmp_path/'hst_rgb_critical_z9_stretch_directions.csv')
    assert saved.image_label.tolist()==['1.a','2.b'] and saved.e_arc_x.iloc[0]==1.
    markers=pd.read_csv(tmp_path/'hst_rgb_markers.csv')
    assert len(markers)==2 and (markers.marker_area_pt2==9).all()
    assert list(markers.color) == ['#FF9500', '#FF3030']        # the second image is not recovered
    candidate_output = output.with_name(output.stem+'_candidates.pdf')
    assert candidate_output.exists() == (cluster == 'm0416')
    if cluster == 'm0416':
        variant = json.loads(candidate_output.with_suffix('.json').read_text())
        assert variant['candidate_marker'] == 'diamond'
        assert variant['candidate_overlays'] == ['K66.c1','A.1']
        colored = pd.read_csv(tmp_path/'hst_rgb_markers_candidates.csv')
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
