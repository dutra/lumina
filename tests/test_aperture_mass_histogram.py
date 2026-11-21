from types import SimpleNamespace as NS
import json
from kappaccino.io import store

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from kappaccino.plotting import aperture_mass_histogram as am


@pytest.mark.parametrize('prefix',['hera','ares'])
def test_uniform_mass_apertures_and_output(tmp_path,monkeypatch,prefix):
    wcs=WCS(naxis=2)
    wcs.wcs.crpix=[101,101];wcs.wcs.crval=[0,0]
    wcs.wcs.cdelt=[-.1/3600,.1/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    truth=tmp_path/'truth.fits'
    fits.writeto(truth,np.full((201,201),.5),wcs.to_header())
    members=pd.DataFrame(dict(member_id=['p:1','p:2','p:3'],catalog_id=['1','2','3'],
        catalog_mag=[22.,22.,22.],x_centre=[0.,5.,100.],y_centre=[0.,0.,0.]))
    state=NS(reference=(3,0.,0.),kpc_per_arcsec=6.,members=NS(member_ids=tuple(members.member_id)))
    class Evaluator:
        def packed_state_for_z(self,p,z):return None
        def total_fields_at(self,xy,packed,chunk):
            return .5*xy,np.broadcast_to(.5*np.eye(2),(len(xy),2,2))
    monkeypatch.setattr(am,'mass_per_kappa_pixel_msun',lambda *a:.01)
    paths=NS(tables=tmp_path/'tables',figures=tmp_path/'figures')
    paths.tables.mkdir();paths.figures.mkdir()
    store.write_table(paths.tables,'member_diagnostics',members)
    images=pd.DataFrame(dict(x_obs_arcsec=[-2,2,2,-2],y_obs_arcsec=[-2,-2,2,2]))
    config=NS(paths=NS(figure_prefix=prefix),outputs=NS(truth=NS(kappa_true_fits=truth),caustic_source_redshift=9.))
    class Draws(NS):
        def __len__(self):return len(self.physical)
    draws=Draws(evaluator=Evaluator(),physical=np.ones((2,1)),has_field=False)
    path=am.write_aperture_mass_histogram(config,state,draws,paths,images,log=lambda s:None)
    table=store.read_table(paths.tables, 'member_aperture_masses')
    np.testing.assert_allclose(table.ratio[:2],1,atol=1e-12)
    np.testing.assert_allclose(table.model_mass_msun[:2],.5*np.pi*table.aperture_arcsec[:2]**2,rtol=1e-12)
    assert table.inside_image_hull.tolist()==[True,False,False]
    assert table.usable.tolist()==[True,True,False]
    assert path.name=='results.h5' and not list(paths.tables.glob('*.json'))
    assert table.n_draws.tolist()==[2,2,2]
    stats=store.read_record(paths.tables, 'aperture_mass_histogram')
    assert stats['samples']['all']['n']==2
    assert stats['samples']['image_hull']['n']==1
    assert stats['skipped_member_ids']==['p:3']
    smaller=am.aperture_masses(state,Evaluator(),np.ones(1),members,
        images.to_numpy(),config.outputs,normalization_kpc=3.)
    np.testing.assert_allclose(smaller.model_mass_msun[:2],table.model_mass_msun[:2]*(3/am.NORMALIZATION_KPC)**2,rtol=1e-12)
    def field(x,y,family):
        return .1*x,.1*y,np.full_like(x,.1),np.zeros_like(x),np.zeros_like(x),np.full_like(x,.1)
    corrected=am.aperture_masses(state,Evaluator(),np.ones(1),members,images.to_numpy(),config.outputs,
                                grf_correction=field,grf_weight=2.)
    np.testing.assert_allclose(corrected.ratio[:2],1.4,atol=1e-12)


def test_median_over_draws_takes_member_medians_and_needs_every_draw_usable():
    base=pd.DataFrame(dict(member_id=['a','b'],truth_mass_msun=[2.,4.],usable=[True,True],model_quadrature_error=[0.,0.]))
    draws=[base.assign(model_mass_msun=m,usable=u,model_quadrature_error=e)
           for m,u,e in [([1.,4.],[True,True],[.001,0.]),([2.,8.],[True,False],[0.,.002]),([3.,6.],[True,True],[0.,0.])]]
    table=am.median_over_draws(draws)
    np.testing.assert_allclose(table.model_mass_msun,[2.,6.])
    assert table.usable.tolist()==[True,False]
    np.testing.assert_allclose(table.ratio[:1],[1.])
    assert np.isnan(table.ratio[1])
    np.testing.assert_allclose(table.model_quadrature_error,[.001,.002])


def test_degenerate_footprint_and_other_clusters():
    assert not am.inside_image_footprint([[0,0]],[[0,0],[1,0],[2,0]]).any()
    assert am.write_aperture_mass_histogram(NS(paths=NS(figure_prefix='as1063')),None,None,None,None) is None


def test_quartiles_and_rendering_do_not_drop_display_tails(tmp_path):
    values=np.r_[np.linspace(.8,1.2,198),.1,4.]
    table=pd.DataFrame(dict(ratio=values,complete_aperture=True,inside_image_hull=np.arange(200)<10))
    stats=am.distribution_statistics(table)['all']
    np.testing.assert_allclose([stats['p25'],stats['median'],stats['p75']],np.quantile(values,[.25,.5,.75]))
    counts=am.render_histogram(table,tmp_path/'hist.pdf')
    assert counts.all_members.sum()==200
    assert counts.inside_image_hull.sum()==10


def test_a_second_pass_reuses_the_saved_masses_until_an_input_changes(tmp_path,monkeypatch):
    wcs=WCS(naxis=2)
    wcs.wcs.crpix=[101,101];wcs.wcs.crval=[0,0]
    wcs.wcs.cdelt=[-.1/3600,.1/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    truth=tmp_path/'truth.fits'
    fits.writeto(truth,np.full((201,201),.5),wcs.to_header())
    members=pd.DataFrame(dict(member_id=['p:1','p:2'],catalog_id=['1','2'],catalog_mag=[22.,22.],x_centre=[0.,5.],y_centre=[0.,0.]))
    state=NS(reference=(3,0.,0.),kpc_per_arcsec=6.,members=NS(member_ids=tuple(members.member_id)))
    calls=[]
    class Evaluator:
        def packed_state_for_z(self,p,z):return None
        def total_fields_at(self,xy,packed,chunk):
            calls.append(len(xy))
            return .5*xy,np.broadcast_to(.5*np.eye(2),(len(xy),2,2))
    monkeypatch.setattr(am,'mass_per_kappa_pixel_msun',lambda *a:.01)
    paths=NS(tables=tmp_path/'tables',figures=tmp_path/'figures')
    paths.tables.mkdir();paths.figures.mkdir()
    store.write_table(paths.tables,'member_diagnostics',members)
    images=pd.DataFrame(dict(x_obs_arcsec=[-2,2,2,-2],y_obs_arcsec=[-2,-2,2,2]))
    config=NS(paths=NS(figure_prefix='hera'),outputs=NS(truth=NS(kappa_true_fits=truth),caustic_source_redshift=9.))
    class Draws(NS):
        def __len__(self):return len(self.physical)
    lines=[]
    def run(key):
        am.write_aperture_mass_histogram(config,state,Draws(evaluator=Evaluator(),physical=np.ones((2,1)),has_field=False,key=key),
                                         paths,images,log=lines.append)
        return store.read_table(paths.tables,'member_aperture_masses'),store.read_record(paths.tables,'aperture_mass_histogram')
    first,record=run('a')
    n=len(calls)
    assert n>0 and 'none saved' in lines[0]
    again,second=run('a')
    assert len(calls)==n and 'reused' in lines[-2]
    pd.testing.assert_frame_equal(again,first,check_exact=True)
    assert second==record
    run('b')
    assert len(calls)==2*n and 'saved with other inputs' in lines[-2]


def test_a_pass_interrupted_between_the_masses_and_their_fingerprint_leaves_the_old_masses_reusable(tmp_path,monkeypatch):
    wcs=WCS(naxis=2)
    wcs.wcs.crpix=[101,101];wcs.wcs.crval=[0,0]
    wcs.wcs.cdelt=[-.1/3600,.1/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    truth=tmp_path/'truth.fits'
    fits.writeto(truth,np.full((201,201),.5),wcs.to_header())
    members=pd.DataFrame(dict(member_id=['p:1','p:2'],catalog_id=['1','2'],catalog_mag=[22.,22.],x_centre=[0.,5.],y_centre=[0.,0.]))
    state=NS(reference=(3,0.,0.),kpc_per_arcsec=6.,members=NS(member_ids=tuple(members.member_id)))
    scale={'v':.5}
    class Evaluator:
        def packed_state_for_z(self,p,z):return None
        def total_fields_at(self,xy,packed,chunk):
            return scale['v']*xy,np.broadcast_to(scale['v']*np.eye(2),(len(xy),2,2))
    monkeypatch.setattr(am,'mass_per_kappa_pixel_msun',lambda *a:.01)
    paths=NS(tables=tmp_path/'tables',figures=tmp_path/'figures')
    paths.tables.mkdir();paths.figures.mkdir()
    store.write_table(paths.tables,'member_diagnostics',members)
    images=pd.DataFrame(dict(x_obs_arcsec=[-2,2,2,-2],y_obs_arcsec=[-2,-2,2,2]))
    config=NS(paths=NS(figure_prefix='hera'),outputs=NS(truth=NS(kappa_true_fits=truth),caustic_source_redshift=9.))
    class Draws(NS):
        def __len__(self):return len(self.physical)
    def run(key):
        am.write_aperture_mass_histogram(config,state,Draws(evaluator=Evaluator(),physical=np.ones((2,1)),has_field=False,key=key),
                                         paths,images,log=lambda m:None)
        return store.read_table(paths.tables,'member_aperture_masses')
    first=run('a')                                   # inputs A: masses and their fingerprint
    scale['v']=.9                                    # inputs B give other masses, and the pass dies at the fingerprint
    write_record=store.write_record
    def dies(stage,name,*a,**k):
        if name=='aperture_mass_histogram':
            raise KeyboardInterrupt
        return write_record(stage,name,*a,**k)
    monkeypatch.setattr(store,'write_record',dies)
    with pytest.raises(KeyboardInterrupt):
        run('b')
    monkeypatch.setattr(store,'write_record',write_record)
    scale['v']=.5
    pd.testing.assert_frame_equal(run('a'),first,check_exact=True)   # back to inputs A: A's masses, not B's
