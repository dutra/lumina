import json
from kappaccino.io import store
import pytest
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.wcs import WCS

from kappaccino.plotting import covariance_cutouts as cc
from kappaccino.likelihood.evaluator import Evaluator
from kappaccino.diagnostics.exact_images import ExactImageSolver
from kappaccino.core.state import build_state


def test_quadratic_contour_linear_limit_and_curved_map():
    a = np.diag([.2, .8]); c = np.diag([.04, .09])
    segments = cc.quadratic_contours(a, np.zeros((2, 2, 2)), c, 4., grid_size=801)
    points = np.concatenate(segments)
    q = np.sum((points @ a.T)**2 / np.diag(c), axis=1)
    np.testing.assert_allclose(q, 1., atol=5e-5)
    np.testing.assert_allclose(np.max(abs(points), axis=0), [1., .375], atol=1e-5)
    # beta_y = .8 y + .15 x^2 bends the same unit contour into a curved strip.
    b = np.zeros((2, 2, 2)); b[1, 0, 0] = .3
    points = np.concatenate(cc.quadratic_contours(a, b, c, 4., grid_size=801))
    expected = np.column_stack([.2*points[:, 0], .8*points[:, 1]+.15*points[:, 0]**2])
    np.testing.assert_allclose(cc.quadratic_offset(points, a, b), expected, atol=1e-15)
    np.testing.assert_allclose(np.sum(expected**2 / np.diag(c), axis=1), 1., atol=7e-5)


def test_exact_geometry_and_derivatives(synthetic_config):
    state = build_state(synthetic_config); evaluator = Evaluator(state)
    g = cc.draw_geometry(state, evaluator, state.initial_theta)
    solver = ExactImageSolver(evaluator, synthetic_config.outputs.image_solver)
    for i in [0, 2]:  # different source redshifts
        family = state.family_data[int(state.images.family_index_per_image[i])]
        packed = evaluator.packed_state_for_z(g['physical'], family.z_source)
        point = np.array([state.images.x_obs[i], state.images.y_obs[i]])
        delta = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])*1e-4
        _, _, a, b, c, d = solver.ray_shoot_and_jacobian(packed, *(point+delta).T)
        jac = np.stack([a,b,c,d], axis=-1).reshape(-1,2,2)
        finite = np.stack([(jac[0]-jac[1])/2e-4, (jac[2]-jac[3])/2e-4], axis=-1)
        np.testing.assert_allclose(g['b'][i], finite, rtol=2e-6, atol=1e-9)
        inverse = np.linalg.inv(g['a'][i]); eig = np.linalg.eigvalsh(inverse@g['c'][i]@inverse.T)
        np.testing.assert_allclose(np.sort(g['widths'][i]**2), eig, rtol=1e-12)


def test_small_catalog_pdf_and_style(synthetic_config, tmp_path, monkeypatch):
    state=build_state(synthetic_config); evaluator=Evaluator(state)
    # FamilyData order is not necessarily the flattened image-family order.
    state.family_data = list(reversed(state.family_data))
    posterior=SimpleNamespace(samples=np.array([state.initial_theta]), physical=np.array([np.asarray(evaluator.physical(state.initial_theta))]))
    from kappaccino.reporting.draws import ReportingDraws
    draws=ReportingDraws(evaluator, posterior, 1, 0)
    paths=SimpleNamespace(tables=tmp_path/'tables',figures=tmp_path/'figures')
    wcs=WCS(naxis=2);wcs.wcs.crpix=[1000,1000];wcs.wcs.crval=list(state.reference[1:]);wcs.wcs.cdelt=[-.05/3600,.05/3600];wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    band=SimpleNamespace(wcs=wcs,shape=(2000,2000),pixel_scale_arcsec=.05,path=tmp_path/'synthetic.fits')
    monkeypatch.setattr(cc,'rgb_cutout',lambda *args: np.full((120,120,3),.2))
    style=dict(plt.rcParams)
    imaging=({'F814W':band},('F814W',),None,'toy')
    assert cc.save_covariance_cutouts(synthetic_config,state,draws,paths,imaging) is not None
    path=cc.render_covariance_cutouts(synthetic_config,state,paths,imaging)
    assert path.read_bytes().startswith(b'%PDF')
    assert dict(plt.rcParams)==style
    metadata=store.read_record(paths.tables, 'small_lambda_covariance_cutouts')
    rows=metadata['records'];assert len(rows)==4
    assert [r['lambda_abs'] for r in rows]==sorted(r['lambda_abs'] for r in rows)
    assert {r['z_source'] for r in rows}=={2.,3.}
    assert all(r['z_source'] == {'1': 2., '2': 3.}[r['family_id']] for r in rows)
    assert all(r['sigma_meas']>0 for r in rows)
    assert all(r['sampled_max_exact_map_error'] is not None for r in rows)


@pytest.mark.parametrize("cluster,missing_hst,expected", [("toy",False,1),("m0416",False,2),("as1063",True,1),("hera",False,1),("ares",False,1)])
def test_output_reuses_loaded_imaging(monkeypatch, synthetic_config, tmp_path, cluster, missing_hst, expected):
    from dataclasses import replace
    from kappaccino.reporting import outputs, results_tex
    from kappaccino.imaging import cutouts
    from kappaccino.diagnostics import image_recovery
    from kappaccino.plotting import mass_figure, plots, covariance_cutouts, fold_pairs
    config=replace(synthetic_config,paths=replace(synthetic_config.paths,run_name=cluster),outputs=replace(synthetic_config.outputs,rgb=replace(synthetic_config.outputs.rgb,image_dir='already-loaded')))
    state=SimpleNamespace(parameter_specs=[],reference=(3,0,0),z_lens=.3,z_source_values=[2.])
    posterior=SimpleNamespace(samples=np.zeros((1,1)),physical=np.zeros((1,1)),log_prob=np.zeros(1))
    evaluator=SimpleNamespace(physical=lambda p:p,source_positions=lambda p:np.zeros((1,2)))
    paths=SimpleNamespace(tables=tmp_path/'tables',figures=tmp_path/'figures',fits=tmp_path/'fits',root=tmp_path)
    monkeypatch.setattr(results_tex,'write_run_results',lambda *a,**k:None)
    for name in ['plot_image_residual_histogram','plot_image_recovery']:
        monkeypatch.setattr(plots,name,lambda *a,**k:None)
    monkeypatch.setattr(image_recovery,'image_recovery_tables',lambda *a,**k:{'image_fit_quality':pd.DataFrame({'image':['a']}),'image_count_recovery':pd.DataFrame(),'image_recovery_extra_images':pd.DataFrame(),
        'posterior_image_residual_draws':pd.DataFrame(columns=['family_id','image_label','recovered','x_obs_arcsec','y_obs_arcsec','x_predicted_arcsec','y_predicted_arcsec'])})
    monkeypatch.setattr(image_recovery,'fit_summary',lambda *a,**k:{'observed_image_count':1})
    imaging=cluster if cluster in {"hera","ares"} else "test-imaging"
    def load(image_config, *args):
        if missing_hst and not image_config.figure_suffix:
            raise FileNotFoundError('HST unavailable')
        bands = ['F115W','F150W','F200W'] if image_config.figure_suffix == '_jwst' else ['F435W','F606W','F814W']
        return (None,bands,None,imaging)
    monkeypatch.setattr(mass_figure,'save_cluster_mass_figure',lambda *a,**k:{})
    monkeypatch.setattr(mass_figure,'draw_cluster_mass_figure',lambda *a,**k:None)
    calls=[];monkeypatch.setattr(covariance_cutouts,'save_covariance_cutouts',lambda *a,**k:calls.append(a[4]) or 'saved')
    renders=[];monkeypatch.setattr(covariance_cutouts,'render_covariance_cutouts',lambda *a,**k:renders.append(a[3]))
    from kappaccino.diagnostics import position_contours
    monkeypatch.setattr(position_contours,"load_or_compute_contours",lambda *a,**k:{})
    monkeypatch.setattr(position_contours,"load_position_contours",lambda *a,**k:{})
    from kappaccino.plotting import small_lambda_jwst, paper_cutouts
    lambda_calls=[]
    monkeypatch.setattr(small_lambda_jwst,'save_small_lambda_jwst',lambda *a,**k:lambda_calls.append(a))
    monkeypatch.setattr(paper_cutouts,'prepare_paper_cutouts',lambda *a,**k:None)
    from kappaccino.plotting import jwst_rgb
    rgb_calls=[]
    monkeypatch.setattr(jwst_rgb,'save_rgb_critical',lambda *a,**k:'saved')
    monkeypatch.setattr(jwst_rgb,'render_saved_rgb_critical',lambda *a,**k:rgb_calls.append(a))
    fold_calls=[]; fold_renders=[]
    monkeypatch.setattr(fold_pairs,"compute_fold_diagnostics",lambda *a,**k: fold_calls.append(1) or {"pairs":[],"systems":[]})
    monkeypatch.setattr(fold_pairs,"save_fold_data",lambda *a:None)
    monkeypatch.setattr(fold_pairs,"load_fold_data",lambda *a:{"pairs":[],"systems":[]})
    monkeypatch.setattr(fold_pairs,"render_fold_diagnostics",lambda *a,**k:fold_renders.append(a[4]))
    loads=[];monkeypatch.setattr(cutouts,'load_cluster_imaging',lambda *a:loads.append(a[0].figure_suffix) or load(*a))
    summary=outputs.write_outputs(config,state,evaluator,posterior,{},paths,log=lambda _:None,only=('rgb',),fail_fast=True)
    assert list(summary['modules'])==['images','rgb'] and summary['modules']['rgb']['status']=='done'
    # Each variant's imaging is loaded once in the pass.
    assert len(loads)==len(set(loads))
    # One RGB critical overview per loaded imaging variant, named by its instrument.
    assert len(rgb_calls)==expected
    assert sorted(call[5].name for call in rgb_calls)==sorted(
        ("jwst" if call[0].outputs.rgb.figure_suffix=="_jwst" else "hst")+"_rgb_critical_z9.pdf" for call in rgb_calls)
    assert any(call[5].name=="jwst_rgb_critical_z9.pdf" for call in rgb_calls)==(cluster in {"m0416","as1063"})
    assert len(lambda_calls)==(1 if cluster in {"m0416","as1063"} else 0)
    assert all(call[4][3] is imaging for call in lambda_calls)
    assert len(fold_calls)==1 and len(fold_renders)==expected
    assert len(calls)==expected and all(call[3] is imaging for call in calls)
    assert len(renders)==expected and all(call[3] is imaging for call in renders)


def test_default_jwst_variants(synthetic_config):
    from dataclasses import replace
    from kappaccino.imaging.cutouts import imaging_variants, variant_config
    from kappaccino.imaging.rgb import DEFAULT_JWST_RGB_BANDS
    from kappaccino.config import RGBDisplayConfig
    for name in ['m0416_bergamini2023', 'as1063']:
        config = replace(synthetic_config, paths=replace(synthetic_config.paths, run_name=name))
        primary, jwst = imaging_variants(config)
        assert primary is config.outputs.rgb
        assert jwst.bands == tuple(DEFAULT_JWST_RGB_BANDS)
        assert jwst.display == RGBDisplayConfig()
        assert jwst.figure_suffix == '_jwst'
        assert jwst.image_dir in ('data/imaging/m0416/canucs_dr1', 'data/imaging/as1063/dja_v7.5')
        assert imaging_variants(variant_config(config, jwst)) == [jwst]
    assert imaging_variants(synthetic_config) == [synthetic_config.outputs.rgb]
