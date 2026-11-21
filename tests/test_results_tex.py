"""Publication exports preserve recorded quantities and produce valid TeX names."""
import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lumina.reporting.results_tex import command_prefix, duration, write_run_results, write_mock_results
from lumina.io import store
from lumina.io.hdf5 import save_arrays


@pytest.mark.parametrize('name,expected', [('ares','Ares'), ('hera','Hera'), ('MJ0416','MJZeroFourOneSix'), ('as1063','ASOneZeroSixThree'), ('cluster-2','ClusterTwo')])
def test_prefix(name, expected):
    assert command_prefix(name) == expected


def test_duration():
    assert duration(3599.6) == r'\ensuremath{1\,\mathrm{h}\;0\,\mathrm{min}\;0\,\mathrm{s}}'
    assert duration(90061) == r'\ensuremath{25\,\mathrm{h}\;1\,\mathrm{min}\;1\,\mathrm{s}}'
    for value in (None, np.nan, -1):
        assert 'not available' in duration(value)
    with pytest.raises(ValueError):
        command_prefix('---')


def test_run_export(tmp_path):
    def save(path, data):
        if path.startswith('tables/'):
            store.write_record(tmp_path, path[len('tables/'):-len('.json')], data)
            return
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))
    save('artifacts/config.json', {'paths': {'figure_prefix': 'm0416'}})
    save('artifacts/summary.json', {'images': 3, 'members': 8, 'exact_members': 2})
    save('artifacts/posterior_diagnostics.json', {'elapsed_seconds': 61, 'max_r_hat': 1.005})
    save('tables/run_summary.json', {'fit': {'observed_image_count': 3, 'recovered_image_count': {'median': 2., 'q16': 1., 'q84': 3.}}})
    path = write_run_results(tmp_path)
    text = path.read_text()
    assert path.name == 'm0416_results.tex'
    assert path.parent == tmp_path / 'figures'
    assert not (tmp_path / 'tables' / path.name).exists()
    assert r'\MJZeroFourOneSixNRecovered}{2}' in text
    assert r'\MJZeroFourOneSixNRecoveredInterval}{{2_{-1}^{+1}}}' in text
    assert r'\MJZeroFourOneSixNMissed}{1}' in text
    assert r'\MJZeroFourOneSixRMS}{\textnormal{not available}}' in text
    assert r'\MJZeroFourOneSixNFarFieldMembers}{6}' in text
    assert r'\MJZeroFourOneSixRuntime}{\textnormal{not available}}' in text
    commands = re.findall(r'\\newcommand\{\\([^}]+)\}', text)
    assert commands and all(re.fullmatch('[A-Za-z]+', c) for c in commands)
    assert r'\MJZeroFourOneSixLensOnlyRMS}{\textnormal{not available}}' in text
    save('tables/run_summary.json', {'fit': {'observed_image_count': 3, 'recovered_image_count': {'median': 2., 'q16': 2., 'q84': 2.},
                                             'lens_only_recovered_image_count': {'median': 3., 'q16': 3., 'q84': 3.}}})
    # Two draws: RMS 0.3 and 0.5 (residuals 0.3, 0.3 and 0.4, 0.6, plus an unrecovered image); lens only doubled.
    rows = pd.DataFrame({'draw_index': [0, 0, 0, 1, 1, 1], 'recovered': [True, True, False, True, True, False],
                         'image_residual_arcsec': [.3, .3, np.nan, .4, .6, np.nan]})
    store.write_table(tmp_path, 'posterior_image_residual_draws', rows)
    store.write_table(tmp_path, 'posterior_image_residual_draws_lens_only',
                      rows.assign(image_residual_arcsec=2 * rows.image_residual_arcsec))
    text = write_run_results(tmp_path).read_text()
    rms = np.percentile([.3, np.sqrt((.16 + .36) / 2)], [16, 50, 84])
    assert rf'\MJZeroFourOneSixRMS}}{{{{{rms[1]:.3f}_{{-{rms[1] - rms[0]:.3f}}}^{{+{rms[2] - rms[1]:.3f}}}}}}}' in text
    spread = np.percentile([.3, .5], [84]) [0] - .4      # per-draw medians 0.3 and 0.5
    assert rf'\MJZeroFourOneSixMedianResidual}}{{{{0.400_{{-{spread:.3f}}}^{{+{spread:.3f}}}}}}}' in text
    assert rf'\MJZeroFourOneSixLensOnlyMedianResidual}}{{{{0.800_{{-{2 * spread:.3f}}}^{{+{2 * spread:.3f}}}}}}}' in text
    assert r'\MJZeroFourOneSixNLensOnlyRecovered}{3}' in text
    save('tables/runtime.json', {'total': 3661})
    assert r'\MJZeroFourOneSixRuntimeHours}{1.017}' in write_run_results(tmp_path).read_text()


def test_posterior_rms_and_position_coverage(tmp_path):
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'artifacts/summary.json').write_text('{}')
    store.write_table(tmp_path, 'posterior_image_rms', pd.DataFrame({'image_rms_arcsec': [.02, .07, .12, np.nan]}))
    store.write_table(tmp_path, 'position_coverage',
                      pd.DataFrame({'image_label': ['a', 'b', 'c'], 'credible_level': [.3, .9, np.nan]}))
    text = write_run_results(tmp_path, name='m0416').read_text()
    for suffix, value in [('NPosteriorRMSDraws', '3'),
                          ('PositionCoverageOneSigmaPercent', '50'), ('NPositionCoverage', '2'),
                          ('PositionCoverageTwoSigmaPercent', '100')]:
        assert rf'\MJZeroFourOneSix{suffix}}}{{{value}}}' in text


def test_field_parameter_medians(tmp_path):
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'artifacts/summary.json').write_text('{}')
    (tmp_path / 'artifacts/config.json').write_text(json.dumps({'likelihood': {'gp_reference_redshift': 2}}))
    store.write_table(tmp_path, 'posterior_summary', pd.DataFrame({'parameter': ['images.sigma_gp', 'images.lambda_b'],
                                                                   'median': [.06805, 14.1126], 'best_fit': [.05, 13.6],
                                                                   'p16': [.055, 13.4], 'p84': [.081, 14.6]}))
    text = write_run_results(tmp_path, name='m0416').read_text()
    assert r'\MJZeroFourOneSixSigmaGP}{0.068}' in text
    assert r'\MJZeroFourOneSixSigmaGPInterval}{{0.068_{-0.013}^{+0.013}}}' in text
    assert r'\MJZeroFourOneSixLambdaB}{14.1}' in text
    assert r'\MJZeroFourOneSixGRFReferenceRedshift}{2.0}' in text
    assert r'\MJZeroFourOneSixSigmaMeas}{\textnormal{not available}}' in text


def test_mock_export(tmp_path):
    campaign = SimpleNamespace(seeds=[1], summary=pd.DataFrame({'seed':[1,2], 'observed_image_count':[3,9], 'recovered_image_count':[2,8], 'missing_image_count':[1,1]}),
        residual_draws=pd.DataFrame({'seed':[1,1,1,1], 'draw_index':[0,0,1,1], 'image_residual_arcsec':[3,4,2,99], 'recovered':[True,True,True,False]}))
    text = write_mock_results(campaign, tmp_path, rhat_cut=1.01, min_ess=50).read_text()
    assert r'\MockNImages}{3}' in text
    assert r'\MockNExcluded}{1}' in text
    # Groups: draw 0 residuals (3, 4) -> RMS 3.536, median 3.5; draw 1 residual (2) -> RMS 2, median 2.
    for name, values in (('RMS', [np.sqrt(12.5), 2.0]), ('MedianResidual', [3.5, 2.0])):
        low, mid, high = np.percentile(values, [16, 50, 84])
        assert rf'\Mock{name}}}{{{{{mid:.3f}_{{-{mid - low:.3f}}}^{{+{high - mid:.3f}}}}}}}' in text
    assert r'\MockInjectedOffset}{\textnormal{not available}}' in text
    # With the mock catalog: one mock whose two images sit 0.03 and 0.04 arcsec from their true positions.
    mock = tmp_path / 'seed_00000001/mock'
    mock.mkdir(parents=True)
    (mock / 'mock_images.json').write_text(json.dumps([
        {'x_true_arcsec': 1.0, 'y_true_arcsec': 2.0, 'x_obs_arcsec': 1.03, 'y_obs_arcsec': 2.0},
        {'x_true_arcsec': -1.0, 'y_true_arcsec': 0.5, 'x_obs_arcsec': -1.0, 'y_obs_arcsec': 0.46}]))
    campaign.root = tmp_path
    text = write_mock_results(campaign, tmp_path, rhat_cut=1.01, min_ess=50).read_text()
    assert r'\MockInjectedOffset}{{0.035_{-0.000}^{+0.000}}}' in text
    assert r'\MockInjectedSigmaGRF}{\textnormal{not available}}' in text
    # With the truth file: the injected scales, their ranges and the member population.
    (mock / 'truth.json').write_text(json.dumps({
        'config': {'measurement_noise_arcsec': 0.0101, 'measurement_noise_range_arcsec': [0.005, 0.02],
                   'field_sigma_grf_range_arcsec': [0.04, 0.1], 'field_knee_range_arcsec': [3.0, 12.0],
                   'field_gamma_range': [3.0, 5.0], 'image_plane_sigma_range_arcsec': [0.02, 0.06],
                   'image_plane_ell_range_arcsec': [0.5, 2.0], 'field_reference_redshift': 2.0},
        'field': {'sigma_grf_arcsec': 0.0634, 'knee_arcsec': 5.46, 'gamma': 3.92},
        'image_plane_error': {'sigma_arcsec': 0.0381, 'ell_arcsec': 1.14},
        'images': [{'system_id': '1', 'family_id': '1A'}, {'system_id': '1', 'family_id': '1A'}, {'system_id': '1', 'family_id': '1B'}],
        'relation': {'schechter_alpha': -0.8, 'schechter_magnitude': 18.7, 'pivot_magnitude': 18.7, 'sigma_scatter_ln': 0.1}}))
    text = write_mock_results(campaign, tmp_path, rhat_cut=1.01, min_ess=50).read_text()
    for macro, value in (('InjectedSigmaGRF', '0.063'), ('InjectedSigmaGP', '0.038'), ('InjectedSigmaMeas', '0.010'),
                         ('InjectedLambdaB', '5.5'), ('InjectedGammaKappa', '3.9'), ('InjectedEllGP', '1.1'),
                         ('SigmaGRFMin', '0.04'), ('SigmaGRFMax', '0.10'), ('LambdaBMin', '3'), ('LambdaBMax', '12'),
                         ('EllGPMin', '0.5'), ('EllGPMax', '2.0'), ('SigmaMeasMin', '0.005'), ('SigmaMeasMax', '0.020'),
                         ('FieldReferenceRedshift', '2'), ('SchechterAlpha', '-0.8'), ('SchechterMagnitude', '18.7'),
                         ('VelocityScatterDex', '0.04'), ('VelocityScatterPercent', '10'),
                         ('NSystemsPerMock', '1'), ('NFamiliesPerMock', '2'), ('NImagesPerMock', '3'), ('NImagesPerMockMax', '3')):
        assert rf'\Mock{macro}}}{{{value}}}' in text, macro


def test_member_table_uses_joint_deviation_draws(tmp_path):
    from lumina.reporting.results_tex import m0416_table_metrics, interval
    final = tmp_path / 'final'
    (final / 'tables').mkdir(parents=True)
    (final / 'artifacts').mkdir()
    (tmp_path / 'init/artifacts').mkdir(parents=True)
    names = ['sigma', 'cutkpc', 'alpha_sigma', 'beta_radius']
    physical = np.array([[100., 20., .3, .6]]*3)
    state = {'kpc_per_arcsec': 2., 'parameter_specs': [{'name':'potfile_1.'+n} for n in names]}
    (tmp_path / 'init/artifacts/state.json').write_text(json.dumps(state))
    store.write_table(final, 'posterior_summary', pd.DataFrame({'parameter':['potfile_1.'+n for n in names], 'median': physical[0],
                                                                'p16':physical[0], 'p84':physical[0]}))
    save_arrays(final/'artifacts/posterior.h5',physical=physical)
    delta = np.tile([np.log(2.),np.log(3.)],(3,1,1))
    store.write_arrays(final, 'member_diagnostic_draws', deviations=delta)
    row = dict(member_id='potfile_1:8971',potfile_id='potfile_1',catalog_mag=21.,luminosity_ratio=1.)
    for prefix, value in [('velocity',np.log(2.)),('radius',np.log(3.))]:
        for suffix in ['p16','median','p84']: row[prefix+'_'+suffix]=value
    store.write_table(final, 'member_diagnostics', pd.DataFrame([row]))
    metrics = m0416_table_metrics(final)
    assert metrics['GalEightNineSevenOneSigma'] == interval(200,0,0,0)
    assert metrics['GalEightNineSevenOneCut'] == interval(30,0,0,1)
    assert metrics['ScalingSigma'] == interval(100*10**(.4*.3*(21-17.02)),0,0,0)
    assert not any(key.startswith('R') for key in metrics)


def test_truth_magnification_macros(tmp_path):
    (tmp_path / 'artifacts').mkdir()
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'artifacts/summary.json').write_text('{}')
    text = write_run_results(tmp_path, name='hera').read_text()
    for suffix in ('MagnificationScatterMuThreePercent', 'LensOnlyMagnificationBiasMuTenPercent', 'SourceAreaRatioMuThirty'):
        assert rf'\Hera{suffix}}}{{\textnormal{{not available}}}}' in text
    scatter = pd.DataFrame(dict(model=['parametric', 'parametric', 'model', 'model'], mu_ref=[3., 10., 3., 10.], bias=[-.18, -.11, -.08, -.1],
                      scatter=[.192, .394, .124, .263], n_pixels=100))
    store.write_table(tmp_path, 'magnification_scatter', scatter)
    mu = np.geomspace(2, 100, 50)
    store.write_table(tmp_path, 'cumulative_source_area', pd.DataFrame(dict(
        mu=mu, area_true_arcsec2=100 / mu**2, area_parametric_median_arcsec2=88 / mu**2, area_model_median_arcsec2=101 / mu**2)))
    text = write_run_results(tmp_path, name='hera').read_text()
    for suffix, value in [('MagnificationScatterMuThreePercent', '12'), ('MagnificationScatterMuTenPercent', '26'),
                          ('LensOnlyMagnificationScatterMuThreePercent', '19'), ('LensOnlyMagnificationScatterMuTenPercent', '39'),
                          ('MagnificationBiasMuThreePercent', '-8'), ('SourceAreaRatioMuTen', '1.01'), ('LensOnlySourceAreaRatioMuThirty', '0.88')]:
        assert rf'\Hera{suffix}}}{{{value}}}' in text
    # Without a field the tables carry the parametric lens only, which is then the model of each draw.
    store.write_table(tmp_path, 'magnification_scatter', scatter.query('model == "parametric"').reset_index(drop=True))
    assert r'\HeraMagnificationScatterMuTenPercent}{39}' in write_run_results(tmp_path, name='hera').read_text()


def test_error_budget_adds_the_scales_in_quadrature_per_draw(tmp_path):
    final = tmp_path / 'final'
    (final / 'artifacts').mkdir(parents=True)
    (tmp_path / 'init/artifacts').mkdir(parents=True)
    (final / 'artifacts/summary.json').write_text('{}')
    assert r'\HeraSigmaTotal}{\textnormal{not available}}' in write_run_results(final, name='hera').read_text()
    names = ['images.sigma_grf', 'images.sigma_gp', 'images.sigma_meas']
    (tmp_path / 'init/artifacts/state.json').write_text(json.dumps({'parameter_specs': [{'name': n} for n in names]}))
    save_arrays(final / 'artifacts/posterior.h5', physical=np.array([[.3, .4, 0.]] * 3 + [[.4, .3, 0.]] * 2))
    text = write_run_results(final, name='hera').read_text()
    assert r'\HeraSigmaTotal}{0.500}' in text and r'\HeraSigmaTotalInterval}{{0.500_{-0.000}^{+0.000}}}' in text
    assert r'\HeraGRFVariancePercent}{36}' in text
    (tmp_path / 'init/artifacts/state.json').write_text(json.dumps({'parameter_specs': [{'name': 'x'}, {'name': names[1]}, {'name': names[2]}]}))
    text = write_run_results(final, name='hera').read_text()
    assert r'\HeraSigmaTotal}{0.400}' in text and r'\HeraGRFVariancePercent}{\textnormal{not available}}' in text


def test_halo_table_macros_and_table(tmp_path):
    """A sampled parameter gives its prior and posterior, a fixed one its value; the cut radius is excess plus core."""
    from lumina.reporting.results_tex import halo_table_metrics, interval, write_halo_table
    final = tmp_path / 'final'
    (final / 'artifacts').mkdir(parents=True)
    (tmp_path / 'init/artifacts').mkdir(parents=True)
    specs = [{'name': '1.x_centre', 'prior': {'family': 'Uniform', 'log': False, 'low': -2.0, 'high': 4.0}},
             {'name': '1.core_radius_kpc', 'prior': {'family': 'Uniform', 'log': True, 'low': np.log(5.0), 'high': np.log(100.0)}},
             {'name': '1.cut_excess_kpc', 'prior': {'family': 'Normal', 'log': True, 'loc': np.log(40.0), 'scale': 0.5}},
             {'name': '1.v_disp', 'prior': {'family': 'TruncatedNormal', 'log': True, 'loc': np.log(800.0), 'scale': 0.25,
                                            'low': np.log(100.0), 'high': 'inf'}},
             {'name': 'shear.gamma1', 'prior': {'family': 'Uniform', 'log': False, 'low': -0.3, 'high': 0.3}}]
    (tmp_path / 'init/artifacts/state.json').write_text(json.dumps({'kpc_per_arcsec': 5.0, 'parameter_specs': specs}))
    halos = [{'id': '1', 'x_centre': 0.0, 'y_centre': 1.5, 'ellipticite': 0.0, 'angle_pos': 0.0, 'core_radius_kpc': 10.0,
              'cut_radius_kpc': 1000.0, 'v_disp': 800.0, 'profile_type': 81},
             {'id': '3', 'x_centre': 1.0, 'y_centre': 1.0, 'ellipticite': 0.2, 'angle_pos': 10.0, 'core_radius_kpc': 10.0,
              'cut_radius_kpc': 1000.0, 'v_disp': 300.0, 'profile_type': 81},
             {'id': 'shear', 'gamma': 0.0, 'angle_pos': 0.0, 'profile_type': 14}]
    (tmp_path / 'init/artifacts/config.json').write_text(json.dumps({'model': {'large_halos': halos}}))
    save_arrays(final / 'artifacts/posterior.h5', physical=np.tile([1.0, 20.0, 30.0, 900.0, 0.01], (5, 1)))
    metrics, rows = halo_table_metrics(final, 'toy')
    assert [key for _, _, key in rows] == ['CompOne' + s for s in ('X', 'Y', 'E', 'Core', 'Cut', 'Sigma')]   # halo 3 and the shear are fixed or not dPIE
    assert metrics['CompOneX'] == interval(1.0, 0, 0, 2) and metrics['CompOneXPrior'] == r'\mathcal{U}(-2,4)'
    assert metrics['CompOneY'] == '1.5' and metrics['CompOneYPrior'] == r'\textnormal{fixed}'
    assert metrics['CompOneCorePrior'] == r'\mathcal{LU}(5,100)'
    assert metrics['CompOneCut'] == interval(50.0, 0, 0, 0) and metrics['CompOneCutPrior'] == r'\mathcal{LN}(\ln 40,0.50)'
    assert metrics['CompOneSigmaPrior'] == r'\mathcal{LN}_{[100,\infty)}(\ln 800,0.25)'
    text = write_halo_table(tmp_path / 'toy_halo_table.tex', 'toy', 'Toy', rows).read_text()
    assert r'Halo 1 & $x$ (arcsec) & $\ToyCompOneXPrior$ & $\ToyCompOneX$ \\' in text and r'\label{tab:halos_toy}' in text


def test_macros_from_results_h5(tmp_path):
    """The macros of a run whose tables and records are in final/results.h5."""
    from lumina.io import store
    tables = {'posterior_image_residual_draws': pd.DataFrame({'draw_index': [0, 0, 1, 1], 'recovered': [True, False, True, True],
                                                              'family_id': ['1', '1', '1', '1'],
                                                              'image_residual_arcsec': [.3, np.nan, .4, .6]}),
              'posterior_image_rms': pd.DataFrame({'image_rms_arcsec': [.02, .07, np.nan]}),
              'position_coverage': pd.DataFrame({'image_label': ['a', 'b'], 'credible_level': [.3, .9]}),
              'posterior_summary': pd.DataFrame({'parameter': ['images.sigma_gp'], 'median': [.068], 'p16': [.055], 'p84': [.081]})}
    records = {'run_summary': {'fit': {'observed_image_count': 3, 'recovered_image_count': {'median': 2., 'q16': 1., 'q84': 3.}}},
               'runtime': {'total': 3661.}, 'aperture_mass_histogram': {'normalization_kpc': 2.0, 'samples': {
                   sample: dict(n=3, median=1.1, p25=.9, p75=1.3, iqr=.4, Q=1.2) for sample in ('all', 'image_hull')}}}
    final = tmp_path / 'final'
    (final / 'artifacts').mkdir(parents=True)
    (final / 'artifacts/summary.json').write_text(json.dumps({'images': 3, 'members': 8, 'exact_members': 2}))
    for name, frame in tables.items():
        store.write_table(final, name, frame)
    for name, record in records.items():
        store.write_record(final, name, record)
    text = write_run_results(final, name='hera').read_text()
    assert r'\HeraRuntimeHours}{1.017}' in text and r'\HeraNPosteriorRMSDraws}{2}' in text
    assert r'\HeraApertureMassRatioAllMedian}{1.100}' in text
    assert not (final / 'tables').exists()
