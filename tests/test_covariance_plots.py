from dataclasses import replace
import json
from kappaccino.io import store
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from kappaccino.plotting.covariance import render_covariance_plots, save_covariance_errors
from kappaccino.likelihood.evaluator import Evaluator
from kappaccino.core.state import build_state


@pytest.mark.parametrize('selection', [True, False])
def test_covariance_outputs_with_one_draw(synthetic_config, tmp_path, selection):
    config = replace(synthetic_config,
                     outputs=replace(synthetic_config.outputs, draws=1))
    if not selection:
        config = replace(config, likelihood=replace(config.likelihood, sigma_sel_prior=None))
    state = build_state(config)
    evaluator = Evaluator(state)
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(42).normal(0, .03, (40, state.n_parameters))
    posterior = SimpleNamespace(samples=samples, physical=np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples))),
                                log_prob=np.arange(40.), grouped_samples=samples.reshape(2, 20, -1))
    paths = SimpleNamespace(figures=tmp_path / 'figures', tables=tmp_path / 'tables')
    style = dict(plt.rcParams)
    from kappaccino.reporting.draws import reporting_draws
    save_covariance_errors(config, state, reporting_draws(config, evaluator, posterior), paths)
    outputs = render_covariance_plots(config, state, posterior, paths, log=lambda _: None)
    assert len(outputs) == 2 and all(p.is_file() for p in outputs)
    assert dict(plt.rcParams) == style
    metadata = store.read_record(paths.tables, 'covariance_errors_both_planes')
    assert len(metadata['posterior_draw_indices']) == 1
    assert metadata['plane'] == 'both'
    table = store.read_table(paths.tables, 'covariance_errors_both_planes')
    source = table[table.plane == 'source'].reset_index(drop=True)
    image = table[table.plane == 'image'].reset_index(drop=True)
    for component in ['measurement', 'selection', 'model', 'total']:
        np.testing.assert_allclose(image[component+'_arcsec'] * image.singular_value_median,
                                   source[component+'_arcsec'], atol=1e-14)
    np.testing.assert_allclose(table.total_p16_arcsec, table.total_p84_arcsec)
    if not selection:
        assert (table.selection_arcsec == 0).all()
    corner = store.read_record(paths.tables, 'covariance_corner')
    assert ('images.lambda_sel' in corner['parameters']) == selection


def test_normal_output_calls_covariance_suite(monkeypatch, synthetic_config, tmp_path):
    from kappaccino.plotting import covariance
    from kappaccino.reporting import outputs
    class ReachedCovariance(Exception):
        pass
    def reached(*args, **kwargs):
        raise ReachedCovariance
    monkeypatch.setattr(covariance, 'save_covariance_errors', reached)
    state = SimpleNamespace(parameter_specs=[])
    posterior = SimpleNamespace(samples=np.zeros((1, 1)), physical=np.ones((1, 1)), log_prob=np.zeros(1))
    evaluator = SimpleNamespace(physical=lambda p: p)
    paths = SimpleNamespace(tables=tmp_path/'tables', figures=tmp_path/'figures', root=tmp_path)
    from kappaccino.reporting import results_tex
    monkeypatch.setattr(results_tex, 'write_run_results', lambda *a, **k: None)
    summary = outputs.write_outputs(synthetic_config, state, evaluator, posterior, {}, paths, only=('fit',), log=lambda _: None)
    assert summary['modules']['fit']['steps']['covariance']['error'].startswith('ReachedCovariance')
