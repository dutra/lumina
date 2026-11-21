"""Check that the normal GP output path includes both shared-field figures."""
from types import SimpleNamespace

import pytest
from unittest.mock import Mock

from kappaccino.plotting import gp, fields


def test_standard_gp_outputs_include_field_figures(monkeypatch, tmp_path):
    import numpy as np
    monkeypatch.setattr(gp, 'apply_style', lambda: None)
    monkeypatch.setattr(gp, 'knee_kernel_curves', lambda *args: {'r': np.arange(3.)})
    monkeypatch.setattr(gp, 'plot_knee_kernel', lambda *args: args[-1])
    monkeypatch.setattr(gp, 'plot_scale_posteriors', lambda *args: args[-2])
    expected = [tmp_path / 'total_grf_dimensionless_power.pdf', tmp_path / 'gp_correction_correlation.pdf']
    saver = Mock(return_value='conditional_field_maps')
    renderer = Mock(return_value=expected)
    monkeypatch.setattr(fields, 'save_field_maps', saver)
    monkeypatch.setattr(fields, 'render_field_plots', renderer)
    config = SimpleNamespace(paths=SimpleNamespace(figure_prefix='test'))
    evaluator = SimpleNamespace(gp_mode=True, gp_knee=True)
    paths = SimpleNamespace(figures=tmp_path, tables=tmp_path / 'tables')
    assert gp.save_gp_diagnostics(config, None, evaluator, None, paths, log=lambda _: None) == gp.GP_DIAGNOSTICS
    outputs = gp.render_gp_diagnostics(config, None, None, paths, log=lambda _: None)
    assert outputs[-2:] == expected
    saver.assert_called_once()
    renderer.assert_called_once()


def test_no_mass_field_skips_the_shared_maps():
    config = SimpleNamespace(likelihood=SimpleNamespace(gp_blocks='all'))
    evaluator = SimpleNamespace(gp_knee=True, sigma_grf_index=-1, gp_knee_spectrum='knee', state=SimpleNamespace(gp_blocks='all'))
    messages = []
    assert fields.save_field_maps(config, None, evaluator, None, None, log=messages.append) is None
    assert 'skipped' in messages[0]


def test_system_blocks_render_through_the_shared_twin(monkeypatch):
    """A system-block run's maps use the reporting draws, converted once to its all-block twin's coordinates."""
    import numpy as np
    from kappaccino.reporting import draws as reporting
    twin = SimpleNamespace(gp_knee=True, sigma_grf_index=0, gp_knee_spectrum='knee', state=SimpleNamespace(gp_blocks='all'))
    run = SimpleNamespace(gp_knee=True, sigma_grf_index=0, gp_knee_spectrum='knee', state=SimpleNamespace(gp_blocks='system'),
                          _shared_field_twin=twin)
    monkeypatch.setattr(reporting, 'shared_theta', lambda evaluator, theta: theta + 100.)
    seen = []
    def compute(config, state, draws, *, log):
        seen.append(draws)
        raise RuntimeError('stop after the swap')
    monkeypatch.setattr(fields, 'compute_fields', compute)
    config = SimpleNamespace(likelihood=SimpleNamespace(gp_blocks='system'), paths=SimpleNamespace(figure_prefix='x'),
                             reporting=SimpleNamespace(draws=8), runtime=SimpleNamespace(seed=3))
    posterior = SimpleNamespace(samples=np.arange(20.)[:, None], physical=np.arange(20.)[:, None])
    messages = []
    with pytest.raises(RuntimeError):
        fields.save_field_maps(config, None, run, posterior, None, log=messages.append)
    draws = seen[0]
    assert draws.evaluator is run and len(draws) == 8
    evaluator, samples = draws.shared()
    assert evaluator is twin
    np.testing.assert_array_equal(samples, draws.samples + 100.)
    assert 'twin' in messages[0]
