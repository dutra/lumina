from types import SimpleNamespace
import json
from kappaccino.config import ImagesOutputConfig, OutputsConfig
from kappaccino.io import store

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import numpyro.distributions as dist

from kappaccino.outputs import fit as posterior_module
from kappaccino.plotting import plots
from kappaccino.reporting import outputs, results_tex
from kappaccino.plotting import potfile
from kappaccino.core.model import ParameterSpec


def make_spec(name, field, prior, family="scaling"):
    return ParameterSpec(name=name, sample_name=name.replace(".", "_"), potential_id=name.split(".")[0], profile_type=0, field=field,
                         prior=prior, component_family=family)


def fixture_draws(population="potfile_members"):
    specs = [make_spec(f"{population}.sigma", "sigma", dist.LogNormal(4.6, .3)),
             make_spec(f"{population}.alpha_sigma", "alpha_sigma", dist.Normal(.25, .05))]
    rng = np.random.default_rng(42)
    samples = np.column_stack((np.exp(rng.normal(4.5, .2, 300)), rng.normal(.28, .04, 300)))
    return samples, specs


def test_run_outputs_create_potfile_corner(tmp_path, monkeypatch):
    samples, specs = fixture_draws()
    posterior = SimpleNamespace(samples=samples, physical=samples, log_prob=np.zeros(len(samples)))
    state = SimpleNamespace(parameter_specs=specs)
    config = SimpleNamespace(paths=SimpleNamespace(figure_prefix="hera"), runtime=SimpleNamespace(seed=0),
                             outputs=OutputsConfig(draws=2, image_draws=2, workers=1,
                                                   images=ImagesOutputConfig(extra_images=False)))
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures", root=tmp_path)
    monkeypatch.setattr(posterior_module, "posterior_summary_table", lambda *args: pd.DataFrame())
    monkeypatch.setattr(posterior_module, "classification_table", lambda *args: pd.DataFrame())
    monkeypatch.setattr(plots, "member_scaling_table", lambda *args: pd.DataFrame())
    monkeypatch.setattr(results_tex, "write_run_results", lambda *args, **kwargs: None)
    summary = outputs.write_outputs(config, state, SimpleNamespace(physical=lambda p: p), posterior, {}, paths,
                                    only=("fit",), skip=(), fail_fast=False, log=lambda _: None)
    assert summary["modules"]["fit"]["render_steps"]["potfiles"]["status"] == "done"
    for kind in ("corner",):
        path = paths.figures / f"hera_potfile_members_{kind}.pdf"
        assert path.read_bytes().startswith(b"%PDF")
        summary = store.read_figure_record(path)
        assert summary["population"] == "potfile_members"
        assert summary["draws"] == len(samples)
    assert not (paths.figures / 'hera_potfile_members_prior_posterior.pdf').exists()


def test_multiple_populations_and_isolated_style(tmp_path):
    samples, specs = fixture_draws("potfile_2")
    other, other_specs = fixture_draws("potfile_named")
    style = plt.rcParams.copy()
    results = potfile.write_potfile_plots(np.column_stack((samples, other)), specs + other_specs, tmp_path)
    assert {p.name for p in results} == {f"potfile_{pop}_{kind}.pdf" for pop in ("2", "named")
                                       for kind in ("corner",)}
    assert plt.rcParams["font.size"] == style["font.size"]
    assert plt.rcParams["mathtext.fontset"] == style["mathtext.fontset"]
    assert not plt.get_fignums()


def test_short_fixed_and_empty_posteriors(tmp_path):
    samples, specs = fixture_draws()
    assert potfile.write_potfile_plots(samples[:2], specs, tmp_path, log=lambda _: None) == []
    assert potfile.write_potfile_plots(np.empty((4, 0)), [], tmp_path) == []
    fixed = np.tile(samples[0], (10, 1))
    paths = potfile.write_potfile_plots(fixed, specs, tmp_path, log=lambda _: None)
    assert paths == []


def test_prior_quantiles_and_density_match_sampler():
    import jax
    from kappaccino.inference.sampling import SampleSite, distribution_for_site
    jax.config.update("jax_enable_x64", True)
    for prior in (dist.Normal(.5, .2), dist.TruncatedNormal(.5, .2, low=0., high=1.5), dist.Uniform(0., 1.5)):
        spec = make_spec("potfile_1.beta_radius", "beta_radius", prior)
        prior = potfile.saved_prior(spec)
        values = prior.ppf(np.linspace(.001, .999, 100))
        distribution = distribution_for_site(SampleSite("beta", (0,), False), [spec])
        np.testing.assert_allclose(prior.logpdf(values), distribution.log_prob(values), atol=2e-14, rtol=2e-14)


def test_corner_traces_preserve_chain_order(tmp_path):
    samples, specs = fixture_draws()
    grouped = samples.reshape(3, 100, 2)
    path = tmp_path / 'corner_with_traces.pdf'
    potfile.plot_potfile_corner(samples, specs, 'potfile_members', path, grouped_physical=grouped)
    metadata = store.read_figure_record(path)
    assert metadata['trace_chains'] == 3 and metadata['trace_draws_per_chain'] == 100
    with pytest.raises(ValueError, match='chain-ordered'):
        potfile.plot_potfile_corner(samples, specs, 'potfile_members', tmp_path/'bad.pdf', grouped_physical=grouped[::-1])
    plt.close('all')


def test_uniform_global_prior_and_coordinate_transforms(tmp_path):
    from kappaccino.plotting.corner import DisplayPrior, display_values
    from kappaccino.plotting.plots import plot_corner
    specs = [make_spec(f'halo.{field}', field, dist.Uniform(-1., 1.), 'large') for field in ('x', 'y')]
    draws = np.random.default_rng(4).uniform(-.8, .8, (100, 2))
    path = plot_corner(draws, specs, tmp_path/'global.pdf')
    assert path.is_file()
    assert 'support boundary' in store.read_figure_record(path)['uniform_joint_prior']
    spec = make_spec('halo.v', 'v', dist.LogNormal(0., 1.))
    prior = DisplayPrior(spec)
    np.testing.assert_allclose(prior.pdf(0.), np.log(10)/np.sqrt(2*np.pi))
    np.testing.assert_allclose(display_values(np.array([1., 10.]), spec), [0., 1.])


def test_corner_headers_physical_with_log_axes(tmp_path, monkeypatch):
    from kappaccino.plotting.corner import plot_parameter_corner
    from matplotlib.figure import Figure
    spec = make_spec('images.sigma_def', 'sigma_def', dist.LogNormal(np.log(.01), .5), 'error_scale')
    samples = np.geomspace(.005, .02, 101)[:,None]
    captured = []
    save = Figure.savefig
    def capture(fig, *args, **kwargs):
        captured.extend(text.get_text() for ax in fig.axes for text in ax.texts)
        captured.extend(ax.get_xlabel() for ax in fig.axes)
        return save(fig, *args, **kwargs)
    monkeypatch.setattr(Figure, 'savefig', capture)
    path = plot_parameter_corner(samples, [(0,spec,r'$\sigma_{\rm def}$')], tmp_path/'corner.pdf')
    assert any(r'\sigma_{\rm def}=0.010' in text for text in captured)
    assert any(r'\log(' in text for text in captured)
    metadata = store.read_figure_record(path)
    assert metadata['marginal_summary_coordinates'] == 'physical'
    assert metadata['coordinates'].startswith('log10')
