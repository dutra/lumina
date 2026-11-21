"""The fit output module split into compute and render: a full pass, then a render pass that evaluates nothing and
draws every figure again from the store and the posterior, for runs with and without a GP, a knee mass field and a
shared GRF, members and recorded chains."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumina.core.state import build_state
from lumina.io import store
from lumina.likelihood.evaluator import Evaluator
from lumina.outputs.base import OutputContext, completion_name, run_modules
from lumina.outputs.fit import FitModule


def likelihood(config, kind):
    """The likelihood of each case: the default, an image-plane GP, a knee mass field shared by every image, and a
    knee GP without a mass field (no shared GRF)."""
    settings = {"smoothed": {},
                "gp": dict(likelihood_mode="gp_image_plane"),
                "knee": dict(likelihood_mode="gp_image_plane", gp_kernel="knee", gp_blocks="all"),
                "knee_no_grf": dict(likelihood_mode="gp_image_plane", gp_kernel="knee", gp_blocks="all", sigma_grf_prior=None)}[kind]
    return replace(config, likelihood=replace(config.likelihood, **settings),
                   outputs=replace(config.outputs, draws=2, image_draws=2)).validate()


def posterior_for(state, evaluator, *, chains=True, n=16):
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(7).normal(0, .005, (n, state.n_parameters))
    physical = np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples)))
    return SimpleNamespace(samples=samples, physical=physical, log_prob=np.arange(n, dtype=float), diagnostics={},
                           grouped_samples=samples.reshape(2, n // 2, -1) if chains else None)


def fit_pass(config, state, evaluator, posterior, root, mode):
    paths = SimpleNamespace(tables=root / "tables", figures=root / "figures", fits=root / "fits", root=root)
    ctx = OutputContext(config, state, evaluator, posterior, {"far_field": "ok"}, paths, log=lambda _: None, mode=mode)
    return run_modules(ctx, [FitModule()])["fit"], paths


def no_evaluation(monkeypatch):
    """Make any lens evaluation fail: what a render pass must never need."""
    from lumina.reporting import draws as draws_module

    def refuse(*args, **kwargs):
        raise AssertionError("a render pass evaluated the lens")
    monkeypatch.setattr(Evaluator, "__init__", refuse)
    monkeypatch.setattr(draws_module, "reporting_draws", refuse)
    monkeypatch.setattr(draws_module.ReportingDraws, "__init__", refuse)


CASES = {
    # kind: (members, chains, the GP figures, whether the shared-field maps are drawn)
    "smoothed": (True, True, (), False),
    "gp": (True, True, ("gp_kernel_sky", "gp_shared_vs_white", "gp_scale_posteriors"), False),
    "knee": (True, True, ("gp_kernel_curves", "gp_scale_posteriors"), True),
    "knee_no_grf": (False, False, ("gp_kernel_curves", "gp_scale_posteriors"), False),
}
FIELD_FIGURES = ("total_grf_dimensionless_power", "gp_correction_correlation")
MEMBER_FIGURES = ("member_deviations", "member_constraints", "whitened_image_residuals", "far_field_accuracy")


@pytest.mark.parametrize("kind", list(CASES))
def test_a_render_pass_draws_every_fit_figure_again_without_evaluating(kind, synthetic_config, tmp_path, monkeypatch):
    members, chains, gp_figures, field_maps = CASES[kind]
    config = likelihood(synthetic_config, kind)
    if not members:
        config = replace(config, model=replace(config.model, member_populations=()))
    state = build_state(config)
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    posterior = posterior_for(state, evaluator, chains=chains)
    root = tmp_path / "final"
    # A stale shared-field entry from some older compute: never drawn when this compute did not write the maps.
    store.write_arrays(root, "conditional_field_maps", {"x": np.zeros(2)})

    record, paths = fit_pass(config, state, evaluator, posterior, root, "full")
    assert record["status"] == "done", record
    completion = store.read_record(root, completion_name("fit"))
    assert completion["complete"] and set(completion["steps"]) == {
        "tables", "chain_traces", "audit", "scaling_table", "covariance", "fields", "members"}
    wrote = {step: r["wrote"] for step, r in completion["steps"].items()}
    assert wrote["chain_traces"] is chains and wrote["members"] is members and wrote["fields"] is bool(gp_figures)
    assert store.read_record(root, "gp_diagnostics")["field_maps"] is field_maps if gp_figures else True
    # What other modules and the macros read is written by compute.
    for table in ("posterior_summary", "member_classification", "member_scaling_relation", "covariance_errors_both_planes"):
        assert store.has_table(root, table)
    assert store.has_table(root, "member_diagnostics") is members

    def stems(folder):         # without the figure prefix (the covariance figures use "cluster" when there is none)
        prefix = config.paths.figure_prefix
        return {path.stem.removeprefix(f"{prefix}_" if prefix else "cluster_") for path in folder.glob("*.pdf")}
    expected = {"corner_globals", "covariance_corner", "covariance_errors_both_planes", *gp_figures}
    expected |= set(FIELD_FIGURES) if field_maps else set()
    expected |= set(MEMBER_FIGURES) | {"scaling_relation"} if members else set()
    expected |= {"chain_traces"} if chains else set()
    figures = stems(paths.figures)
    assert expected <= figures and not (set(FIELD_FIGURES) - expected) & figures
    assert not set(MEMBER_FIGURES) & figures if not members else True

    for path in paths.figures.glob("*.pdf"):
        path.unlink()
    no_evaluation(monkeypatch)
    again, _ = fit_pass(config, state, None, posterior, root, "render")
    assert again["status"] == "done", again
    assert stems(paths.figures) == figures
    left_out = {"chain_traces"} if not chains else set()
    left_out |= {"members"} if not members else set()          # scaling_plot runs and draws nothing for an empty table
    left_out |= {"fields"} if not gp_figures else set()
    assert set(FitModule.render_steps) - set(again["render_steps"]) == left_out


def test_render_reads_what_compute_saved(synthetic_config, tmp_path):
    """The covariance figure is drawn from its table alone and the member figures from the saved diagnostics, which
    come back as they were computed."""
    from lumina.plotting.covariance import render_covariance_errors, save_covariance_errors
    from lumina.plotting.members import load_member_diagnostics, save_member_diagnostics
    from lumina.diagnostics.members import compute_member_diagnostics
    from lumina.reporting.draws import reporting_draws
    config = likelihood(synthetic_config, "smoothed")
    state = build_state(config)
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    posterior = posterior_for(state, evaluator)
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures")
    assert save_covariance_errors(config, state, reporting_draws(config, evaluator, posterior), paths) == "covariance_errors_both_planes"
    table = store.read_table(paths.tables, "covariance_errors_both_planes")
    assert table.groupby(["plane", "direction"]).image_label.apply(list).map(lambda labels: labels == list(state.images.image_labels)).all()
    assert render_covariance_errors(config, paths).is_file()

    assert save_member_diagnostics(config, state, evaluator, posterior, paths) == "member_diagnostics"
    saved = load_member_diagnostics(paths.tables)
    fresh = compute_member_diagnostics(state, evaluator, posterior, config)
    assert list(saved.variables) == list(fresh.variables) and saved.variables == fresh.variables
    np.testing.assert_array_equal(saved.deviations, fresh.deviations)
    np.testing.assert_array_equal(saved.prior_deviations, fresh.prior_deviations)
    assert set(saved.fields) == set(fresh.fields)
    for key, value in fresh.fields.items():
        np.testing.assert_array_equal(saved.fields[key], value)
        assert saved.fields[key].dtype == np.asarray(value).dtype
    assert saved.members.equals(fresh.members) and saved.images.equals(fresh.images)
    assert {"failed_guard_members", "failed_error_members", "failed_error_images"} <= set(saved.provenance)
