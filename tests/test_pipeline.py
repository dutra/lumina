"""End-to-end run on the synthetic toy cluster: the one test of the default suite that runs NUTS (a few draws)."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lumina.io import artifacts, store
from lumina import pipeline
from lumina.config import OutputsConfig, RuntimeConfig, SamplingConfig, SearchConfig
from lumina.likelihood.evaluator import Evaluator
from lumina.core.state import build_state


def _fast(config):
    return replace(config, runtime=RuntimeConfig(seed=3, chains=2, nuts_chain_method="sequential", resume=True),
                   search=SearchConfig(n_starts=2, member_starts=1, lbfgs_max_iter=30, polish_max_iter=20), sampling=SamplingConfig(warmup=10, samples=10, max_tree_depth=4),
                   outputs=OutputsConfig(draws=2))


@pytest.mark.slow                       # full pipeline run: about two minutes
def test_single_stage_end_to_end(synthetic_config, monkeypatch):
    """NUTS smoke test: two chains of a few draws through both stages, the saved artifacts, the adaptation
    diagnostics and resume."""
    config = _fast(synthetic_config)
    result = pipeline.run(config)
    state, evaluator, posterior = result.state, result.evaluator, result.posterior
    assert posterior.samples.shape == (20, state.n_parameters)
    assert np.isfinite(posterior.log_prob).all()

    # Adapted step sizes and energies, per chain, saved and reloaded.
    assert posterior.step_size.shape == (2,) and np.all(posterior.step_size > 0)
    assert posterior.energy.shape == (20,) and np.isfinite(posterior.energy).all()
    assert np.all(posterior.energy >= -posterior.log_prob - 1e-9)        # H = U + K with K >= 0
    adaptation = posterior.diagnostics["adaptation"]
    assert [c["chain"] for c in adaptation["per_chain"]] == [0, 1] and adaptation["max_tree_depth"] == 4
    np.testing.assert_allclose([c["step_size"] for c in adaptation["per_chain"]], posterior.step_size)
    reloaded = artifacts.load_posterior(result.paths.final.artifacts)
    np.testing.assert_array_equal(reloaded.step_size, posterior.step_size)
    np.testing.assert_array_equal(reloaded.energy, posterior.energy)
    assert reloaded.diagnostics["adaptation"]["per_chain"] == adaptation["per_chain"]
    stiffness = posterior.diagnostics["stiffness"]                      # one row per chain, from its own adapted mass
    assert [row["chain"] for row in stiffness] == [0, 1]
    assert all(np.isfinite(row["step_limit"]) and row["step_limit"] > 0 and row["direction"] for row in stiffness)
    assert posterior.diagnostics["start_check"]["warnings"] == []

    # Sources are the Gaussian centroid plus the whitened offsets, beta = c + L u.
    theta = jnp.asarray(posterior.samples[-1])
    stats = evaluator.statistics(theta)
    whitening = stats["whitening"]
    offsets = np.stack((posterior.samples[-1][state.images.source_x_param_index], posterior.samples[-1][state.images.source_y_param_index]), -1)
    expected = np.asarray(whitening["centroid"]) + np.einsum("fij,fj->fi", np.asarray(whitening["cholesky"]), offsets)
    np.testing.assert_allclose(np.asarray(stats["source_by_family"]), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(evaluator.source_positions(theta)), expected, rtol=1e-12, atol=1e-12)

    # The final classification was made at the best mode in the init stage and is the model the final stage sampled.
    init_dir = result.paths.init.artifacts
    saved = json.loads((init_dir / artifacts.CLASSIFICATION_FILE).read_text())
    assert saved["exact_count"] == len(state.members.exact_positions) == state.members.classification["exact_count"]
    assert set(np.asarray(artifacts.load_exact_positions(init_dir)).tolist()) == set(state.members.exact_positions.tolist())
    np.testing.assert_array_equal(artifacts.load_start(init_dir), result.init.theta)
    assert state.members.classification["exact_count"] + len(state.members.far_positions) == len(state.members.member_ids)

    # Audit and outputs exist.
    assert result.audit["approximation_validation"] in {"passed", "failed"}
    final = result.paths.final.root
    for name in ("posterior_summary", "member_classification", "image_fit_quality", "member_scaling_relation"):
        assert not store.read_table(final, name).empty
    assert "fit" in store.read_record(final, "run_summary")
    assert (final / store.RESULTS_FILE).is_file()
    # Every table and record is in results.h5: tables/ holds plain-text reports only, and no figure has a json file.
    assert {p.suffix for p in result.paths.final.tables.iterdir()} <= {".txt"}
    assert not list(result.paths.final.figures.glob("*.json"))
    assert (result.paths.final.figures / "critical_curves.pdf").is_file()
    assert (result.paths.init.tables / "classification_report.txt").is_file()
    assert {p.suffix for p in result.paths.init.tables.iterdir()} <= {".txt"}
    assert {name for name, record in store.read_record(final, "run_summary")["modules"].items()
            if record["status"] == "done"} == {"fit", "images", "mass"}

    # The outputs stage again on the saved run, fit module only (lumina outputs RUN --only fit): the
    # saved fit numbers stay in run_summary.
    from lumina.outputs.command import main as outputs_main
    fit = store.read_record(final, "run_summary")["fit"]
    assert outputs_main([str(result.paths.root), "--only", "fit", "--fail-fast"]) == 0
    rerun = store.read_record(final, "run_summary")
    assert json.dumps(rerun["fit"], sort_keys=True, default=float) == json.dumps(fit, sort_keys=True, default=float)
    assert rerun["modules"]["fit"]["status"] == "done" and "images" in rerun["modules"]

    # A full pass of the real modules, then every figure deleted and drawn again by a render pass that can evaluate
    # nothing: the same figures come back, and the numbers and macros stay as the full pass wrote them.
    # (The toy cluster has no truth maps and no imaging, so truth and rgb are inactive here.)
    assert outputs_main([str(result.paths.root), "--all"]) == 0
    full = store.read_record(final, "run_summary")
    done = {name for name, record in full["modules"].items() if record["status"] == "done"}
    assert done == {"fit", "images", "mass"}
    figures = result.paths.final.figures
    drawn = sorted(p.name for p in figures.glob("*.pdf"))
    macros = {p.name: p.read_text() for p in figures.glob("*_results.tex")}
    assert drawn and macros
    for path in figures.glob("*.pdf"):
        path.unlink()
    from lumina.outputs import saved_run
    from lumina.reporting import draws as draws_module

    def forbidden(*args, **kwargs):
        raise AssertionError("a render pass evaluated the lens")

    with monkeypatch.context() as patch:
        for target, name in ((Evaluator, "__init__"), (draws_module, "reporting_draws"),
                             (draws_module.ReportingDraws, "__init__"), (saved_run, "load_run")):
            patch.setattr(target, name, forbidden)
        assert outputs_main([str(result.paths.root), "--render"]) == 0
    rendered = store.read_record(final, "run_summary")
    assert {name for name, record in rendered["modules"].items() if record["status"] == "done"} == done
    assert all(rendered["modules"][name]["mode"] == "render" for name in done)
    assert sorted(p.name for p in figures.glob("*.pdf")) == drawn
    assert {p.name: p.read_text() for p in figures.glob("*_results.tex")} == macros
    numbers = lambda record: json.dumps({k: v for k, v in record.items() if k != "modules"}, sort_keys=True, default=float)
    assert numbers(rendered) == numbers(full)

    # Resuming reloads both stages without searching or sampling again (the outputs, written above, are not redrawn here).
    monkeypatch.setattr(pipeline, "write_outputs", lambda *args, **kwargs: {})
    again = pipeline.run(config)
    assert again.resumed and again.init.resumed
    np.testing.assert_array_equal(again.posterior.samples, posterior.samples)
    np.testing.assert_array_equal(again.init.theta, result.init.theta)
    # The init stage alone resumes too; a changed sampling setting under the same run name is refused
    # rather than silently overwriting the saved posterior.
    assert pipeline.run(config, stage="init").resumed
    # A resume with other outputs settings (outside the fit settings) saves them: the saved configuration describes the
    # products that pass rewrites, and the rest of it is left as it was.
    before = artifacts.load_config(result.paths.final.artifacts)
    fewer = replace(config, outputs=replace(config.outputs, draws=3, image_draws=3))
    assert pipeline.run(fewer).resumed
    after = artifacts.load_config(result.paths.final.artifacts)
    assert (after.outputs.draws, after.outputs.image_draws) == (3, 3)
    assert after.to_json_dict() | {"outputs": None} == before.to_json_dict() | {"outputs": None}
    assert pipeline.run(config).resumed and artifacts.load_config(result.paths.final.artifacts).outputs == config.outputs

    # Model comparison: the init stage writes BIC, the final stage copies it.  Turning on WBIC for the finished run adds
    # only the tempered NUTS run; the posterior is not sampled again.
    init_record = json.loads((init_dir / "model_comparison.json").read_text())
    assert np.isfinite(init_record["bic"]["bic"]) and init_record["bic"]["n"] == 2 * state.images.n_images
    final_file = result.paths.final.artifacts / "model_comparison.json"
    assert json.loads(final_file.read_text())["bic"] == init_record["bic"] and "wbic" not in json.loads(final_file.read_text())
    with_wbic = pipeline.run(replace(config, sampling=replace(config.sampling, wbic=True)))
    np.testing.assert_array_equal(with_wbic.posterior.samples, posterior.samples)
    record = json.loads(final_file.read_text())
    assert record["bic"] == init_record["bic"] and np.isfinite(record["wbic"]["wbic"]) and record["wbic"]["draws"] == 20
    assert record["wbic"]["beta"] == pytest.approx(1.0 / np.log(record["wbic"]["n"]))
    assert (result.paths.final.artifacts / "wbic_draws.h5").is_file()
    from lumina.inference.model_comparison import compare
    assert compare([result.paths.root])["delta_wbic"].tolist() == [0.0]
    table = (result.paths.final.tables / "run_summary.txt").read_text()
    assert "Run summary" in table and "recovered / observed" in table and "total" in table
    assert "step size per chain" in table and "E-BFMI per chain" in table and "tree-depth cap" in table
    changed = replace(config, sampling=SamplingConfig(warmup=10, samples=12, max_tree_depth=4))
    with pytest.raises(pipeline.SavedStageMismatch, match=r"saved final stage of this run folder was made with other fit settings(.|\n)*sampling\.samples: 10 -> 12"):
        pipeline.run(changed)


def test_far_field_barrier_is_finite_smooth_and_pushes_back(synthetic_config):
    from lumina.core.state import with_member_classes
    state = build_state(synthetic_config)
    state = with_member_classes(state, np.asarray([0, 1], dtype=np.int32), {"exact_count": 2, "mandatory_count": 2})   # two far-field members
    evaluator = Evaluator(state)
    theta = np.asarray(state.initial_theta, dtype=float)
    cut_index = next(i for i, s in enumerate(state.parameter_specs) if s.field == "cutkpc")

    def barrier(latent):
        physical = evaluator.physical(latent)
        return evaluator.far_field_log_barrier(evaluator.fields(physical)["far_cut_radii"])

    ratio0 = float(np.min(np.asarray(evaluator.far_min_separation) / np.asarray(evaluator.fields(evaluator.physical(jnp.asarray(theta)))["far_cut_radii"])))
    # Scale the relation's cut radius so the nearest far-field member sits at these multiples of the guard distance.
    targets = (3.0, 1.5, 1.1, 1.0, 0.9, 0.5)
    values, slopes = [], []
    for target in targets:
        trial = theta.copy()
        trial[cut_index] += np.log(ratio0 / (target * evaluator.guard_cut_radii))
        values.append(float(barrier(jnp.asarray(trial))))
        slopes.append(float(jax.grad(barrier)(jnp.asarray(trial))[cut_index]))
        value, grad = evaluator.value_and_grad(jnp.asarray(trial))
        assert np.isfinite(float(value)) and np.isfinite(np.asarray(grad)).all()
    assert all(v <= 0 for v in values) and values == sorted(values, reverse=True)      # never positive, worse as r_cut grows
    assert values[0] > -1e-3                                                            # negligible three guard distances out
    assert abs(values[3] + np.log(2.0) * evaluator.far_field_barrier_strength) < 1.0    # about -0.7 strength at the guard (one member there)
    assert slopes[-1] < 0 and values[-1] < -evaluator.far_field_barrier_strength        # steep inside, gradient points back out


def test_gp_image_plane_model(synthetic_config):
    """The gp likelihood: sigma_gp and sigma_meas are its error scales, and the sources are the family
    conditional."""
    from lumina.config import LikelihoodConfig
    state = build_state(replace(synthetic_config, likelihood=LikelihoodConfig(likelihood_mode="gp_image_plane")).validate())
    evaluator = Evaluator(state, "gp_image_plane")
    names = [s.name for s in state.parameter_specs]
    assert "images.sigma_gp" in names and "images.sigma_meas" in names
    assert not any(n in names for n in ("images.sigma_def", "images.sigma_sel", "images.sigma_img"))
    assert evaluator.gp_mode
    rng = np.random.default_rng(0)
    theta = np.asarray(state.initial_theta, dtype=float).copy()
    for index in (state.images.source_x_param_index, state.images.source_y_param_index):
        theta[index] = rng.normal(size=len(index))
    stats = evaluator.statistics(jnp.asarray(theta))
    offsets = np.stack((theta[state.images.source_x_param_index], theta[state.images.source_y_param_index]), -1)
    expected = np.asarray(stats["whitening"]["centroid"]) + np.einsum("fij,fj->fi", np.asarray(stats["whitening"]["cholesky"]), offsets)
    np.testing.assert_allclose(np.asarray(stats["source_by_family"]), expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(float(evaluator.loglike(jnp.asarray(theta))))


def test_a_saved_configuration_is_written_atomically(monkeypatch, synthetic_config, tmp_path):
    """config.json goes through a temporary file in its own folder and a rename; a failure before the rename leaves
    the old file whole and no temporary file behind."""
    import os
    config = synthetic_config
    tmp_path = tmp_path / "artifacts"
    artifacts.save_config(tmp_path, config)
    before = (tmp_path / artifacts.CONFIG_FILE).read_bytes()
    renames = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: renames.append((Path(src), Path(dst))) or real(src, dst))
    changed = replace(config, outputs=replace(config.outputs, draws=3, image_draws=3))
    artifacts.save_config(tmp_path, changed)
    (source, target), = renames
    assert source.parent == target.parent == tmp_path and target.name == artifacts.CONFIG_FILE and source != target
    assert artifacts.load_config(tmp_path).outputs.draws == 3
    written = (tmp_path / artifacts.CONFIG_FILE).read_bytes()

    def crash(src, dst):
        raise OSError("crashed before the rename")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="before the rename"):
        artifacts.save_config(tmp_path, config)
    assert (tmp_path / artifacts.CONFIG_FILE).read_bytes() == written != before
    assert sorted(p.name for p in tmp_path.iterdir()) == [artifacts.CONFIG_FILE]
