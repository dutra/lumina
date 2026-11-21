"""The relative-displacement product (``plotting.relative_displacement``): pair statistic, mode removal, skips of its
compute."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kappaccino.plotting import relative_displacement as rd


def _points(n=300, seed=0):
    return np.random.default_rng(seed).uniform(-40, 40, (n, 2))


def test_uniform_shift_gives_zero():
    xy = _points()
    geometry = rd.pair_geometry(xy)
    edges = rd.log_edges(geometry[2], geometry[2].min() * .999, geometry[2].max() * 1.001, 10)
    s_l, s_t, counts = rd.relative_displacement(np.tile([0.3, -0.7], (len(xy), 1)), geometry, edges)
    assert counts.sum() == len(geometry[0])
    np.testing.assert_allclose(s_l, 0, atol=1e-15)
    np.testing.assert_allclose(s_t, 0, atol=1e-15)


def test_independent_noise_gives_sqrt2_sigma():
    xy = _points(2000, 1)
    sigma = 0.15
    v = np.random.default_rng(2).normal(0, sigma, xy.shape)
    geometry = rd.pair_geometry(xy)
    edges = rd.log_edges(geometry[2], 3.0, 120, 5)
    s_l, s_t, counts = rd.relative_displacement(v, geometry, edges)
    assert (counts > 15000).all()
    np.testing.assert_allclose(s_l, np.sqrt(2) * sigma, rtol=0.03)
    np.testing.assert_allclose(s_t, np.sqrt(2) * sigma, rtol=0.03)


def test_longitudinal_identity():
    """S_L^2 = mean(v_L,i^2 + v_L,j^2) - 2 mean(v_L,i v_L,j) over each bin's pairs."""
    xy = _points(200, 3)
    v = np.random.default_rng(4).normal(0, 1, xy.shape) + 0.01 * xy
    geometry = rd.pair_geometry(xy)
    i, j, r, n, _ = geometry
    edges = rd.log_edges(r, 0.5, 120, 8)
    s_l, _, _ = rd.relative_displacement(v, geometry, edges)
    vi, vj = np.sum(v[i] * n, 1), np.sum(v[j] * n, 1)
    expected = rd.binned_mean(vi ** 2 + vj ** 2, r, edges) - 2 * rd.binned_mean(vi * vj, r, edges)
    np.testing.assert_allclose(s_l ** 2, expected, rtol=1e-10, atol=1e-12)


def test_nonfinite_vectors_leave_their_pairs_out():
    xy = _points(50, 5)
    v = np.random.default_rng(6).normal(0, 1, xy.shape)
    geometry = rd.pair_geometry(xy)
    edges = rd.log_edges(geometry[2], 0.5, 120, 4, min_pairs=1)
    full = rd.relative_displacement(v, geometry, edges)[2].sum()
    v[0] = np.nan
    s_l, s_t, counts = rd.relative_displacement(v, geometry, edges)
    assert counts.sum() == full - (len(xy) - 1)
    assert np.isfinite(s_l).all() and np.isfinite(s_t).all()


def test_log_edges_merge_until_min_pairs():
    r = np.r_[np.full(5, 1.5), np.full(40, 30.0), np.full(40, 80.0)]
    edges = rd.log_edges(r, 1.0, 100, 10, min_pairs=25)
    counts = np.histogram(r, edges)[0]
    assert (counts >= 25).all() and edges[0] == 1.0 and np.isclose(edges[-1], 100)


def test_mode_removal_removes_translation_convergence_shear():
    xy = _points(80, 7)
    modes = ("translation", "convergence", "shear")
    centre = xy.mean(0)
    u = (xy - centre) / 37.0
    pattern = (np.array([0.4, -1.1]) + 0.8 * u + 0.3 * np.column_stack([u[:, 0], -u[:, 1]])
               - 0.5 * np.column_stack([u[:, 1], u[:, 0]]))
    basis = rd.mode_basis(xy, modes)
    assert np.abs(rd.remove_modes(pattern, basis)).max() < 1e-12
    bumps = np.random.default_rng(8).normal(0, 0.1, xy.shape)
    kept = rd.remove_modes(pattern + bumps, basis)
    np.testing.assert_allclose(kept, rd.remove_modes(bumps, basis), atol=1e-12)
    assert rd.mode_basis(xy, ()) is None
    np.testing.assert_array_equal(rd.remove_modes(bumps, None), bumps)


def test_gp_structure_rises_to_plateau():
    r = np.array([0.0, 1e-3, 100.0])
    s = rd.gp_structure(r, 0.15, 1.2)
    assert s[0] == 0 and s[1] < 1e-3 and np.isclose(s[2], np.sqrt(2) * 0.15)
    np.testing.assert_array_equal(rd.gp_structure(r, 0.0, np.nan), 0)


def test_compute_skips_without_truth_map(tmp_path):
    lines = []
    config = SimpleNamespace(outputs=SimpleNamespace(truth=SimpleNamespace(deflection_true_fits=None)), paths=SimpleNamespace(figure_prefix="x"))
    recovery = {"image_fit_quality": pd.DataFrame({"x_obs_arcsec": [0.0]}),
                "posterior_image_residual_draws": pd.DataFrame({"posterior_index": [0]})}
    out = rd.save_relative_displacement(config, None, None, SimpleNamespace(has_field=True), recovery, {"amplitude": 1.0},
                                        tmp_path, log=lines.append)
    assert out is None
    assert lines == ["[outputs] relative displacement skipped: no outputs.truth.deflection_true_fits"]
    assert not (tmp_path / "results.h5").exists() and not (tmp_path / "figures").exists()


def test_compute_skips_without_true_lens_field_or_images(tmp_path):
    lines = []
    config = SimpleNamespace(outputs=SimpleNamespace(truth=SimpleNamespace(deflection_true_fits="map.fits")), paths=SimpleNamespace(figure_prefix="x"))
    recovery = {"image_fit_quality": pd.DataFrame({"x_obs_arcsec": [0.0]}),
                "posterior_image_residual_draws": pd.DataFrame({"posterior_index": [0]})}
    assert rd.save_relative_displacement(config, None, None, SimpleNamespace(has_field=True), recovery, None, tmp_path,
                                         log=lines.append) is None
    assert rd.save_relative_displacement(config, None, None, SimpleNamespace(has_field=False), recovery, {"amplitude": 1.0},
                                         tmp_path, log=lines.append) is None
    empty = {**recovery, "posterior_image_residual_draws": pd.DataFrame()}
    assert rd.save_relative_displacement(config, None, None, SimpleNamespace(has_field=True), empty, {"amplitude": 1.0},
                                         tmp_path, log=lines.append) is None
    assert lines == ["[outputs] relative displacement skipped: no true-lens record",
                     "[outputs] relative displacement skipped: no shared mass field",
                     "[outputs] relative displacement skipped: no image table"]
    assert not (tmp_path / "results.h5").exists()


def test_an_unreadable_map_fails_the_compute(tmp_path):
    from kappaccino.io import store
    lines = []
    store.write_table(tmp_path, "true_lens_residuals", pd.DataFrame({"status": ["used"]}))
    config = SimpleNamespace(outputs=SimpleNamespace(truth=SimpleNamespace(deflection_true_fits=str(tmp_path / "missing.fits"))),
                             paths=SimpleNamespace(figure_prefix="x"))
    state = SimpleNamespace(cosmo_config={"H0": 70.0, "Om0": 0.3}, gp_reference_redshift=2.0)
    recovery = {"image_fit_quality": pd.DataFrame({"x_obs_arcsec": [0.0]}),
                "posterior_image_residual_draws": pd.DataFrame({"posterior_index": [0]})}
    # A configured map that cannot be read is an error (a failed step), never a quiet skip.
    with pytest.raises(OSError):
        rd.save_relative_displacement(config, state, None, SimpleNamespace(has_field=True), recovery, {"amplitude": 1.0},
                                      tmp_path, log=lines.append)
    assert lines == [] and not store.has_table(tmp_path, "relative_displacement")


def test_mode_removal_ignores_nonfinite_points():
    xy = _points(60, 9)
    basis = rd.mode_basis(xy, ("translation", "convergence", "shear"))
    v = np.tile([0.2, -0.3], (len(xy), 1))
    v[5] = np.nan
    out = rd.remove_modes(v, basis)
    assert np.isnan(out[5]).all()
    assert np.abs(np.delete(out, 5, axis=0)).max() < 1e-12


def test_grid_over_the_map_within_budget():
    lens = SimpleNamespace(origin=(-100.0, -50.0), step=0.1, shape=(1000, 2000))     # map x from -100 to 99.9
    extent = rd.map_extent(lens, margin=2)
    np.testing.assert_allclose(extent, (-99.7, 99.8, -49.8, 49.7))                    # model x = -map x
    step = rd.grid_step(extent, budget=60_000)
    assert step == 1.0                                                                 # 0.5 would give 80k points
    mesh = rd.map_grid(extent, step)
    assert mesh.shape[0] * mesh.shape[1] <= 60_000 and mesh.shape[-1] == 2
    assert mesh[..., 0].min() >= extent[0] and mesh[..., 0].max() <= extent[1]
    assert mesh[..., 1].min() >= extent[2] and mesh[..., 1].max() <= extent[3]
    np.testing.assert_allclose(np.diff(mesh[0, :, 0]), step)



# ----------------------------------------------------------------------------- the truth module's steps
def synthetic_table():
    """A table with every curve the figure draws: two draws, image and grid panels, the GP and the scatter."""
    rng = np.random.default_rng(10)
    image_edges, grid_edges = np.geomspace(0.5, 80, 6), np.geomspace(1.0, 200, 8)
    rows = []
    for k in range(2):
        for panel, edges in (("images", image_edges), ("grid", grid_edges)):
            n = len(edges) - 1
            for curve in ("truth", "mass_field"):
                rows += rd._rows(panel, curve, k, 10 + k, edges, rng.uniform(.1, .5, n), rng.uniform(.1, .5, n), np.full(n, 30))
        n = len(image_edges) - 1
        rows += rd._rows("images", "gp", k, 10 + k, image_edges, rng.uniform(.05, .2, n), rng.uniform(.05, .2, n), np.full(n, 30))
        axis = np.geomspace(rd.ANALYTIC_START_ARCSEC, grid_edges[-1], 20)
        value = rd.gp_structure(axis, 0.1, 2.0)
        rows += [dict(panel="grid", curve="gp_analytic", draw=k, posterior_row=10 + k, bin=b, r_lo=np.nan, r_hi=np.nan,
                      r_centre=float(r), n_pairs=0, S_L_arcsec=float(v), S_T_arcsec=float(v))
                 for b, (r, v) in enumerate(zip(axis, value))]
    n = len(image_edges) - 1
    rows += rd._rows("images", "image_position_scatter", -1, -1, image_edges, rng.uniform(.05, .3, n),
                     rng.uniform(.05, .3, n), np.full(n, 30))
    return pd.DataFrame(rows)


class TruthSetup:
    """A stand-in ``images`` module (its tables and the true-lens record in the store), fake draws, the real
    ``truth`` module with the relative-displacement table and the true map faked, and passes."""

    def __init__(self, monkeypatch, tmp_path):
        from kappaccino.diagnostics import truth_recovery
        from kappaccino.io import store
        from kappaccino.outputs.base import OutputModule
        from kappaccino.reporting import draws as draws_module
        self.monkeypatch, self.tmp_path, self.lines = monkeypatch, tmp_path, []
        self.kappa, self.deflection, self.has_field, self.true_lens, self.empty = None, "defl.fits", True, True, False
        self.mode, self.computed, self.rendered = "full", [], []
        setup = self

        class FakeDraws(list):
            seed, cache_stage, field_is_rebuilt, key = 7, None, False, "draws-key"
            image_indices = indices = list(range(2))

            @property
            def has_field(self):
                return setup.has_field

        class Images(OutputModule):
            name, compute_steps = "images", ("recovery", "true_lens")

            def compute(self, ctx):
                def recovery():
                    ctx.draws
                    image_df = pd.DataFrame({"x_obs_arcsec": [] if setup.empty else [1.0, -1.0]})
                    residuals = pd.DataFrame({"posterior_index": [] if setup.empty else [0, 0]})
                    with store.batch(ctx.stage):
                        store.write_table(ctx.stage, "image_fit_quality", image_df)
                        store.write_table(ctx.stage, "posterior_image_residual_draws", residuals)
                    return "recovery"

                def true_lens():
                    if not setup.true_lens:
                        return None
                    with store.batch(ctx.stage):
                        store.write_table(ctx.stage, "true_lens_residuals", pd.DataFrame({"status": ["used", "used"]}))
                        store.write_record(ctx.stage, "true_lens_rms", {"amplitude": 0.98})
                    return "true_lens"
                ctx.step("recovery", recovery)
                ctx.step("true_lens", true_lens)

            def product(self, ctx):
                recovery = {name: store.read_table(ctx.stage, name)
                            for name in ("image_fit_quality", "posterior_image_residual_draws")}
                true_lens = store.read_record(ctx.stage, "true_lens_rms") if ctx.computed("true_lens") else None
                return {"image_df": recovery["image_fit_quality"], "recovery": recovery, "true_lens": true_lens}

        def draws_factory(*args):
            if setup.mode == "render":
                raise AssertionError("a render pass built the reporting draws")
            return FakeDraws(range(2))

        def table(state, evaluator, draws, residual_draws, true_table, alpha_true, lens_map, *, log=print):
            if setup.mode == "render":
                raise AssertionError("a render pass computed the relative displacement")
            setup.computed.append(len(residual_draws))
            return synthetic_table(), dict(n_image_points=2, n_grid_points=100, grid_step_arcsec=1.0, n_grid_pairs=4000,
                                           modes=["translation"])

        render = rd.render_relative_displacement

        def spy(table, path):
            setup.rendered.append(table)
            return render(table, path)

        self.images = Images()
        monkeypatch.setattr(draws_module, "reporting_draws", draws_factory)
        monkeypatch.setattr(rd, "relative_displacement_table", table)
        monkeypatch.setattr(rd, "true_deflection", lambda path, cosmology, z, amplitude: (None, None, 1.5))
        monkeypatch.setattr(rd, "render_relative_displacement", spy)
        monkeypatch.setattr(truth_recovery, "compute_truth_recovery", lambda *a, **k: None)

    def context(self, mode="full"):
        from kappaccino.outputs import OutputContext
        self.mode = mode
        config = SimpleNamespace(outputs=SimpleNamespace(truth=SimpleNamespace(kappa_true_fits=self.kappa, deflection_true_fits=self.deflection)),
                                 paths=SimpleNamespace(figure_prefix="toy"))
        state = SimpleNamespace(cosmo_config={"H0": 70.0, "Om0": 0.3}, gp_reference_redshift=2.0)
        paths = SimpleNamespace(tables=self.tmp_path / "tables", figures=self.tmp_path / "figures",
                                fits=self.tmp_path / "fits", root=self.tmp_path)
        return OutputContext(config, state, None, None, {}, paths, log=self.lines.append, mode=mode)

    def run(self, mode="full"):
        from kappaccino.outputs import run_modules
        from kappaccino.outputs.truth import TruthModule
        ctx = self.context(mode)
        return ctx, run_modules(ctx, [self.images, TruthModule()])

    @property
    def figure(self):
        return self.tmp_path / "figures" / "toy_relative_displacement.pdf"


@pytest.fixture
def truth_setup(monkeypatch, tmp_path):
    return TruthSetup(monkeypatch, tmp_path)


def test_a_deflection_map_alone_activates_truth_and_the_render_pass_redraws_the_figure(truth_setup):
    from kappaccino.io import store
    from kappaccino.outputs.base import ComputeOnly
    s = truth_setup
    ctx, records = s.run()
    truth = records["truth"]
    assert truth["status"] == "done" and s.computed == [2]
    assert truth["steps"]["relative_displacement"]["wrote"] is True and truth["steps"]["truth_maps"]["wrote"] is False
    assert list(truth["render_steps"]) == ["relative_displacement"] and s.figure.exists()
    saved = store.read_table(s.tmp_path, "relative_displacement")
    pd.testing.assert_frame_equal(saved, synthetic_table(), check_exact=True)
    record = store.read_record(s.tmp_path, "relative_displacement")
    assert record["amplitude"] == 0.98 and record["scale_to_reference"] == 1.5 and record["deflection_fits"] == "defl.fits"
    assert any(line.startswith("[outputs] relative displacement: 2 draws, 2 images, 100 grid points") for line in s.lines)
    # Render only: the figure comes back from the saved table, without the draws, the evaluator or the computation.
    s.figure.unlink()
    ctx, records = s.run("render")
    assert records["truth"]["status"] == "done" and records["truth"]["mode"] == "render" and s.figure.exists()
    assert list(records["truth"]["render_steps"]) == ["relative_displacement"] and s.computed == [2]
    assert len(s.rendered) == 2
    pd.testing.assert_frame_equal(s.rendered[1], s.rendered[0], check_exact=True)
    for what in ("draws", "evaluator"):
        with pytest.raises(ComputeOnly):
            getattr(ctx, what)


@pytest.mark.parametrize("change, reason", [
    (dict(deflection=None, kappa="kappa.fits"), "no outputs.truth.deflection_true_fits"),
    (dict(true_lens=False), "no true-lens record"),
    (dict(has_field=False), "no shared mass field"),
    (dict(empty=True), "no image table"),
])
def test_a_skipped_compute_draws_no_figure_and_never_a_stale_table(truth_setup, change, reason):
    from kappaccino.io import store
    s = truth_setup
    s.run()
    assert s.figure.exists()
    s.figure.unlink()
    s.lines.clear()
    for key, value in change.items():
        setattr(s, key, value)
    _, records = s.run()
    truth = records["truth"]
    assert truth["status"] == "done" and truth["steps"]["relative_displacement"]["wrote"] is False
    assert f"[outputs] relative displacement skipped: {reason}" in s.lines
    assert store.has_table(s.tmp_path, "relative_displacement")          # the older table stays in the store
    assert "relative_displacement" not in truth.get("render_steps", {}) and not s.figure.exists()
    _, records = s.run("render")
    assert "relative_displacement" not in records["truth"].get("render_steps", {}) and not s.figure.exists()
    assert len(s.rendered) == 1


def test_truth_is_inactive_without_either_map(truth_setup):
    s = truth_setup
    s.deflection = None
    _, records = s.run()
    assert records["truth"] == {"status": "inactive", "seconds": 0.0, "reason": (
        "no truth convergence or deflection map (outputs.truth.kappa_true_fits, outputs.truth.deflection_true_fits)")}


def test_the_figure_survives_failed_truth_maps_in_the_full_pass_and_a_later_render(truth_setup):
    from kappaccino.diagnostics import truth_recovery
    s = truth_setup
    s.kappa = "kappa.fits"

    def broken(*a, **k):
        raise OSError("kappa.fits is not a FITS file")
    s.monkeypatch.setattr(truth_recovery, "compute_truth_recovery", broken)
    _, records = s.run()
    truth = records["truth"]
    # The module is failed without a product, but the figure of the step that succeeded is drawn.
    assert truth["status"] == "failed" and truth["error"] == "ModuleUnavailable: no truth maps (the truth_maps step failed)"
    assert truth["steps"]["relative_displacement"]["wrote"] is True and truth["steps"]["truth_maps"]["status"] == "failed"
    assert truth["render_steps"]["relative_displacement"]["status"] == "done" and s.figure.exists()
    # The figures of the truth maps are reported as not drawn, with the reason.
    assert {step: record["reason"] for step, record in truth["render_steps"].items() if record["status"] == "skipped"} == {
        "magnification_recovery": "compute step magnification_recovery skipped",
        "cumulative_source_area": "compute step cumulative_source_area skipped",
        "truth_residuals": "compute step truth_maps failed", "mass_profile": "compute step truth_maps failed",
        "kappa_mass_residuals": "compute step truth_maps failed"}
    # A later render-only pass draws it again from the saved table, and truth stays failed.
    s.figure.unlink()
    _, records = s.run("render")
    truth = records["truth"]
    assert truth["status"] == "failed" and truth["mode"] == "render" and s.figure.exists()
    assert truth["render_steps"]["relative_displacement"]["status"] == "done" and len(s.rendered) == 2
    assert "gave no product" in truth["error"] and "the truth_maps step failed" in truth["error"]
