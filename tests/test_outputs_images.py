"""The ``images`` output module split into compute, product and render: the product read back from the store, the
figures drawn from it alone, the failed recovery, the true-lens record and the fit numbers of a render pass."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kappaccino.io import store
from kappaccino.outputs import OutputContext, OutputModule, run_modules
from kappaccino.outputs.images import ImagesModule


def recovery_tables(empty=False):
    """Stand-ins for ``image_recovery_tables`` with the column kinds the real tables have (strings, floats with NaN,
    ints, bools), or with no observed image."""
    if empty:
        draws = pd.DataFrame(columns=["draw_index", "family_id", "image_label", "recovered", "x_obs_arcsec", "y_obs_arcsec",
                                      "x_predicted_arcsec", "y_predicted_arcsec", "image_residual_arcsec"])
        return {"image_fit_quality": pd.DataFrame(columns=["family_id", "image_label", "x_obs_arcsec", "image_recovery_status"]),
                "image_count_recovery": pd.DataFrame(), "model_magnification": pd.DataFrame(),
                "image_recovery_extra_images": pd.DataFrame(), "posterior_extra_images": pd.DataFrame(),
                "posterior_image_rms": pd.DataFrame(), "posterior_image_residual_draws": draws}
    rng = np.random.default_rng(3)
    labels, families = ["1.a", "1.b", "2.a"], ["1", "1", "2"]
    draws = pd.DataFrame({"draw_index": np.repeat(np.arange(4), 3), "posterior_index": np.repeat([5, 9, 2, 7], 3),
                          "family_id": families * 4, "image_label": labels * 4,
                          "recovered": [True, True, False] * 4, "x_obs_arcsec": [1.0, -1.0, 3.0] * 4,
                          "y_obs_arcsec": [0.5, -0.5, 2.0] * 4, "x_predicted_arcsec": rng.normal(0, 1, 12),
                          "y_predicted_arcsec": rng.normal(0, 1, 12), "image_residual_arcsec": np.abs(rng.normal(0, .3, 12))})
    image_df = pd.DataFrame({"family_id": families, "image_label": labels, "x_obs_arcsec": [1.0, -1.0, 3.0],
                             "image_recovery_status": ["recovered", "recovered", "missing"],
                             "recovered_fraction": [1.0, 1.0, 0.0], "x_model_q50": [1.01, -0.98, np.nan],
                             "n_draws": np.array([4, 4, 0], dtype=np.int64)})
    extra = pd.DataFrame({"family_id": ["2"], "x_model_arcsec": [4.5], "y_model_arcsec": [-2.0], "draw_fraction": [0.75]})
    return {"image_fit_quality": image_df, "image_count_recovery": pd.DataFrame({"family_id": ["1", "2"], "multiplicity_failed": [False, True]}),
            "model_magnification": pd.DataFrame({"image_label": labels, "mu": [3.0, -2.0, np.nan]}),
            "image_recovery_extra_images": extra, "posterior_extra_images": extra.assign(draw_index=[1]),
            "posterior_image_rms": pd.DataFrame({"draw_index": np.arange(4), "rms_arcsec": [.2, .3, .25, .4]}),
            "posterior_image_residual_draws": draws}


class Dependent(OutputModule):
    """A converted module that requires ``images`` and records the product it got."""

    name, requires, compute_steps, render_steps = "dependent", ("images",), ("use",), ()

    def __init__(self, seen):
        self.seen = seen

    def compute(self, ctx):
        ctx.step("use", lambda: self.seen.append(ctx.products["images"]) or "used")


class Setup:
    """A stage, the fakes of the draws, the solver, the fit numbers, the true-lens RMS and the two figures, and passes."""

    def __init__(self, monkeypatch, tmp_path):
        from kappaccino.diagnostics import image_recovery
        from kappaccino.outputs import images as images_module
        from kappaccino.plotting import plots
        from kappaccino.reporting import draws as draws_module
        self.monkeypatch, self.tmp_path = monkeypatch, tmp_path
        self.solved, self.drawn, self.true_lens_calls, self.lines = [], [], [], []
        self.tables, self.fail, self.true_lens_writes = recovery_tables(), False, True

        class FakeDraws(list):
            seed, has_field, cache_stage, image_key = 7, False, None, "draws-key"
            image_indices = list(range(4))

        def draws_factory(*args):
            if self.mode == "render":
                raise AssertionError("a render pass built the reporting draws")
            return FakeDraws(range(4))

        def solve(*args, **kwargs):
            if self.fail:
                raise RuntimeError("solver broke")
            self.solved.append(1)
            return {name: frame.copy() for name, frame in self.tables.items()}

        def true_lens(stage, image_df, deflection_fits, cosmo_config, *, log=print):
            self.true_lens_calls.append(str(deflection_fits))
            if not self.true_lens_writes:
                return None
            record = {"rms_arcsec": 0.1 * len(self.true_lens_calls), "p16_arcsec": 0.05, "p84_arcsec": 0.2}
            store.write_record(stage, "true_lens_rms", record)
            return record

        def figure(name):
            def draw(*args, **kwargs):
                path = next(a for a in args if isinstance(a, type(tmp_path)))
                path.write_text(name)
                self.drawn.append((name, args, kwargs))
                return path
            return draw

        monkeypatch.setattr(draws_module, "reporting_draws", draws_factory)
        monkeypatch.setattr(image_recovery, "image_recovery_tables", solve)
        monkeypatch.setattr(image_recovery, "fit_summary", lambda state, recovery, extra_image_draws=0: {
            "observed_image_count": 3, "rows": len(recovery["posterior_image_residual_draws"]), "extra_draws": extra_image_draws})
        monkeypatch.setattr(images_module, "write_true_lens_rms", true_lens)
        monkeypatch.setattr(plots, "plot_image_residual_histogram", figure("histogram"))
        monkeypatch.setattr(plots, "plot_image_recovery", figure("recovery"))
        self.deflection = None
        self.mode = "full"

    def context(self, mode="full"):
        self.mode = mode
        config = SimpleNamespace(image_diagnostics=SimpleNamespace(exact_image_finder="global-jax"),
                                 reporting=SimpleNamespace(extra_images=True),
                                 truth=SimpleNamespace(deflection_true_fits=self.deflection),
                                 paths=SimpleNamespace(figure_prefix="toy"))
        state = SimpleNamespace(family_data=[], parameter_specs=[], cosmo_config={"H0": 70.0, "Om0": 0.3})
        paths = SimpleNamespace(tables=self.tmp_path / "tables", figures=self.tmp_path / "figures", root=self.tmp_path)
        return OutputContext(config, state, None, None, {}, paths, log=self.lines.append, mode=mode)

    def run(self, mode="full", modules=None):
        ctx = self.context(mode)
        records = run_modules(ctx, modules or [ImagesModule()])
        return ctx, records

    def figures(self):
        return sorted(p.name for p in (self.tmp_path / "figures").glob("*.pdf")) if (self.tmp_path / "figures").exists() else []


@pytest.fixture
def setup(monkeypatch, tmp_path):
    return Setup(monkeypatch, tmp_path)


def test_the_product_is_the_saved_tables(setup):
    ctx, records = setup.run()
    assert records["images"]["status"] == "done" and list(records["images"]["steps"]) == ["recovery"]
    product = ctx.products["images"]
    assert set(product) == {"image_df", "recovery", "true_lens"} and product["true_lens"] is None
    computed = recovery_tables()
    assert list(product["recovery"]) == [*computed, "position_coverage"]
    for name, frame in computed.items():
        pd.testing.assert_frame_equal(product["recovery"][name], frame, check_exact=True)
    pd.testing.assert_frame_equal(product["recovery"]["position_coverage"], store.read_table(setup.tmp_path, "position_coverage"))
    assert len(product["recovery"]["position_coverage"]) == 3
    assert product["image_df"] is product["recovery"]["image_fit_quality"]
    assert ctx.summary["fit"] == {"observed_image_count": 3, "rows": 12, "extra_draws": 4}


def test_an_empty_image_table_round_trips_and_draws_no_figure(setup):
    from kappaccino.outputs import module_named
    setup.tables = recovery_tables(empty=True)
    ctx, records = setup.run()
    image_df = ctx.products["images"]["image_df"]
    assert image_df.empty and list(image_df.columns) == ["family_id", "image_label", "x_obs_arcsec", "image_recovery_status"]
    for name, frame in recovery_tables(empty=True).items():
        pd.testing.assert_frame_equal(ctx.products["images"]["recovery"][name], frame, check_exact=True)
    assert ctx.products["images"]["recovery"]["position_coverage"].empty
    assert records["images"]["status"] == "done" and "render_steps" not in records["images"] and setup.figures() == []
    ctx.config.image_catalog = SimpleNamespace(image_dir="imaging")
    assert module_named("rgb").active(ctx) == "no observed image in the image recovery"


def test_a_render_pass_draws_both_figures_from_the_store(setup):
    setup.run()
    first = list(setup.drawn)
    assert setup.figures() == ["toy_image_recovery.pdf", "toy_image_residual_histogram.pdf"]
    for path in (setup.tmp_path / "figures").glob("*.pdf"):
        path.unlink()
    setup.drawn.clear()
    ctx, records = setup.run("render")
    assert setup.solved == [1] and records["images"]["status"] == "done" and records["images"]["mode"] == "render"
    assert list(records["images"]["render_steps"]) == ["residual_histogram", "recovery_figure"]
    assert setup.figures() == ["toy_image_recovery.pdf", "toy_image_residual_histogram.pdf"]
    # The same arguments as in the full pass, and those are the saved tables.
    computed = recovery_tables()
    assert [(name, args[-1]) for name, args, _ in setup.drawn] == [(name, args[-1]) for name, args, _ in first]
    for name, args, kwargs in setup.drawn:
        pd.testing.assert_frame_equal(args[0], computed["image_fit_quality"], check_exact=True)
        if name == "histogram":
            pd.testing.assert_frame_equal(kwargs["draws"], computed["posterior_image_residual_draws"], check_exact=True)
            assert kwargs["true_lens"] is None
        else:
            pd.testing.assert_frame_equal(args[1], computed["image_recovery_extra_images"], check_exact=True)
    # The guard: in a render pass the draws and the evaluator are compute-only.
    from kappaccino.outputs.base import ComputeOnly
    for what in ("draws", "evaluator"):
        with pytest.raises(ComputeOnly):
            getattr(ctx, what)


def test_a_render_step_that_asks_for_the_draws_fails(setup):
    from kappaccino.plotting import plots
    setup.run()
    ctx = setup.context("render")
    setup.monkeypatch.setattr(plots, "plot_image_recovery", lambda *a, **k: ctx.draws)
    records = run_modules(ctx, [ImagesModule()])
    error = records["images"]["render_steps"]["recovery_figure"]["error"]
    assert error.startswith("ComputeOnly: images.recovery_figure asked for ctx.draws in the render phase of a render pass")
    assert records["images"]["status"] == "failed" and records["images"]["render_steps"]["residual_histogram"]["status"] == "done"


def test_a_failed_recovery_makes_the_module_unavailable(setup):
    seen = []
    setup.fail, setup.deflection = True, "defl.fits"
    ctx, records = setup.run(modules=[ImagesModule(), Dependent(seen)])
    assert records["images"]["status"] == "failed" and "images" not in ctx.products
    assert records["images"]["steps"]["recovery"]["status"] == "failed"
    assert records["images"]["steps"]["true_lens"] == {"status": "skipped", "seconds": 0.0, "reason": "recovery failed"}
    assert records["dependent"]["status"] == "skipped" and seen == [] and setup.figures() == [] and "fit" not in ctx.summary
    assert setup.true_lens_calls == []
    # A render pass then refuses the module: its last compute is incomplete.
    ctx, records = setup.run("render", modules=[ImagesModule(), Dependent(seen)])
    assert records["images"]["status"] == "failed" and "incomplete" in records["images"]["error"]
    assert records["dependent"]["status"] == "skipped"


def test_dependents_get_the_product(setup):
    seen = []
    ctx, records = setup.run(modules=[ImagesModule(), Dependent(seen)])
    assert records["dependent"]["status"] == "done" and seen == [ctx.products["images"]]


def test_the_true_lens_record_only_from_this_compute(setup):
    setup.deflection = "defl.fits"
    ctx, records = setup.run()
    assert setup.true_lens_calls == ["defl.fits"] and ctx.products["images"]["true_lens"]["rms_arcsec"] == 0.1
    assert records["images"]["steps"]["true_lens"]["wrote"] is True
    assert setup.drawn[0][0] == "histogram" and setup.drawn[0][2]["true_lens"]["rms_arcsec"] == 0.1
    ctx, _ = setup.run("render")
    assert ctx.products["images"]["true_lens"] == {"rms_arcsec": 0.1, "p16_arcsec": 0.05, "p84_arcsec": 0.2}
    # The writer skips (unreadable map): the record of the earlier compute stays in the store and is not returned.
    setup.true_lens_writes = False
    ctx, records = setup.run()
    assert records["images"]["steps"]["true_lens"]["wrote"] is False and ctx.products["images"]["true_lens"] is None
    assert store.read_record(setup.tmp_path, "true_lens_rms")["rms_arcsec"] == 0.1
    assert setup.run("render")[0].products["images"]["true_lens"] is None
    # No deflection map: the step does not run and nothing is returned either.
    setup.true_lens_writes, setup.deflection = True, "defl.fits"
    setup.run()
    setup.deflection = None
    ctx, records = setup.run()
    assert "true_lens" not in records["images"]["steps"] and ctx.products["images"]["true_lens"] is None
    assert setup.run("render")[0].products["images"]["true_lens"] is None
    # With an empty image table (on a new stage, so the recovery is not reused) the true-lens RMS does not run.
    setup.deflection, setup.tables, calls = "defl.fits", recovery_tables(empty=True), len(setup.true_lens_calls)
    setup.tmp_path = setup.tmp_path / "empty"
    ctx, records = setup.run()
    assert len(setup.true_lens_calls) == calls and ctx.products["images"]["true_lens"] is None


def test_a_second_compute_reuses_the_saved_recovery(setup):
    first, _ = setup.run()
    again, _ = setup.run()
    assert setup.solved == [1] and any("image recovery: 7 tables reused" in line for line in setup.lines)
    for name, frame in first.products["images"]["recovery"].items():
        pd.testing.assert_frame_equal(again.products["images"]["recovery"][name], frame, check_exact=True)
    assert again.summary["fit"] == first.summary["fit"]


def test_the_fit_numbers_survive_a_render_pass(setup):
    from kappaccino.reporting import results_tex
    from kappaccino.reporting.outputs import write_run_summary
    setup.monkeypatch.setattr(results_tex, "write_run_results", lambda *a, **k: None)
    modules = [ImagesModule()]
    ctx, _ = setup.run(modules=modules)
    write_run_summary(ctx, partial=False, modules=modules)
    fit = store.read_record(setup.tmp_path, "run_summary")["fit"]
    assert fit == {"observed_image_count": 3, "rows": 12, "extra_draws": 4}
    ctx, _ = setup.run("render", modules=modules)
    assert "fit" not in ctx.summary
    write_run_summary(ctx, partial=True, modules=modules)
    saved = store.read_record(setup.tmp_path, "run_summary")
    assert saved["fit"] == fit and saved["modules"]["images"]["mode"] == "render"
