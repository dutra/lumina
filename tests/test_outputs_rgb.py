"""The rgb output module in compute and render: what compute saves, that render draws from the store alone, and the
imaging variants in both phases."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kappaccino.io import store
from kappaccino.outputs import OutputContext, OutputModule, run_modules
from kappaccino.outputs.rgb import RGBModule


class Images(OutputModule):
    """The images product the rgb module reads."""

    name = "images"

    def compute(self, ctx):
        pass

    def product(self, ctx):
        return {"image_df": pd.DataFrame({"image_label": ["a"]}),
                "recovery": {"image_recovery_extra_images": pd.DataFrame()}}


class Fit(OutputModule):
    """rgb runs after fit (it reads a table fit writes)."""

    name = "fit"

    def compute(self, ctx):
        pass


class FakeDraws(list):
    seed, has_field, field_is_rebuilt, key = 7, False, False, "draws"
    image_indices = indices = [0, 1]


@pytest.fixture
def rgb_pass(monkeypatch, synthetic_config, tmp_path):
    """Runs the images stand-in and the rgb module with every compute and render function of rgb replaced by a
    recorder; ``missing`` names the imaging variants (``figure_suffix``) whose imaging folder is missing."""
    from kappaccino.diagnostics import position_contours
    from kappaccino.imaging import cutouts
    from kappaccino.plotting import (apj_fold_cutouts, covariance_cutouts, fold_pairs, jwst_rgb, mass_figure,
                                     paper_cutouts, small_lambda_jwst)
    from kappaccino.reporting import draws as draws_module
    events, lines = [], []

    def record(name, result=None):
        return lambda *a, **k: events.append(name) or result

    for module, name, result in (
            (covariance_cutouts, "save_covariance_cutouts", "saved"), (covariance_cutouts, "render_covariance_cutouts", None),
            (jwst_rgb, "save_rgb_critical", "saved"), (jwst_rgb, "render_saved_rgb_critical", None),
            (position_contours, "load_or_compute_contours", {"summaries": []}), (position_contours, "load_position_contours", {}),
            (small_lambda_jwst, "save_small_lambda_jwst", "saved"), (small_lambda_jwst, "render_small_lambda_jwst", None),
            (mass_figure, "save_cluster_mass_figure", {}), (mass_figure, "draw_cluster_mass_figure", None),
            (fold_pairs, "compute_fold_diagnostics", {}), (fold_pairs, "save_fold_data", None),
            (fold_pairs, "load_fold_data", {}), (fold_pairs, "render_fold_diagnostics", None),
            (apj_fold_cutouts, "render_apj_fold_cutouts", None),
            (paper_cutouts, "prepare_paper_cutouts", ["fold_cutouts"]), (paper_cutouts, "render_paper_cutouts", None)):
        monkeypatch.setattr(module, name, record(name, result))
    config = replace(synthetic_config, paths=replace(synthetic_config.paths, run_name="m0416"),
                     outputs=replace(synthetic_config.outputs, rgb=replace(synthetic_config.outputs.rgb, image_dir="hst")))
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures", fits=tmp_path / "fits", root=tmp_path)

    def run(mode="full", missing=()):
        def load(image_config, *args):
            if image_config.figure_suffix in missing:
                raise FileNotFoundError(f"no imaging {image_config.figure_suffix or 'primary'}")
            return ({}, (), None, "m0416")
        monkeypatch.setattr(cutouts, "load_cluster_imaging", load)
        if mode == "render":
            monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: pytest.fail("draws built in a render pass"))
        else:
            monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: FakeDraws([0, 1]))
        events.clear(), lines.clear()
        ctx = OutputContext(config, SimpleNamespace(reference=(3, 0., 0.)), None if mode == "render" else "evaluator",
                            None, {}, paths, log=lines.append, mode=mode)
        return run_modules(ctx, [Fit(), Images(), RGBModule()])["rgb"]
    run.events, run.lines, run.monkeypatch = events, lines, monkeypatch
    return run


def test_compute_saves_and_render_draws_every_variant(rgb_pass):
    record = rgb_pass()
    assert record["status"] == "done"
    # M0416 without CANUCS: the primary (HST) and the JWST variant, each its own overview.
    assert list(record["steps"]) == [
        "imaging:primary", "covariance_cutouts:primary", "rgb_critical:primary", "cluster_mass:primary", "fold_data",
        "imaging:jwst", "covariance_cutouts:jwst", "rgb_critical:jwst", "position_contours:jwst", "small_lambda:jwst",
        "cluster_mass:jwst", "paper_cutouts:jwst"]
    assert list(record["render_steps"]) == [
        "imaging:primary", "covariance_cutouts:primary", "rgb_critical:primary", "cluster_mass:primary", "fold_cutouts:primary",
        "imaging:jwst", "covariance_cutouts:jwst", "rgb_critical:jwst", "small_lambda:jwst", "cluster_mass:jwst",
        "fold_cutouts:jwst", "apj_fold_cutouts:jwst", "paper_cutouts:jwst"]
    saves = [e for e in rgb_pass.events if e.startswith(("save", "compute", "load_or", "prepare"))]
    assert rgb_pass.events[:len(saves)] == saves                    # every figure after the whole compute
    assert rgb_pass.events.count("compute_fold_diagnostics") == 1


def test_a_render_pass_without_the_imaging_of_a_computed_variant_fails_it(rgb_pass):
    from kappaccino.outputs.base import rendered_anything
    rgb_pass()
    # On a machine without the HST imaging that the compute drew: that variant fails, naming its folder and --remap; the
    # JWST figures are still drawn, and nothing is computed.
    record = rgb_pass("render", missing=("",))
    assert record["status"] == "failed" and record["mode"] == "render"
    assert not any(e.startswith(("save", "compute", "load_or", "prepare")) for e in rgb_pass.events)
    failed = record["render_steps"]["imaging:primary"]
    assert failed["status"] == "failed" and "outputs.rgb.image_dir = hst" in failed["error"] and "--remap" in failed["error"]
    assert not any("skipped: no imaging" in line for line in rgb_pass.lines)
    assert [step for step in record["render_steps"] if step.endswith(":primary")] == ["imaging:primary"]
    assert rgb_pass.events.count("render_saved_rgb_critical") == 1 and "render_paper_cutouts" in rgb_pass.events
    assert record["render_steps"]["imaging:jwst"]["input"] is True
    # Loading imaging is not drawing: a pass that only loaded it drew nothing.
    loaded_only = {"rgb": {"status": "done", "render_steps": {"imaging:jwst": record["render_steps"]["imaging:jwst"]}}}
    assert not rendered_anything(loaded_only) and rendered_anything({"rgb": record})


def test_a_variant_computed_without_imaging_is_left_out_of_the_render(rgb_pass):
    record = rgb_pass("compute", missing=("",))
    assert record["steps"]["imaging:primary"]["wrote"] is False and "rgb_critical:primary" not in record["steps"]
    record = rgb_pass("render")
    assert [step for step in record["render_steps"] if step.endswith(":primary")] == []
    assert rgb_pass.events.count("render_saved_rgb_critical") == 1


def test_render_skips_the_figures_of_a_failed_compute_step(rgb_pass):
    from kappaccino.plotting import fold_pairs

    def broken(*a, **k):
        raise ValueError("no folds")
    rgb_pass.monkeypatch.setattr(fold_pairs, "compute_fold_diagnostics", broken)
    record = rgb_pass()
    assert record["status"] == "failed" and record["steps"]["fold_data"]["status"] == "failed"
    assert record["steps"]["paper_cutouts:jwst"] == {"status": "skipped", "seconds": 0.0, "reason": "fold_data failed"}
    for step in ("fold_cutouts:primary", "fold_cutouts:jwst", "apj_fold_cutouts:jwst"):
        assert record["render_steps"][step]["reason"] == "compute step fold_data failed"
    assert record["render_steps"]["paper_cutouts:jwst"]["reason"] == "compute step paper_cutouts:jwst skipped"
    assert "render_saved_rgb_critical" in rgb_pass.events


def test_position_contours_read_back_as_computed(tmp_path):
    from kappaccino.diagnostics.position_contours import load_or_compute_contours, load_position_contours
    rng = np.random.default_rng(3)
    rows = []
    for draw in range(12):
        for label, (x, y), n in (("1.a", (0., 0.), 12), ("1.b", (2., 1.), 5), ("2.a", (-1., 3.), 2)):
            rows.append(dict(posterior_index=draw, family_id=label[0], image_label=label, recovered=draw < n,
                             x_model_arcsec=x, y_model_arcsec=y, x_true_arcsec=x, y_true_arcsec=y,
                             x_predicted_arcsec=x + rng.normal(0, .1), y_predicted_arcsec=y + rng.normal(0, .05)))
    store.write_table(tmp_path, "posterior_image_residual_draws", pd.DataFrame(rows))
    computed = load_or_compute_contours(None, None, SimpleNamespace(indices=np.arange(12), key="k"), None, tmp_path,
                                        log=lambda _: None)
    loaded = load_position_contours(tmp_path)
    assert [len(curve["segments"]) for curve in computed["curves"]] == [len(curve["segments"]) for curve in loaded["curves"]]
    assert len(loaded["curves"]) == 4                                # KDE and ellipse levels; none with two draws
    for a, b in zip(computed["curves"], loaded["curves"]):
        assert {k: v for k, v in a.items() if k != "segments"} == {k: v for k, v in b.items() if k != "segments"}
        for sa, sb in zip(a["segments"], b["segments"]):
            np.testing.assert_array_equal(sa, sb)
    assert loaded["summaries"] == computed["summaries"] and loaded["fingerprint"] == "k"
    for key, value in computed["position_data"].items():
        np.testing.assert_array_equal(loaded["position_data"][key], value)


def _inset_inputs(tmp_path, monkeypatch):
    """The saved ranking, positions and fold data of two image panels ``a`` and ``b``; median_cutout_determinants
    replaced by a stand-in that saves fields named after its entry (``calls`` counts them)."""
    from kappaccino.plotting import critical_fields, rgb_overview_cutouts as rc
    store.write_table(tmp_path, "toy_small_lambda_a_image_posteriors_jwst", pd.DataFrame(dict(
        image_label=["a", "b"], x_arcsec=[0., 1.], y_arcsec=[0., 0.], z_source=[2., 3.], family_id=["1", "2"], shown=[False, False])))
    store.write_arrays(tmp_path, "posterior_overview_positions", image_labels=np.array(["a", "b"]),
                       recovered=np.random.default_rng(1).normal(0, .02, (32, 2, 2)), draw_indices=np.arange(32))
    monkeypatch.setattr(rc, "load_fold_data", lambda _: dict(systems=[], partial_systems=[]))
    calls = []

    def compute(draws, centers, fovs, redshifts, path, **k):
        calls.append(path.stem)
        median = np.full((len(centers), 4, 4), float(len(path.stem))) + np.arange(len(centers))[:, None, None]
        store.write_arrays(path.parent, path.stem, median_determinant=median, lower=median - 1, upper=median + 1,
                           centers=np.asarray(centers), fovs=np.asarray(fovs), redshifts=np.asarray(redshifts))
        return None, median, median - 1, median + 1
    monkeypatch.setattr(critical_fields, "median_cutout_determinants", compute)
    return rc, calls


def test_inset_critical_fields_of_a_render_are_those_the_compute_recorded(tmp_path, monkeypatch):
    rc, calls = _inset_inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="recorded no critical fields"):
        rc.overview_cutout_data(tmp_path, "toy", image_labels=("b",))
    computed = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), draws=object(), neighbors=True)
    assert computed["critical_source"] == {"entry": "toy_rgb_inset_critical", "index": [0, 1]} and len(calls) == 1
    loaded = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), neighbors=True,
                                     critical=computed["critical_source"])
    assert len(calls) == 1                                                     # no draws: nothing computed
    for a, b in zip(computed["images"], loaded["images"]):
        for key in ("median_determinant", "lower", "upper"):
            np.testing.assert_array_equal(a["critical"][key], b["critical"][key])
    with pytest.raises(ValueError, match="does not hold the panels"):         # rows of other panels
        rc.overview_cutout_data(tmp_path, "toy", image_labels=("a",), critical={"entry": "toy_rgb_inset_critical", "index": [0]})


def test_each_group_of_panels_keeps_its_own_inset_fields(tmp_path, monkeypatch):
    # Both paper-cutout figures need inset fields: the second must not overwrite the first's.
    rc, _ = _inset_inputs(tmp_path, monkeypatch)
    first = rc.overview_cutout_data(tmp_path, "toy", image_labels=("a",), draws=object(), group="small_lambda_posteriors")
    second = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), draws=object(), group="fold_cutouts")
    assert first["critical_source"]["entry"] == "toy_rgb_inset_critical_small_lambda_posteriors"
    assert second["critical_source"]["entry"] == "toy_rgb_inset_critical_fold_cutouts"
    loaded = rc.overview_cutout_data(tmp_path, "toy", image_labels=("a",), group="small_lambda_posteriors",
                                     critical=first["critical_source"])
    np.testing.assert_array_equal(loaded["images"][0]["critical"]["median_determinant"],
                                  first["images"][0]["critical"]["median_determinant"])
    assert len(rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), group="fold_cutouts",
                                       critical=second["critical_source"])["images"]) == 2


def test_small_lambda_critical_fields_are_used_only_when_this_compute_wrote_them(tmp_path, monkeypatch):
    rc, calls = _inset_inputs(tmp_path, monkeypatch)
    # A small-lambda entry holding exactly these panels, as an older compute may have left it.
    stale = "toy_small_lambda_a_image_posteriors_jwst_critical"
    from kappaccino.plotting import critical_fields
    first = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), draws=object())
    centers = [item["center"] for item in first["images"]]
    critical_fields.median_cutout_determinants(None, centers[::-1], [item["fov"] for item in first["images"]][::-1],
                                               [item["z"] for item in first["images"]][::-1], tmp_path / f"{stale}.h5")
    calls.clear()
    # This compute did not write the small-lambda fields: they are not used, the panels get their own.
    fresh = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), draws=object(), small_lambda_critical=False)
    assert fresh["critical_source"]["entry"] == "toy_rgb_inset_critical" and calls == ["toy_rgb_inset_critical"]
    # Written by this compute: used, at the rows that hold each panel.
    calls.clear()
    shared = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), draws=object(), small_lambda_critical=True)
    assert shared["critical_source"] == {"entry": stale, "index": [1, 0]} and calls == []
    rendered = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), critical=shared["critical_source"])
    for a, b in zip(shared["images"], rendered["images"]):
        np.testing.assert_array_equal(a["critical"]["median_determinant"], b["critical"]["median_determinant"])
    # A render after a compute that recorded the inset fields never reads the small-lambda entry.
    rendered = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), critical=fresh["critical_source"])
    for a, b in zip(fresh["images"], rendered["images"]):
        np.testing.assert_array_equal(a["critical"]["median_determinant"], b["critical"]["median_determinant"])


def test_covariance_cutouts_render_from_the_store(synthetic_config, tmp_path, monkeypatch):
    from astropy.wcs import WCS
    from kappaccino.core.state import build_state
    from kappaccino.likelihood.evaluator import Evaluator
    from kappaccino.plotting import covariance_cutouts as cc
    from kappaccino.reporting.draws import ReportingDraws
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    posterior = SimpleNamespace(samples=np.array([state.initial_theta]),
                                physical=np.array([np.asarray(evaluator.physical(state.initial_theta))]))
    draws = ReportingDraws(evaluator, posterior, 1, 0)
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures")
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1000, 1000]; wcs.wcs.crval = list(state.reference[1:])
    wcs.wcs.cdelt = [-.05 / 3600, .05 / 3600]; wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    band = SimpleNamespace(wcs=wcs, shape=(2000, 2000), pixel_scale_arcsec=.05, path=tmp_path / "synthetic.fits")
    imaging = ({"F814W": band}, ("F814W",), None, "toy")
    monkeypatch.setattr(cc, "rgb_cutout", lambda *args: np.full((120, 120, 3), .2))
    labels = list(state.images.image_labels)
    fit_table = pd.DataFrame(dict(image_label=labels * 2, plane="image", direction=["arc"] * len(labels) + ["perpendicular"] * len(labels),
                                  total_p16_arcsec=np.arange(2. * len(labels)), total_p84_arcsec=np.arange(2. * len(labels)) + 1))
    assert cc.save_covariance_cutouts(synthetic_config, state, draws, paths, imaging,
                                      intervals=fit_table) == "small_lambda_covariance_cutouts"
    shown = store.read_table(paths.tables, "small_lambda_covariance_cutouts_intervals")
    arc = fit_table[fit_table.direction == "arc"].set_index("image_label")
    assert set(shown.image_label) <= set(labels) and len(shown) == 4
    np.testing.assert_array_equal(shown.total_p16_arcsec, arc.loc[shown.image_label, "total_p16_arcsec"])
    g = cc.median_geometry(state, draws)
    saved = store.read_arrays(paths.tables, "small_lambda_covariance_cutouts_ellipses")
    np.testing.assert_array_equal(saved["directions"], g["directions"][saved["image_index"]])
    np.testing.assert_array_equal(saved["widths"], g["widths"][saved["image_index"]])
    record = store.read_record(paths.tables, "small_lambda_covariance_cutouts")
    assert [r["image_label"] for r in record["records"]] == [state.images.image_labels[i] for i in saved["image_index"]]
    # The figure comes from the store: no draws, no evaluator, and not the table of fit, which a later compute of fit
    # (other draws) may have replaced.
    read_table = store.read_table
    monkeypatch.setattr(store, "read_table", lambda stage, name, **k: pytest.fail("render read fit's table")
                        if name == "covariance_errors_both_planes" else read_table(stage, name, **k))
    path = cc.render_covariance_cutouts(synthetic_config, state, paths, imaging, log=lambda _: None)
    assert path.read_bytes().startswith(b"%PDF")
    assert store.read_record(paths.tables, "small_lambda_covariance_cutouts") == record


def _toy_imaging(folder, reference, *, size=640, scale=.1):
    """Three HST bands of noise and a few sources around the toy reference, named so that the toy cluster finds them."""
    from astropy.io import fits
    from astropy.wcs import WCS
    folder.mkdir(parents=True)
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:size, 0:size]
    for band in ("F435W", "F606W", "F814W"):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = list(reference[1:])
        wcs.wcs.crpix = [size / 2 + .5, size / 2 + .5]
        wcs.wcs.cdelt = [-scale / 3600, scale / 3600]
        data = rng.normal(0, .01, (size, size)).astype(np.float32)
        for x, y in ((320, 320), (350, 380), (250, 300)):
            data += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 18.).astype(np.float32)
        fits.writeto(folder / f"toy_{band.lower()}_drz.fits", data, wcs.to_header())


def _forbid_evaluation(monkeypatch):
    """Make every way of evaluating the lens raise, for a render pass."""
    from kappaccino.likelihood.evaluator import Evaluator
    from kappaccino.outputs import saved_run
    from kappaccino.reporting import draws as draws_module

    def forbidden(*args, **kwargs):
        raise AssertionError("a render pass evaluated the lens")
    for target, name in ((Evaluator, "__init__"), (draws_module, "reporting_draws"),
                         (draws_module.ReportingDraws, "__init__"), (saved_run, "load_run")):
        monkeypatch.setattr(target, name, forbidden)


def test_the_real_rgb_module_renders_its_figures_again_from_the_store(synthetic_config, tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    from kappaccino.core.state import build_state
    from kappaccino.likelihood.evaluator import Evaluator
    from kappaccino.pipeline import StagePaths
    outputs = synthetic_config.outputs
    config = replace(synthetic_config, outputs=replace(outputs, draws=2, rgb=replace(outputs.rgb, image_dir=str(tmp_path / "toy_imaging"),
                                                                                     bands=("F435W", "F606W", "F814W"))))
    state = build_state(config)
    _toy_imaging(tmp_path / "toy_imaging", state.reference)
    evaluator = Evaluator(state)
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(2).normal(0, .002, (4, state.n_parameters))
    posterior = SimpleNamespace(samples=samples, physical=np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples))),
                                log_prob=np.zeros(4))
    paths = StagePaths(tmp_path / "run" / "final")
    records = run_modules(OutputContext(config, state, evaluator, posterior, {}, paths, log=lambda _: None), only=("rgb",))
    assert records["rgb"]["status"] == "done", records["rgb"]
    # The figures of rgb (images, pulled in by --only rgb, gives only its product in a render pass).
    drawn = sorted(p.name for p in paths.figures.glob("*.pdf") if not p.name.startswith("image_re"))
    assert drawn == ["hst_rgb_critical_z9.pdf", "image_catalog_family_cluster_mass.pdf", "small_lambda_covariance_cutouts.pdf"]
    for path in paths.figures.glob("*.pdf"):
        path.unlink()
    _forbid_evaluation(monkeypatch)
    records = run_modules(OutputContext(config, state, None, posterior, {}, paths, log=lambda _: None, mode="render"),
                          only=("rgb",))
    assert records["rgb"]["status"] == "done" and records["rgb"]["mode"] == "render", records["rgb"]
    assert sorted(p.name for p in paths.figures.glob("*.pdf")) == drawn
    # Without the imaging the compute drew, the render fails rgb instead of drawing nothing in silence.
    (tmp_path / "toy_imaging").rename(tmp_path / "moved")
    records = run_modules(OutputContext(config, state, None, posterior, {}, paths, log=lambda _: None, mode="render"),
                          only=("rgb",))
    assert records["rgb"]["status"] == "failed" and list(records["rgb"]["render_steps"]) == ["imaging:primary"]
    assert "toy_imaging" in records["rgb"]["render_steps"]["imaging:primary"]["error"]


def test_the_covariance_cutouts_read_the_table_of_fit_only_when_its_last_compute_wrote_it(tmp_path):
    from kappaccino.outputs.base import completion_name
    from kappaccino.outputs.rgb import fit_covariance_errors
    table = pd.DataFrame(dict(image_label=["a"], plane=["image"], direction=["arc"], total_p16_arcsec=[1.], total_p84_arcsec=[2.]))
    store.write_table(tmp_path, "covariance_errors_both_planes", table)
    assert fit_covariance_errors(tmp_path) is None                                   # no compute of fit recorded
    for step in ({"status": "failed", "error": "x"}, {"status": "done", "wrote": False}):
        store.write_record(tmp_path, completion_name("fit"), {"complete": False, "steps": {"covariance": step}})
        assert fit_covariance_errors(tmp_path) is None
    store.write_record(tmp_path, completion_name("fit"), {"complete": True, "steps": {"covariance": {"status": "done", "wrote": True}}})
    pd.testing.assert_frame_equal(fit_covariance_errors(tmp_path), table)
