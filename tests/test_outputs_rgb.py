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
                     image_catalog=replace(synthetic_config.image_catalog, image_dir="hst"))
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


def test_a_render_pass_reads_the_store_and_skips_missing_imaging(rgb_pass):
    rgb_pass()
    # On a machine without the HST imaging: logged once, the JWST figures are drawn, and nothing is computed.
    record = rgb_pass("render", missing=("",))
    assert record["status"] == "done" and record["mode"] == "render"
    assert not any(e.startswith(("save", "compute", "load_or", "prepare")) for e in rgb_pass.events)
    assert [line for line in rgb_pass.lines if "skipped: no imaging" in line] == [
        "[outputs] cluster mass figure and covariance RGB cutouts skipped: no imaging primary"]
    assert [step for step in record["render_steps"] if step.endswith(":primary")] == ["imaging:primary"]
    assert rgb_pass.events.count("render_saved_rgb_critical") == 1 and "render_paper_cutouts" in rgb_pass.events


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


def test_inset_critical_fields_without_draws_come_from_the_store(tmp_path, monkeypatch):
    from kappaccino.plotting import critical_fields, rgb_overview_cutouts as rc
    store.write_table(tmp_path, "toy_small_lambda_a_image_posteriors_jwst", pd.DataFrame(dict(
        image_label=["a", "b"], x_arcsec=[0., 1.], y_arcsec=[0., 0.], z_source=[2., 3.], family_id=["1", "2"], shown=[False, False])))
    store.write_arrays(tmp_path, "posterior_overview_positions", image_labels=np.array(["a", "b"]),
                       recovered=np.random.default_rng(1).normal(0, .02, (32, 2, 2)), draw_indices=np.arange(32))
    store.write_record(tmp_path, "fold_pairs_source_caustics", {})
    store.write_arrays(tmp_path, "fold_pairs_source_caustics", {})
    monkeypatch.setattr(rc, "load_fold_data", lambda _: dict(systems=[]))
    with pytest.raises(ValueError, match="toy_rgb_inset_critical holds none"):
        rc.overview_cutout_data(tmp_path, "toy", image_labels=("b",))

    def compute(draws, centers, fovs, redshifts, path, **k):
        median = np.arange(len(centers) * 16.).reshape(-1, 4, 4)
        store.write_arrays(path.parent, path.stem, median_determinant=median, lower=median - 1, upper=median + 1,
                           centers=np.asarray(centers), fovs=np.asarray(fovs), redshifts=np.asarray(redshifts))
        return None, median, median - 1, median + 1
    monkeypatch.setattr(critical_fields, "median_cutout_determinants", compute)
    computed = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), draws=object(), neighbors=True)
    monkeypatch.setattr(critical_fields, "median_cutout_determinants", lambda *a, **k: pytest.fail("computed without draws"))
    loaded = rc.overview_cutout_data(tmp_path, "toy", image_labels=("b", "a"), neighbors=True)
    for a, b in zip(computed["images"], loaded["images"]):
        for key in ("median_determinant", "lower", "upper"):
            np.testing.assert_array_equal(a["critical"][key], b["critical"][key])
    with pytest.raises(ValueError, match="holds none for these panels"):         # other panels than the saved ones
        rc.overview_cutout_data(tmp_path, "toy", image_labels=("a",))


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
    assert cc.save_covariance_cutouts(synthetic_config, state, draws, paths, imaging) == "small_lambda_covariance_cutouts"
    g = cc.median_geometry(state, draws)
    saved = store.read_arrays(paths.tables, "small_lambda_covariance_cutouts_ellipses")
    np.testing.assert_array_equal(saved["directions"], g["directions"][saved["image_index"]])
    np.testing.assert_array_equal(saved["widths"], g["widths"][saved["image_index"]])
    record = store.read_record(paths.tables, "small_lambda_covariance_cutouts")
    assert [r["image_label"] for r in record["records"]] == [state.images.image_labels[i] for i in saved["image_index"]]
    # The figure comes from the store: no draws, no evaluator.
    path = cc.render_covariance_cutouts(synthetic_config, state, paths, imaging, log=lambda _: None)
    assert path.read_bytes().startswith(b"%PDF")
    assert store.read_record(paths.tables, "small_lambda_covariance_cutouts") == record
