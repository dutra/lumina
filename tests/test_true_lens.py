"""The true-lens RMS of a simulated cluster and the green reference of the residual histogram."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

from scipy.optimize import least_squares

from kappaccino.diagnostics.true_lens import true_lens_residuals, true_lens_rms
from kappaccino.plotting import plots

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
CENTER = np.array([3.0, -2.0])             # map frame (x east), off axis so a flipped x would not close


def _efficiency(z):
    return (COSMO.angular_diameter_distance(0.5, z) / COSMO.angular_diameter_distance(z)).value


def _sis_map(path, theta_e=10.0, half=40.0, n=1600):
    """A singular isothermal sphere at ZS = 1 on a grid of pixel centers from -half to half."""
    step = 2 * half / n
    g = -half + step * (np.arange(n) + 0.5)
    x, y = np.meshgrid(g - CENTER[0], g - CENTER[1])
    r = np.hypot(x, y)
    header = fits.Header({"XMIN": -half, "XMAX": half, "YMIN": -half, "YMAX": half, "ZS": 1.0, "ZL": 0.5})
    fits.HDUList([fits.PrimaryHDU((theta_e * x / r).astype(np.float32), header=header),
                  fits.ImageHDU((theta_e * y / r).astype(np.float32))]).writeto(path)
    return theta_e


def _catalog(theta_e_map, families=300, sigma=0.05, z=2.0, seed=1):
    """Two SIS images per source at z, displaced by Gaussian scatter of sigma per axis, in the model frame."""
    rng = np.random.default_rng(seed)
    theta_e = theta_e_map * _efficiency(z) / _efficiency(1.0)
    rows = []
    for f in range(families):
        b, phi = rng.uniform(2.0, 4.0), rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(phi), np.sin(phi)])
        for k, radius in enumerate((b + theta_e, b - theta_e)):
            x, y = CENTER + radius * u + rng.normal(0, sigma, 2)
            rows.append({"family_id": str(f), "image_label": f"{f}.{k}", "x_obs_arcsec": -x, "y_obs_arcsec": y, "z_source": z})
    return pd.DataFrame(rows)


def test_true_lens_rms_recovers_the_injected_scatter(tmp_path):
    theta_e = _sis_map(tmp_path / "defl.fits")
    images, sigma = _catalog(theta_e), 0.05
    table, amplitude = true_lens_residuals(images, tmp_path / "defl.fits", COSMO)
    summary = true_lens_rms(table)
    n, f = len(images), images.family_id.nunique()
    expected = sigma * np.sqrt(2 * (1 - f / n))          # 2D scatter after removing a fitted source per family
    assert summary["n_images"] == n and summary["n_families"] == f
    np.testing.assert_allclose(amplitude, 1.0, atol=0.003)
    np.testing.assert_allclose(summary["rms_arcsec"], expected, rtol=0.07)
    assert summary["p16_arcsec"] < summary["rms_arcsec"] < summary["p84_arcsec"]
    assert (table.status == "used").all()


def test_true_lens_rms_is_zero_without_scatter(tmp_path):
    theta_e = _sis_map(tmp_path / "defl.fits")
    table, _ = true_lens_residuals(_catalog(theta_e, families=20, sigma=0.0), tmp_path / "defl.fits", COSMO)
    assert true_lens_rms(table)["rms_arcsec"] < 2e-3


def _legend(monkeypatch, call):
    figures = []
    monkeypatch.setattr(plots, "save_page", lambda fig, path: figures.append(fig) or path)
    call()
    legend = figures[0].axes[0].get_legend()
    return [t.get_text() for t in legend.get_texts()], figures[0].axes[0].get_xlim()


def _draws():
    return pd.DataFrame([{"draw_index": d, "recovered": True, "image_residual_arcsec": 0.05 * (k + 1) + 0.01 * d}
                         for d in range(5) for k in range(4)])


def test_run_histogram_gets_the_true_lens_entry(monkeypatch, tmp_path):
    record = {"rms_arcsec": 0.154, "p16_arcsec": 0.141, "p84_arcsec": 0.171}
    labels, _ = _legend(monkeypatch, lambda: plots.plot_image_residual_histogram(pd.DataFrame(), tmp_path / "h.pdf", draws=_draws(), true_lens=record))
    assert len(labels) == 3 and labels[2] == r"True-lens RMS $= 0.154^{+0.017}_{-0.013}$ arcsec"
    labels, _ = _legend(monkeypatch, lambda: plots.plot_image_residual_histogram(pd.DataFrame(), tmp_path / "h.pdf", draws=_draws()))
    assert len(labels) == 2 and labels[1].startswith("RMS $=")


def test_mock_campaign_histogram_is_unchanged(monkeypatch, tmp_path):
    from kappaccino.mock_validation.aggregate_plots import plot_image_residual_distribution
    offsets = {}
    for seed, shift in ((1, 0.30), (2, 0.40), (3, 0.50)):
        (tmp_path / f"seed_{seed:08d}" / "mock").mkdir(parents=True)
        images = [{"x_obs_arcsec": shift, "y_obs_arcsec": 0.0, "x_true_arcsec": 0.0, "y_true_arcsec": 0.0}]
        (tmp_path / f"seed_{seed:08d}" / "mock" / "mock_images.json").write_text(json.dumps(images))
        offsets[seed] = shift
    draws = pd.concat([_draws().assign(seed=s) for s in offsets])
    campaign = SimpleNamespace(root=tmp_path, seeds=list(offsets), residual_draws=draws)
    labels, xlim = _legend(monkeypatch, lambda: plot_image_residual_distribution(campaign, tmp_path / "m.pdf"))
    low, mid, high = np.percentile(list(offsets.values()), [16, 50, 84])     # the earlier code's own formula
    assert len(labels) == 3 and labels[2] == rf"Injected $= {mid:.3f}^{{+{high - mid:.3f}}}_{{-{mid - low:.3f}}}$ arcsec"
    assert xlim[1] == np.ceil(1.08 * high / 0.05) * 0.05


def test_outputs_save_the_true_lens_table_and_record(tmp_path):
    from kappaccino.io import store
    from kappaccino.outputs.images import write_true_lens_rms
    theta_e = _sis_map(tmp_path / "defl.fits")
    record = write_true_lens_rms(tmp_path, _catalog(theta_e, families=30), tmp_path / "defl.fits", {"H0": 70.0, "Om0": 0.3}, log=lambda m: None)
    assert store.read_record(tmp_path, "true_lens_rms")["rms_arcsec"] == record["rms_arcsec"]
    assert len(store.read_table(tmp_path, "true_lens_residuals")) == 60


# A point mass: two images per source with magnifications a factor 10-30 apart, and the demagnified image scattered
# 7 times more, so the family mean of the back-projected sources is far from the best source.
PM_CENTER, PM_THETA_E = np.array([1.5, -1.0]), 5.0


def _point_mass_map(path, half=20.0, n=1600):
    step = 2 * half / n
    g = -half + step * (np.arange(n) + 0.5)
    x, y = np.meshgrid(g - PM_CENTER[0], g - PM_CENTER[1])
    r2 = x * x + y * y
    header = fits.Header({"XMIN": -half, "XMAX": half, "YMIN": -half, "YMAX": half, "ZS": 1.0, "ZL": 0.5})
    fits.HDUList([fits.PrimaryHDU((PM_THETA_E ** 2 * x / r2).astype(np.float32), header=header),
                  fits.ImageHDU((PM_THETA_E ** 2 * y / r2).astype(np.float32))]).writeto(path)


def _point_mass_images(beta, theta_e2):
    """Map-frame images of a map-frame source (analytic)."""
    d = beta - PM_CENTER
    b = np.hypot(*d)
    s = np.sqrt(b * b + 4 * theta_e2)
    return np.array([PM_CENTER + d / b * (b + s) / 2, PM_CENTER + d / b * (b - s) / 2])


def test_true_lens_source_is_the_least_squares_optimum(tmp_path):
    _point_mass_map(tmp_path / "pm.fits")
    rng = np.random.default_rng(3)
    rows = []
    for f in range(40):
        b, phi = rng.uniform(6, 9), rng.uniform(0, 2 * np.pi)
        for k, (theta, sigma) in enumerate(zip(_point_mass_images(PM_CENTER + b * np.array([np.cos(phi), np.sin(phi)]), PM_THETA_E ** 2), (0.02, 0.15))):
            x, y = theta + rng.normal(0, sigma, 2)
            rows.append({"family_id": str(f), "image_label": f"{f}.{k}", "x_obs_arcsec": -x, "y_obs_arcsec": y, "z_source": 1.0})
    images = pd.DataFrame(rows)
    table, amplitude = true_lens_residuals(images, tmp_path / "pm.fits", COSMO)
    rms = true_lens_rms(table)["rms_arcsec"]
    theta_e2 = amplitude * PM_THETA_E ** 2
    best, mean = [], []
    for _, g in images.groupby("family_id"):
        obs = np.stack([-g.x_obs_arcsec, g.y_obs_arcsec], 1)
        residual = lambda beta: (_point_mass_images(beta, theta_e2) - obs).ravel()
        back = obs - theta_e2 * (obs - PM_CENTER) / np.sum((obs - PM_CENTER) ** 2, 1)[:, None]
        best.append(least_squares(residual, back[0], xtol=1e-12, ftol=1e-12).fun)
        mean.append(residual(back.mean(0)))
    rms_best, rms_mean = (np.sqrt(np.sum(np.concatenate(v) ** 2) / len(images)) for v in (best, mean))
    np.testing.assert_allclose(rms, rms_best, rtol=1e-4)      # the linearized source alone is 2e-3 off
    assert rms_mean > 1.3 * rms_best


def _sis_images(theta_e, centers):
    return pd.DataFrame([{"family_id": f, "image_label": f"{f}.{k}", "x_obs_arcsec": -x, "y_obs_arcsec": y, "z_source": 2.0}
                         for f, k, x, y in centers])


def test_true_lens_leaves_out_and_counts_unusable_images(tmp_path):
    theta_e = _sis_map(tmp_path / "defl.fits")
    images = pd.concat([_catalog(theta_e, families=20),
                        _sis_images(theta_e, [("far", 0, 55.0, 0.0), ("far", 1, -60.0, 3.0),     # outside the 40" half-width
                                              ("noz", 0, 5.0, 5.0), ("noz", 1, -5.0, -5.0),
                                              ("one", 0, 12.0, 4.0)])], ignore_index=True)
    images.loc[images.family_id == "noz", "z_source"] = np.nan
    table, _ = true_lens_residuals(images, tmp_path / "defl.fits", COSMO)
    summary = true_lens_rms(table)
    assert (summary["n_images"], summary["n_families"]) == (40, 20)
    assert (summary["n_outside_map"], summary["n_no_redshift"], summary["n_single_image_family"]) == (2, 2, 1)
    assert summary["rms_arcsec"] < 0.1 and table.loc[table.status != "used", "true_lens_residual_arcsec"].isna().all()


def test_true_lens_with_no_usable_image_raises(tmp_path):
    theta_e = _sis_map(tmp_path / "defl.fits")
    table, _ = true_lens_residuals(_sis_images(theta_e, [("far", 0, 55.0, 0.0), ("far", 1, -60.0, 3.0)]), tmp_path / "defl.fits", COSMO)
    with np.testing.assert_raises(ValueError):
        true_lens_rms(table)


def test_outputs_skip_the_true_lens_rms_on_a_missing_map(monkeypatch, tmp_path):
    from kappaccino.io import store
    from kappaccino.outputs.images import write_true_lens_rms
    messages = []
    record = write_true_lens_rms(tmp_path, _catalog(10.0, families=5), tmp_path / "missing.fits", {"H0": 70.0, "Om0": 0.3}, log=messages.append)
    assert record is None and messages[0].startswith("[outputs] true-lens RMS skipped: ")
    assert not store.has_record(tmp_path, "true_lens_rms")
    labels, _ = _legend(monkeypatch, lambda: plots.plot_image_residual_histogram(pd.DataFrame(), tmp_path / "h.pdf", draws=_draws(), true_lens=record))
    assert len(labels) == 2


def test_true_lens_counts_solutions_far_from_the_catalog_position(monkeypatch, tmp_path):
    import kappaccino.diagnostics.true_lens as true_lens
    theta_e = _sis_map(tmp_path / "defl.fits")
    monkeypatch.setattr(true_lens, "MATCH_LIMIT_ARCSEC", 0.03)          # below the 0.05" scatter: most images fail
    table, _ = true_lens_residuals(_catalog(theta_e, families=30), tmp_path / "defl.fits", COSMO)
    counts = table.status.value_counts()
    assert counts.get("not_matched", 0) > 10 and (table.loc[table.status == "used", "true_lens_residual_arcsec"] <= 0.03).all()
