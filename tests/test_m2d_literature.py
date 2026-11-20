from dataclasses import replace

import numpy as np
import pandas as pd

from kappaccino.config import TruthRecoveryConfig
from kappaccino.diagnostics.truth_recovery import LITERATURE_RATIOS, annular_kappa_profile, literature_m2d_ratios, literature_ratios
from kappaccino.plotting import plots


def test_literature_ratios_are_read_beside_the_truth_map_and_drawn(tmp_path):
    config = replace(TruthRecoveryConfig(), kappa_true_fits=str(tmp_path/"kappa.fits"))
    assert literature_m2d_ratios(config) is None
    radius = np.geomspace(5., 90., 60)
    rows = [dict(quantity=q, approach=a, model=m, r_arcsec=r, ratio=1.02, width_px=3.0, flag=0) for q in ("mass", "kappa") for m, a in (("CATS", "parametric"), ("GRALE", "free-form")) for r in radius]
    (tmp_path/LITERATURE_RATIOS).write_text("# header\n" + pd.DataFrame(rows).to_csv(index=False))
    literature = literature_m2d_ratios(config)
    assert len(literature) == 120 and "quantity" not in literature and set(literature["model"]) == {"CATS", "GRALE"} and set(literature["approach"]) == {"parametric", "free-form"}
    profile = pd.DataFrame(dict(radius_arcsec=radius, kappa_true_sum=radius**1.5, kappa_model_sum=1.01 * radius**1.5,
                                kappa_model_sum_q16=radius**1.5, kappa_model_sum_q84=1.02 * radius**1.5,
                                m2d_ratio_median=1.01, m2d_ratio_q16=1.0, m2d_ratio_q84=1.02))
    path = plots.plot_m2d_aperture_ratio(profile, dict(center_x_arcsec=0., center_y_arcsec=0.), None, 1e10, tmp_path/"m2d.pdf", literature=literature)
    assert path.exists() and path.stat().st_size > 0

    kappa = literature_ratios(config, "kappa")
    grid = np.linspace(-100., 100., 201)
    x, y = np.meshgrid(grid, grid)
    truth = 1. / (1. + np.hypot(x, y) / 10.)
    kappa_profile = annular_kappa_profile(truth, np.stack([1.02 * truth, 0.98 * truth, truth]), x, y, dict(center_x_arcsec=0., center_y_arcsec=0.))
    np.testing.assert_allclose(kappa_profile["kappa_ratio_median"], 1.0, atol=1e-12)
    assert kappa_profile["radius_hi_arcsec"].max() <= 100. + 1e-9
    path = plots.plot_truth_ratio_panels(profile, kappa_profile, dict(center_x_arcsec=0., center_y_arcsec=0.), None, tmp_path/"ratios.pdf",
                                         literature_mass=literature, literature_kappa=kappa)
    assert path.exists() and path.stat().st_size > 0


def test_footprint_mean_ratios_shared_range():
    import numpy as np
    from kappaccino.diagnostics.truth_recovery import footprint_mean_ratios, mean_ratio
    radius = np.geomspace(1.0, 200.0, 400)
    assert np.allclose(mean_ratio(radius, np.full(400, 1.02), 5.0, 50.0), (1.02, 0.02, 0.02))
    # Half the log range at 1.03 and half at 0.97: the mean ratio cancels, the rms does not.
    assert np.allclose(mean_ratio(radius, np.where(radius < np.sqrt(5.0 * 50.0), 1.03, 0.97), 5.0, 50.0), (1.0, 0.03, 0.03), atol=1e-3)
    assert mean_ratio(radius, np.ones(400), 300.0, 400.0) is None
    profile = pd.DataFrame({"radius_arcsec": radius, "m2d_ratio_median": np.full(400, 0.99), "kappa_ratio_median": np.full(400, 1.03)})
    r = np.geomspace(3.0, 40.0, 300)
    literature = pd.DataFrame({"approach": "parametric", "model": "CATS", "r_arcsec": r, "ratio": 1.05, "flag": 0})
    found = footprint_mean_ratios(profile, profile, np.array([10.0, 60.0]), literature, literature)
    # The outermost image (60") is cut at the edge of the published curves (40").
    assert found["outer_arcsec"] == 40.0
    assert np.allclose(found["mass"]["This work"], (0.99, 0.01, 0.01)) and np.allclose(found["kappa"]["This work"], (1.03, 0.03, 0.03))
    assert np.allclose(found["mass"]["CATS"], (1.05, 0.05, 0.05), atol=1e-6)


def test_footprint_mean_ratios_from_draws():
    import numpy as np
    from kappaccino.diagnostics.truth_recovery import footprint_mean_ratios, ratio_draws_table
    radius = np.geomspace(1.0, 200.0, 100)
    profile = pd.DataFrame({"radius_arcsec": radius, "m2d_ratio_median": 1.0, "kappa_ratio_median": 1.0})
    # Four draws, each flat at 0.98, 0.99, 1.01, 1.02: per-draw rms 2, 1, 1, 2 per cent.
    draws = ratio_draws_table(radius, np.tile([0.98, 0.99, 1.01, 1.02], (100, 1)))
    found = footprint_mean_ratios(profile, profile, np.array([40.0]), None, None, draws, draws)
    median, low, high = found["mass"]["This work"]
    assert np.allclose(median, (1.0, 0.015, 0.015)) and low[2] < 0.015 < high[2]


def test_truth_recovery_reads_back_what_its_figures_draw(tmp_path):
    from kappaccino.diagnostics.truth_recovery import load_truth_recovery, save_truth_recovery
    from kappaccino.io import store
    rng = np.random.default_rng(1)
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 4))
    maps = {q: rng.normal(size=x.shape) for q in ("kappa", "abs_mu", "detA", "mu")}
    center = dict(center_mode="smoothed_truth_kappa_peak", center_x_arcsec=0.1, center_y_arcsec=-0.2,
                  center_smoothing_sigma_pix=2.0, center_smoothed_kappa_peak=1.5)
    profile = pd.DataFrame({"radius_arcsec": [1.0, 2.0], "m2d_ratio_median": [1.0, 1.1]})
    literature = pd.DataFrame({"approach": ["parametric"], "model": ["CATS"], "r_arcsec": [3.0], "ratio": [1.0], "width_px": [2.0], "flag": [0]})
    recovery = {"x": x, "y": y, "truth": maps, "model": {q: {"median": 2 * v} for q, v in maps.items()}, "shape": x.shape,
                "z_source": 9.0, "grf_included": True, "mass_per_kappa_pixel_msun": 1.25e9, "center": center,
                "profile": profile, "kappa_profile": profile, "literature_m2d": literature, "literature_kappa": None}
    store.write_table(tmp_path, "truth_recovery_m2d_aperture_profile", profile)
    store.write_table(tmp_path, "truth_recovery_kappa_annulus_profile", profile)
    save_truth_recovery(tmp_path, recovery, n_draws=8)
    view = load_truth_recovery(tmp_path)
    assert set(view["truth"]) == {"kappa", "abs_mu", "detA"}                 # only what the residual figure draws
    for q in view["truth"]:
        np.testing.assert_array_equal(view["truth"][q], maps[q])
        np.testing.assert_array_equal(view["model"][q]["median"], 2 * maps[q])
    np.testing.assert_array_equal(view["x"], x)
    assert view["center"] == center and view["mass_per_kappa_pixel_msun"] == 1.25e9 and view["z_source"] == 9.0
    pd.testing.assert_frame_equal(view["profile"], profile)
    pd.testing.assert_frame_equal(view["literature_m2d"], literature)
    assert view["literature_kappa"] is None
    assert store.read_record(tmp_path, "truth_recovery")["n_draws"] == 8
