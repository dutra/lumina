"""The truth output module with the real code on the toy cluster: what its compute saves is enough for a render pass
that evaluates nothing to draw every figure again."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from lumina.outputs import OutputContext, run_modules


def _truth_maps(folder, reference, *, size=121, scale=.5, einstein=12.):
    """True convergence and shear maps of a singular isothermal sphere at the toy reference, at z = 9."""
    from astropy.io import fits
    from astropy.wcs import WCS
    folder.mkdir(parents=True)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = list(reference[1:])
    wcs.wcs.crpix = [size / 2 + .5, size / 2 + .5]
    wcs.wcs.cdelt = [-scale / 3600, scale / 3600]
    header = wcs.to_header()
    header["ZSOURCE"] = 9.
    yy, xx = (np.mgrid[0:size, 0:size] - (size - 1) / 2) * scale
    x = -xx                                                    # RA grows to the east (left)
    r = np.maximum(np.hypot(x, yy), scale / 2)
    kappa = einstein / (2 * r)
    gamma1, gamma2 = -kappa * (x ** 2 - yy ** 2) / r ** 2, -kappa * 2 * x * yy / r ** 2
    paths = {}
    for name, values in (("kappa", kappa), ("gamma1", gamma1), ("gamma2", gamma2)):
        paths[name] = str(folder / f"{name}_z9.fits")
        fits.writeto(paths[name], values, header)
    return paths


def test_the_real_truth_module_renders_its_figures_again_from_the_store(synthetic_config, tmp_path, monkeypatch):
    import jax
    import jax.numpy as jnp
    from lumina.core.state import build_state
    from lumina.likelihood.evaluator import Evaluator
    from lumina.outputs import saved_run
    from lumina.pipeline import StagePaths
    from lumina.reporting import draws as draws_module
    state = build_state(synthetic_config)
    maps = _truth_maps(tmp_path / "truth", state.reference)
    outputs = synthetic_config.outputs
    config = replace(synthetic_config, outputs=replace(outputs, draws=2, truth=replace(
        outputs.truth, kappa_true_fits=maps["kappa"], gammax_true_fits=maps["gamma1"], gammay_true_fits=maps["gamma2"], grid_size=48)))
    evaluator = Evaluator(state)
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(2).normal(0, .002, (4, state.n_parameters))
    posterior = SimpleNamespace(samples=samples, physical=np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples))),
                                log_prob=np.zeros(4))
    paths = StagePaths(tmp_path / "run" / "final")
    records = run_modules(OutputContext(config, state, evaluator, posterior, {}, paths, log=lambda _: None), only=("truth",))
    assert records["truth"]["status"] == "done", records["truth"]
    # The figures of truth (images, pulled in by --only truth, gives only its product in a render pass).
    drawn = sorted(p.name for p in paths.figures.glob("*.pdf") if not p.name.startswith("image_re"))
    assert {"magnification_recovery.pdf", "cumulative_source_area.pdf", "truth_recovery_residuals.pdf"} <= set(drawn), drawn
    for path in paths.figures.glob("*.pdf"):
        path.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("a render pass evaluated the lens")
    for target, name in ((Evaluator, "__init__"), (draws_module, "reporting_draws"),
                         (draws_module.ReportingDraws, "__init__"), (saved_run, "load_run")):
        monkeypatch.setattr(target, name, forbidden)
    records = run_modules(OutputContext(config, state, None, posterior, {}, paths, log=lambda _: None, mode="render"),
                          only=("truth",))
    assert records["truth"]["status"] == "done" and records["truth"]["mode"] == "render", records["truth"]
    assert sorted(p.name for p in paths.figures.glob("*.pdf")) == drawn
