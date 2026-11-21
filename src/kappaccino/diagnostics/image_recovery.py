"""Image-recovery tables from exact lens-equation solutions at the reporting posterior draws.

Every draw is solved in its own lens: parametric plus its conditional realization of the GRF
(``ReportingDraws.correction``), the lens behind the maps and critical curves.  The residuals, RMS and recovered
counts, the positions, the images a lens adds and the magnifications all come from that one lens, so their ranges over
the draws hold the uncertainty of the parameters and of the field.

The lens position is not where a catalog entry is expected: the likelihood lets the two differ by the image-plane GP
and the measurement error.  Each draw therefore also carries a predicted observed position, its lens position plus one
fresh draw of that scatter (:func:`image_plane_scatter`), the position the figures draw contours of, and a true
position, its lens position plus the GP displacement given the catalog (:func:`true_positions`): the catalog position
without its measurement error.
"""
from __future__ import annotations

import math
import time
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..config import ImageSolverConfig
from ..likelihood.evaluator import Evaluator
from .exact_images import ExactImageSolver
from ..core.model import ModelState, PosteriorResults
from ..core.parallel import map_draws


def image_plane_scales(evaluator: Evaluator, theta: np.ndarray) -> dict[str, float]:
    """The image-plane error scales of one draw, in arcsec: ``sigma_gp`` and ``ell_gp`` of the correlated displacement
    (0 and NaN when the likelihood has none) and the independent ``sigma_meas``."""
    scales = evaluator.error_scales(evaluator.physical(jnp.asarray(np.asarray(theta, dtype=float))))
    return {"sigma_gp": float(scales.get("sigma_gp", 0.0)), "ell_gp": float(scales.get("ell_gp", np.nan)),
            "sigma_meas": float(np.mean(np.asarray(scales["sigma_meas"], dtype=float)))}


def image_plane_scatter(xy: np.ndarray, scales: dict[str, float], rng: np.random.Generator, *,
                        leading: int | None = None) -> np.ndarray:
    """One draw of the displacement between the lens positions of images at ``xy`` (arcsec, shape (point, 2)) and their
    observed positions: the image-plane GP, correlated between the points with its squared-exponential kernel, plus
    independent measurement error.  It is drawn from the prior of these terms, not conditioned on the catalog.  Shape
    (point, 2), arcsec.

    The scatter of the first ``leading`` points (all of them by default) does not depend on the points after them:
    their random numbers are drawn first, and the Cholesky factor of the kernel is lower triangular.  The image
    recovery puts the observed images first, so their scatter is the same however many extra images a draw has."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    blocks = (len(xy),) if leading is None else (int(leading), len(xy) - int(leading))
    correlated = scales["sigma_gp"] > 0 and len(xy)
    white, smooth = [], []
    for size in blocks:
        white.append(rng.standard_normal((size, 2)))
        if correlated:
            smooth.append(rng.standard_normal((size, 2)))
    scatter = scales["sigma_meas"] * np.vstack(white)
    if correlated:
        scatter = scatter + scales["sigma_gp"] * np.linalg.cholesky(_gp_kernel(xy, scales["ell_gp"])) @ np.vstack(smooth)
    return scatter


def _gp_kernel(xy: np.ndarray, ell: float) -> np.ndarray:
    """Squared-exponential correlation of the image-plane GP between the points ``xy`` (identity without a finite
    correlation length), with the jitter its Cholesky factor needs."""
    if not (np.isfinite(ell) and ell > 0):
        return np.eye(len(xy))
    return np.exp(-np.sum((xy[:, None] - xy[None]) ** 2, axis=-1) / (2.0 * ell ** 2)) + 1e-9 * np.eye(len(xy))


def true_positions(catalog: np.ndarray, lens: np.ndarray, extra_lens: np.ndarray, scales: dict[str, float],
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One draw of the true image positions given the catalog: the lens positions plus the GP displacement.

    In the draw's lens the catalog position of an image is its lens position plus the GP displacement d and the
    measurement error e, so the offsets r = catalog - lens of the recovered images are data for d + e.  d is drawn
    jointly at all points given r (Matheron's rule), with covariance sigma_gp^2 K for d and sigma_meas^2 I for e.

    Parameters
    ----------
    catalog, lens : numpy.ndarray
        Catalog and lens positions of the observed images in arcsec, shape (image, 2); a lens position is NaN where the
        draw does not recover the image.
    extra_lens : numpy.ndarray
        Lens positions of the images the draw adds, shape (extra, 2).
    scales : dict
        :func:`image_plane_scales` of the draw.

    Returns
    -------
    tuple of numpy.ndarray
        True positions of the observed images (image, 2), equal to the catalog position minus the draw of e and NaN
        where not recovered, and of the extra images (extra, 2), whose displacement is only constrained by observed
        images within the correlation length.
    """
    catalog, lens = np.asarray(catalog, dtype=float).reshape(-1, 2), np.asarray(lens, dtype=float).reshape(-1, 2)
    extra_lens = np.asarray(extra_lens, dtype=float).reshape(-1, 2)
    found = np.isfinite(lens).all(axis=1)
    n = int(found.sum())
    points = np.vstack((catalog[found], extra_lens))
    kernel = _gp_kernel(points, scales["ell_gp"])
    # The random numbers of the observed images are drawn before those of the extra images, and the Cholesky factor
    # is lower triangular, so the true positions of the observed images do not depend on the extra images.
    normal, noise = rng.standard_normal((n, 2)), scales["sigma_meas"] * rng.standard_normal((n, 2))
    normal = np.vstack((normal, rng.standard_normal((len(extra_lens), 2))))
    displacement = scales["sigma_gp"] * np.linalg.cholesky(kernel) @ normal if len(points) else np.zeros((0, 2))
    if n:
        covariance = scales["sigma_gp"] ** 2 * kernel[:n, :n] + scales["sigma_meas"] ** 2 * np.eye(n)
        weights = np.linalg.solve(covariance, catalog[found] - lens[found] - displacement[:n] - noise)
        displacement = displacement + scales["sigma_gp"] ** 2 * kernel[:, :n] @ weights
    true = np.full_like(lens, np.nan)
    true[found] = lens[found] + displacement[:n]
    return true, extra_lens + displacement[n:]


def predictions_at(solver: ExactImageSolver, state: ModelState, evaluator: Evaluator, theta: np.ndarray, *,
                   adaptive: bool | None = None, correction=None) -> dict[str, Any]:
    """Per-image model positions, residuals and magnifications for one latent vector, in the parametric lens plus
    ``correction``: a deflection field called as ``correction(x, y, family_index)`` (the draw's GRF realization), or
    None for the parametric lens alone."""
    theta = np.asarray(theta, dtype=float)
    physical = evaluator.physical(jnp.asarray(theta))
    sources = evaluator.source_positions(jnp.asarray(theta))
    images = state.images
    family_by_id = {f.family_id: f for f in state.family_data}
    families = [family_by_id[family_id] for family_id in images.family_ids]
    extra_kwargs = {} if correction is None else dict(correction=correction)
    if adaptive is False:        # posterior draws: all families in one batched local solve
        batched = solver.predict_all_local(images, families, evaluator.packed_state(physical)[0], np.asarray(sources), **extra_kwargs)
    elif solver.config.exact_image_finder == "global-jax":   # a global search that also reports extra images
        batched, adaptive = solver.predict_all_global(images, families, np.asarray(physical), np.asarray(sources), **extra_kwargs), False
    elif correction is not None:
        raise ValueError("only the global-jax and batched local image finders add a deflection field to the lens; "
                         f"outputs.image_solver.exact_image_finder is {solver.config.exact_image_finder!r}")
    rows, extra_rows, count_rows = [], [], []
    for family_index, family in enumerate(families):
        family_id = family.family_id
        if adaptive is False:
            details = batched[family_index]
        else:
            packed = evaluator.packed_state_for_z(physical, family.z_source)
            details = solver.predict_family(family, packed, float(sources[family_index, 0]), float(sources[family_index, 1]), adaptive=adaptive)
        recovered = np.asarray(details["recovered_image_mask"], dtype=bool)
        mx, my = details["matched_model_x_arcsec"], details["matched_model_y_arcsec"]
        mu = np.asarray(details["magnification"], dtype=float)
        mu_model = np.asarray(details["magnification_at_model"], dtype=float)
        count_rows.append({"family_id": family_id, "z_source": family.z_source, "observed_image_count": family.n_images,
                           "produced_image_count": details["produced_image_count"], "recovered_image_count": details["recovered_image_count"],
                           "missing_image_count": details["missing_image_count"], "extra_image_count": details["extra_image_count"],
                           "multiplicity_failed": bool(details["multiplicity_failed"]),
                           "multiplicity_failure_reason": str(details["multiplicity_failure_reason"]),
                           "exact_image_rms_arcsec": float(details["exact_image_rms"]),
                           "source_x_arcsec": float(sources[family_index, 0]), "source_y_arcsec": float(sources[family_index, 1])})
        for k, label in enumerate(family.image_labels):
            x_obs, y_obs = float(family.x_obs[k]), float(family.y_obs[k])
            residual = math.hypot(mx[k] - x_obs, my[k] - y_obs) if recovered[k] else np.nan
            rows.append({"family_id": family_id, "image_label": label, "x_obs_arcsec": x_obs, "y_obs_arcsec": y_obs,
                         "z_source": family.z_source,
                         "radius_arcsec": math.hypot(x_obs, y_obs), "angle_deg": math.degrees(math.atan2(y_obs, x_obs)),
                         "image_recovery_status": "recovered" if recovered[k] else "not_recovered",
                         "x_model_arcsec": float(mx[k]), "y_model_arcsec": float(my[k]), "image_residual_arcsec": float(residual),
                         "magnification_model": float(mu[k]), "magnification_model_at_model_position": float(mu_model[k]),
                         "model_produced_image_count": details["produced_image_count"], "model_recovered_image_count": details["recovered_image_count"],
                         "model_missing_image_count": details["missing_image_count"], "model_extra_image_count": details["extra_image_count"],
                         "model_multiplicity_failed": bool(details["multiplicity_failed"]),
                         "model_multiplicity_failure_reason": str(details["multiplicity_failure_reason"])})
        for ex, ey in zip(details["extra_model_x_arcsec"], details["extra_model_y_arcsec"]):
            extra_rows.append({"family_id": family_id, "x_model_arcsec": float(ex), "y_model_arcsec": float(ey)})
    return {"image_rows": rows, "extra_rows": extra_rows, "count_rows": count_rows}


# Bump whenever the image solver or these tables can give other numbers for the same draws, catalog and settings (the
# shared-lattice global search and the fixed order of the extra images of 2026-11-18 were such changes; 4: the field's
# distance tables sized from the search grids, not the windows): tables saved with another version are computed again.
IMAGE_RECOVERY_VERSION = 4
# Record in results.h5, next to the tables, of what they were computed from (``recovery_fingerprint``).
RECOVERY_RECORD = "image_recovery_fingerprint"


def _digest(*parts) -> str:
    import hashlib
    h = hashlib.sha256()
    for part in parts:
        h.update(part if isinstance(part, bytes) else str(part).encode())
        h.update(b"\x00")
    return h.hexdigest()


def recovery_fingerprint(state: ModelState, draws, config: ImageSolverConfig, *, extra_images: bool) -> dict[str, str]:
    """What ``image_recovery_tables`` depends on, part by part: ``version`` (``IMAGE_RECOVERY_VERSION`` and the
    solver's ``shared_lattice`` switch), ``draws`` (seed, product and image draw indices and parameter vectors,
    and the model they map to, ``ReportingDraws.image_key``), ``image_diagnostics`` (every setting of ``config``, the ``outputs.image_solver`` section), ``extra_images`` and
    ``catalog`` (each family's id, redshift, labels, observed positions and search window, the image order and the
    parameter names)."""
    import dataclasses
    import json
    settings = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(getattr(config, "__dict__", {}))
    # Test doubles of the state may lack the catalog: an empty catalog part then (a real state always has one).
    images = getattr(state, "images", None)
    catalog = [_digest(json.dumps([str(f.family_id), float(f.z_source), [str(label) for label in f.image_labels],
                                   float(getattr(f, "search_window", np.nan)), float(getattr(f, "x_center", np.nan)),
                                   float(getattr(f, "y_center", np.nan))]),
                       np.asarray(f.x_obs, dtype=float).tobytes(), np.asarray(f.y_obs, dtype=float).tobytes())
               for f in getattr(state, "family_data", ())]
    catalog.append(_digest(json.dumps([str(v) for v in getattr(images, "image_labels", ())]),
                           json.dumps([str(v) for v in getattr(images, "family_ids", ())]),
                           json.dumps([spec.name for spec in getattr(state, "parameter_specs", ())])))
    return {"version": f"{IMAGE_RECOVERY_VERSION}:shared_lattice={bool(ExactImageSolver.shared_lattice)}",
            "draws": str(draws.image_key),
            "image_diagnostics": _digest(json.dumps(settings, sort_keys=True, default=str)),
            "extra_images": str(bool(extra_images)),
            "catalog": _digest(*catalog)}


def saved_image_recovery(stage, fingerprint: dict[str, str]) -> tuple[dict[str, pd.DataFrame] | None, str]:
    """The image-recovery tables saved in the stage's results.h5 with ``fingerprint`` (``recovery_fingerprint``), or
    (None, why they cannot be reused): no record, a part of the fingerprint changed, or a table missing or damaged."""
    from ..io import store
    try:
        saved = store.read_record(stage, RECOVERY_RECORD)
    except store.MissingEntry:
        return None, "no saved image recovery"
    except (store.DamagedEntry, store.StoreError) as error:
        return None, f"its record cannot be read ({type(error).__name__}: {error})"
    old = saved.get("fingerprint", {}) if isinstance(saved, dict) else {}
    changed = [part for part in fingerprint if old.get(part) != fingerprint[part]]
    if changed:
        return None, "changed since it was saved: " + ", ".join(changed)
    tables = {}
    for name in saved.get("tables", []):
        try:
            tables[str(name)] = store.read_table(stage, str(name))
        except store.MissingEntry:
            return None, f"table {name} missing"
        except (store.DamagedEntry, store.StoreError) as error:
            return None, f"table {name} cannot be read ({type(error).__name__}: {error})"
    if not tables:
        return None, "no saved tables"
    return tables, ""


def image_recovery_with_reuse(stage, state: ModelState, evaluator: Evaluator, draws, config: ImageSolverConfig, *,
                              extra_images: bool = True, compute=None, log=print) -> dict[str, pd.DataFrame]:
    """``image_recovery_tables``, or the tables an earlier pass saved in the stage's results.h5 when they were computed
    from the same draws, settings, catalog and code (``recovery_fingerprint``); ``compute`` replaces
    ``image_recovery_tables`` (the output stage passes its own name, which tests stub).  Computed tables are written to the
    stage with their fingerprint in one commit.  Either way the tables are the same: the store returns a table exactly
    as written."""
    from ..io import store
    fingerprint = recovery_fingerprint(state, draws, config, extra_images=extra_images)
    tables, reason = saved_image_recovery(stage, fingerprint)
    if tables is not None:
        log(f"[outputs] image recovery: {len(tables)} tables reused from {store.results_path(stage)} "
            f"(same draws, settings, catalog and solver version {IMAGE_RECOVERY_VERSION})")
        return tables
    log(f"[outputs] image recovery: computed ({reason})")
    tables = (compute or image_recovery_tables)(state, evaluator, draws, config, extra_images=extra_images, log=log)
    with store.batch(stage):
        for name, frame in tables.items():
            store.write_table(stage, name, frame)
        store.write_record(stage, RECOVERY_RECORD, {"fingerprint": fingerprint, "tables": list(tables)})
    return tables


LENS_ONLY_COLUMNS = ["family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "image_recovery_status", "recovered_fraction",
                     "x_model_arcsec", "y_model_arcsec", "image_residual_arcsec", "image_residual_q16", "image_residual_q84"]
# Extra model images of different draws within this distance (arcsec) are one image.
EXTRA_IMAGE_MATCH_ARCSEC = 1.0


def lens_only_summary(draw_rows: pd.DataFrame | None) -> dict[str, Any]:
    """Headline numbers of the lens-only draws (the parametric lens without the GRF), for
    run_summary.json and the results macros: median and 16-84% over draws of the recovered count, RMS and median
    residual."""
    if draw_rows is None or draw_rows.empty:
        return {}
    return {f"lens_only_{key}": value for key, value in draw_summary(draw_rows).items()}


def draw_summary(draw_rows: pd.DataFrame) -> dict[str, Any]:
    """Per-draw recovered count, RMS and median residual of ``draw_rows`` (one row per draw and image), as
    ``{"recovered_image_count": {"median", "q16", "q84"}, "image_rms_arcsec": ..., "image_median_residual_arcsec": ...}``."""
    per_draw = []
    for _, group in draw_rows.groupby("draw_index", sort=True):
        residual = pd.to_numeric(group["image_residual_arcsec"], errors="coerce").to_numpy(float)
        valid = group["recovered"].astype(bool).to_numpy() & np.isfinite(residual)
        per_draw.append((int(valid.sum()), float(np.sqrt(np.mean(residual[valid] ** 2))) if valid.any() else np.nan,
                         float(np.median(residual[valid])) if valid.any() else np.nan))
    values = np.asarray(per_draw, dtype=float)
    out = {}
    for column, name in enumerate(("recovered_image_count", "image_rms_arcsec", "image_median_residual_arcsec")):
        q16, q50, q84 = (float(v) for v in np.nanpercentile(values[:, column], [16, 50, 84]))
        out[name] = {"median": q50, "q16": q16, "q84": q84}
    return out


def per_image_table(draw_rows: pd.DataFrame, observed: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """One row per observed image from the per-draw rows: the fraction of draws that recover it, its status
    (``"recovered"`` when at least half the draws recover it) and, over the draws that recover it, the median of each
    of ``columns`` (same name) with its 16th, 50th and 84th percentiles (``<stem>_q16`` and so on, ``<stem>`` the
    column name without ``_arcsec``)."""
    rows = []
    for label, group in draw_rows.groupby("image_label", sort=False):
        recovered = group["recovered"].astype(bool).to_numpy()
        row = {"image_label": label, "recovered_fraction": float(recovered.mean()),
               "image_recovery_status": "recovered" if recovered.mean() >= .5 else "not_recovered",
               "posterior_valid_draws": int(recovered.sum()), "posterior_failed_draws": int((~recovered).sum())}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)[recovered]
            values = values[np.isfinite(values)]
            q16, q50, q84 = np.percentile(values, [16, 50, 84]) if values.size else (np.nan,) * 3
            stem = column.removesuffix("_arcsec")
            row.update({column: float(q50), f"{stem}_q16": float(q16), f"{stem}_q50": float(q50), f"{stem}_q84": float(q84)})
        rows.append(row)
    return observed.merge(pd.DataFrame(rows), on="image_label", how="left")


def common_extra_images(extra: pd.DataFrame, n_draws: int) -> pd.DataFrame:
    """Extra model images found in at least half of the draws: the per-draw extras of a family within
    ``EXTRA_IMAGE_MATCH_ARCSEC`` of each other are one image, reported at the median position with the fraction of draws
    that produce it."""
    rows = []
    if extra.empty:
        return pd.DataFrame(columns=["family_id", "x_model_arcsec", "y_model_arcsec", "x_model_q16", "x_model_q84",
                                     "y_model_q16", "y_model_q84", "draw_fraction"])
    for family_id, group in extra.groupby("family_id", sort=False):
        xy = group[["x_model_arcsec", "y_model_arcsec"]].to_numpy(float)
        draw = group["draw_index"].to_numpy(int)
        unassigned = np.ones(len(xy), bool)
        while unassigned.any():
            seed = np.flatnonzero(unassigned)[0]
            centre = xy[seed]
            for _ in range(5):          # recentre on the median of the members
                member = unassigned & (np.hypot(*(xy - centre).T) <= EXTRA_IMAGE_MATCH_ARCSEC)
                centre = np.median(xy[member], axis=0)
            member = unassigned & (np.hypot(*(xy - centre).T) <= EXTRA_IMAGE_MATCH_ARCSEC)
            member[seed] = True
            unassigned &= ~member
            fraction = len(np.unique(draw[member])) / n_draws
            if fraction >= .5:
                (x16, x50, x84), (y16, y50, y84) = (np.percentile(xy[member, i], [16, 50, 84]) for i in range(2))
                rows.append({"family_id": family_id, "x_model_arcsec": x50, "y_model_arcsec": y50, "x_model_q16": x16,
                             "x_model_q84": x84, "y_model_q16": y16, "y_model_q84": y84, "draw_fraction": fraction})
    return pd.DataFrame(rows)


def image_recovery_tables(state: ModelState, evaluator: Evaluator, draws, config: ImageSolverConfig, *,
                          extra_images: bool = True, log=print) -> dict[str, pd.DataFrame]:
    """Image-recovery tables over the reporting draws (``reporting.draws.ReportingDraws``); nothing at a single sample.

    Every image draw (``draws.image_samples``) is solved in its own lens, parametric plus the draw's GRF realization,
    with the batched local solver from the observed positions.  ``posterior_image_residual_draws`` has one row per draw
    and image: whether the lens recovers it, its lens position (``x_model_arcsec``, ``y_model_arcsec``), the residual
    from the catalog position, the magnification at the observed and at the lens position, the predicted observed
    position (``x_predicted_arcsec``, ``y_predicted_arcsec``: the lens position plus one draw of the image-plane
    scatter, :func:`image_plane_scatter`) and the true position (``x_true_arcsec``, ``y_true_arcsec``,
    :func:`true_positions`).  Per image, ``image_fit_quality`` and ``model_magnification`` give the fraction of draws
    that recover it and the medians with their 16-84% ranges; per draw, ``posterior_image_rms``; per family, the median
    and 16-84% of the per-draw counts.  With a mass field the parametric lens alone is solved alongside
    (``image_fit_quality_lens_only``, ``posterior_image_residual_draws_lens_only``).

    With ``extra_images``, the global search (``config.exact_image_finder``) gives the images each product draw
    (``draws.indices``) adds, with the same three positions; their ``draw_index`` counts the product draws.  Those
    found in at least half of them are reported.  The scatter and the true positions are joint over the observed (at
    their catalog positions) and extra images of a draw and seeded by the run seed and the draw's posterior row.
    Every row carries its ``posterior_index``.

    The draws are solved in ``draws.workers`` threads (one solver shared by all), the first product draw alone; the
    rows are assembled in draw order, so the tables do not depend on the number of threads."""
    solver = ExactImageSolver(evaluator, config)
    field = bool(getattr(draws, "has_field", False))
    reach = draws.reach(window_corners(solver, state)) if field else 0.0        # the realization must reach every grid point of the search
    product = {int(index): k for k, index in enumerate(draws.indices)}
    magnification = ("magnification_model", "magnification_model_at_model_position")

    def solve_draw(draw_index: int) -> dict[str, Any]:
        """Every row of image draw ``draw_index``, as the tables below collect them."""
        index, theta = draws.image_indices[draw_index], draws.image_samples[draw_index]
        lens = draws.image_correction(draw_index, reach) if field else None
        prediction = predictions_at(solver, state, evaluator, theta, adaptive=False, correction=lens)
        residuals = np.asarray([r["image_residual_arcsec"] for r in prediction["image_rows"]], dtype=float)
        rms = {"draw_index": draw_index, "posterior_index": int(index),
               "recovered_image_count": int(np.isfinite(residuals).sum()),
               "image_rms_arcsec": float(np.sqrt(np.nanmean(residuals ** 2))) if np.isfinite(residuals).any() else np.nan}
        own = _position_rows(prediction, draw_index, index, ("image_residual_arcsec",) + magnification)
        counts = [{"draw_index": draw_index, **row} for row in prediction["count_rows"]]
        added = ([{"draw_index": product[int(index)], "posterior_index": int(index), **row}
                  for row in predictions_at(solver, state, evaluator, theta, correction=lens)["extra_rows"]]
                 if extra_images and int(index) in product else [])
        # A fixed order within each family (by lens position): the solver returns a family's roots ordered by a
        # residual at round-off, and the scatter below is handed out in row order, so the order must not depend on it.
        rank = {}
        for row in added:
            rank.setdefault(str(row["family_id"]), len(rank))
        added.sort(key=lambda row: (rank[str(row["family_id"])], float(row["x_model_arcsec"]), float(row["y_model_arcsec"])))
        catalog = np.array([[row["x_obs_arcsec"], row["y_obs_arcsec"]] for row in prediction["image_rows"]], dtype=float).reshape(-1, 2)
        lens_xy = np.array([[row["x_model_arcsec"], row["y_model_arcsec"]] for row in own + added], dtype=float).reshape(-1, 2)
        scales, seed = image_plane_scales(evaluator, theta), [int(getattr(draws, "seed", 0)), int(index)]
        scatter = image_plane_scatter(np.vstack((catalog, lens_xy[len(own):])), scales, np.random.default_rng(seed + [1]),
                                      leading=len(catalog))
        true = np.vstack(true_positions(catalog, lens_xy[:len(own)], lens_xy[len(own):], scales, np.random.default_rng(seed + [2])))
        for row, predicted, actual in zip(own + added, lens_xy + scatter, true):
            row["x_predicted_arcsec"], row["y_predicted_arcsec"] = map(float, predicted)
            row["x_true_arcsec"], row["y_true_arcsec"] = map(float, actual)
        lens_only = []
        if field:        # the parametric lens alone shows how much of the misfit the field takes
            lens_only = _position_rows(predictions_at(solver, state, evaluator, theta, adaptive=False), draw_index, index,
                                       ("image_residual_arcsec",))
        return dict(image_rows=prediction["image_rows"], rms=rms, own=own, counts=counts, added=added, lens_only=lens_only)

    workers = int(getattr(draws, "workers", 1))
    # The first draw with a global search runs alone, so that both solvers compile before the threads start.
    first = next((j for j, index in enumerate(draws.image_indices) if int(index) in product), 0) if extra_images else 0
    started = time.perf_counter()
    per_draw = map_draws(solve_draw, range(len(draws.image_indices)), workers=workers, desc="images of each draw's lens", first=first)
    wall = time.perf_counter() - started
    observed = pd.DataFrame([{key: row[key] for key in ("family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "z_source",
                                                         "radius_arcsec", "angle_deg")} for row in per_draw[0]["image_rows"]])
    rows = [row for d in per_draw for row in d["own"]]
    extra_rows = [row for d in per_draw for row in d["added"]]
    count_rows = [row for d in per_draw for row in d["counts"]]
    lens_only_rows = [row for d in per_draw for row in d["lens_only"]]
    rms_rows = [d["rms"] for d in per_draw]
    del per_draw
    log(f"[diagnostics] observed images in each draw's lens ({'parametric + GRF realization' if field else 'parametric'}) at "
        f"{len(draws.image_indices)} posterior draws (local solver)"
        + (f"; global search for extra images at {len(draws.indices)} draws" if extra_images else "")
        + f": {wall:.1f}s wall with {workers} worker thread{'s' if workers > 1 else ''}, "
        f"{solver.timing['exact_solver']:.1f}s of solver time summed over the draws")
    draw_df, extra_df, counts = pd.DataFrame(rows), pd.DataFrame(extra_rows), pd.DataFrame(count_rows)
    image_df = per_image_table(draw_df, observed, ("x_model_arcsec", "y_model_arcsec", "image_residual_arcsec") + magnification)
    magnification_df = image_df[["family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "z_source", "image_recovery_status",
                                 "x_model_arcsec", "y_model_arcsec"] + [c for c in image_df.columns if c.startswith(magnification)]]
    count_df = family_count_table(counts.drop(columns=["extra_image_count"]))
    if extra_images:
        per_draw = (extra_df.groupby(["family_id", "draw_index"]).size() if not extra_df.empty else pd.Series(dtype=int))
        for name, q in (("extra_image_count", 50), ("extra_image_count_q16", 16), ("extra_image_count_q84", 84)):
            count_df[name] = [float(np.percentile([per_draw.get((f, d), 0) for d in range(len(draws.indices))], q))
                              for f in count_df.family_id]
    extra_tables = {}
    if lens_only_rows:
        lens_rows = pd.DataFrame(lens_only_rows)
        lens_only = per_image_table(lens_rows, observed[["family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec"]],
                                    ("x_model_arcsec", "y_model_arcsec", "image_residual_arcsec"))
        extra_tables = {"image_fit_quality_lens_only": lens_only[[c for c in LENS_ONLY_COLUMNS if c in lens_only.columns]],
                        "posterior_image_residual_draws_lens_only": lens_rows}
    return {**extra_tables, "image_fit_quality": image_df, "image_count_recovery": count_df, "model_magnification": magnification_df,
            "image_recovery_extra_images": common_extra_images(extra_df, len(draws.indices)) if extra_images else common_extra_images(pd.DataFrame(), 1),
            "posterior_extra_images": extra_df, "posterior_image_rms": pd.DataFrame(rms_rows), "posterior_image_residual_draws": draw_df}


def _position_rows(prediction: dict[str, Any], draw_index: int, posterior_index: int, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Per-image rows of one draw: whether the image is recovered, its model position and ``columns``."""
    return [{"draw_index": draw_index, "posterior_index": int(posterior_index), "family_id": row["family_id"],
             "image_label": row["image_label"], "x_obs_arcsec": row["x_obs_arcsec"], "y_obs_arcsec": row["y_obs_arcsec"],
             "recovered": row["image_recovery_status"] == "recovered",
             "x_model_arcsec": row["x_model_arcsec"], "y_model_arcsec": row["y_model_arcsec"], **{c: row[c] for c in columns}}
            for row in prediction["image_rows"]]


def window_corners(solver: ExactImageSolver, state: ModelState) -> np.ndarray:
    """Corners of the region the global search's grids cover, shape (point, 2), arcsec: the families' search windows
    snapped outward onto the coarse lattice and widened by the fine sub-grid (``ExactImageSolver.search_extent``)."""
    family_by_id = {f.family_id: f for f in state.family_data}
    return solver.search_extent([family_by_id[family_id] for family_id in state.images.family_ids])


def family_count_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Per family: the median and 16-84% over draws of the produced, recovered, missing and extra image counts, and the
    fraction of draws whose multiplicity fails."""
    rows = []
    for family_id, group in counts.groupby("family_id", sort=False):
        row = {"family_id": family_id, "z_source": float(group["z_source"].iloc[0]), "observed_image_count": int(group["observed_image_count"].iloc[0]),
               "multiplicity_failed_fraction": float(group["multiplicity_failed"].astype(bool).mean()),
               "multiplicity_failed": bool(group["multiplicity_failed"].astype(bool).mean() >= .5)}
        for name in [n for n in ("produced_image_count", "recovered_image_count", "missing_image_count", "extra_image_count") if n in group]:
            q16, q50, q84 = np.percentile(group[name].to_numpy(float), [16, 50, 84])
            row.update({name: float(q50), f"{name}_q16": float(q16), f"{name}_q84": float(q84)})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_summary(state: ModelState, recovery: dict[str, pd.DataFrame], *, extra_image_draws: int = 0) -> dict[str, Any]:
    """Headline recovery numbers over the reporting draws: the median and 16-84% of the per-draw recovered count, RMS
    and median residual (with and without the GRF), the median extra-image count and the families whose multiplicity
    holds in at least half the draws (``extra_image_draws``: how many draws had the global search, 0 if none).  The effective parameter count follows the lenstool convention,
    2 N_images - 2 N_families - N_global, where the global parameters are the halo, relation and error-model
    parameters: the member deviations are prior-dominated and add nothing to the effective count."""
    n_images = int(sum(f.n_images for f in state.family_data))
    n_global = int(sum(spec.component_family not in {"source_position", "member_scatter_offset", "member_shape_offset"} for spec in state.parameter_specs))
    k = n_global + 2 * len(state.family_data)
    out: dict[str, Any] = {"observed_image_count": n_images, "n_effective_parameters": k, "n_global_parameters": n_global}
    draws = recovery.get("posterior_image_residual_draws")
    if draws is None or draws.empty:
        return out
    out.update(draw_summary(draws))
    out["n_draws"] = int(draws["draw_index"].nunique())
    counts = recovery["image_count_recovery"]
    if extra_image_draws:
        extra = recovery["posterior_extra_images"]
        per_draw_extra = np.array([int((extra["draw_index"] == d).sum()) if not extra.empty else 0 for d in range(extra_image_draws)])
        q16, q50, q84 = np.percentile(per_draw_extra, [16, 50, 84])
        out["extra_image_count"] = {"median": float(q50), "q16": float(q16), "q84": float(q84)}
    out["families_fully_recovered"] = int((~counts["multiplicity_failed"].astype(bool)).sum()) if not counts.empty else 0
    out.update(lens_only_summary(recovery.get("posterior_image_residual_draws_lens_only")))
    return out
