"""Relative displacement against separation for a simulated cluster with a true deflection map
(``outputs.truth.deflection_true_fits``): the rms difference of a vector field between two points, split along and across
their separation, as a function of the separation.

For a field v at points i, j, Delta = v_i - v_j; per separation bin S_L = sqrt(mean Delta_L^2) along the separation
vector and S_T = sqrt(mean Delta_T^2) across it, in arcsec.  The curves, each per reporting draw k:

- truth: d_k = alpha_true - alpha_param(theta_k), the true map's deflection (with the fitted amplitude of
  ``diagnostics.true_lens``, scaled to the GRF reference redshift by D_ls/D_s) minus the draw's parametric lens;
- mass field: g_k, the deflection of draw k's GRF realization (``ReportingDraws.field``) at the same points;
- image-plane GP: on the images, the draw's GP displacement (true minus lens position, table
  ``posterior_image_residual_draws``, image draws); on the grid, the analytic
  sqrt(2 sigma_gp^2 (1 - exp(-r^2 / 2 ell^2))) of the draw's scales;
- image position scatter: the true lens's residual vectors on the catalog (table ``true_lens_residuals``), measured
  over image pairs and drawn in both panels.

Before the differences, the low-order modes the likelihood projects out of the field (``gp_project_modes``) are
removed by least squares from d_k and from g_k, separately and over each panel's own points.  The left panel uses
every pair of catalog images, the right panel a fixed random set of pairs on a regular grid over the whole true map,
its step the smallest of ``GRID_STEPS_ARCSEC`` that keeps the grid within ``GRID_POINT_BUDGET`` points.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io import store
from .style import BLUE, GREEN, ORANGE, PAPER_STYLE

STYLE = Path(__file__).with_name("style.mplstyle")
GRID_POINT_BUDGET = 60_000      # most grid points: 1" on Hera's map (51k points), 1.5" on Ares's (41k)
GRID_STEPS_ARCSEC = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0)
MAP_MARGIN_PIXELS = 2           # grid points stay this many map pixels inside the map's edge
GRID_PAIRS = 4_000_000          # random grid-point pairs, the same for every draw and field
GRID_PAIR_SEED = 12345
MIN_PAIRS = 25                  # bins are merged until each holds this many pairs
IMAGE_BINS = 14                 # log bins before merging
GRID_BINS = 14
MIN_IMAGE_SEPARATION = 0.3      # arcsec, lower edge of the image bins at the least
ANALYTIC_START_ARCSEC = 0.05    # the analytic GP curve starts here, well below the grid step and ell_gp
PANEL_IN = 2.8                  # side of each square panel, inches


# ------------------------------------------------------------------ pure statistics
def pair_geometry(xy: np.ndarray, pairs: np.ndarray | None = None):
    """Pairs (i, j) of the points ``xy`` (point, 2) (all pairs i < j, or the given ``pairs`` (pair, 2)), their
    separations r, unit separation vectors n (from i to j) and the perpendicular unit vectors t."""
    xy = np.asarray(xy, dtype=float)
    i, j = np.triu_indices(len(xy), 1) if pairs is None else (np.asarray(pairs)[:, 0], np.asarray(pairs)[:, 1])
    sep = xy[j] - xy[i]
    r = np.hypot(*sep.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = sep / r[:, None]
    t = np.stack([-n[:, 1], n[:, 0]], 1)
    return i, j, r, n, t


def log_edges(r: np.ndarray, lo: float, hi: float, bins: int, min_pairs: int = MIN_PAIRS) -> np.ndarray:
    """``bins`` log-spaced edges from ``lo`` to ``hi``, merged from the small-separation end until every bin holds at
    least ``min_pairs`` of the separations ``r`` (or two bins are left)."""
    edges = list(np.geomspace(lo, hi, bins + 1))
    while True:
        counts = np.histogram(r, edges)[0]
        bad = np.flatnonzero(counts < min_pairs)
        if not len(bad) or len(edges) <= 3:
            return np.asarray(edges)
        k = bad[0]
        del edges[k + 1 if k + 1 < len(edges) - 1 else k]


def binned_mean(values: np.ndarray, r: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of ``values`` over the pairs whose separation ``r`` falls in each bin of ``edges``, leaving out nonfinite
    values; NaN for an empty bin."""
    b = np.digitize(r, edges) - 1
    ok = (b >= 0) & (b < len(edges) - 1) & np.isfinite(values)
    s = np.bincount(b[ok], values[ok], len(edges) - 1)
    c = np.bincount(b[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c


def relative_displacement(v: np.ndarray, geometry, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S_L, S_T (units of ``v``) and the pair counts per bin of ``edges``: the rms over each bin's pairs of
    v_i - v_j projected on the separation vector and across it.  ``geometry`` is :func:`pair_geometry` of the points
    of ``v`` (point, 2); pairs with a nonfinite vector are left out."""
    i, j, r, n, t = geometry
    delta = np.asarray(v, dtype=float)[j] - np.asarray(v, dtype=float)[i]
    longitudinal, transverse = np.sum(delta * n, 1), np.sum(delta * t, 1)
    good = np.isfinite(longitudinal) & np.isfinite(transverse)
    counts = np.histogram(r[good], edges)[0]
    return (np.sqrt(binned_mean(np.where(good, longitudinal ** 2, np.nan), r, edges)),
            np.sqrt(binned_mean(np.where(good, transverse ** 2, np.nan), r, edges)), counts)


def mode_basis(xy: np.ndarray, modes) -> np.ndarray | None:
    """Least-squares design matrix (2 point, mode) of the low-order deflection modes ``modes`` at ``xy`` about their
    mean (``likelihood.gp_knee.low_order_basis``), rows in ravelled (point, component) order; None without modes."""
    if not modes:
        return None
    from ..likelihood.gp_knee import low_order_basis
    xy = np.asarray(xy, dtype=float)
    basis, _ = low_order_basis(xy[None], np.ones((1, len(xy)), bool), tuple(modes))
    return np.asarray(basis, dtype=float)[0].reshape(2 * len(xy), -1)


def remove_modes(v: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
    """``v`` (point, 2) minus its least-squares fit by the columns of ``basis`` (:func:`mode_basis`), fitted over the
    points where ``v`` is finite (the others stay NaN); ``v`` itself when ``basis`` is None."""
    v = np.asarray(v, dtype=float)
    if basis is None:
        return v
    finite = np.repeat(np.isfinite(v).all(1), 2)
    coefficients = np.linalg.lstsq(basis[finite], v.ravel()[finite], rcond=None)[0]
    return v - (basis @ coefficients).reshape(v.shape)


def gp_structure(r: np.ndarray, sigma_gp: float, ell_gp: float) -> np.ndarray:
    """Relative displacement of a squared-exponential GP of amplitude ``sigma_gp`` and length ``ell_gp`` per
    component, sqrt(2 sigma^2 (1 - exp(-r^2 / 2 ell^2))), the same along and across the separation."""
    if not sigma_gp > 0 or not np.isfinite(ell_gp):
        return np.zeros_like(np.asarray(r, dtype=float))
    return np.sqrt(2 * sigma_gp ** 2 * (1 - np.exp(-np.asarray(r, dtype=float) ** 2 / (2 * ell_gp ** 2))))


def isotropic(s_l: np.ndarray, s_t: np.ndarray) -> np.ndarray:
    """One isotropic relative displacement per bin, sqrt((S_L^2 + S_T^2) / 2)."""
    return np.sqrt((np.asarray(s_l) ** 2 + np.asarray(s_t) ** 2) / 2)


# ------------------------------------------------------------------ inputs
def true_deflection(deflection_fits, cosmology, z_reference: float, amplitude: float):
    """A function giving the true map's deflection (model frame, arcsec) at points (point, 2) for a source at
    ``z_reference``: the map's deflection (``diagnostics.true_lens.DeflectionMap``, its own ``ZS``) times
    ``amplitude`` times D_ls/D_s at ``z_reference`` over D_ls/D_s at ``ZS``; NaN outside the map.  Also the map."""
    from ..diagnostics.true_lens import DeflectionMap
    lens = DeflectionMap(deflection_fits)
    efficiency = lambda zs: float((cosmology.angular_diameter_distance(lens.z_lens, zs) / cosmology.angular_diameter_distance(zs)).value)
    scale = float(amplitude) * efficiency(z_reference) / efficiency(lens.z_source)

    def alpha(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        theta = np.column_stack([-points[:, 0], points[:, 1]])
        beta, _ = lens(theta, np.ones(len(points)))
        deflection = (theta - beta) * scale
        return np.column_stack([-deflection[:, 0], deflection[:, 1]])
    return alpha, lens, scale


def map_extent(lens, margin: int = MAP_MARGIN_PIXELS) -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max), model frame in arcsec, of the true map ``lens`` (``DeflectionMap``, map x = -model
    x) less ``margin`` pixels on every side."""
    x0, y0 = lens.origin
    lo_x, hi_x = x0 + margin * lens.step, x0 + (lens.shape[1] - 1 - margin) * lens.step
    lo_y, hi_y = y0 + margin * lens.step, y0 + (lens.shape[0] - 1 - margin) * lens.step
    return -hi_x, -lo_x, lo_y, hi_y


def grid_step(extent, budget: int = GRID_POINT_BUDGET, steps=GRID_STEPS_ARCSEC) -> float:
    """The smallest of ``steps`` whose grid over ``extent`` (:func:`map_extent`) has at most ``budget`` points."""
    x_min, x_max, y_min, y_max = extent
    for step in steps:
        if (np.floor((x_max - x_min) / step) + 1) * (np.floor((y_max - y_min) / step) + 1) <= budget:
            return float(step)
    return float(steps[-1])


def map_grid(extent, step: float) -> np.ndarray:
    """A regular grid (row, column, 2) of spacing ``step`` (arcsec) centred in ``extent`` (:func:`map_extent`)."""
    x_min, x_max, y_min, y_max = extent
    axes = []
    for lo, hi in ((x_min, x_max), (y_min, y_max)):
        count = int(np.floor((hi - lo) / step)) + 1
        start = (lo + hi) / 2 - (count - 1) * step / 2
        axes.append(start + step * np.arange(count))
    return np.stack(np.meshgrid(*axes), -1)


def grid_pairs(count: int, n_pairs: int = GRID_PAIRS, seed: int = GRID_PAIR_SEED) -> np.ndarray:
    """A fixed-seed random set of pairs (pair, 2) of distinct points out of ``count``."""
    pairs = np.random.default_rng(seed).integers(0, count, (n_pairs, 2))
    return pairs[pairs[:, 0] != pairs[:, 1]]


# ------------------------------------------------------------------ the product
def _rows(panel, curve, draw, row, edges, s_l, s_t, counts):
    """Table rows of one curve, from the output of :func:`relative_displacement`."""
    centres = np.sqrt(edges[1:] * edges[:-1])
    return [dict(panel=panel, curve=curve, draw=int(draw), posterior_row=int(row), bin=b, r_lo=float(edges[b]),
                 r_hi=float(edges[b + 1]), r_centre=float(centres[b]), n_pairs=int(counts[b]),
                 S_L_arcsec=float(s_l[b]), S_T_arcsec=float(s_t[b])) for b in range(len(centres))]


def relative_displacement_table(state, evaluator, draws, residual_draws: pd.DataFrame, true_table: pd.DataFrame,
                                alpha_true, lens_map, *, log=print) -> tuple[pd.DataFrame, dict]:
    """The table of every curve (columns ``panel`` (images, grid), ``curve`` (truth, mass_field, gp, gp_analytic,
    image_position_scatter), ``draw``, ``posterior_row``, ``bin``, ``r_lo``, ``r_hi``, ``r_centre``, ``n_pairs``,
    ``S_L_arcsec``, ``S_T_arcsec``) and a provenance record.  ``gp`` rows count the image draws, ``image_position_scatter``
    rows have draw and row -1, and ``gp_analytic`` rows are points of a fine separation axis (``r_centre``; no
    edges, no pairs, S_L = S_T)."""
    from ..likelihood.grf_fields import shared_field_evaluator
    from ..diagnostics.image_recovery import image_plane_scales
    z_ref = float(state.gp_reference_redshift)
    modes = tuple(getattr(shared_field_evaluator(evaluator), "gp_project_modes", ()) or ())
    images = np.column_stack([np.asarray(state.images.x_obs, float), np.asarray(state.images.y_obs, float)])
    extent = map_extent(lens_map)
    step = grid_step(extent)
    mesh = map_grid(extent, step)
    grid = mesh.reshape(-1, 2)
    points = {"images": images, "grid": grid}
    inputs = {"images": images, "grid": mesh}        # the grid as a meshgrid: the field's separable evaluation
    truth = {name: alpha_true(p) for name, p in points.items()}
    basis = {name: mode_basis(p, modes) for name, p in points.items()}
    geometry = {"images": pair_geometry(images), "grid": pair_geometry(grid, grid_pairs(len(grid)))}
    r_images, r_grid = geometry["images"][2], geometry["grid"][2]
    edges = {"images": log_edges(r_images, max(r_images.min() * .999, MIN_IMAGE_SEPARATION), r_images.max() * 1.001, IMAGE_BINS),
             "grid": log_edges(r_grid, step * .999, r_grid.max() * 1.001, GRID_BINS)}
    rows, n_nonfinite = [], 0
    for k in range(len(draws)):
        row = int(draws.indices[k])
        for name, p in inputs.items():
            field = np.asarray(draws.field(k, p, z_ref)[0], dtype=float)
            parametric = np.asarray(draws.lens(k, p, z_ref)[0], dtype=float) - field
            if name == "grid":
                n_nonfinite = max(n_nonfinite, int((~np.isfinite(parametric + field).all(1)).sum()))
            for curve, v in (("truth", truth[name] - parametric), ("mass_field", field)):
                rows += _rows(name, curve, k, row, edges[name], *relative_displacement(remove_modes(v, basis[name]), geometry[name], edges[name]))
    # Image-plane GP at the images: each image draw's true minus lens position, nothing removed.
    for j, (row, table) in enumerate(residual_draws.groupby("posterior_index", sort=True)):
        xy = table[["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(float)
        gp = (table[["x_true_arcsec", "y_true_arcsec"]].to_numpy(float) - table[["x_model_arcsec", "y_model_arcsec"]].to_numpy(float))
        rows += _rows("images", "gp", j, row, edges["images"], *relative_displacement(gp, pair_geometry(xy), edges["images"]))
    # Image-plane GP on the grid: the analytic curve of each product draw's scales on a fine axis.
    scales = [image_plane_scales(evaluator, draws.samples[k]) for k in range(len(draws))]
    axis = np.geomspace(ANALYTIC_START_ARCSEC, edges["grid"][-1], 160)
    for k, s in enumerate(scales):
        value = gp_structure(axis, s["sigma_gp"], s["ell_gp"])
        rows += [dict(panel="grid", curve="gp_analytic", draw=k, posterior_row=int(draws.indices[k]), bin=b, r_lo=np.nan,
                      r_hi=np.nan, r_centre=float(r), n_pairs=0, S_L_arcsec=float(v), S_T_arcsec=float(v))
                 for b, (r, v) in enumerate(zip(axis, value))]
    # Image position scatter of the true lens (one truth, no draws): catalog minus true-lens image.
    used = (true_table["status"] == "used").to_numpy()
    scatter_xy = true_table.loc[used, ["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(float)
    scatter = -true_table.loc[used, ["dx_arcsec", "dy_arcsec"]].to_numpy(float)
    rows += _rows("images", "image_position_scatter", -1, -1, edges["images"],
                  *relative_displacement(scatter, pair_geometry(scatter_xy), edges["images"]))
    record = dict(reference_redshift=z_ref, modes=list(modes), grid_step_arcsec=float(step), n_image_points=int(len(images)),
                  n_image_pairs=int(len(r_images)), n_grid_points=int(len(grid)), n_grid_pairs=int(len(r_grid)), grid_extent_arcsec=[float(v) for v in extent],
                  grid_shape=list(mesh.shape[:2]), grid_point_budget=GRID_POINT_BUDGET, n_grid_points_nonfinite=n_nonfinite,
                  grid_pair_seed=GRID_PAIR_SEED, min_pairs=MIN_PAIRS, image_edges_arcsec=edges["images"].tolist(),
                  grid_edges_arcsec=edges["grid"].tolist(), posterior_rows=[int(v) for v in draws.indices],
                  image_draw_rows=[int(v) for v in sorted(residual_draws["posterior_index"].unique())],
                  n_image_position_scatter_images=int(used.sum()),
                  sigma_gp_arcsec=[s["sigma_gp"] for s in scales], ell_gp_arcsec=[s["ell_gp"] for s in scales])
    return pd.DataFrame(rows), record


def _band(table: pd.DataFrame, panel: str, curve: str, column: str):
    s = table[(table.panel == panel) & (table.curve == curve)]
    r = s.groupby("bin").r_centre.first().sort_index().to_numpy()
    values = s.pivot(index="bin", columns="draw", values=column).sort_index().to_numpy()
    return r, values


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def render_relative_displacement(table: pd.DataFrame, path: Path) -> Path:
    """Two square panels (image pairs, grid) of the relative displacement against separation: median over the
    reporting draws; truth dashed, mass field solid with its 16-84% band, longitudinal blue, transverse orange, the
    image-plane GP green with its band and the image position scatter as open circles."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    left, right, bottom, top, gap = .55, .1, .45, .62, .1
    width, height = left + 2 * PANEL_IN + gap + right, bottom + PANEL_IN + top
    fig = plt.figure(figsize=(width, height))
    axes = [fig.add_axes([(left + c * (PANEL_IN + gap)) / width, bottom / height, PANEL_IN / width, PANEL_IN / height])
            for c in range(2)]
    scatter = table[(table.panel == "images") & (table.curve == "image_position_scatter")]

    axes[1].sharey(axes[0])
    alpha = .2
    ymax = 0.
    for ax, panel, label in zip(axes, ("images", "grid"), ("Image pairs", "Grid")):
        # Colour is the component, line style truth (dashed, drawn on top) or mass field (solid, with its band).
        for curve, style, z in (("mass_field", "-", 2), ("truth", "--", 4)):
            for component, color in (("L", BLUE), ("T", ORANGE)):
                r, values = _band(table, panel, curve, f"S_{component}_arcsec")
                lo, med, hi = np.nanpercentile(values, [16, 50, 84], axis=1)
                if curve == "mass_field":
                    ax.fill_between(r, lo, hi, color=color, alpha=alpha, lw=0)
                ymax = max(ymax, np.nanmax(hi if curve == "mass_field" else med))
                ax.plot(r, med, color=color, ls=style, lw=1.2, zorder=z)
        gp = "gp" if panel == "images" else "gp_analytic"
        r, s_l = _band(table, panel, gp, "S_L_arcsec")
        _, s_t = _band(table, panel, gp, "S_T_arcsec")
        lo, med, hi = np.nanpercentile(isotropic(s_l, s_t), [16, 50, 84], axis=1)
        ax.fill_between(r, lo, hi, color=GREEN, alpha=alpha, lw=0)
        ax.plot(r, med, color=GREEN, lw=1.2)
        ymax = max(ymax, np.nanmax(hi))
        # The image position scatter exists only at the images: the same image-pair points in both panels.
        s = scatter
        value = isotropic(s.S_L_arcsec.to_numpy(), s.S_T_arcsec.to_numpy())
        ax.plot(s.r_centre, value, ls="none", marker="o", ms=3.5, mfc="white", mec="k", mew=.9, zorder=5)
        ymax = max(ymax, np.nanmax(value))
        if panel == "images":
            lo_r, hi_r = s.r_lo.min(), s.r_hi.max()
        else:
            lo_r = r.min()
            hi_r = max(table[(table.panel == panel) & (table.curve == "truth")].r_hi.max(), s.r_centre.max())
        ax.set_xscale("log")
        ax.set_xlim(lo_r * .9, hi_r * 1.1)
        # A panel spanning less than two decades (the image pairs) gets labels at 1, 2 and 5 of each decade.
        subs = (1, 2, 5) if hi_r / lo_r < 100 else (1,)
        ax.xaxis.set_major_locator(LogLocator(subs=subs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(LogLocator(subs=[s for s in range(2, 10) if s not in subs]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Separation (arcsec)")
        ax.text(.05, .95, label, transform=ax.transAxes, ha="left", va="top")
    axes[0].set_ylim(0, ymax * 1.05)
    axes[0].set_ylabel("Relative displacement (arcsec)")
    axes[1].tick_params(labelleft=False)
    handles = [Line2D([], [], color=".45", lw=1.2, ls="--"), Line2D([], [], color=".45", lw=1.2),
               Line2D([], [], color=BLUE, lw=1.2), Line2D([], [], color=ORANGE, lw=1.2),
               (Patch(color=GREEN, alpha=alpha, lw=0), Line2D([], [], color=GREEN, lw=1.2)),
               Line2D([], [], ls="none", marker="o", ms=3.5, mfc="white", mec="k", mew=.9)]
    labels = ["True lens minus parametric lens", "Mass-field realization", "Longitudinal", "Transverse", "Image-plane GP",
              "Image position scatter (true lens)"]
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=((left + PANEL_IN + gap / 2) / width, 1.0), ncol=3,
               frameon=False, handlelength=2.0, columnspacing=1.2, handletextpad=.5)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_relative_displacement(config, state, evaluator, draws, recovery: dict, true_lens: dict | None, stage: Path, *,
                               log=print) -> dict | None:
    """The numbers of ``relative_displacement.pdf`` for a run with a true deflection map: the table and record
    ``relative_displacement`` in the results of ``stage`` (one batch), and the record returned.  ``recovery`` holds the
    image-recovery tables (``image_fit_quality``, ``posterior_image_residual_draws``); ``true_lens`` is the record
    ``true_lens_rms`` (``outputs.images.write_true_lens_rms``), whose table ``true_lens_residuals`` is read from the
    stage.  Report-only: any missing input or an unreadable map logs one line and returns None, and nothing is
    written."""
    image_df = recovery.get("image_fit_quality") if recovery else None
    residual_draws = recovery.get("posterior_image_residual_draws") if recovery else None
    reason = None
    if not config.outputs.truth.deflection_true_fits:
        reason = "no outputs.truth.deflection_true_fits"
    elif true_lens is None:
        reason = "no true-lens record"
    elif not draws.has_field:
        reason = "no shared mass field"
    elif image_df is None or image_df.empty or residual_draws is None or residual_draws.empty:
        reason = "no image table"
    if reason is not None:
        log(f"[outputs] relative displacement skipped: {reason}")
        return None
    from ..diagnostics.exact_images import astropy_cosmology
    start = time.perf_counter()
    try:
        true_table = store.read_table(stage, "true_lens_residuals")
        alpha_true, lens_map, scale = true_deflection(config.outputs.truth.deflection_true_fits, astropy_cosmology(state.cosmo_config),
                                               float(state.gp_reference_redshift), float(true_lens["amplitude"]))
    except (OSError, KeyError, IndexError, ValueError) as error:
        log(f"[outputs] relative displacement skipped: {type(error).__name__}: {error}")
        return None
    table, record = relative_displacement_table(state, evaluator, draws, residual_draws, true_table, alpha_true, lens_map, log=log)
    record.update(amplitude=float(true_lens["amplitude"]), scale_to_reference=scale, deflection_fits=str(config.outputs.truth.deflection_true_fits))
    with store.batch(stage):
        store.write_table(stage, "relative_displacement", table)
        store.write_record(stage, "relative_displacement", record)
    log(f"[outputs] relative displacement: {len(draws)} draws, {record['n_image_points']} images, {record['n_grid_points']} grid points "
        f"({record['grid_step_arcsec']:g}\" step over the true map, {record['n_grid_pairs']} pairs), modes removed: {', '.join(record['modes']) or 'none'}; "
        f"{time.perf_counter() - start:.0f} s")
    return record
