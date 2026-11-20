"""Image-position residuals of the true lens of a simulated cluster (``truth.deflection_true_fits``).

The true-lens RMS is the rms over catalog images of the separation between each observed image and the image the
simulation's true deflection map gives, with a free source per family and one free amplitude on the deflection.  It
is a reference value, not a lower bound on a fitted model's RMS: that RMS is a per-draw statistic with each draw's own
sources and correction field, and single draws can go below it.  The map's deflection, at its ``ZS`` behind its
``ZL``, is scaled to each family's redshift by :math:`D_{ls}/D_s` in the run's cosmology and by the amplitude, chosen
to minimize the rms (0.4-0.8 per cent on the FF-SIMS maps, which a fit absorbs in its mass normalization).  Each
family's source minimizes its summed squared image-plane residuals (Gauss-Newton from the magnification-weighted
back-projection), and each image is solved exactly near its observed position by Newton iteration on the map.  The
map (``XMIN``..``YMAX`` in arcsec, primary HDU :math:`\\alpha_x`, first extension :math:`\\alpha_y`, values at pixel
centers) runs with x positive east: map x = -model x.

Images are left out, and counted by ``status``, when they have no redshift, lie outside the map or leave it during the
solve, do not converge, or converge farther than ``MATCH_LIMIT_ARCSEC`` from their catalog position (another image of
the source); families left with fewer than two images are left out too (a free source would fit them exactly).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar

AMPLITUDE_BOUNDS = (0.95, 1.05)
# Five times the per-axis catalog scatter of FF-SIMS (0.13-0.16"): a Gaussian residual beyond it has probability
# below 1e-8, while the fold partner a wandering solve would find is typically 1-3" away.
MATCH_LIMIT_ARCSEC = 1.0
STATUSES = ("used", "no_redshift", "outside_map", "not_converged", "not_matched", "single_image_family")


class DeflectionMap:
    """Bilinear interpolation of a true deflection map and its Jacobian, in the map frame (x = -model x); NaN
    outside the map."""

    def __init__(self, path: str | Path):
        with fits.open(path) as hdus:
            header, ax, ay = hdus[0].header, hdus[0].data.astype(float), hdus[1].data.astype(float)
        self.step = (float(header["XMAX"]) - float(header["XMIN"])) / ax.shape[1]
        self.origin = (float(header["XMIN"]) + self.step / 2, float(header["YMIN"]) + self.step / 2)
        self.shape = ax.shape
        self.z_lens, self.z_source = float(header["ZL"]), float(header["ZS"])
        (axy, axx), (ayy, ayx) = np.gradient(ax, self.step), np.gradient(ay, self.step)
        self.fields = (ax, ay, axx, axy, ayx, ayy)

    def inside(self, theta: np.ndarray) -> np.ndarray:
        col, row = (theta[:, 0] - self.origin[0]) / self.step, (theta[:, 1] - self.origin[1]) / self.step
        return (col >= 0) & (col <= self.shape[1] - 1) & (row >= 0) & (row <= self.shape[0] - 1)

    def __call__(self, theta: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Source positions (N, 2) and lens Jacobians A (N, 2, 2) at map-frame positions ``theta`` (N, 2), with the
        deflection multiplied by ``scale`` (N,)."""
        pixel = [(theta[:, 1] - self.origin[1]) / self.step, (theta[:, 0] - self.origin[0]) / self.step]
        ax, ay, axx, axy, ayx, ayy = (scale * map_coordinates(f, pixel, order=1, cval=np.nan) for f in self.fields)
        jacobian = np.stack([np.stack([1 - axx, -axy], -1), np.stack([-ayx, 1 - ayy], -1)], -2)
        return theta - np.stack([ax, ay], 1), jacobian


def _family_solve(family: np.ndarray, use: np.ndarray, magnification: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Per family (rows of ``family``, only the ``use`` ones), (sum M^T M)^-1 sum M^T v: the source shift that
    minimizes the image-plane residuals; zero for a family without used images."""
    out = np.zeros((family.max() + 1, 2))
    for f in np.unique(family[use]):
        i = use & (family == f)
        out[f] = np.linalg.solve(np.einsum("nji,njk->ik", magnification[i], magnification[i]),
                                 np.einsum("nji,nj->i", magnification[i], vectors[i]))
    return out


def _solve_images(lens, obs, beta, scale, newton: int = 30, tol: float = 1e-5):
    """Newton solve from the observed positions for the images of ``beta``: positions, magnification matrices and
    the status code of each image."""
    theta = obs.copy()
    for _ in range(newton):
        b, a = lens(theta, scale)
        theta = theta + np.clip(np.linalg.solve(np.where(np.isfinite(a), a, np.eye(2)), (beta - b)[..., None])[..., 0], -0.5, 0.5)
        theta[~np.isfinite(b).all(1)] = np.nan
    b, a = lens(theta, scale)
    status = np.full(len(obs), "used", dtype=object)
    status[np.hypot(*(b - beta).T) >= tol] = "not_converged"
    status[~np.isfinite(theta).all(1) | ~lens.inside(np.nan_to_num(theta, nan=np.inf))] = "outside_map"
    status[(status == "used") & (np.hypot(*(theta - obs).T) > MATCH_LIMIT_ARCSEC)] = "not_matched"
    return theta, np.linalg.inv(np.where(np.isfinite(a), a, np.eye(2))), status


def _single_families(family, status):
    used = status == "used"
    counts = np.bincount(family[used], minlength=family.max() + 1)
    status[used & (counts[family] < 2)] = "single_image_family"
    return status


def _residuals(lens: DeflectionMap, obs: np.ndarray, family: np.ndarray, scale: np.ndarray, start: np.ndarray, *, outer: int = 4):
    """Residual vectors (N, 2) of the exact true-lens images at the final best source per family and the status of
    each image (only ``start == "used"`` images are solved)."""
    use0 = start == "used"
    beta_obs, a_obs = lens(obs, scale)
    m_obs = np.linalg.inv(np.where(np.isfinite(a_obs), a_obs, np.eye(2)))
    beta = _family_solve(family, use0, m_obs, np.einsum("nij,nj->ni", m_obs, beta_obs))[family]
    for k in range(outer + 1):
        theta, m, status = _solve_images(lens, obs, beta, scale)
        status = _single_families(family, np.where(use0, status, start))
        residual = theta - obs
        if k < outer:
            beta = beta - _family_solve(family, status == "used", m, np.nan_to_num(residual))[family]
    return residual, status


def true_lens_residuals(images: pd.DataFrame, deflection_fits: str | Path, cosmology) -> tuple[pd.DataFrame, float]:
    """Per-image residuals of the true lens (``family_id``, ``x_obs_arcsec``, ``y_obs_arcsec``, ``z_source`` of every
    catalog image, model frame) and the fitted amplitude.  Columns added: ``status`` (one of ``STATUSES``),
    ``dx_arcsec``, ``dy_arcsec`` (model minus observed, model frame), ``true_lens_residual_arcsec`` (NaN unless used)
    and ``magnification_true`` at the observed position."""
    lens = DeflectionMap(deflection_fits)
    z = images["z_source"].to_numpy(float)
    obs = np.stack([-images["x_obs_arcsec"].to_numpy(float), images["y_obs_arcsec"].to_numpy(float)], 1)
    family = pd.factorize(images["family_id"].astype(str))[0]
    start = np.full(len(images), "used", dtype=object)
    start[~lens.inside(obs)] = "outside_map"
    start[~(np.isfinite(z) & (z > lens.z_lens))] = "no_redshift"
    efficiency = lambda zs: (cosmology.angular_diameter_distance(lens.z_lens, zs) / cosmology.angular_diameter_distance(zs)).value
    ratio = np.where(start == "no_redshift", 1.0, efficiency(np.where(start == "no_redshift", lens.z_source, z)) / efficiency(lens.z_source))

    def rms(amplitude):
        r, status = _residuals(lens, obs, family, amplitude * ratio, start)
        used = status == "used"
        return float(np.sqrt(np.mean(np.sum(r[used] ** 2, 1)))) if used.any() else np.inf

    amplitude = float(minimize_scalar(rms, bounds=AMPLITUDE_BOUNDS, method="bounded", options={"xatol": 1e-5}).x)
    residual, status = _residuals(lens, obs, family, amplitude * ratio, start)
    _, a_obs = lens(obs, amplitude * ratio)
    used = status == "used"
    table = images[[c for c in ("family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "z_source") if c in images]].copy()
    table["status"] = status
    table["dx_arcsec"], table["dy_arcsec"] = np.where(used, -residual[:, 0], np.nan), np.where(used, residual[:, 1], np.nan)
    table["true_lens_residual_arcsec"] = np.where(used, np.hypot(*np.nan_to_num(residual).T), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        table["magnification_true"] = 1.0 / np.linalg.det(a_obs)   # NaN outside the map
    return table.reset_index(drop=True), amplitude


def true_lens_rms(table: pd.DataFrame, *, resamples: int = 2000, seed: int = 0) -> dict:
    """The rms of ``true_lens_residual_arcsec`` over the used images, its 16th and 84th percentiles under a bootstrap
    over families (families drawn with replacement, all their images kept), and the images left out by status.

    Raises
    ------
    ValueError
        If no image is used.
    """
    good = table[table["status"] == "used"]
    if good.empty:
        raise ValueError(f"no usable image ({table['status'].value_counts().to_dict()})")
    r2 = good["true_lens_residual_arcsec"].to_numpy(float) ** 2
    family = pd.factorize(good["family_id"].astype(str))[0]
    sums, counts = np.bincount(family, r2), np.bincount(family)
    picks = np.random.default_rng(seed).integers(0, len(counts), (resamples, len(counts)))
    boot = np.sqrt(sums[picks].sum(1) / counts[picks].sum(1))
    p16, p84 = np.percentile(boot, [16, 84])
    dropped = {f"n_{s}": int((table["status"] == s).sum()) for s in STATUSES[1:]}
    return {"rms_arcsec": float(np.sqrt(r2.mean())), "p16_arcsec": float(p16), "p84_arcsec": float(p84),
            "n_images": int(len(good)), "n_catalog_images": int(len(table)), "n_families": int(len(counts)), **dropped,
            "match_limit_arcsec": MATCH_LIMIT_ARCSEC, "bootstrap_resamples": resamples, "bootstrap_seed": seed}
