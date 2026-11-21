"""Exact image prediction for diagnostics: solve the lens equation per family and match images."""
from __future__ import annotations

import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..config import ImageSolverConfig
from ..likelihood.evaluator import Evaluator
from ..core.model import (CORED_POWER_LAW_PROFILE, DP_IE_PROFILE, MASS_SHEET_PROFILE, NFW_PROFILE, SHEAR_PROFILE, FamilyData,
                          PackedLensState)

_TRUST_RADIUS_DEFAULT = 0.3
_LM_DAMPING_RELATIVE = 1.0e-3
_LM_DAMPING_ABSOLUTE = 1.0e-6
_DET_GUARD = 1.0e-30
# Global-search starts of one family closer than this (arcsec) iterate identically and are merged; far below the
# root separation (exact_image_root_separation_arcsec, 1e-4").
MERGE_ARCSEC = 1.0e-9
# Largest Newton step (arcsec) while polishing a candidate root: keeps a polish inside the basin it started in.
POLISH_STEP_ARCSEC = 0.01
# Profiles whose deflection is linear in the source's lensing efficiency (dPIE and cored power-law sigma0, NFW alpha_rs,
# mass-sheet kappa): at one lens plane their fields at any source redshift are those at a reference redshift times one
# scalar.  The external shear (SHEAR_PROFILE) does not depend on the source and is evaluated apart.
SCALED_PROFILES = frozenset({DP_IE_PROFILE, CORED_POWER_LAW_PROFILE, NFW_PROFILE, MASS_SHEET_PROFILE})
# Largest difference allowed, relative to the field's size, between the lens at a family's redshift and the scaled
# reference lens at the probe points of the per-draw check; the scaling holds to ~1e-15, a missing term fails by %.
SCALING_CHECK_RTOL = 1.0e-10


def scaled_lens_factors(evaluator: Evaluator, redshifts, reference: float) -> np.ndarray | None:
    """The factors s(z) = (dPIE lensing factor at z) / (at ``reference``), one per entry of ``redshifts``, that take
    the scaled components (``SCALED_PROFILES``) of the lens at ``reference`` to each redshift at one lens plane; the
    external shear (``SHEAR_PROFILE``) is added unscaled.  None when the lens does not scale so: a sampled cosmology or
    a profile outside ``SCALED_PROFILES`` and the shear."""
    codes = frozenset(int(code) for code in np.unique(np.asarray(evaluator.state.packed_lens_spec.profile_type)))
    if evaluator.cosmology_sampled or not codes <= SCALED_PROFILES | {SHEAR_PROFILE}:
        return None
    from ..likelihood.grf_fields import _sigma0_factor
    redshifts = [float(z) for z in np.atleast_1d(np.asarray(redshifts, dtype=float))]
    reference = float(reference)
    factor = {z: _sigma0_factor(evaluator.state, z) for z in sorted(set(redshifts) | {reference})}
    return np.array([factor[z] / factor[reference] for z in redshifts])


def astropy_cosmology(cosmo_config: dict[str, Any]):
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM
    h0, om0 = float(cosmo_config["H0"]), float(cosmo_config["Om0"])
    if str(cosmo_config.get("class", "FlatLambdaCDM")) == "FlatwCDM":
        return FlatwCDM(H0=h0, Om0=om0, w0=float(cosmo_config.get("w0", -1.0)))
    return FlatLambdaCDM(H0=h0, Om0=om0)


class ExactImageSolver:
    """Solve the lens equation for every family at a latent vector and match the model images.

    One solver may serve several threads at once (the posterior draws of ``image_recovery``).  ``timing`` then sums
    the time of every call over all threads (seconds of solver time, more than the wall time of the threads together),
    and ``grid_points``, ``seed_count`` and ``search_path`` are those of the calling thread's last global search.
    ``search_counts`` counts the global searches by path (``"shared"``, ``"per_redshift"``).
    """

    # The global search uses one lattice and one batch over all families when the lens scales with the source
    # redshift (``solve_all_global``); False forces the search per source redshift (kept for comparison and tests).
    shared_lattice = True

    def __init__(self, evaluator: Evaluator, config: ImageSolverConfig) -> None:
        self.evaluator = evaluator
        self.config = config
        self.timing: dict[str, float] = {"exact_solver": 0.0, "global_grid": 0.0, "global_refine": 0.0}
        self.all_components = np.arange(evaluator.n_components, dtype=np.int32)
        self._per_image_fields = None
        self._lock = threading.Lock()
        self._last_search = threading.local()
        self.search_counts = {"shared": 0, "per_redshift": 0}
        profile = np.asarray(evaluator.state.packed_lens_spec.profile_type)
        self._scaled_components = np.flatnonzero(profile != SHEAR_PROFILE).astype(np.int32)
        self._shear_components = np.flatnonzero(profile == SHEAR_PROFILE).astype(np.int32)
        self._profile_codes = frozenset(int(code) for code in np.unique(profile))

        def split(points, packed_state):
            """Summed deflection and Jacobian of the scaled components and of the shear, separately."""
            out = []
            for components in (self._scaled_components, self._shear_components):
                if components.size:
                    alpha_rows, u_rows = evaluator.component_rows(points, packed_state, components)
                    out += [jnp.sum(alpha_rows, axis=0), jnp.sum(u_rows, axis=0)]
                else:
                    out += [jnp.zeros((points.shape[0], 2)), jnp.zeros((points.shape[0], 2, 2))]
            return tuple(out)
        self._split_fields_jit = jax.jit(split)          # created here, before any worker thread calls it

    def _add_time(self, key: str, started: float) -> None:
        """Add the time since ``started`` to ``timing[key]`` (under a lock: threads share the solver)."""
        elapsed = time.perf_counter() - started
        with self._lock:
            self.timing[key] += elapsed

    @property
    def grid_points(self) -> int:
        """Points ray-shot by the calling thread's last global search (coarse grid plus fine sub-grids)."""
        return getattr(self._last_search, "grid_points", 0)

    @property
    def seed_count(self) -> int:
        """Seeds refined by the calling thread's last global search."""
        return getattr(self._last_search, "seed_count", 0)

    @property
    def search_path(self) -> str:
        """``"shared"`` or ``"per_redshift"``: how the calling thread's last global search ran ("" before any)."""
        return getattr(self._last_search, "path", "")

    # ------------------------------------------------------------------ lens fields at arbitrary points
    def ray_shoot_and_jacobian(self, packed: PackedLensState, x: np.ndarray, y: np.ndarray):
        xy = np.stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)), -1)
        alpha, u = self.evaluator.total_fields_at(xy, packed, self.all_components)
        a = np.eye(2) - u
        return (xy[:, 0] - alpha[:, 0], xy[:, 1] - alpha[:, 1], a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1])

    def ray_shoot_and_jacobian_per_image(self, packed_images: PackedLensState, x: np.ndarray, y: np.ndarray):
        """As ray_shoot_and_jacobian, for the N constrained images at once with the per-image packed state (normalizations of
        shape (component, image), so every point carries its own source redshift); one jitted call per iteration."""
        with self._lock:
            if self._per_image_fields is None:
                components = self.all_components

                def fields(xy, packed_state):
                    alpha_rows, u_rows = self.evaluator.component_rows(xy, packed_state, components)
                    return jnp.sum(alpha_rows, axis=0), jnp.sum(u_rows, axis=0)
                self._per_image_fields = jax.jit(fields)
        xy = jnp.stack((jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)), -1)
        alpha, u = self._per_image_fields(xy, packed_images)
        alpha, a = np.asarray(alpha), np.eye(2) - np.asarray(u)
        return (np.asarray(x, dtype=float) - alpha[:, 0], np.asarray(y, dtype=float) - alpha[:, 1], a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1])

    def magnification(self, packed: PackedLensState, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        _, _, a00, a01, a10, a11 = self.ray_shoot_and_jacobian(packed, x, y)
        return 1.0 / (a00 * a11 - a01 * a10)

    # ------------------------------------------------------------------ local Levenberg--Marquardt solver
    def _lm_step(self, fx, fy, a00, a01, a10, a11):
        finite = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(a00) & np.isfinite(a01) & np.isfinite(a10) & np.isfinite(a11)
        n00 = a00 ** 2 + a10 ** 2
        n01 = a00 * a01 + a10 * a11
        n11 = a01 ** 2 + a11 ** 2
        lam = _LM_DAMPING_ABSOLUTE + _LM_DAMPING_RELATIVE * np.maximum(n00 + n11, 0.0)
        h00, h11, h01 = n00 + lam, n11 + lam, n01
        rhs0 = -(a00 * fx + a10 * fy)
        rhs1 = -(a01 * fx + a11 * fy)
        det = np.maximum(h00 * h11 - h01 ** 2, _DET_GUARD)
        dx = (h11 * rhs0 - h01 * rhs1) / det
        dy = (-h01 * rhs0 + h00 * rhs1) / det
        radius = np.sqrt(dx ** 2 + dy ** 2 + 1.0e-30)
        trust = float(self.config.exact_image_lm_trust_radius_arcsec)
        if trust > 0.0:
            scaled = radius / trust
            scale = np.tanh(scaled) / np.maximum(scaled, 1.0e-12)
            dx, dy = dx * scale, dy * scale
        return dx, dy, finite & np.isfinite(dx) & np.isfinite(dy)

    @staticmethod
    def _corrected(shoot, correction):
        """``shoot`` with a deflection correction added to the lens: beta = theta - alpha - delta, A = I - d alpha - d delta.
        ``correction(x, y)`` returns (delta_x, delta_y, J00, J01, J10, J11) aligned with the points; None leaves ``shoot``."""
        if correction is None:
            return shoot

        def shoot_corrected(packed, x, y):
            bx, by, a00, a01, a10, a11 = shoot(packed, x, y)
            dx, dy, j00, j01, j10, j11 = correction(x, y)
            return bx - dx, by - dy, a00 - j00, a01 - j01, a10 - j10, a11 - j11
        return shoot_corrected

    def _refine(self, packed, source_x, source_y, start_x, start_y, *, shoot=None, correction=None, groups=None) -> dict[str, np.ndarray]:
        """Batched Levenberg-Marquardt solve of the lens equation beta(theta) = source from the given starts.

        ``shoot(packed, x, y)`` returns the source positions and the Jacobian A = I - d alpha / d theta at (x, y); the
        default evaluates one scalar-normalized packed state, ``ray_shoot_and_jacobian_per_image`` one with a
        normalization per point.  Every start is iterated until its step and source residual are below the tolerances
        or the iteration budget is spent; starts that produce non-finite values are frozen.  Returns per start: the
        final position, whether it converged, whether it passes the identification tolerance (the Newton step at the
        final position is small, so it sits on a root), the iteration count and the final step and source residual.
        ``correction(x, y)`` adds a deflection field to the lens (a realization of the GRF), aligned with the
        starts.

        With ``groups`` (an integer per start, e.g. its family; requires a ``packed`` state shared by all starts and
        ``correction(x, y, index)`` taking the start indices) only the starts still iterating are evaluated, and starts
        of one group that reach the same position (to ``MERGE_ARCSEC``) are merged: from there their iterations would
        be identical, so the merged starts get the result of the one that continues."""
        if groups is not None:
            return self._refine_merged(packed, source_x, source_y, start_x, start_y, groups, correction)
        cfg = self.config
        shoot = self._corrected(self.ray_shoot_and_jacobian if shoot is None else shoot, correction)
        x, y = np.array(start_x, dtype=float), np.array(start_y, dtype=float)
        source_x, source_y = np.broadcast_to(source_x, x.shape).astype(float), np.broadcast_to(source_y, y.shape).astype(float)
        alive = np.isfinite(x) & np.isfinite(y)            # still iterating and finite so far
        converged = np.zeros(x.shape, dtype=bool)
        iterations = np.zeros(x.shape, dtype=int)
        source_tol = max(cfg.exact_image_identification_tol_arcsec, cfg.exact_image_precision_limit)
        for iteration in range(1, int(cfg.exact_image_lm_max_iter) + 1):
            bx, by, a00, a01, a10, a11 = shoot(packed, x, y)
            rx, ry = bx - source_x, by - source_y
            dx, dy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
            active = alive & ok & ~converged
            x, y = np.where(active, x + dx, x), np.where(active, y + dy, y)
            iterations = np.where(active, iteration, iterations)
            converged |= active & (np.hypot(dx, dy) <= cfg.exact_image_displacement_tol_arcsec) & (np.hypot(rx, ry) <= source_tol)
            alive &= ok & np.isfinite(x) & np.isfinite(y)
            if not np.any(alive & ~converged):
                break
        # Final Newton step at the end points: its size says how close to a root each start ended.
        bx, by, a00, a01, a10, a11 = shoot(packed, x, y)
        rx, ry = bx - source_x, by - source_y
        cx, cy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
        root_error, source_residual = np.hypot(cx, cy), np.hypot(rx, ry)
        finite = alive & ok & np.isfinite(root_error) & np.isfinite(source_residual)
        identified = finite & (root_error <= cfg.exact_image_identification_tol_arcsec)
        converged |= identified & (root_error <= cfg.exact_image_displacement_tol_arcsec)
        return dict(x=x, y=y, converged=converged, finite=finite, identification_pass=identified,
                    iterations=iterations, final_step_arcsec=root_error, final_source_residual_arcsec=source_residual)

    def _refine_merged(self, packed, source_x, source_y, start_x, start_y, groups, correction, shoot_rows=None) -> dict[str, np.ndarray]:
        """``_refine`` for many starts sharing one packed state: each iteration evaluates only the starts still moving,
        and starts of one group at the same position (rounded to ``MERGE_ARCSEC``) follow a single representative.
        ``shoot_rows(x, y, index)``, when given, replaces the packed state and the correction: it returns the source
        positions and A at the starts ``index`` (each start may then have its own lens, e.g. its family's redshift)."""
        cfg = self.config
        x, y = np.array(start_x, dtype=float), np.array(start_y, dtype=float)
        n = x.size
        source_x, source_y = np.broadcast_to(source_x, x.shape).astype(float), np.broadcast_to(source_y, y.shape).astype(float)
        groups = np.asarray(groups, dtype=np.int64)
        representative = np.arange(n)                       # each start follows itself until merged
        alive = np.isfinite(x) & np.isfinite(y)
        converged = np.zeros(n, dtype=bool)
        iterations = np.zeros(n, dtype=int)
        source_tol = max(cfg.exact_image_identification_tol_arcsec, cfg.exact_image_precision_limit)

        def shoot(index):
            if shoot_rows is not None:
                return shoot_rows(x[index], y[index], index)
            bx, by, a00, a01, a10, a11 = self.ray_shoot_and_jacobian(packed, x[index], y[index])
            if correction is not None:
                dx, dy, j00, j01, j10, j11 = correction(x[index], y[index], index)
                bx, by, a00, a01, a10, a11 = bx - dx, by - dy, a00 - j00, a01 - j01, a10 - j10, a11 - j11
            return bx, by, a00, a01, a10, a11

        for iteration in range(1, int(cfg.exact_image_lm_max_iter) + 1):
            todo = np.flatnonzero(alive & ~converged & (representative == np.arange(n)))
            if not todo.size:
                break
            # Merge the moving starts of one group that now share a position: they would iterate identically.
            key = np.column_stack([groups[todo], np.round(x[todo] / MERGE_ARCSEC), np.round(y[todo] / MERGE_ARCSEC)])
            _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
            representative[todo] = todo[first[inverse.ravel()]]
            todo = todo[np.sort(first)]
            bx, by, a00, a01, a10, a11 = shoot(todo)
            rx, ry = bx - source_x[todo], by - source_y[todo]
            dx, dy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
            x[todo], y[todo] = np.where(ok, x[todo] + dx, x[todo]), np.where(ok, y[todo] + dy, y[todo])
            iterations[todo] = np.where(ok, iteration, iterations[todo])
            converged[todo] |= ok & (np.hypot(dx, dy) <= cfg.exact_image_displacement_tol_arcsec) & (np.hypot(rx, ry) <= source_tol)
            alive[todo] &= ok & np.isfinite(x[todo]) & np.isfinite(y[todo])
        # Merged starts take their representative's end point (chains of merges resolved to the root).
        while np.any(representative[representative] != representative):
            representative = representative[representative]
        x, y = x[representative], y[representative]
        alive, converged, iterations = alive[representative], converged[representative], iterations[representative]
        final = np.unique(representative)
        bx, by, a00, a01, a10, a11 = shoot(final)
        rx, ry = bx - source_x[final], by - source_y[final]
        cx, cy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
        root_error, source_residual = np.full(n, np.nan), np.full(n, np.nan)
        step_ok = np.zeros(n, dtype=bool)
        position = np.searchsorted(final, representative)
        root_error[:] = np.hypot(cx, cy)[position]
        source_residual[:] = np.hypot(rx, ry)[position]
        step_ok[:] = ok[position]
        finite = alive & step_ok & np.isfinite(root_error) & np.isfinite(source_residual)
        identified = finite & (root_error <= cfg.exact_image_identification_tol_arcsec)
        converged |= identified & (root_error <= cfg.exact_image_displacement_tol_arcsec)
        return dict(x=x, y=y, converged=converged, finite=finite, identification_pass=identified,
                    iterations=iterations, final_step_arcsec=root_error, final_source_residual_arcsec=source_residual)

    def _polish(self, packed, source_x, source_y, x, y, correction, shoot_rows=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Undamped Newton iterations (steps capped at ``POLISH_STEP_ARCSEC``) on candidate roots sharing one packed state,
        each until its source residual stops falling; returns the positions and the final source residual.  Damped steps
        stall near a critical curve, where A is nearly singular; a Newton step converges quadratically at a true root and
        cannot lower the residual of a point stuck on the critical curve below the caustic offset.
        ``correction(x, y, index)`` takes the candidate indices; ``shoot_rows`` is as in ``_refine_merged``."""
        x, y = np.array(x, dtype=float), np.array(y, dtype=float)
        residual = np.full(x.size, np.inf)
        active = np.isfinite(x) & np.isfinite(y)
        for _ in range(int(self.config.exact_image_polish_iterations) + 1):
            index = np.flatnonzero(active)
            if not index.size:
                break
            if shoot_rows is not None:
                bx, by, a00, a01, a10, a11 = shoot_rows(x[index], y[index], index)
            else:
                bx, by, a00, a01, a10, a11 = self.ray_shoot_and_jacobian(packed, x[index], y[index])
                if correction is not None:
                    dx, dy, j00, j01, j10, j11 = correction(x[index], y[index], index)
                    bx, by, a00, a01, a10, a11 = bx - dx, by - dy, a00 - j00, a01 - j01, a10 - j10, a11 - j11
            rx, ry = bx - source_x[index], by - source_y[index]
            r = np.hypot(rx, ry)
            improved = np.isfinite(r) & (r < residual[index])
            residual[index] = np.where(improved, r, residual[index])
            det = a00 * a11 - a01 * a10
            step_x, step_y = -(a11 * rx - a01 * ry) / det, -(-a10 * rx + a00 * ry) / det
            length = np.hypot(step_x, step_y)
            scale = np.where(length > POLISH_STEP_ARCSEC, POLISH_STEP_ARCSEC / np.maximum(length, 1e-300), 1.0)
            move = improved & np.isfinite(step_x) & np.isfinite(step_y) & (r > 1e-14)
            x[index] = np.where(move, x[index] + scale * step_x, x[index])
            y[index] = np.where(move, y[index] + scale * step_y, y[index])
            active[index] = move
        return x, y, residual

    def solve_local_lm(self, family: FamilyData, packed, source_x, source_y, *, adaptive: bool):
        cfg = self.config
        start_x, start_y = np.asarray(family.x_obs, dtype=float), np.asarray(family.y_obs, dtype=float)
        result = self._refine(packed, source_x, source_y, start_x, start_y)
        ok = np.asarray(result["identification_pass"], dtype=bool)
        if adaptive and not ok.all():
            radius = min(float(cfg.exact_image_lm_trust_radius_arcsec), max(float(cfg.exact_image_min_distance_arcsec), 1.0e-3))
            for idx in np.flatnonzero(~ok):
                best, best_score, r = None, np.inf, radius
                for _ in range(int(cfg.exact_image_adaptive_max_levels)):
                    offsets = np.asarray([-r, 0.0, r])
                    gx, gy = np.meshgrid(offsets, offsets)
                    refined = self._refine(packed, source_x, source_y, start_x[idx] + gx.ravel(), start_y[idx] + gy.ravel())
                    score = np.where(refined["finite"], refined["final_step_arcsec"] + 0.01 * refined["final_source_residual_arcsec"], np.inf)
                    k = int(np.argmin(score))
                    if np.isfinite(score[k]) and score[k] < best_score:
                        best_score, best = float(score[k]), {key: np.asarray(value)[k] for key, value in refined.items()}
                    if best is not None and bool(best["identification_pass"]):
                        break
                    if not np.isfinite(score).any() and best is None:
                        break                                   # nothing finite from nine starts: a smaller box will not help
                    r *= 0.5
                if best is not None and bool(best["identification_pass"]):
                    for key in result:
                        result[key][idx] = best[key]
            ok = np.asarray(result["identification_pass"], dtype=bool)
        if not ok.any():
            raise RuntimeError("local exact image solver failed identification tolerance")
        return np.asarray(result["x"])[ok], np.asarray(result["y"])[ok]

    def solve(self, family: FamilyData, packed, source_x: float, source_y: float, *, adaptive: bool | None = None):
        """Model image positions of one family.  ``adaptive`` overrides the configured finder's fallback search; the
        posterior-draw diagnostics switch it off because a draw whose images the local solver cannot find from the
        observed positions is a failed recovery, and the fallback costs minutes per draw on a bad model."""
        started = time.perf_counter()
        try:
            finder = self.config.exact_image_finder
            return self.solve_local_lm(family, packed, source_x, source_y,
                                       adaptive=(finder == "local-lm-adaptive") if adaptive is None else bool(adaptive))
        finally:
            self._add_time("exact_solver", started)

    # ------------------------------------------------------------------ matching
    def match(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> dict[str, Any]:
        x_pred, y_pred = np.asarray(x_pred, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
        n_obs, n_pred = family.n_images, int(min(x_pred.size, y_pred.size))
        matched_x, matched_y = np.full(n_obs, np.nan), np.full(n_obs, np.nan)
        recovered = np.zeros(n_obs, dtype=bool)
        out = dict(produced_image_count=n_pred, recovered_image_count=0, missing_image_count=n_obs, extra_image_count=n_pred,
                   multiplicity_failed=True, multiplicity_failure_reason="no_model_images" if n_pred == 0 else "no_matches",
                   matched_model_x_arcsec=matched_x, matched_model_y_arcsec=matched_y, recovered_image_mask=recovered,
                   extra_model_x_arcsec=np.asarray([], dtype=float), extra_model_y_arcsec=np.asarray([], dtype=float))
        if n_pred == 0 or n_obs == 0:
            return out
        pred = np.column_stack([x_pred[:n_pred], y_pred[:n_pred]])
        obs = np.column_stack([family.x_obs, family.y_obs])
        cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        cost = np.where(np.isfinite(pred).all(axis=1)[:, None] & np.isfinite(obs).all(axis=1)[None, :], cost, np.inf)
        if not np.isfinite(cost).any():
            out["multiplicity_failure_reason"] = "nonfinite_prediction"
            return out
        tolerance = float(self.config.match_tolerance_arcsec)
        assignment_cost = np.where(np.isfinite(cost), cost, cost[np.isfinite(cost)].max() + tolerance + 1.0)
        rows, cols = linear_sum_assignment(assignment_cost)
        accepted = np.isfinite(cost[rows, cols]) & (cost[rows, cols] <= tolerance)
        for p, o in zip(rows[accepted], cols[accepted]):
            matched_x[o], matched_y[o], recovered[o] = x_pred[p], y_pred[p], True
        extra = np.ones(n_pred, dtype=bool)
        extra[rows[accepted]] = False
        n_rec = int(accepted.sum())
        missing, surplus = n_obs - n_rec, n_pred - n_rec
        reason = "" if missing == 0 and surplus == 0 else ("missing_and_extra_model_images" if missing and surplus and n_pred != n_obs
                                                            else "match_tolerance_exceeded" if missing and surplus
                                                            else "missing_model_images" if missing else "extra_model_images")
        out.update(recovered_image_count=n_rec, missing_image_count=missing, extra_image_count=surplus,
                   multiplicity_failed=bool(missing or surplus), multiplicity_failure_reason=reason,
                   matched_model_x_arcsec=matched_x, matched_model_y_arcsec=matched_y, recovered_image_mask=recovered,
                   extra_model_x_arcsec=x_pred[:n_pred][extra], extra_model_y_arcsec=y_pred[:n_pred][extra])
        return out

    # ------------------------------------------------------------------ per family prediction
    def _details(self, family: FamilyData, x_pred: np.ndarray, y_pred: np.ndarray, mu: np.ndarray, *, error: str | None = None) -> dict[str, Any]:
        details: dict[str, Any] = {"failed": True, "exact_image_rms": np.nan}
        details.update(self.match(np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float), family))
        if error is not None:
            details["multiplicity_failure_reason"] = f"exact_image_prediction_failed: {error}"
        recovered = details["recovered_image_mask"]
        if recovered.any():
            residual = np.sqrt((details["matched_model_x_arcsec"] - family.x_obs) ** 2 + (details["matched_model_y_arcsec"] - family.y_obs) ** 2)
            details["exact_image_rms"] = float(np.sqrt(np.nanmean(residual[recovered] ** 2)))
            details["failed"] = not bool(recovered.all())
        details["magnification"] = mu                                        # model magnification at the observed positions
        details["magnification_at_model"] = np.full(family.n_images, np.nan)  # and at the recovered model positions (filled by the caller)
        return details

    def predict_family(self, family: FamilyData, packed: PackedLensState, source_x: float, source_y: float, *,
                       adaptive: bool | None = None) -> dict[str, Any]:
        """Model images of one family: matched positions, recovery mask, extra images, magnifications."""
        mu = self.magnification(packed, family.x_obs, family.y_obs)
        try:
            x_pred, y_pred = self.solve(family, packed, source_x, source_y, adaptive=adaptive)
        except Exception as error:
            return self._details(family, np.asarray([]), np.asarray([]), mu, error=str(error))
        details = self._details(family, x_pred, y_pred, mu)
        recovered = details["recovered_image_mask"]
        if recovered.any():
            details["magnification_at_model"][recovered] = self.magnification(
                packed, details["matched_model_x_arcsec"][recovered], details["matched_model_y_arcsec"][recovered])
        return details

    # ------------------------------------------------------------------ global image search (batched, JAX)
    def search_windows(self, families: list[FamilyData]) -> list[tuple[float, float, float]]:
        """The square searched for each family's images, as (x centre, y centre, half side) in arcsec: the span of its
        images plus ``exact_image_window_margin_arcsec``."""
        margin = float(self.config.exact_image_window_margin_arcsec)
        return [(family.x_center, family.y_center, 0.5 * float(family.search_window) + margin) for family in families]

    def solve_all_global(self, families: list[FamilyData], physical: np.ndarray, sources: np.ndarray, correction=None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Every image of every family at one model.

        Each family's search window (its images' span plus a margin) is ray-shot on a coarse grid.  A cell can hold an
        image when the source lies within the cell's mapped reach, |beta - beta_source| below a few pitches times the
        local stretch ||A||, so demagnified images count too.  Seeds are the coarse local minima of that distance, plus,
        where such cells lie close to a critical curve (small |det A|) and image pairs can sit closer than the coarse
        pitch, the points of a fine sub-grid that pass the same reach test on their own fields.  The seeds are refined
        in a batched Levenberg-Marquardt solve that iterates only the seeds still moving and merges seeds of one family
        that reach the same position (``_refine``), then polished; distinct roots inside each window are the images.
        Returns (x, y) per family, in order.  ``correction(x, y, family_index)`` adds a deflection field per family (a
        realization of the GRF), with family_index the position in ``families``.

        When the lens at every family's redshift is the lens at the first family's redshift times one scalar per family
        (plus the external shear, which does not scale; :meth:`_redshift_scaling` checks it at every call), all families
        share one coarse lattice (``_solve_shared``): the windows are snapped outward onto it, every lattice point and
        every distinct fine point is evaluated once for all families, and all seeds are refined in one batch.
        Otherwise, or with ``shared_lattice`` False, each source redshift is searched on its own
        (``_solve_per_redshift``).  The two place the grid points differently, so their seeds differ and a root close
        to the limit of the coarse search can be found by one and not the other."""
        plan = self._redshift_scaling(families, physical, correction) if self.shared_lattice else None
        path = "per_redshift" if plan is None else "shared"
        self._last_search.path = path
        with self._lock:
            self.search_counts[path] += 1
        if plan is None:
            return self._solve_per_redshift(families, physical, sources, correction)
        return self._solve_shared(families, sources, correction, plan)

    def _distinct_roots(self, windows, seed_family, x, y, residual) -> list[tuple[np.ndarray, np.ndarray]]:
        """The images of each family from the polished seeds: a root must solve the lens equation to round-off (points
        stuck on a critical curve reach only the caustic offset) and lie in the family's window; roots closer than the
        root separation are one image, the best kept."""
        cfg = self.config
        root = np.isfinite(x) & np.isfinite(y) & (residual <= float(cfg.exact_image_root_residual_arcsec))
        separation = float(cfg.exact_image_root_separation_arcsec)
        solutions = []
        for k, (cx, cy, half) in enumerate(windows):
            keep = root & (seed_family == k) & (np.abs(x - cx) <= half) & (np.abs(y - cy) <= half)
            px, py, quality = x[keep], y[keep], residual[keep]
            distinct: list[int] = []
            for i in np.argsort(quality):
                if all(np.hypot(px[i] - px[j], py[i] - py[j]) > separation for j in distinct):
                    distinct.append(int(i))
            solutions.append((px[distinct], py[distinct]))
        return solutions

    def _solve_per_redshift(self, families, physical, sources, correction) -> list[tuple[np.ndarray, np.ndarray]]:
        """``solve_all_global`` one source redshift at a time: each family's window on its own coarse grid, the
        families of one redshift sharing one packed state and one field evaluation, one refinement per redshift."""
        cfg = self.config
        coarse, fine = float(cfg.exact_image_grid_pitch_arcsec), float(cfg.exact_image_fine_pitch_arcsec)
        reach = float(cfg.exact_image_seed_tolerance_pitches)
        physical = jnp.asarray(physical, dtype=jnp.float64)
        sources = np.asarray(sources, dtype=float)
        windows = self.search_windows(families)
        by_z: dict[float, list[int]] = {}
        for k, family in enumerate(families):
            by_z.setdefault(float(family.z_source), []).append(k)
        packed_by_z = {z: self.evaluator.packed_state_for_z(physical, z) for z in by_z}
        sub = np.arange(-0.5 * coarse, 0.5 * coarse + 1e-9, fine)
        sub_x, sub_y = (v.ravel() for v in np.meshgrid(sub, sub))

        def mapped(points: np.ndarray, z: float, point_family: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Source positions, det A (signed) and the stretch ||A||_F at image-plane points for one source redshift."""
            alpha, u = self.evaluator.total_fields_at(points, packed_by_z[z], self.all_components)
            alpha, jac = np.asarray(alpha), np.eye(2) - np.asarray(u)
            if correction is not None:
                dx, dy, j00, j01, j10, j11 = correction(points[:, 0], points[:, 1], point_family)
                alpha = alpha + np.column_stack([dx, dy])
                jac = jac - np.stack([np.stack([j00, j01], -1), np.stack([j10, j11], -1)], -2)
            det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
            return points - alpha, det, np.maximum(1.0, np.sqrt(np.sum(np.square(jac), axis=(1, 2))))

        started = time.perf_counter()
        seed_x, seed_y, seed_family = [], [], []
        grid_points = 0
        for z, group in by_z.items():
            grids = []
            for k in group:
                cx, cy, half = windows[k]
                n = int(np.ceil(2.0 * half / coarse)) + 1
                grids.append(np.meshgrid(cx + np.linspace(-half, half, n), cy + np.linspace(-half, half, n)))
            points = np.concatenate([np.column_stack([xx.ravel(), yy.ravel()]) for xx, yy in grids])
            point_family = np.concatenate([np.full(xx.size, k) for k, (xx, _) in zip(group, grids)])
            grid_points += points.shape[0]
            beta, det, stretch = mapped(points, z, point_family)
            fine_points, fine_family = [], []
            offset = 0
            for k, (xx, yy) in zip(group, grids):
                block = slice(offset, offset + xx.size)
                offset += xx.size
                distance = np.sqrt(np.sum(np.square(beta[block] - sources[k]), axis=1)).reshape(xx.shape)
                within = distance <= reach * coarse * stretch[block].reshape(xx.shape)
                padded = np.pad(distance, 1, constant_values=np.inf)
                minima = within & (distance <= np.min([padded[i:i + xx.shape[0], j:j + xx.shape[1]] for i in range(3) for j in range(3)], axis=0))
                seed_x.append(xx[minima]); seed_y.append(yy[minima]); seed_family.append(np.full(int(minima.sum()), k))
                critical = within & (np.abs(det[block]).reshape(xx.shape) <= float(cfg.exact_image_critical_det_threshold))
                if critical.any():
                    fine_points.append(np.column_stack([(xx[critical][:, None] + sub_x[None, :]).ravel(), (yy[critical][:, None] + sub_y[None, :]).ravel()]))
                    fine_family.append(np.full(fine_points[-1].shape[0], k))
            if fine_points:
                # The fine sub-grid is screened on its own fields: one cheap evaluation instead of a solve per point.
                fine_points, fine_family = np.concatenate(fine_points), np.concatenate(fine_family)
                grid_points += fine_points.shape[0]
                beta_f, _, stretch_f = mapped(fine_points, z, fine_family)
                distance_f = np.sqrt(np.sum(np.square(beta_f - sources[fine_family]), axis=1))
                # Every fine point that passes the reach test is a seed: images of a tight multiplet near a critical
                # curve can share one local minimum at this pitch (Hera, family 10: three images within 0.3"), so no
                # point is dropped; the refinement merges the seeds that reach the same position.
                keep = distance_f <= reach * fine * stretch_f
                seed_x.append(fine_points[keep, 0]); seed_y.append(fine_points[keep, 1]); seed_family.append(fine_family[keep])
        self._add_time("global_grid", started)
        seed_x, seed_y = np.concatenate(seed_x), np.concatenate(seed_y)
        seed_family = np.concatenate(seed_family).astype(int)
        self._last_search.grid_points, self._last_search.seed_count = grid_points, int(seed_x.size)
        solutions = [(np.asarray([], dtype=float), np.asarray([], dtype=float)) for _ in families]
        if seed_x.size == 0:
            return solutions
        started = time.perf_counter()
        x, y = np.full(seed_x.shape, np.nan), np.full(seed_y.shape, np.nan)
        residual = np.full(seed_x.shape, np.inf)
        for z, group in by_z.items():
            rows = np.flatnonzero(np.isin(seed_family, group))
            if not rows.size:
                continue
            fam = seed_family[rows]
            family_correction = None if correction is None else (lambda xx, yy, i, fam=fam: correction(xx, yy, fam[i]))
            part = self._refine(packed_by_z[z], sources[fam, 0], sources[fam, 1], seed_x[rows], seed_y[rows],
                                correction=family_correction, groups=fam)
            # Candidates: finite end points near a root.  One polish per distinct end point (to MERGE_ARCSEC).
            loose = max(float(cfg.exact_image_identification_tol_arcsec), float(cfg.exact_image_precision_limit))  # where the refinement stops
            candidate = np.flatnonzero(np.asarray(part["finite"]) & (np.asarray(part["final_source_residual_arcsec"]) <= loose))
            if not candidate.size:
                continue
            key = np.column_stack([fam[candidate], np.round(part["x"][candidate] / MERGE_ARCSEC), np.round(part["y"][candidate] / MERGE_ARCSEC)])
            _, first = np.unique(key, axis=0, return_index=True)
            candidate = candidate[np.sort(first)]
            cf = fam[candidate]
            px, py, pr = self._polish(packed_by_z[z], sources[cf, 0], sources[cf, 1], part["x"][candidate], part["y"][candidate],
                                      None if correction is None else (lambda xx, yy, i, cf=cf: correction(xx, yy, cf[i])))
            x[rows[candidate]], y[rows[candidate]], residual[rows[candidate]] = px, py, pr
        self._add_time("global_refine", started)
        return self._distinct_roots(windows, seed_family, x, y, residual)

    # ------------------------------------------------------------------ one lattice for all families
    def _split_fields(self, xy: np.ndarray, packed: PackedLensState) -> tuple[np.ndarray, ...]:
        """Deflection (P, 2) and Jacobian (P, 2, 2) of the scaled components and of the shear at ``xy`` (P, 2), in
        chunks as ``Evaluator.total_fields_at`` (the same block sizes)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        n = xy.shape[0]
        chunk = int(min(16384, max(64, 1 << max(0, int(n - 1)).bit_length())))
        out = [np.empty((n, 2)), np.empty((n, 2, 2)), np.empty((n, 2)), np.empty((n, 2, 2))]
        for start in range(0, n, chunk):
            block = xy[start:start + chunk]
            pad = chunk - block.shape[0]
            values = self._split_fields_jit(jnp.asarray(np.pad(block, ((0, pad), (0, 0))) if pad else block), packed)
            for array, value in zip(out, values):
                array[start:start + chunk] = np.asarray(value)[:block.shape[0]]
        return tuple(out)

    def _redshift_scaling(self, families, physical, correction) -> dict[str, Any] | None:
        """How the lens of every family follows from one evaluation at the first family's redshift, or None when it
        does not and the search must run per redshift.

        The scaled components (``SCALED_PROFILES``) are linear in the lensing efficiency of the source, so at one lens
        plane with a fixed cosmology their fields at a family's redshift are those at the reference redshift times the
        ratio of the dPIE factors; the external shear is added unscaled.  A field realization with ``unweighted`` and
        ``weights`` (``plotting.grf.FieldRealization``) is evaluated once, unweighted, and weighted per family; any
        other correction is called per family.  None for an unknown profile or a sampled cosmology, and when the
        scaled lens differs from the lens at any family's redshift (or a weighted field from the correction itself) by
        more than ``SCALING_CHECK_RTOL`` at the corners and centres of the windows."""
        evaluator = self.evaluator
        if not families or not self._profile_codes <= SCALED_PROFILES | {SHEAR_PROFILE}:
            return None
        redshift = np.array([float(family.z_source) for family in families])
        reference = float(redshift[0])
        scale = scaled_lens_factors(evaluator, redshift, reference)
        if scale is None:
            return None
        physical = jnp.asarray(physical, dtype=jnp.float64)
        packed = evaluator.packed_state_for_z(physical, reference)
        windows = np.asarray(self.search_windows(families))
        probes = np.concatenate([windows[:, :2]] + [windows[:, :2] + windows[:, 2:] * np.asarray(sign) for sign in ((-1, -1), (1, 1))])
        probes = probes[np.linspace(0, len(probes) - 1, min(len(probes), 12)).round().astype(int)]

        def close(value, expected):
            return bool(np.all(np.isfinite(value)) and np.all(np.isfinite(expected))
                        and np.abs(value - expected).max() <= SCALING_CHECK_RTOL * max(1.0, float(np.abs(expected).max())))

        checked = sorted({float(z) for z in redshift} - {reference})
        if checked:
            scaled_alpha, scaled_u, shear_alpha, shear_u = self._split_fields(probes, packed)
            for z in checked:
                alpha, u = evaluator.total_fields_at(probes, evaluator.packed_state_for_z(physical, z), self.all_components)
                s = float(scale[np.flatnonzero(redshift == z)[0]])
                if not (close(s * scaled_alpha + shear_alpha, alpha) and close(s * scaled_u + shear_u, u)):
                    return None
        weights = None
        if correction is not None and callable(getattr(correction, "unweighted", None)) and getattr(correction, "weights", None) is not None:
            weights = np.asarray(correction.weights, dtype=float)
            if len(weights) < len(families):
                return None
            weights = weights[:len(families)]
            unweighted = correction.unweighted(probes[:, 0], probes[:, 1])
            for k in sorted({0, int(np.argmax(np.abs(weights - weights[0])))}):
                direct = np.column_stack(correction(probes[:, 0], probes[:, 1], np.full(len(probes), k)))
                if not close(weights[k] * unweighted, direct):
                    return None
        return dict(packed=packed, scale=scale, weights=weights)

    def _lens_pieces(self, points: np.ndarray, plan, correction, grid=None) -> dict[str, np.ndarray]:
        """The parts of the lens shared by every family at ``points`` (P, 2): scaled components and shear at the
        reference redshift and, for a field realization, its unweighted field (P, 6).  ``grid`` = (gx, gy, used) says
        that the points are the ``used`` nodes (row-major) of the meshgrid of gx and gy, so the field is summed by rows
        and columns."""
        scaled_alpha, scaled_u, shear_alpha, shear_u = self._split_fields(points, plan["packed"])
        pieces = dict(points=points, scaled_alpha=scaled_alpha, scaled_u=scaled_u, shear_alpha=shear_alpha, shear_u=shear_u, field=None)
        if correction is not None and plan["weights"] is not None:
            if grid is not None:
                gx, gy, used = grid
                xx, yy = np.meshgrid(gx, gy)
                pieces["field"] = correction.unweighted(xx, yy, select=used)
            else:
                pieces["field"] = correction.unweighted(points[:, 0], points[:, 1])
        return pieces

    def _family_lens(self, pieces, rows, families, plan, correction) -> tuple[np.ndarray, np.ndarray]:
        """Source positions (P, 2) and A = I - d alpha / d theta (P, 2, 2) at the points ``rows`` of ``pieces``, each
        for the family ``families`` (an index into the plan, per row or one for all)."""
        families = np.broadcast_to(np.asarray(families, dtype=int), np.shape(rows))
        points = pieces["points"][rows]
        s = plan["scale"][families]
        alpha = s[:, None] * pieces["scaled_alpha"][rows] + pieces["shear_alpha"][rows]
        u = s[:, None, None] * pieces["scaled_u"][rows] + pieces["shear_u"][rows]
        if pieces["field"] is not None:
            w = plan["weights"][families]
            field = pieces["field"][rows]
            alpha = alpha + w[:, None] * field[:, :2]
            u = u + w[:, None, None] * field[:, 2:].reshape(-1, 2, 2)
        elif correction is not None:
            dx, dy, j00, j01, j10, j11 = correction(points[:, 0], points[:, 1], families)
            alpha = alpha + np.column_stack([dx, dy])
            u = u + np.stack([np.stack([j00, j01], -1), np.stack([j10, j11], -1)], -2)
        return points - alpha, np.eye(2) - u

    @staticmethod
    def lattice_boxes(windows, pitch: float) -> np.ndarray:
        """The node range (i0, i1, j0, j1) of each window (x centre, y centre, half side) on the lattice of nodes at
        integer multiples of ``pitch``: the window snapped outward, so its nodes cover it."""
        windows = np.asarray(windows, dtype=float).reshape(-1, 3)
        cx, cy, half = windows.T
        return np.column_stack([np.floor((cx - half) / pitch), np.ceil((cx + half) / pitch),
                                np.floor((cy - half) / pitch), np.ceil((cy + half) / pitch)]).astype(np.int64)

    def search_extent(self, families: list[FamilyData]) -> np.ndarray:
        """Corners (2 per family, (point, 2), arcsec) of the squares holding every grid point the global search
        evaluates: each window snapped outward onto the coarse lattice (``lattice_boxes``) and widened by half a coarse
        pitch, the reach of the fine sub-grid around an edge node.  They hold the grids of both search paths (the
        per-redshift grids span the windows themselves); the refinement's iterates are not bounded by them."""
        coarse = float(self.config.exact_image_grid_pitch_arcsec)
        i0, i1, j0, j1 = (v * coarse for v in self.lattice_boxes(self.search_windows(families), coarse).T)
        return np.concatenate([np.column_stack([i0, j0]) - .5 * coarse, np.column_stack([i1, j1]) + .5 * coarse])

    def _solve_shared(self, families, sources, correction, plan) -> list[tuple[np.ndarray, np.ndarray]]:
        """``solve_all_global`` on one coarse lattice shared by all families (nodes at integer multiples of the pitch,
        each window snapped outward onto it), the fine sub-grids on the matching global fine lattice with every distinct
        point evaluated once, and one refinement and polish over the seeds of all families."""
        cfg = self.config
        coarse, fine = float(cfg.exact_image_grid_pitch_arcsec), float(cfg.exact_image_fine_pitch_arcsec)
        reach = float(cfg.exact_image_seed_tolerance_pitches)
        sources = np.asarray(sources, dtype=float)
        windows = self.search_windows(families)
        started = time.perf_counter()
        boxes = self.lattice_boxes(windows, coarse)
        i_min, j_min = boxes[:, 0].min(), boxes[:, 2].min()
        used = np.zeros((boxes[:, 3].max() - j_min + 1, boxes[:, 1].max() - i_min + 1), dtype=bool)
        for i0, i1, j0, j1 in boxes:
            used[j0 - j_min:j1 - j_min + 1, i0 - i_min:i1 - i_min + 1] = True
        gx, gy = (np.arange(i_min, i_min + used.shape[1]) * coarse, np.arange(j_min, j_min + used.shape[0]) * coarse)
        jj, ii = np.nonzero(used)
        index = np.full(used.shape, -1)
        index[jj, ii] = np.arange(jj.size)
        pieces = self._lens_pieces(np.column_stack([gx[ii], gy[jj]]), plan, correction, grid=(gx, gy, used))
        grid_points = int(jj.size)
        seed_x, seed_y, seed_family = [], [], []
        centre_i, centre_j, centre_family = [], [], []
        for k, (i0, i1, j0, j1) in enumerate(boxes):
            rows = index[j0 - j_min:j1 - j_min + 1, i0 - i_min:i1 - i_min + 1]
            shape = rows.shape
            beta, jac = self._family_lens(pieces, rows.ravel(), k, plan, correction)
            det = (jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]).reshape(shape)
            stretch = np.maximum(1.0, np.sqrt(np.sum(np.square(jac), axis=(1, 2)))).reshape(shape)
            distance = np.sqrt(np.sum(np.square(beta - sources[k]), axis=1)).reshape(shape)
            within = distance <= reach * coarse * stretch
            padded = np.pad(distance, 1, constant_values=np.inf)
            minima = within & (distance <= np.min([padded[a:a + shape[0], b:b + shape[1]] for a in range(3) for b in range(3)], axis=0))
            node_x, node_y = gx[i0 - i_min:i1 - i_min + 1], gy[j0 - j_min:j1 - j_min + 1]
            mj, mi = np.nonzero(minima)
            seed_x.append(node_x[mi]); seed_y.append(node_y[mj]); seed_family.append(np.full(mi.size, k))
            cj, ci = np.nonzero(within & (np.abs(det) <= float(cfg.exact_image_critical_det_threshold)))
            centre_i.append(ci + i0); centre_j.append(cj + j0); centre_family.append(np.full(ci.size, k))
        centre_i, centre_j, centre_family = (np.concatenate(v).astype(np.int64) for v in (centre_i, centre_j, centre_family))
        if centre_i.size:
            # Fine points: offsets sub[a] (x) and sub[b] (y) from each critical node.  When the coarse pitch is a whole
            # number R of fine pitches, the sub-grids of neighbouring nodes share their edge points: node i offset a is
            # fine-lattice point i R + a, and each distinct point is evaluated once for every family that asks for it.
            sub = np.arange(-0.5 * coarse, 0.5 * coarse + 1e-9, fine)
            ratio = coarse / fine
            step = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-9 else sub.size
            b, a = (v.ravel() for v in np.meshgrid(np.arange(sub.size), np.arange(sub.size), indexing="ij"))
            key = np.column_stack([(centre_i[:, None] * step + a[None]).ravel(), (centre_j[:, None] * step + b[None]).ravel()])
            fine_family = np.repeat(centre_family, a.size)
            fine_xy = np.column_stack([(centre_i[:, None] * coarse + sub[a][None]).ravel(), (centre_j[:, None] * coarse + sub[b][None]).ravel()])
            _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
            inverse = inverse.ravel()
            fine_pieces = self._lens_pieces(fine_xy[first], plan, correction)
            grid_points += int(first.size)
            for k in np.unique(fine_family):
                rows = inverse[fine_family == k]
                beta, jac = self._family_lens(fine_pieces, rows, k, plan, correction)
                stretch = np.maximum(1.0, np.sqrt(np.sum(np.square(jac), axis=(1, 2))))
                # Every fine point that passes the reach test is a seed: images of a tight multiplet near a critical
                # curve can share one local minimum at this pitch (Hera, family 10: three images within 0.3"), so no
                # point is dropped; the refinement merges the seeds that reach the same position.
                keep = np.sqrt(np.sum(np.square(beta - sources[k]), axis=1)) <= reach * fine * stretch
                points = fine_pieces["points"][rows[keep]]
                seed_x.append(points[:, 0]); seed_y.append(points[:, 1]); seed_family.append(np.full(int(keep.sum()), k))
        del pieces
        self._add_time("global_grid", started)
        seed_x, seed_y = np.concatenate(seed_x), np.concatenate(seed_y)
        seed_family = np.concatenate(seed_family).astype(int)
        self._last_search.grid_points, self._last_search.seed_count = grid_points, int(seed_x.size)
        if seed_x.size == 0:
            return [(np.asarray([], dtype=float), np.asarray([], dtype=float)) for _ in families]
        started = time.perf_counter()

        def shooter(family_of_row):
            def shoot(x, y, index):
                beta, a = self._family_lens(self._lens_pieces(np.column_stack([x, y]), plan, correction), np.arange(len(x)),
                                            family_of_row[index], plan, correction)
                return beta[:, 0], beta[:, 1], a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1]
            return shoot
        fam = seed_family
        part = self._refine_merged(None, sources[fam, 0], sources[fam, 1], seed_x, seed_y, fam, None, shoot_rows=shooter(fam))
        x, y = np.full(seed_x.shape, np.nan), np.full(seed_y.shape, np.nan)
        residual = np.full(seed_x.shape, np.inf)
        # Candidates: finite end points near a root.  One polish per distinct end point (to MERGE_ARCSEC).
        loose = max(float(cfg.exact_image_identification_tol_arcsec), float(cfg.exact_image_precision_limit))
        candidate = np.flatnonzero(np.asarray(part["finite"]) & (np.asarray(part["final_source_residual_arcsec"]) <= loose))
        if candidate.size:
            key = np.column_stack([fam[candidate], np.round(part["x"][candidate] / MERGE_ARCSEC), np.round(part["y"][candidate] / MERGE_ARCSEC)])
            _, first = np.unique(key, axis=0, return_index=True)
            candidate = candidate[np.sort(first)]
            cf = fam[candidate]
            px, py, pr = self._polish(None, sources[cf, 0], sources[cf, 1], part["x"][candidate], part["y"][candidate], None,
                                      shoot_rows=shooter(cf))
            x[candidate], y[candidate], residual[candidate] = px, py, pr
        self._add_time("global_refine", started)
        return self._distinct_roots(windows, seed_family, x, y, residual)

    def predict_all_global(self, images, families: list[FamilyData], physical: np.ndarray, sources: np.ndarray, correction=None) -> list[dict[str, Any]]:
        """Family predictions from the global search: matched positions, extra images, magnifications at the observed and
        the recovered positions.  ``families`` is in the order of ``images.family_ids`` and ``sources`` is (F, 2).
        ``correction`` (see ``solve_all_global``) is part of the lens: it moves the positions and enters the magnifications."""
        started = time.perf_counter()
        solutions = self.solve_all_global(families, physical, sources, correction)
        self._add_time("exact_solver", started)
        packed_images = self.evaluator.packed_state(jnp.asarray(physical, dtype=jnp.float64))[0]
        x_obs, y_obs = np.asarray(images.x_obs, dtype=float), np.asarray(images.y_obs, dtype=float)
        family_index = np.asarray(images.family_index_per_image, dtype=int)
        shoot = self._corrected(self.ray_shoot_and_jacobian_per_image, None if correction is None else (lambda x, y: correction(x, y, family_index)))
        _, _, a00, a01, a10, a11 = shoot(packed_images, x_obs, y_obs)
        mu_obs = 1.0 / (a00 * a11 - a01 * a10)
        details = [self._details(family, *solutions[f], mu_obs[family_index == f]) for f, family in enumerate(families)]
        self._fill_magnification_at_model(details, shoot, packed_images, x_obs, y_obs, family_index)
        return details

    def _fill_magnification_at_model(self, details, shoot, packed_images, x_obs, y_obs, family_index) -> None:
        """Model magnification at the recovered positions of every family, one per-image evaluation of ``shoot`` (the
        lens the positions were solved in; unrecovered images keep the observed position and are masked)."""
        x_model, y_model, recovered = x_obs.copy(), y_obs.copy(), np.zeros(x_obs.size, dtype=bool)
        for f, d in enumerate(details):
            rows = np.flatnonzero(family_index == f)[d["recovered_image_mask"]]
            x_model[rows], y_model[rows] = d["matched_model_x_arcsec"][d["recovered_image_mask"]], d["matched_model_y_arcsec"][d["recovered_image_mask"]]
            recovered[rows] = True
        _, _, b00, b01, b10, b11 = shoot(packed_images, x_model, y_model)
        mu_model = np.where(recovered, 1.0 / (b00 * b11 - b01 * b10), np.nan)
        for f, d in enumerate(details):
            d["magnification_at_model"] = mu_model[family_index == f]

    def predict_all_local(self, images, families: list[FamilyData], packed_images: PackedLensState, sources: np.ndarray, correction=None) -> list[dict[str, Any]]:
        """Every family of one model in a single batched local solve from the observed positions (no fallback search):
        one Levenberg-Marquardt pass over all N images with per-image source redshifts.  ``families`` is in the order of
        ``images.family_ids`` and ``sources`` is (F, 2).  ``correction(x, y, family_index)`` adds a deflection field per
        family (family_index as in ``images.family_index_per_image``) and is part of the lens: it moves the positions and
        enters the magnifications."""
        started = time.perf_counter()
        x0, y0 = np.asarray(images.x_obs, dtype=float), np.asarray(images.y_obs, dtype=float)
        family_index = np.asarray(images.family_index_per_image, dtype=int)
        sources = np.asarray(sources, dtype=float)
        per_image = None if correction is None else (lambda x, y: correction(x, y, family_index))
        shoot = self._corrected(self.ray_shoot_and_jacobian_per_image, per_image)
        result = self._refine(packed_images, sources[family_index, 0], sources[family_index, 1], x0, y0, shoot=self.ray_shoot_and_jacobian_per_image,
                              correction=per_image)
        _, _, a00, a01, a10, a11 = shoot(packed_images, x0, y0)
        mu = 1.0 / (a00 * a11 - a01 * a10)
        self._add_time("exact_solver", started)
        details = []
        separation = float(self.config.exact_image_min_separation_arcsec)
        for f, family in enumerate(families):
            sel = family_index == f
            ok = np.asarray(result["identification_pass"])[sel]
            px, py = np.asarray(result["x"])[sel][ok], np.asarray(result["y"])[sel][ok]
            # Two observed images that converge onto the same root are one model image, as in the global search.
            distinct = [i for i in range(px.size) if all(np.hypot(px[i] - px[j], py[i] - py[j]) > separation for j in range(i))]
            if distinct:
                details.append(self._details(family, px[distinct], py[distinct], mu[sel]))
            else:
                details.append(self._details(family, np.asarray([]), np.asarray([]), mu[sel], error="local exact image solver failed identification tolerance"))
        self._fill_magnification_at_model(details, shoot, packed_images, x0, y0, family_index)
        return details
