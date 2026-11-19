"""Lens maps of small grids from a larger grid evaluated before (the overview).

The fold and cutout grids (4 to 21 arcsec across, pixels of 0.017 to 0.08") lie inside the overview (512 x 512 over
the cluster, pixels of 0.18 to 0.24"), which every reporting draw has already evaluated.  The parametric deflection
Jacobian is a sum over components, U = U_near + U_far.  The far part varies slowly on the scale of the overview
pixels, so it is interpolated with a bicubic spline from the overview nodes, where it is known exactly as
U_overview - U_near; the near part and the draw's GRF realization (its spectrum has power down to 0.2", below the
overview pixel) are evaluated exactly on the small grid.

The overview is at another source redshift.  The components in ``diagnostics.exact_images.SCALED_PROFILES`` scale
with one factor per redshift and the external shear does not (``exact_images.scaled_lens_factors``), so the far part
at the grid's redshift is s (U_overview - U_shear - U_near,overview) + U_shear, with U_near evaluated at the grid's
redshift.  A lens that does not scale this way (a sampled cosmology, another profile) is evaluated directly.

Which components are near.  A cubic spline through nodes h apart misses a function by at most (5/384) h^4 times its
fourth derivative along each axis.  A dPIE with core a, cut s and normalization sigma0 (the cored power law is treated
alike) has kappa ~ E / (2 L), E = sigma0 a s / (s - a), at L = sqrt(xi^2 + a^2) from its centre (xi the elliptical
radius), and a fourth derivative of U of at most about 12 E / L^5, so the spline misses its part of U by at most about
(5/16) E h^4 / L^5 in a box at elliptical distance xi.  An NFW (kappa ~ kappa_s ln r inside its scale radius, kappa_s =
alpha_rs / (4 r_s (1 - ln 2))) gives about (5/32) kappa_s h^4 / xi^4.  These estimates hold for round profiles; an
elliptical one falls off faster along its minor axis, and on single components (round and with ellipticity e up to 0.6,
1.6 to 12" from a 2" box, h = 0.25") the spline missed by up to 3.1 times them.  The code multiplies both by
(1 + e) / (1 - e) and the NFW one also by 2, which put every measured miss at or below 0.84 of its bound.  A component is
near when this bound, at its distance from the grid and for any draw, exceeds ``NEAR_TOLERANCE``, when its centre lies
within the patch of overview nodes the spline is fitted on (a component unresolved by the nodes would ring through the
spline), or when it is a dPIE with its cut not above its core (no bound).

Every draw is checked: at up to ``CHECK_POINTS`` x ``CHECK_POINTS`` points of the grid the interpolated parametric
Jacobian is compared with all components evaluated exactly, and a draw that misses by more than ``CHECK_TOLERANCE`` in
any element is evaluated directly on the whole grid.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..core.model import CORED_POWER_LAW_PROFILE, DP_IE_PROFILE, NFW_PROFILE, SHEAR_PROFILE

# Bump when the maps derived from a reference can change (the tolerances below are part of the cache key already).
LOCAL_MAPS_VERSION = 2       # 2: the near bound made conservative for elliptical components (more of them near)
# Overview nodes beyond the grid on each side in the patch the spline is fitted on.
PATCH_MARGIN = 4
# Largest a priori interpolation error (element of U) of a component left to the spline.
NEAR_TOLERANCE = 1.0e-6
# Largest miss of the interpolated parametric U at the check points before a draw is evaluated directly.
CHECK_TOLERANCE = 1.0e-5
CHECK_POINTS = 24
# The reference lens: per draw the parametric U on the reference grid (u00, u01, u11; U is symmetric) and the
# Jacobian of the external shear.  float32 rounds U by 6e-8 of its size, two orders below CHECK_TOLERANCE.
REFERENCE_ARRAYS = ("jacobian", "shear")
REFERENCE_DTYPE = np.float32


def reference_entry(grid_entry: str) -> str:
    """Name in cache.h5 of the reference lens of the grid saved as ``grid_entry``."""
    return "lens_reference_" + grid_entry.removeprefix("grid_maps_")


def grid_axes(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(x of the columns, y of the rows) when ``x``, ``y`` are a finite meshgrid with strictly monotonic axes."""
    if x.ndim != 2 or x.shape != y.shape or min(x.shape) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    xs, ys = x[0], y[:, 0]
    if not (np.array_equal(x, np.broadcast_to(xs, x.shape)) and np.array_equal(y, np.broadcast_to(ys[:, None], y.shape))):
        return None
    for axis in (xs, ys):
        step = np.diff(axis)
        if not (np.all(step > 0) or np.all(step < 0)):
            return None
    return xs.copy(), ys.copy()


@dataclass(frozen=True, eq=False)
class Reference:
    """A grid whose parametric Jacobians are kept as a reference lens (``ReportingDraws.grid_maps(reference=True)``):
    its cache entry, name, source redshift and axes."""
    entry: str
    name: str
    z: float
    xs: np.ndarray
    ys: np.ndarray

    @classmethod
    def of(cls, entry: str, name: str, x: np.ndarray, y: np.ndarray, z: float) -> "Reference | None":
        axes = grid_axes(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if axes is None or min(len(axes[0]), len(axes[1])) < 2 * PATCH_MARGIN + 4:
            return None
        return cls(entry, str(name), float(z), *axes)

    @property
    def pixel(self) -> float:
        return float(max(np.abs(np.diff(self.xs)).max(), np.abs(np.diff(self.ys)).max()))

    @property
    def tag(self) -> str:
        """What a grid derived from this reference depends on besides its own points (part of its cache key)."""
        return (f"local-v{LOCAL_MAPS_VERSION}:{self.entry}:near{NEAR_TOLERANCE:g}:check{CHECK_TOLERANCE:g}x{CHECK_POINTS}:"
                f"margin{PATCH_MARGIN}:{np.dtype(REFERENCE_DTYPE).name}")

    def key(self, draws) -> str:
        return json.dumps([LOCAL_MAPS_VERSION, draws.key, self.entry, repr(self.z), np.dtype(REFERENCE_DTYPE).name,
                           hashlib.sha256(self.xs.tobytes() + self.ys.tobytes()).hexdigest()])

    def covers(self, x: np.ndarray, y: np.ndarray) -> bool:
        margin = (PATCH_MARGIN + 1) * self.pixel
        return bool(x.min() - margin >= self.xs.min() and x.max() + margin <= self.xs.max()
                    and y.min() - margin >= self.ys.min() and y.max() + margin <= self.ys.max())


def _shear_components(evaluator) -> np.ndarray:
    return np.flatnonzero(np.asarray(evaluator.state.packed_lens_spec.profile_type) == SHEAR_PROFILE).astype(np.int32)


def shear_jacobian(evaluator, physical, z: float) -> np.ndarray:
    """The (constant) deflection Jacobian (2, 2) of the external shear components, zero without any."""
    shear = _shear_components(evaluator)
    if not shear.size:
        return np.zeros((2, 2))
    packed = evaluator.packed_state_for_z(physical, float(z))
    return np.asarray(evaluator.total_fields_at(np.zeros((1, 2)), packed, shear)[1], dtype=float)[0]


def _symmetric(u: np.ndarray) -> np.ndarray:
    """(..., 3) elements u00, u01, u11 of Jacobians (..., 2, 2), the off-diagonal averaged."""
    return np.stack([u[..., 0, 0], .5 * (u[..., 0, 1] + u[..., 1, 0]), u[..., 1, 1]], axis=-1)


def _full(u: np.ndarray) -> np.ndarray:
    """Jacobians (..., 2, 2) from their (..., 3) symmetric elements."""
    return np.stack([np.stack([u[..., 0], u[..., 1]], -1), np.stack([u[..., 1], u[..., 2]], -1)], -2)


class ReferenceRows:
    """The reference lens of one grid, filled draw by draw (each thread writes its own rows)."""

    def __init__(self, draws, reference: Reference):
        self.draws, self.reference = draws, reference
        self.shape = (len(reference.ys), len(reference.xs))
        self.jacobian = np.full((len(draws),) + self.shape + (3,), np.nan, dtype=REFERENCE_DTYPE)
        self.shear = np.zeros((len(draws), 2, 2))

    def add(self, k: int, parametric: np.ndarray) -> None:
        """Draw ``k``'s parametric Jacobian (point, 2, 2) on the reference grid, points in ravelled order."""
        e = self.draws.evaluator
        self.jacobian[k] = _symmetric(np.asarray(parametric, dtype=float).reshape(self.shape + (2, 2)))
        self.shear[k] = shear_jacobian(e, e.physical(jnp.asarray(self.draws.samples[k])), self.reference.z)

    def arrays(self) -> dict[str, np.ndarray]:
        return dict(jacobian=self.jacobian, shear=self.shear)


def choose_reference(draws, x: np.ndarray, y: np.ndarray, z: float) -> tuple[Reference | None, str]:
    """The first reference grid of ``draws`` that covers the grid (x, y) with the patch margin, or (None, why not)."""
    from ..diagnostics.exact_images import scaled_lens_factors
    references = getattr(draws, "_references", [])
    if not references:
        return None, "no reference grid evaluated before it"
    if grid_axes(x, y) is None:
        return None, "not a finite meshgrid"
    for reference in references:
        if reference.covers(x, y):
            if scaled_lens_factors(draws.evaluator, [z], reference.z) is None:
                return None, "the lens does not scale with the source redshift"
            return reference, ""
    return None, "no reference grid covers it"


def near_components(evaluator, packed_states, box, patch, pixel: float) -> np.ndarray:
    """Indices of the components evaluated exactly on a grid (see the module docstring): those whose interpolation
    bound at the grid's box (x0, x1, y0, y1) exceeds ``NEAR_TOLERANCE`` for any of ``packed_states`` (one per draw, at
    the grid's redshift) or whose centre lies within ``patch`` (x0, x1, y0, y1)."""
    profile = np.asarray(evaluator.state.packed_lens_spec.profile_type)
    cored = (profile == DP_IE_PROFILE) | (profile == CORED_POWER_LAW_PROFILE)
    cusp = profile == NFW_PROFILE
    near = np.zeros(profile.shape, bool)
    h4 = pixel ** 4
    for packed in packed_states:
        cx, cy = np.asarray(packed.center_x, dtype=float), np.asarray(packed.center_y, dtype=float)
        dx = np.maximum(np.maximum(box[0] - cx, cx - box[1]), 0.0)
        dy = np.maximum(np.maximum(box[2] - cy, cy - box[3]), 0.0)
        ellipticity = np.clip(np.hypot(np.asarray(packed.e1, dtype=float), np.asarray(packed.e2, dtype=float)), 0.0, 0.99)
        xi = np.hypot(dx, dy) * np.sqrt((1.0 - ellipticity) / (1.0 + ellipticity))
        inside = (cx >= patch[0]) & (cx <= patch[1]) & (cy >= patch[2]) & (cy <= patch[3])
        bound = np.zeros(profile.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            a, s, sigma0 = (np.asarray(getattr(packed, f), dtype=float) for f in ("Ra", "Rs", "sigma0"))
            strength = np.where(cored & (s > a), sigma0 * a * s / np.where(s > a, s - a, 1.0), 0.0)
            radius = np.sqrt(xi ** 2 + a ** 2)
            stretch = (1.0 + ellipticity) / (1.0 - ellipticity)      # an elliptical profile is steeper along its minor axis
            bound = np.where(cored, 5.0 / 16.0 * stretch * np.abs(strength) * h4 / np.maximum(radius, 1e-12) ** 5, bound)
            rs, alpha_rs = np.asarray(packed.nfw_rs, dtype=float), np.asarray(packed.alpha_rs, dtype=float)
            kappa_s = np.where(cusp & (rs > 0), np.abs(alpha_rs) / (4.0 * np.where(rs > 0, rs, 1.0) * (1.0 - math.log(2.0))), 0.0)
            bound = np.where(cusp, 5.0 / 16.0 * stretch * kappa_s * h4 / np.maximum(xi, pixel) ** 4, bound)
        degenerate = cored & ~(s > a)                                    # cut not above the core: no bound, so near
        near |= (cored | cusp) & (inside | degenerate | ~np.isfinite(bound) | (bound > NEAR_TOLERANCE))
    return np.flatnonzero(near).astype(np.int32)


def _bucket(n: int) -> int:
    return max(4, 1 << max(0, int(n) - 1).bit_length())


def _near_kernel(draws, codes: tuple[int, ...]):
    """Jitted U (point, 2, 2) summed over the components given as (index, mask) arrays per profile code ``codes``;
    the indices are traced, so grids with other near components of the same counts share one compilation."""
    from ..lens import profiles as lens_profiles
    kernels = draws.__dict__.setdefault("_local_kernels", {})
    if codes not in kernels:
        def fields(points, packed, groups):
            u = jnp.zeros((points.shape[0], 2, 2), dtype=jnp.float64)
            for code, (index, mask) in zip(codes, groups):
                _, rows, _ = lens_profiles.profile(code).grouped_fields(points[:, 0], points[:, 1], packed, index, derivative=False)
                u = u + jnp.sum(jnp.where(mask[:, None, None, None], rows, 0.0), axis=0)
            return u
        kernels[codes] = jax.jit(fields)
    return kernels[codes]


class LocalLens:
    """The parametric Jacobian of every draw on one grid from a reference (module docstring).  Built in the calling
    thread (it reads the reference lens); :meth:`parametric` may run in worker threads."""

    def __init__(self, draws, reference: Reference, x: np.ndarray, y: np.ndarray, z: float, *, log=print):
        from ..diagnostics.exact_images import scaled_lens_factors
        e = self.evaluator = draws.evaluator
        self.reference, self.z = reference, float(z)
        arrays = draws.reference_arrays(reference, log=log)
        self.jacobian, self.shear = arrays["jacobian"], np.asarray(arrays["shear"], dtype=float)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.shape = x.shape
        self.xf, self.yf = grid_axes(x, y)
        h = reference.pixel
        box = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
        margin = (PATCH_MARGIN + .5) * h
        columns = np.flatnonzero((reference.xs >= box[0] - margin) & (reference.xs <= box[1] + margin))
        rows = np.flatnonzero((reference.ys >= box[2] - margin) & (reference.ys <= box[3] + margin))
        self.rows, self.columns = slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1)
        px, py = reference.xs[self.columns], reference.ys[self.rows]
        self.order = (np.argsort(py), np.argsort(px))                      # the spline takes ascending axes
        self.axes = (py[self.order[0]], px[self.order[1]])
        self.fold_order = (np.argsort(self.yf), np.argsort(self.xf))
        self.fold_inverse = (np.argsort(self.fold_order[0]), np.argsort(self.fold_order[1]))
        self.nodes = np.stack(np.meshgrid(px, py), axis=-1).reshape(-1, 2)
        self.points = np.stack((x, y), axis=-1).reshape(-1, 2)
        self.scale = float(scaled_lens_factors(e, [self.z], reference.z)[0])
        physical = [e.physical(jnp.asarray(theta)) for theta in draws.samples]
        self.packed_reference = [e.packed_state_for_z(p, reference.z) for p in physical]
        self.packed = [e.packed_state_for_z(p, self.z) for p in physical]
        patch = (float(px.min()), float(px.max()), float(py.min()), float(py.max()))
        self.near = near_components(e, self.packed, box, patch, h)
        profile = np.asarray(e.state.packed_lens_spec.profile_type)
        codes = tuple(sorted(int(c) for c in np.unique(profile[self.near])))
        self.groups = []
        for code in codes:
            index = self.near[profile[self.near] == code]
            padded = np.zeros(_bucket(len(index)), dtype=np.int32)
            padded[:len(index)] = index
            padded[len(index):] = index[0]
            self.groups.append((jnp.asarray(padded), jnp.asarray(np.arange(len(padded)) < len(index))))
        self.groups = tuple(self.groups)
        self.kernel = _near_kernel(draws, codes) if codes else None
        check_rows = np.unique(np.linspace(0, self.shape[0] - 1, min(self.shape[0], CHECK_POINTS)).round().astype(int))
        check_columns = np.unique(np.linspace(0, self.shape[1] - 1, min(self.shape[1], CHECK_POINTS)).round().astype(int))
        self.check = (check_rows[:, None] * self.shape[1] + check_columns[None, :]).ravel()
        self.direct: list[int] = []
        self.misses: list[float] = []

    def _near_fields(self, points: np.ndarray, packed) -> np.ndarray:
        """U (point, 2, 2) of the near components at ``points``."""
        out = np.zeros((len(points), 2, 2))
        if self.kernel is None or not len(points):
            return out
        chunk = int(min(16384, max(64, 1 << max(0, len(points) - 1).bit_length())))
        for start in range(0, len(points), chunk):
            block = points[start:start + chunk]
            pad = chunk - len(block)
            values = self.kernel(jnp.asarray(np.pad(block, ((0, pad), (0, 0))) if pad else block), packed, self.groups)
            out[start:start + len(block)] = np.asarray(values)[:len(block)]
        return out

    def smooth_part(self, k: int) -> np.ndarray:
        """The far part of draw ``k``'s parametric U at the patch nodes, at the reference redshift, without the shear:
        (row, column, 3) symmetric elements."""
        reference = self.jacobian[k][self.rows, self.columns].astype(float)
        near = self._near_fields(self.nodes, self.packed_reference[k]).reshape(reference.shape[:2] + (2, 2))
        return reference - _symmetric(self.shear[k]) - _symmetric(near)

    def interpolate(self, smooth: np.ndarray) -> np.ndarray:
        """The patch values ``smooth`` (row, column, 3) at the grid points, (point, 2, 2) in ravelled order."""
        from scipy.interpolate import RectBivariateSpline
        oy, ox = self.order
        fy, fx = self.fold_order
        iy, ix = self.fold_inverse
        parts = [RectBivariateSpline(*self.axes, smooth[oy][:, ox, c], kx=3, ky=3, s=0)(self.yf[fy], self.xf[fx], grid=True)[iy][:, ix]
                 for c in range(3)]
        return _full(np.stack(parts, axis=-1)).reshape(-1, 2, 2)

    def parametric(self, k: int) -> np.ndarray:
        """Draw ``k``'s parametric deflection Jacobian on the grid, (point, 2, 2) in ravelled order, at the grid's
        redshift; evaluated directly when the check misses."""
        e = self.evaluator
        u = self.scale * self.interpolate(self.smooth_part(k)) + self.shear[k] + self._near_fields(self.points, self.packed[k])
        exact = np.asarray(e.total_fields_at(self.points[self.check], self.packed[k], chunk=4096)[1], dtype=float)
        miss = float(np.max(np.abs(u[self.check] - exact))) if len(self.check) else 0.0
        self.misses.append(miss)
        if not (miss <= CHECK_TOLERANCE):
            self.direct.append(int(k))
            return np.asarray(e.total_fields_at(self.points, self.packed[k], chunk=4096)[1], dtype=float)
        return u

    def summary(self) -> str:
        worst = max(self.misses) if self.misses else float("nan")
        return (f", from the reference lens of {self.reference.name}: {len(self.near)} near components exact, the rest "
                f"interpolated (largest check miss {worst:.1e}); {len(self.direct)} of {len(self.packed)} draws evaluated "
                f"directly")
