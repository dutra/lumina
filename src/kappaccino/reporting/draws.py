"""The posterior draws behind every reported product, chosen once and evaluated once.

``reporting_draws`` picks ``config.outputs.draws`` random posterior draws over all chains with the run seed.  The
returned :class:`ReportingDraws` evaluates each draw lazily and keeps what it computed, so the figures and tables of one
output pass share the work.  Each draw carries one conditional realization of the shared GRF (``plotting.grf``), drawn
once with a seed fixed by the run seed and the draw's posterior row, and a grid of lens maps is evaluated once per draw
however many products read it.  Every product reports the median over these draws and their 16th to 84th percentile
range, which therefore holds the uncertainty of the lens parameters and of the field; none uses the single best sample.

The grids evaluate the draws in ``workers`` threads (``config.outputs.workers``, :func:`core.parallel.map_draws`);
the caches of this object are filled under locks, so the threads see one realization per draw.

Grids on disk.  With ``cache_stage`` set (the output stage, ``reporting.outputs.write_outputs``), the maps of every grid
also go to the stage's ``cache.h5``, keyed by the draws (:attr:`ReportingDraws.key`), the redshift, the grid points and
``GRID_MAPS_VERSION``, and a later pass over the same draws (a resumed run) reads them instead of evaluating the lens.
Only the most recently used grids stay in memory (``memory_bytes``); a grid asked for again after that is read back
from the cache.  A grid that could not be written stays in memory.  The cache only saves time: deleting ``cache.h5``
changes no map.

Small grids (``from_reference``) may take the smooth part of the parametric lens from a larger grid evaluated before
(``reference``, the overview) and evaluate exactly only the components that vary on the scale of its pixels and the
GRF realization: :mod:`reporting.local_maps`, switched by ``ReportingDraws.maps_from_reference``.
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..core import parallel
from ..core.parallel import map_draws
from ..io import store
from ..likelihood.grf_fields import _sigma0_factor, shared_field_evaluator, shared_theta

# The lens maps of a grid (``ReportingDraws.grid_maps``).
MAP_QUANTITIES = ("kappa_parametric", "kappa_grf", "detA_parametric", "kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu")
# Bump when the maps of a grid can change for the same draws, redshift and points: grids saved in cache.h5 with another
# version are evaluated again.
GRID_MAPS_VERSION = 1
# The maps of the overview grid, which two products read (``plotting.jwst_rgb`` and ``plotting.mass_figure``): the first
# request evaluates them all.
OVERVIEW_QUANTITIES = ("kappa_parametric", "kappa_grf", "kappa", "detA")
# Bytes of grid maps kept in memory between requests, least recently used out first; the rest is read from cache.h5.
GRID_MEMORY_BYTES = 1 << 30


def map_values(parametric: np.ndarray, grf: np.ndarray, quantities) -> dict[str, np.ndarray]:
    """The maps ``quantities`` (names in ``MAP_QUANTITIES``) at points with parametric and GRF deflection Jacobians
    ``parametric`` and ``grf`` (point, 2, 2), float64: convergence, shear, det A and magnification of the parametric
    lens and of parametric plus GRF."""
    total = parametric + grf
    a = np.eye(2) - total
    out: dict[str, np.ndarray] = {}
    det = None
    for q in quantities:
        if q == "kappa_parametric":
            out[q] = .5 * np.trace(parametric, axis1=-2, axis2=-1)
        elif q == "kappa_grf":
            out[q] = .5 * np.trace(grf, axis1=-2, axis2=-1)
        elif q == "detA_parametric":
            out[q] = np.linalg.det(np.eye(2) - parametric)
        elif q == "kappa":
            out[q] = .5 * np.trace(total, axis1=-2, axis2=-1)
        elif q == "gamma1":
            out[q] = .5 * (a[:, 1, 1] - a[:, 0, 0])
        elif q == "gamma2":
            out[q] = -.5 * (a[:, 0, 1] + a[:, 1, 0])
        else:                                   # detA, mu, abs_mu
            if det is None:
                det = np.linalg.det(a)
            if q == "detA":
                out[q] = det
            else:
                with np.errstate(divide="ignore"):
                    mu = 1.0 / det
                out[q] = mu if q == "mu" else np.abs(mu)
    return out


def _hash_value(h, value) -> None:
    """Feed ``value`` (an array, a scalar, a string, a sequence or mapping of them, or an object with attributes) into
    the hash ``h``, tagged with its type and shape so that different values never give the same bytes."""
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        h.update(f"{type(value).__name__}:{value!r};".encode())
    elif isinstance(value, dict):
        for name in sorted(value, key=str):
            h.update(f"<{name}>".encode())
            _hash_value(h, value[name])
    elif hasattr(value, "__dataclass_fields__") or (hasattr(value, "__dict__") and not hasattr(value, "__array__")):
        _hash_value(h, dict(vars(value)))
    else:
        try:
            array = np.asarray(value)
        except Exception:
            array = None
        if array is None or array.dtype == object:
            try:
                items = list(value)
            except TypeError:           # an opaque object (a function, a lock): its type only, never its address
                h.update(f"<{type(value).__qualname__}>".encode())
                return
            h.update(b"[")
            for item in items:
                _hash_value(h, item)
            h.update(b"]")
        else:
            h.update(f"{array.dtype.str}{array.shape};".encode() + np.ascontiguousarray(array).tobytes())


def _field_code() -> tuple:
    """The field's code versions, ``plotting.grf.PLOT_FIELD_VERSION`` and ``N_MODES``."""
    from ..plotting import grf
    return grf.PLOT_FIELD_VERSION, grf.N_MODES


def lens_digest(evaluator) -> str:
    """sha256 of what turns a latent vector into the lens maps besides the vector itself: the parameter names, their
    log-sampled mask and centering, the packed lens spec (every component's profile, family, fixed position, shape,
    radii and dispersion, the members' luminosity ratios and the parameter indices that override each field), the
    members' identities, exact or far-field classes and membership priors, the cosmology and redshifts, the constrained
    images and the GP settings (which condition the shared field), and the field's code (``plotting.grf.PLOT_FIELD_VERSION``
    and ``N_MODES``).  Evaluators without these attributes (test doubles) hash what they have."""
    h = hashlib.sha256(f"field:{_field_code()};".encode())
    state = getattr(evaluator, "state", None)
    for name in ("parameter_specs", "packed_lens_spec", "images", "z_lens", "kpc_per_arcsec", "z_source_values", "cosmo_config",
                 "gp_kernel", "gp_knee", "gp_blocks", "gp_reference_redshift", "gp_correlation_length_arcsec", "gp_floor_arcsec"):
        h.update(f"<state.{name}>".encode())
        _hash_value(h, getattr(state, name, None))
    members = getattr(state, "members", None)
    for name in ("member_ids", "component_indices", "exact_positions", "far_positions", "centered"):
        h.update(f"<members.{name}>".encode())
        _hash_value(h, getattr(members, name, None))
    for name in ("likelihood_mode", "log_positive_mask", "member_components", "membership_prior", "membership_calibration_index",
                 "membership_uncertain_positions", "membership_enumerated_positions", "factor_by_z", "nfw_rho_crit", "cosmology_fixed"):
        h.update(f"<{name}>".encode())
        _hash_value(h, getattr(evaluator, name, None))
    return h.hexdigest()


class ReportingDraws:
    """The reporting draws of one run and their cached lens evaluations.

    Attributes
    ----------
    evaluator : Evaluator
        Evaluator of the run.
    indices : numpy.ndarray
        Rows of the posterior behind the maps and other products, shape (draw,), sorted.
    samples, physical : numpy.ndarray
        Latent and physical parameter vectors of those draws, shape (draw, parameter).
    image_indices, image_samples : numpy.ndarray
        Rows and latent vectors of the (more numerous) draws behind the image positions and residuals, each solved in
        its own lens (:meth:`image_correction`); ``indices`` is a subset of ``image_indices``, both the leading entries
        of one seeded shuffle of the posterior.
    seed : int
        Seed of the choice.
    workers : int
        Threads that evaluate the draws of a grid (:meth:`grid_maps`) and of the image recovery
        (``diagnostics.image_recovery``); 1 is a plain loop.  The results do not depend on it.
    cache_stage : pathlib.Path or None
        Stage folder whose ``cache.h5`` holds the grid maps (None: no cache, every grid stays in memory).
    memory_bytes : int
        Bytes of grid maps kept in memory when they are also in the cache.
    """

    # Small grids requested with ``from_reference`` take the smooth part of the parametric lens from a reference grid
    # (``reporting.local_maps``); False evaluates every component on every grid (kept for comparison and tests).  On
    # every fold system, fold pair and cutout of Hera and AS1063 the zero contours of det A moved by at most 6e-6 arcsec.
    maps_from_reference = True

    def __init__(self, evaluator, posterior, count: int, seed: int, image_count: int | None = None, rows=None, *, workers: int = 1):
        self.evaluator = evaluator
        self.seed = int(seed)
        self.workers = int(workers)
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if rows is None:
            order = np.random.default_rng(self.seed).permutation(len(posterior.samples))
            self.indices = np.sort(order[:int(count)])
            self.image_indices = np.sort(order[:max(int(count), int(image_count or count))])
        else:        # chosen posterior rows, in the given order (scripts that put the best draw first)
            self.indices = self.image_indices = np.asarray(rows, dtype=int)
        self.samples = np.asarray(posterior.samples)[self.indices]
        self.image_samples = np.asarray(posterior.samples)[self.image_indices]
        self.physical = np.asarray(posterior.physical)[self.indices]
        self._field_draws: dict[int, dict] = {}
        self._corrections: dict[int, Any] = {}
        # Grid maps and reference lenses held in memory, by cache entry name: {"maps": {name: array}, "pinned": bool};
        # pinned entries (not in cache.h5) are never dropped.
        self._grids: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._references: list = []                         # reporting.local_maps.Reference of the reference grids
        self.cache_stage = None
        self.memory_bytes = GRID_MEMORY_BYTES
        self._lock = threading.Lock()                       # guards _shared and the creation of the per-draw locks
        self._draw_locks: dict[int, threading.Lock] = {}

    def _draw_lock(self, k: int) -> threading.Lock:
        """The lock of draw ``k``'s field realization, created once."""
        with self._lock:
            return self._draw_locks.setdefault(int(k), threading.Lock())

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def key(self) -> str:
        """A fingerprint of the draws (seed, indices, latent and physical parameter vectors) and of the lens they map to
        (:func:`lens_digest`: member catalog, fixed parameters, images, field code), for caches written to disk."""
        code = _field_code()
        with self._lock:                # hashed once per evaluator and field code
            cached = getattr(self, "_lens_digest", None)
            if cached is None or cached[0] != code:
                cached = self._lens_digest = (code, lens_digest(self.evaluator))
        return hashlib.sha256(str(self.seed).encode() + self.indices.tobytes() + self.samples.tobytes()
                              + np.ascontiguousarray(self.physical, dtype=float).tobytes() + cached[1].encode()).hexdigest()

    @property
    def image_key(self) -> str:
        """:attr:`key` together with the image draws (``image_indices``, ``image_samples``): the fingerprint of every
        draw behind the image recovery."""
        return hashlib.sha256(self.key.encode() + np.asarray(self.image_indices).tobytes()
                              + np.asarray(self.image_samples).tobytes()).hexdigest()

    @property
    def has_field(self) -> bool:
        """Whether the run has one shared GRF mass field (its own, or the all-block twin of a system-block run)."""
        return shared_field_evaluator(self.evaluator) is not None

    @property
    def field_is_rebuilt(self) -> bool:
        """Whether the shared field is rebuilt for a run that did not sample it (``gp_blocks`` other than ``"all"``).
        The posterior gave each block its own field, so a realization of one shared field is not a posterior sample of
        the lens: residuals, RMS and position scatter in the draws' own lenses come out too large."""
        twin = shared_field_evaluator(self.evaluator)
        return twin is not None and twin is not self.evaluator

    def shared(self):
        """The shared-field evaluator (the run's, or the all-block twin of a system-block run) and the draws in its
        latent coordinates (same physical parameters and sources), converted once."""
        with self._lock:
            cached = getattr(self, '_shared', None)
            if cached is None:
                twin = shared_field_evaluator(self.evaluator)
                if twin is None or twin is self.evaluator:
                    cached = (self.evaluator, np.asarray(self.samples))
                else:
                    cached = (twin, np.stack([shared_theta(self.evaluator, theta) for theta in self.samples]))
                self._shared = cached
            return cached

    def correction(self, k: int, max_separation: float):
        """Draw ``k``'s conditional realization of the shared field (``plotting.grf.FieldRealization``, its distance
        tables reaching ``max_separation`` arcsec); None without a shared field.  The realization is drawn once per
        draw, so every product and every grid sees the same field."""
        from ..plotting.grf import field_draw, field_realization
        twin = shared_field_evaluator(self.evaluator)
        if twin is None:
            return None
        base = twin._padded_evaluator()
        with self._draw_lock(k):        # one draw and one realization per draw, also when threads ask at once
            cached = self._corrections.get(k)
            if cached is None or cached[0] < max_separation:
                if k not in self._field_draws:
                    rng = np.random.default_rng([self.seed, int(self.indices[k])])
                    self._field_draws[k] = field_draw(base, shared_theta(self.evaluator, self.samples[k]), rng)
                cached = (float(max_separation), field_realization(base, self._field_draws[k], float(max_separation)))
                self._corrections[k] = cached
            return cached[1]

    def image_correction(self, j: int, max_separation: float):
        """The conditional realization of the shared field of image draw ``j`` (``image_indices[j]``): the product
        draw's own (:meth:`correction`) when the row is one, otherwise drawn with the same seeding and not kept."""
        from ..plotting.grf import field_draw, field_realization
        row = int(self.image_indices[j])
        k = np.flatnonzero(self.indices == row)
        if len(k):
            return self.correction(int(k[0]), max_separation)
        twin = shared_field_evaluator(self.evaluator)
        if twin is None:
            return None
        base = twin._padded_evaluator()
        draw = field_draw(base, shared_theta(self.evaluator, self.image_samples[j]), np.random.default_rng([self.seed, row]))
        return field_realization(base, draw, float(max_separation))

    def grf_weight(self, z_source: float) -> float:
        """Factor from the field's units (family zero at the reference redshift) to a source at ``z_source``."""
        twin = shared_field_evaluator(self.evaluator)
        state = twin.state
        return _sigma0_factor(state, z_source) / _sigma0_factor(state, state.gp_reference_redshift) / float(twin.gp_weights_per_family[0])

    def reach(self, xy: np.ndarray) -> float:
        """Largest distance (arcsec) from the bounding box of ``xy`` to an image of the shared field: what the
        realization's distance tables must reach to be evaluated at ``xy``."""
        base = shared_field_evaluator(self.evaluator)._padded_evaluator()
        observed = np.asarray(base.gp['xy'])[np.asarray(base.gp['mask'], bool)]
        corners = np.array([(a, b) for a in (xy[:, 0].min(), xy[:, 0].max()) for b in (xy[:, 1].min(), xy[:, 1].max())])
        return float(np.linalg.norm(corners[:, None] - observed[None], axis=-1).max())

    def lens(self, k: int, xy: np.ndarray, z_source: float, *, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``k``'s lens at ``xy`` (arcsec, shape (point, 2), or (row, column, 2) for a grid) for a source at
        ``z_source``: the total deflection (point, 2) of parametric plus the draw's GRF realization, and the parametric
        and GRF deflection Jacobians, each (point, 2, 2), points in ravelled order.  The GRF part is zero without a
        shared field.  A grid whose points are a meshgrid or another affine grid gets the field's separable evaluation
        (``plotting.grf.FieldRealization``), the same values to round-off."""
        grid = np.asarray(xy, dtype=float)
        xy = grid.reshape(-1, 2)
        e = self.evaluator
        alpha, parametric = e.total_fields_at(xy, e.packed_state_for_z(e.physical(jnp.asarray(self.samples[k])), z_source), chunk=chunk)
        alpha, parametric = np.array(alpha, dtype=float, copy=True), np.asarray(parametric, dtype=float)
        grf = np.zeros_like(parametric)
        if self.has_field and len(xy):
            field_alpha, grf[:] = self.field(k, grid, z_source)
            alpha += field_alpha
        return alpha, parametric, grf

    def field(self, k: int, xy: np.ndarray, z_source: float) -> tuple[np.ndarray, np.ndarray]:
        """Deflection (point, 2) and deflection Jacobian (point, 2, 2) of draw ``k``'s GRF realization at ``xy`` (as in
        :meth:`lens`) for a source at ``z_source``; the field part of :meth:`lens`."""
        grid = np.asarray(xy, dtype=float)
        xy = grid.reshape(-1, 2)
        weight = self.grf_weight(z_source)
        x, y = (grid[..., 0], grid[..., 1]) if grid.ndim == 3 else (xy[:, 0], xy[:, 1])
        values = [np.ravel(v) for v in self.correction(k, self.reach(xy))(x, y, np.zeros(len(xy), dtype=int))]   # chunked inside, compiled
        return np.stack(values[:2], axis=-1) * weight, np.stack(values[2:], axis=-1).reshape(-1, 2, 2) * weight

    def jacobians(self, k: int, xy: np.ndarray, z_source: float, *, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Parametric and GRF deflection Jacobians of draw ``k`` at ``xy`` (arcsec, shape (point, 2)) for a source at
        ``z_source``, each of shape (point, 2, 2); the GRF one is zero without a shared field."""
        _, parametric, grf = self.lens(k, xy, z_source, chunk=chunk)
        if not (np.isfinite(parametric).all() and np.isfinite(grf).all()):
            raise ValueError("Nonfinite lens Jacobian of a reporting draw")
        return parametric, grf

    def deflections(self, k: int, xy: np.ndarray, z_source: float) -> np.ndarray:
        """Total deflection (arcsec, shape (point, 2)) of draw ``k`` at ``xy`` for a source at ``z_source``: parametric
        plus its realization of the shared GRF."""
        return self.lens(k, xy, z_source)[0]

    def grid_maps(self, name: str, x: np.ndarray, y: np.ndarray, z_source: float, *, quantities=None, reference: bool = False,
                  from_reference: bool = False, log=print) -> dict[str, np.ndarray]:
        """Per-draw lens maps on a grid, evaluated once per (grid, redshift) and kept: ``kappa_parametric``,
        ``kappa_grf`` (the convergence of the draw's GRF realization), ``detA_parametric``, and ``kappa``, ``gamma1``, ``gamma2``,
        ``detA``, ``mu`` and ``abs_mu`` of parametric plus GRF, each of shape (draw,) + x.shape, float32.  Nonfinite input points stay NaN.
        The draws are evaluated in ``workers`` threads, the first alone; each writes its own rows of the maps.

        Parameters
        ----------
        name : str
            Label of the grid in the log.
        x, y : numpy.ndarray
            Points of the grid, arcsec, any shape.
        z_source : float
            Source redshift.
        quantities : iterable of str, optional
            The maps needed (names in ``MAP_QUANTITIES``); None (default) for all.  The returned dict may hold more.
        reference : bool, optional
            Keep the parametric Jacobian of every draw on this grid (a meshgrid) as a reference lens for the small grids
            asked with ``from_reference`` (:mod:`reporting.local_maps`).  Default False.
        from_reference : bool, optional
            With ``maps_from_reference``, evaluate this grid from a reference grid that covers it (exactly: the GRF
            realization and the components near the grid; interpolated: the rest of the parametric lens), and every
            component directly otherwise.  Default False.
        log : callable, optional
            Progress lines.

        Returns
        -------
        dict of numpy.ndarray
            The maps, by name.
        """
        from . import local_maps
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        wanted = tuple(MAP_QUANTITIES) if quantities is None else tuple(dict.fromkeys(str(q) for q in quantities))
        unknown = sorted(set(wanted) - set(MAP_QUANTITIES))
        if unknown:
            raise ValueError(f"unknown grid map(s) {unknown}; known: {list(MAP_QUANTITIES)}")
        z_source = float(z_source)
        source, why = None, ""
        if from_reference and self.maps_from_reference:
            source, why = local_maps.choose_reference(self, x, y, z_source)
        method = "direct" if source is None else source.tag
        key = json.dumps([GRID_MAPS_VERSION, self.key, repr(z_source), list(x.shape),
                          hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(), method])
        entry = "grid_maps_" + hashlib.sha256(key.encode()).hexdigest()[:40]
        size = (f"{x.size}-point list" if x.ndim == 1 else f"{x.shape[0]}x{x.shape[-1]} grid") + f" at z = {z_source:g}"
        held = self._grids.get(entry)
        maps = dict(held["maps"]) if held is not None else {}
        pinned = bool(held["pinned"]) if held is not None else False
        missing = [q for q in wanted if q not in maps]
        reason = ""
        if missing and self._uses_cache():
            stored, reason = self._read_cached(entry, key, (len(self),) + x.shape, MAP_QUANTITIES)
            if stored is not None:
                maps, pinned = {**stored, **maps}, False
                missing = [q for q in wanted if q not in maps]
                reason = f"cache.h5 lacks {', '.join(missing)}" if missing else ""
        keep = local_maps.Reference.of(local_maps.reference_entry(entry), name, x, y, z_source) if reference else None
        if keep is not None and not missing and not self._has_reference(keep.entry):
            missing, reason = list(wanted), "its reference lens is not in cache.h5"
        if not missing:
            if held is None:
                log(f"[draws] {name}: {len(self)} posterior draws on a {size}: reused from cache.h5")
            maps = self._remember(entry, maps, pinned=pinned)
            if keep is not None:
                self._register_reference(keep)
            return maps
        computed = tuple(dict.fromkeys(wanted + tuple(maps)))
        finite = np.isfinite(x) & np.isfinite(y)
        # A 2-D grid of finite points keeps its shape, so the field can be summed by rows and columns.
        xy = np.stack((x, y), axis=-1) if x.ndim == 2 and finite.all() else np.column_stack((x[finite], y[finite]))
        out = {q: np.full((len(self),) + x.shape, np.nan, dtype=np.float32) for q in computed}
        lens_reference = local_maps.ReferenceRows(self, keep) if keep is not None else None
        local = local_maps.LocalLens(self, source, x, y, z_source, log=log) if source is not None else None

        def evaluate(k: int) -> None:
            if local is None:
                parametric, grf = self.jacobians(k, xy, z_source)
            else:
                parametric = local.parametric(k)
                grf = self.field(k, xy, z_source)[1] if self.has_field else np.zeros_like(parametric)
                if not (np.isfinite(parametric).all() and np.isfinite(grf).all()):
                    raise ValueError("Nonfinite lens Jacobian of a reporting draw")
            values = map_values(parametric, grf, computed)
            for q in computed:
                out[q][k][finite] = values[q]
            if lens_reference is not None:
                lens_reference.add(k, parametric)
        map_draws(evaluate, range(len(self)), workers=self.workers, desc=f"{name} draws")
        note = f" ({reason})" if reason else (" (not in cache.h5)" if self._uses_cache() else "")
        log(f"[draws] {name}: {len(self)} posterior draws on a {size}" + (local.summary() if local is not None else "")
            + (f"; {why}" if why and local is None else "") + (f": evaluated{note}" if self._uses_cache() else ""))
        pinned = True
        if self._uses_cache():
            store.write_cache_arrays(self.cache_stage, entry, {**out, "key": np.array(key)})
            pinned = not store.has_cache_arrays(self.cache_stage, entry)
        maps = self._remember(entry, out, pinned=pinned)
        if lens_reference is not None:
            self._store_reference(keep, lens_reference.arrays())
        return maps

    # ------------------------------------------------------------------ memory and cache.h5
    def _uses_cache(self) -> bool:
        """Whether grids go to ``cache.h5``: a cache stage is set and this is not a worker thread of ``map_draws``
        (the store is written from the calling thread only)."""
        return self.cache_stage is not None and not getattr(parallel._inside, "active", False)

    def _read_cached(self, entry: str, key: str, shape: tuple, names) -> tuple[dict[str, np.ndarray] | None, str]:
        """The arrays of a cache entry written for ``key``, each of shape ``shape`` and float32 or (for a reference)
        as written, or (None, why not)."""
        try:
            arrays = store.read_cache_arrays(self.cache_stage, entry)
        except store.MissingEntry:
            return None, "not in cache.h5"
        except (store.DamagedEntry, store.StoreError) as error:      # a cache problem is a miss
            return None, f"cache.h5 unreadable ({type(error).__name__}: {error})"
        if str(arrays.pop("key", "")) != key:
            return None, "cache.h5 holds another key under its name"
        if not arrays or not set(arrays) <= set(names) or any(a.shape[:len(shape)] != shape for a in arrays.values()):
            return None, "cache.h5 holds arrays of another shape"
        return arrays, ""

    def _remember(self, entry: str, maps: dict[str, np.ndarray], *, pinned: bool) -> dict[str, np.ndarray]:
        """Hold ``maps`` in memory as the most recently used entry, then drop the least recently used entries that
        are in cache.h5 until the held bytes fit ``memory_bytes`` (the newest entry is kept)."""
        held = self._grids.pop(entry, None)
        if held is not None and set(held["maps"]) >= set(maps):
            held["pinned"] = bool(held["pinned"] and pinned)
            self._grids[entry] = held
        else:
            self._grids[entry] = held = dict(maps=maps, pinned=bool(pinned))
        total = sum(sum(a.nbytes for a in h["maps"].values()) for h in self._grids.values())
        for name in list(self._grids):
            if total <= self.memory_bytes or name == entry:
                break
            if not self._grids[name]["pinned"]:
                total -= sum(a.nbytes for a in self._grids.pop(name)["maps"].values())
        return held["maps"]

    def held_bytes(self) -> int:
        """Bytes of grid maps and reference lenses held in memory."""
        return int(sum(sum(a.nbytes for a in h["maps"].values()) for h in self._grids.values()))

    def _has_reference(self, entry: str) -> bool:
        return entry in self._grids or (self._uses_cache() and store.has_cache_arrays(self.cache_stage, entry))

    def _register_reference(self, reference) -> None:
        if all(r.entry != reference.entry for r in self._references):
            self._references.append(reference)

    def _store_reference(self, reference, arrays: dict[str, np.ndarray]) -> None:
        pinned = True
        if self._uses_cache():
            store.write_cache_arrays(self.cache_stage, reference.entry, {**arrays, "key": np.array(reference.key(self))})
            pinned = not store.has_cache_arrays(self.cache_stage, reference.entry)
        self._remember(reference.entry, arrays, pinned=pinned)
        self._register_reference(reference)

    def reference_arrays(self, reference, *, log=print) -> dict[str, np.ndarray]:
        """The reference lens of ``reference`` (``reporting.local_maps.ReferenceRows.arrays``): from memory, from
        cache.h5, or evaluated again on its grid when neither holds it."""
        from . import local_maps
        held = self._grids.get(reference.entry)
        if held is not None:
            return self._remember(reference.entry, held["maps"], pinned=held["pinned"])
        reason = "no cache"
        if self._uses_cache():
            stored, reason = self._read_cached(reference.entry, reference.key(self), (len(self),), local_maps.REFERENCE_ARRAYS)
            if stored is not None:
                return self._remember(reference.entry, stored, pinned=False)
        log(f"[draws] reference lens of {reference.name}: {reason}; evaluated again")
        rows = local_maps.ReferenceRows(self, reference)
        xy = np.stack(np.meshgrid(reference.xs, reference.ys), axis=-1)

        def evaluate(k: int) -> None:
            _, parametric = self.evaluator.total_fields_at(xy.reshape(-1, 2), self.evaluator.packed_state_for_z(
                self.evaluator.physical(jnp.asarray(self.samples[k])), reference.z), chunk=4096)
            rows.add(k, np.asarray(parametric, dtype=float))
        map_draws(evaluate, range(len(self)), workers=self.workers, desc=f"{reference.name} reference draws")
        self._store_reference(reference, rows.arrays())
        return self._grids[reference.entry]["maps"]


def reporting_draws(config, evaluator, posterior) -> ReportingDraws:
    """The reporting draws of a run (``config.outputs.draws`` product draws within ``config.outputs.image_draws``
    image draws, ``config.runtime.seed``, evaluated in ``config.outputs.workers`` threads), cached on the
    posterior object so every product of one output pass shares them and their evaluations."""
    count, seed = int(config.outputs.draws), int(config.runtime.seed)
    image_count, workers = int(config.outputs.image_draws), int(config.outputs.workers)
    cached = getattr(posterior, "_reporting_draws", None)
    if (cached is not None and cached.evaluator is evaluator and len(cached) == min(count, len(posterior.samples)) and cached.seed == seed
            and len(cached.image_indices) == min(max(count, image_count), len(posterior.samples))):
        cached.workers = workers
        return cached
    draws = ReportingDraws(evaluator, posterior, count, seed, image_count, workers=workers)
    try:
        object.__setattr__(posterior, "_reporting_draws", draws)
    except (AttributeError, TypeError):
        pass
    return draws


def percentiles(values: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    """``{"median", "q16", "q84"}`` of ``values`` along ``axis``, ignoring NaN."""
    q16, q50, q84 = np.nanpercentile(np.asarray(values, dtype=float), [16, 50, 84], axis=axis)
    return {"median": q50, "q16": q16, "q84": q84}
