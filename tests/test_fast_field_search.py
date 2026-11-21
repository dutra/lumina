"""The fast field evaluation (vectorisable sincos, separable grids, finer padding) and the extra-image search on one
shared lattice: agreement with the direct sums and with the search per source redshift, and the fallback to it."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from lumina.config import NFWHaloConfig, ShearHaloConfig
from lumina.core.state import build_state
from lumina.diagnostics.exact_images import ExactImageSolver
from lumina.likelihood.evaluator import Evaluator
from lumina.plotting import grf


# ----------------------------------------------------------------------------- sincos
def test_sincos_matches_numpy_over_a_wide_range_of_arguments():
    rng = np.random.default_rng(1)
    quarter = np.pi / 4
    x = np.concatenate([np.linspace(-10, 10, 20001), rng.uniform(-1e3, 1e3, 20000), rng.uniform(-1e5, 1e5, 20000),
                        rng.uniform(-1e6, 1e6, 20000), np.arange(-4000, 4001) * quarter,        # quadrant boundaries
                        np.arange(-4000, 4001) * quarter + 1e-12, [0., -0., 1e-300, -1e-300, 5e-324]])
    sine, cosine = map(np.asarray, grf.sincos(jnp.asarray(x)))
    assert np.abs(sine - np.sin(x)).max() <= 4e-16 and np.abs(cosine - np.cos(x)).max() <= 4e-16
    assert np.abs(sine ** 2 + cosine ** 2 - 1).max() <= 1e-15
    # Beyond |x| ~ 1.6e6 the three-piece reduction is no longer exact: the error grows like eps |x|, as in a plain
    # float64 reduction.
    big = rng.uniform(-1e9, 1e9, 1000)
    assert np.abs(np.asarray(grf.sincos(jnp.asarray(big))[0]) - np.sin(big)).max() <= 1e-15 * 1e9
    special = np.asarray(grf.sincos(jnp.asarray([np.nan, np.inf, -np.inf])))
    assert np.isnan(special).all()


# ----------------------------------------------------------------------------- plane waves
def _modes(n=512, seed=2):
    rng = np.random.default_rng(seed)
    magnitude = np.geomspace(.05, 16., n)
    angle = rng.uniform(0, 2 * np.pi, n)
    kvec = magnitude[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
    a, b = rng.normal(0, 2e-3, (2, n))
    return kvec, a, b


def _reference(points, kvec, a, b):
    value, derivative = grf._mode_fields(jnp.asarray(points), jnp.asarray(kvec), jnp.asarray(a), jnp.asarray(b))
    return np.column_stack([np.asarray(value), np.asarray(derivative).reshape(-1, 4)])


def test_the_mode_sum_with_the_fast_sincos_matches_the_direct_sum():
    kvec, a, b = _modes()
    points = np.random.default_rng(3).uniform(-80, 80, (3000, 2))
    fast = grf._chunked(grf._mode_points, points, jnp.asarray(kvec), jnp.asarray(grf.mode_weights(kvec, a, b)))[0]
    reference = _reference(points, kvec, a, b)
    assert np.abs(fast - reference).max() <= 1e-14 * np.abs(reference).max()


@pytest.mark.parametrize("kind", ["meshgrid", "nonuniform", "rotated", "row", "column"])
def test_the_separable_kernel_matches_the_point_path(kind):
    kvec, a, b = _modes()
    rng = np.random.default_rng(4)
    if kind == "meshgrid":
        x, y = np.meshgrid(np.linspace(-40, 35, 150), np.linspace(-20, 50, 70))
    elif kind == "nonuniform":
        x, y = np.meshgrid(np.sort(rng.uniform(-40, 40, 90)), np.sort(rng.uniform(-30, 30, 130)))
    elif kind == "rotated":
        angle, origin = .3, np.array([12.5, -7.25])
        i, j = np.meshgrid(np.arange(80.), np.arange(66.))
        u, v = np.array([np.cos(angle), np.sin(angle)]) * .37, np.array([-np.sin(angle), np.cos(angle)]) * .41
        x, y = origin[0] + i * u[0] + j * v[0], origin[1] + i * u[1] + j * v[1]
    elif kind == "row":
        x, y = np.linspace(-30, 30, 333)[None], np.full((1, 333), 4.2)
    else:
        x, y = np.full((200, 1), -3.), np.linspace(-30, 30, 200)[:, None]
    grid = grf.separable_grid(x, y)
    assert grid is not None
    rows, columns = grid
    np.testing.assert_allclose(rows[:, None] + columns[None], np.stack([x, y], -1), rtol=0, atol=1e-12)
    separable = grf._mode_grid_values(rows, columns, jnp.asarray(kvec), jnp.asarray(grf.mode_weights(kvec, a, b)))
    reference = _reference(np.column_stack([x.ravel(), y.ravel()]), kvec, a, b)
    assert np.abs(separable - reference).max() <= 1e-13 * np.abs(reference).max()


def test_separable_grid_detection_rejects_what_is_not_a_grid():
    x, y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-2, 2, 15))
    assert grf.separable_grid(x, y) is not None
    jitter = np.random.default_rng(5).normal(0, 1e-6, x.shape)
    assert grf.separable_grid(x + jitter, y) is None                       # irregular points
    assert grf.separable_grid(x, y + jitter) is None
    holed = x.copy()
    holed[3, 4] = np.nan
    assert grf.separable_grid(holed, y) is None                             # a NaN point
    assert grf.separable_grid(x.ravel(), y.ravel()) is None                 # not 2-D
    assert grf.separable_grid(x, y[:, :-1]) is None                         # shapes differ
    assert grf.separable_grid(np.zeros((0, 3)), np.zeros((0, 3))) is None


def test_the_padding_ladder_is_powers_of_two_up_to_the_chunk():
    assert [grf._padded_size(n) for n in (1, 64, 65, 300, 512, 513, 4096)] == [64, 64, 128, 512, 512, 1024, 4096]


# ----------------------------------------------------------------------------- the realization on grids
def _realization():
    from test_grf_plotting import _toy_conditioning
    toy = _toy_conditioning(('translation', 'convergence', 'shear'))
    draw = grf.draw_field(toy['layout'], toy['tables'], toy['cholesky'], toy['v'], toy['sigma'], toy['lam'], 4.,
                          modes=('translation', 'convergence', 'shear'), rng=np.random.default_rng([3, 17]), n_modes=1024)
    layout = dict(toy['layout'], block_of_family=jnp.array([0, 0]))
    return grf.FieldRealization(layout, toy['tables'], draw, np.array([1., .5]), modes=('translation', 'convergence', 'shear'))


def test_a_realization_on_a_grid_equals_its_values_point_by_point():
    field = _realization()
    angle = .4
    i, j = np.meshgrid(np.arange(-20., 21.), np.arange(-15., 16.))
    x, y = .1 * (i * np.cos(angle) - j * np.sin(angle)) + .3, .1 * (i * np.sin(angle) + j * np.cos(angle)) - .2
    families = np.ones(x.size, int)
    on_grid = np.asarray(field(x, y, families))
    by_point = np.asarray(field(x.ravel(), y.ravel(), families))
    assert np.abs(on_grid - by_point).max() <= 1e-13 * np.abs(by_point).max()
    select = np.random.default_rng(6).random(x.shape) < .3
    np.testing.assert_allclose(field.unweighted(x, y, select=select), field.unweighted(x[select], y[select]), rtol=0, atol=1e-13)
    np.testing.assert_allclose(by_point, .5 * np.asarray(field(x.ravel(), y.ravel(), 0 * families)), rtol=1e-15, atol=0)


# ----------------------------------------------------------------------------- the shared lattice
def test_the_shared_lattice_covers_every_window():
    rng = np.random.default_rng(7)
    windows = np.column_stack([rng.uniform(-60, 60, 300), rng.uniform(-60, 60, 300), rng.uniform(.3, 40, 300)])
    windows[:5] = [[0, 0, 1.], [.25, -.25, .25], [1e-13, 0, 3.0], [-17.5, 4.5, 2.5], [10., 10., .1]]   # edges on nodes
    for pitch in (.5, .3):
        boxes = ExactImageSolver.lattice_boxes(windows, pitch)
        cx, cy, half = windows.T
        assert np.all(boxes[:, 0] * pitch <= cx - half + 1e-12) and np.all(boxes[:, 1] * pitch >= cx + half - 1e-12)
        assert np.all(boxes[:, 2] * pitch <= cy - half + 1e-12) and np.all(boxes[:, 3] * pitch >= cy + half - 1e-12)
        # snapped outward by less than one pitch on each side
        assert np.all(boxes[:, 0] * pitch > cx - half - pitch) and np.all(boxes[:, 1] * pitch < cx + half + pitch)
        assert np.all(boxes[:, 2] * pitch > cy - half - pitch) and np.all(boxes[:, 3] * pitch < cy + half + pitch)


def _sheared(synthetic_config, *, nfw=False):
    halos = synthetic_config.model.large_halos + (ShearHaloConfig('shear', .06, 25., .3),)
    if nfw:
        halos = halos + (NFWHaloConfig(id='nfw', x_centre=5., y_centre=-3., angle_pos=10., ellipticite=.1, log10_m200=13.5,
                                       scale_radius_kpc=150., z_lens=.3),)
    config = replace(synthetic_config, model=replace(synthetic_config.model, large_halos=halos))
    state = build_state(config)
    evaluator = Evaluator(state)
    diagnostics = replace(config.outputs.image_solver, exact_image_finder="global-jax", exact_image_window_margin_arcsec=12.0)
    solver = ExactImageSolver(evaluator, diagnostics)
    theta = jnp.asarray(state.initial_theta)
    physical, sources = np.asarray(evaluator.physical(theta)), np.asarray(evaluator.source_positions(theta))
    family_by_id = {f.family_id: f for f in state.family_data}
    return state, evaluator, solver, physical, sources, [family_by_id[fid] for fid in state.images.family_ids]


def test_the_lens_at_any_redshift_is_the_reference_lens_scaled_with_the_shear_apart(synthetic_config):
    from lumina.likelihood.grf_fields import _sigma0_factor
    state, evaluator, solver, physical, _, _ = _sheared(synthetic_config, nfw=True)
    points = np.random.default_rng(8).uniform(-20, 20, (500, 2))
    reference = evaluator.packed_state_for_z(jnp.asarray(physical), 2.)
    scaled_alpha, scaled_u, shear_alpha, shear_u = solver._split_fields(points, reference)
    assert np.abs(shear_u).max() > .04                                     # the shear is there (0.06)
    for z in (1.1, 3., 6.):
        alpha, u = evaluator.total_fields_at(points, evaluator.packed_state_for_z(jnp.asarray(physical), z))
        s = _sigma0_factor(state, z) / _sigma0_factor(state, 2.)
        np.testing.assert_allclose(s * scaled_alpha + shear_alpha, alpha, rtol=0, atol=1e-13 * np.abs(alpha).max())
        np.testing.assert_allclose(s * scaled_u + shear_u, u, rtol=0, atol=1e-13 * np.abs(u).max())
        # Scaling the shear too is wrong by the shear's share of the lens.
        assert np.abs(s * (scaled_u + shear_u) - u).max() > 1e-3


class PlaneWaveField:
    """A field realization stand-in: one curl-free plane wave of unit weight, times a weight per family."""

    def __init__(self, weights, amplitude=.02, k=(.7, -.4)):
        self.weights, self.amplitude, self.k = np.asarray(weights, float), amplitude, np.asarray(k)

    def unweighted(self, x, y, select=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if select is not None:
            x, y = x[np.asarray(select, bool)], y[np.asarray(select, bool)]
        x, y = x.ravel(), y.ravel()
        phase = self.k[0] * x + self.k[1] * y
        khat, norm = self.k / np.linalg.norm(self.k), np.linalg.norm(self.k)
        value = self.amplitude * np.cos(phase)[:, None] * khat
        jac = -self.amplitude * norm * np.sin(phase)[:, None, None] * np.outer(khat, khat)
        return np.column_stack([value, jac.reshape(-1, 4)])

    def __call__(self, x, y, families):
        return tuple((self.unweighted(x, y) * self.weights[np.asarray(families, int).ravel()][:, None]).T)


def _same_images(a, b, tolerance):
    for (ax, ay), (bx, by) in zip(a, b):
        assert ax.size == bx.size
        if ax.size:
            d = np.hypot(ax[:, None] - bx[None], ay[:, None] - by[None])
            assert d.min(axis=1).max() < tolerance and d.min(axis=0).max() < tolerance


@pytest.mark.parametrize("field", ["none", "realization", "callable"])
def test_the_shared_lattice_finds_the_images_of_the_search_per_redshift(synthetic_config, field):
    state, evaluator, solver, physical, sources, families = _sheared(synthetic_config)
    correction = {"none": None, "realization": PlaneWaveField([1., 1.6]),
                  "callable": lambda x, y, f: PlaneWaveField([1., 1.6])(x, y, f)}[field]
    shared = solver.solve_all_global(families, physical, sources, correction)
    assert solver.search_path == "shared"
    per_redshift = ExactImageSolver(evaluator, solver.config)
    per_redshift.shared_lattice = False
    reference = per_redshift.solve_all_global(families, physical, sources, correction)
    assert per_redshift.search_path == "per_redshift"
    assert sum(x.size for x, _ in shared) >= len(families)
    _same_images(shared, reference, 1e-8)
    # every image solves the lens equation in its family's lens
    for k, (x, y) in enumerate(shared):
        packed = evaluator.packed_state_for_z(jnp.asarray(physical), families[k].z_source)
        bx, by, *_ = ExactImageSolver._corrected(solver.ray_shoot_and_jacobian, None if correction is None else
                                                 (lambda xx, yy: correction(xx, yy, np.full(len(xx), k))))(packed, x, y)
        np.testing.assert_allclose(np.column_stack([bx, by]), np.broadcast_to(sources[k], (x.size, 2)), atol=1e-10)


def _fallback_gives_the_per_redshift_search(solver, evaluator, families, physical, sources, correction=None):
    found = solver.solve_all_global(families, physical, sources, correction)
    assert solver.search_path == "per_redshift" and solver.search_counts == {"shared": 0, "per_redshift": 1}
    reference = ExactImageSolver(evaluator, solver.config)
    reference.shared_lattice = False
    expected = reference.solve_all_global(families, physical, sources, correction)
    for (x, y), (ex, ey) in zip(found, expected):
        np.testing.assert_array_equal(x, ex)
        np.testing.assert_array_equal(y, ey)


def test_a_lens_that_does_not_scale_with_redshift_takes_the_search_per_redshift(synthetic_config, monkeypatch):
    from lumina.likelihood import grf_fields
    state, evaluator, solver, physical, sources, families = _sheared(synthetic_config)
    assert len({f.z_source for f in families}) == 2
    true_factor = grf_fields._sigma0_factor
    # A wrong efficiency ratio: the numerical check at the probe points must catch it.
    monkeypatch.setattr(grf_fields, "_sigma0_factor", lambda s, z: true_factor(s, z) * (1.01 if float(z) == 3. else 1.))
    assert solver._redshift_scaling(families, physical, None) is None
    _fallback_gives_the_per_redshift_search(solver, evaluator, families, physical, sources)


def test_an_unknown_profile_or_an_inconsistent_field_takes_the_search_per_redshift(synthetic_config):
    state, evaluator, solver, physical, sources, families = _sheared(synthetic_config)
    assert solver._redshift_scaling(families, physical, None) is not None
    solver._profile_codes = frozenset({999})
    _fallback_gives_the_per_redshift_search(solver, evaluator, families, physical, sources)

    class Inconsistent(PlaneWaveField):          # its families are not weights times one field
        def __call__(self, x, y, families):
            values = self.unweighted(x, y) * self.weights[np.asarray(families, int).ravel()][:, None]
            return tuple((values * (1 + .1 * np.asarray(families).ravel()[:, None])).T)
    solver = ExactImageSolver(evaluator, solver.config)
    _fallback_gives_the_per_redshift_search(solver, evaluator, families, physical, sources, Inconsistent([1., 1.6]))


def test_the_fine_sub_grid_of_the_shared_lattice_finds_a_close_pair_the_coarse_lattice_misses(synthetic_config):
    # A source 0.01" inside a fold caustic of family 1's lens: two of its three images straddle the critical curve
    # closer together than the 0.5" coarse pitch, so the coarse lattice seeds only one of them.
    state, evaluator, solver, physical, sources, families = _sheared(synthetic_config)
    sources = sources.copy()
    sources[0] = [-2.3800155898071496, -4.797666823826104]
    shared = solver.solve_all_global(families, physical, sources)
    assert solver.search_path == "shared" and shared[0][0].size == 3
    coarse_only = ExactImageSolver(evaluator, replace(solver.config, exact_image_critical_det_threshold=0.))   # no fine sub-grid
    assert coarse_only.solve_all_global(families, physical, sources)[0][0].size == 2
    x, y = shared[0]
    assert np.sort(np.hypot(x[:, None] - x[None], y[:, None] - y[None])[np.triu_indices(3, 1)])[0] < solver.config.exact_image_grid_pitch_arcsec
    per_redshift = ExactImageSolver(evaluator, solver.config)
    per_redshift.shared_lattice = False
    _same_images(shared, per_redshift.solve_all_global(families, physical, sources), 1e-8)
    packed = evaluator.packed_state_for_z(jnp.asarray(physical), families[0].z_source)
    bx, by, *_ = solver.ray_shoot_and_jacobian(packed, x, y)
    np.testing.assert_allclose(np.column_stack([bx, by]), np.broadcast_to(sources[0], (3, 2)), atol=1e-10)


def test_a_lens_that_scales_at_the_extreme_redshifts_but_not_between_takes_the_search_per_redshift(synthetic_config, monkeypatch):
    from lumina.likelihood import grf_fields
    state, evaluator, solver, physical, sources, families = _sheared(synthetic_config)
    assert [f.z_source for f in families] == [2., 3.]
    families = families + [replace(families[0], family_id="between", z_source=2.5)]       # a third redshift, between
    sources = np.vstack([sources, sources[:1]])
    assert solver._redshift_scaling(families, physical, None) is not None
    true_factor = grf_fields._sigma0_factor
    # The efficiency ratio is wrong at z = 2.5 only: the reference (2) and the farthest redshift (3) still scale.
    monkeypatch.setattr(grf_fields, "_sigma0_factor", lambda s, z: true_factor(s, z) * (1.01 if float(z) == 2.5 else 1.))
    assert solver._redshift_scaling(families[:2], physical, None) is not None
    assert solver._redshift_scaling(families, physical, None) is None
    _fallback_gives_the_per_redshift_search(solver, evaluator, families, physical, sources)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("shared", [True, False])
def test_every_grid_point_of_the_search_lies_within_the_reach_of_the_field(synthetic_config, monkeypatch, shared):
    # A coarse pitch of 2" (fine 1") and windows whose edges miss the lattice; every node within the seed reach and
    # near a critical curve, so fine sub-grids surround the edge nodes too.
    from lumina.diagnostics.image_recovery import window_corners
    state, evaluator, solver, physical, sources, families = _sheared(synthetic_config)
    config = replace(solver.config, exact_image_grid_pitch_arcsec=2., exact_image_fine_pitch_arcsec=1., exact_image_window_margin_arcsec=.3,
                     exact_image_critical_det_threshold=1e30, exact_image_seed_tolerance_pitches=1e30)
    solver = ExactImageSolver(evaluator, config)
    solver.shared_lattice = shared
    observed = np.column_stack([state.images.x_obs, state.images.y_obs])

    def reach(xy):                     # as ReportingDraws.reach: from the bounding box of xy to the farthest image
        corners = np.array([(a, b) for a in (xy[:, 0].min(), xy[:, 0].max()) for b in (xy[:, 1].min(), xy[:, 1].max())])
        return float(np.linalg.norm(corners[:, None] - observed[None], axis=-1).max())

    visited = []

    class Recording(PlaneWaveField):
        def unweighted(self, x, y, select=None):
            x, y = np.asarray(x, float), np.asarray(y, float)
            keep = np.ones(x.shape, bool) if select is None else np.asarray(select, bool)
            visited.append(np.column_stack([x[keep], y[keep]]))
            return super().unweighted(x, y, select)

    def stop(*args, **kwargs):         # the grid stage is over: the refinement's iterates are not grid points
        raise _Stop
    monkeypatch.setattr(solver, "_refine_merged", stop)
    with pytest.raises(_Stop):
        solver.solve_all_global(families, physical, sources, Recording([1., 1.6]))
    assert solver.search_path == ("shared" if shared else "per_redshift")
    visited = np.concatenate(visited)
    farthest = float(np.linalg.norm(visited[:, None] - observed[None], axis=-1).max())
    windows = np.asarray(solver.search_windows(families))
    corners = np.concatenate([windows[:, :2] + windows[:, 2:] * sign for sign in ((-1, -1), (1, 1))])
    assert farthest > reach(corners) + .5            # the windows alone fall short of the grid
    assert farthest <= reach(window_corners(solver, state)) + 1e-9
