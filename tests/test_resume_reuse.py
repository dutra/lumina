"""Reuse on resume and fold maps from the overview: the image-recovery fingerprint, the grid maps in cache.h5 (hit,
miss, damaged file, memory policy) and the local maps of ``reporting.local_maps`` (exact algebra, the check, the
switch)."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from kappaccino.io import store
from kappaccino.reporting import draws as reporting
from kappaccino.reporting import local_maps


# ----------------------------------------------------------------------------- a lens that counts its evaluations
def _counting_run(monkeypatch):
    """The lens of test_reporting_draws: parametric Jacobian theta[0] I, GRF Jacobian theta[1] I at the reference
    redshift; ``calls`` collects the number of points of every parametric evaluation."""
    from kappaccino.lens import cosmology
    monkeypatch.setattr(cosmology, 'dpie_sigma0_factor_from_config', lambda zl, z, c: z)
    calls = []

    def total_fields_at(xy, packed, chunk=4096):
        calls.append(len(xy))
        return np.zeros((len(xy), 2)), np.broadcast_to(packed * np.eye(2), (len(xy), 2, 2)).copy()
    evaluator = SimpleNamespace(gp_knee=True, sigma_grf_index=0, gp_knee_spectrum='knee',
                                state=SimpleNamespace(gp_blocks='all', z_lens=.3, gp_reference_redshift=2., cosmo_config=None),
                                gp_weights_per_family=np.array([1.]), physical=lambda theta: np.asarray(theta),
                                packed_state_for_z=lambda physical, z: float(physical[0]), total_fields_at=total_fields_at)
    posterior = SimpleNamespace(samples=np.array([[.1, .0], [.2, .05], [.3, -.05], [.4, .1]]),
                                physical=np.array([[.1, .0], [.2, .05], [.3, -.05], [.4, .1]]))
    monkeypatch.setattr(reporting.ReportingDraws, 'reach', lambda self, xy: 1.)
    monkeypatch.setattr(reporting.ReportingDraws, 'correction', lambda self, k, reach: (
        lambda x, y, f: (0*x, 0*x, self.samples[k, 1] + 0*x, 0*x, 0*x, self.samples[k, 1] + 0*x)))
    return evaluator, posterior, calls


def _quiet(_):
    pass


def _grid():
    return np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-2, 2, 5))


def test_a_second_pass_reads_the_grid_from_cache_h5_with_identical_maps(monkeypatch, tmp_path):
    evaluator, posterior, calls = _counting_run(monkeypatch)
    x, y = _grid()
    first = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    first.cache_stage = tmp_path
    lines = []
    maps = first.grid_maps('grid', x, y, 4., log=lines.append)
    assert len(calls) == 4 and 'evaluated (not in cache.h5)' in lines[-1]
    second = reporting.ReportingDraws(evaluator, posterior, 4, 0)       # a resumed run: a new object, same draws
    second.cache_stage = tmp_path
    again = second.grid_maps('grid', x, y, 4., log=lines.append)
    assert len(calls) == 4 and lines[-1].endswith('reused from cache.h5')
    assert set(again) == set(maps)
    for q in maps:
        assert again[q].dtype == np.float32
        np.testing.assert_array_equal(again[q], maps[q], err_msg=q)


def test_other_draws_a_version_bump_or_a_deleted_cache_evaluate_again(monkeypatch, tmp_path):
    evaluator, posterior, calls = _counting_run(monkeypatch)
    x, y = _grid()

    def run(seed=0, count=4):
        draws = reporting.ReportingDraws(evaluator, posterior, count, seed)
        draws.cache_stage = tmp_path
        return draws.grid_maps('grid', x, y, 4., log=_quiet)
    reference = run()
    n = len(calls)
    run(seed=1, count=3)
    assert len(calls) == n + 3                                          # other draws: another key
    run()
    assert len(calls) == n + 3                                          # the first key is still cached
    monkeypatch.setattr(reporting, 'GRID_MAPS_VERSION', reporting.GRID_MAPS_VERSION + 1)
    run()
    assert len(calls) == n + 7
    store.cache_path(tmp_path).unlink()
    monkeypatch.setattr(reporting, 'GRID_MAPS_VERSION', reporting.GRID_MAPS_VERSION - 1)
    again = run()
    assert len(calls) == n + 11
    for q in reference:
        np.testing.assert_array_equal(again[q], reference[q])


@pytest.mark.parametrize('damage', ['truncated', 'garbage', 'empty'])
def test_a_damaged_cache_h5_is_a_miss_not_a_crash(monkeypatch, tmp_path, damage):
    evaluator, posterior, calls = _counting_run(monkeypatch)
    x, y = _grid()
    draws = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    draws.cache_stage = tmp_path
    reference = draws.grid_maps('grid', x, y, 4., log=_quiet)
    path = store.cache_path(tmp_path)
    data = path.read_bytes()
    path.write_bytes({'truncated': data[:len(data) // 3], 'garbage': b'not an hdf5 file' * 100, 'empty': b''}[damage])
    store_lines = []
    previous = store.set_log(store_lines.append)
    try:
        again = reporting.ReportingDraws(evaluator, posterior, 4, 0)
        again.cache_stage = tmp_path
        lines = []
        maps = again.grid_maps('grid', x, y, 4., log=lines.append)
    finally:
        store.set_log(previous)
    assert len(calls) == 8 and 'evaluated' in lines[-1]
    for q in reference:
        np.testing.assert_array_equal(maps[q], reference[q])
    third = reporting.ReportingDraws(evaluator, posterior, 4, 0)       # the next writer replaced the damaged file
    third.cache_stage = tmp_path
    third.grid_maps('grid', x, y, 4., log=_quiet)
    assert len(calls) == 8


def test_only_the_quantities_asked_for_are_kept_and_more_are_evaluated_when_asked(monkeypatch, tmp_path):
    evaluator, posterior, calls = _counting_run(monkeypatch)
    x, y = _grid()
    draws = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    draws.cache_stage = tmp_path
    maps = draws.grid_maps('grid', x, y, 4., quantities=('detA',), log=_quiet)
    assert set(maps) == {'detA'} and len(calls) == 4
    assert set(draws.grid_maps('grid', x, y, 4., quantities=('detA',), log=_quiet)) == {'detA'} and len(calls) == 4
    lines = []
    both = draws.grid_maps('grid', x, y, 4., quantities=('kappa', 'detA'), log=lines.append)
    assert set(both) == {'kappa', 'detA'} and len(calls) == 8 and 'lacks kappa' in lines[-1]
    np.testing.assert_array_equal(both['detA'], maps['detA'])
    with pytest.raises(ValueError, match='unknown grid map'):
        draws.grid_maps('grid', x, y, 4., quantities=('kapa',), log=_quiet)
    full = reporting.ReportingDraws(evaluator, posterior, 4, 0).grid_maps('grid', x, y, 4., log=_quiet)
    for q in both:
        np.testing.assert_array_equal(both[q], full[q])


def test_memory_holds_the_newest_grids_and_reads_the_others_back(monkeypatch, tmp_path):
    evaluator, posterior, calls = _counting_run(monkeypatch)
    grids = [np.meshgrid(np.linspace(-1, 1, 30) + i, np.linspace(-1, 1, 20)) for i in range(4)]
    one = 4 * 30 * 20 * 4 * len(reporting.MAP_QUANTITIES)              # bytes of one grid's maps
    draws = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    draws.cache_stage = tmp_path
    draws.memory_bytes = int(1.5 * one)
    first = [draws.grid_maps(f'g{i}', x, y, 4., log=_quiet) for i, (x, y) in enumerate(grids)]
    assert len(calls) == 16
    assert draws.held_bytes() <= draws.memory_bytes and len(draws._grids) == 1
    lines = []
    again = draws.grid_maps('g0', *grids[0], 4., log=lines.append)
    assert len(calls) == 16 and lines[-1].endswith('reused from cache.h5')
    for q in again:
        np.testing.assert_array_equal(again[q], first[0][q])
    # Without a cache nothing is dropped (as before), and a grid whose cache write fails stays in memory.
    plain = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    plain.memory_bytes = 0
    for i, (x, y) in enumerate(grids):
        plain.grid_maps(f'g{i}', x, y, 4., log=_quiet)
    assert len(plain._grids) == 4 and plain.held_bytes() == 4 * one
    failing = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    failing.cache_stage, failing.memory_bytes = tmp_path / 'unwritable', 0
    monkeypatch.setattr(store, 'write_cache_arrays', lambda *args, **kwargs: None)
    for i, (x, y) in enumerate(grids):
        failing.grid_maps(f'h{i}', x + 50, y, 4., log=_quiet)
    assert len(failing._grids) == 4 and all(entry['pinned'] for entry in failing._grids.values())


def test_the_maps_are_identical_with_and_without_the_cache_in_a_worker_thread(monkeypatch, tmp_path):
    from kappaccino.core.parallel import map_draws
    evaluator, posterior, calls = _counting_run(monkeypatch)
    x, y = _grid()
    draws = reporting.ReportingDraws(evaluator, posterior, 4, 0, workers=2)
    draws.cache_stage = tmp_path
    inside = map_draws(lambda k: draws.grid_maps('grid', x + k, y, 4., log=_quiet), range(2), workers=2)
    # The first item runs in the calling thread and writes its grid; the worker thread's grid stays out of the store.
    assert len(store.entries(tmp_path, cache=True)['arrays']) == 1
    outside = reporting.ReportingDraws(evaluator, posterior, 4, 0).grid_maps('grid', x + 1, y, 4., log=_quiet)
    for q in outside:
        np.testing.assert_array_equal(inside[1][q], outside[q])


# ----------------------------------------------------------------------------- image-recovery fingerprint
def _recovery_inputs(seed=7, count=3, image_count=4):
    family = SimpleNamespace(family_id='1', z_source=2., image_labels=['1.a', '1.b'], x_obs=np.array([1., -1.]),
                             y_obs=np.array([2., -2.]), search_window=4., x_center=0., y_center=0.)
    state = SimpleNamespace(family_data=[family], images=SimpleNamespace(image_labels=['1.a', '1.b'], family_ids=['1']),
                            parameter_specs=[SimpleNamespace(name='halo.v_disp')])
    posterior = SimpleNamespace(samples=np.arange(20.)[:, None], physical=np.arange(20.)[:, None])
    draws = reporting.ReportingDraws(None, posterior, count, seed, image_count)
    config = SimpleNamespace(exact_image_finder='global-jax', exact_image_grid_pitch_arcsec=.5)
    return state, draws, config


def test_the_recovery_fingerprint_changes_with_every_input():
    from kappaccino.diagnostics import image_recovery as recovery
    state, draws, config = _recovery_inputs()
    base = recovery.recovery_fingerprint(state, draws, config, extra_images=True)
    assert recovery.recovery_fingerprint(state, draws, config, extra_images=True) == base
    changed = {
        'draws': [recovery.recovery_fingerprint(state, _recovery_inputs(seed=8)[1], config, extra_images=True),
                  recovery.recovery_fingerprint(state, _recovery_inputs(count=4)[1], config, extra_images=True),
                  recovery.recovery_fingerprint(state, _recovery_inputs(image_count=5)[1], config, extra_images=True)],
        'extra_images': [recovery.recovery_fingerprint(state, draws, config, extra_images=False)],
        'image_diagnostics': [recovery.recovery_fingerprint(state, draws, SimpleNamespace(**{**vars(config), 'exact_image_grid_pitch_arcsec': .4}),
                                                            extra_images=True)],
    }
    moved = SimpleNamespace(**{**vars(state), 'family_data': [SimpleNamespace(**{**vars(state.family_data[0]), 'x_obs': np.array([1., -1.001])})]})
    changed['catalog'] = [recovery.recovery_fingerprint(moved, draws, config, extra_images=True)]
    for part, prints in changed.items():
        for fingerprint in prints:
            assert [k for k in base if base[k] != fingerprint[k]] == [part]


def test_saved_recovery_tables_are_reused_only_with_the_same_fingerprint(monkeypatch, tmp_path):
    from kappaccino.diagnostics import image_recovery as recovery
    state, draws, config = _recovery_inputs()
    tables = {'image_fit_quality': pd.DataFrame({'image_label': ['1.a', '1.b'], 'x': [1.5, np.nan]}),
              'posterior_extra_images': pd.DataFrame(columns=['family_id', 'x_model_arcsec']),
              'posterior_image_rms': pd.DataFrame({'draw_index': np.arange(4), 'flag': [True, False, True, True]})}
    calls = []
    monkeypatch.setattr(recovery, 'image_recovery_tables', lambda *args, **kwargs: (calls.append(1), {k: v.copy() for k, v in tables.items()})[1])
    lines = []
    first = recovery.image_recovery_with_reuse(tmp_path, state, None, draws, config, extra_images=True, log=lines.append)
    assert calls == [1] and 'computed (no saved image recovery)' in lines[-1]
    again = recovery.image_recovery_with_reuse(tmp_path, state, None, draws, config, extra_images=True, log=lines.append)
    assert calls == [1] and 'reused from' in lines[-1] and list(again) == list(first)
    for name, frame in first.items():
        pd.testing.assert_frame_equal(again[name], frame, check_exact=True)
    recovery.image_recovery_with_reuse(tmp_path, state, None, draws, config, extra_images=False, log=lines.append)
    assert calls == [1, 1] and 'changed since it was saved: extra_images' in lines[-1]
    monkeypatch.setattr(recovery, 'IMAGE_RECOVERY_VERSION', recovery.IMAGE_RECOVERY_VERSION + 1)
    recovery.image_recovery_with_reuse(tmp_path, state, None, draws, config, extra_images=False, log=lines.append)
    assert calls == [1, 1, 1] and 'changed since it was saved: version' in lines[-1]
    with store.batch(tmp_path):                                         # a table lost from results.h5
        record = store.read_record(tmp_path, recovery.RECOVERY_RECORD)
        store.write_record(tmp_path, recovery.RECOVERY_RECORD, {**record, 'tables': record['tables'] + ['lost_table']})
    recovery.image_recovery_with_reuse(tmp_path, state, None, draws, config, extra_images=False, log=lines.append)
    assert calls == [1, 1, 1, 1] and 'table lost_table missing' in lines[-1]
    # A saved table that cannot be decoded is a recomputation, not a stop.
    def damaged(stage, name, **kwargs):
        raise store.DamagedEntry(f'{name}: cannot be decoded')
    monkeypatch.setattr(store, 'read_table', damaged)
    recovery.image_recovery_with_reuse(tmp_path, state, None, draws, config, extra_images=False, log=lines.append)
    assert calls == [1, 1, 1, 1, 1] and 'cannot be read (DamagedEntry' in lines[-1]


# ----------------------------------------------------------------------------- local maps from a reference grid
@pytest.fixture(scope='module')
def toy_lens(tmp_path_factory):
    """The toy cluster with an external shear and an NFW halo (every profile kind the scaling covers) and four draws
    around its start."""
    from conftest import synthetic_cluster_config
    from kappaccino.config import NFWHaloConfig, ShearHaloConfig
    from kappaccino.core.state import build_state
    from kappaccino.likelihood.evaluator import Evaluator
    config = synthetic_cluster_config(tmp_path_factory.mktemp('local'))
    halos = config.model.large_halos + (ShearHaloConfig('shear', .06, 25., .3),
                                        NFWHaloConfig(id='nfw', x_centre=5., y_centre=-3., angle_pos=10., ellipticite=.1,
                                                      log10_m200=13.5, scale_radius_kpc=150., z_lens=.3))
    config = replace(config, model=replace(config.model, large_halos=halos))
    state = build_state(config)
    evaluator = Evaluator(state)
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(5).normal(0, .003, (4, state.n_parameters))
    posterior = SimpleNamespace(samples=samples, physical=np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples))))
    return evaluator, posterior


def _overview(evaluator, posterior, monkeypatch=None, *, pitch=.25):
    draws = reporting.ReportingDraws(evaluator, posterior, 4, 3)
    assert not draws.has_field
    xs = np.arange(-15., 15. + 1e-9, pitch)
    x, y = np.meshgrid(xs, xs[::-1])                                    # a descending y axis, as an image's
    draws.grid_maps('overview', x, y, 9., quantities=('detA',), reference=True, log=_quiet)
    assert len(draws._references) == 1
    return draws, xs


def test_the_smooth_and_near_split_is_exact_at_the_reference_nodes(toy_lens, monkeypatch):
    evaluator, posterior = toy_lens
    monkeypatch.setattr(local_maps, 'REFERENCE_DTYPE', np.float64)
    draws, xs = _overview(evaluator, posterior)
    x, y = np.meshgrid(xs[30:71], xs[40:81])                             # reference nodes, ascending, well inside
    for z in (9., 2.5):                                                 # the overview's redshift and another one
        local = local_maps.LocalLens(draws, draws._references[0], x, y, z, log=_quiet)
        assert len(local.near) >= 1                                     # the members near the box
        for k in range(len(draws)):
            direct = np.asarray(evaluator.total_fields_at(np.stack((x, y), -1).reshape(-1, 2), evaluator.packed_state_for_z(
                evaluator.physical(jnp.asarray(draws.samples[k])), z), chunk=4096)[1])
            smooth = local.smooth_part(k)
            # total = smooth + shear + near at the nodes, to round-off
            np.testing.assert_allclose(local.interpolate(smooth), local_maps._full(smooth).reshape(-1, 2, 2)[
                _node_rows(local, x, y)], atol=1e-13, rtol=0)
            u = local.parametric(k)
            assert np.abs(u - direct).max() < 1e-12
        assert local.direct == []


def _node_rows(local, x, y):
    """Rows of the patch nodes (ravelled) at the points of the grid (x, y)."""
    nodes = {tuple(p): i for i, p in enumerate(local.nodes)}
    return np.array([nodes[(a, b)] for a, b in zip(x.ravel(), y.ravel())])


def test_between_the_nodes_the_near_members_are_exact_and_the_check_falls_back_to_direct(toy_lens, monkeypatch):
    evaluator, posterior = toy_lens
    draws, xs = _overview(evaluator, posterior)
    x, y = np.meshgrid(np.linspace(-2., 6., 96), np.linspace(-1., 7., 96))      # around member g1 at (2, 3)
    local = local_maps.LocalLens(draws, draws._references[0], x, y, 2., log=_quiet)
    profile = np.asarray(evaluator.state.packed_lens_spec.profile_type)
    assert profile[local.near].tolist() and local_maps.SHEAR_PROFILE not in profile[local.near].tolist()
    packed = evaluator.packed_state_for_z(evaluator.physical(jnp.asarray(draws.samples[0])), 2.)
    centres = np.column_stack([np.asarray(packed.center_x), np.asarray(packed.center_y)])
    inside = np.flatnonzero((np.abs(centres[:, 0] - 2.) < 6) & (np.abs(centres[:, 1] - 3.) < 6) & (profile == local_maps.DP_IE_PROFILE))
    assert set(inside) <= set(local.near)
    direct = [np.asarray(evaluator.total_fields_at(local.points, local.packed[k], chunk=4096)[1]) for k in range(4)]
    worst = max(np.abs(local.parametric(k) - direct[k]).max() for k in range(4))
    assert worst < 1e-4 and local.direct == []
    monkeypatch.setattr(local_maps, 'CHECK_TOLERANCE', -1.0)            # every check misses: every draw direct
    strict = local_maps.LocalLens(draws, draws._references[0], x, y, 2., log=_quiet)
    for k in range(4):
        np.testing.assert_array_equal(strict.parametric(k), direct[k])
    assert strict.direct == [0, 1, 2, 3]


def test_the_switch_chooses_between_direct_and_reference_maps(toy_lens, monkeypatch):
    evaluator, posterior = toy_lens
    draws, xs = _overview(evaluator, posterior)
    x, y = np.meshgrid(np.linspace(-2., 6., 40), np.linspace(-1., 7., 40))
    plain = reporting.ReportingDraws(evaluator, posterior, 4, 3).grid_maps('fold', x, y, 2., quantities=('detA',), log=_quiet)
    monkeypatch.setattr(reporting.ReportingDraws, 'maps_from_reference', False)
    lines = []
    off = draws.grid_maps('fold', x, y, 2., quantities=('detA',), from_reference=True, log=lines.append)
    np.testing.assert_array_equal(off['detA'], plain['detA'])
    assert 'reference' not in lines[-1]
    monkeypatch.setattr(reporting.ReportingDraws, 'maps_from_reference', True)
    on = draws.grid_maps('fold', x, y, 2., quantities=('detA',), from_reference=True, log=lines.append)
    assert 'from the reference lens of overview' in lines[-1] and 'near components exact' in lines[-1]
    assert np.abs(on['detA'] - plain['detA']).max() < 1e-4 and not np.array_equal(on['detA'], plain['detA'])
    # Outside the reference (or with no reference at all) the grid is evaluated directly, and says why.
    far = draws.grid_maps('far', x + 40, y, 2., quantities=('detA',), from_reference=True, log=lines.append)
    assert 'no reference grid covers it' in lines[-1]
    np.testing.assert_array_equal(far['detA'], reporting.ReportingDraws(evaluator, posterior, 4, 3).grid_maps(
        'far', x + 40, y, 2., quantities=('detA',), log=_quiet)['detA'])
    fresh = reporting.ReportingDraws(evaluator, posterior, 4, 3)
    fresh.grid_maps('fold', x, y, 2., quantities=('detA',), from_reference=True, log=lines.append)
    assert 'no reference grid evaluated before it' in lines[-1]


def test_a_reference_lost_from_the_cache_is_evaluated_again_with_the_same_maps(toy_lens, monkeypatch, tmp_path):
    evaluator, posterior = toy_lens
    monkeypatch.setattr(reporting.ReportingDraws, 'maps_from_reference', True)
    xs = np.arange(-15., 15. + 1e-9, .25)
    ox, oy = np.meshgrid(xs, xs)
    x, y = np.meshgrid(np.linspace(-2., 6., 40), np.linspace(-1., 7., 40))

    def run(drop=None):
        draws = reporting.ReportingDraws(evaluator, posterior, 4, 3)
        draws.cache_stage = tmp_path
        lines = []
        draws.grid_maps('overview', ox, oy, 9., quantities=('detA',), reference=True, log=lines.append)
        if drop is not None:
            draws._grids.clear()
            drop(draws)
        return draws.grid_maps('fold', x, y, 2., quantities=('detA',), from_reference=True, log=lines.append)['detA'], lines
    first, lines = run()
    assert 'evaluated' in lines[0] and 'from the reference lens' in lines[1]
    again, lines = run()
    assert all(line.endswith('reused from cache.h5') for line in lines)
    np.testing.assert_array_equal(again, first)

    def forget(draws):                                                  # the reference entry is gone from cache.h5
        monkeypatch.setattr(store, 'read_cache_arrays', lambda stage, name: (_ for _ in ()).throw(store.MissingEntry(name)))
    store.cache_path(tmp_path).unlink()
    third, lines = run(drop=forget)
    assert any('evaluated again' in line for line in lines)
    np.testing.assert_array_equal(third, first)


# ----------------------------------------------------------------------------- what the draws key covers
@pytest.mark.parametrize('constant, value', [('PLOT_FIELD_VERSION', 'another-field-code'), ('N_MODES', 2048)])
def test_the_field_code_enters_the_grid_recovery_and_aperture_keys(monkeypatch, tmp_path, constant, value):
    from kappaccino.diagnostics import image_recovery as recovery
    from kappaccino.plotting import aperture_mass_histogram as am
    from kappaccino.plotting import grf
    evaluator, posterior, calls = _counting_run(monkeypatch)
    x, y = _grid()
    state, _, config = _recovery_inputs()
    truth = tmp_path / 'truth.fits'
    truth.write_bytes(b'0')
    members = pd.DataFrame(dict(member_id=['p:1'], catalog_id=['1'], x_centre=[0.], y_centre=[0.], catalog_mag=[20.]))
    aperture_config = SimpleNamespace(outputs=SimpleNamespace(truth=SimpleNamespace(kappa_true_fits=truth), caustic_source_redshift=9.))

    def keys():
        draws = reporting.ReportingDraws(evaluator, posterior, 4, 0)
        draws.cache_stage = tmp_path
        draws.grid_maps('grid', x, y, 4., log=_quiet)
        return (recovery.recovery_fingerprint(state, draws, config, extra_images=True)['draws'],
                am.aperture_fingerprint(aperture_config, draws, members, np.zeros((3, 2)), 1.))
    before = keys()
    n = len(calls)
    assert keys() == before and len(calls) == n                         # same code: the grid is reused
    monkeypatch.setattr(grf, constant, value)
    after = keys()
    assert len(calls) == 2 * n                                          # the grid is evaluated again
    assert after[0] != before[0] and after[1] != before[1]


def test_a_moved_or_brighter_member_changes_the_key_of_the_same_latent_draws(toy_lens):
    from kappaccino.likelihood.evaluator import Evaluator
    evaluator, posterior = toy_lens
    state = evaluator.state
    component = int(np.asarray(state.members.component_indices)[0])
    spec = state.packed_lens_spec

    def key(**changes):
        moved = replace(state, packed_lens_spec=replace(spec, **{k: f(np.array(getattr(spec, k), dtype=float)) for k, f in changes.items()}))
        draws = reporting.ReportingDraws(Evaluator(moved), posterior, 4, 3, image_count=4)
        return draws.key, draws.image_key
    base = reporting.ReportingDraws(evaluator, posterior, 4, 3, image_count=4)
    assert key() == (base.key, base.image_key)                         # a rebuilt evaluator of the same model: the same key

    def bump(array, by):
        array[component] += by
        return array
    for changes in ({'x_center_base': lambda a: bump(a, .01)}, {'luminosity_ratio': lambda a: bump(a, 1e-3)}):
        moved = key(**changes)
        assert moved[0] != base.key and moved[1] != base.image_key, list(changes)


def test_the_near_bound_covers_elliptical_components_and_a_cut_inside_the_core():
    from kappaccino.core.model import DP_IE_PROFILE
    h, distance, e = .25, 5., .5
    xi = distance * np.sqrt((1 - e) / (1 + e))
    round_bound = 5 / 16 * h ** 4 / xi ** 5                            # per unit E = sigma0 a s / (s - a), core << xi
    a, s = 1e-4, 10.
    strength = .5 * local_maps.NEAR_TOLERANCE / round_bound           # half the tolerance without the ellipticity factor
    packed = SimpleNamespace(center_x=np.array([distance, distance]), center_y=np.zeros(2), e1=np.array([e, 0.]), e2=np.zeros(2),
                             Ra=np.array([a, 2.]), Rs=np.array([s, 1.]), sigma0=np.array([strength * (s - a) / (a * s), 1e-9]),
                             nfw_rs=np.zeros(2), alpha_rs=np.zeros(2))
    evaluator = SimpleNamespace(state=SimpleNamespace(packed_lens_spec=SimpleNamespace(profile_type=np.full(2, DP_IE_PROFILE))))
    near = local_maps.near_components(evaluator, [packed], (-1., 0., -1., 1.), (-2., 1., -2., 2.), h)
    assert near.tolist() == [0, 1]                  # 0: elliptical, 1.5 times the tolerance; 1: cut inside its core


def test_the_log_counts_the_points_of_a_point_list_and_the_shape_of_a_grid(monkeypatch):
    evaluator, posterior, _ = _counting_run(monkeypatch)
    x, y = _grid()
    lines = []
    reporting.ReportingDraws(evaluator, posterior, 4, 0).grid_maps('points', x.ravel(), y.ravel(), 4., log=lines.append)
    reporting.ReportingDraws(evaluator, posterior, 4, 0).grid_maps('grid', x, y, 4., log=lines.append)
    assert 'on a 35-point list at z = 4' in lines[0] and 'on a 5x7 grid at z = 4' in lines[-1]
