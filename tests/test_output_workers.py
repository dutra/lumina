"""The output stage's worker threads: the draw map (order, errors, one worker), the setting left out of the run hash and
the resume comparison, and grids and image recovery identical with one and three threads."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from kappaccino.core.parallel import map_draws


# ----------------------------------------------------------------------------- the draw map
def test_results_come_back_in_item_order_whatever_finishes_first():
    def slow_first(i):
        time.sleep(.02 * (5 - i % 5))
        return i * i
    assert map_draws(slow_first, range(10), workers=4) == [i * i for i in range(10)]


def test_one_worker_is_a_plain_loop_in_the_calling_thread():
    seen = []
    assert map_draws(lambda i: seen.append((i, threading.current_thread())) or -i, range(4), workers=1) == [0, -1, -2, -3]
    assert seen == [(i, threading.main_thread()) for i in range(4)]


def test_the_first_item_runs_alone_before_the_others():
    started, lock = [], threading.Lock()
    def work(i):
        with lock:
            started.append(i)
        time.sleep(.01)
        return i
    assert map_draws(work, range(6), workers=3, first=2) == list(range(6))
    assert started[0] == 2 and sorted(started) == list(range(6))


def test_the_earliest_failure_is_raised_and_unstarted_items_are_cancelled():
    ran = []
    def work(i):
        ran.append(i)
        if i in (3, 5):
            time.sleep(.05 if i == 3 else 0.)
            raise ValueError(f"draw {i}")
        time.sleep(.01)
        return i
    with pytest.raises(ValueError, match="draw 3"):
        map_draws(work, range(40), workers=2, first=0)
    assert len(ran) < 40
    with pytest.raises(ValueError, match="draw 3"):
        map_draws(work, range(8), workers=1)


def test_a_map_inside_a_worker_runs_serially_and_workers_must_be_positive():
    inner = []
    def outer(i):
        return map_draws(lambda j: inner.append(threading.current_thread()) or (i, j), range(3), workers=4)
    assert map_draws(outer, range(3), workers=3) == [[(i, j) for j in range(3)] for i in range(3)]
    with pytest.raises(ValueError):
        map_draws(lambda i: i, range(3), workers=0)
    assert map_draws(lambda i: i, [], workers=4) == []


# ----------------------------------------------------------------------------- the setting, the hash and a resume
def test_output_workers_is_left_out_of_the_hash_and_of_the_resume_comparison(synthetic_config, tmp_path):
    from kappaccino import cli
    from kappaccino.config import LensClusterSolverConfig
    from kappaccino.pipeline import resume_payload
    from kappaccino.reporting.draws import reporting_draws
    base = synthetic_config
    assert base.outputs.workers == 4
    eight = replace(base, outputs=replace(base.outputs, workers=8)).validate()
    assert cli.config_hash(eight) == cli.config_hash(base)
    assert resume_payload(eight.to_json_dict()) == resume_payload(base.to_json_dict())
    assert cli.config_hash(replace(base, runtime=replace(base.runtime, seed=1))) != cli.config_hash(base)
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match=r"outputs\.workers"):
            replace(base, outputs=replace(base.outputs, workers=bad)).validate()
    assert LensClusterSolverConfig.from_json_dict(eight.to_json_dict()).outputs.workers == 8
    # The reporting draws evaluate in that many threads.
    posterior = SimpleNamespace(samples=np.zeros((6, 1)), physical=np.zeros((6, 1)))
    assert reporting_draws(eight, None, posterior).workers == 8


# ----------------------------------------------------------------------------- identical products with 1 and 3 threads
@pytest.fixture(scope="module")
def knee_run(tmp_path_factory):
    """The toy cluster with the shared knee mass field and five posterior draws around its start."""
    from conftest import synthetic_cluster_config
    from test_gp_knee import knee_config
    from kappaccino.core.state import build_state
    from kappaccino.likelihood.evaluator import Evaluator
    config = knee_config(synthetic_cluster_config(tmp_path_factory.mktemp("knee")), "all")
    state = build_state(config)
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    samples = np.asarray(state.initial_theta)[None] + np.random.default_rng(3).normal(0, .005, (5, state.n_parameters))
    posterior = SimpleNamespace(samples=samples, physical=np.asarray(jax.vmap(evaluator.physical)(jnp.asarray(samples))))
    return config, state, evaluator, posterior


def test_grid_maps_are_identical_with_one_and_three_workers(knee_run):
    from kappaccino.reporting.draws import ReportingDraws
    config, state, evaluator, posterior = knee_run
    x, y = np.meshgrid(np.linspace(-12, 12, 23), np.linspace(-10, 10, 19))
    x[0, 0] = np.nan
    maps = {}
    for workers in (1, 3):
        draws = ReportingDraws(evaluator, posterior, 4, 11, workers=workers)
        assert draws.has_field
        maps[workers] = draws.grid_maps("grid", x, y, 2.5, log=lambda _: None)
    for name, values in maps[1].items():
        np.testing.assert_array_equal(maps[3][name], values, err_msg=name)
    assert np.isnan(maps[3]["kappa"][:, 0, 0]).all() and np.isfinite(maps[3]["kappa"][:, 1:, 1:]).all()


def test_image_recovery_is_identical_with_one_and_three_workers(knee_run):
    from kappaccino.diagnostics.image_recovery import image_recovery_tables
    from kappaccino.reporting.draws import ReportingDraws
    config, state, evaluator, posterior = knee_run
    tables, logs = {}, []
    for workers in (1, 3):
        draws = ReportingDraws(evaluator, posterior, 3, 11, image_count=5, workers=workers)
        tables[workers] = image_recovery_tables(state, evaluator, draws, config.outputs.image_solver, extra_images=True, log=logs.append)
    assert tables[1].keys() == tables[3].keys() and "posterior_image_residual_draws_lens_only" in tables[1]
    for name, frame in tables[1].items():
        pd.testing.assert_frame_equal(tables[3][name], frame, check_exact=True, obj=name)
    rows = tables[3]["posterior_image_residual_draws"]
    assert rows.draw_index.tolist() == [d for d in range(5) for _ in range(4)]
    assert "3 worker threads" in logs[-1] and "solver time summed over the draws" in logs[-1]


def test_an_interrupt_drops_the_draws_not_yet_started():
    """Ctrl-C in the calling thread stops after the running items: the queued ones are cancelled, not run."""
    import _thread, threading, time
    from kappaccino.core.parallel import map_draws
    ran = []
    def slow(item):
        ran.append(item)
        time.sleep(.2)
        return item
    timer = threading.Timer(.5, _thread.interrupt_main)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            map_draws(slow, range(30), workers=2)
    finally:
        timer.cancel()
    assert len(ran) < 15
