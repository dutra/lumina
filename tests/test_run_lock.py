"""The lock files of the results tree (:mod:`lumina.run_lock`)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

import pytest

from lumina import run_lock


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _write(path, **record):
    path.write_text(json.dumps({"started": "2026-11-21T10:00:00", **record}))


def test_a_lock_is_taken_once_and_released_only_by_its_holder(tmp_path):
    path = tmp_path / "run.lock"
    lock = run_lock.acquire(path, what="the run x")
    info = run_lock.read_lock(path)
    assert (info.host, info.pid, info.stale) == (run_lock.host(), os.getpid(), False)
    with pytest.raises(run_lock.RunLocked, match=rf"the run x is in use: locked by host .* pid {os.getpid()} since "):
        run_lock.acquire(path, what="the run x")
    # Released once; a lock someone else took in between is left alone.
    path.unlink()
    _write(path, host=run_lock.host(), pid=os.getppid())
    lock.release()
    assert path.is_file()
    path.unlink()
    with run_lock.acquire(path, what="the run x"):
        assert path.is_file()
    assert not path.exists()


def test_stale_other_host_and_unreadable_locks(tmp_path):
    path = tmp_path / "run.lock"
    lines = []
    dead = _dead_pid()
    _write(path, host=run_lock.host(), pid=dead)
    assert run_lock.read_lock(path).stale and run_lock.run_holder(tmp_path) is None
    lock = run_lock.acquire(path, what="the run x", log=lines.append)
    assert lines == [f"[lock] took over the stale lock {path} of host {run_lock.host()} pid {dead} since "
                     f"2026-11-21T10:00:00: that process no longer runs"]
    assert run_lock.read_lock(path).pid == os.getpid()
    lock.release()
    _write(path, host="vega.example", pid=dead)          # another host: never checked, never taken over
    with pytest.raises(run_lock.RunLocked, match=rf"another host's lock cannot be checked from .*: if that process is "
                                                  rf"gone, delete {path}"):
        run_lock.acquire(path, what="the run x")
    assert run_lock.run_holder(tmp_path).host == "vega.example"
    path.write_text("")                                  # cannot come from this code: a lock is published whole
    with pytest.raises(run_lock.RunLocked, match=rf"an incomplete lock file \(empty\) {path}; no process can be "
                                                  rf"named from it: if no command is working there, delete {path}"):
        run_lock.acquire(path, what="the run x")
    path.write_text('{"host": "x", "pi')
    with pytest.raises(run_lock.RunLocked, match=r"an incomplete lock file \(cut short or not a lock\)"):
        run_lock.acquire(path, what="the run x")


def test_the_short_host_name_decides_whether_a_lock_is_from_this_machine(tmp_path, monkeypatch):
    path = tmp_path / "run.lock"
    dead = _dead_pid()
    monkeypatch.setattr(run_lock, "host", lambda: "FC-Mac")
    for name in ("fc-mac.local", "FC-MAC", "fc-mac.lan.example"):
        _write(path, host=name, pid=dead)
        assert run_lock.read_lock(path).stale, name
        run_lock.acquire(path, what="x", log=lambda _: None).release()
    _write(path, host="vega.local", pid=dead)
    assert not run_lock.read_lock(path).stale


def test_a_lock_is_published_whole_and_release_never_raises(tmp_path):
    path = tmp_path / "run.lock"
    lock = run_lock.acquire(path, what="x")
    assert json.loads(path.read_text())["pid"] == os.getpid()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.lock"]          # no temporary file left
    path.unlink()
    path.mkdir()                                         # replaced by a folder: release warns, never raises
    lines = []
    lock._log = lines.append
    lock.release()
    assert len(lines) == 1 and lines[0].startswith(f"[lock] WARNING could not release the lock {path}")


def test_a_takeover_survives_what_others_do_underneath(tmp_path, monkeypatch):
    """The stale lock removed by another process, the guard file removed, or a fresh lock taken in between: no
    exception, and never two holders."""
    path = tmp_path / "run.lock"
    guard = tmp_path / "run.lock.takeover"
    dead = _dead_pid()
    real_read, real_flock = run_lock._read_bytes, run_lock.fcntl.flock

    def stale():
        _write(path, host=run_lock.host(), pid=dead)

    # The stale lock goes (another process took it over and released it) just before it is read again.
    stale()
    monkeypatch.setattr(run_lock, "_read_bytes", lambda p: (p.unlink(), real_read(p))[1])
    lock, holder = run_lock.try_acquire(path, log=lambda _: None)
    assert lock is not None and holder is None
    lock.release()
    # The guard file is removed underneath its holder: its removal in finally does not fail.
    stale()
    monkeypatch.setattr(run_lock, "_read_bytes", lambda p: (guard.unlink(), real_read(p))[1])
    lock, _ = run_lock.try_acquire(path, log=lambda _: None)
    assert lock is not None
    lock.release()
    monkeypatch.setattr(run_lock, "_read_bytes", real_read)
    # The guard file is removed between its open and its flock: the process locks the new guard instead.
    calls = []

    def flock(descriptor, operation):
        real_flock(descriptor, operation)
        if not calls:
            calls.append(1)
            guard.unlink()

    stale()
    monkeypatch.setattr(run_lock.fcntl, "flock", flock)
    lock, _ = run_lock.try_acquire(path, log=lambda _: None)
    assert lock is not None and calls
    lock.release()
    monkeypatch.setattr(run_lock.fcntl, "flock", real_flock)
    # A live lock replaces the stale one before it is read again: it is left alone and reported.
    stale()
    monkeypatch.setattr(run_lock, "_read_bytes",
                        lambda p: (_write(p, host=run_lock.host(), pid=os.getppid()), real_read(p))[1])
    lock, holder = run_lock.try_acquire(path, log=lambda _: None)
    assert lock is None and holder.pid == os.getppid() and path.is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.lock"]


def test_one_of_many_takes_over_a_stale_lock(tmp_path):
    path = tmp_path / "run.lock"
    dead = _dead_pid()
    for round in range(20):
        _write(path, host=run_lock.host(), pid=dead)
        if round % 2:
            (tmp_path / "run.lock.takeover").write_text("")      # left by a process killed while it held the guard
        barrier = threading.Barrier(8)
        got = []

        def contender():
            barrier.wait()
            lock, holder = run_lock.try_acquire(path, log=lambda _: None)
            got.append(lock)

        threads = [threading.Thread(target=contender) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sum(lock is not None for lock in got) == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run.lock"]
        next(lock for lock in got if lock is not None).release()
