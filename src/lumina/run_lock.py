"""Locks of the results tree: one process at a time works in a run folder.

A lock is a file holding one JSON line: the host name, the process id, the local time it was taken and the command.
It is written whole to a temporary file of the same folder and published with ``os.link``, which fails when the lock
exists, so exactly one process creates it and no process ever sees it empty.  ``run.lock`` in a run folder is held by
``lumina <cluster>``, ``lumina resume`` and ``lumina outputs`` (except ``--render``) for as long as they work in that
folder.  It is removed when the process leaves the ``with`` block or calls :meth:`Lock.release`, also on an exception
or Ctrl-C.

A lock left by a process that died without removing it (SIGTERM, SIGKILL, a crash, a power cut) is stale when it
names this host (its short name, before the first dot, in any case) and its process id no longer runs: the next
command takes it over and says so.  Of several processes that find the same stale lock, only the one holding the
``flock`` of ``<lock>.takeover`` removes it, after reading it again.  A lock from another host cannot be checked (the
results tree may be mirrored between machines by a file-sync tool, so lock files travel), so it counts as held; the
message names the file to delete when that process is known to be gone.  An unreadable or incomplete lock file counts
as held too, and its message says which file to delete.  Locks work per machine: two machines that mirror one results
tree see each other's locks only after the sync delay.
"""
from __future__ import annotations

import fcntl
import json
import os
import secrets
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RUN_LOCK = "run.lock"                   # in a run folder
TAKEOVER_PAUSE = 0.005                  # seconds a process waits when another one is taking over the same stale lock


class RunLocked(ValueError):
    """A lock is held by another process (a usage error: the command refuses, it does not crash)."""


def host() -> str:
    return socket.gethostname()


def same_host(name: str | None) -> bool:
    """Whether the host ``name`` is this machine: the same short name (before the first dot), in any case, so that
    ``mac`` and ``mac.local`` are one host."""
    return name is not None and name.split(".")[0].lower() == host().split(".")[0].lower()


@dataclass(frozen=True)
class LockInfo:
    """What a lock file says: ``host``, ``pid``, ``started`` (local ISO time) and ``command``; ``raw`` is the file's
    bytes (None when it could not be read or parsed, ``problem`` then says why)."""
    path: Path
    host: str | None
    pid: int | None
    started: str | None
    command: str | None
    raw: bytes | None
    problem: str | None = None

    @property
    def stale(self) -> bool:
        """The lock names this host and a process that no longer runs."""
        return same_host(self.host) and self.pid is not None and not _alive(self.pid)

    def holder(self) -> str:
        """``host vega pid 1234 since 2026-11-21T10:02:03 (lumina catalogs/hera)``, or what is wrong with the file."""
        if self.host is None:
            return f"{self.problem} {self.path}"
        command = f" ({self.command})" if self.command else ""
        return f"host {self.host} pid {self.pid} since {self.started}{command}"

    def advice(self) -> str:
        """What to do about the lock: wait, or delete the file when its process is known to be gone."""
        if self.host is None:
            return f"no process can be named from it: if no command is working there, delete {self.path}"
        if not same_host(self.host):
            return (f"another host's lock cannot be checked from {host()}: if that process is gone, delete {self.path}")
        return f"wait for it to end, or if that process is gone, delete {self.path}"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:             # another user's process
        return True
    return True


def read_lock(path) -> LockInfo | None:
    """The :class:`LockInfo` of the lock file ``path``, None when there is none."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        return LockInfo(path, None, None, None, None, None, f"an unreadable lock file ({error.strerror or error})")
    try:
        record = json.loads(raw.decode("utf-8"))
        return LockInfo(path, str(record["host"]), int(record["pid"]), str(record["started"]),
                        record.get("command"), raw)
    except (ValueError, KeyError, TypeError, AttributeError):
        what = "empty" if not raw.strip() else "cut short or not a lock"
        return LockInfo(path, None, None, None, None, None, f"an incomplete lock file ({what})")


class Lock:
    """A lock this process holds (:func:`acquire`); :meth:`release` removes its file, once, if it is still ours."""

    def __init__(self, path: Path, raw: bytes, log=print):
        self.path, self._raw, self._held, self._log = path, raw, True, log

    def release(self) -> None:
        """Remove the lock file if it still holds this lock.  Never raises (it runs in ``finally`` blocks, where an
        error would hide the one being handled): a file that cannot be read or removed is one WARNING line."""
        if not self._held:
            return
        self._held = False
        try:
            if self.path.read_bytes() == self._raw:
                self.path.unlink()
        except FileNotFoundError:
            pass
        except Exception as error:                  # noqa: BLE001 - see the docstring
            try:
                self._log(f"[lock] WARNING could not release the lock {self.path} "
                          f"({type(error).__name__}: {error}); delete it if no command is working there")
            except Exception:                       # noqa: BLE001
                pass

    def __enter__(self) -> "Lock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _content() -> bytes:
    record = {"host": host(), "pid": os.getpid(), "started": datetime.now().isoformat(timespec="microseconds"),
              "command": " ".join(["lumina", *sys.argv[1:]]) if sys.argv else "lumina",
              "token": secrets.token_hex(4)}
    return (json.dumps(record) + "\n").encode("utf-8")


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _take_over(info: LockInfo) -> bool:
    """Remove the stale lock ``info`` if its file still holds exactly that lock; True when it was removed, False when
    another process holds the guard or the lock changed or went in between (the caller then tries to take the lock
    as usual).  The guard is the ``flock`` of ``<lock>.takeover``, which the kernel releases when its holder dies, so a
    guard is never abandoned; the guard file is removed by its holder, and a process that locked a file removed in
    between (another inode than the path's) starts again."""
    guard = info.path.with_name(info.path.name + ".takeover")
    while True:
        descriptor = os.open(guard, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            time.sleep(TAKEOVER_PAUSE)
            return False
        try:
            same = os.fstat(descriptor).st_ino == os.stat(guard).st_ino
        except FileNotFoundError:
            same = False
        if same:
            break
        os.close(descriptor)                     # the guard file was removed in between: lock the new one
    try:
        try:
            if _read_bytes(info.path) != info.raw:
                return False
            info.path.unlink()
        except FileNotFoundError:                # another process took it over: acquire as usual
            return False
        return True
    finally:
        try:
            guard.unlink()                       # while still holding it
        except FileNotFoundError:
            pass
        os.close(descriptor)


def _publish(path: Path, raw: bytes) -> bool:
    """Create the lock file ``path`` holding ``raw``, whole: written to a temporary file of the same folder, then
    linked to ``path`` (which fails when it exists).  False when ``path`` exists."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        try:
            os.write(descriptor, raw)
        finally:
            os.close(descriptor)
        os.link(temporary, path)
        return True
    except FileExistsError:
        return False
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def try_acquire(path, *, log=print) -> tuple[Lock | None, LockInfo | None]:
    """Take the lock ``path``: ``(Lock, None)``, or ``(None, holder)`` when a live process holds it (or the file
    cannot be read).  A stale lock is taken over, with one log line.

    Raises
    ------
    OSError
        When the lock file cannot be created (a folder that cannot be written).
    """
    path = Path(path)
    while True:
        raw = _content()
        if _publish(path, raw):
            return Lock(path, raw, log), None
        info = read_lock(path)
        if info is None:                    # released in between
            continue
        if info.stale:
            if _take_over(info):
                log(f"[lock] took over the stale lock {path} of {info.holder()}: that process no longer runs")
            continue
        return None, info


def acquire(path, *, what: str, log=print) -> Lock:
    """Take the lock ``path`` (:func:`try_acquire`) or refuse.

    Raises
    ------
    RunLocked
        When a live process holds it: ``<what> is in use: locked by <holder>; <advice>``.
    """
    lock, holder = try_acquire(path, log=log)
    if lock is None:
        raise RunLocked(f"{what} is in use: locked by {holder.holder()}; {holder.advice()}")
    return lock


def run_lock_path(run_dir) -> Path:
    return Path(run_dir) / RUN_LOCK


def lock_run(run_dir, *, log=print) -> Lock:
    """Take the lock of the run folder ``run_dir`` (:data:`RUN_LOCK`) or refuse (:class:`RunLocked`)."""
    return acquire(run_lock_path(run_dir), what=f"the run {run_dir}", log=log)


def run_holder(run_dir) -> LockInfo | None:
    """The live holder of the lock of ``run_dir``, None when it is free or stale (a dry run takes nothing)."""
    info = read_lock(run_lock_path(run_dir))
    return None if info is None or info.stale else info
