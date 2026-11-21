"""The products of a run stage in two HDF5 files: ``results.h5`` and ``cache.h5``.

Each stage folder (``<run>/init``, ``<run>/final``) holds

``results.h5``
    Every table, record and array bundle the stage produces: what used to be ``tables/<name>.csv``,
    ``tables/<name>.json``, ``tables/<name>.h5`` and the ``figures/<figure>.json`` record next to each figure.
``cache.h5``
    What only exists to avoid recomputation.  Deleting it never loses or changes a result; it only costs time.

The figures (PDF), FITS maps, plain-text reports, logs and ``artifacts/`` stay separate files.

Every entry is addressed by its stage folder and a name (a stage's ``tables/`` or ``figures/`` folder is accepted for
the stage itself), through the functions below; nothing else opens these files.

Tables
    :func:`write_table`, :func:`read_table`.  A DataFrame comes back with the same columns in the same order and the
    same dtypes (strings stay strings, bools stay bools, floats bit for bit, NaN and None where they were, NaN and NA
    of nullable columns kept apart, empty frames with their columns).  The index is not stored, as with the
    ``to_csv(index=False)`` the tables were written with: a table reads back with a default ``RangeIndex``.  The
    dtypes are the in-memory ones, not what parsing a CSV would give: a string column of numeric-looking ids stays a
    string column.  Anything that cannot come back exactly is refused with :class:`Unstorable` (a ``TypeError``).
Records
    :func:`write_record`, :func:`read_record`: a JSON-serialisable object, read back as ``json.loads`` of the JSON it
    was written as (NumPy scalars and arrays, paths and tuples become numbers, strings and lists; dict keys become
    strings as ``json.dumps`` writes them: ``None`` as ``"null"``, ``True`` as ``"true"``).
Figure records
    :func:`write_figure_record`, :func:`read_figure_record`: the record of one figure, addressed by the figure's path
    (``<stage>/figures/<stem>.pdf``); stored under the figure's stem.
Arrays
    :func:`write_arrays`, :func:`read_arrays`: a bundle of named arrays, as :func:`lumina.io.hdf5.save_arrays`.
Cache
    :func:`write_cache_arrays`, :func:`read_cache_arrays`, :func:`write_cache_record`, :func:`read_cache_record`,
    :func:`write_cache_table`, :func:`read_cache_table`.

Crash safety of ``results.h5``.  The file is never modified in place.  A write copies it to a temporary file in the
same folder (``results.h5.tmp-<host>-<pid>-<token>``), applies the change there, flushes and fsyncs it, renames it over
``results.h5`` (``os.replace``) and fsyncs the folder, so the file on disk is always either the complete old version
or the complete new one, whatever kills the process.  :func:`batch` groups writes into one such commit::

    with store.batch(stage):
        for name, frame in tables.items():
            store.write_table(stage, name, frame)

A write outside a batch is a batch of one.  Inside a batch, reads see the batch's own writes.  An exception that leaves
the ``with`` block of a batch, at any level of nesting, or that leaves a write inside it, fails the whole batch: its
writes are discarded at once (the temporary removed, the lock released), later writes in it raise
:class:`BatchFailed`, and the outermost ``with`` raises :class:`BatchFailed` if the exception was caught inside it, so
that nothing is committed half.  The one exception that does not fail a batch is input the codec refuses
(:class:`Unstorable`, raised before anything changed) caught inside the block.  Keep a batch to writes made in a row: a
product that took minutes to compute should be committed when it is done, not held until the end of the stage.  A
symlinked ``results.h5`` is followed: the file it points to is replaced, and the link stays.

Other processes.  Writers of one file are serialised across processes by an advisory lock on ``<file>.lock``
(``fcntl.flock``); a writer waits for it up to :data:`WRITER_TIMEOUT` seconds (with a log line while it waits), then
raises :class:`StoreBusy`.  Readers of ``results.h5`` take no lock: they open whichever complete version is on disk
(read-only, without HDF5's own file locking), so a lookup never answers "missing" because another process is writing.
Temporary files left by dead writers are removed by the next writer, while it holds the lock (one whose pid is alive
again, reused by another process, once it is :data:`STALE_AGE` old).

Space.  Replacing an entry does not shrink an HDF5 file, so a commit rebuilds the file entry by entry (instead of
copying its bytes) when the space freed by replaced entries passes a quarter of the file and 4 MiB; :func:`compact`
does it at the end of a stage.  Rebuilding is only an optimisation: if any entry fails to copy, the rebuild is given up
(logged) and the commit copies the file's bytes instead, so it never loses an entry.

Disk full.  HDF5 does not report a failed write reliably (h5py only prints it from a destructor) and the next HDF5 call
on that file can crash the process.  So before a commit copies ``results.h5`` and before each write, the store checks
that the folder has room for the copy and the new entry (with a margin of :data:`SPACE_MARGIN` bytes plus 2%), and that
the process's file size limit (``ulimit -f``) allows it; if not, it raises :class:`StoreError` without starting (for
the cache: logs a miss).  A write that fails all the same (the disk filled up between the check and the write) is
detected, its temporary is abandoned without further HDF5 calls on it, and the write raises :class:`StoreError`;
``results.h5`` keeps its old version.  What cannot be guaranteed in that case is the process itself: the HDF5 library
may already be in a broken state and crash later (at the latest when the process exits), and a cache file written in
place may be left damaged (the next run treats it as such).

Damage.  A file that is positively not a valid HDF5 file (no signature, empty, truncated) is moved aside to
``<file>.unreadable-<time>`` by the next writer, which holds the lock; for ``results.h5`` it logs an error naming the
entries that are lost.  Any other failure to open or write ``results.h5`` (permissions, a read-only file, too many open
files, a full disk, an I/O error) raises :class:`StoreError` and leaves the file as it was.  A reader never moves
anything.

Reading.  A name that is not in the file raises :class:`MissingEntry`, a
``FileNotFoundError``: the product was never written.  An entry that is there but cannot be decoded, or a
``results.h5`` that is not a valid HDF5 file, is logged and raises :class:`DamagedEntry`, which is *not* a
``FileNotFoundError`` (nor an ``OSError``), so that callers that carry on when a product is missing stop instead of
producing a product without it.  ``MemoryError``, interrupts and ``OSError`` from the device (an I/O error) are raised
as they are.  ``has_*`` answers whether the name is there, without decoding it.  :func:`entries`, :func:`export` and
the command line list a damaged entry as damaged and carry on with the rest.  The only way an entry is ever dropped is
writing a name whose stored entry cannot even be replaced in place (its header is damaged): the writer reads the entry
from ``results.h5`` a second time, and only if that fails too, it rebuilds the file without that one entry and logs an
ERROR that names it as LOST.

``cache.h5`` is written in place (it can grow to gigabytes), under the same kind of lock (shared for readers).  Every
failure to read or write it that concerns the file is logged and treated as a cache miss; it never stops a run.  Input
the codec refuses (:class:`Unstorable`, a bug in the caller), ``MemoryError`` and interrupts are raised.

Threads.  All access goes through one process-wide lock, held for the whole of a batch: write only from the calling
thread of the output stage, never from the worker threads of :mod:`lumina.core.parallel`.

Inspecting.  ``python -m lumina.io.store RUN_OR_STAGE_DIR`` lists the entries with their shapes;
``--table NAME`` prints one table as CSV, ``--record NAME`` one record as JSON, ``--arrays NAME`` the arrays of one
bundle; ``--export DIR`` writes everything as loose ``.csv``, ``.json`` and ``.h5`` files.  Damaged entries are
listed as such and make the command exit with status 1.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import math
import os
import resource
import secrets
import shutil
import socket
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import h5py
import numpy as np
import pandas as pd

from .hdf5 import STRING, save_arrays

RESULTS_FILE = "results.h5"
CACHE_FILE = "cache.h5"
FORMAT_VERSION = 1

TABLES, RECORDS, ARRAYS, FIGURE_RECORDS = "tables", "records", "arrays", "figure_records"
KINDS = (TABLES, RECORDS, ARRAYS, FIGURE_RECORDS)
EXPORT_SUFFIX = {TABLES: ".csv", RECORDS: ".json", ARRAYS: ".h5", FIGURE_RECORDS: ".json"}

# Repack when the space freed by replaced entries passes both of these.
REPACK_FRACTION = 0.25
REPACK_MIN_BYTES = 4 << 20
FREED_ATTRIBUTE = "freed_bytes"

# Seconds a writer of results.h5 waits for another process's commit; a writer or reader of cache.h5 gives up sooner
# (a miss).  WAIT_LOG_AFTER: seconds of waiting before the first log line (then every WAIT_LOG_EVERY).
WRITER_TIMEOUT = 600.0
CACHE_TIMEOUT = 30.0
WAIT_LOG_AFTER = 2.0
WAIT_LOG_EVERY = 30.0
# Old-style temporaries without an owner (``<file>.repack``), temporaries of another host, and temporaries whose pid is
# alive (a reused pid: no writer of the file can hold one while another has the lock) are removed only when they are
# at least this old (seconds).
STALE_AGE = 3600.0
# Room a write must leave free on the disk and below the file size limit: this many bytes plus SPACE_MARGIN_FRACTION
# of the file.
SPACE_MARGIN = 4 << 20
SPACE_MARGIN_FRACTION = 0.02

LOCK_SUFFIX = ".lock"
TEMP_MARK = ".tmp-"
PENDING_SUFFIX = ".writing"          # an entry being encoded inside an uncommitted file
_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_HOST = socket.gethostname().replace("-", "_").replace("/", "_") or "host"

_LOCK = threading.RLock()
_log: Callable[[str], Any] = lambda message: print(message, file=sys.stderr, flush=True)


class MissingEntry(FileNotFoundError):
    """No entry of that name in the stage's HDF5 file."""


class DamagedEntry(Exception):
    """The entry exists but cannot be decoded (or ``results.h5`` is not a valid HDF5 file, so whether it exists cannot
    be told).  Logged with its name when raised.  Deliberately not a :class:`MissingEntry`, ``FileNotFoundError`` or
    ``OSError``: code that carries on without a missing product must stop here instead."""


class StoreError(OSError):
    """A store file could not be opened or written for a reason other than damage (permissions, a read-only file,
    too many open files, a full disk, an I/O error).  Nothing was changed."""


class StoreBusy(StoreError, BlockingIOError):
    """Another process held the file's writer lock for longer than the timeout."""


class BatchFailed(StoreError):
    """A write or an inner ``with batch()`` of this batch failed earlier and its exception was caught inside the
    batch: the batch's writes were discarded, and nothing more is written in it."""


class Unstorable(TypeError):
    """A value the store cannot write so that it reads back exactly."""


def set_log(log: Callable[[str], Any] | None) -> Callable[[str], Any]:
    """Send the store's messages (files moved aside, damaged entries, waits for a lock) to ``log``; None restores
    the default (standard error).  Returns the previous function."""
    global _log
    previous = _log
    _log = log if log is not None else (lambda message: print(message, file=sys.stderr, flush=True))
    return previous


@contextmanager
def logging_to(log: Callable[[str], Any] | None) -> Iterator[None]:
    """:func:`set_log` for the duration of a ``with`` block."""
    previous = set_log(log)
    try:
        yield
    finally:
        set_log(previous)


def stage_dir(location: str | Path) -> Path:
    """The stage folder of ``location``: the folder itself, or its parent for a stage's ``tables`` or ``figures``."""
    path = Path(location)
    return path.parent if path.name in ("tables", "figures") else path


def results_path(stage: str | Path) -> Path:
    return stage_dir(stage) / RESULTS_FILE


def cache_path(stage: str | Path) -> Path:
    return stage_dir(stage) / CACHE_FILE


def _check_name(name: str) -> str:
    name = str(name)
    if not name or "/" in name or "\x00" in name or name in (".", "..") or name.endswith(PENDING_SUFFIX):
        raise ValueError(f"invalid entry name {name!r}: it must be non-empty, contain no '/' or NUL and not end "
                         f"with {PENDING_SUFFIX!r}")
    return name


# ----------------------------------------------------------------------------------------------------------- locks

def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + LOCK_SUFFIX)


def _holder(lock: Path) -> str:
    """Who holds ``lock``, as well as it can be known: a writer records itself in the lock file when it takes the lock
    and clears it when it lets go; readers of the cache (shared lock) do not record themselves."""
    try:
        text = lock.read_text().strip()
    except OSError:
        text = ""
    if not text:
        return "another process (not recorded: a reader of the cache, or a writer that has not recorded itself yet)"
    if text.startswith(f"pid {os.getpid()} on {socket.gethostname()} "):
        return f"this process itself ({text}; a lock taken twice, a bug in the store)"
    return f"{text} (the last holder recorded in the lock file)"


@contextmanager
def _flock(path: Path, *, exclusive: bool, timeout: float) -> Iterator[None]:
    """Hold the advisory lock ``<path>.lock`` (exclusive for writers, shared for cache readers), waiting up to
    ``timeout`` seconds; :class:`StoreBusy` after that.  Callers already hold ``_LOCK``, so the threads of this process
    never contend for it among themselves."""
    lock = _lock_path(path)
    try:
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as error:
        raise StoreError(error.errno, f"cannot create the lock file {lock} ({error.strerror})") from error
    try:
        start, next_log = time.monotonic(), WAIT_LOG_AFTER
        while True:
            try:
                fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno in (errno.ENOLCK, errno.EOPNOTSUPP, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)):
                    _log(f"[store] WARNING: {lock.parent} does not support file locks ({error.strerror}); writing "
                         "without the lock between processes")
                    break
                if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                    raise StoreError(error.errno, f"cannot lock {lock} ({error.strerror})") from error
            waited = time.monotonic() - start
            if waited >= timeout:
                raise StoreBusy(errno.EAGAIN, f"{path} is busy: its lock ({lock.name}) has been held for more than "
                                              f"{timeout:g} s by {_holder(lock)}; nothing was written")
            if waited >= next_log:
                _log(f"[store] waiting for {path}: locked by {_holder(lock)} for {waited:.0f} s "
                     f"(giving up after {timeout:g} s)")
                next_log += WAIT_LOG_EVERY
            time.sleep(0.02)
        recorded = False
        if exclusive:
            try:
                os.ftruncate(fd, 0)
                os.pwrite(fd, f"pid {os.getpid()} on {socket.gethostname()} since {time.strftime('%H:%M:%S')}\n".encode(), 0)
                recorded = True
            except OSError:
                pass
        try:
            yield
        finally:
            if recorded:
                try:
                    os.ftruncate(fd, 0)              # still holding the lock: the next holder records itself
                except OSError:
                    pass
    finally:
        os.close(fd)


# ----------------------------------------------------------------------------------------------------------- files

def _key(path: Path) -> str:
    return os.path.realpath(path)


def _real(path: Path) -> Path:
    """``path`` with symlinks resolved: a write replaces the file a link points to, never the link."""
    return Path(os.path.realpath(path))


def _device_error(error: BaseException) -> bool:
    """An error that says nothing about the stored bytes: out of memory, or an ``OSError`` with an errno (an I/O
    error, a full disk, too many open files...).  HDF5 reports damaged structures without an errno."""
    return isinstance(error, MemoryError) or (isinstance(error, OSError) and error.errno is not None)


def _no_room(path: Path, needed: int, final_size: int) -> str | None:
    """Why the folder of ``path`` has no room to write ``needed`` more bytes, so that ``path`` (or its temporary)
    grows to ``final_size`` bytes, with the margin; None when it has.  Checks the free space for this user and the
    process's file size limit (``ulimit -f``)."""
    margin = SPACE_MARGIN + int(SPACE_MARGIN_FRACTION * max(final_size, 0))
    try:
        info = os.statvfs(path.parent)
        free = info.f_bavail * info.f_frsize
    except OSError:
        free = None
    if free is not None and free < needed + margin:
        return (f"the disk is (nearly) full: {free / 2**20:.1f} MiB free in {path.parent}, {(needed + margin) / 2**20:.1f} "
                "MiB needed")
    limit = resource.getrlimit(resource.RLIMIT_FSIZE)[0]
    if limit != resource.RLIM_INFINITY and final_size + margin > limit:
        return (f"the file size limit of this process (ulimit -f) is {limit} bytes, and the file would need "
                f"{final_size + margin} bytes")
    return None


def _size(path: Path | None) -> int:
    try:
        return path.stat().st_size if path is not None else 0
    except OSError:
        return 0


class _WriteFailed(Exception):
    """An HDF5 write failed: raised by h5py, or only reported from one of its destructors.  The handle is not touched
    again (the next HDF5 call on it can crash the process)."""


# Handles of files whose writes failed, kept referenced so that their destructors (which call HDF5) never run.
_ABANDONED: list[Any] = []


@contextmanager
def _watching_writes() -> Iterator[list[BaseException]]:
    """Collect the write errors h5py reports only from its destructors (``Exception ignored in ...__dealloc__``) while
    the block runs; the block checks the list before each further HDF5 call.  An ``OSError`` with an errno raised in
    the block, or any exception once a failure was seen, becomes :class:`_WriteFailed`."""
    seen: list[BaseException] = []
    previous = sys.unraisablehook

    def hook(unraisable):
        if isinstance(unraisable.exc_value, (OSError, RuntimeError)) and "h5py" in repr(unraisable.object):
            seen.append(unraisable.exc_value)
        else:
            previous(unraisable)
    sys.unraisablehook = hook
    try:
        yield seen
    except _WriteFailed:
        raise
    except Exception as error:
        if seen or (isinstance(error, OSError) and error.errno is not None):
            raise _WriteFailed(f"{type(seen[0] if seen else error).__name__}: {seen[0] if seen else error}") from error
        raise
    finally:
        sys.unraisablehook = previous
    if seen:
        raise _WriteFailed(f"{type(seen[0]).__name__}: {seen[0]}")


def _check_writes(seen: list[BaseException]) -> None:
    if seen:
        raise _WriteFailed(f"{type(seen[0]).__name__}: {seen[0]}")


def _open_for_reading(path: Path) -> h5py.File:
    """``path`` open read-only without HDF5's file locking (readers rely on the atomic replace instead); with it when
    this process already has the file open with locking, which HDF5 requires to match."""
    try:
        return h5py.File(path, "r", locking=False)
    except OSError as error:
        if "locking flag values don't match" not in str(error):
            raise
        return h5py.File(path, "r")


def _damage(path: Path, error: BaseException) -> str | None:
    """Why ``path`` is positively not a valid HDF5 file (empty, no signature, truncated), or None when the failure to
    open it (``error``) has another cause or cannot be told apart from one."""
    try:
        size = path.stat().st_size
        found = False
        with open(path, "rb") as handle:
            offset = 0
            while offset + len(_SIGNATURE) <= size:          # the superblock sits at 0, 512, 1024, 2048, ...
                handle.seek(offset)
                if handle.read(len(_SIGNATURE)) == _SIGNATURE:
                    found = True
                    break
                offset = 512 if offset == 0 else 2 * offset
    except OSError:
        return None
    if size == 0:
        return "the file is empty"
    if not found:
        return "no HDF5 signature"
    if "truncated file" in str(error):
        return f"truncated ({error})"
    return None


def _names_in(path: Path) -> list[str]:
    """``kind/name`` of every entry an unreadable file still lists, best effort."""
    try:
        with h5py.File(path, "r", locking=False) as file:
            return [f"{kind}/{name}" for kind in KINDS if kind in file for name in file[kind]]
    except Exception:
        return []


def _move_aside(path: Path, reason: str) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    target = path.with_name(f"{path.name}.unreadable-{stamp}")
    count = 1
    while target.exists():
        target = path.with_name(f"{path.name}.unreadable-{stamp}-{count}")
        count += 1
    listed = _names_in(path)
    os.replace(path, target)
    _fsync_dir(path.parent)
    if path.name == CACHE_FILE:
        _log(f"[store] WARNING: {path} could not be opened: it is not a valid HDF5 file ({reason}); moved to "
             f"{target.name}. The stage starts an empty cache; the cached values are recomputed when needed.")
    else:
        lost = ", ".join(listed) if listed else "their names could not be read"
        _log(f"[store] ERROR: {path} could not be opened: it is not a valid HDF5 file ({reason}). It was moved to "
             f"{target.name} and the stage starts an empty results file. Every entry the old file held is LOST unless "
             f"the stage writes it again: {lost}.")
    return target


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _temporary(path: Path) -> Path:
    return path.with_name(f"{path.name}{TEMP_MARK}{_HOST}-{os.getpid()}-{secrets.token_hex(4)}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _active_temporaries() -> set[str]:
    return {str(batch.temp) for batch in _BATCHES.values() if batch.temp is not None}


def _clean_stale(path: Path) -> None:
    """Remove the temporaries a dead writer of ``path`` left behind.  Called with the writer lock held, so no live
    writer of this file can be using them; a temporary of a live process or of another host is kept all the same
    until it is :data:`STALE_AGE` old (the pid of a dead writer can be reused by an unrelated process)."""
    now, active = time.time(), _active_temporaries()
    candidates = list(path.parent.glob(path.name + TEMP_MARK + "*"))
    for temporary in candidates:
        if str(temporary) in active:
            continue
        fields = temporary.name[len(path.name + TEMP_MARK):].rsplit("-", 2)
        try:
            host, pid = fields[0], int(fields[1])
        except (IndexError, ValueError):
            host, pid = None, None
        try:
            age = now - temporary.stat().st_mtime
        except OSError:
            continue
        if host == _HOST and pid is not None:
            if pid != os.getpid() and _pid_alive(pid) and age < STALE_AGE:
                continue
        elif age < STALE_AGE:
            continue
        _remove(temporary)
    repack = path.with_name(path.name + ".repack")           # written by an earlier version of this module
    try:
        if repack.exists() and now - repack.stat().st_mtime >= STALE_AGE:
            _remove(repack)
    except OSError:
        pass


def _remove(temporary: Path) -> None:
    try:
        temporary.unlink()
        _log(f"[store] removed {temporary.name}, left behind by an interrupted write")
    except FileNotFoundError:
        pass
    except OSError as error:
        _log(f"[store] WARNING: could not remove the stale {temporary} ({error})")


def _storage_bytes(node) -> int:
    if isinstance(node, h5py.Dataset):
        return int(node.id.get_storage_size()) + 512
    total = 512
    for child in node.values():
        total += _storage_bytes(child)
    return total


class _DamagedSlot(Exception):
    """The old entry of a name (``name`` None: its whole group) cannot be removed in place; HDF5 cannot unlink an
    object whose header is unreadable."""

    def __init__(self, kind: str, name: str | None, error: BaseException):
        super().__init__(f"{kind}/{name if name is not None else ''}: {type(error).__name__}: {error}")
        self.kind, self.name, self.error = kind, name, error


def _replace_slot(file: h5py.File, kind: str, name: str) -> h5py.Group:
    """The group of ``kind``, with any entry ``name`` removed and its space counted as freed."""
    try:
        group = file.require_group(kind)
        present = name in group
    except Exception as error:                  # noqa: BLE001
        if _device_error(error):
            raise
        raise _DamagedSlot(kind, None, error) from error
    if present:
        try:
            freed = _storage_bytes(group[name])
        except Exception as error:              # noqa: BLE001 - a damaged entry: its size is unknown
            if _device_error(error):
                raise
            freed = 0
        try:
            del group[name]
        except Exception as error:              # noqa: BLE001
            if _device_error(error):
                raise
            raise _DamagedSlot(kind, name, error) from error
        file.attrs[FREED_ATTRIBUTE] = int(file.attrs.get(FREED_ATTRIBUTE, 0)) + freed
    return group


def _needs_repack(freed: int, size: int, force: bool) -> bool:
    return bool(freed) and (force or (freed >= REPACK_MIN_BYTES and freed >= REPACK_FRACTION * size))


class _CopyFailed(Exception):
    """An entry (or a group) of the source could not be copied while rebuilding a file; nothing was dropped."""


def _copy_entries(source_path: Path, target: Path, *, skip: frozenset = frozenset()) -> None:
    """Write a fresh ``target`` holding every entry of ``source_path`` but the ``(kind, name)`` pairs in ``skip`` and
    the ``.writing`` leftovers (the space of replaced entries is not copied).  Any entry that cannot be copied raises
    :class:`_CopyFailed` (``target`` is then incomplete and must be discarded): a rebuild never drops an entry on its
    own.  A failed write or a device error raises :class:`_WriteFailed` (the target's handle is then abandoned, not
    closed); ``MemoryError`` is raised as it is."""
    with _open_for_reading(source_path) as source:
        out = h5py.File(target, "w", locking=False)
        try:
            with _watching_writes() as failed:
                for key, value in source.attrs.items():
                    if key != FREED_ATTRIBUTE:
                        out.attrs[key] = value
                for key in source:
                    _check_writes(failed)
                    try:
                        if key not in KINDS:
                            source.copy(source[key], out, name=key)
                            continue
                        group = out.require_group(key)
                        names = list(source[key])
                    except Exception as error:  # noqa: BLE001
                        if _device_error(error) or failed:
                            error.add_note(f"while copying {key}")
                            raise
                        raise _CopyFailed(f"{key}: {type(error).__name__}: {error}") from error
                    for name in names:
                        _check_writes(failed)
                        if name.endswith(PENDING_SUFFIX) or (key, name) in skip:
                            continue
                        try:
                            source.copy(source[key][name], group, name=name)
                        except Exception as error:  # noqa: BLE001
                            if _device_error(error) or failed:
                                error.add_note(f"while copying {key}/{name}")
                                raise
                            raise _CopyFailed(f"{key}/{name}: {type(error).__name__}: {error}") from error
                    del group
                _check_writes(failed)
                out.close()
        except _WriteFailed:
            _ABANDONED.append(out)
            raise
        except BaseException:
            try:
                out.close()
            except Exception:                   # noqa: BLE001 - the target is discarded anyway
                pass
            raise


def _read_fully(node) -> None:
    """Read every attribute and dataset under ``node`` (raises what reading them raises)."""
    dict(node.attrs)
    if isinstance(node, h5py.Dataset):
        node[()]
        return
    for child in node.values():
        _read_fully(child)


def _confirm_damage(path: Path, kind: str, name: str) -> BaseException | None:
    """A second, independent attempt to read the stored ``kind/name`` from ``path`` (a fresh handle on the committed
    file): the error it fails with when the entry is damaged there, or None when it reads (or is not there).  Device
    errors and ``MemoryError`` are raised."""
    try:
        with _open_for_reading(path) as file:
            if kind not in file or name not in file[kind]:
                return None
            _read_fully(file[kind][name])
    except Exception as error:                  # noqa: BLE001
        if _device_error(error):
            raise
        return error
    return None


def _drop_pending(file: h5py.File) -> None:
    """Remove ``<name>.writing`` leftovers (an earlier version wrote entries in place under that name first)."""
    for kind in KINDS:
        try:
            names = [name for name in file[kind] if name.endswith(PENDING_SUFFIX)] if kind in file else []
            for name in names:
                del file[kind][name]
        except Exception as error:              # noqa: BLE001
            if _device_error(error):
                raise
            _log(f"[store] WARNING: {file.filename}: could not remove leftovers in {kind} ({error})")


# --------------------------------------------------------------------------------------------- results: commits

class _Batch:
    """The writes to one ``results.h5`` that become one atomic commit.  The writer lock is taken and the file copied
    at the first write, so a batch without writes costs nothing."""

    def __init__(self, path: Path, *, force_repack: bool = False):
        self.path, self.force_repack = _real(path), force_repack
        self.file: h5py.File | None = None
        self.temp: Path | None = None
        self.stack = ExitStack()
        self.failure: BaseException | None = None   # the first exception that left a write or a level of the batch
        self.poisoned = False                       # an HDF5 write on the temporary failed: never touch self.file
        self.repacked = False
        self.lost: list[str] = []                   # damaged entries dropped (logged as LOST once committed)

    def writable(self, size: int = 0) -> h5py.File:
        """The new version, open for writing, with room checked for ``size`` more bytes."""
        if self.failure is not None:
            raise BatchFailed(errno.EIO, f"{self.path}: nothing more is written in this batch: it failed earlier "
                                         f"({type(self.failure).__name__}: {self.failure}) and its writes were "
                                         "discarded") from self.failure
        if self.file is None:
            self._prepare(size)
            return self.file
        reason = _no_room(self.temp, size, _size(self.temp) + size)
        if reason:
            raise StoreError(errno.ENOSPC, f"{self.path}: not written, {reason}; the file was left as it was")
        return self.file

    def fail(self, error: BaseException) -> None:
        """Mark the batch failed (the first error is kept) and discard its new version and lock now."""
        if self.failure is None:
            self.failure = error
        if isinstance(error, _WriteFailed) or isinstance(error.__cause__, _WriteFailed):
            self.poisoned = True
        self.abort()

    def _prepare(self, size: int) -> None:
        path = self.path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise StoreError(error.errno, f"cannot create {path.parent} ({error.strerror})") from error
        self.stack.enter_context(_flock(path, exclusive=True, timeout=WRITER_TIMEOUT))
        _clean_stale(path)
        repack = False
        if path.exists():
            if not os.access(path, os.W_OK):
                raise StoreError(errno.EACCES, f"{path} is read-only; nothing was written")
            try:
                with _open_for_reading(path) as source:
                    freed = int(source.attrs.get(FREED_ATTRIBUTE, 0))
                repack = _needs_repack(freed, path.stat().st_size, self.force_repack)
            except OSError as error:
                reason = _damage(path, error)
                if reason is None:
                    raise StoreError(getattr(error, "errno", None) or errno.EIO,
                                     f"{path} could not be opened ({error}); nothing was written and the file was "
                                     "left as it is") from error
                _move_aside(path, reason)
        existing = _size(path) if path.exists() else 0
        reason = _no_room(path, existing + size, existing + size)
        if reason:
            raise StoreError(errno.ENOSPC, f"{path}: not written, {reason}; the file was left as it was")
        self.temp = _temporary(path)
        try:
            if path.exists() and repack:
                self._repack(path)
            if path.exists() and not self.repacked:
                shutil.copyfile(path, self.temp)
            if path.exists():
                shutil.copymode(path, self.temp)
            with _watching_writes() as failed:
                self.file = h5py.File(self.temp, "a" if self.temp.exists() else "w", locking=False)
                self.file.attrs.setdefault("format_version", FORMAT_VERSION)
                _check_writes(failed)
                _drop_pending(self.file)
        except _WriteFailed as error:
            self.poisoned = True
            raise StoreError(errno.EIO, f"{path}: writing the new version {self.temp.name} failed ({error}); nothing "
                                        "was written and the file was left as it is") from error
        except OSError as error:
            raise StoreError(getattr(error, "errno", None) or errno.EIO,
                             f"{path}: could not prepare the new version in {self.temp.name} ({error}); nothing was "
                             "written and the file was left as it is") from error

    def _repack(self, path: Path) -> None:
        """Rebuild ``path`` entry by entry into the temporary (to drop the space of replaced entries).  If any entry
        fails to copy, the rebuild is given up and logged; the caller then copies the bytes, so no entry is lost."""
        try:
            _copy_entries(path, self.temp)
            self.repacked = True
        except _WriteFailed as error:
            raise StoreError(errno.EIO, f"{path}: writing the rebuilt version {self.temp.name} failed ({error}); "
                                        "nothing was written and the file was left as it is") from error
        except Exception as error:              # noqa: BLE001 - any failure: the bytes are copied instead
            try:
                self.temp.unlink()
            except FileNotFoundError:
                pass
            notes = "".join(f" ({note})" for note in getattr(error, "__notes__", ()))
            _log(f"[store] WARNING: {path} was not rebuilt to reclaim the space of replaced entries: "
                 f"{type(error).__name__}: {error}{notes}. Its bytes are copied as they are instead (nothing is dropped; the "
                 "unused space stays until a later rebuild succeeds).")

    def replace_damaged(self, damaged: _DamagedSlot) -> None:
        """Make room for a new entry whose stored namesake cannot be removed in place (see :class:`_DamagedSlot`):
        after a second, independent read of the stored entry in ``results.h5`` fails as well, rebuild the new version
        without that one entry.  Anything else (the stored entry reads after all, its whole group is damaged, another
        entry cannot be copied) raises :class:`StoreError` and drops nothing."""
        kind, name, path = damaged.kind, damaged.name, self.path
        if name is None:
            raise StoreError(errno.EIO, f"{path}: the {kind} group cannot be changed ({damaged}); nothing was written "
                                        "and nothing was dropped")
        second = None
        if not name.endswith(PENDING_SUFFIX):    # a leftover of an interrupted write is never data
            second = _confirm_damage(path, kind, name)
            if second is None:
                raise StoreError(errno.EIO, f"{path}: the old {kind}/{name} could not be replaced in the new version "
                                            f"({damaged.error}), but it reads back from {path.name} on a second "
                                            "attempt, so it is not dropped; nothing was written")
        old = self.temp
        with _watching_writes():
            self.file.close()
        self.file = None
        self.temp = _temporary(path)
        reason = _no_room(self.temp, _size(old), _size(old))
        try:
            if reason:
                raise StoreError(errno.ENOSPC, f"{path}: not written, {reason}; the file was left as it was")
            _copy_entries(old, self.temp, skip=frozenset({(kind, name)}))
            shutil.copymode(old, self.temp)
            with _watching_writes():
                self.file = h5py.File(self.temp, "a", locking=False)
        except _CopyFailed as error:
            raise StoreError(errno.EIO, f"{path}: the stored {kind}/{name} is damaged and could be dropped to write "
                                        f"its new version, but the file also holds an entry that cannot be copied "
                                        f"({error}); nothing was written and nothing was dropped") from error
        except _WriteFailed as error:
            raise StoreError(errno.EIO, f"{path}: writing the rebuilt version failed ({error}); nothing was written "
                                        "and the file was left as it is") from error
        except OSError as error:
            raise StoreError(getattr(error, "errno", None) or errno.EIO,
                             f"{path}: could not rebuild the new version ({error}); nothing was written and the "
                             "file was left as it is") from error
        finally:
            try:
                old.unlink()
            except FileNotFoundError:
                pass
        if second is not None:
            self.lost.append(f"{kind}/{name} (in the new version: {type(damaged.error).__name__}: {damaged.error}; "
                             f"read again from {path.name}: {type(second).__name__}: {second})")

    def commit(self) -> None:
        if self.file is None:
            self.stack.close()
            return
        try:
            with _watching_writes() as failed:
                self.file.flush()
                _check_writes(failed)
                self.file.close()
                _check_writes(failed)
            self.file = None
            _fsync_file(self.temp)
            os.replace(self.temp, self.path)
            self.temp = None
            _fsync_dir(self.path.parent)
        except BaseException as error:
            if self.file is not None:            # the flush or the close failed: the handle is not touched again
                self.poisoned = True
            self.fail(error)
            if isinstance(error, (OSError, _WriteFailed)):
                raise StoreError(getattr(error, "errno", None) or errno.EIO,
                                 f"{self.path}: the new version could not be committed ({error}); the file was left "
                                 "as it was") from error
            raise
        finally:
            self.stack.close()
        for lost in self.lost:
            _log(f"[store] ERROR: {self.path}: a damaged entry is LOST: {lost}. It could not be read in two "
                 "independent attempts and was dropped so that the new entry of that name could be written.")

    def abort(self) -> None:
        """Discard the new version and release the lock.  After a failed write the handle is abandoned, not closed
        (closing it would call HDF5 on a file it failed to write); the temporary is removed all the same."""
        try:
            if self.file is not None:
                if self.poisoned:
                    _ABANDONED.append(self.file)
                else:
                    try:
                        with _watching_writes():
                            self.file.close()
                    except Exception:           # noqa: BLE001 - discarded anyway
                        pass
                self.file = None
            if self.temp is not None:
                try:
                    self.temp.unlink()
                except OSError:
                    pass
                self.temp = None
        finally:
            self.stack.close()


_BATCHES: dict[str, _Batch] = {}


def _clean_refusal(error: BaseException) -> bool:
    """Input the codec refused before anything was changed (see :func:`_put`): it does not fail the batch."""
    return isinstance(error, Unstorable) and getattr(error, "_store_clean", False)


@contextmanager
def _batch(path: Path, *, force_repack: bool = False, write: bool = False) -> Iterator[_Batch]:
    """One level of a transaction on ``path`` (``write``: the implicit level of a single write).  An exception that
    leaves any level fails the whole transaction (:meth:`_Batch.fail`), except a clean refusal leaving a write; the
    outermost level commits only if nothing failed, and raises :class:`BatchFailed` if something failed and the
    exception was caught inside it."""
    with _LOCK:
        key = _key(path)
        active = _BATCHES.get(key)
        if active is not None:                  # nested: part of the enclosing batch
            try:
                yield active
            except BaseException as error:
                if not (write and _clean_refusal(error)):
                    active.fail(error)
                raise
            return
        transaction = _Batch(path, force_repack=force_repack)
        _BATCHES[key] = transaction
        try:
            try:
                yield transaction
            except BaseException as error:
                transaction.fail(error)
                raise
            if transaction.failure is not None:
                failure = transaction.failure
                raise BatchFailed(errno.EIO, f"{path}: nothing was written: a write or an inner batch failed inside "
                                             f"this batch ({type(failure).__name__}: {failure}) and the exception was "
                                             "caught inside it; the batch's writes were discarded and the file was "
                                             "left as it was") from failure
            transaction.commit()
        finally:
            _BATCHES.pop(key, None)


@contextmanager
def batch(stage: str | Path) -> Iterator[None]:
    """Make every write to the stage's ``results.h5`` inside the ``with`` block one atomic commit: the file is copied
    once, at the first write, and replaced once, at the end.  Reads inside the block see its writes.  An exception
    that leaves this block, a nested ``with batch()`` or a write inside it fails the whole batch: its writes are
    discarded at once and nothing is committed, and if the exception was caught inside the block, the outermost
    ``with`` raises :class:`BatchFailed`.  (Input a write refuses with :class:`Unstorable` is the exception: caught,
    it leaves the batch as it was.)  Nested batches of one stage join the outermost.  The process-wide lock is held
    for the whole block, so other threads cannot use the store until it ends."""
    with _batch(results_path(stage)):
        yield


@contextmanager
def _open_results(path: Path) -> Iterator[tuple[h5py.File | None, str | None]]:
    """``(file, None)``: the file open for reading (the batch's own new version inside a batch); ``(None, None)`` when
    it does not exist; ``(None, reason)`` when it is not a valid HDF5 file.  Any other failure raises
    :class:`StoreError`."""
    with _LOCK:
        active = _BATCHES.get(_key(path))
        if active is not None and active.file is not None:
            yield active.file, None
            return
        if not path.is_file():
            yield None, None
            return
        try:
            file = _open_for_reading(path)
        except OSError as error:
            reason = _damage(path, error)
            if reason is None:
                raise StoreError(getattr(error, "errno", None) or errno.EIO,
                                 f"{path} could not be opened for reading ({error})") from error
            yield None, f"{path} is not a valid HDF5 file ({reason})"
            return
        try:
            yield file, None
        finally:
            file.close()


# ------------------------------------------------------------------------------------------------ cache: in place

@contextmanager
def _open_cache(path: Path) -> Iterator[tuple[h5py.File | None, str | None]]:
    """The cache open for reading under the shared lock, as :func:`_open_results`, but every failure (busy, damaged,
    unopenable) is ``(None, reason)``: a miss."""
    with _LOCK, ExitStack() as stack:
        if _key(path) in _BROKEN_CACHES:
            yield None, f"{path}: a write to it failed earlier in this process, which no longer uses it"
            return
        if not path.is_file():
            yield None, None
            return
        try:
            stack.enter_context(_flock(path, exclusive=False, timeout=CACHE_TIMEOUT))
        except StoreBusy as error:
            yield None, str(error)
            return
        except StoreError:
            pass                                 # a read-only folder: read without the lock
        try:
            file = _open_for_reading(path)
        except Exception as error:              # noqa: BLE001
            if isinstance(error, MemoryError):
                raise
            reason = _damage(path, error)
            yield None, (f"{path} is not a valid HDF5 file ({reason})" if reason else f"{path} could not be opened ({error})")
            return
        try:
            yield file, None
        finally:
            file.close()


def _maybe_repack_cache(path: Path, *, force: bool = False) -> bool:
    """Rebuild ``cache.h5`` without the space of replaced entries; called with its writer lock held."""
    if not path.is_file():
        return False
    with _open_for_reading(path) as file:
        freed = int(file.attrs.get(FREED_ATTRIBUTE, 0))
    if not _needs_repack(freed, path.stat().st_size, force):
        return False
    _rebuild_cache(path)
    return True


def _rebuild_cache(path: Path, *, skip: frozenset = frozenset()) -> None:
    """Replace ``cache.h5`` by a copy of its entries but ``skip``; called with its writer lock held.  Any entry that
    cannot be copied gives the rebuild up (:class:`_CopyFailed`) and leaves the file as it was."""
    temporary = _temporary(path)
    reason = _no_room(temporary, _size(path), _size(path))
    if reason:
        raise StoreError(errno.ENOSPC, f"{path}: not rebuilt, {reason}")
    try:
        _copy_entries(path, temporary, skip=skip)
        shutil.copymode(path, temporary)
        _fsync_file(temporary)
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


# Cache files a write failed on in this process: HDF5 may hold them in a broken state, so they are not touched again.
_BROKEN_CACHES: set[str] = set()


def _put_in_place(path: Path, kind: str, name: str, encode: Callable[[h5py.Group, str], None]) -> None:
    """Open the cache and write one entry into it.  A failed write abandons the handle (:class:`_WriteFailed`)."""
    try:
        file = h5py.File(path, "a", locking=False)
    except OSError as error:
        reason = _damage(path, error) if path.exists() else None
        if reason is None:
            raise
        _move_aside(path, reason)
        file = h5py.File(path, "a", locking=False)
    try:
        with _watching_writes() as failed:
            file.attrs.setdefault("format_version", FORMAT_VERSION)
            _check_writes(failed)
            _put(file, kind, name, encode, failed)
            file.flush()
            _check_writes(failed)
            file.close()
    except _WriteFailed:
        _ABANDONED.append(file)
        raise
    except BaseException:
        file.close()
        raise


def _write_cache(path: Path, kind: str, name: str, encode: Callable[[h5py.Group, str], None], size: int = 0) -> None:
    """Write an entry into the cache in place.  Every failure but refused input and ``MemoryError`` is logged and
    skipped."""
    name = _check_name(name)
    path = _real(path)
    with _LOCK:
        try:
            if _key(path) in _BROKEN_CACHES:
                raise StoreError(errno.EIO, "a write to it failed earlier in this process, which no longer uses it")
            path.parent.mkdir(parents=True, exist_ok=True)       # before the lock file is created in it
            with _flock(path, exclusive=True, timeout=CACHE_TIMEOUT):
                _clean_stale(path)
                existing = _size(path) if path.exists() else 0
                reason = _no_room(path, size, existing + size)
                if reason:
                    raise StoreError(errno.ENOSPC, reason)
                try:
                    _put_in_place(path, kind, name, encode)
                except _DamagedSlot as damaged:
                    if damaged.name is None:
                        raise
                    _log(f"[store] WARNING: {path}: the old {damaged} cannot be replaced in place; rebuilding the "
                         f"cache without that entry")
                    _rebuild_cache(path, skip=frozenset({(kind, damaged.name)}))
                    _put_in_place(path, kind, name, encode)
                try:
                    _maybe_repack_cache(path)
                except MemoryError:
                    raise
                except Exception as error:      # noqa: BLE001 - the entry is written; only reclaiming space failed
                    _log(f"[store] WARNING: {path} not repacked ({error})")
        except (Unstorable, MemoryError):
            raise
        except _WriteFailed as error:
            _BROKEN_CACHES.add(_key(path))
            _log(f"[store] WARNING: {path}: writing cached {kind}/{name} failed ({error}); this process no longer uses "
                 "the cache file (written in place, it may be damaged; the next run treats it as such). The value is "
                 "recomputed when needed.")
        except Exception as error:              # noqa: BLE001 - the cache never stops a run
            _log(f"[store] WARNING: {path}: cached {kind}/{name} not written ({type(error).__name__}: {error}); "
                 "it is recomputed when needed")


# ---------------------------------------------------------------------------------------------------- table codec

def _plain_text(value: str) -> bool:
    """A string HDF5 can store as UTF-8 as it is: no NUL, no lone surrogates."""
    if "\x00" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _json_cell(value: Any, column: str) -> Any:
    """A cell of an object column as the JSON value it is stored as: None, bools, numbers (NumPy scalars as their
    Python values), strings, and lists and string-keyed dicts of those, which all read back as they were.  Anything else
    (tuples, which would come back as lists, pd.NA, other objects) is refused rather than changed."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if type(value) is list:
        return [_json_cell(item, column) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_cell(item, column) for key, item in value.items()}
    if value is pd.NA or value is pd.NaT:
        raise Unstorable(f"column {column!r}: pd.NA/pd.NaT in an object column cannot be stored; use NaN or None")
    raise Unstorable(f"column {column!r}: a {type(value).__name__} cell cannot be stored exactly; "
                     "tables hold numbers, bools, strings, None, NaN and lists or string-keyed dicts of them")


_MASKED = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "boolean"}


def _write_column(group: h5py.Group, key: str, series: pd.Series, label: str) -> None:
    dtype = series.dtype
    if isinstance(dtype, pd.StringDtype):
        mask = series.isna().to_numpy()
        texts = [("" if missing else str(v)) for v, missing in zip(series.tolist(), mask)]
        encoded = not all(_plain_text(text) for text in texts)
        if encoded:                              # NUL or lone surrogates: stored as JSON strings, exactly
            texts = [json.dumps(text) for text in texts]
        dataset = group.create_dataset(key, data=np.asarray(texts, dtype=object), dtype=STRING, shape=(len(texts),))
        group.create_dataset(key + "_missing", data=mask)
        dataset.attrs["kind"] = "string"
        dataset.attrs["storage"] = str(dtype.storage)
        dataset.attrs["na_value"] = "nan" if (isinstance(dtype.na_value, float) and math.isnan(dtype.na_value)) else "NA"
        if encoded:
            dataset.attrs["encoding"] = "json"
        return
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and not isinstance(dtype, np.dtype):
        name = str(dtype)
        if name in _MASKED:
            array = series.array
            # The values and the mask as they are: a NaN in a Float column is a value, not a missing entry.
            values = np.array(array._data, dtype=dtype.numpy_dtype, copy=True)
            mask = np.array(array._mask, dtype=bool, copy=True)
            values[mask] = dtype.numpy_dtype.type(0)
            dataset = group.create_dataset(key, data=values)
            group.create_dataset(key + "_missing", data=mask)
            dataset.attrs["kind"] = "masked"
            dataset.attrs["dtype"] = name
            return
        raise Unstorable(f"column {label!r}: dtype {name} cannot be stored")
    if dtype.kind in "biufc":
        dataset = group.create_dataset(key, data=series.to_numpy(), **_compression(len(series)))
        dataset.attrs["kind"] = "numpy"
        return
    if dtype.kind in "mM":
        dataset = group.create_dataset(key, data=series.to_numpy().view("int64"))
        dataset.attrs["kind"] = "datetime"
        dataset.attrs["dtype"] = str(dtype)
        return
    if dtype.kind == "O":
        values = series.tolist()
        if values and all(isinstance(v, str) and _plain_text(v) for v in values):
            dataset = group.create_dataset(key, data=np.asarray(values, dtype=object), dtype=STRING)
            dataset.attrs["kind"] = "object_str"
            return
        encoded = np.asarray([json.dumps(_json_cell(v, label), allow_nan=True) for v in values], dtype=object)
        dataset = group.create_dataset(key, data=encoded, dtype=STRING, shape=(len(encoded),))
        dataset.attrs["kind"] = "object_json"
        return
    raise Unstorable(f"column {label!r}: dtype {dtype} cannot be stored")


_MASKED_ARRAY = {"Float": pd.arrays.FloatingArray, "Int": pd.arrays.IntegerArray, "UInt": pd.arrays.IntegerArray,
                 "boolean": pd.arrays.BooleanArray}


def _read_column(group: h5py.Group, key: str) -> pd.Series:
    dataset = group[key]
    kind = dataset.attrs["kind"]
    if kind == "numpy":
        return pd.Series(dataset[()], copy=False)
    if kind == "string":
        values = dataset.asstr()[()].tolist() if dataset.shape[0] else []
        if dataset.attrs.get("encoding") == "json":
            values = [json.loads(value) for value in values]
        mask = group[key + "_missing"][()]
        na_value = np.nan if dataset.attrs["na_value"] == "nan" else pd.NA
        dtype = pd.StringDtype(storage=str(dataset.attrs["storage"]), na_value=na_value)
        cells = [na_value if missing else value for value, missing in zip(values, mask.tolist())]
        return pd.Series(pd.array(cells, dtype=dtype))
    if kind == "masked":
        values = np.asarray(dataset[()])
        mask = np.asarray(group[key + "_missing"][()], dtype=bool)
        name = str(dataset.attrs["dtype"])
        constructor = next(array for prefix, array in _MASKED_ARRAY.items() if name.startswith(prefix))
        return pd.Series(constructor(values, mask))
    if kind == "datetime":
        return pd.Series(dataset[()].view(str(dataset.attrs["dtype"])))
    if kind == "object_str":
        return pd.Series(np.asarray(dataset.asstr()[()].tolist() if dataset.shape[0] else [], dtype=object), dtype=object)
    if kind == "object_json":
        cells = [json.loads(text) for text in (dataset.asstr()[()].tolist() if dataset.shape[0] else [])]
        array = np.empty(len(cells), dtype=object)
        array[:] = cells
        return pd.Series(array, dtype=object)
    raise ValueError(f"unknown column kind {kind!r}")


def _compression(size: int) -> dict:
    return {"compression": "gzip", "shuffle": True} if size > 1 else {}


def _encode_table(group: h5py.Group, frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise Unstorable(f"a table must be a pandas DataFrame, not {type(frame).__name__}")
    if isinstance(frame.columns, pd.MultiIndex):
        raise Unstorable("tables with MultiIndex columns cannot be stored")
    labels = [label.item() if isinstance(label, np.generic) else label for label in frame.columns]
    for label in labels:
        if not isinstance(label, (str, int, float, bool)) or label is None:
            raise Unstorable(f"column label {label!r}: labels must be strings or numbers")
    group.attrs["columns"] = json.dumps(labels)
    group.attrs["rows"] = int(len(frame))
    for position in range(frame.shape[1]):
        _write_column(group, f"c{position:04d}", frame.iloc[:, position], str(labels[position]))


def _decode_table(group: h5py.Group) -> pd.DataFrame:
    labels = json.loads(group.attrs["columns"])
    rows = int(group.attrs["rows"])
    columns = [_read_column(group, f"c{position:04d}") for position in range(len(labels))]
    frame = pd.concat(columns, axis=1, ignore_index=True) if columns else pd.DataFrame(index=pd.RangeIndex(rows))
    frame.index = pd.RangeIndex(rows)
    frame.columns = pd.Index(labels) if labels else frame.columns
    return frame


# ---------------------------------------------------------------------------------------------------- record codec

def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _key_text(key: Any) -> str:
    """A dict key as ``json.dumps`` writes it (``None`` -> ``"null"``, ``True`` -> ``"true"``, ``1.5`` -> ``"1.5"``,
    NaN -> ``"NaN"``); NumPy scalars as their Python values and paths as strings.  Other keys are refused, as
    ``json.dumps`` refuses them."""
    if isinstance(key, str):
        return key
    if isinstance(key, np.generic):
        key = key.item()
    if key is None or isinstance(key, (bool, int, float)):
        return json.dumps(key)
    if isinstance(key, Path):
        return str(key)
    raise Unstorable(f"record key {key!r}: keys must be strings, numbers, bools or None")


def _plain(value: Any) -> Any:
    """Dict keys as strings and NumPy containers as lists, as :func:`lumina.io.artifacts.write_json` writes them."""
    if isinstance(value, dict):
        return {_key_text(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def encode_record(record: Any, *, default: Callable[[Any], Any] | None = None) -> str:
    """The JSON text a record is stored as.  With ``default`` it is ``json.dumps(record, default=default)`` (as the
    writers that passed ``default=float`` wrote their files); without, NumPy values, paths and non-string keys are
    converted first.  :class:`Unstorable` for what JSON cannot hold."""
    try:
        if default is not None:
            return json.dumps(record, default=default, allow_nan=True)
        return json.dumps(_plain(record), default=_json_default, allow_nan=True)
    except Unstorable:
        raise
    except (TypeError, ValueError) as error:
        raise Unstorable(f"record cannot be stored as JSON: {error}") from error


def _write_text(group: h5py.Group, name: str, text: str) -> None:
    group.create_dataset(name, data=text, dtype=STRING)


def _read_text(dataset: h5py.Dataset) -> str:
    return dataset.asstr()[()]


# ---------------------------------------------------------------------------------------------------- array codec

def _encode_arrays(group: h5py.Group, arrays: dict[str, Any]) -> None:
    for key, value in arrays.items():
        _check_name(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.dtype.kind == "U":
            texts = array.astype(object)
            if not all(_plain_text(text) for text in texts.ravel()):
                raise Unstorable(f"{key}: strings with NUL characters or lone surrogates cannot be saved")
            group.create_dataset(key, data=texts, dtype=STRING)
            continue
        if array.dtype.kind == "O":
            raise Unstorable(f"{key}: object arrays cannot be saved; convert to numbers or strings")
        group.create_dataset(key, data=array, **(_compression(array.size) if array.ndim else {}))


def _decode_arrays(group: h5py.Group) -> dict[str, np.ndarray]:
    out = {}
    for key, dataset in group.items():
        if h5py.check_string_dtype(dataset.dtype) is not None:
            out[key] = np.asarray(dataset.asstr()[()], dtype=str)
        else:
            out[key] = np.asarray(dataset[()])
    return out


# ------------------------------------------------------------------------------------------------- generic access

def _put(file: h5py.File, kind: str, name: str, encode: Callable[[h5py.Group, str], None],
         failed: list[BaseException]) -> None:
    """Encode the entry under a temporary name and only then replace the old one, so that an entry that fails to
    encode (an unsupported column, say) leaves the stored one as it was.  ``failed`` is the list of the enclosing
    :func:`_watching_writes`: after a failed write, no further HDF5 call is made on ``file``."""
    pending = name + PENDING_SUFFIX
    group = _replace_slot(file, kind, pending)
    _check_writes(failed)
    try:
        encode(group, pending)
    except BaseException as error:
        if failed or _device_error(error):
            _ABANDONED.append(group)
            raise
        if pending in group:
            del group[pending]
        if isinstance(error, TypeError):
            refusal = error if isinstance(error, Unstorable) else Unstorable(str(error))
            refusal._store_clean = True          # nothing was changed: it does not fail an enclosing batch
            if refusal is error:
                raise
            raise refusal from error
        raise
    _check_writes(failed)
    _replace_slot(file, kind, name)
    _check_writes(failed)
    group.move(pending, name)


def _put_watched(file: h5py.File, kind: str, name: str, encode: Callable[[h5py.Group, str], None]) -> None:
    with _watching_writes() as failed:
        _put(file, kind, name, encode, failed)


def _write(path: Path, kind: str, name: str, encode: Callable[[h5py.Group, str], None], size: int = 0) -> None:
    """Write one entry into the stage's ``results.h5`` (``size``: about how many bytes it adds, for the room check)."""
    name = _check_name(name)
    with _batch(path, write=True) as transaction:
        file = transaction.writable(size)
        try:
            try:
                _put_watched(file, kind, name, encode)
            except _DamagedSlot as damaged:
                _log(f"[store] WARNING: {path}: the old {damaged} cannot be replaced in place; reading it again from "
                     f"{path.name} to tell whether it is damaged there")
                transaction.replace_damaged(damaged)
                _put_watched(transaction.writable(size), kind, name, encode)
        except _WriteFailed as error:
            transaction.poisoned = True
            raise StoreError(errno.EIO, f"{path}: writing {kind}/{name} failed ({error}); nothing was written and the "
                                        "file was left as it was. The HDF5 library may not recover from a failed "
                                        "write: if the process crashes later, the file still holds its old "
                                        "version.") from error


FOUND, ABSENT, DAMAGED = "found", "absent", "damaged"


def _lookup(path: Path, kind: str, name: str, decode: Callable[[Any], Any] | None, *, cache: bool) -> tuple[str, Any]:
    """``(FOUND, value)``, ``(ABSENT, None)`` or ``(DAMAGED, reason)`` of the entry in the HDF5 file (``decode`` None:
    presence only, the entry itself is not opened).  ``MemoryError`` is raised; so are device errors (an ``OSError``
    with an errno) for ``results.h5``, while for the cache they are a miss like any failure of the file."""
    with (_open_cache(path) if cache else _open_results(path)) as (file, problem):
        if file is None:
            return (DAMAGED, problem) if problem else (ABSENT, None)
        try:
            if kind not in file or name not in file[kind]:
                return ABSENT, None
            return FOUND, (decode(file[kind][name]) if decode is not None else None)
        except Exception as error:              # noqa: BLE001 - a damaged entry
            if isinstance(error, MemoryError) or (not cache and _device_error(error)):
                raise
            return DAMAGED, f"{path}: {kind}/{name} cannot be read ({type(error).__name__}: {error})"


def _get(path: Path, kind: str, name: str, decode: Callable[[Any], Any], missing: str, *, cache: bool = False) -> Any:
    state, value = _lookup(path, kind, name, decode, cache=cache)
    if state == FOUND:
        return value
    if state == DAMAGED and not cache:
        _log(f"[store] ERROR: {value}; raised as damaged (not as missing)")
        raise DamagedEntry(value)
    if state == DAMAGED:
        _log(f"[store] WARNING: {value}; treated as a cache miss")
    raise MissingEntry(missing)


def _has(path: Path, kind: str, name: str, *, cache: bool = False) -> bool:
    """Whether the name is there (decodable or not; the entry is not opened).  For ``results.h5``, a file or group
    whose names cannot be read raises :class:`DamagedEntry`; for the cache it is a miss."""
    state, value = _lookup(path, kind, name, None, cache=cache)
    if state == FOUND:
        return True
    if state == DAMAGED and not cache:
        _log(f"[store] ERROR: {value}; whether it holds {kind}/{name} cannot be told")
        raise DamagedEntry(value)
    if state == DAMAGED:
        _log(f"[store] WARNING: {value}; treated as a cache miss")
    return False


def _table_group(group: h5py.Group, name: str, frame: pd.DataFrame) -> None:
    _encode_table(group.create_group(name), frame)


def _record_text(group: h5py.Group, name: str, text: str) -> None:
    _write_text(group, name, text)


def _arrays_group(group: h5py.Group, name: str, arrays: dict) -> None:
    _encode_arrays(group.create_group(name), arrays)


def _table_bytes(frame: Any) -> int:
    """About how many bytes a table adds to a file (its in-memory size), for the room check."""
    try:
        return int(frame.memory_usage(index=False, deep=True).sum()) + 4096 * (frame.shape[1] + 1)
    except Exception:                           # noqa: BLE001 - not a DataFrame: refused when it is encoded
        return 0


def _bundle(arrays: dict[str, Any] | None, more: dict[str, Any]) -> dict[str, Any]:
    """The arrays of a bundle, with their names checked before any write starts."""
    bundle = {**(arrays or {}), **more}
    for key in bundle:
        _check_name(key)
    return bundle


def _arrays_bytes(bundle: dict[str, Any]) -> int:
    total = 0
    for value in bundle.values():
        try:
            total += int(np.asarray(value).nbytes) + 4096 if value is not None else 0
        except Exception:                       # noqa: BLE001 - refused when it is encoded
            pass
    return total


# ------------------------------------------------------------------------------------------------------- results

def write_table(stage: str | Path, name: str, frame: pd.DataFrame) -> None:
    """Store ``frame`` as the table ``name`` of the stage (replacing one of that name)."""
    _write(results_path(stage), TABLES, name, lambda group, key: _table_group(group, key, frame), _table_bytes(frame))


def read_table(stage: str | Path, name: str) -> pd.DataFrame:
    """The table ``name`` of the stage, exactly as written.

    Raises
    ------
    MissingEntry
        If the stage has no such table.
    DamagedEntry
        If it is there but cannot be decoded, or ``results.h5`` is not a valid HDF5 file (not a
        ``MissingEntry``).  ``MemoryError`` and device errors (``OSError``) are raised as they are.
    """
    stage = stage_dir(stage)
    name = _check_name(name)
    return _get(results_path(stage), TABLES, name, _decode_table, f"{stage}: no table {name!r} in {RESULTS_FILE}")


def has_table(stage: str | Path, name: str) -> bool:
    stage = stage_dir(stage)
    return _has(results_path(stage), TABLES, _check_name(name))


def write_record(stage: str | Path, name: str, record: Any, *, default: Callable[[Any], Any] | None = None) -> None:
    """Store a JSON-serialisable ``record`` as ``name`` (see :func:`encode_record` for ``default``)."""
    text = encode_record(record, default=default)
    _write(results_path(stage), RECORDS, name, lambda group, key: _record_text(group, key, text), len(text) * 4)


def read_record(stage: str | Path, name: str) -> Any:
    """The record ``name`` of the stage (``json.loads`` of its JSON).

    Raises
    ------
    MissingEntry
        If the stage has no such record.
    DamagedEntry
        If it is there but cannot be decoded, or ``results.h5`` is not a valid HDF5 file (not a
        ``MissingEntry``).  ``MemoryError`` and device errors (``OSError``) are raised as they are.
    """
    stage = stage_dir(stage)
    name = _check_name(name)
    return _get(results_path(stage), RECORDS, name, lambda node: json.loads(_read_text(node)),
                f"{stage}: no record {name!r} in {RESULTS_FILE}")


def has_record(stage: str | Path, name: str) -> bool:
    stage = stage_dir(stage)
    return _has(results_path(stage), RECORDS, _check_name(name))


def write_arrays(stage: str | Path, name: str, arrays: dict[str, Any] | None = None, **more: Any) -> None:
    """Store a bundle of arrays as ``name``: one dataset per key, None values skipped, strings as UTF-8, no object
    arrays (as :func:`lumina.io.hdf5.save_arrays`)."""
    bundle = _bundle(arrays, more)
    _write(results_path(stage), ARRAYS, name, lambda group, key: _arrays_group(group, key, bundle), _arrays_bytes(bundle))


def read_arrays(stage: str | Path, name: str) -> dict[str, np.ndarray]:
    """The arrays of bundle ``name`` (as :func:`lumina.io.hdf5.load_arrays`).

    Raises
    ------
    MissingEntry
        If the stage has no such bundle.
    DamagedEntry
        If it is there but cannot be decoded, or ``results.h5`` is not a valid HDF5 file (not a
        ``MissingEntry``).  ``MemoryError`` and device errors (``OSError``) are raised as they are.
    """
    stage = stage_dir(stage)
    name = _check_name(name)
    return _get(results_path(stage), ARRAYS, name, _decode_arrays, f"{stage}: no arrays {name!r} in {RESULTS_FILE}")


def has_arrays(stage: str | Path, name: str) -> bool:
    stage = stage_dir(stage)
    return _has(results_path(stage), ARRAYS, _check_name(name))


def _figure_location(figure: str | Path) -> tuple[Path, str]:
    figure = Path(figure)
    return stage_dir(figure.parent), _check_name(figure.stem)


def write_figure_record(figure: str | Path, record: Any, *, default: Callable[[Any], Any] | None = None) -> None:
    """Store the record of the figure at ``figure`` (``<stage>/figures/<stem>.pdf``) under its stem, in the stage's
    ``results.h5`` (in the figure's own folder's ``results.h5`` when that folder is not a stage's ``figures``)."""
    stage, name = _figure_location(figure)
    text = encode_record(record, default=default)
    _write(results_path(stage), FIGURE_RECORDS, name, lambda group, key: _record_text(group, key, text), len(text) * 4)


def read_figure_record(figure: str | Path) -> Any:
    """The record of the figure at ``figure``.

    Raises
    ------
    MissingEntry
        If the figure has no record.
    DamagedEntry
        If it is there but cannot be decoded, or ``results.h5`` is not a valid HDF5 file (not a
        ``MissingEntry``).  ``MemoryError`` and device errors (``OSError``) are raised as they are.
    """
    stage, name = _figure_location(figure)
    return _get(results_path(stage), FIGURE_RECORDS, name, lambda node: json.loads(_read_text(node)),
                f"{figure}: no figure record in {RESULTS_FILE}")


def has_figure_record(figure: str | Path) -> bool:
    stage, name = _figure_location(figure)
    return _has(results_path(stage), FIGURE_RECORDS, name)


def compact(stage: str | Path) -> None:
    """Rebuild the stage's ``results.h5`` and ``cache.h5`` without the space of replaced entries (when there is any).
    The results file is rebuilt as an atomic commit; a failure (an entry that cannot be copied, say) is logged and
    leaves the files as they were."""
    path = _real(results_path(stage))
    with _LOCK:
        try:
            if path.is_file() and _key(path) not in _BATCHES:
                with _open_for_reading(path) as file:
                    freed = int(file.attrs.get(FREED_ATTRIBUTE, 0))
                if freed:
                    with _batch(path, force_repack=True) as transaction:
                        transaction.writable()
                        if not transaction.repacked:          # given up (logged): a byte copy gains nothing
                            raise _NotRepacked
        except _NotRepacked:
            pass
        except OSError as error:
            _log(f"[store] WARNING: {path} not compacted ({error})")
        cache = _real(cache_path(stage))
        try:
            if cache.is_file() and _key(cache) not in _BROKEN_CACHES:
                with _flock(cache, exclusive=True, timeout=CACHE_TIMEOUT):
                    _maybe_repack_cache(cache, force=True)
        except MemoryError:
            raise
        except Exception as error:              # noqa: BLE001
            _log(f"[store] WARNING: {cache} not compacted ({error})")


class _NotRepacked(Exception):
    pass


# --------------------------------------------------------------------------------------------------------- cache

def write_cache_arrays(stage: str | Path, name: str, arrays: dict[str, Any] | None = None, **more: Any) -> None:
    """Store arrays in the stage's ``cache.h5`` (see :func:`write_arrays`).  A failure is logged, never raised."""
    bundle = _bundle(arrays, more)
    _write_cache(cache_path(stage), ARRAYS, name, lambda group, key: _arrays_group(group, key, bundle),
                 _arrays_bytes(bundle))


def read_cache_arrays(stage: str | Path, name: str) -> dict[str, np.ndarray]:
    """Arrays from the stage's cache.  :class:`MissingEntry` if absent or unreadable (logged)."""
    stage = stage_dir(stage)
    name = _check_name(name)
    return _get(cache_path(stage), ARRAYS, name, _decode_arrays, f"{stage}: no cached arrays {name!r}", cache=True)


def has_cache_arrays(stage: str | Path, name: str) -> bool:
    stage = stage_dir(stage)
    return _has(cache_path(stage), ARRAYS, _check_name(name), cache=True)


def write_cache_record(stage: str | Path, name: str, record: Any, *, default: Callable[[Any], Any] | None = None) -> None:
    text = encode_record(record, default=default)
    _write_cache(cache_path(stage), RECORDS, name, lambda group, key: _record_text(group, key, text), len(text) * 4)


def read_cache_record(stage: str | Path, name: str) -> Any:
    """A record from the stage's cache.  :class:`MissingEntry` if absent or unreadable (logged)."""
    stage = stage_dir(stage)
    name = _check_name(name)
    return _get(cache_path(stage), RECORDS, name, lambda node: json.loads(_read_text(node)),
                f"{stage}: no cached record {name!r}", cache=True)


def write_cache_table(stage: str | Path, name: str, frame: pd.DataFrame) -> None:
    _write_cache(cache_path(stage), TABLES, name, lambda group, key: _table_group(group, key, frame), _table_bytes(frame))


def read_cache_table(stage: str | Path, name: str) -> pd.DataFrame:
    """A table from the stage's cache.  :class:`MissingEntry` if absent or unreadable (logged)."""
    stage = stage_dir(stage)
    name = _check_name(name)
    return _get(cache_path(stage), TABLES, name, _decode_table, f"{stage}: no cached table {name!r}", cache=True)


# ------------------------------------------------------------------------------------------------------ listing

def _describe(kind: str, node) -> str:
    if kind == TABLES:
        return f"{int(node.attrs['rows'])} rows x {len(json.loads(node.attrs['columns']))} columns"
    if kind in (RECORDS, FIGURE_RECORDS):
        value = json.loads(_read_text(node))
        size = f"{len(value)} keys" if isinstance(value, dict) else (f"{len(value)} items" if isinstance(value, list) else type(value).__name__)
        return f"JSON {type(value).__name__}, {size}"
    return ", ".join(f"{key} {tuple(dataset.shape)} {dataset.dtype if h5py.check_string_dtype(dataset.dtype) is None else 'str'}"
                     for key, dataset in node.items())


def entries(stage: str | Path, *, cache: bool = False) -> dict[str, dict[str, str]]:
    """``{kind: {name: source}}`` of every entry of the stage (``cache=True``: of its cache); the source is the file
    name.  A damaged file or group is logged and its entries are left out; damaged entries whose names can still be read are
    listed (reading them raises :class:`DamagedEntry`)."""
    stage = stage_dir(stage)
    path = cache_path(stage) if cache else results_path(stage)
    out: dict[str, dict[str, str]] = {kind: {} for kind in KINDS}
    with (_open_cache(path) if cache else _open_results(path)) as (file, problem):
        if problem:
            _log(f"[store] WARNING: {problem}; its entries are not listed")
        if file is not None:
            for kind in KINDS:
                try:
                    names = list(file[kind]) if kind in file else []
                except MemoryError:
                    raise
                except Exception as error:      # noqa: BLE001 - a damaged group
                    _log(f"[store] WARNING: {path}: the {kind} cannot be listed ({error})")
                    continue
                for name in names:
                    if not name.endswith(PENDING_SUFFIX):
                        out[kind][name] = path.name
    return out


def _shape(stage: Path, kind: str, name: str, source: str, *, cache: bool) -> str:
    path = stage / source
    state, value = _lookup(path, kind, name, lambda node: _describe(kind, node), cache=cache)
    if state == FOUND:
        return value
    raise DamagedEntry(value or f"{path}: {kind}/{name} is no longer there")


def stages_of(location: str | Path) -> list[Path]:
    """The stage folders of a run folder (``init``, ``final`` that exist), or ``[location]`` for a stage folder."""
    location = stage_dir(location)
    stages = [location / name for name in ("init", "final") if (location / name).is_dir()]
    return stages or [location]


def _reader(kind: str, cache: bool):
    if cache:
        return {TABLES: read_cache_table, RECORDS: read_cache_record, ARRAYS: read_cache_arrays}[kind]
    return {TABLES: read_table, RECORDS: read_record, ARRAYS: read_arrays}[kind]


def _export(location: str | Path, directory: str | Path) -> tuple[list[Path], list[str]]:
    directory = Path(directory)
    written, damaged = [], []
    for stage in stages_of(location):
        for cache in (False, True):
            for kind, names in entries(stage, cache=cache).items():
                for name in names:
                    folder = directory / stage.name / ("cache" if cache else ("figures" if kind == FIGURE_RECORDS else "tables"))
                    target = folder / f"{name}{EXPORT_SUFFIX[kind]}"
                    try:
                        if kind == FIGURE_RECORDS:
                            value = read_figure_record(stage / "figures" / f"{name}.pdf")
                        else:
                            value = _reader(kind, cache)(stage, name)
                    except MemoryError:
                        raise
                    except Exception as error:   # noqa: BLE001 - reported, the rest is exported
                        label = f"{stage}{' [cache]' if cache else ''} {kind}/{name}"
                        _log(f"[store] WARNING: {label} not exported: {error}")
                        damaged.append(label)
                        continue
                    folder.mkdir(parents=True, exist_ok=True)
                    if kind == TABLES:
                        value.to_csv(target, index=False)
                    elif kind == ARRAYS:
                        save_arrays(target, **value)
                    else:
                        target.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
                    written.append(target)
    return written, damaged


def export(location: str | Path, directory: str | Path) -> list[Path]:
    """Write every entry of the run or stage at ``location`` as loose files under ``directory/<stage>/``: tables as
    ``tables/<name>.csv``, records as ``tables/<name>.json``, arrays as ``tables/<name>.h5``, figure records as
    ``figures/<stem>.json`` and the cache as ``cache/<name>.*``.  Returns the files written.  An entry that cannot be
    read is logged by name and skipped."""
    return _export(location, directory)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lumina.io.store",
                                     description="List, print or export the results and cache of a run or stage.")
    parser.add_argument("location", metavar="RUN_OR_STAGE_DIR", help="a run folder (results/<cluster>/<run>) or a stage folder (its init or final)")
    show = parser.add_mutually_exclusive_group()
    show.add_argument("--table", metavar="NAME", help="print the table NAME as CSV")
    show.add_argument("--record", metavar="NAME", help="print the record NAME as JSON")
    show.add_argument("--figure-record", metavar="STEM", help="print the record of the figure STEM as JSON")
    show.add_argument("--arrays", metavar="NAME", help="print the arrays of bundle NAME (shape, dtype and values)")
    show.add_argument("--export", metavar="DIR", help="write every entry as loose csv/json/h5 files under DIR")
    parser.add_argument("--cache", action="store_true", help="--table/--record/--arrays read the cache instead of the results")
    parser.add_argument("--stage", choices=("init", "final"), default="final",
                        help="for a run folder, the stage --table/--record/--arrays read (default final)")
    args = parser.parse_args(argv)
    location = Path(args.location)
    if not location.is_dir():
        parser.error(f"not a folder: {location}")
    if args.export:
        written, damaged = _export(location, args.export)
        for path in written:
            print(path)
        for label in damaged:
            print(f"not exported (damaged): {label}", file=sys.stderr)
        return 1 if damaged else 0
    stages = stages_of(location)
    stage = next((s for s in stages if s.name == args.stage), stages[0]) if len(stages) > 1 else stages[0]
    try:
        if args.table:
            reader = read_cache_table if args.cache else read_table
            frame = reader(stage, args.table)
            frame.to_csv(sys.stdout, index=False)
            return 0
        if args.record:
            reader = read_cache_record if args.cache else read_record
            print(json.dumps(reader(stage, args.record), indent=2, allow_nan=True))
            return 0
        if args.figure_record:
            print(json.dumps(read_figure_record(stage / "figures" / f"{args.figure_record}.pdf"), indent=2, allow_nan=True))
            return 0
        if args.arrays:
            reader = read_cache_arrays if args.cache else read_arrays
            with np.printoptions(threshold=50, edgeitems=3):
                for key, value in reader(stage, args.arrays).items():
                    print(f"{key} {tuple(np.shape(value))} {np.asarray(value).dtype}\n{value}")
            return 0
    except (MissingEntry, DamagedEntry) as error:
        print(error, file=sys.stderr)
        return 1
    status = 0
    for stage in stages:
        for cache in (False, True):
            listed = entries(stage, cache=cache)
            if not any(listed.values()):
                continue
            print(f"{stage} [{'cache' if cache else 'results'}]")
            for kind, names in listed.items():
                for name, source in sorted(names.items()):
                    try:
                        shape = _shape(stage, kind, name, source, cache=cache)
                    except MemoryError:
                        raise
                    except Exception as error:   # noqa: BLE001 - listed as damaged, the rest is listed
                        shape, status = f"DAMAGED: {error}", 1
                    print(f"  {kind[:-1] if kind != ARRAYS else kind:<14} {name:<48} {shape}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
