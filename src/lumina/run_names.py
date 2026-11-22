"""Run folders of the ``lumina`` command: a new folder for every run, named once.

A new run of a cluster goes to ``<results>/<name>/<mon><day><letter>_<description>``, e.g.
``results/hera/nov21a_w500s250t7_allblocks``: the local date the folder is created (lowercase three-letter month, day
without a leading zero), the next letter of that cluster and date (``a`` ... ``z``, ``aa``, ``ab``, ...: every run of
the date counts, whatever its description), and ``cli.describe`` of the run.  The name never changes; nothing in it
but the description comes from the configuration.  Names carry no year: on the same date of a later year the letters
go on after those already given.

Every run of the command is a new folder, so the same command run twice makes two runs.  ``lumina resume RUN_DIR``
(:mod:`lumina.resume`) is the only way to continue an existing one.

Letters are reserved by creating ``<results>/<name>/.names/<mon><day><letter>`` (an atomic directory creation); a
reserved letter is never given again, even when its run folder is deleted.  The new folder is created and locked
(``run.lock``) under the cluster's lock (``.names.lock``, :mod:`lumina.run_lock`).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

from . import run_lock

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
RESERVED = ".names"                 # the letters given so far, one empty folder each


class RunRefused(ValueError):
    """The command cannot create the run folder it was asked for."""


def today() -> date:
    """The local date a new run is named after (tests replace it)."""
    return date.today()


def date_prefix(day: date) -> str:
    """``nov21`` for 21 November."""
    return f"{MONTHS[day.month - 1]}{day.day}"


def letters(index: int) -> str:
    """The letter of the ``index``-th run of a date, from 0: ``a`` ... ``z``, then ``aa``, ``ab``, ..."""
    text, index = "", index + 1
    while index:
        index, rest = divmod(index - 1, 26)
        text = chr(ord("a") + rest) + text
    return text


def letter_index(text: str) -> int:
    """The inverse of :func:`letters`."""
    value = 0
    for character in text:
        value = 26 * value + ord(character) - ord("a") + 1
    return value - 1


def _reserved_dir(cluster_dir: Path) -> Path:
    reserved = cluster_dir / RESERVED
    if reserved.exists() and not reserved.is_dir():
        raise RunRefused(f"{reserved} is not a folder: it holds the letters given to the runs of {cluster_dir}; "
                         "move that file away")
    return reserved


def taken(cluster_dir: Path, prefix: str) -> set[int]:
    """The letter indices of the date ``prefix`` already given in ``cluster_dir``: the entries ``<prefix><letters>``
    of ``.names/`` and the folders named like an assigned name, ``<prefix>`` and one or two letters then ``_`` and a
    description.  A label made by hand (``nov21final``, ``nov21best_fit``) takes no letter."""
    cluster_dir = Path(cluster_dir)
    reserved = _reserved_dir(cluster_dir)
    given = re.compile(rf"^{re.escape(prefix)}([a-z]+)$")
    named = re.compile(rf"^{re.escape(prefix)}([a-z]{{1,2}})_.")
    found = [given.match(p.name) for p in reserved.iterdir()] if reserved.is_dir() else []
    found += [named.match(p.name) for p in cluster_dir.iterdir()] if cluster_dir.is_dir() else []
    return {letter_index(match.group(1)) for match in found if match}


def next_name(cluster_dir, description: str, *, day: date | None = None) -> str:
    """The name the next new run of ``cluster_dir`` would get on ``day`` (default :func:`today`), reserving
    nothing: one letter past the last one given that date."""
    prefix = date_prefix(day or today())
    used = taken(Path(cluster_dir), prefix)
    return f"{prefix}{letters(max(used) + 1 if used else 0)}_{description}"


def reserve(cluster_dir, description: str, *, day: date | None = None) -> str:
    """Create the folder of a new run of ``cluster_dir`` and return its name (:func:`next_name`).  The letter is
    taken by creating ``.names/<prefix><letter>``, which fails when another command took it first; the next letter is
    tried then, and also when a folder of that exact name is already there.

    Raises
    ------
    RunRefused
        When ``.names`` is not a folder.
    OSError
        When the cluster folder cannot be written.
    """
    cluster_dir = Path(cluster_dir)
    prefix = date_prefix(day or today())
    reserved = _reserved_dir(cluster_dir)
    reserved.mkdir(parents=True, exist_ok=True)
    used = taken(cluster_dir, prefix)
    index = max(used) + 1 if used else 0
    while True:
        letter = f"{prefix}{letters(index)}"
        index += 1
        try:
            (reserved / letter).mkdir()
        except FileExistsError:
            continue
        name = f"{letter}_{description}"
        try:
            (cluster_dir / name).mkdir()
        except FileExistsError:
            continue
        return name


def _exists_refusal(path: Path) -> RunRefused:
    return RunRefused(f"{path} already exists: --run-label names a new run.  Continue that run with: "
                      f"lumina resume {path}")


def new_run(cluster_dir, description: str, *, run_label: str | None = None, dry_run: bool = False,
            day: date | None = None, log=print, wait: float = run_lock.CLUSTER_LOCK_WAIT):
    """``(path, lock)``: the folder of a new run of ``cluster_dir``, created and locked (``run.lock``) unless
    ``dry_run`` (``lock`` is then None and nothing is written).  The folder is ``run_label`` when given, else the next
    assigned name (:func:`reserve`; :func:`next_name` for a dry run).  It is created under the cluster's lock
    (:func:`lumina.run_lock.acquire_waiting`, at most ``wait`` seconds); the caller releases the run's lock.

    Raises
    ------
    RunRefused
        When the ``run_label`` folder exists (an existing run is continued with ``lumina resume``), ``.names`` is not
        a folder, or the results folder cannot be written.
    lumina.run_lock.RunLocked
        When the cluster's lock stays held for ``wait`` seconds.
    """
    cluster_dir = Path(cluster_dir)
    if dry_run:
        path = cluster_dir / (run_label or next_name(cluster_dir, description, day=day))
        if run_label is not None and path.exists():
            raise _exists_refusal(path)
        return path, None
    try:
        cluster_dir.mkdir(parents=True, exist_ok=True)
        guard = run_lock.acquire_waiting(cluster_dir / run_lock.CLUSTER_LOCK, what=f"the runs of {cluster_dir}",
                                         wait=wait, log=log)
    except OSError as error:
        raise RunRefused(f"cannot write in the results folder {cluster_dir} ({error.strerror or error})") from None
    with guard:
        try:
            if run_label is None:
                path = cluster_dir / reserve(cluster_dir, description, day=day)
            else:
                path = cluster_dir / run_label
                try:
                    path.mkdir()
                except FileExistsError:
                    raise _exists_refusal(path) from None
            return path, run_lock.lock_run(path, log=log)
        except OSError as error:
            raise RunRefused(f"cannot create a run in {cluster_dir} ({error.strerror or error})") from None
