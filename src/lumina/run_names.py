"""Run folders of the ``lumina`` command: names assigned once, and the saved run a command resumes.

A new run of a cluster goes to ``<results>/<name>/<mon><day><letter>_<description>``, e.g.
``results/hera/nov21a_w500s250t7_allblocks``: the local date the folder is created (lowercase three-letter month, day
without a leading zero), the next letter of that cluster and date (``a`` ... ``z``, ``aa``, ``ab``, ...: every run of
the date counts, whatever its description), and ``cli.describe`` of the run.  The name never changes; nothing in it
but the description comes from the configuration.  Names carry no year: on the same date of a later year the letters
go on after those already given.

The command resumes the saved run of the cluster whose fit settings equal its own (:func:`fit_settings`, the
comparison of ``pipeline.resume_payload``), whatever the run is called: the newest such run when there are several
(the one whose ``init/artifacts/config.json`` was written last).  With ``--stage final`` a run matches when its
complete init stage is the one these settings would make and its final stage has not started, whatever the sampling
settings, or when its final stage started with these fit settings.  Without a match the command creates the next
name and says which settings differ from the closest saved run.  A run is a folder of ``<results>/<name>/`` holding
``init/artifacts/config.json``; its settings are ``final/artifacts/config.json`` when the final stage started, else
that file.  A new run's folder gets that file at once, before its model is built.

Looking for the matching run and reserving a new name happen under the cluster's lock (``.names.lock``,
:mod:`lumina.run_lock`), and the chosen run is locked (``run.lock``) before that lock is released, so two identical
commands started together never make two runs: the second finds the first one's run locked and refuses.  Letters are
reserved by creating ``<results>/<name>/.names/<mon><day><letter>`` (an atomic directory creation); a reserved letter
is never given again, even when its run folder is deleted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from . import run_lock

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
RESERVED = ".names"                 # the letters given so far, one empty folder each
SHOWN_DIFFERENCES = 12              # setting lines shown for the closest run
SHOWN_ERROR_LINES = 3               # lines of an error shown for a saved run that cannot be read


class RunRefused(ValueError):
    """The command cannot use the run folder it was given, or has none to continue."""


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


def fit_settings(config) -> dict:
    """The fit settings of a configuration, as a JSON-ready dictionary: what a resume compares
    (``pipeline.resume_payload``), so the run folder, the outputs section, the ``runtime`` fields that change only
    what is printed or drawn and the WBIC settings are left out."""
    from .pipeline import resume_payload
    return json.loads(json.dumps(resume_payload(config.to_json_dict()), default=str))


@dataclass(frozen=True)
class SavedRun:
    """A run folder of a cluster: ``settings`` (:func:`fit_settings` of its saved configuration), or ``error`` (why
    it cannot be read) when they are None; ``created`` is the time its ``init/artifacts/config.json`` was written."""
    path: Path
    created: float
    settings: dict | None
    error: str | None = None


def settings_file(run_dir) -> Path | None:
    """The saved configuration that holds the fit settings of a run folder: ``final/artifacts/config.json`` once the
    final stage started, else ``init/artifacts/config.json``; None when there is no ``init/artifacts/config.json``
    (not a run).

    Raises
    ------
    OSError
        When the folder cannot be read.
    """
    from .io.artifacts import CONFIG_FILE
    run_dir = Path(run_dir)
    init, final = run_dir / "init" / "artifacts" / CONFIG_FILE, run_dir / "final" / "artifacts" / CONFIG_FILE
    if not init.is_file():
        return None
    return final if final.is_file() else init


def _shown(error: BaseException) -> str:
    """An error as text: its type and at most :data:`SHOWN_ERROR_LINES` lines, never cut inside a line."""
    if isinstance(error, OSError) and error.strerror:
        return f"{type(error).__name__}: {error.strerror}"
    lines = str(error).splitlines() or [""]
    more = f" (... {len(lines) - SHOWN_ERROR_LINES} more lines)" if len(lines) > SHOWN_ERROR_LINES else ""
    return f"{type(error).__name__}: {' / '.join(line.strip() for line in lines[:SHOWN_ERROR_LINES])}{more}"


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def load_saved_config(source: Path):
    """The configuration saved in ``source`` (a ``config.json``).

    Raises
    ------
    SavedConfigUnreadable
        When it is missing, cannot be read, is not valid JSON, or is not a configuration this code can load (unknown,
        missing or misnamed fields, a section that is not a mapping, a ``_class`` that is not a configuration class);
        the message says which.  Any other error is a bug and propagates.
    """
    from .config import InvalidSection, LensClusterSolverConfig, MissingField, UnknownField
    source = Path(source)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SavedConfigUnreadable(f"{source} is missing") from None
    except OSError as error:
        raise SavedConfigUnreadable(f"{source} cannot be read ({_shown(error)})") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SavedConfigUnreadable(f"{source} is not valid JSON ({_shown(error)})") from None
    if not isinstance(payload, dict):
        raise SavedConfigUnreadable(f"{source} is not a configuration this code can load (a JSON "
                                    f"{type(payload).__name__}, not an object)")
    from dataclasses import fields
    try:
        config = LensClusterSolverConfig.from_json_dict(payload)
    except (UnknownField, MissingField, InvalidSection) as error:
        raise SavedConfigUnreadable(f"{source} is not a configuration this code can load ({_shown(error)})") from None
    absent = [f.name for f in fields(LensClusterSolverConfig) if f.name not in payload]
    if absent:                          # config.json holds every section (to_json_dict); a section-less one is no run's
        raise SavedConfigUnreadable(f"{source} is not a configuration this code can load (it lacks the section(s) "
                                    f"{', '.join(absent)})")
    return config


class SavedConfigUnreadable(ValueError):
    """A saved ``config.json`` that is missing, cannot be read, is not JSON, or is not a configuration of this code."""


def read_saved_run(run_dir) -> SavedRun | None:
    """The :class:`SavedRun` of ``run_dir``, or None when it is not a run (:func:`settings_file`).  A folder that
    cannot be read, or whose saved configuration cannot be loaded (:func:`load_saved_config`), gives a SavedRun with
    ``error`` and no settings."""
    from .io.artifacts import CONFIG_FILE
    run_dir = Path(run_dir)
    try:
        source = settings_file(run_dir)
    except OSError as error:
        return SavedRun(run_dir, _mtime(run_dir), None, f"the folder cannot be read ({_shown(error)})")
    if source is None:
        return None
    created = _mtime(run_dir / "init" / "artifacts" / CONFIG_FILE)
    try:
        return SavedRun(run_dir, created, fit_settings(load_saved_config(source)))
    except SavedConfigUnreadable as error:
        return SavedRun(run_dir, created, None, str(error))


def saved_runs(cluster_dir) -> list[SavedRun]:
    """Every run of ``cluster_dir``, newest first (ties: by name, last first)."""
    cluster_dir = Path(cluster_dir)
    if not cluster_dir.is_dir():
        return []
    try:
        entries = [p for p in cluster_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except OSError as error:
        raise RunRefused(f"cannot read the results folder {cluster_dir} ({_shown(error)})") from None
    runs = [read_saved_run(p) for p in entries]
    return sorted((r for r in runs if r is not None), key=lambda r: (r.created, r.path.name), reverse=True)


class _Absent:
    """A setting one side of a comparison does not have."""

    def __repr__(self) -> str:
        return "(absent)"


ABSENT = _Absent()


def differences(saved: dict, current: dict) -> list[tuple[str, object, object]]:
    """``(setting, saved value, current value)`` of each leaf setting that differs, by dotted name (:data:`ABSENT`
    for a setting one side does not have)."""
    from .outputs.saved_run import _flatten
    old, new = _flatten(saved), _flatten(current)
    return [(name, old.get(name, ABSENT), new.get(name, ABSENT)) for name in sorted(old.keys() | new.keys())
            if old.get(name, ABSENT) != new.get(name, ABSENT)]


def _value(value) -> str:
    """A setting's value as shown: ``(absent)``, a string in full (paths are never cut), anything else as JSON, cut
    at 60 characters."""
    if value is ABSENT:
        return "(absent)"
    text = json.dumps(value, default=str)
    return text if isinstance(value, str) or len(text) <= 60 else text[:57] + "..."


def difference_lines(saved: dict, current: dict, *, indent: str = "    ") -> list[str]:
    """The lines ``setting: saved -> current`` of :func:`differences`, at most :data:`SHOWN_DIFFERENCES` of them."""
    found = differences(saved, current)
    lines = [f"{indent}{name}: {_value(old)} -> {_value(new)}" for name, old, new in found[:SHOWN_DIFFERENCES]]
    if len(found) > SHOWN_DIFFERENCES:
        lines.append(f"{indent}... and {len(found) - SHOWN_DIFFERENCES} more")
    return lines


def path_prefix_differences(found: list[tuple[str, object, object]]) -> list[str]:
    """The settings of ``found`` (:func:`differences`) that are one path written from two folders: two strings, one
    ending in ``/`` and the other (``catalogs/hera/images.yaml`` against ``hera/images.yaml``)."""
    return [name for name, old, new in found if isinstance(old, str) and isinstance(new, str)
            and (old.endswith("/" + new) or new.endswith("/" + old))]


PATH_PREFIX_HINT = ("differ only by a path prefix: catalog paths are compared as given, relative to the working "
                    "directory, so run the command from the folder the saved run was started from (the repository "
                    "root)")


def explain_no_match(runs: list[SavedRun], current: dict, *, log=print) -> None:
    """Log why none of ``runs`` (newest first) matches the fit settings ``current``: the settings that differ from the
    closest readable run (fewest differing settings; ties: the newest), and each run whose saved configuration cannot
    be read."""
    readable = [r for r in runs if r.settings is not None]
    log(f"[run] none of the {len(runs)} saved run(s) of {runs[0].path.parent} has these fit settings")
    if readable:
        closest = min(readable, key=lambda r: len(differences(r.settings, current)))     # min keeps the first: the newest
        found = differences(closest.settings, current)
        log(f"[run]   closest: {closest.path}, {len(found)} setting(s) differ (saved -> current):")
        for line in difference_lines(closest.settings, current):
            log(f"[run] {line}")
        prefixed = path_prefix_differences(found)
        if prefixed:
            which = "these settings" if len(prefixed) == len(found) else f"{len(prefixed)} of these settings"
            log(f"[run]   {which} {PATH_PREFIX_HINT}")
        log(f"[run]   lumina resume {closest.path} continues that run with its own settings")
    for run in runs:
        if run.settings is None:
            log(f"[run]   {run.path}: skipped, {run.error}")


@dataclass(frozen=True)
class RunChoice:
    """The folder a command runs in: ``path`` (``<results>/<name>/<name>``), whether it continues a saved run
    (``resume``), and why (``reason``).  A real run holds the folder's ``lock`` (:mod:`lumina.run_lock`) until
    :meth:`release`; a dry run holds none, and ``locked_by`` names a live process that holds it."""
    path: Path
    resume: bool
    reason: str
    lock: run_lock.Lock | None = None
    locked_by: run_lock.LockInfo | None = None

    @property
    def name(self) -> str:
        return self.path.name

    def release(self) -> None:
        if self.lock is not None:
            self.lock.release()


def _in(config, run_dir: Path):
    """``config`` writing to ``run_dir``."""
    return replace(config, paths=replace(config.paths, output_dir=str(run_dir.parent), run_name=run_dir.name))


def _start(config, path: Path, *, log) -> run_lock.Lock:
    """Lock the new run folder ``path`` and save ``config``'s settings in it (``init/artifacts/config.json``), so
    that the next command matches it at once, before its model is built."""
    from .io import artifacts
    lock = run_lock.lock_run(path, log=log)
    try:
        artifacts.save_config(path / "init" / "artifacts", _in(config, path))
    except BaseException:
        lock.release()
        raise
    return lock


def _init_matches(run: SavedRun, config) -> bool:
    """Whether ``run`` holds a complete init stage the pipeline would reuse with ``config``
    (``pipeline.saved_init_conflict``)."""
    from .io import artifacts
    from .pipeline import saved_init_conflict
    init = run.path / "init" / "artifacts"
    if run.settings is None or not artifacts.is_init_complete(init):
        return False
    try:
        return saved_init_conflict(_in(config, run.path), init) is None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _init_complete(run_dir: Path) -> bool:
    from .io import artifacts
    return artifacts.is_init_complete(run_dir / "init" / "artifacts")


def _final_started(run_dir: Path) -> bool:
    from .io import artifacts
    return (run_dir / "final" / "artifacts" / artifacts.CONFIG_FILE).is_file()


def _final_stage_match(run: SavedRun, config, current: dict) -> bool:
    """Whether ``--stage final`` can continue ``run``: a complete init stage the pipeline would reuse with ``config``
    (:func:`_init_matches`), and a final stage either not started or started with the fit settings ``current``."""
    return _init_matches(run, config) and (not _final_started(run.path) or run.settings == current)


def _final_started_refusal(run: SavedRun, current: dict) -> str:
    """Why ``--stage final`` does not continue ``run``, whose init stage matches but whose final stage started with
    other fit settings, and what works instead."""
    from .io import artifacts
    lines = "\n".join(difference_lines(run.settings, current))
    complete = artifacts.is_final_complete(run.path / "final" / "artifacts")
    copy = (f"to sample from that init with these settings and keep it, copy the run without its final stage and run "
            f"the final stage there: cp -R {run.path} {run.path.parent}/NEW && rm -rf {run.path.parent}/NEW/final, "
            "then this command with --run-label NEW (no command starts a new final stage from a saved init by itself)")
    if complete:
        advice = f"its final stage is complete and is kept; {copy}"
    else:
        advice = (f"its final stage did not finish: this command with --run-label {run.path.name} samples it again "
                  f"there with these settings, replacing it; or, {copy}")
    return (f"--stage final: {run.path} has the init stage these settings would make, but its final stage already "
            f"started with other fit settings (saved -> current):\n{lines}\n{advice}")


def _label_refusal(config, path: Path, stage: str) -> str | None:
    """Why the pipeline would refuse to run ``stage`` of ``config`` in the existing folder ``path``, None when it
    would not: the comparisons the pipeline makes for that stage (``pipeline.resume_conflicts``)."""
    from .config import InvalidSection, MissingField, UnknownField
    from .io import artifacts
    from .pipeline import resume_conflicts
    if stage == "final" and not artifacts.is_init_complete(path / "init" / "artifacts"):
        return f"--stage final needs a run with a completed init stage; {path} holds none"
    try:
        conflicts = resume_conflicts(_in(config, path), stage)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, UnknownField, MissingField, InvalidSection) as error:
        return f"{path} holds a saved stage that cannot be read ({_shown(error)}); choose another --run-label"
    if not conflicts:
        return None
    lines = "\n".join(f"  {name} stage:\n" + "\n".join(difference_lines(saved, current))
                      for name, saved, current in conflicts)
    return (f"{path} holds a run with other fit settings for --stage {stage} (saved -> current):\n{lines}\n"
            f"choose another --run-label, or continue it with its own settings: lumina resume {path}")


def choose_run(config, description: str, *, run_label: str | None = None, stage: str = "both", dry_run: bool = False,
               day: date | None = None, log=print, wait: float = run_lock.CLUSTER_LOCK_WAIT) -> RunChoice:
    """The run folder of the ``lumina`` command for the validated ``config`` (its ``paths.output_dir`` is
    ``<results>/<name>``), created and locked unless ``dry_run``.

    With ``run_label`` the folder is exactly that name: created when absent, otherwise used when the pipeline would
    continue it for ``stage`` (``pipeline.resume_conflicts``: a complete init stage with the same init signature or
    none, and for the final stage a complete final stage with the same fit settings or none), refused when it would
    not.  Without it, the newest run of the cluster with the same fit settings is continued (several: said, with their
    names); for ``stage`` ``"final"`` a run whose complete init stage matches and whose final stage has not started
    matches too, whatever its sampling settings.  With none, the next name (:func:`reserve`, :func:`next_name` for a
    dry run) is created, after :func:`explain_no_match`, and its settings are saved in it at once.  With
    ``runtime.resume`` false no run is matched (a ``run_label`` folder is used as it is, as the pipeline overwrites
    it).  ``stage`` ``"final"`` needs an existing run: it never creates one.

    The choice is made under the cluster's lock (:func:`lumina.run_lock.acquire_waiting`, at most ``wait``
    seconds), and the chosen folder is locked before that lock is released; the caller releases the folder's lock
    (:meth:`RunChoice.release`).  A dry run takes no lock and writes nothing; it reports a live lock in ``locked_by``.

    Raises
    ------
    RunRefused
        For a ``run_label`` folder the pipeline would not continue, ``stage`` ``"final"`` without a run to continue,
        a ``.names`` that is not a folder, or a results folder that cannot be written.
    lumina.run_lock.RunLocked
        When the chosen run folder is locked by a live process, or the cluster's lock stays held for ``wait`` seconds.
    """
    cluster_dir = Path(config.paths.output_dir)
    if stage == "final" and not cluster_dir.is_dir():
        where = cluster_dir / run_label if run_label is not None else cluster_dir
        raise RunRefused(f"--stage final needs a run with a completed init stage; {where} holds none")
    if dry_run:
        return _choose(config, description, cluster_dir, run_label, stage, day, log, create=False)
    try:
        cluster_dir.mkdir(parents=True, exist_ok=True)
        guard = run_lock.acquire_waiting(cluster_dir / run_lock.CLUSTER_LOCK, what=f"the runs of {cluster_dir}",
                                         wait=wait, log=log)
    except OSError as error:
        raise RunRefused(f"cannot write in the results folder {cluster_dir} ({_shown(error)})") from None
    with guard:
        try:
            return _choose(config, description, cluster_dir, run_label, stage, day, log, create=True)
        except OSError as error:
            raise RunRefused(f"cannot create a run in {cluster_dir} ({_shown(error)})") from None


def _choose(config, description, cluster_dir: Path, run_label, stage, day, log, *, create: bool) -> RunChoice:
    current = fit_settings(config)

    def take(path: Path, resume: bool, reason: str, *, new: bool = False) -> RunChoice:
        if not create:
            return RunChoice(path, resume, reason, locked_by=None if new else run_lock.run_holder(path))
        if new:
            return RunChoice(path, resume, reason, lock=_start(config, path, log=log))
        return RunChoice(path, resume, reason, lock=run_lock.lock_run(path, log=log))

    if run_label is not None:
        path = cluster_dir / run_label
        if not path.is_dir():
            if stage == "final":
                raise RunRefused(f"--stage final needs a run with a completed init stage; {path} holds none")
            if create:
                path.mkdir()
            return take(path, False, "a new run in the folder named by --run-label", new=True)
        if create:              # locked first: what is read below is not being written by another process
            choice = take(path, True, "")
            try:
                refusal = _label_refusal(config, path, stage) if config.runtime.resume or stage == "final" else None
            except BaseException:
                choice.release()
                raise
            if refusal:
                choice.release()
                raise RunRefused(refusal)
        else:
            refusal = _label_refusal(config, path, stage) if config.runtime.resume or stage == "final" else None
            if refusal:
                raise RunRefused(refusal)
            choice = take(path, True, "")
        if not config.runtime.resume:
            reason, resume = "runtime.resume is false: the run starts again in the existing folder", False
        elif settings_file(path) is None:
            reason, resume = "a new run in the folder named by --run-label", False
        elif not _init_complete(path):
            reason, resume = ("its init stage is not complete: the init stage runs again from the start there, with "
                              "these settings"), True
        else:
            reason, resume = f"the pipeline continues its saved stages for --stage {stage}", True
        return replace(choice, resume=resume, reason=reason)
    if not config.runtime.resume:
        runs, matches = [], []
    else:
        runs = saved_runs(cluster_dir)
        if stage == "final":
            matches = [r for r in runs if _final_stage_match(r, config, current)]
        else:
            matches = [r for r in runs if r.settings == current]
    if matches:
        if len(matches) > 1:
            log(f"[run] {len(matches)} saved runs match these fit settings: {', '.join(r.path.name for r in matches)}; "
                f"continuing the newest, {matches[0].path.name}")
        reason = ("its saved fit settings match" if matches[0].settings == current
                  else "its init stage matches and its final stage has not started")
        if not _init_complete(matches[0].path):
            reason += "; its init stage is not complete, so it runs again from the start"
        return take(matches[0].path, True, reason)
    if runs:
        explain_no_match(runs, current, log=log)
    if stage == "final":
        started = [r for r in runs if _init_matches(r, config)]         # matching init, final started otherwise
        if started:
            raise RunRefused(_final_started_refusal(started[0], current))
        raise RunRefused(f"--stage final needs a run with a completed init stage; no saved run of {cluster_dir} has "
                         "one these fit settings would make")
    reason = "no saved run has these fit settings" if config.runtime.resume else "runtime.resume is false"
    if not create:
        return RunChoice(cluster_dir / next_name(cluster_dir, description, day=day), False, reason)
    return take(cluster_dir / reserve(cluster_dir, description, day=day), False, reason, new=True)
