"""``lumina resume RUN_DIR``: continue a run folder with its own saved configuration.

This is the only way back into an existing run folder: the run command always starts a new one.  Run it from the
folder the run was started from (usually the repository root), as ``lumina outputs``::

    lumina resume results/hera/nov21a_w500s250t7_allblocks
    lumina resume results/hera/nov21a_w500s250t7_allblocks --stage final
    lumina resume RUN --set outputs.rgb.enabled=false --draws 64
    lumina resume RUN --remap /home/me/burkelenscluster=$PWD
    lumina resume RUN --dry-run

The settings are the run's saved configuration (``final/artifacts/config.json`` once the final stage started, else
``init/artifacts/config.json``), so a run continues the same way whatever the defaults or the run files say now.
Whatever stage is unfinished is run:

- init complete, final never started: the final stage runs from the saved chain start;
- interrupted during sampling (the final stage saved its configuration but no posterior): the final stage samples
  again from the saved chain start (sampling has no checkpoint), with the final stage's saved settings;
- interrupted during the init stage: the init stage runs again from the start;
- finished: what a resume reruns, the posterior is reloaded and the outputs stage runs again (WBIC is added when the
  saved settings ask for it and it is missing).

``--stage init`` runs (or reloads) the init stage only; ``--stage final`` needs a completed init stage.

``--set`` changes only the ``outputs`` section (``outputs.FIELD`` or ``outputs.MODULE.FIELD``), checked as
``lumina outputs --set`` checks it, and ``--draws N`` sets ``outputs.draws`` and ``outputs.image_draws``; any fit
setting is refused, since the saved fit would no longer describe the run.  Unlike ``lumina outputs``, a resumed final
stage saves the outputs settings it used into ``final/artifacts/config.json``.  ``--remap OLD=NEW`` relocates the
saved paths of a run made on another machine (as ``lumina outputs --remap``) for this command only: it is never
saved.  Every file the run writes (``config.json``, ``init_signature.json``) holds the paths as first saved, and the
saved stages are compared in those terms, so the folder stays the same run with or without the remap.  The run
folder may have been moved or renamed: the run writes where it is now.

The run folder is locked (``run.lock``, :mod:`lumina.run_lock`) while the command works in it: a folder another
process is working in is refused (exit 2), with that process's host, pid and start time.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import replace
from pathlib import Path

SHOWN_ERROR_LINES = 3               # lines of an error shown for a saved configuration that cannot be read


def parser() -> argparse.ArgumentParser:
    from .outputs.command import _positive_int, _remap
    p = argparse.ArgumentParser(prog="lumina resume", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("run", type=Path, help="the run folder, results/<name>/<run>")
    p.add_argument("--stage", choices=("init", "final", "both"), default="both",
                   help="the stages to run (default both, each resumed when complete)")
    p.add_argument("--remap", type=_remap, action="append", default=[], metavar="OLD=NEW",
                   help="replace the absolute prefix OLD of every saved path (catalogs, imaging, truth maps) by NEW; "
                        "repeatable")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="outputs.FIELD=VALUE",
                   help="override a field of the outputs section, e.g. outputs.workers=8; repeatable")
    p.add_argument("--draws", type=_positive_int, metavar="N",
                   help="reporting draws (outputs.draws and outputs.image_draws).  Default: the run's")
    p.add_argument("--dry-run", action="store_true", help="print what would run and the settings; write nothing")
    return p


def next_step(run_dir, finished: bool) -> str:
    """The line a command prints when it leaves the run folder ``run_dir`` (whose name cannot be derived from the run
    command): how to continue it when it stopped before the end (``finished`` false), else what to run next given the
    stages it now holds."""
    from .io import artifacts
    run = shlex.quote(str(run_dir))
    if not finished:
        return f"[run] {run_dir} stopped before the end; continue it with: lumina resume {run}"
    if artifacts.is_final_complete(Path(run_dir) / "final" / "artifacts"):
        return f"[run] finished {run_dir}; rerun its outputs with: lumina outputs {run}"
    return f"[run] init stage done in {run_dir}; run its final stage with: lumina resume {run} --stage final"


def run_and_report(fit, run_dir, *, prog: str) -> int:
    """Call ``fit()`` (the pipeline in the locked run folder ``run_dir``), print the elapsed time and the
    :func:`next_step` line, and return the exit code: 0; 2 for a saved stage the run cannot continue
    (``pipeline.SavedStageMismatch``: one line, no hint, since resuming again would fail the same way); 130 after
    Ctrl-C.  Any other error is raised after the hint."""
    from .pipeline import SavedStageMismatch
    start = time.monotonic()

    def done(finished):
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s", flush=True)
        if finished is not None:
            print(next_step(run_dir, finished), flush=True)
    try:
        fit()
    except SavedStageMismatch as error:
        done(None)
        print(f"{prog}: {run_dir}: {error}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        done(False)
        return 130
    except BaseException:
        done(False)
        raise
    done(True)
    return 0


def settings_file(run_dir) -> Path | None:
    """The saved configuration a resume of ``run_dir`` runs with: ``final/artifacts/config.json`` once the final
    stage started, else ``init/artifacts/config.json``; None when there is no ``init/artifacts/config.json`` (not a
    run).

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


class SavedConfigUnreadable(ValueError):
    """A saved ``config.json`` that is missing, cannot be read, is not JSON, or is not a configuration of this code."""


def _shown(error: BaseException) -> str:
    """An error as text: its type and at most :data:`SHOWN_ERROR_LINES` lines, never cut inside a line."""
    if isinstance(error, OSError) and error.strerror:
        return f"{type(error).__name__}: {error.strerror}"
    lines = str(error).splitlines() or [""]
    more = f" (... {len(lines) - SHOWN_ERROR_LINES} more lines)" if len(lines) > SHOWN_ERROR_LINES else ""
    return f"{type(error).__name__}: {' / '.join(line.strip() for line in lines[:SHOWN_ERROR_LINES])}{more}"


def load_saved_config(source: Path):
    """The configuration saved in ``source`` (a ``config.json``).

    Raises
    ------
    SavedConfigUnreadable
        When it is missing, cannot be read, is not valid JSON, or is not a configuration this code can load (unknown,
        missing or misnamed fields, a section that is not a mapping, a ``_class`` that is not a configuration class);
        the message says which.  Any other error is a bug and propagates.
    """
    from dataclasses import fields
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
    try:
        config = LensClusterSolverConfig.from_json_dict(payload)
    except (UnknownField, MissingField, InvalidSection) as error:
        raise SavedConfigUnreadable(f"{source} is not a configuration this code can load ({_shown(error)})") from None
    absent = [f.name for f in fields(LensClusterSolverConfig) if f.name not in payload]
    if absent:                          # config.json holds every section (to_json_dict); a section-less one is no run's
        raise SavedConfigUnreadable(f"{source} is not a configuration this code can load (it lacks the section(s) "
                                    f"{', '.join(absent)})")
    return config


def run_status(run_dir) -> str:
    """The state of ``run_dir`` in words (what a resume then runs is in the module docstring)."""
    from .io import artifacts
    run_dir = Path(run_dir)
    init, final = run_dir / "init" / "artifacts", run_dir / "final" / "artifacts"
    if not artifacts.is_init_complete(init):
        return "init stage not complete (it runs again from the start)"
    if artifacts.is_final_complete(final):
        return "finished (the posterior is reloaded)"
    if (final / artifacts.CONFIG_FILE).is_file():
        return "interrupted in the final stage (sampling starts again from the saved chain start)"
    return "init stage complete, final stage not started"


def resume_config(run_dir, *, remap=(), overrides=(), draws=None, log=print):
    """``(config, saved_form)``: the configuration a resume of ``run_dir`` runs with, the saved one
    (:func:`settings_file`) with ``remap`` applied to its paths, the ``outputs`` ``overrides`` and ``draws``,
    ``runtime.resume`` on, and the run folder where ``run_dir`` is now; and the function that gives a configuration
    back with the remapped paths as saved (``pipeline.run(saved_form=...)``; None without a remap).

    Raises
    ------
    FileNotFoundError
        When ``run_dir`` holds no saved run, or a catalog of the model does not exist (the message names the field).
    ValueError
        For an override outside ``outputs``, an unknown or invalid one, a bad remap, or a saved configuration this
        code cannot load (:class:`SavedConfigUnreadable`).
    """
    from .io import artifacts
    from .outputs.saved_run import REMAP_HINT, apply_overrides, catalog_fields, missing_paths, relocate
    run_dir = Path(run_dir)
    source = settings_file(run_dir)
    if source is None:
        raise FileNotFoundError(f"{run_dir} is not a run folder (no init/artifacts/{artifacts.CONFIG_FILE})")
    if draws is not None:
        overrides = (*overrides, f"outputs.draws={int(draws)}", f"outputs.image_draws={int(draws)}")
    moved, saved_form = relocate(load_saved_config(source), remap, log=log)
    config = apply_overrides(moved, overrides)
    missing = missing_paths(config, catalog_fields(config))
    if missing:
        raise FileNotFoundError(f"run {run_dir}: {'; '.join(missing)} not found; {REMAP_HINT}")
    return replace(config, runtime=replace(config.runtime, resume=True),
                   paths=replace(config.paths, output_dir=str(run_dir.parent), run_name=run_dir.name)).validate(), saved_form


def main(argv=None) -> int:
    """``lumina resume``; returns the exit code: 2 for a usage error, a folder that is not a run, a refused
    override, a missing file, a refused stage, a run folder locked by another process, or a saved stage the run cannot
    continue (``pipeline.SavedStageMismatch``); 130 after Ctrl-C."""
    args = parser().parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lumina")
    from .config import UnknownField
    from .cpu_runtime import configure_cpu_runtime
    from .outputs.saved_run import check_outputs_only
    from .run_lock import lock_run, run_holder
    log = lambda text: print(text, flush=True)
    try:
        check_outputs_only(args.overrides)              # before anything starts: a fit setting is refused at once
        if not args.run.exists():
            raise FileNotFoundError(f"{args.run} does not exist")
        try:
            source = settings_file(args.run)
        except OSError as error:
            raise FileNotFoundError(f"{args.run} cannot be read ({error.strerror or error})") from None
        if source is None:
            raise FileNotFoundError(f"{args.run} is not a run folder (no init/artifacts/config.json)")
        # The CPU devices are set before anything starts JAX (reading the configuration builds priors).
        try:
            saved = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError):               # reported by resume_config (load_saved_config)
            saved = None
        runtime = saved.get("runtime") if isinstance(saved, dict) else None
        runtime = runtime if isinstance(runtime, dict) else {}         # a bad one is reported by resume_config
        if not args.dry_run:
            configure_cpu_runtime(int(runtime.get("chains", 1)), runtime.get("nuts_chain_method", "parallel"))
        config, saved_form = resume_config(args.run, remap=args.remap, overrides=args.overrides, draws=args.draws, log=log)
        from .io import artifacts
        if args.stage == "final" and not artifacts.is_init_complete(Path(args.run) / "init" / "artifacts"):
            raise ValueError(f"--stage final needs a completed init stage in {args.run}")
    except (FileNotFoundError, ValueError, UnknownField) as error:
        print(f"lumina resume: {error}", file=sys.stderr, flush=True)
        return 2
    status = run_status(args.run)
    if args.dry_run:
        holder = run_holder(args.run)
        locked = f", currently locked by {holder.holder()}" if holder is not None else ""
        log(f"[resume] {args.run} (settings of {source.relative_to(args.run)}): {status}; would run stage={args.stage}{locked}")
        print(json.dumps({"dry_run": True, "run": str(args.run), "status": status, "stage": args.stage,
                          "locked_by": None if holder is None else holder.holder(),
                          "config": config.to_json_dict()}, indent=2, default=str))
        return 0
    try:
        lock = lock_run(args.run, log=log)
    except (ValueError, OSError) as error:          # RunLocked, or a folder that cannot be written
        print(f"lumina resume: {error}", file=sys.stderr, flush=True)
        return 2
    from .pipeline import run
    log(f"[resume] {args.run} (settings of {source.relative_to(args.run)}): {status}; running stage={args.stage}")
    try:
        return run_and_report(lambda: run(config, stage=args.stage, saved_form=saved_form), args.run, prog="lumina resume")
    finally:
        lock.release()
