"""``kappaccino outputs RUN_DIR``: rerun the outputs stage of a finished run, without any fitting.

Run it from the folder the run was started from (usually the repository root): the run's saved paths are relative to
it, or absolute.  ``--remap OLD=NEW`` relocates the saved paths of a run made on another machine::

    kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c
    kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --only truth,mass
    kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --render
    kappaccino outputs results/m0416_canucs2025/<run> --skip rgb --fail-fast
    kappaccino outputs RUN --remap /home/me/burkelenscluster=$PWD --set truth.deflection_true_fits=defl.fits
    kappaccino outputs --list [RUN_DIR]

Module selection: ``--only``/``--skip`` replace the run's saved ``outputs.only`` and ``outputs.skip`` (both of them,
an option not given counts as empty), ``--all`` runs every module, and with none of the three the saved selection
applies.  The products go to the run's ``final/`` folder, as after the fit.

Pass mode: by default each module computes its results into the store and draws its figures from the store.
``--render`` only draws the figures again from what an earlier pass saved, without building the evaluator or the
reporting draws (a module whose compute never completed is reported unavailable); ``--compute`` writes the results
and no figures.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path


def _names(text: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in text.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("must name at least one output module, e.g. truth,rgb")
    return names


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _remap(text: str) -> tuple[str, str]:
    old, separator, new = text.partition("=")
    if not separator or not old.strip() or not new.strip():
        raise argparse.ArgumentTypeError("takes OLD=NEW, e.g. /home/me/burkelenscluster=/Users/me/burkelenscluster")
    return old.strip(), new.strip()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kappaccino outputs", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("run", nargs="?", type=Path, help="the run folder, results/<name>/<run> (optional with --list)")
    p.add_argument("--only", type=_names, metavar="A,B", help="run only these modules and the modules they require")
    p.add_argument("--skip", type=_names, metavar="C,D", help="do not run these modules nor the modules that require them")
    p.add_argument("--all", action="store_true", help="run every module, whatever the run's saved outputs.only and outputs.skip")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first step that raises")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--render", action="store_const", const="render", dest="mode",
                      help="only draw the figures again from the saved results: no evaluator, no reporting draws")
    mode.add_argument("--compute", action="store_const", const="compute", dest="mode",
                      help="only compute the results into the store: no figures")
    p.add_argument("--draws", type=_positive_int, metavar="N",
                   help="reporting draws (reporting.draws and reporting.image_draws, as the run command's --draws).  "
                        "Default: the run's")
    p.add_argument("--remap", type=_remap, action="append", default=[], metavar="OLD=NEW",
                   help="replace the prefix OLD of every saved path (catalogs, imaging, truth maps) by NEW; repeatable")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE",
                   help="override a report setting outside the run hash, e.g. truth.deflection_true_fits=FILE; repeatable")
    p.add_argument("--list", action="store_true", help="list the modules in run order; with a run, whether each is active")
    return p


def step_line(module) -> str:
    """The indented line of ``--list`` with the compute and the render steps of ``module``."""
    return f"    compute: {', '.join(module.compute_steps) or '-'}; render: {', '.join(module.render_steps) or '-'}"


def module_table(config=None) -> str:
    """The modules in run order with their ``requires`` and ``after``, each followed by an indented line of its steps
    by kind (:func:`step_line`), and, given the run's ``config``, whether each is active (from the configuration alone:
    conditions on the products of other modules show only in a pass)."""
    from .base import OutputContext, order_modules
    from .registry import MODULES
    ctx = OutputContext(config, None, None, None, None, None) if config is not None else None
    rows = [("module", "requires", "after", "active" if ctx is not None else "description")]
    for module in order_modules(MODULES):
        if ctx is not None:
            reason = module.active(ctx)
            last = "yes" if reason is None else f"no: {reason}"
        else:
            last = module.description
        rows.append((module.name, ",".join(module.requires) or "-", ",".join(module.after) or "-", last, step_line(module)))
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines = []
    for row in rows:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row[:3], widths)) + "  " + row[3])
        if len(row) > 4:
            lines.append(row[4])
    return "\n".join(lines)


def select_modules(config, *, only=None, skip=None, all_modules=False, fail_fast=False):
    """``config`` with the command-line module selection: ``all_modules`` runs every module; ``only`` or ``skip``
    replace the saved ``outputs.only`` and ``outputs.skip`` together (the one not given is empty); with none of them
    the saved selection stays.  ``fail_fast`` True turns ``outputs.fail_fast`` on."""
    outputs = config.outputs
    if all_modules:
        outputs = replace(outputs, only=None, skip=())
    elif only is not None or skip is not None:
        outputs = replace(outputs, only=None if only is None else tuple(only), skip=tuple(skip or ()))
    if fail_fast:
        outputs = replace(outputs, fail_fast=True)
    return replace(config, outputs=outputs)


def check_module_paths(config) -> None:
    """Refuse a pass whose active modules read a file that does not exist, naming the configuration field.

    Raises
    ------
    FileNotFoundError
        With every missing ``field = path`` and the hint to use ``--remap``.
    """
    from .base import OutputContext, plan_modules
    from .registry import MODULES
    from .saved_run import REMAP_HINT, missing_paths
    plan, _ = plan_modules(MODULES, only=config.outputs.only, skip=config.outputs.skip)
    ctx = OutputContext(config, None, None, None, None, None)
    missing = []
    for module in plan:
        if module.name in config.outputs.skip or module.active(ctx) is not None:
            continue
        missing += [f"{item} (module {module.name})" for item in missing_paths(config, module.path_fields)]
    if missing:
        raise FileNotFoundError(f"{'; '.join(dict.fromkeys(missing))} not found; {REMAP_HINT}")


def prepare_rerun(run_directory, *, only=None, skip=None, all_modules=False, fail_fast=False, draws=None, remap=(),
                  overrides=(), transform=None, mode="full", log=print):
    """The arguments of ``reporting.outputs.write_outputs`` for the finished run in ``run_directory``:
    ``(config, state, evaluator, posterior, audit, final StagePaths)``, loaded with ``saved_run.load_run`` (``remap``,
    ``overrides``, ``transform``).  The selection is :func:`select_modules`; ``draws`` sets ``reporting.draws`` and
    ``reporting.image_draws``, the one change to the fit's settings a rerun allows (both are in the run hash).  For a
    render pass (``mode`` "render") the run is loaded without its evaluator (``saved_run.load_saved_run``; evaluator
    None) and the files the modules' computes read are not checked, since render reads only the store.

    Raises
    ------
    FileNotFoundError
        When a catalog of the model or a file the selected modules read does not exist (:func:`check_module_paths`).
    ValueError
        For an override of a setting inside the run hash, or an unknown one.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kappaccino")
    import matplotlib
    matplotlib.use("Agg")
    from ..io import artifacts
    from ..pipeline import StagePaths
    from .saved_run import load_run, load_saved_run
    run = Path(run_directory).resolve()
    if mode == "render":
        if draws is not None:
            raise ValueError("--draws chooses the draws a compute uses; a render pass draws what was computed")
        config, state, posterior = load_saved_run(run, remap=remap, overrides=overrides, transform=transform, log=log)
        evaluator = None
    else:
        config, state, evaluator, posterior, _ = load_run(run, remap=remap, overrides=overrides, transform=transform, log=log)
    config = select_modules(config, only=only, skip=skip, all_modules=all_modules, fail_fast=fail_fast)
    if draws is not None:
        config = replace(config, reporting=replace(config.reporting, draws=int(draws), image_draws=int(draws)))
    if mode != "render":
        check_module_paths(config)
    final = StagePaths(run / "final")
    audit_file = final.artifacts / artifacts.AUDIT_FILE
    audit = json.loads(audit_file.read_text()) if audit_file.is_file() else {}
    return config, state, evaluator, posterior, audit, final


def rerun_outputs(run_directory, *, log=print, mode="full", **options) -> dict:
    """Run the outputs stage of the finished run in ``run_directory`` into its ``final/`` folder in ``mode``
    (:func:`prepare_rerun` with ``options``, then ``reporting.outputs.write_outputs``); returns this pass's summary."""
    from ..reporting.outputs import write_outputs
    return write_outputs(*prepare_rerun(run_directory, log=log, mode=mode, **options), log=log, mode=mode)


def main(argv=None) -> int:
    """``kappaccino outputs``; returns the exit code: 1 when a module failed or the pass stopped (``--fail-fast``),
    2 for a usage error, a missing file or a refused override."""
    p = parser()
    args = p.parse_args(argv)
    if args.list:
        config = None
        if args.run is not None:
            from ..io import artifacts
            config = artifacts.load_config(Path(args.run) / "final" / "artifacts")
        print(module_table(config))
        return 0
    if args.run is None:
        p.error("the run folder is required (or use --list)")
    if args.all and (args.only is not None or args.skip is not None):
        p.error("--all runs every module; it cannot be combined with --only or --skip")
    mode = args.mode or "full"
    if mode == "render" and args.draws is not None:
        p.error("--draws chooses the draws of a compute; --render draws what the last compute saved")
    from .registry import check_module_names
    for label, names in (("--only", args.only), ("--skip", args.skip)):
        if names is not None:
            try:
                check_module_names(label, names)
            except ValueError as error:
                p.error(str(error))
    log = lambda text: print(text, flush=True)
    try:
        prepared = prepare_rerun(args.run, only=args.only, skip=args.skip, all_modules=args.all, fail_fast=args.fail_fast,
                                 draws=args.draws, remap=args.remap, overrides=args.overrides, mode=mode, log=log)
    except (FileNotFoundError, ValueError) as error:
        print(f"kappaccino outputs: {error}", flush=True)
        return 2
    from ..reporting.outputs import write_outputs
    try:
        summary = write_outputs(*prepared, log=log, mode=mode)
    except Exception as error:          # --fail-fast: the step logged its traceback; one line here
        print(f"kappaccino outputs: stopped at the first error ({type(error).__name__}: {error})", flush=True)
        return 1
    from ..io import store
    try:
        saved = store.read_record(Path(args.run) / "final", "run_summary")
    except (store.MissingEntry, store.DamagedEntry, store.StoreError):
        saved = {}
    print(json.dumps(saved.get("fit", {}), indent=2, default=float), flush=True)
    failed = [name for name, record in summary.get("modules", {}).items() if record["status"] == "failed"]
    return 1 if failed else 0
