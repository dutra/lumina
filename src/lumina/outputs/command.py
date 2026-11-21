"""``lumina outputs RUN_DIR``: rerun the outputs stage of a finished run, without any fitting.

Run it from the folder the run was started from (usually the repository root): the run's saved paths are relative to
it, or absolute.  ``--remap OLD=NEW`` relocates the saved paths of a run made on another machine (OLD absolute; a
relative saved path is compared as the absolute path it names from here)::

    lumina outputs results/hera/w500s250t7_allblocks-3f9a1c
    lumina outputs results/hera/w500s250t7_allblocks-3f9a1c --only truth,mass
    lumina outputs results/hera/w500s250t7_allblocks-3f9a1c --render
    lumina outputs results/m0416_canucs2025/<run> --skip rgb --fail-fast
    lumina outputs RUN --remap /home/me/burkelenscluster=$PWD --set outputs.truth.deflection_true_fits=defl.fits
    lumina outputs RUN --set outputs.rgb.enabled=false
    lumina outputs RUN --all --skip rgb
    lumina outputs --list [RUN_DIR]

Module selection: a module runs when its ``outputs.<module>.enabled`` is true (in the run's saved configuration, or as
``--set`` changes it) and it is active.  ``--only A,B`` runs A, B and the modules they require, enabled or not (an
explicit request); ``--all`` runs every module, enabled or not; ``--skip C,D`` also leaves out C, D and the modules
that require them, on top of the switches (or of ``--all``: ``--all --skip rgb`` runs everything but rgb).  The
products go to the run's ``final/`` folder, as after the fit.

Pass mode: by default each module computes its results into the store and draws its figures from the store.
``--render`` only draws the figures again from what an earlier pass saved, without building the evaluator or the
reporting draws: each render step is drawn when the compute step it needs wrote in the module's last compute; a module
never computed here fails (``NotComputed``) with the hint to compute it first; a render step that cannot be drawn (its
compute step failed or did not run, or ``rgb`` cannot load the imaging of a variant its compute drew) is reported and
fails the module; a render pass that draws nothing at all says why and exits with 1.  ``--compute`` writes the results
and no figures (on a machine without the imaging: ``--compute --skip rgb``).

``--set`` changes only the ``outputs`` section (``outputs.FIELD`` or ``outputs.MODULE.FIELD``), which never changes
the fit, checked (``config.validate_outputs``) before the pass.  ``--draws N`` sets ``outputs.draws`` and
``outputs.image_draws``.  Both hold for this pass only: the command never writes ``final/artifacts/config.json``, so
the next pass runs with the saved settings again unless it repeats them.  Each module's completeness record keeps
the outputs settings its last compute used; when a module's recorded settings differ from this pass's in a field its
compute reads, the pass ends with a warning naming the module and the fields (old -> current).  ``--list RUN``
applies ``--remap`` and ``--set`` before listing.
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
    p = argparse.ArgumentParser(prog="lumina outputs", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("run", nargs="?", type=Path, help="the run folder, results/<name>/<run> (optional with --list)")
    p.add_argument("--only", type=_names, metavar="A,B", help="run only these modules and the modules they require, enabled or not")
    p.add_argument("--skip", type=_names, metavar="C,D",
                   help="also leave out these modules and the modules that require them (disabled modules stay off)")
    p.add_argument("--all", action="store_true",
                   help="run every module, whatever the outputs.<module>.enabled switches (with --skip: all but those)")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first step that raises")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--render", action="store_const", const="render", dest="mode",
                      help="only draw the figures again from the saved results: no evaluator, no reporting draws")
    mode.add_argument("--compute", action="store_const", const="compute", dest="mode",
                      help="only compute the results into the store: no figures")
    p.add_argument("--draws", type=_positive_int, metavar="N",
                   help="reporting draws (outputs.draws and outputs.image_draws, as the run command's --draws).  "
                        "Default: the run's")
    p.add_argument("--remap", type=_remap, action="append", default=[], metavar="OLD=NEW",
                   help="replace the absolute prefix OLD of every saved path (catalogs, imaging, truth maps) by NEW; "
                        "repeatable")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="outputs.FIELD=VALUE",
                   help="override a field of the outputs section, e.g. outputs.workers=8, outputs.rgb.enabled=false; "
                        "repeatable")
    p.add_argument("--list", action="store_true",
                   help="list the modules in run order; with a run, whether each is enabled and active")
    return p


def step_line(module) -> str:
    """The indented line of ``--list`` with the compute and the render steps of ``module``."""
    return f"    compute: {', '.join(module.compute_steps) or '-'}; render: {', '.join(module.render_steps) or '-'}"


def module_table(config=None) -> str:
    """The modules in run order with their ``requires`` and ``after``, each followed by an indented line of its steps
    by kind (:func:`step_line`), and, given the run's ``config``, whether each is enabled (``outputs.<module>.enabled``)
    and active (from the configuration alone: conditions on the products of other modules show only in a pass)."""
    from .base import OutputContext, order_modules
    from .registry import MODULES
    ctx = OutputContext(config, None, None, None, None, None) if config is not None else None
    modules = order_modules(MODULES)
    rows = [("module", "requires", "after") + (("enabled", "active") if ctx is not None else ("description",))]
    for module in modules:
        cells = (module.name, ",".join(module.requires) or "-", ",".join(module.after) or "-")
        if ctx is not None:
            reason = ctx.inactive_reason(module)
            cells += ("yes" if getattr(config.outputs, module.name).enabled else "no",
                      "yes" if reason is None else f"no: {reason}")
        else:
            cells += (module.description,)
        rows.append(cells)
    padded = len(rows[0]) - 1                   # every column but the last is padded
    widths = [max(len(row[i]) for row in rows) for i in range(padded)]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row[:padded], widths)) + "  " + row[padded])
        if index:
            lines.append(step_line(modules[index - 1]))
    return "\n".join(lines)


def check_module_paths(config, *, only=None, skip=None, all_modules=False) -> None:
    """Refuse a pass whose active modules read a file that does not exist, naming the configuration field.  The
    modules are those the pass runs: the selection ``only``, ``skip``, ``all_modules`` over the enabled modules, as
    ``reporting.outputs.write_outputs`` takes it, without the modules that require a module left out.

    Raises
    ------
    FileNotFoundError
        With every missing ``field = path`` and the hint to use ``--remap``.
    """
    from .base import OutputContext, disabled_modules, plan_modules
    from .registry import MODULES
    from .saved_run import REMAP_HINT, missing_paths
    skipped = set(skip or ()) | (set() if only is not None or all_modules else set(disabled_modules(config, MODULES)))
    plan, _ = plan_modules(MODULES, only=only, skip=tuple(skip or ()))
    ctx = OutputContext(config, None, None, None, None, None)
    missing = []
    for module in plan:
        if (module.name in skipped or set(module.requires) & skipped
                or ctx.inactive_reason(module) is not None):
            skipped.add(module.name)            # so are the modules that require it
            continue
        missing += [f"{item} (module {module.name})"
                    for item in missing_paths(config, [f"outputs.{name}" for name in module.path_fields])]
    if missing:
        raise FileNotFoundError(f"{'; '.join(dict.fromkeys(missing))} not found; {REMAP_HINT}")


def prepare_rerun(run_directory, *, only=None, skip=None, all_modules=False, fail_fast=False, draws=None, remap=(),
                  overrides=(), transform=None, mode="full", log=print):
    """The arguments of ``reporting.outputs.write_outputs`` for the finished run in ``run_directory``:
    ``(config, state, evaluator, posterior, audit, final StagePaths)``, loaded with ``saved_run.load_run`` (``remap``,
    ``overrides``, ``transform``).  ``fail_fast`` True turns ``outputs.fail_fast`` on; ``draws`` sets
    ``outputs.draws`` and ``outputs.image_draws``.  The module selection ``only``, ``skip``, ``all_modules`` is passed
    to ``write_outputs`` by the caller; here it chooses the modules whose files are checked
    (:func:`check_module_paths`).  For a render pass (``mode`` "render") the run is loaded without its evaluator
    (``saved_run.load_saved_run``; evaluator None) and the files the modules' computes read are not checked, since
    render reads only the store.

    Raises
    ------
    FileNotFoundError
        When a catalog of the model or a file the selected modules read does not exist (:func:`check_module_paths`).
    ValueError
        For an override of a setting inside the run hash, or an unknown one.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lumina")
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
    if fail_fast:
        config = replace(config, outputs=replace(config.outputs, fail_fast=True))
    if draws is not None:
        config = replace(config, outputs=replace(config.outputs, draws=int(draws), image_draws=int(draws)))
    if mode != "render":
        check_module_paths(config, only=only, skip=skip, all_modules=all_modules)
    final = StagePaths(run / "final")
    audit_file = final.artifacts / artifacts.AUDIT_FILE
    audit = json.loads(audit_file.read_text()) if audit_file.is_file() else {}
    return config, state, evaluator, posterior, audit, final


def rerun_outputs(run_directory, *, log=print, mode="full", only=None, skip=None, all_modules=False, **options) -> dict:
    """Run the outputs stage of the finished run in ``run_directory`` into its ``final/`` folder in ``mode``
    (:func:`prepare_rerun` with ``options``, then ``reporting.outputs.write_outputs`` with the module selection
    ``only``, ``skip``, ``all_modules``: none of them runs the enabled modules); returns this pass's summary.  The
    run's ``final/artifacts/config.json`` is never written: ``--set`` and ``--draws`` hold for this pass only."""
    from ..reporting.outputs import write_outputs
    selection = dict(only=only, skip=skip, all_modules=all_modules)
    prepared = prepare_rerun(run_directory, log=log, mode=mode, **selection, **options)
    return write_outputs(*prepared, log=log, mode=mode, **selection)


def main(argv=None) -> int:
    """``lumina outputs``; returns the exit code: 1 when a module failed, the pass stopped (``--fail-fast``) or
    raised, or a render pass drew nothing; 2 for a usage error, a missing file, a refused override or remap."""
    p = parser()
    args = p.parse_args(argv)
    if args.list:
        config = None
        if args.run is not None:            # the configuration a pass would run with: remapped, then --set applied
            from ..io import artifacts
            from .saved_run import apply_overrides, remap_paths
            try:
                config = apply_overrides(remap_paths(artifacts.load_config(Path(args.run) / "final" / "artifacts"),
                                                     args.remap, log=lambda _: None), args.overrides)
            except ValueError as error:
                print(f"lumina outputs: {error}", flush=True)
                return 2
        elif args.remap or args.overrides:
            p.error("--remap and --set with --list need the run folder")
        print(module_table(config))
        return 0
    if args.run is None:
        p.error("the run folder is required (or use --list)")
    if args.all and args.only is not None:
        p.error("--all runs every module; it cannot be combined with --only")
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
        print(f"lumina outputs: {error}", flush=True)
        return 2
    from ..reporting.outputs import write_outputs
    try:
        summary = write_outputs(*prepared, log=log, mode=mode, only=args.only, skip=args.skip, all_modules=args.all)
    except Exception as error:
        if prepared[0].outputs.fail_fast:       # the step logged its traceback; one line here
            print(f"lumina outputs: stopped at the first error (fail_fast: {type(error).__name__}: {error})", flush=True)
        else:                                   # outside the modules (the run summary, the store): nothing logged it
            import traceback
            print(f"lumina outputs: the pass failed outside the modules ({type(error).__name__}: {error})\n"
                  f"{traceback.format_exc().rstrip()}", flush=True)
        return 1
    from ..io import store
    try:
        saved = store.read_record(Path(args.run) / "final", "run_summary")
    except (store.MissingEntry, store.DamagedEntry, store.StoreError):
        saved = {}
    print(json.dumps(saved.get("fit", {}), indent=2, default=float), flush=True)
    from .base import nothing_drawn_reason, rendered_anything
    if mode == "render" and not rendered_anything(summary["modules"]):
        print(f"lumina outputs: the render pass drew nothing: {nothing_drawn_reason(summary['modules'])}", flush=True)
        return 1
    failed = [name for name, record in summary["modules"].items() if record["status"] == "failed"]
    return 1 if failed else 0
