"""The outputs stage after a run: the output modules (:mod:`kappaccino.outputs`), the record ``run_summary`` and the
results macros."""
from __future__ import annotations

from typing import Any, Iterable

from ..config import LensClusterSolverConfig
from ..core.model import ModelState, PosteriorResults
from ..likelihood.evaluator import Evaluator
from ..outputs.base import OutputContext, plan_modules, run_modules


def write_outputs(config: LensClusterSolverConfig, state: ModelState, evaluator: Evaluator | None, posterior: PosteriorResults,
                  audit: dict[str, Any], paths, *, log=print, only: Iterable[str] | None = None,
                  skip: Iterable[str] | None = None, fail_fast: bool | None = None, mode: str = "full") -> dict[str, Any]:
    """Run the output modules (``outputs.registry.MODULES``) and write the record ``run_summary`` and the results
    macros; returns the summary.  ``paths`` is the final StagePaths.  ``only``, ``skip`` and ``fail_fast`` default to
    the ``outputs`` section of ``config`` (``config.OutputsConfig``).  ``mode`` is the pass mode
    (``outputs.base.MODES``): ``full``, ``compute`` (no figures) or ``render`` (figures from the store only; the
    evaluator may be None and is never used).  A render pass always merges its summary into the saved one and keeps
    every module's numbers (``summary_keys``) from the last compute.

    Every product is computed over one set of reporting draws (``reporting.draws.reporting_draws``), each draw
    evaluated once and shared by all of them; none uses the single best sample.  The summary and the macros are written
    before the first imaging module (the slowest and most failure-prone part: the Ares run of 2026-11-08 ended inside
    it with no run_summary and "not available" macros) and again at the end, also when a module failed.  A pass that
    does not run every module (``only`` or ``skip``, or a module failed or was skipped) merges its summary into the
    saved ``run_summary`` (:func:`merged_summary`).  ``cache.h5`` and ``results.h5`` are compacted at the end, also when
    ``fail_fast`` raises.

    A second pass over the same stage (a resumed run) reuses what the first one saved: the image-recovery tables from
    results.h5 when their fingerprint matches (``diagnostics.image_recovery.image_recovery_with_reuse``), the lens maps
    of every grid from cache.h5 (``ReportingDraws.grid_maps``), and the aperture masses; each reuse or recomputation is
    logged with its reason."""
    from ..io import store
    from ..outputs.registry import MODULES
    settings = config.outputs if None in (only, skip, fail_fast) else None
    only = settings.only if only is None else tuple(only)
    skip = settings.skip if skip is None else tuple(skip)
    fail_fast = settings.fail_fast if fail_fast is None else fail_fast
    plan_modules(MODULES, only=only, skip=skip)          # unknown names fail here, before anything is written
    ctx = OutputContext(config, state, evaluator, posterior, audit, paths, log=log, mode=mode)
    partial = only is not None or bool(skip) or mode == "render"
    try:
        run_modules(ctx, MODULES, only=only, skip=skip, fail_fast=fail_fast,
                    checkpoint=lambda: write_run_summary(ctx, partial=partial))
    except Exception:                   # fail_fast: the summary of what ran, then the error
        try:
            write_run_summary(ctx, partial=partial)
        except Exception as error:
            log(f"[outputs] run summary not written: {type(error).__name__}: {error}")
        raise
    else:
        write_run_summary(ctx, partial=partial)
    finally:
        store.compact(ctx.stage)
    return ctx.summary


def merged_summary(saved: dict[str, Any], summary: dict[str, Any], modules) -> dict[str, Any]:
    """The record ``run_summary`` after a pass that did not run every module: ``summary`` (this pass) over ``saved``
    (the record before it).  For each module that ran in this pass (done or failed), its ``summary_keys`` are dropped
    from ``saved`` first, so a failed module leaves no old numbers; a module that did not run (not selected, skipped or
    inactive) keeps its saved keys and its saved module record.  A module that ran in a render pass (``mode``
    "render") gets this pass's record but keeps its saved keys, which come from its last compute.  ``warnings`` belong
    to the reporting draws and are replaced when this pass built them; ``reporting_draws`` is that of the last pass
    that built draws.  The module records are in the order of ``modules`` (the registry), then any other saved ones."""
    records = summary.get("modules", {})
    ran = {name for name, record in records.items() if record["status"] in ("done", "failed")}
    computed = {name for name in ran if records[name].get("mode") != "render"}
    base = {key: value for key, value in saved.items() if key != "modules"}
    for module in modules:
        if module.name in computed:
            for key in module.summary_keys:
                base.pop(key, None)
    if "reporting_draws" in summary:
        base.pop("warnings", None)
    saved_records = saved.get("modules", {}) if isinstance(saved.get("modules"), dict) else {}
    chosen = {**saved_records, **{name: record for name, record in records.items() if name in ran or name not in saved_records}}
    order = [m.name for m in modules] + [name for name in chosen if name not in {m.name for m in modules}]
    return {**base, **{key: value for key, value in summary.items() if key != "modules"},
            "modules": {name: chosen[name] for name in order if name in chosen}}


def write_run_summary(ctx: OutputContext, *, partial: bool, modules=None) -> None:
    """Write the record ``run_summary`` and the results macros (``results_tex.write_run_results``).  The record is
    ``ctx.summary``; with ``partial``, or when a module failed or was skipped, it is :func:`merged_summary` of the saved
    record and ``ctx.summary``.  ``modules`` defaults to the registry."""
    from ..io import store
    from ..outputs.registry import MODULES
    from . import results_tex
    record = ctx.summary
    statuses = {r["status"] for r in record.get("modules", {}).values()}
    if partial or statuses & {"failed", "skipped"}:
        try:
            saved = store.read_record(ctx.stage, "run_summary")
        except (store.MissingEntry, store.DamagedEntry, store.StoreError):
            saved = {}
        record = merged_summary(saved if isinstance(saved, dict) else {}, record, MODULES if modules is None else modules)
    store.write_record(ctx.stage, "run_summary", record, default=float)
    results_tex.write_run_results(ctx.paths.root, name=ctx.prefix)
