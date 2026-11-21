"""The outputs stage after a run: the output modules (:mod:`lumina.outputs`), the record ``run_summary`` and the
results macros."""
from __future__ import annotations

from typing import Any, Iterable

from ..config import LensClusterSolverConfig
from ..core.model import ModelState, PosteriorResults
from ..likelihood.evaluator import Evaluator
from ..outputs.base import OutputContext, disabled_modules, plan_modules, run_modules


def write_outputs(config: LensClusterSolverConfig, state: ModelState, evaluator: Evaluator | None, posterior: PosteriorResults,
                  audit: dict[str, Any], paths, *, log=print, only: Iterable[str] | None = None,
                  skip: Iterable[str] | None = None, all_modules: bool = False, fail_fast: bool | None = None,
                  mode: str = "full") -> dict[str, Any]:
    """Run the output modules (``outputs.registry.MODULES``) and write the record ``run_summary`` and the results
    macros; returns the summary.  ``paths`` is the final StagePaths.  The modules run are chosen by the ``enabled``
    switch of each module's section of ``config.outputs`` (``config.OutputsConfig``) and the selection
    (``lumina outputs --only/--skip/--all``): ``only`` runs these modules and the modules they require, enabled or
    not; ``all_modules`` runs every module, enabled or not; ``skip`` takes these modules and those that require them
    out of what would run otherwise (the enabled modules, or with ``all_modules`` every module).  ``fail_fast``
    defaults to ``config.outputs.fail_fast``.  ``mode`` is the pass mode
    (``outputs.base.MODES``): ``full``, ``compute`` (no figures) or ``render`` (figures from the store only; the
    evaluator may be None and is never used).  A render pass always merges its summary into the saved one and keeps
    every module's numbers (``summary_keys``) from the last compute.

    Every product is computed over one set of reporting draws (``reporting.draws.reporting_draws``), each draw
    evaluated once and shared by all of them; none uses the single best sample.  The summary and the macros are written
    before the first imaging module (the slowest and most failure-prone part: the Ares run of 2026-11-08 ended inside
    it with no run_summary and "not available" macros) and again at the end, also when a module failed.  A pass that
    does not run every module (``only`` or ``skip``, a disabled module, or a module failed or was skipped) merges its summary into the
    saved ``run_summary`` (:func:`merged_summary`).  ``cache.h5`` and ``results.h5`` are compacted at the end, also when
    ``fail_fast`` raises.

    A second pass over the same stage (a resumed run) reuses what the first one saved: the image-recovery tables from
    results.h5 when their fingerprint matches (``diagnostics.image_recovery.image_recovery_with_reuse``), the lens maps
    of every grid from cache.h5 (``ReportingDraws.grid_maps``), and the aperture masses; each reuse or recomputation is
    logged with its reason."""
    from ..io import store
    from ..outputs.registry import MODULES
    disabled = () if only is not None or all_modules else disabled_modules(config, MODULES)
    only = None if only is None else tuple(only)
    skip = () if skip is None else tuple(skip)
    fail_fast = config.outputs.fail_fast if fail_fast is None else fail_fast
    plan_modules(MODULES, only=only, skip=skip)          # unknown names fail here, before anything is written
    ctx = OutputContext(config, state, evaluator, posterior, audit, paths, log=log, mode=mode)
    partial = only is not None or bool(skip) or bool(disabled) or mode == "render"
    try:
        run_modules(ctx, MODULES, only=only, skip=skip, disabled=disabled, fail_fast=fail_fast,
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
    from ``saved`` first, so a failed module leaves no old numbers.  So are those of a module skipped because a module
    it requires failed in this pass (its record has ``failed_requirement``): what its numbers came from was computed
    again and failed, and its record is this pass's.  A disabled module (``outputs.<name>.enabled`` false) keeps its
    saved keys and gets this pass's record, "disabled"; so does a module skipped because a module it requires is
    disabled in this pass (its record has ``disabled_requirement``): nothing was recomputed or failed, so its numbers
    stay, but its record says it did not run.  A module that did not run otherwise (not selected, in the skip
    list or requiring one that is, or inactive) keeps its saved keys and its saved module record.  In a render pass
    (``mode`` "render") the modules get this pass's records but keep their saved keys, which come from their last
    compute.  ``warnings`` belong to the reporting draws and are replaced when this pass built them; the warning of
    ``outputs.base.check_draws`` and those of ``outputs.base.check_settings`` are this pass's in any case.
    ``reporting_draws`` is that of the last pass that built draws.  The module records are in the order of ``modules`` (the registry), then any other saved ones."""
    from ..outputs.base import DRAWS_WARNING, SETTINGS_WARNING
    records = summary.get("modules", {})
    blocked = {name for name, record in records.items() if record["status"] == "skipped" and "failed_requirement" in record}
    ran = {name for name, record in records.items() if record["status"] in ("done", "failed")} | blocked
    computed = {name for name in ran if records[name].get("mode") != "render"}
    base = {key: value for key, value in saved.items() if key != "modules"}
    for module in modules:
        if module.name in computed:
            for key in module.summary_keys:
                base.pop(key, None)
    kept = [] if "reporting_draws" in summary else [w for w in base.get("warnings", [])
                                                    if not str(w).startswith((DRAWS_WARNING, SETTINGS_WARNING))]
    warnings = list(dict.fromkeys(kept + list(summary.get("warnings", []))))
    base.pop("warnings", None)
    if warnings:
        base["warnings"] = warnings
    saved_records = saved.get("modules", {}) if isinstance(saved.get("modules"), dict) else {}
    disabled = {name for name, record in records.items()
                if record["status"] == "disabled" or "disabled_requirement" in record}
    chosen = {**saved_records, **{name: record for name, record in records.items()
                                  if name in ran or name in disabled or name not in saved_records}}
    order = [m.name for m in modules] + [name for name in chosen if name not in {m.name for m in modules}]
    return {**base, **{key: value for key, value in summary.items() if key not in ("modules", "warnings")},
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
