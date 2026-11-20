"""The framework of the outputs stage: the context every module reads, the module base class and the runner.

A module (:class:`OutputModule`) writes one group of tables and figures as a sequence of named steps
(:meth:`OutputContext.step`).  A step that raises is logged with its traceback and recorded as failed, and the
module goes on with its other steps; a step that needs the result of a failed one is skipped with the reason
(:meth:`OutputContext.skip_step`).  A module with a failed step ends as "failed" but keeps its product; a module that
cannot give its dependents what they need raises (:class:`ModuleUnavailable` after its failed steps), and is then
failed with no product.

The runner (:func:`run_modules`) orders the modules by what they need (``requires``, hard: the module reads
``ctx.products`` of those) and by what they should follow (``after``, ordering only), stably with respect to the
registry order (``outputs.registry.MODULES``), and runs each one that is selected, not skipped and active.  A module
runs only when every module it requires returned from its ``run`` (its product may be None).  Each module's outcome,
wall time, steps and the reporting draws it used go to ``ctx.summary["modules"]``, and so to the record
``run_summary``.

The modules import matplotlib, the plotting and diagnostics modules and the image solver inside their ``run``, so
importing the registry (``config.validate`` does, to check module names) loads none of them.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REBUILT_FIELD_WARNING = (
    "gp_blocks = {blocks}: the posterior gave each block its own mass field, and the products rebuild one shared field "
    "from it. That field is not a posterior sample of the lens, so the image residuals, RMS, position contours and map "
    "uncertainties in the draws' own lenses are not trustworthy and likely inflated. Use an all-block run for reported numbers.")


class OutputContext:
    """What the modules of one output pass share.

    ``config``, ``state``, ``evaluator``, ``posterior`` and ``audit`` are those of the fit; ``paths`` is the final
    ``StagePaths`` (``root``, ``tables``, ``figures``, ``fits``).  ``summary`` is the record ``run_summary`` being
    built, and ``products`` holds what each module's ``run`` returned, by module name.  The reporting draws
    (:attr:`draws`) are built on first access, so a pass of modules that need none never builds them, and
    :meth:`imaging` loads the imaging of each variant once per pass."""

    def __init__(self, config, state, evaluator, posterior, audit: dict[str, Any] | None, paths, *, log=print):
        self.config, self.state, self.evaluator, self.posterior = config, state, evaluator, posterior
        self.audit = audit or {}
        self.paths, self.log = paths, log
        self.summary: dict[str, Any] = {}
        self.products: dict[str, Any] = {}
        self._draws = None
        self._draws_error: Exception | None = None    # a failed build, raised again on every later access
        self._used_draws = False                      # the current module touched the draws
        self._logged: Exception | None = None         # an error a step already logged with its traceback
        self._imaging: list[tuple[Any, Any]] = []
        self._module: str | None = None              # the module running, its step records and fail_fast
        self._steps: dict[str, dict] = {}
        self._fail_fast = False

    @property
    def tables(self) -> Path:
        return self.paths.tables

    @property
    def figures(self) -> Path:
        return self.paths.figures

    @property
    def prefix(self) -> str:
        return self.config.paths.figure_prefix

    @property
    def stage(self) -> Path:
        """The stage folder, whose ``results.h5`` and ``cache.h5`` receive the products."""
        from ..io import store
        return store.stage_dir(self.paths.tables)

    @property
    def draws_built(self) -> bool:
        return self._draws is not None

    @property
    def draws(self):
        """The reporting draws (``reporting.draws.reporting_draws``), built on first access with their grid maps cached
        in the stage's ``cache.h5``.  Building them sets ``summary["reporting_draws"]`` and, for a field rebuilt from
        per-block fields, ``summary["warnings"]``.  A build that fails raises the same error on every later access."""
        self._used_draws = True
        if self._draws_error is not None:
            raise self._draws_error
        if self._draws is None:
            from ..reporting.draws import reporting_draws
            try:
                draws = reporting_draws(self.config, self.evaluator, self.posterior)
            except Exception as error:
                self._draws_error = error
                raise
            draws.cache_stage = self.stage
            self.log(f"[outputs] {len(draws)} reporting draws (seed {draws.seed})")
            self.summary["reporting_draws"] = len(draws)
            if draws.has_field and draws.field_is_rebuilt:
                self.summary["warnings"] = [REBUILT_FIELD_WARNING.format(blocks=self.state.gp_blocks)]
                self.log(f"[outputs] WARNING: {self.summary['warnings'][0]}")
            self._draws = draws
        return self._draws

    def imaging(self, image_config):
        """The imaging (``imaging.cutouts.load_cluster_imaging``) of the variant ``image_config``, loaded once per pass;
        variants that differ only in ``figure_suffix`` share it.  None when the imaging folder is missing, which is
        logged once and leaves the other variants and outputs alone."""
        key = replace(image_config, figure_suffix="")
        for known, imaging in self._imaging:
            if known == key:
                return imaging
        from ..imaging.cutouts import load_cluster_imaging
        try:
            imaging = load_cluster_imaging(image_config, self.config.model, self.config.paths.run_name)
        except FileNotFoundError as error:
            # Missing external imaging must not prevent the other outputs.
            self.log(f"[outputs] cluster mass figure and covariance RGB cutouts skipped: {error}")
            imaging = None
        self._imaging.append((key, imaging))
        return imaging

    def step(self, name: str, fn: Callable, *args, **kwargs):
        """Run one step of the current module, ``fn(*args, **kwargs)``, and return its result.  The step's status and
        wall time go to the module's record (``steps``) and the log (``[outputs] <module>.<step>``).  An exception
        (not KeyboardInterrupt or SystemExit) is logged with its traceback and recorded, and the step returns None so
        that the module goes on; :meth:`ok` tells a failed step from one that returned None.  With ``fail_fast`` the
        exception is raised again."""
        if self._module is None:
            raise RuntimeError("OutputContext.step called outside a module run")
        label = f"{self._module}.{name}"
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as error:
            seconds = time.perf_counter() - started
            self._steps[name] = {"status": "failed", "seconds": seconds, "error": f"{type(error).__name__}: {error}"}
            self.log(f"[outputs] {label}: FAILED after {seconds:.1f}s: {type(error).__name__}: {error}\n"
                     f"{traceback.format_exc().rstrip()}")
            if self._fail_fast:
                self._logged = error
                raise
            return None
        seconds = time.perf_counter() - started
        self._steps[name] = {"status": "done", "seconds": seconds}
        self.log(f"[outputs] {label}: done in {seconds:.1f}s")
        return result

    def ok(self, name: str) -> bool:
        """Whether the step ``name`` of the current module ran without an error."""
        return self._steps.get(name, {}).get("status") == "done"

    def skip_step(self, name: str, reason: str) -> None:
        """Record the step ``name`` of the current module as skipped, with the reason (logged)."""
        self._steps[name] = {"status": "skipped", "seconds": 0.0, "reason": reason}
        self.log(f"[outputs] {self._module}.{name}: skipped ({reason})")


class ModuleUnavailable(RuntimeError):
    """Raised by a module that cannot give its dependents what they need (its essential step failed, which was logged):
    the module is failed with no product and the modules that require it are skipped."""


class OutputModule:
    """One group of output products, written as named steps (:meth:`OutputContext.step`).

    Subclasses set ``name`` and, when needed, ``requires`` (modules whose products this one reads from
    ``ctx.products``; the module runs only when each of them returned from ``run``) and ``after`` (modules it should follow
    when both run, for example because it reads a table they write).  ``checkpoint`` marks the imaging module: the run
    summary and the results macros are written before it runs, because it is the slowest and most failure-prone part
    of the stage.  ``description`` is the one line of ``kappaccino outputs --list`` and ``steps`` names its steps.
    ``summary_keys`` are the top-level keys of ``run_summary`` the module writes: a later pass that runs the module
    drops them before merging, so a failed module leaves no old numbers behind.  ``path_fields`` are the configuration
    fields (``section.field``) of the files the module reads, checked before a rerun (``outputs.command``)."""

    name: str = ""
    requires: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    checkpoint: bool = False
    description: str = ""
    steps: tuple[str, ...] = ()
    summary_keys: tuple[str, ...] = ()
    path_fields: tuple[str, ...] = ()

    def active(self, ctx: OutputContext) -> str | None:
        """None when the module applies to this run, else a short reason why not (logged, recorded as inactive).
        Decided from ``ctx.config`` and, at run time, the products of the modules it requires (absent when listing)."""
        return None

    def run(self, ctx: OutputContext) -> Any:
        """Write the products; the return value is ``ctx.products[self.name]`` (may be None).  Raise
        :class:`ModuleUnavailable` when the dependents cannot run."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<output module {self.name}>"


def order_modules(modules: Sequence[OutputModule]) -> list[OutputModule]:
    """``modules`` ordered so that each follows its ``requires`` and ``after``; among the modules free to go next, the
    earliest in ``modules`` goes first, so modules already in a valid order keep it.

    Raises
    ------
    ValueError
        For a duplicate name, a ``requires`` or ``after`` naming no module, or a dependency cycle.
    """
    names = [m.name for m in modules]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate output module names in {names}")
    for module in modules:
        for field in ("requires", "after"):
            unknown = [n for n in getattr(module, field) if n not in names]
            if unknown:
                raise ValueError(f"output module {module.name!r}: {field} names unknown modules {unknown}; known: {names}")
    pending, placed, ordered = list(modules), set(), []
    while pending:
        ready = next((m for m in pending if set(m.requires) | set(m.after) <= placed), None)
        if ready is None:
            raise ValueError("output modules depend on each other in a cycle: "
                             + ", ".join(f"{m.name} (requires {list(m.requires)}, after {list(m.after)})" for m in pending))
        pending.remove(ready)
        placed.add(ready.name)
        ordered.append(ready)
    return ordered


def plan_modules(modules: Sequence[OutputModule], *, only: Iterable[str] | None = None,
                 skip: Iterable[str] = ()) -> tuple[list[OutputModule], dict[str, str]]:
    """The modules a pass runs, in run order, and those added to ``only`` because a selected module requires them
    (``{added: the module that required it}``).  ``only`` None selects every module.

    Raises
    ------
    ValueError
        For a name in ``only`` or ``skip`` that is no module, and the errors of :func:`order_modules`.
    """
    ordered = order_modules(modules)
    by_name = {m.name: m for m in ordered}
    for label, names in (("only", only or ()), ("skip", skip)):
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise ValueError(f"outputs.{label} names unknown output modules {unknown}; known: {list(by_name)}")
    if only is None:
        return ordered, {}
    selected, added = set(only), {}
    stack = list(only)
    while stack:
        name = stack.pop()
        for required in by_name[name].requires:
            if required not in selected:
                selected.add(required)
                added[required] = name
                stack.append(required)
    return [m for m in ordered if m.name in selected], added


def run_modules(ctx: OutputContext, modules: Sequence[OutputModule] | None = None, *, only: Iterable[str] | None = None,
                skip: Iterable[str] = (), fail_fast: bool = False, checkpoint: Callable[[], None] | None = None) -> dict[str, dict]:
    """Run the output modules (default: the registry, ``outputs.registry.MODULES``) on ``ctx``.

    ``only`` selects modules plus whatever they require; ``skip`` removes modules, and a module that requires a skipped,
    inactive or failed-without-product one is skipped too.  A failing step is recorded and the module goes on
    (:meth:`OutputContext.step`); an error outside the steps fails the module, and the other modules still run.
    ``fail_fast`` re-raises the first error instead.  ``checkpoint`` is called once, before the first module with
    ``checkpoint`` set that runs.  Every module of the plan gets a record in ``ctx.summary["modules"]``: ``status``
    ("done", "inactive", "skipped" or "failed"), ``seconds``, ``steps`` (each step's ``status``, ``seconds`` and
    ``error`` or ``reason``), ``reporting_draws`` and ``image_draws`` when it used the draws, and ``reason`` or
    ``error``.  Returns those records."""
    if modules is None:
        from .registry import MODULES as modules
    skip = tuple(skip)
    plan, added = plan_modules(modules, only=only, skip=skip)
    log = ctx.log
    if added:
        log("[outputs] modules added because a selected module requires them: "
            + ", ".join(f"{name} (by {by})" for name, by in added.items()))
    log(f"[outputs] modules: {', '.join(m.name for m in plan) or 'none'}")
    records = ctx.summary.setdefault("modules", {})
    checkpointed = False
    for module in plan:
        name = module.name
        if name in skip:
            records[name] = {"status": "skipped", "seconds": 0.0, "reason": "in outputs.skip"}
            log(f"[outputs] {name}: skipped (in outputs.skip)")
            continue
        missing = next((r for r in module.requires if r not in ctx.products), None)
        if missing is not None:
            status = records.get(missing, {}).get("status")
            reason = f"requires {missing}, which was {status}" if status != "failed" else f"requires {missing}, which failed without a product"
            records[name] = {"status": "skipped", "seconds": 0.0, "reason": reason}
            log(f"[outputs] {name}: skipped ({reason})")
            continue
        started = time.perf_counter()
        ctx._module, ctx._steps, ctx._fail_fast, ctx._used_draws = name, {}, fail_fast, False
        error_text = None
        try:
            reason = module.active(ctx)
            if reason is None:
                if module.checkpoint and checkpoint is not None and not checkpointed:
                    checkpointed = True
                    _checkpoint(checkpoint, ctx, fail_fast)
                    started = time.perf_counter()
                ctx.products[name] = module.run(ctx)
        except Exception as error:
            reason, error_text = None, f"{type(error).__name__}: {error}"
            seconds = time.perf_counter() - started
            # Once per error: a step (or the checkpoint) logged its traceback, and ModuleUnavailable follows logged steps.
            told = error is ctx._logged or isinstance(error, ModuleUnavailable)
            if not told:                # the module's own error: its traceback (the closing line below names it)
                log(f"[outputs] {name}: {error_text}\n{traceback.format_exc().rstrip()}")
            if fail_fast:
                log(f"[outputs] {name}: FAILED after {seconds:.1f}s ({error_text})")
                records[name] = {"status": "failed", "seconds": seconds, "error": error_text, "steps": ctx._steps,
                                 **_draws_record(ctx)}
                ctx._module, ctx._steps = None, {}
                raise
        finally:
            steps, ctx._module, ctx._steps = ctx._steps, None, {}
        if reason is not None:
            records[name] = {"status": "inactive", "seconds": 0.0, "reason": str(reason)}
            log(f"[outputs] {name}: inactive ({reason})")
            continue
        seconds = time.perf_counter() - started
        failed_steps = [step for step, record in steps.items() if record["status"] == "failed"]
        if error_text is None and failed_steps:
            error_text = f"failed steps: {', '.join(failed_steps)}"
        records[name] = {"status": "failed" if error_text else "done", "seconds": seconds, "steps": steps, **_draws_record(ctx)}
        if error_text:
            records[name]["error"] = error_text
        log(f"[outputs] {name}: {'FAILED (' + error_text + ')' if error_text else 'done'} in {seconds:.1f}s")
    failed = [name for name, record in records.items() if record["status"] == "failed"]
    if failed:
        log(f"[outputs] failed modules: {', '.join(failed)} (tracebacks above)")
    return records


def _draws_record(ctx: OutputContext) -> dict[str, int]:
    """The reporting draws the current module used, ``{"reporting_draws": ..., "image_draws": ...}``, or {}."""
    if not ctx._used_draws or ctx._draws is None:
        return {}
    return {"reporting_draws": len(ctx._draws), "image_draws": len(ctx._draws.image_indices)}


def _checkpoint(checkpoint: Callable[[], None], ctx: OutputContext, fail_fast: bool) -> None:
    """Call ``checkpoint``; an error is logged and does not stop the pass (unless ``fail_fast``)."""
    try:
        checkpoint()
    except Exception as error:
        ctx._logged = error
        ctx.log(f"[outputs] run summary before the imaging module not written: {type(error).__name__}: {error}\n"
            f"{traceback.format_exc().rstrip()}")
        if fail_fast:
            raise
