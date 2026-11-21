"""The framework of the outputs stage: the context every module reads, the module base class and the runner.

A module (:class:`OutputModule`) works in three phases.  ``compute`` evaluates what the module needs (the lens of the
reporting draws, the image solver) and writes every result to the stage's store (``results.h5``, ``cache.h5``);
``product`` loads from the store what the modules that require this one read (``ctx.products[name]``); ``render``
draws the figures from the store alone.  Render never touches the evaluator or the reporting draws, so a finished
run's figures can be drawn again without evaluating a lens (``kappaccino outputs RUN --render``).  A pass runs in one
of three modes (:data:`MODES`): ``full`` (compute, product, render), ``compute`` (compute and product, no figures) and
``render`` (product and render, from what an earlier compute saved).  Even a full pass draws from the store, so the
render path is the only path.

Each phase is a sequence of named steps (:meth:`OutputContext.step`).  A step that raises is logged with its traceback
and recorded as failed, and the module goes on with its other steps; a step that needs the result of a failed one is
skipped with the reason (:meth:`OutputContext.skip_step`, :meth:`OutputContext.render_step`).  A module with a failed
step ends as "failed" but keeps its product; a module that cannot give its dependents what they need raises
(:class:`ModuleUnavailable` after its failed steps), and is then failed with no product.

At the start of its compute a module's completeness record ``output_module_<name>`` (in ``results.h5``) is set to
incomplete, and at the end it says whether every compute step ran (``complete``: none failed or was skipped), with the
step statuses and the reporting draws the compute used.  A render pass renders only modules whose record is complete;
any other module is failed with :class:`ModuleUnavailable` and the hint to run its compute first.  At the end of every
pass the saved records are checked for modules computed over different draws (:func:`check_draws`).

The runner (:func:`run_modules`) orders the modules by what they need (``requires``, hard: the module reads
``ctx.products`` of those) and by what they should follow (``after``, ordering only), stably with respect to the
registry order (``outputs.registry.MODULES``), and runs each one that is selected, not skipped and active.  A module
runs only when every module it requires gave its product (which may be None).  Each module's outcome, wall time,
steps and the reporting draws it used go to ``ctx.summary["modules"]``, and so to the record ``run_summary``.

The modules import matplotlib, the plotting and diagnostics modules and the image solver inside their phases, so
importing the registry (``config.validate`` does, to check module names) loads none of them.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REBUILT_FIELD_WARNING = (
    "gp_blocks = {blocks}: the posterior gave each block its own mass field, and the products rebuild one shared field "
    "from it. That field is not a posterior sample of the lens, so the image residuals, RMS, position contours and map "
    "uncertainties in the draws' own lenses are not trustworthy and likely inflated. Use an all-block run for reported numbers.")

MODES = ("full", "compute", "render")
COMPLETION_PREFIX = "output_module_"


def completion_name(module: str) -> str:
    """The name of the completeness record of ``module`` in ``results.h5``."""
    return f"{COMPLETION_PREFIX}{module}"


class ComputeOnly(RuntimeError):
    """Raised when a module asks for the evaluator or the reporting draws outside its compute (in ``product``,
    ``render``, or any phase of a render pass): those phases read only the store."""


class ModuleUnavailable(RuntimeError):
    """Raised by a module that cannot give its dependents what they need (its essential step failed, which was logged,
    or in a render pass its compute was never completed): the module is failed with no product and the modules that
    require it are skipped."""


class OutputContext:
    """What the modules of one output pass share.

    ``config``, ``state``, ``posterior`` and ``audit`` are those of the fit; ``paths`` is the final ``StagePaths``
    (``root``, ``tables``, ``figures``, ``fits``); ``mode`` is the pass mode (:data:`MODES`).  ``summary`` is the record
    ``run_summary`` being built, and ``products`` holds what each module's ``product`` returned, by module name.  The
    evaluator (:attr:`evaluator`) and the reporting draws (:attr:`draws`, built on first access, so a pass of modules
    that need none never builds them) are for compute steps only: anywhere else they raise :class:`ComputeOnly`, which
    names the module and step that asked.  :meth:`imaging` loads the imaging of each variant once per pass."""

    def __init__(self, config, state, evaluator, posterior, audit: dict[str, Any] | None, paths, *, log=print,
                 mode: str = "full"):
        if mode not in MODES:
            raise ValueError(f"output pass mode {mode!r}; known: {', '.join(MODES)}")
        self.config, self.state, self.posterior, self.mode = config, state, posterior, mode
        self._evaluator = evaluator
        self.audit = audit or {}
        self.paths, self.log = paths, log
        self.summary: dict[str, Any] = {}
        self.products: dict[str, Any] = {}
        self._draws = None
        self._draws_error: Exception | None = None    # a failed build, raised again on every later access
        self._used_draws = False                      # the current module touched the draws
        self._logged: Exception | None = None         # an error a step already logged with its traceback
        self._imaging: list[tuple[Any, Any]] = []
        self._module: OutputModule | None = None      # the module running, its phase and step, its step records
        self._phase: str | None = None
        self._step: str | None = None
        self._steps: dict[str, dict] = {}             # the compute steps
        self._render_steps: dict[str, dict] = {}
        self._completion: dict[str, Any] | None = None  # the completeness record render and product read
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

    def _compute_only(self, what: str) -> None:
        """Raise :class:`ComputeOnly` unless the lens may be evaluated here: in a module's compute, or outside the
        modules, in a pass that is not a render pass."""
        if self.mode != "render" and (self._phase == "compute" or self._module is None):
            return
        where = "outside any module" if self._module is None else (
            f"{self._module.name}.{self._step}" if self._step else f"{self._module.name} ({self._phase})")
        raise ComputeOnly(f"{where} asked for ctx.{what} in the {self._phase or 'no'} phase of a {self.mode} pass: only "
                          "compute steps may evaluate the lens; product and render read the store "
                          "(save what this needs from compute)")

    @property
    def evaluator(self):
        """The evaluator of the fit; compute steps only (:class:`ComputeOnly` elsewhere)."""
        self._compute_only("evaluator")
        return self._evaluator

    @property
    def draws(self):
        """The reporting draws (``reporting.draws.reporting_draws``), built on first access with their grid maps cached
        in the stage's ``cache.h5``; compute steps only (:class:`ComputeOnly` elsewhere).  Building them sets
        ``summary["reporting_draws"]`` and, for a field rebuilt from per-block fields, ``summary["warnings"]``.  A
        build that fails raises the same error on every later access."""
        self._compute_only("draws")
        self._used_draws = True
        if self._draws_error is not None:
            raise self._draws_error
        if self._draws is None:
            from ..reporting.draws import reporting_draws
            try:
                draws = reporting_draws(self.config, self._evaluator, self.posterior)
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

    def imaging(self, image_config, *, required: bool = False):
        """The imaging (``imaging.cutouts.load_cluster_imaging``) of the variant ``image_config``, loaded once per pass;
        variants that differ only in ``figure_suffix`` share it.  When the imaging folder is missing: None, logged once,
        which leaves the other variants and outputs alone (a compute leaves the variant out); with ``required`` (a
        render of a variant its compute drew) a FileNotFoundError that names the folder and suggests ``--remap``."""
        key = replace(image_config, figure_suffix="")
        known = next((entry for entry in self._imaging if entry[0] == key), None)
        if known is None:
            from ..imaging.cutouts import load_cluster_imaging
            try:
                known = (key, load_cluster_imaging(image_config, self.config.model, self.config.paths.run_name), None)
            except FileNotFoundError as error:
                known = (key, None, error)
                if not required:            # missing external imaging must not prevent the other outputs
                    self.log(f"[outputs] cluster mass figure and covariance RGB cutouts skipped: {error}")
            self._imaging.append(known)
        _, imaging, error = known
        if imaging is None and required:
            from .saved_run import REMAP_HINT
            raise FileNotFoundError(f"the last compute drew this imaging variant, but its imaging cannot be loaded from "
                                    f"image_catalog.image_dir = {image_config.image_dir} ({error}); {REMAP_HINT}")
        return imaging

    def _records(self) -> dict[str, dict]:
        return self._render_steps if self._phase == "render" else self._steps

    def _label(self, name: str) -> str:
        return f"{self._module.name}.{name}" + (" (render)" if self._phase == "render" else "")

    def step(self, name: str, fn: Callable, *args, **kwargs):
        """Run one step of the current module's phase, ``fn(*args, **kwargs)``, and return its result.  The step's
        status and wall time go to the module's record (``steps`` for compute, ``render_steps`` for render) and the
        log (``[outputs] <module>.<step>``).  A compute step that returns None is recorded as having written nothing
        (``wrote`` False), so render skips what depends on it (:meth:`computed`).  An exception (not
        KeyboardInterrupt or SystemExit) is logged with its traceback and recorded, and the step returns None so that
        the module goes on; :meth:`ok` tells a failed step from one that returned None.  With ``fail_fast`` the
        exception is raised again.

        Raises
        ------
        RuntimeError
            Outside a module's compute or render, or for a step the module does not declare in ``compute_steps`` or
            ``render_steps`` (its part before any ``:``).
        """
        module = self._module
        if module is None or self._phase not in ("compute", "render"):
            raise RuntimeError("OutputContext.step called outside a module's compute or render")
        self._check_declared(name, self._phase)
        label, records = self._label(name), self._records()
        started = time.perf_counter()
        self._step = name
        try:
            result = fn(*args, **kwargs)
        except Exception as error:
            seconds = time.perf_counter() - started
            records[name] = {"status": "failed", "seconds": seconds, "error": f"{type(error).__name__}: {error}"}
            self.log(f"[outputs] {label}: FAILED after {seconds:.1f}s: {type(error).__name__}: {error}\n"
                     f"{traceback.format_exc().rstrip()}")
            if self._fail_fast:
                self._logged = error
                raise
            return None
        finally:
            self._step = None
        seconds = time.perf_counter() - started
        records[name] = {"status": "done", "seconds": seconds}
        if self._phase == "compute":
            records[name]["wrote"] = result is not None
        elif name.split(":")[0] in module.input_steps:
            records[name]["input"] = True         # loaded what the figures read, drew nothing (rendered_anything)
        self.log(f"[outputs] {label}: done in {seconds:.1f}s")
        return result

    def _check_declared(self, name: str, phase: str) -> None:
        """Raise RuntimeError unless ``name`` (its part before any ``:``) is a declared step of ``phase`` of the
        current module."""
        module = self._module
        declared = module.compute_steps if phase == "compute" else module.render_steps
        if name.split(":")[0] not in declared:
            raise RuntimeError(f"output module {module.name!r}: {phase} step {name!r} is not in its "
                               f"{phase}_steps {list(declared)}")

    def ok(self, name: str) -> bool:
        """Whether the step ``name`` of the current module's phase ran without an error."""
        return self._records().get(name, {}).get("status") == "done"

    def skip_step(self, name: str, reason: str) -> None:
        """Record the step ``name`` of the current module's phase as skipped, with the reason (logged).  Only for a
        step that cannot run because another one failed: in a compute, a skipped step makes the compute incomplete
        (so its module cannot be rendered until it is computed again).  A step that does not apply to this run runs
        and returns None instead (``wrote`` false), and render leaves its figure out."""
        self._check_declared(name, self._phase)
        self._records()[name] = {"status": "skipped", "seconds": 0.0, "reason": reason}
        self.log(f"[outputs] {self._label(name)}: skipped ({reason})")

    @property
    def completion(self) -> dict[str, Any]:
        """The current module's completeness record: the one its compute just wrote, or in a render pass the saved
        one.  Product and render only."""
        if self._completion is None:
            raise RuntimeError("OutputContext.completion is read by product and render, after the compute")
        return self._completion

    def computed(self, name: str) -> bool:
        """Whether the compute step ``name`` of the current module ran and wrote its entries (did not return None),
        according to :attr:`completion`."""
        record = self.completion.get("steps", {}).get(name, {})
        return record.get("status") == "done" and bool(record.get("wrote"))

    def render_step(self, name: str, fn: Callable, *args, needs: str | Sequence[str] | None = None, **kwargs):
        """:meth:`step` for a render step that draws what the compute steps ``needs`` (default: the compute step of the
        same name) saved.  When one of them failed or was skipped, the step is skipped with the reason; when one ran
        but wrote nothing (it did not apply), the step is left out silently; otherwise it runs.

        Raises
        ------
        RuntimeError
            Outside a module's render, for a ``name`` not in ``render_steps`` or a ``needs`` not in ``compute_steps``
            (each by its part before any ``:``): a misdeclared step is an error, never a silent skip.
        """
        needs = (name,) if needs is None else ((needs,) if isinstance(needs, str) else tuple(needs))
        if self._module is None or self._phase != "render":
            raise RuntimeError("OutputContext.render_step called outside a module's render")
        self._check_declared(name, "render")
        for need in needs:
            self._check_declared(need, "compute")
        steps = self.completion.get("steps", {})
        for need in needs:
            status = steps.get(need, {}).get("status", "not run")
            if status != "done":
                self.skip_step(name, f"compute step {need} {status}")
                return None
        if not all(self.computed(need) for need in needs):
            return None
        return self.step(name, fn, *args, **kwargs)


class OutputModule:
    """One group of output products: compute steps that write results to the store, a product for the modules that
    require this one, and render steps that draw the figures from the store (see the module docstring).

    Subclasses set ``name`` and, when needed, ``requires`` (modules whose products this one reads from
    ``ctx.products``; the module runs only when each of them gave its product) and ``after`` (modules it should follow
    when both run, for example because it reads a table they write).  ``checkpoint`` marks the imaging module: the run
    summary and the results macros are written before it runs, because it is the slowest and most failure-prone part
    of the stage.  ``description`` is the one line of ``kappaccino outputs --list``; ``compute_steps`` and
    ``render_steps`` name the steps of each phase (a step's name before any ``:``, for steps named per variant), and
    ``input_steps`` those render steps that only load what the figures read (the imaging), which do not count as
    having drawn something (:func:`rendered_anything`).  ``summary_keys`` are the top-level keys of ``run_summary``
    the module writes, from its compute: a later full or compute pass that runs the module drops them before merging,
    so a failed module leaves no old numbers behind, and a render pass keeps them.  ``path_fields`` are the configuration fields (``section.field``) of the files the
    compute reads, checked before a rerun (``outputs.command``)."""

    name: str = ""
    requires: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    checkpoint: bool = False
    description: str = ""
    compute_steps: tuple[str, ...] = ()
    render_steps: tuple[str, ...] = ()
    input_steps: tuple[str, ...] = ()
    summary_keys: tuple[str, ...] = ()
    path_fields: tuple[str, ...] = ()

    def active(self, ctx: OutputContext) -> str | None:
        """None when the module applies to this run, else a short reason why not (logged, recorded as inactive).
        Decided from ``ctx.config`` and, at run time, the products of the modules it requires (absent when listing)."""
        return None

    def compute(self, ctx: OutputContext) -> None:
        """Evaluate what the module needs and write every result to the store, as compute steps; no figures.  Raise
        :class:`ModuleUnavailable` when the dependents cannot run."""
        raise NotImplementedError

    def product(self, ctx: OutputContext) -> Any:
        """``ctx.products[self.name]``, loaded from the store (never passed in memory from :meth:`compute`); None by
        default.  A missing store entry (``store.MissingEntry``) fails the module with :class:`ModuleUnavailable`."""
        return None

    def render(self, ctx: OutputContext) -> None:
        """Draw the figures (and write any purely presentational table) as render steps, reading only the store, the
        configuration, ``ctx.state`` (the catalogs), ``ctx.posterior`` and :attr:`OutputContext.completion`; never
        ``ctx.evaluator`` nor ``ctx.draws``."""

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
    """Run the output modules (default: the registry, ``outputs.registry.MODULES``) on ``ctx`` in its mode
    (``ctx.mode``): per module compute, product and render in a full pass, compute and product in a compute pass,
    product and render in a render pass.

    ``only`` selects modules plus whatever they require (in a render pass a module added that way gives only its
    product); ``skip`` removes modules, and a module that requires a skipped, inactive or failed-without-product one is
    skipped too.  A failing step is recorded and the module goes on (:meth:`OutputContext.step`); an error outside the
    steps fails the module, and the other modules still run.  ``fail_fast`` re-raises the first error instead.
    ``checkpoint`` is called once, before the first module with ``checkpoint`` set that runs.  Every module of the
    plan gets a record in ``ctx.summary["modules"]``: ``status`` ("done", "inactive", "skipped" or "failed"),
    ``mode``, ``seconds``, ``steps`` (each compute step's ``status``, ``seconds``, ``wrote`` and ``error`` or
    ``reason``; in a render pass those of the saved completeness record), ``render_steps`` when it rendered,
    ``reporting_draws`` and ``image_draws`` when its compute used the draws, and ``reason`` or ``error``.  Returns
    those records."""
    if modules is None:
        from .registry import MODULES as modules
    skip = tuple(skip)
    plan, added = plan_modules(modules, only=only, skip=skip)
    log = ctx.log
    if added:
        log("[outputs] modules added because a selected module requires them: "
            + ", ".join(f"{name} (by {by})" for name, by in added.items()))
    log(f"[outputs] {ctx.mode} pass; modules: {', '.join(m.name for m in plan) or 'none'}")
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
            required = records.get(missing, {})
            status = required.get("status")
            reason = f"requires {missing}, which was {status}" if status != "failed" else f"requires {missing}, which failed without a product"
            records[name] = {"status": "skipped", "mode": ctx.mode, "seconds": 0.0, "reason": reason}
            if status == "failed" or "failed_requirement" in required:
                # Not the user's choice: what this module would have computed from is gone (merged_summary).
                records[name]["failed_requirement"] = required.get("failed_requirement", missing)
            log(f"[outputs] {name}: skipped ({reason})")
            continue
        started = time.perf_counter()
        ctx._module, ctx._phase, ctx._steps, ctx._render_steps = module, None, {}, {}
        ctx._completion, ctx._fail_fast, ctx._used_draws = None, fail_fast, False
        error_text = None
        try:
            reason = module.active(ctx)
            if reason is None:
                if module.checkpoint and checkpoint is not None and not checkpointed:
                    checkpointed = True
                    _checkpoint(checkpoint, ctx, fail_fast)
                    started = time.perf_counter()
                _run_phases(ctx, module, product_only=ctx.mode == "render" and name in added)
        except Exception as error:
            reason, error_text = None, f"{type(error).__name__}: {error}"
            seconds = time.perf_counter() - started
            # Once per error: a step (or the checkpoint) logged its traceback, and ModuleUnavailable follows logged steps.
            told = error is ctx._logged or isinstance(error, ModuleUnavailable)
            if not told:                # the module's own error: its traceback (the closing line below names it)
                log(f"[outputs] {name}: {error_text}\n{traceback.format_exc().rstrip()}")
            if fail_fast:
                log(f"[outputs] {name}: FAILED after {seconds:.1f}s ({error_text})")
                records[name] = {"status": "failed", "mode": ctx.mode, "seconds": seconds, "error": error_text,
                                 **_step_records(ctx), **_draws_record(ctx)}
                ctx._module, ctx._phase, ctx._steps, ctx._render_steps = None, None, {}, {}
                raise
        finally:
            step_records, draws_record = _step_records(ctx), _draws_record(ctx)
            ctx._module, ctx._phase, ctx._steps, ctx._render_steps, ctx._completion = None, None, {}, {}, None
        if reason is not None:
            records[name] = {"status": "inactive", "seconds": 0.0, "reason": str(reason)}
            log(f"[outputs] {name}: inactive ({reason})")
            continue
        seconds = time.perf_counter() - started
        failed_steps = [step for step, record in step_records.get("steps", {}).items() if record["status"] == "failed"]
        failed_steps += [f"{step} (render)" for step, record in step_records.get("render_steps", {}).items()
                         if record["status"] == "failed"]
        if error_text is None and failed_steps:
            error_text = f"failed steps: {', '.join(failed_steps)}"
        records[name] = {"status": "failed" if error_text else "done", "mode": ctx.mode, "seconds": seconds,
                         **step_records, **draws_record}
        if error_text:
            records[name]["error"] = error_text
        log(f"[outputs] {name}: {'FAILED (' + error_text + ')' if error_text else 'done'} in {seconds:.1f}s")
    check_draws(ctx, modules)
    if ctx.mode == "render" and not rendered_anything(records):
        log("[outputs] WARNING: this render pass drew nothing: no selected module with a completed compute drew a "
            "figure (run a full or --compute pass first)")
    failed = [name for name, record in records.items() if record["status"] == "failed"]
    if failed:
        log(f"[outputs] failed modules: {', '.join(failed)} (tracebacks above)")
    return records


def rendered_anything(records: dict[str, dict]) -> bool:
    """Whether any module of a pass's ``records`` ran a render step that draws (not one of its ``input_steps``,
    which only load what the figures read) without failing."""
    return any(step.get("status") == "done" and not step.get("input") for record in records.values()
               for step in (record.get("render_steps") or {}).values())


DRAWS_WARNING = "modules computed over different reporting draws"


def check_draws(ctx: OutputContext, modules: Sequence[OutputModule]) -> str | None:
    """Warn when the saved results of ``modules`` come from different reporting draws: from their completeness records
    in ``ctx.stage``, the modules whose compute used the draws, grouped by ``draws_key``.  A module whose required
    module was computed later with other draws is named as stale.  The warning (logged as ``[outputs] WARNING`` and
    added to ``ctx.summary["warnings"]``) never blocks the pass; returns it, or None when the draws agree."""
    from ..io import store
    saved = {}
    for module in modules:
        try:
            record = store.read_record(ctx.stage, completion_name(module.name))
        except (store.MissingEntry, store.DamagedEntry, store.StoreError):
            continue
        if isinstance(record, dict) and isinstance(record.get("draws_key"), str):
            saved[module.name] = record
    if len({record["draws_key"] for record in saved.values()}) < 2:
        return None
    groups: dict[str, list[str]] = {}
    for name, record in saved.items():
        groups.setdefault(record["draws_key"], []).append(name)
    described = "; ".join(", ".join(names) + f" ({saved[names[0]].get('reporting_draws', '?')} draws, seed "
                          f"{saved[names[0]].get('draws_seed', '?')}, key {key[:8]})" for key, names in groups.items())
    stale = [f"{module.name} (requires {required})" for module in modules if module.name in saved
             for required in module.requires
             if required in saved and saved[required]["draws_key"] != saved[module.name]["draws_key"]
             and str(saved[required].get("computed_at", "")) > str(saved[module.name].get("computed_at", ""))]
    warning = f"{DRAWS_WARNING}: {described}"
    if stale:
        warning += (f". Stale, because a module they require was computed again with other draws since: "
                    f"{', '.join(stale)}")
    warning += ". Compute them together (one pass over the same draws) for consistent numbers and figures"
    ctx.summary.setdefault("warnings", []).append(warning)
    ctx.log(f"[outputs] WARNING: {warning}")
    return warning


def _run_phases(ctx: OutputContext, module: OutputModule, *, product_only: bool) -> None:
    """Run the phases of ``module`` that ``ctx.mode`` asks for and set its product."""
    name = module.name
    if ctx.mode == "render":
        ctx._completion = load_completion(ctx.stage, name)
    else:
        ctx._phase = "compute"
        _mark_incomplete(ctx, name)
        try:
            module.compute(ctx)
        except Exception:
            try:                        # the steps that ran, still incomplete; the marker above already says so
                _write_completion(ctx, name, complete=False)
            except Exception as error:
                ctx.log(f"[outputs] {name}: completeness record not updated: {type(error).__name__}: {error}")
            raise
        failed = [step for step, record in ctx._steps.items() if record["status"] != "done"]
        ctx._completion = _write_completion(ctx, name, complete=not failed)
    from ..io import store
    ctx._phase = "product"
    try:
        ctx.products[name] = module.product(ctx)
    except store.MissingEntry as error:
        raise ModuleUnavailable(f"{error}; {_compute_hint(name)}") from error
    if ctx.mode != "compute" and not product_only:
        ctx._phase = "render"
        module.render(ctx)


def _compute_hint(name: str) -> str:
    return f"run its compute first: kappaccino outputs RUN --only {name} (or --compute --only {name})"


def load_completion(stage, name: str) -> dict[str, Any]:
    """The completeness record of module ``name`` in ``stage``.

    Raises
    ------
    ModuleUnavailable
        When there is none, it cannot be read, or the last compute of the module did not complete; the message tells
        to run the compute first.
    """
    from ..io import store
    try:
        record = store.read_record(stage, completion_name(name))
    except (store.MissingEntry, store.DamagedEntry, store.StoreError) as error:
        raise ModuleUnavailable(f"no record {completion_name(name)} of a completed compute ({type(error).__name__}); "
                                f"{_compute_hint(name)}") from error
    if not isinstance(record, dict) or not record.get("complete"):
        failed = [step for step, r in (record.get("steps") or {}).items() if r.get("status") != "done"] if isinstance(record, dict) else []
        detail = f"steps not done: {', '.join(failed)}" if failed else "it did not finish"
        raise ModuleUnavailable(f"the last compute of {name} is incomplete ({detail}); {_compute_hint(name)}")
    return record


def _mark_incomplete(ctx: OutputContext, name: str) -> None:
    """Set the module's completeness record to incomplete before its compute writes anything, so that a compute that
    stops half way never leaves a record of an earlier complete one."""
    from ..io import store
    store.write_record(ctx.stage, completion_name(name),
                       {"module": name, "complete": False, "started_at": datetime.now(timezone.utc).isoformat()})


def _write_completion(ctx: OutputContext, name: str, *, complete: bool) -> dict[str, Any]:
    """Write and return the module's completeness record after its compute: ``complete`` (every compute step done),
    when, the compute steps and the reporting draws the compute used (``reporting_draws``, ``image_draws``,
    ``draws_seed``, ``draws_key``)."""
    from ..io import store
    record = {"module": name, "complete": bool(complete), "computed_at": datetime.now(timezone.utc).isoformat(),
              "steps": ctx._steps, **_draws_record(ctx)}
    if ctx._used_draws and ctx._draws is not None:
        record["draws_seed"] = int(ctx._draws.seed)
        key = getattr(ctx._draws, "key", None)
        if isinstance(key, str):
            record["draws_key"] = key
    store.write_record(ctx.stage, completion_name(name), record, default=float)
    return record


def _step_records(ctx: OutputContext) -> dict[str, dict]:
    """``steps`` (and ``render_steps`` when the module rendered) of the current module; in a render pass ``steps`` are
    the compute steps of the saved completeness record."""
    if ctx.mode == "render" and ctx._completion is not None:
        out = {"steps": ctx._completion.get("steps", {})}
    else:
        out = {"steps": ctx._steps}
    if ctx._render_steps:
        out["render_steps"] = ctx._render_steps
    return out


def _draws_record(ctx: OutputContext) -> dict[str, int]:
    """The reporting draws the current module's compute used, ``{"reporting_draws": ..., "image_draws": ...}``, or {};
    in a render pass those of its saved completeness record."""
    if ctx.mode == "render":
        saved = ctx._completion or {}
        return {key: saved[key] for key in ("reporting_draws", "image_draws") if key in saved}
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
