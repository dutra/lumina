"""The framework of the outputs stage: the context every module reads, the module base class and the runner.

A module (:class:`OutputModule`) works in three phases.  ``compute`` evaluates what the module needs (the lens of the
reporting draws, the image solver) and writes every result to the stage's store (``results.h5``, ``cache.h5``);
``product`` loads from the store what the modules that require this one read (``ctx.products[name]``); ``render``
draws the figures from the store alone.  Render never touches the evaluator or the reporting draws, so a finished
run's figures can be drawn again without evaluating a lens (``lumina outputs RUN --render``).  A pass runs in one
of three modes (:data:`MODES`): ``full`` (compute, product, render), ``compute`` (compute and product, no figures) and
``render`` (product and render, from what an earlier compute saved).  Even a full pass draws from the store, so the
render path is the only path.

Each phase is a sequence of named steps (:meth:`OutputContext.step`).  A step that raises is logged with its traceback
and recorded as failed, and the module goes on with its other steps; a step that needs the result of a failed one is
skipped with the reason (:meth:`OutputContext.skip_step`, :meth:`OutputContext.render_step`).  A module with a failed
step ends as "failed" but keeps its product; a module that cannot give its dependents what they need raises
(:class:`ModuleUnavailable` after its failed steps), and is then failed with no product, after its render (in a full
pass) has drawn the figures whose compute steps succeeded.

At the start of its compute a module's completeness record ``output_module_<name>`` (in ``results.h5``) is set to
incomplete, and at the end it holds the step statuses, whether every compute step ran (``complete``: none failed or
was skipped; for information), whether the compute gave no product (``unavailable``: the error it raised, else None)
and the reporting draws the compute used.  Completeness is per step: a render step is drawn when the compute step it
needs finished and wrote in the module's last compute, whatever the other steps did (a step that did not run in that
compute, after an error stopped it, is reported as not drawn).  A render pass refuses a module
(:class:`NotComputed`) only when it has no record of a compute with its steps (never computed here, or interrupted
before the end); otherwise it draws what it can, and the module ends as failed when a compute step of its last compute
failed or a render step could not be drawn.  The record also keeps the outputs settings of the compute
(:func:`settings_snapshot`).  At the end of every pass the saved records are checked for modules computed over
different draws (:func:`check_draws`) and for modules computed with other settings than the current ones
(:func:`check_settings`); both only warn.

The runner (:func:`run_modules`) orders the modules by what they need (``requires``, hard: the module reads
``ctx.products`` of those) and by what they should follow (``after``, ordering only), stably with respect to the
registry order (``outputs.registry.MODULES``), and runs each one that is selected, not skipped, not disabled and
active.  Each module has a section of the configuration under ``outputs`` by its name (``config.OutputsConfig``), with
its ``enabled`` switch and its own settings; the module reads it as :attr:`OutputContext.settings`.  A module runs only
when every module it requires gave its product (which may be None).  Each module's outcome, wall time,
steps and the reporting draws it used go to ``ctx.summary["modules"]``, and so to the record ``run_summary``.

The modules import matplotlib, the plotting and diagnostics modules and the image solver inside their phases, so
importing the registry (``config.validate`` does, to check module names) loads none of them.
"""
from __future__ import annotations

import json
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
    or in a render pass its last compute gave no product or never recorded its steps): the module is failed with no
    product and the modules that require it are skipped."""


class NotComputed(ModuleUnavailable):
    """Raised in a render pass for a module with no record of a compute that recorded its steps (never computed in this
    stage, or interrupted before the end of its compute): there is nothing to draw from."""


class OutputContext:
    """What the modules of one output pass share.

    ``config``, ``state``, ``posterior`` and ``audit`` are those of the fit; ``paths`` is the final ``StagePaths``
    (``root``, ``tables``, ``figures``, ``fits``); ``mode`` is the pass mode (:data:`MODES`).  ``summary`` is the record
    ``run_summary`` being built, and ``products`` holds what each module's ``product`` returned, by module name;
    :attr:`settings` is the running module's section of ``config.outputs``.  The
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
        self._notes: list[str] | None = None          # what the running step noted (:meth:`note`)
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
    def settings(self):
        """The running module's own section of the configuration, ``config.outputs.<module name>`` (in ``active`` too).

        Raises
        ------
        RuntimeError
            Outside a module.
        """
        if self._module is None:
            raise RuntimeError("OutputContext.settings is the running module's section: read it inside a module")
        return getattr(self.config.outputs, self._module.name)

    def inactive_reason(self, module: "OutputModule") -> str | None:
        """``module.active(self)`` with ``module`` as the running module, so that it reads its own :attr:`settings`
        (for listing and checking modules outside a pass)."""
        previous, self._module = self._module, module
        try:
            return module.active(self)
        finally:
            self._module = previous

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
            if Path(image_config.image_dir) != Path(self.config.outputs.rgb.image_dir):
                # The extra JWST variant: imaging.cutouts.imaging_variants builds its folder, no field names it.
                raise FileNotFoundError(
                    f"the last compute drew this imaging variant, but its imaging cannot be loaded from "
                    f"{Path(image_config.image_dir).absolute()} ({error}); that folder ({image_config.image_dir}) is "
                    f"built by imaging.cutouts.imaging_variants relative to the working directory, not read from a "
                    f"configuration field, so --remap cannot move it: run the command from the directory that "
                    f"contains {image_config.image_dir}")
            from .saved_run import REMAP_HINT
            raise FileNotFoundError(f"the last compute drew this imaging variant, but its imaging cannot be loaded from "
                                    f"outputs.rgb.image_dir = {image_config.image_dir} ({error}); {REMAP_HINT}")
        return imaging

    def _records(self) -> dict[str, dict]:
        return self._render_steps if self._phase == "render" else self._steps

    def _label(self, name: str) -> str:
        return f"{self._module.name}.{name}" + (" (render)" if self._phase == "render" else "")

    def step(self, name: str, fn: Callable, *args, **kwargs):
        """Run one step of the current module's phase, ``fn(*args, **kwargs)``, and return its result.  The step's
        status and wall time go to the module's record (``steps`` for compute, ``render_steps`` for render) and the
        log (``[outputs] <module>.<step>``).  A compute step that returns None is recorded as having written nothing
        (``wrote`` False), so render skips what depends on it (:meth:`computed`).  What the step noted
        (:meth:`note`) goes to its record as ``notes``.  An exception (not KeyboardInterrupt or SystemExit) is logged
        with its traceback and recorded, and the step returns None so that the module goes on; :meth:`ok` tells a
        failed step from one that returned None.  With ``fail_fast`` the exception is raised again.

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
        self._step, self._notes = name, []
        input_step = self._phase == "render" and name.split(":")[0] in module.input_steps
        try:
            result = fn(*args, **kwargs)
        except Exception as error:
            seconds = time.perf_counter() - started
            records[name] = {"status": "failed", "seconds": seconds, "error": f"{type(error).__name__}: {error}",
                             **({"notes": self._notes} if self._notes else {}), **({"input": True} if input_step else {})}
            self.log(f"[outputs] {label}: FAILED after {seconds:.1f}s: {type(error).__name__}: {error}\n"
                     f"{traceback.format_exc().rstrip()}")
            if self._fail_fast:
                self._logged = error
                raise
            return None
        finally:
            notes, self._step, self._notes = self._notes, None, None
        seconds = time.perf_counter() - started
        records[name] = {"status": "done", "seconds": seconds}
        if self._phase == "compute":
            records[name]["wrote"] = result is not None
        elif input_step:
            records[name]["input"] = True         # loaded what the figures read, drew nothing (rendered_anything)
        if notes:
            records[name]["notes"] = notes
        self.log(f"[outputs] {label}: done in {seconds:.1f}s")
        return result

    def note(self, message: str) -> None:
        """Log ``message`` for the running step (``[outputs] <module>.<step>: <message>``) and add it to the step's
        record (``notes``): for a step that goes on without something it would have used, so that the omission
        leaves a trace.

        Raises
        ------
        RuntimeError
            Outside a step.
        """
        if self._notes is None:
            raise RuntimeError("OutputContext.note is for the running step: call it inside ctx.step")
        self._notes.append(message)
        self.log(f"[outputs] {self._label(self._step)}: {message}")

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
        step that cannot run because another one failed: in a compute, the render steps that need a skipped step are
        skipped too (not drawn), and the compute is recorded as incomplete.  A step that does not apply to this run
        runs and returns None instead (``wrote`` false), and render leaves its figure out."""
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
        record = self.completion["steps"].get(name)              # None: the step did not run
        return record is not None and record["status"] == "done" and record["wrote"]

    def wrote_nothing(self, name: str) -> bool:
        """Whether the compute step ``name`` of the current module ran without an error and wrote nothing (it did not
        apply), according to :attr:`completion`; False for a step that failed, was skipped or did not run.  Render
        reads the steps through this, :meth:`computed` and :meth:`render_step`, never ``completion["steps"][name]``:
        a compute stopped by an error (``fail_fast``) records only the steps that ran."""
        record = self.completion["steps"].get(name)
        return record is not None and record["status"] == "done" and not record["wrote"]

    def render_step(self, name: str, fn: Callable, *args, needs: str | Sequence[str] | None = None, **kwargs):
        """:meth:`step` for a render step that draws what the compute steps ``needs`` (default: the compute step of the
        same name) saved in the module's last compute, whatever its other steps did.  When one of them failed, was
        skipped or did not run, the step is skipped with the reason (it could not be drawn); when one ran but wrote
        nothing (it did not apply), the step is left out silently; otherwise it runs.

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
        steps = self.completion["steps"]
        for need in needs:
            if need not in steps:
                self.skip_step(name, f"compute step {need} did not run")
                return None
            if steps[need]["status"] != "done":
                self.skip_step(name, f"compute step {need} {steps[need]['status']}")
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
    of the stage.  ``description`` is the one line of ``lumina outputs --list``; ``compute_steps`` and
    ``render_steps`` name the steps of each phase (a step's name before any ``:``, for steps named per variant), and
    ``input_steps`` those render steps that only load what the figures read (the imaging), which do not count as
    having drawn something (:func:`rendered_anything`).  ``summary_keys`` are the top-level keys of ``run_summary``
    the module writes, from its compute: a later full or compute pass that runs the module drops them before merging,
    so a failed module leaves no old numbers behind, and a render pass keeps them.  ``path_fields`` are the fields of
    the files the compute reads, as dotted names under ``outputs`` (``rgb.image_dir``, ``truth.kappa_true_fits``),
    checked before a rerun (``outputs.command``).  The module's settings are the section ``config.outputs.<name>``
    (:attr:`OutputContext.settings`), whose ``enabled`` switch the runner reads; ``name`` must be the field name of
    that section.  ``shared_settings`` are the fields of ``outputs`` outside any module's section that the compute
    reads (``draws``, ``image_draws``, ``caustic_source_redshift``, ``image_solver``; for a module that reads the
    ``images`` product, ``image_solver``, which made it).  The completeness record keeps the settings the compute used
    (:func:`settings_snapshot`: these, the module's own section and its ``path_fields`` in other sections), and
    :func:`check_settings` compares them with the current ones, except ``render_settings``: the fields under ``outputs``
    (dotted, ``rgb.display``) that only render reads from the current configuration, which change no saved result."""

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
    shared_settings: tuple[str, ...] = ()
    render_settings: tuple[str, ...] = ()

    def active(self, ctx: OutputContext) -> str | None:
        """None when the module applies to this run, else a short reason why not (logged, recorded as inactive).
        Decided from ``ctx.settings`` (or another part of ``ctx.config``) and, at run time, the products of the modules
        it requires (absent when listing)."""
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


def disabled_modules(config, modules: Sequence[OutputModule] | None = None) -> tuple[str, ...]:
    """The names of the ``modules`` (default: the registry) whose section ``config.outputs.<name>`` has ``enabled``
    false."""
    if modules is None:
        from .registry import MODULES as modules
    return tuple(m.name for m in modules if not getattr(config.outputs, m.name).enabled)


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
    for label, names in (("--only", only or ()), ("--skip", skip)):
        unknown = [n for n in names if n not in by_name]
        if unknown:
            raise ValueError(f"{label} names unknown output modules {unknown}; known: {list(by_name)}")
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
                skip: Iterable[str] = (), disabled: Iterable[str] = (), fail_fast: bool = False,
                checkpoint: Callable[[], None] | None = None) -> dict[str, dict]:
    """Run the output modules (default: the registry, ``outputs.registry.MODULES``) on ``ctx`` in its mode
    (``ctx.mode``): per module compute, product and render in a full pass, compute and product in a compute pass,
    product and render in a render pass.

    ``only`` selects modules plus whatever they require (in a render pass a module added that way gives only its
    product); ``skip`` removes modules; ``disabled`` names the modules switched off in the configuration
    (:func:`disabled_modules`), which are recorded as disabled and do not run.  A module that requires a skipped,
    disabled, inactive or failed-without-product one is skipped too.  A failing step is recorded and the module goes on
    (:meth:`OutputContext.step`); an error outside the steps fails the module, and the other modules still run.  A
    module ends as failed when a compute step failed (in a render pass: in its last compute) or a render step failed or
    could not be drawn; the record of a module refused in a render pass for want of a compute has ``not_computed``.
    ``fail_fast`` re-raises the first error instead.  ``checkpoint`` is called once, before the first module with
    ``checkpoint`` set that runs.  Every module of the plan gets a record in ``ctx.summary["modules"]``: ``status``
    ("done", "disabled", "inactive", "skipped" or "failed"),
    ``mode``, ``seconds``, ``steps`` (each compute step's ``status``, ``seconds``, ``wrote`` and ``error`` or
    ``reason``; in a render pass those of the saved completeness record), ``render_steps`` when it rendered,
    ``reporting_draws`` and ``image_draws`` when its compute used the draws, and ``reason`` or ``error``.  Returns
    those records."""
    if modules is None:
        from .registry import MODULES as modules
    skip, disabled = tuple(skip), tuple(disabled)
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
            records[name] = {"status": "skipped", "seconds": 0.0, "reason": "in --skip"}
            log(f"[outputs] {name}: skipped (in --skip)")
            continue
        if name in disabled:
            records[name] = {"status": "disabled", "seconds": 0.0, "reason": f"outputs.{name}.enabled is false"}
            log(f"[outputs] {name}: disabled (outputs.{name}.enabled is false)")
            continue
        missing = next((r for r in module.requires if r not in ctx.products), None)
        if missing is not None:
            required = records[missing]
            status = required["status"]
            reason = f"requires {missing}, which was {status}" if status != "failed" else f"requires {missing}, which failed without a product"
            records[name] = {"status": "skipped", "mode": ctx.mode, "seconds": 0.0, "reason": reason}
            if status == "failed" or "failed_requirement" in required:
                # Not the user's choice: what this module would have computed from is gone (merged_summary).
                records[name]["failed_requirement"] = required.get("failed_requirement", missing)
            elif status == "disabled" or "disabled_requirement" in required:
                # A requirement switched off in this pass: this record replaces the saved one (merged_summary).
                records[name]["disabled_requirement"] = required.get("disabled_requirement", missing)
            log(f"[outputs] {name}: skipped ({reason})")
            continue
        started = time.perf_counter()
        ctx._module, ctx._phase, ctx._steps, ctx._render_steps = module, None, {}, {}
        ctx._completion, ctx._fail_fast, ctx._used_draws = None, fail_fast, False
        error_text, not_computed = None, False
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
            not_computed = isinstance(error, NotComputed)
            seconds = time.perf_counter() - started
            # Once per error: a step (or the checkpoint) logged its traceback, and ModuleUnavailable follows logged steps.
            told = error is ctx._logged or isinstance(error, ModuleUnavailable)
            if not told:                # the module's own error: its traceback (the closing line below names it)
                log(f"[outputs] {name}: {error_text}\n{traceback.format_exc().rstrip()}")
            if fail_fast:
                log(f"[outputs] {name}: FAILED after {seconds:.1f}s ({error_text})")
                records[name] = {"status": "failed", "mode": ctx.mode, "seconds": seconds, "error": error_text,
                                 **_step_records(ctx), **_draws_record(ctx), **_settings_record(ctx)}
                ctx._module, ctx._phase, ctx._steps, ctx._render_steps = None, None, {}, {}
                raise
        finally:
            step_records, draws_record = _step_records(ctx), {**_draws_record(ctx), **_settings_record(ctx)}
            ctx._module, ctx._phase, ctx._steps, ctx._render_steps, ctx._completion = None, None, {}, {}, None
        if reason is not None:
            records[name] = {"status": "inactive", "seconds": 0.0, "reason": str(reason)}
            log(f"[outputs] {name}: inactive ({reason})")
            continue
        seconds = time.perf_counter() - started
        render_records = step_records.get("render_steps", {})      # absent when the module drew nothing
        failed_steps = [step for step, record in step_records["steps"].items() if record["status"] == "failed"]
        failed_steps += [f"{step} (render)" for step, record in render_records.items() if record["status"] == "failed"]
        not_drawn = [step for step, record in render_records.items() if record["status"] == "skipped"]
        if error_text is None and (failed_steps or not_drawn):   # each step not drawn was logged with its reason
            error_text = (f"failed steps: {', '.join(failed_steps)}" if failed_steps else
                          f"render steps not drawn: {', '.join(not_drawn)}")
        records[name] = {"status": "failed" if error_text else "done", "mode": ctx.mode, "seconds": seconds,
                         **step_records, **draws_record}
        if error_text:
            records[name]["error"] = error_text
        if not_computed:
            records[name]["not_computed"] = True
        log(f"[outputs] {name}: {'FAILED (' + error_text + ')' if error_text else 'done'} in {seconds:.1f}s")
    check_draws(ctx, modules)
    check_settings(ctx, modules, records)
    if ctx.mode == "render" and not rendered_anything(records):
        log(f"[outputs] WARNING: this render pass drew nothing: {nothing_drawn_reason(records)}")
    failed = [name for name, record in records.items() if record["status"] == "failed"]
    if failed:
        log(f"[outputs] failed modules: {', '.join(failed)} (tracebacks above)")
    return records


def rendered_anything(records: dict[str, dict]) -> bool:
    """Whether any module of a pass's ``records`` ran a render step that draws (not one of its ``input_steps``,
    which only load what the figures read) without failing."""
    return any(step["status"] == "done" and not step.get("input") for record in records.values()
               for step in record.get("render_steps", {}).values())


def nothing_drawn_reason(records: dict[str, dict]) -> str:
    """Why a render pass whose module ``records`` drew nothing (:func:`rendered_anything` false) drew nothing: no module
    ran; an input a figure needs is missing (a failed input step, such as the imaging of ``rgb``, named with its
    error); nothing was computed for the modules that ran (``not_computed``); render steps were skipped or failed;
    modules failed outside their steps (a product that could not be loaded, a render that raised outside its steps),
    named with their errors; or the last compute saved nothing to draw."""
    ran = {name: record for name, record in records.items() if record["status"] in ("done", "failed")}
    if not ran:
        return "no selected module ran (each was disabled, inactive or skipped)"
    render_steps = [(f"{name}.{step}", record) for name, module in ran.items()
                    for step, record in module.get("render_steps", {}).items()]
    missing = [f"{step} ({record['error']})" for step, record in render_steps if record.get("input") and record["status"] == "failed"]
    if missing:
        return f"the compute is there, but an input needed to draw is missing: {'; '.join(missing)}"
    uncomputed = [name for name, record in ran.items() if record.get("not_computed")]
    if len(uncomputed) == len(ran):
        return (f"nothing was computed for the selected modules ({', '.join(uncomputed)}): run a full or --compute "
                "pass first")
    not_drawn = [f"{step} ({record['status']}: {record['error'] if record['status'] == 'failed' else record['reason']})"
                 for step, record in render_steps if record["status"] != "done"]
    if not_drawn:
        return f"the compute is there, but no figure could be drawn: {'; '.join(not_drawn)}"
    failed = [f"{name} ({record['error']})" for name, record in ran.items()
              if record["status"] == "failed" and not record.get("not_computed")]
    if failed:
        return (f"modules failed before drawing: {'; '.join(failed)}"
                + (f" (never computed: {', '.join(uncomputed)})" if uncomputed else ""))
    return ("the last compute of the selected modules saved nothing to draw"
            + (f" (never computed: {', '.join(uncomputed)})" if uncomputed else ""))


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
    described = "; ".join(", ".join(names) + f" ({saved[names[0]]['reporting_draws']} draws, seed "
                          f"{saved[names[0]]['draws_seed']}, key {key[:8]})" for key, names in groups.items())
    stale = [f"{module.name} (requires {required})" for module in modules if module.name in saved
             for required in module.requires
             if required in saved and saved[required]["draws_key"] != saved[module.name]["draws_key"]
             and saved[required]["computed_at"] > saved[module.name]["computed_at"]]
    warning = f"{DRAWS_WARNING}: {described}"
    if stale:
        warning += (f". Stale, because a module they require was computed again with other draws since: "
                    f"{', '.join(stale)}")
    warning += ". Compute them together (one pass over the same draws) for consistent numbers and figures"
    ctx.summary.setdefault("warnings", []).append(warning)
    ctx.log(f"[outputs] WARNING: {warning}")
    return warning


SETTINGS_WARNING = "outputs settings changed since the last compute of"


def _shown(value) -> str:
    return "null" if value is None else str(value)


def check_settings(ctx: OutputContext, modules: Sequence[OutputModule], records: dict[str, dict]) -> list[str]:
    """Warn, once per module, when the saved results of one of ``modules`` were computed with other outputs settings
    than those of this pass (``ctx.config.outputs``): the ``settings`` of its completeness record against
    :func:`settings_snapshot` of the current configuration, over every field but the module's ``render_settings``
    (which render reads from the current configuration).  Modules disabled or inactive in this pass (``records``) and
    modules without a record of a finished compute are left out.  Each warning (``[outputs] WARNING``, and in
    ``ctx.summary["warnings"]``) names the module and the fields as ``old -> current``; it never blocks the pass.
    Returns the warnings."""
    from ..io import store
    warnings = []
    for module in modules:
        if records.get(module.name, {}).get("status") in ("disabled", "inactive"):
            continue
        try:
            record = store.read_record(ctx.stage, completion_name(module.name))
        except (store.MissingEntry, store.DamagedEntry, store.StoreError):
            continue
        if "steps" not in record:              # an interrupted compute: render refuses it (NotComputed)
            continue
        saved = record["settings"]
        current = json.loads(json.dumps(settings_snapshot(ctx.config.outputs, module), default=str))
        changed = [f"outputs.{key} {_shown(saved.get(key))} -> {_shown(current.get(key))}"
                   for key in dict.fromkeys([*saved, *current])
                   if not any(key == field or key.startswith(field + ".") for field in module.render_settings)
                   and saved.get(key) != current.get(key)]
        if not changed:
            continue
        warning = (f"{SETTINGS_WARNING} {module.name} (at {record['computed_at']}): {'; '.join(changed)}. Its saved "
                   f"results, and the figures drawn from them, come from the old values; compute it again "
                   f"(lumina outputs RUN --only {module.name}) for results of the current ones")
        warnings.append(warning)
        ctx.summary.setdefault("warnings", []).append(warning)
        ctx.log(f"[outputs] WARNING: {warning}")
    return warnings


def _run_phases(ctx: OutputContext, module: OutputModule, *, product_only: bool) -> None:
    """Run the phases of ``module`` that ``ctx.mode`` asks for and set its product.  A compute that raises
    :class:`ModuleUnavailable` gives no product, but in a full pass its render still draws the figures whose compute
    steps succeeded before the error is raised again; in a render pass the same holds for a module whose last compute
    gave no product (``unavailable`` in its record)."""
    from ..io import store
    name = module.name
    unavailable: ModuleUnavailable | None = None
    if ctx.mode == "render":
        ctx._completion = load_completion(ctx.stage, name)
        if ctx._completion["unavailable"] is not None:
            unavailable = ModuleUnavailable(f"the last compute of {name} gave no product "
                                            f"({ctx._completion['unavailable']}); {_compute_hint(name)}")
    else:
        ctx._phase = "compute"
        _mark_incomplete(ctx, name)
        try:
            module.compute(ctx)
        except ModuleUnavailable as error:
            unavailable = error
            ctx._completion = _write_completion(ctx, name, unavailable=error)
        except Exception as error:
            try:                        # the steps that ran, still incomplete; the marker above already says so
                _write_completion(ctx, name, unavailable=error)
            except Exception as failure:
                ctx.log(f"[outputs] {name}: completeness record not updated: {type(failure).__name__}: {failure}")
            raise
        else:
            ctx._completion = _write_completion(ctx, name)
    if unavailable is None:
        ctx._phase = "product"
        try:
            ctx.products[name] = module.product(ctx)
        except store.MissingEntry as error:
            raise ModuleUnavailable(f"{error}; {_compute_hint(name)}") from error
    if ctx.mode != "compute" and not product_only:
        ctx._phase = "render"
        module.render(ctx)
    if unavailable is not None:
        raise unavailable


def _compute_hint(name: str) -> str:
    return f"run its compute first: lumina outputs RUN --only {name} (or --compute --only {name})"


def _flat_settings(value, prefix: str) -> dict[str, Any]:
    """``{dotted name: value}`` of a configuration value under ``prefix``: the fields of a section one by one (nested
    sections too), as JSON values (paths as the strings given, tuples as lists)."""
    from dataclasses import fields, is_dataclass
    if is_dataclass(value):
        out = {}
        for item in fields(value):
            out.update(_flat_settings(getattr(value, item.name), f"{prefix}.{item.name}"))
        return out
    if isinstance(value, Path):
        return {prefix: str(value)}
    if isinstance(value, tuple):
        return {prefix: [str(v) if isinstance(v, Path) else v for v in value]}
    return {prefix: value}


def settings_snapshot(outputs, module: OutputModule) -> dict[str, Any]:
    """The outputs settings behind the compute of ``module`` under the section ``outputs`` (``config.OutputsConfig``),
    as ``{dotted name under outputs: value}``: its ``shared_settings``, its own section without ``enabled`` (a switch
    that selects the module, not a result) and its ``path_fields`` in other modules' sections.  ``workers`` and
    ``fail_fast`` are never part of it: they change how a pass runs, not what it computes."""
    snapshot: dict[str, Any] = {}
    for name in module.shared_settings:
        snapshot.update(_flat_settings(getattr(outputs, name), name))
    own = getattr(outputs, module.name)
    snapshot.update({key: value for key, value in _flat_settings(own, module.name).items()
                     if key != f"{module.name}.enabled"})
    for dotted in module.path_fields:
        section, field = dotted.split(".", 1)
        if section != module.name:
            snapshot.update(_flat_settings(getattr(getattr(outputs, section), field), dotted))
    return snapshot


def load_completion(stage, name: str) -> dict[str, Any]:
    """The completeness record of module ``name`` in ``stage``.

    Raises
    ------
    NotComputed
        When there is none, it cannot be read, or the last compute of the module was interrupted before it recorded
        its steps (the record is still the marker of :func:`_mark_incomplete`); the message tells to run the compute
        first.  A compute with failed or skipped steps is returned: render draws what its other steps saved.
    """
    from ..io import store
    try:
        record = store.read_record(stage, completion_name(name))
    except (store.MissingEntry, store.DamagedEntry, store.StoreError) as error:
        raise NotComputed(f"no record {completion_name(name)} of a compute of {name} ({type(error).__name__}); "
                          f"{_compute_hint(name)}") from error
    if "steps" not in record:
        raise NotComputed(f"the last compute of {name}, started at {record['started_at']}, was interrupted before it "
                          f"recorded its steps; {_compute_hint(name)}")
    return record


def _mark_incomplete(ctx: OutputContext, name: str) -> None:
    """Set the module's completeness record to incomplete before its compute writes anything, so that a compute that
    stops half way never leaves a record of an earlier complete one."""
    from ..io import store
    store.write_record(ctx.stage, completion_name(name),
                       {"module": name, "complete": False, "started_at": datetime.now(timezone.utc).isoformat()})


def _write_completion(ctx: OutputContext, name: str, *, unavailable: Exception | None = None) -> dict[str, Any]:
    """Write and return the module's completeness record after its compute: ``complete`` (every compute step done and
    the compute gave its product), ``unavailable`` (the error with which the compute ended without a product, as
    text, else None), when, the compute steps and the reporting draws the compute used (``reporting_draws``,
    ``image_draws``, ``draws_seed``, ``draws_key``) and the outputs settings it used (``settings``,
    :func:`settings_snapshot`)."""
    from ..io import store
    done = all(record["status"] == "done" for record in ctx._steps.values())
    record = {"module": name, "complete": done and unavailable is None,
              "unavailable": None if unavailable is None else f"{type(unavailable).__name__}: {unavailable}",
              "computed_at": datetime.now(timezone.utc).isoformat(), "steps": ctx._steps, **_draws_record(ctx),
              **_settings_record(ctx)}
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
        out = {"steps": ctx._completion["steps"]}
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


def _settings_record(ctx: OutputContext) -> dict[str, dict]:
    """``{"settings": ...}``: the outputs settings of the current module's compute (:func:`settings_snapshot`), in a
    render pass those of its saved completeness record; {} when no compute of it is described."""
    if ctx.mode == "render":
        return {} if ctx._completion is None else {"settings": ctx._completion["settings"]}
    return {"settings": settings_snapshot(ctx.config.outputs, ctx._module)} if ctx._phase is not None else {}


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
