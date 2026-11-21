"""The outputs stage as modules: ordering, selection, skipping, failures, lazy draws, compute and render, the config
section and the ``lumina outputs`` command."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from lumina.config import LensClusterSolverConfig, OutputsConfig
from lumina.io import artifacts, store
from lumina.outputs import MODULES, OutputContext, OutputModule, module_named, order_modules, plan_modules, run_modules
from lumina.outputs.base import ComputeOnly, ModuleUnavailable, completion_name


class Fake(OutputModule):
    def __init__(self, name, requires=(), after=(), inactive=None, error=None, checkpoint=False, calls=None, draws=False):
        self.name, self.requires, self.after, self.checkpoint = name, tuple(requires), tuple(after), checkpoint
        self.inactive, self.error, self.calls, self.draws = inactive, error, calls if calls is not None else [], draws

    def active(self, ctx):
        return self.inactive

    def compute(self, ctx):
        self.calls.append(self.name)
        if self.draws:
            ctx.draws
        if self.error is not None:
            raise self.error

    def product(self, ctx):
        return f"product of {self.name}"


@dataclass(frozen=True)
class ToySection:
    """The configuration section of a toy module: its switch and one setting."""
    enabled: bool = True
    value: int = 1


class ToyOutputs(SimpleNamespace):
    """``config.outputs`` of the toy modules: a :class:`ToySection` for any module name, ``draws`` 8 and whatever the
    test sets."""

    def __init__(self, **fields):
        super().__init__(**{"draws": 8, **fields})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return ToySection()


def toy_config(**outputs):
    return SimpleNamespace(paths=SimpleNamespace(figure_prefix="toy"), outputs=ToyOutputs(**outputs))


def context(tmp_path, log=None, mode="full", config=None):
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures", root=tmp_path)
    return OutputContext(toy_config() if config is None else config, None, None, None, {}, paths,
                         log=log if log is not None else (lambda _: None), mode=mode)


def names(modules):
    return [m.name for m in modules]


def test_default_order_is_the_registry_order():
    assert names(order_modules(MODULES)) == names(MODULES) == [
        "fit", "images", "mass", "truth", "rgb"]
    assert module_named("rgb").requires == ("images",) and module_named("rgb").after == ("fit",)
    assert module_named("mass").after == ("fit",) and module_named("truth").requires == ("images",)
    with pytest.raises(KeyError, match="no output module 'nope'"):
        module_named("nope")


def test_requires_and_after_order_stably():
    modules = [Fake("c", requires=["b"]), Fake("a"), Fake("b", after=["a"]), Fake("d")]
    assert names(order_modules(modules)) == ["a", "b", "c", "d"]
    # Modules already in a valid order keep it.
    assert names(order_modules([Fake("x"), Fake("y", after=["x"]), Fake("z")])) == ["x", "y", "z"]


def test_cycles_and_unknown_names_are_errors():
    with pytest.raises(ValueError, match="cycle"):
        order_modules([Fake("a", after=["b"]), Fake("b", requires=["a"])])
    with pytest.raises(ValueError, match="unknown modules"):
        order_modules([Fake("a", requires=["ghost"])])
    with pytest.raises(ValueError, match="unknown modules"):
        order_modules([Fake("a", after=["ghost"])])
    with pytest.raises(ValueError, match="duplicate"):
        order_modules([Fake("a"), Fake("a")])
    with pytest.raises(ValueError, match="--only names unknown"):
        plan_modules([Fake("a")], only=["b"])
    with pytest.raises(ValueError, match="--skip names unknown"):
        plan_modules([Fake("a")], skip=["b"])


def test_only_pulls_in_what_is_required(tmp_path):
    lines = []
    calls = []
    modules = [Fake("a", calls=calls), Fake("b", requires=["a"], calls=calls), Fake("c", requires=["b"], calls=calls),
               Fake("d", calls=calls)]
    ctx = context(tmp_path, log=lines.append)
    records = run_modules(ctx, modules, only=["c"])
    assert calls == ["a", "b", "c"] and list(records) == ["a", "b", "c"]
    assert ctx.products == {"a": "product of a", "b": "product of b", "c": "product of c"}
    assert any("added because a selected module requires them: b (by c), a (by b)" in line for line in lines)


def test_skip_cascades_to_dependents(tmp_path):
    calls = []
    modules = [Fake("a", calls=calls), Fake("b", requires=["a"], calls=calls), Fake("c", requires=["b"], calls=calls),
               Fake("d", after=["a"], calls=calls)]
    ctx = context(tmp_path)
    records = run_modules(ctx, modules, skip=["a"])
    assert calls == ["d"]
    assert records["a"] == {"status": "skipped", "seconds": 0.0, "reason": "in --skip"}
    assert records["b"]["status"] == records["c"]["status"] == "skipped"
    assert records["c"]["reason"] == "requires b, which was skipped" and records["d"]["status"] == "done"


def test_a_disabled_module_is_recorded_and_its_dependents_skipped(tmp_path):
    calls, lines = [], []
    modules = [Fake("a", calls=calls), Fake("b", requires=["a"], calls=calls), Fake("c", calls=calls)]
    records = run_modules(context(tmp_path, log=lines.append), modules, disabled=["a"])
    assert calls == ["c"]
    assert records["a"] == {"status": "disabled", "seconds": 0.0, "reason": "outputs.a.enabled is false"}
    assert records["b"]["status"] == "skipped" and records["b"]["reason"] == "requires a, which was disabled"
    assert "[outputs] a: disabled (outputs.a.enabled is false)" in lines


def test_inactive_module_gives_its_reason(tmp_path):
    lines, calls = [], []
    modules = [Fake("truth", inactive="no truth maps configured", calls=calls), Fake("plots", requires=["truth"], calls=calls)]
    ctx = context(tmp_path, log=lines.append)
    records = run_modules(ctx, modules)
    assert calls == []
    assert records["truth"] == {"status": "inactive", "seconds": 0.0, "reason": "no truth maps configured"}
    assert records["plots"]["reason"] == "requires truth, which was inactive"
    assert "[outputs] truth: inactive (no truth maps configured)" in lines


def test_a_failing_module_is_isolated_and_recorded(tmp_path):
    lines, calls = [], []
    modules = [Fake("a", error=RuntimeError("broken map"), calls=calls), Fake("b", requires=["a"], calls=calls),
               Fake("c", calls=calls)]
    ctx = context(tmp_path, log=lines.append)
    records = run_modules(ctx, modules)
    assert calls == ["a", "c"]
    assert records["a"]["status"] == "failed" and records["a"]["error"] == "RuntimeError: broken map"
    assert records["b"] == {"status": "skipped", "mode": "full", "seconds": 0.0,
                            "reason": "requires a, which failed without a product", "failed_requirement": "a"}
    assert records["c"]["status"] == "done" and ctx.summary["modules"] is records
    failure = next(line for line in lines if line.startswith("[outputs] a: RuntimeError"))
    assert "broken map" in failure and "Traceback" in failure
    assert "[outputs] a: FAILED (RuntimeError: broken map) in" in "\n".join(lines)
    assert sum("Traceback" in line for line in lines) == 1
    assert lines[-1].startswith("[outputs] failed modules: a")


def test_fail_fast_reraises(tmp_path):
    calls = []
    modules = [Fake("a", error=RuntimeError("broken"), calls=calls), Fake("c", calls=calls)]
    ctx = context(tmp_path)
    with pytest.raises(RuntimeError, match="broken"):
        run_modules(ctx, modules, fail_fast=True)
    assert calls == ["a"] and ctx.summary["modules"]["a"]["status"] == "failed"
    # KeyboardInterrupt is never caught.
    with pytest.raises(KeyboardInterrupt):
        run_modules(context(tmp_path), [Fake("a", error=KeyboardInterrupt())])


def test_checkpoint_runs_once_before_the_first_imaging_module(tmp_path):
    events = []
    modules = [Fake("a", calls=events), Fake("rgb", checkpoint=True, calls=events), Fake("folds", checkpoint=True, calls=events)]
    # (two checkpoint modules: the checkpoint still runs once)
    run_modules(context(tmp_path), modules, checkpoint=lambda: events.append("checkpoint"))
    assert events == ["a", "checkpoint", "rgb", "folds"]
    events.clear()
    run_modules(context(tmp_path), modules, skip=["rgb", "folds"], checkpoint=lambda: events.append("checkpoint"))
    assert events == ["a"]


def test_draws_are_built_lazily(monkeypatch, tmp_path):
    from lumina.reporting import draws as draws_module
    built = []

    class FakeDraws(list):
        seed, has_field, cache_stage, image_indices = 7, False, None, [0, 1, 2]

    monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: built.append(1) or FakeDraws([0, 1]))
    ctx = context(tmp_path)
    records = run_modules(ctx, [Fake("a"), Fake("b")])
    assert built == [] and "reporting_draws" not in ctx.summary
    assert "reporting_draws" not in records["a"]
    records = run_modules(ctx, [Fake("a", draws=True), Fake("b")])
    assert built == [1] and ctx.summary["reporting_draws"] == 2 and ctx.draws.cache_stage == tmp_path
    # Each module records the draws it used.
    assert records["a"]["reporting_draws"] == 2 and records["a"]["image_draws"] == 3 and "reporting_draws" not in records["b"]


def test_a_failed_draws_build_is_not_repeated(monkeypatch, tmp_path):
    from lumina.reporting import draws as draws_module
    built = []

    def broken(*a):
        built.append(1)
        raise RuntimeError("no room for the draws")

    monkeypatch.setattr(draws_module, "reporting_draws", broken)
    ctx = context(tmp_path)
    records = run_modules(ctx, [Fake("a", draws=True), Fake("b", draws=True)])
    assert built == [1]
    assert records["a"]["error"] == records["b"]["error"] == "RuntimeError: no room for the draws"


def test_outputs_stage_builds_draws_only_in_the_step_that_needs_them(monkeypatch, synthetic_config, tmp_path):
    from lumina.outputs import fit
    from lumina.plotting import block_corner, chain_traces, gp, members, plots, potfile
    from lumina.reporting import draws as draws_module, outputs, results_tex

    def no_draws(*a):
        raise RuntimeError("draws built")

    monkeypatch.setattr(draws_module, "reporting_draws", no_draws)
    for module, name in ((members, "save_member_diagnostics"), (gp, "save_gp_diagnostics"), (chain_traces, "save_chain_traces"),
                         (potfile, "write_potfile_plots"), (block_corner, "write_posterior_block_corner"),
                         (block_corner, "write_cosmology_corner"), (plots, "plot_corner"), (chain_traces, "trace_data")):
        monkeypatch.setattr(module, name, lambda *a, **k: None)
    monkeypatch.setattr(fit, "posterior_summary_table", lambda *a: pd.DataFrame())
    monkeypatch.setattr(fit, "classification_table", lambda *a: pd.DataFrame())
    monkeypatch.setattr(plots, "member_scaling_table", lambda *a: pd.DataFrame())
    monkeypatch.setattr(results_tex, "write_run_results", lambda *a, **k: None)
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures", root=tmp_path)
    # A saved summary from an earlier full pass: a partial pass keeps its numbers.
    store.write_record(tmp_path, "run_summary", {"fit": {"rms": 0.5}, "modules": {"images": {"status": "done"}}})
    summary = outputs.write_outputs(synthetic_config, SimpleNamespace(parameter_specs=[]), None, SimpleNamespace(physical=[[0.0]]), {}, paths,
                                    only=["fit"], log=lambda _: None)
    steps = summary["modules"]["fit"]["steps"]
    assert {name for name, step in steps.items() if step["status"] != "done"} == {"covariance"}
    assert steps["covariance"]["error"] == "RuntimeError: draws built" and "reporting_draws" not in summary
    assert summary["modules"]["fit"]["status"] == "failed" and summary["modules"]["fit"]["error"] == "failed steps: covariance"
    saved = store.read_record(tmp_path, "run_summary")
    assert saved["fit"] == {"rms": 0.5} and set(saved["modules"]) == {"images", "fit"}
    with pytest.raises(ValueError, match="unknown output modules"):
        outputs.write_outputs(synthetic_config, SimpleNamespace(), None, None, {}, paths, skip=["nope"])


class Stepped(OutputModule):
    """A module of three steps: ``table`` (whose result ``figure`` needs) and ``other``."""

    compute_steps = ("table", "figure", "other")

    def __init__(self, name="stepped", fail=(), product=True):
        self.name, self.fail, self.gives, self.calls = name, set(fail), product, []

    def _work(self, step):
        self.calls.append(step)
        if step in self.fail:
            raise ValueError(f"{step} broke")
        return step

    def compute(self, ctx):
        ctx.step("table", self._work, "table")
        if ctx.ok("table"):
            ctx.step("figure", self._work, "figure")
        else:
            ctx.skip_step("figure", "table failed")
        ctx.step("other", self._work, "other")
        if not ctx.ok("table"):
            raise ModuleUnavailable("no table (the table step failed)")

    def product(self, ctx):
        return "table" if self.gives else None


def test_a_failing_step_does_not_stop_the_module(tmp_path):
    lines = []
    module = Stepped(fail={"table"})
    ctx = context(tmp_path, log=lines.append)
    records = run_modules(ctx, [module, Fake("after", requires=["stepped"])])
    assert module.calls == ["table", "other"]
    steps = records["stepped"]["steps"]
    assert steps["table"]["status"] == "failed" and steps["table"]["error"] == "ValueError: table broke"
    assert steps["figure"] == {"status": "skipped", "seconds": 0.0, "reason": "table failed"}
    assert steps["other"]["status"] == "done"
    # Without its table the module raised: failed with no product, and the module that requires it is skipped.
    assert records["stepped"]["status"] == "failed" and records["stepped"]["error"] == "ModuleUnavailable: no table (the table step failed)"
    assert "stepped" not in ctx.products
    assert records["after"]["reason"] == "requires stepped, which failed without a product"
    assert sum("Traceback" in line for line in lines) == 1          # the step's; the module's own line has none
    assert "[outputs] stepped.figure: skipped (table failed)" in lines
    # A failed step that leaves the product: the module is failed and the module that requires it still runs.
    records = run_modules(context(tmp_path), [Stepped(fail={"other"}), Fake("after", requires=["stepped"])])
    assert records["stepped"]["status"] == "failed" and records["stepped"]["error"] == "failed steps: other"
    assert records["after"]["status"] == "done"
    # A module that returned None satisfies its dependents.
    records = run_modules(context(tmp_path), [Stepped(product=False), Fake("after", requires=["stepped"])])
    assert records["stepped"]["status"] == records["after"]["status"] == "done"


def test_a_failing_step_with_fail_fast_reraises(tmp_path):
    lines = []
    module = Stepped(fail={"table"})
    ctx = context(tmp_path, log=lines.append)
    with pytest.raises(ValueError, match="table broke"):
        run_modules(ctx, [module], fail_fast=True)
    assert module.calls == ["table"]
    assert sum("Traceback" in line for line in lines) == 1          # logged once, by the step
    record = ctx.summary["modules"]["stepped"]
    assert record["status"] == "failed" and record["steps"]["table"]["status"] == "failed"
    with pytest.raises(RuntimeError, match="outside a module.s compute or render"):
        context(tmp_path).step("x", print)


def test_imaging_is_loaded_once_per_variant(monkeypatch, synthetic_config):
    from lumina.imaging import cutouts
    loads, lines = [], []

    def load(image_config, *args):
        loads.append(image_config.image_dir)
        if image_config.image_dir == "missing":
            raise FileNotFoundError("no such folder")
        return ("imaging", image_config.image_dir)

    monkeypatch.setattr(cutouts, "load_cluster_imaging", load)
    ctx = OutputContext(synthetic_config, None, None, None, {}, None, log=lines.append)
    view = replace(synthetic_config.outputs.rgb, image_dir="hst")
    assert ctx.imaging(view) is ctx.imaging(replace(view, figure_suffix="_other"))
    missing = replace(view, image_dir="missing")
    assert ctx.imaging(missing) is None and ctx.imaging(missing) is None
    assert loads == ["hst", "missing"]
    assert lines == ["[outputs] cluster mass figure and covariance RGB cutouts skipped: no such folder"]


def test_outputs_config_is_validated(synthetic_config):
    from lumina.config import ImageSolverConfig, RGBOutputConfig, TruthOutputConfig
    synthetic_config.validate()
    outputs = synthetic_config.outputs
    replace(synthetic_config, outputs=replace(outputs, fail_fast=True, rgb=replace(outputs.rgb, enabled=False))).validate()
    for bad, match in ((OutputsConfig(draws=0), r"^outputs\.draws must be a positive integer"),
                       (OutputsConfig(draws=8, image_draws=4), r"^outputs\.image_draws must be an integer of at least outputs\.draws"),
                       (OutputsConfig(workers=0), r"^outputs\.workers must be a positive integer"),
                       (OutputsConfig(caustic_source_redshift=float("nan")), r"^outputs\.caustic_source_redshift"),
                       (OutputsConfig(fail_fast="yes"), r"^outputs\.fail_fast must be true or false"),
                       (OutputsConfig(rgb=RGBOutputConfig(enabled="no")), r"^outputs\.rgb\.enabled must be true or false"),
                       (OutputsConfig(rgb=RGBOutputConfig(fold_pair_max_separation_arcsec=0.0)),
                        r"^outputs\.rgb\.fold_pair_max_separation_arcsec must be finite and positive"),
                       (OutputsConfig(truth=TruthOutputConfig(grid_size=-1)), r"^outputs\.truth\.grid_size must be"),
                       (OutputsConfig(truth=TruthOutputConfig(kappa_true_fits=3)), r"^outputs\.truth\.kappa_true_fits must be"),
                       (OutputsConfig(image_solver=ImageSolverConfig(exact_image_finder="newton")),
                        r"^outputs\.image_solver\.exact_image_finder must be"),
                       (OutputsConfig(images=5), r"^outputs\.images must be a section"),
                       (5, r"^outputs must be a section")):
        with pytest.raises(ValueError, match=match):
            replace(synthetic_config, outputs=bad).validate()


def test_every_module_has_its_own_section():
    from dataclasses import fields
    from lumina.outputs.base import disabled_modules
    sections = {f.name: f for f in fields(OutputsConfig)}
    for module in MODULES:
        section = getattr(OutputsConfig(), module.name)
        assert module.name in sections and section.enabled is True and callable(section.validate)
        for name in module.path_fields:           # dotted names under outputs
            node = OutputsConfig()
            for part in name.split("."):
                node = getattr(node, part)
    assert disabled_modules(SimpleNamespace(outputs=OutputsConfig())) == ()


def test_outputs_settings_leave_the_hash_and_the_resume_alone(synthetic_config, tmp_path):
    from lumina import cli
    from lumina.config import ImageSolverConfig, ImagesOutputConfig, RGBDisplayConfig, RGBOutputConfig, TruthOutputConfig
    from lumina.pipeline import resume_payload
    base = synthetic_config
    every = OutputsConfig(draws=3, image_draws=5, workers=2, caustic_source_redshift=2.5, fail_fast=True,
                          image_solver=ImageSolverConfig(match_tolerance_arcsec=1.0),
                          images=ImagesOutputConfig(enabled=False, extra_images=False),
                          truth=TruthOutputConfig(enabled=False, kappa_true_fits="k.fits", deflection_true_fits="d.fits", grid_size=8),
                          rgb=RGBOutputConfig(enabled=False, image_dir="imaging", display=RGBDisplayConfig(q=2.0),
                                              fold_pair_max_separation_arcsec=1.0))
    chosen = replace(base, outputs=every).validate()
    assert cli.config_hash(chosen) == cli.config_hash(base)
    assert resume_payload(chosen.to_json_dict()) == resume_payload(base.to_json_dict())
    assert artifacts.init_signature(chosen) == artifacts.init_signature(base)
    assert "outputs" not in resume_payload(base.to_json_dict()) and "outputs" not in artifacts.init_signature(base)
    # The section round-trips through config.json.
    artifacts.save_config(tmp_path, chosen)
    assert artifacts.load_config(tmp_path).outputs == chosen.outputs
    assert LensClusterSolverConfig.from_json_dict(chosen.to_json_dict()).outputs == chosen.outputs
    # A config.json of the layout before the outputs section is refused like any unknown field.
    for section, value in (("truth", {"kappa_true_fits": None}), ("reporting", {"draws": 8}), ("image_catalog", {}),
                           ("image_diagnostics", {})):
        payload = chosen.to_json_dict()
        payload[section] = value
        with pytest.raises(TypeError, match=section):
            LensClusterSolverConfig.from_json_dict(payload)
    payload = chosen.to_json_dict()
    payload["runtime"]["output_workers"] = 4
    with pytest.raises(TypeError, match="output_workers"):
        LensClusterSolverConfig.from_json_dict(payload)
    payload = chosen.to_json_dict()
    payload["outputs"]["only"] = ["fit"]
    with pytest.raises(TypeError, match="only"):
        LensClusterSolverConfig.from_json_dict(payload)


def test_outputs_section_from_run_files_and_set():
    from lumina import cli
    from lumina.run_files import load_run
    config = load_run("catalogs/hera", overrides=["outputs.rgb.enabled=false", "outputs.fail_fast=true", "outputs.workers=2"])
    assert not config.outputs.rgb.enabled and config.outputs.fail_fast and config.outputs.workers == 2
    assert config.outputs.rgb.image_dir.endswith("data/imaging/hera/ffsims")         # resolved against the file's folder
    assert (cli.build_config("catalogs/hera", overrides=["outputs.rgb.enabled=false", "outputs.draws=2", "outputs.image_draws=2",
                                                         "outputs.truth.deflection_true_fits=null"]).paths.run_name
            == cli.build_config("catalogs/hera").paths.run_name)
    # The sections and fields before the outputs section are unknown keys, with no special message.
    with pytest.raises(ValueError, match="unknown keys"):
        load_run("catalogs/hera", overrides=["reporting.draws=8"])
    for old in ("outputs.only=[truth]", "outputs.skip=[rgb]", "runtime.output_workers=2", "outputs.rgb.rgb={q: 1}"):
        with pytest.raises(TypeError):
            load_run("catalogs/hera", overrides=[old])


def test_outputs_list_command(capsys, synthetic_config, tmp_path):
    from lumina import cli
    assert cli.main(["outputs", "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[1::2]] == [m.name for m in MODULES]
    assert lines[0].split()[:4] == ["module", "requires", "after", "description"]
    assert lines[5].split()[:3] == ["mass", "images", "fit"]
    assert lines[6] == "    compute: critical_curves, aperture_masses, zeta_masses; render: critical_curves, zeta_masses"
    assert lines[2] == ("    compute: tables, chain_traces, audit, scaling_table, covariance, fields, members; render: "
                        "chain_traces, potfiles, block_corner, cosmology_corner, covariance, fields, members, scaling_plot, "
                        "globals_corner")
    # With a run: whether each module is enabled and whether it applies, from its saved configuration.
    run = tmp_path / "run"
    outputs = synthetic_config.outputs
    config = replace(synthetic_config, outputs=replace(outputs, rgb=replace(outputs.rgb, image_dir=None),
                                                       truth=replace(outputs.truth, kappa_true_fits=None),
                                                       mass=replace(outputs.mass, enabled=False)))
    artifacts.save_config(run / "final" / "artifacts", config)
    assert cli.main(["outputs", "--list", str(run)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == ["module", "requires", "after", "enabled", "active"]
    rows = {line.split()[0]: line for line in out[1::2]}
    assert rows["fit"].split()[3:] == ["yes", "yes"] and rows["mass"].split()[3:] == ["no", "yes"]
    assert rows["truth"].split()[3] == "yes"
    assert rows["truth"].rstrip().endswith("no: no truth convergence or deflection map (outputs.truth.kappa_true_fits, "
                                           "outputs.truth.deflection_true_fits)")
    assert rows["rgb"].rstrip().endswith("no: no imaging (outputs.rgb.image_dir)")
    with pytest.raises(SystemExit):
        cli.main(["outputs", str(run), "--only", "nope"])
    with pytest.raises(SystemExit):
        cli.main(["outputs"])


# ----------------------------------------------------------------------------- run_summary across passes
class SummaryModule(OutputModule):
    """Writes ``summary[key]`` from the draws, or fails."""

    def __init__(self, name, key, requires=(), fail=False):
        self.name, self.summary_keys, self.requires, self.fail = name, (key,), tuple(requires), fail

    def compute(self, ctx):
        draws = ctx.draws
        if self.fail:
            raise RuntimeError(f"{self.name} broke")
        ctx.summary[self.summary_keys[0]] = {"value": f"{self.name} over {len(draws)} draws"}

    def product(self, ctx):
        return {}


def summary_pass(monkeypatch, tmp_path, modules, *, draws, only=None, disabled=()):
    from lumina.reporting import draws as draws_module, results_tex
    from lumina.reporting.outputs import write_run_summary

    class FakeDraws(list):
        seed, has_field, cache_stage = 7, False, None
        image_indices = list(range(2 * draws))

    monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: FakeDraws(range(draws)))
    monkeypatch.setattr(results_tex, "write_run_results", lambda *a, **k: None)
    ctx = context(tmp_path)
    ctx.config = toy_config()
    run_modules(ctx, modules, only=only, disabled=disabled)
    write_run_summary(ctx, partial=only is not None or bool(disabled), modules=modules)
    return store.read_record(tmp_path, "run_summary")


def test_a_partial_pass_leaves_no_stale_numbers(monkeypatch, tmp_path):
    images = SummaryModule("images", "fit")
    truth = SummaryModule("truth", "truth", requires=["images"])
    full = summary_pass(monkeypatch, tmp_path, [images, truth], draws=8)
    assert full["truth"] == {"value": "truth over 8 draws"} and full["reporting_draws"] == 8
    # --only truth, and truth fails: its old numbers go, images' are this pass's.
    broken = SummaryModule("truth", "truth", requires=["images"], fail=True)
    failed = summary_pass(monkeypatch, tmp_path, [images, broken], draws=8, only=["truth"])
    assert "truth" not in failed and failed["modules"]["truth"]["status"] == "failed"
    # --only images --draws 4: the draws are those of this pass, truth keeps the record of the pass that ran it.
    again = summary_pass(monkeypatch, tmp_path, [images, broken], draws=4, only=["images"])
    assert again["reporting_draws"] == 4 and again["fit"] == {"value": "images over 4 draws"}
    assert "truth" not in again and again["modules"]["truth"] == failed["modules"]["truth"]
    assert again["modules"]["images"]["reporting_draws"] == 4 and again["modules"]["images"]["image_draws"] == 8
    assert list(again["modules"]) == ["images", "truth"]           # registry order, not saved order
    # A module not in the pass keeps its numbers and its record.
    fixed = SummaryModule("truth", "truth", requires=["images"])
    summary_pass(monkeypatch, tmp_path, [images, fixed], draws=4, only=["truth"])
    kept = summary_pass(monkeypatch, tmp_path, [images, fixed], draws=4, only=["images"])
    assert kept["truth"] == {"value": "truth over 4 draws"} and kept["modules"]["truth"]["status"] == "done"


def test_a_disabled_module_keeps_its_numbers_and_says_disabled(monkeypatch, tmp_path):
    images = SummaryModule("images", "fit")
    truth = SummaryModule("truth", "truth", requires=["images"])
    summary_pass(monkeypatch, tmp_path, [images, truth], draws=8)
    off = summary_pass(monkeypatch, tmp_path, [images, truth], draws=4, disabled=["truth"])
    assert off["truth"] == {"value": "truth over 8 draws"} and off["fit"] == {"value": "images over 4 draws"}
    assert off["modules"]["truth"] == {"status": "disabled", "seconds": 0.0, "reason": "outputs.truth.enabled is false"}


def test_fail_fast_still_writes_the_summary_and_compacts(monkeypatch, synthetic_config, tmp_path):
    from lumina.io import store as store_module
    from lumina.outputs import fit
    from lumina.reporting import outputs, results_tex
    compacted = []
    monkeypatch.setattr(store_module, "compact", lambda stage: compacted.append(stage))
    monkeypatch.setattr(results_tex, "write_run_results", lambda *a, **k: None)
    monkeypatch.setattr(fit, "posterior_summary_table", lambda *a: (_ for _ in ()).throw(RuntimeError("table broke")))
    paths = SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures", root=tmp_path)
    with pytest.raises(RuntimeError, match="table broke"):
        outputs.write_outputs(synthetic_config, SimpleNamespace(parameter_specs=[]), None, None, {}, paths, only=["fit"],
                              fail_fast=True, log=lambda _: None)
    assert compacted == [tmp_path]
    assert store.read_record(tmp_path, "run_summary")["modules"]["fit"]["steps"]["tables"]["status"] == "failed"


# ----------------------------------------------------------------------------- lumina outputs on a saved run
def test_module_selection_on_the_command_line(monkeypatch, synthetic_config, tmp_path):
    from lumina.io import store as store_module
    from lumina.reporting import outputs
    seen = []
    monkeypatch.setattr(outputs, "run_modules", lambda ctx, modules, **k: seen.append(k) or {})
    monkeypatch.setattr(outputs, "write_run_summary", lambda *a, **k: None)
    monkeypatch.setattr(store_module, "compact", lambda stage: None)
    settings = synthetic_config.outputs
    off = replace(synthetic_config, outputs=replace(settings, truth=replace(settings.truth, enabled=False),
                                                    rgb=replace(settings.rgb, enabled=False)))
    paths = SimpleNamespace(tables=tmp_path, figures=tmp_path, root=tmp_path)

    def selection(**options):
        outputs.write_outputs(off, None, None, None, {}, paths, log=lambda _: None, **options)
        return {key: seen[-1][key] for key in ("only", "skip", "disabled", "fail_fast")}

    # Nothing given: the enabled switches of the configuration.
    assert selection() == {"only": None, "skip": (), "disabled": ("truth", "rgb"), "fail_fast": False}
    # --only and --all replace them: an explicit request wins.
    assert selection(only=["rgb"]) == {"only": ("rgb",), "skip": (), "disabled": (), "fail_fast": False}
    assert selection(all_modules=True, fail_fast=True) == {"only": None, "skip": (), "disabled": (), "fail_fast": True}
    # --skip goes on top of the switches (a disabled module stays off), or of --all.
    assert selection(skip=["fit"]) == {"only": None, "skip": ("fit",), "disabled": ("truth", "rgb"), "fail_fast": False}
    assert selection(all_modules=True, skip=["rgb"]) == {"only": None, "skip": ("rgb",), "disabled": (), "fail_fast": False}


def test_skip_keeps_disabled_modules_off_in_a_real_pass(monkeypatch, synthetic_config, tmp_path):
    from lumina.io import store as store_module
    from lumina.reporting import outputs
    monkeypatch.setattr(outputs, "write_run_summary", lambda *a, **k: None)
    monkeypatch.setattr(store_module, "compact", lambda stage: None)
    settings = synthetic_config.outputs
    off = replace(synthetic_config, outputs=replace(settings, rgb=replace(settings.rgb, enabled=False)))
    paths = SimpleNamespace(tables=tmp_path, figures=tmp_path, root=tmp_path)
    modules = [Fake("fit"), Fake("images"), Fake("mass", requires=["images"]), Fake("truth", requires=["images"]),
               Fake("rgb", requires=["images"])]
    monkeypatch.setattr("lumina.outputs.registry.MODULES", tuple(modules))
    summary = outputs.write_outputs(off, None, None, None, {}, paths, log=lambda _: None, skip=["truth"])
    statuses = {name: record["status"] for name, record in summary["modules"].items()}
    assert statuses == {"fit": "done", "images": "done", "mass": "done", "truth": "skipped", "rgb": "disabled"}
    summary = outputs.write_outputs(off, None, None, None, {}, paths, log=lambda _: None, all_modules=True, skip=["truth"])
    assert {name: record["status"] for name, record in summary["modules"].items()}["rgb"] == "done"


def test_remap_and_report_overrides(synthetic_config, tmp_path):
    from lumina.outputs.saved_run import apply_overrides, remap_paths
    outputs = synthetic_config.outputs
    config = replace(synthetic_config, outputs=replace(outputs, truth=replace(outputs.truth, kappa_true_fits="/far/away/data/kappa.fits"),
                                                       rgb=replace(outputs.rgb, image_dir="/far/away/data/imaging")))
    lines = []
    moved = remap_paths(config, [("/far/away", str(tmp_path))], log=lines.append)
    assert (moved.outputs.truth.kappa_true_fits == f"{tmp_path}/data/kappa.fits"
            and moved.outputs.rgb.image_dir == f"{tmp_path}/data/imaging")
    assert sorted(line.split()[2].rstrip(":") for line in lines) == ["outputs.rgb.image_dir", "outputs.truth.kappa_true_fits"]
    assert remap_paths(config, [("/far/aw", "/x")], log=lambda _: None).outputs.truth.kappa_true_fits == "/far/away/data/kappa.fits"
    # Any field of the outputs section may change; nothing else may, also outside the run hash (paths, WBIC).
    changed = apply_overrides(config, ["outputs.truth.deflection_true_fits=defl.fits", "outputs.rgb.enabled=false",
                                       "outputs.workers=2", "outputs.truth.kappa_true_fits=other.fits", "outputs.draws=4"])
    assert changed.outputs.truth.deflection_true_fits == "defl.fits" and not changed.outputs.rgb.enabled
    assert (changed.outputs.workers, changed.outputs.truth.kappa_true_fits, changed.outputs.draws) == (2, "other.fits", 4)
    assert changed.to_json_dict() | {"outputs": None} == config.to_json_dict() | {"outputs": None}
    for bad in ("sampling.warmup=3", "reporting.draws=4", "truth.kappa_true_fits=other.fits", "paths.figure_prefix=../../X",
                "paths.run_name=other", "paths.output_dir=/tmp", "sampling.wbic=true", "runtime.seed=3",
                "runtime.output_workers=2", "truth=5", "nosuch.field=1"):
        with pytest.raises(ValueError, match=r"^--set may change only the outputs section \(outputs\.\*\); refused: "):
            apply_overrides(config, [bad])
    for bad in ("outputs.nosuch=1", "outputs.only=[fit]", "outputs.rgb.rgb={q: 1}"):
        with pytest.raises(ValueError, match="^--set: "):
            apply_overrides(config, [bad])
    # A setting of the wrong type is refused before the pass, with the field named.
    for bad, field in (("outputs.truth.deflection_true_fits={a: 1}", "outputs.truth.deflection_true_fits"),
                       ("outputs.truth.deflection_true_fits=[a, b]", "outputs.truth.deflection_true_fits"),
                       ("outputs.truth.deflection_true_fits.x=1", "outputs.truth.deflection_true_fits"),
                       ("outputs.truth.deflection_true_fits=true", "outputs.truth.deflection_true_fits"),
                       ("outputs.workers=abc", "outputs.workers"), ("outputs.workers=-5", "outputs.workers"),
                       ("outputs.workers=2.5", "outputs.workers"), ("outputs.rgb.enabled=3", "outputs.rgb.enabled"),
                       ("outputs.fail_fast=3", "outputs.fail_fast"), ("outputs=5", "outputs")):
        with pytest.raises(ValueError, match=rf"^--set: {field.replace('.', r'.')} "):
            apply_overrides(config, [bad])


def test_a_bad_report_setting_is_a_usage_error(synthetic_config, tmp_path, capsys):
    from lumina import cli
    from lumina.io import artifacts
    run = tmp_path / "run"
    artifacts.save_config(run / "final" / "artifacts", synthetic_config)
    assert cli.main(["outputs", str(run), "--set", "outputs.workers=-5"]) == 2
    assert "outputs.workers must be a positive integer" in capsys.readouterr().out


def test_missing_files_name_the_field_and_suggest_remap(synthetic_config):
    from lumina.outputs.command import check_module_paths
    outputs = synthetic_config.outputs
    config = replace(synthetic_config, outputs=replace(outputs, truth=replace(outputs.truth, kappa_true_fits="/far/away/kappa.fits")))
    with pytest.raises(FileNotFoundError, match=r"outputs.truth.kappa_true_fits = /far/away/kappa.fits \(module mass\).*--remap"):
        check_module_paths(config)
    check_module_paths(config, only=("fit",))                  # fit reads no such file
    # Disabled modules read nothing; --only runs (and checks) a module whatever its switch.
    off = replace(config, outputs=replace(config.outputs, **{name: replace(getattr(config.outputs, name), enabled=False)
                                                               for name in ("mass", "truth", "rgb")}))
    check_module_paths(off)
    with pytest.raises(FileNotFoundError, match=r"\(module mass\)"):
        check_module_paths(off, only=("mass",))
    # --skip keeps the disabled ones off; --all checks them, and --all --skip leaves out the skipped ones.
    check_module_paths(off, skip=("fit",))
    with pytest.raises(FileNotFoundError, match=r"\(module mass\)"):
        check_module_paths(off, all_modules=True)
    check_module_paths(off, all_modules=True, skip=("mass", "truth", "rgb"))
    # A module that requires a skipped one is not run, so not checked.
    check_module_paths(config, skip=("images",))


def test_outputs_command_options(monkeypatch, tmp_path, capsys):
    from lumina.outputs import command
    with pytest.raises(SystemExit):
        command.main([str(tmp_path), "--all", "--only", "fit"])
    seen = {}

    def prepare(run, **options):
        seen.update(options)
        return ("config",)

    monkeypatch.setattr(command, "prepare_rerun", prepare)
    from lumina.reporting import outputs
    store.write_record(tmp_path / "final", "run_summary", {"fit": {"rms": 0.25}})
    monkeypatch.setattr(outputs, "write_outputs", lambda *a, **k: {"modules": {"fit": {"status": "done"}}})
    assert command.main([str(tmp_path), "--only", "fit", "--draws", "4", "--remap", "/a=/b", "--set", "outputs.fail_fast=true"]) == 0
    assert seen["only"] == ("fit",) and seen["draws"] == 4 and seen["remap"] == [("/a", "/b")]
    assert seen["overrides"] == ["outputs.fail_fast=true"]
    # The fit numbers printed are the saved record's, which a pass of fit alone does not touch.
    assert json.loads(capsys.readouterr().out) == {"rms": 0.25}
    # --fail-fast: one line, exit code 1.
    monkeypatch.setattr(outputs, "write_outputs", lambda *a, **k: (_ for _ in ()).throw(ValueError("step broke")))
    settings = SimpleNamespace(outputs=SimpleNamespace(fail_fast=True))
    monkeypatch.setattr(command, "prepare_rerun", lambda run, **options: (settings,))
    assert command.main([str(tmp_path), "--fail-fast"]) == 1
    assert capsys.readouterr().out.strip() == "lumina outputs: stopped at the first error (fail_fast: ValueError: step broke)"
    # Without fail_fast an error can only come from outside the modules, which nothing logged: its traceback.
    settings.outputs.fail_fast = False
    assert command.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("lumina outputs: the pass failed outside the modules (ValueError: step broke)")
    assert "Traceback" in out
    monkeypatch.setattr(command, "prepare_rerun", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x not found")))
    assert command.main([str(tmp_path)]) == 2


def test_draws_option_sets_both_draw_counts(monkeypatch, synthetic_config, tmp_path):
    from lumina.outputs import command, saved_run
    monkeypatch.setattr(saved_run, "load_run", lambda run, **k: (synthetic_config, None, None, None, None))
    monkeypatch.setattr(command, "check_module_paths", lambda config, **k: None)
    config, *_ = command.prepare_rerun(tmp_path, draws=4)
    assert (config.outputs.draws, config.outputs.image_draws) == (4, 4)


# ----------------------------------------------------------------------------- compute, product and render
class Phased(OutputModule):
    """A converted module: compute writes the table ``<name>_table`` (step ``table``) and nothing (step ``nothing``),
    product reads the table back, render draws ``<name>.pdf`` from it (step ``figure``).  ``use`` names what a phase
    asks for that it should not: ``{"render": "draws"}``."""

    compute_steps = ("table", "nothing")
    render_steps = ("figure", "unused")

    def __init__(self, name, requires=(), events=None, fail=(), use=None, error=None):
        self.name, self.requires = name, tuple(requires)
        self.events = events if events is not None else []
        self.fail, self.use, self.error = set(fail), dict(use or {}), error

    def _touch(self, ctx, phase):
        if phase in self.use:
            getattr(ctx, self.use[phase])

    def compute(self, ctx):
        self.events.append(f"{self.name}.compute")

        def table():
            self._touch(ctx, "compute")
            if "table" in self.fail:
                raise ValueError("table broke")
            store.write_table(ctx.stage, f"{self.name}_table", pd.DataFrame({"x": [1.0, 2.0]}))
            return "written"
        ctx.step("table", table)
        ctx.step("nothing", lambda: None)
        if self.error is not None:
            raise self.error

    def product(self, ctx):
        self.events.append(f"{self.name}.product")
        self._touch(ctx, "product")
        return store.read_table(ctx.stage, f"{self.name}_table")

    def render(self, ctx):
        self.events.append(f"{self.name}.render")

        def figure():
            self._touch(ctx, "render")
            ctx.figures.mkdir(parents=True, exist_ok=True)
            (ctx.figures / f"{self.name}.pdf").write_text(str(store.read_table(ctx.stage, f"{self.name}_table").x.sum()))
            return "drawn"
        ctx.render_step("figure", figure, needs="table")
        ctx.render_step("unused", lambda: self.events.append("unused drawn"), needs="nothing")


def test_a_full_pass_runs_compute_product_render_in_order(tmp_path):
    events = []
    modules = [Phased("a", events=events), Phased("b", requires=["a"], events=events)]
    ctx = context(tmp_path)
    records = run_modules(ctx, modules)
    assert events == ["a.compute", "a.product", "a.render", "b.compute", "b.product", "b.render"]
    # The product is what product() loaded from the store.
    assert ctx.products["a"].x.tolist() == [1.0, 2.0] and (tmp_path / "figures" / "a.pdf").read_text() == "3.0"
    record = records["a"]
    assert record["status"] == "done" and record["mode"] == "full"
    assert set(record["steps"]) == {"table", "nothing"} and record["steps"]["table"]["wrote"] is True
    assert record["steps"]["nothing"]["wrote"] is False
    # A render step whose compute step wrote nothing is left out; the step that drew is recorded under render_steps.
    assert list(record["render_steps"]) == ["figure"] and "unused drawn" not in events
    completion = store.read_record(tmp_path, completion_name("a"))
    assert completion["complete"] is True and completion["steps"]["table"]["status"] == "done" and "computed_at" in completion


def test_a_render_pass_never_computes_and_redraws_from_the_store(tmp_path):
    events = []
    run_modules(context(tmp_path), [Phased("a", events=events)])
    (tmp_path / "figures" / "a.pdf").unlink()
    events.clear()
    ctx = context(tmp_path, mode="render")
    records = run_modules(ctx, [Phased("a", events=events)])
    assert events == ["a.product", "a.render"] and (tmp_path / "figures" / "a.pdf").read_text() == "3.0"
    assert records["a"]["status"] == "done" and records["a"]["mode"] == "render"
    # The compute steps are those of the saved completeness record.
    assert records["a"]["steps"]["table"]["status"] == "done" and list(records["a"]["render_steps"]) == ["figure"]


def test_a_compute_pass_draws_no_figures(tmp_path):
    events = []
    ctx = context(tmp_path, mode="compute")
    records = run_modules(ctx, [Phased("a", events=events)])
    assert events == ["a.compute", "a.product"] and not (tmp_path / "figures").exists()
    assert "render_steps" not in records["a"] and ctx.products["a"] is not None
    assert store.read_record(tmp_path, completion_name("a"))["complete"] is True


@pytest.mark.parametrize("what", ["draws", "evaluator"])
def test_draws_and_evaluator_raise_outside_compute(tmp_path, what):
    lines = []
    # In the render phase of a full pass: the step fails and names the module and the step.
    records = run_modules(context(tmp_path, log=lines.append), [Phased("a", use={"render": what})])
    error = records["a"]["render_steps"]["figure"]["error"]
    assert error.startswith("ComputeOnly: a.figure asked for ctx." + what + " in the render phase of a full pass")
    # In product: the module fails without a product.
    records = run_modules(context(tmp_path), [Phased("a", use={"product": what}), Phased("b", requires=["a"])])
    assert records["a"]["status"] == "failed" and "ComputeOnly: a (product) asked for ctx." + what in records["a"]["error"]
    assert records["b"]["status"] == "skipped"
    # In a render pass even a compute step's request is refused (compute never runs there), and outside a module too.
    ctx = context(tmp_path, mode="render")
    with pytest.raises(ComputeOnly, match="outside any module"):
        getattr(ctx, what)
    # In a compute step of a full pass it is allowed.
    assert run_modules(context(tmp_path), [Phased("a", use={"compute": "evaluator"})])["a"]["status"] == "done"


def test_render_without_a_completed_compute_is_unavailable(tmp_path):
    events = []
    records = run_modules(context(tmp_path, mode="render"), [Phased("a", events=events), Phased("b", requires=["a"], events=events)])
    assert events == []
    assert records["a"]["status"] == "failed" and records["a"]["not_computed"] is True
    assert records["a"]["error"].startswith("NotComputed: no record output_module_a of a compute of a")
    assert "run its compute first: lumina outputs RUN --only a" in records["a"]["error"]
    assert records["b"]["reason"] == "requires a, which failed without a product"


def test_a_failed_compute_leaves_no_record_of_completeness(tmp_path):
    # A complete compute first, then one whose step fails: the record says incomplete and which step failed.
    run_modules(context(tmp_path), [Phased("a")])
    records = run_modules(context(tmp_path), [Phased("a", fail={"table"})])
    assert records["a"]["status"] == "failed" and records["a"]["error"] == "failed steps: table"
    # The full pass still rendered what it could: the figure step is skipped with its reason.
    assert records["a"]["render_steps"]["figure"] == {"status": "skipped", "seconds": 0.0, "reason": "compute step table failed"}
    saved = store.read_record(tmp_path, completion_name("a"))
    assert saved["complete"] is False and saved["steps"]["table"]["status"] == "failed" and saved["unavailable"] is None
    # A render pass draws what the last compute saved (nothing here), never the figure of the older complete compute.
    (tmp_path / "figures" / "a.pdf").unlink()
    records = run_modules(context(tmp_path, mode="render"), [Phased("a")])
    assert records["a"]["status"] == "failed" and records["a"]["error"] == "failed steps: table"
    assert records["a"]["render_steps"]["figure"]["status"] == "skipped" and not (tmp_path / "figures" / "a.pdf").exists()
    # A compute that raises outside its steps.
    run_modules(context(tmp_path), [Phased("a")])
    run_modules(context(tmp_path), [Phased("a", error=ModuleUnavailable("no table"))])
    saved = store.read_record(tmp_path, completion_name("a"))
    assert saved["complete"] is False and saved["unavailable"] == "ModuleUnavailable: no table"
    # A compute interrupted half way (no chance to write the end record) leaves the incomplete marker.
    run_modules(context(tmp_path), [Phased("a")])
    with pytest.raises(KeyboardInterrupt):
        run_modules(context(tmp_path), [Phased("a", error=KeyboardInterrupt())])
    saved = store.read_record(tmp_path, completion_name("a"))
    assert saved["complete"] is False and "started_at" in saved and "steps" not in saved


def test_only_in_a_render_pass_takes_just_the_product_of_a_required_module(tmp_path):
    events = []
    run_modules(context(tmp_path), [Phased("a"), Phased("b", requires=["a"])])
    ctx = context(tmp_path, mode="render")
    records = run_modules(ctx, [Phased("a", events=events), Phased("b", requires=["a"], events=events)], only=["b"])
    assert events == ["a.product", "b.product", "b.render"]
    assert ctx.products["a"].x.tolist() == [1.0, 2.0] and "render_steps" not in records["a"]
    # In a full pass the module pulled in by only runs entirely, as before.
    events.clear()
    run_modules(context(tmp_path), [Phased("a", events=events), Phased("b", requires=["a"], events=events)], only=["b"])
    assert events == ["a.compute", "a.product", "a.render", "b.compute", "b.product", "b.render"]


def test_an_undeclared_step_is_an_error(tmp_path):
    class Wrong(Phased):
        def compute(self, ctx):
            ctx.step("other", lambda: 1)

    records = run_modules(context(tmp_path), [Wrong("a")])
    assert "compute step 'other' is not in its compute_steps ['table', 'nothing']" in records["a"]["error"]


def test_a_render_pass_keeps_the_numbers_of_the_last_compute(monkeypatch, tmp_path):
    from lumina.reporting import results_tex
    from lumina.reporting.outputs import write_run_summary
    monkeypatch.setattr(results_tex, "write_run_results", lambda *a, **k: None)

    class Numbers(Phased):
        summary_keys = ("numbers",)

        def compute(self, ctx):
            super().compute(ctx)
            ctx.summary["numbers"] = {"value": 7}

    modules = [Numbers("a")]
    ctx = context(tmp_path)
    ctx.config = toy_config()
    run_modules(ctx, modules)
    write_run_summary(ctx, partial=False, modules=modules)
    ctx = context(tmp_path, mode="render")
    ctx.config = toy_config()
    run_modules(ctx, modules)
    write_run_summary(ctx, partial=True, modules=modules)
    saved = store.read_record(tmp_path, "run_summary")
    assert saved["numbers"] == {"value": 7} and saved["modules"]["a"]["mode"] == "render"


def test_render_and_compute_flags(monkeypatch, tmp_path, capsys):
    from lumina.outputs import command
    from lumina.reporting import outputs
    for argv in ([str(tmp_path), "--render", "--compute"], [str(tmp_path), "--render", "--draws", "4"]):
        with pytest.raises(SystemExit):
            command.main(argv)
    seen = []
    drew = {"modules": {"a": {"status": "done", "render_steps": {"figure": {"status": "done"}}}}}
    monkeypatch.setattr(command, "prepare_rerun", lambda run, **options: seen.append(options["mode"]) or ("config",))
    monkeypatch.setattr(outputs, "write_outputs", lambda *a, **k: seen.append(k["mode"]) or drew)
    for flag, mode in (("--render", "render"), ("--compute", "compute"), (None, "full")):
        assert command.main([str(tmp_path)] + ([flag] if flag else [])) == 0
        assert seen[-2:] == [mode, mode]
    # A render pass that draws nothing says so and fails.
    drew["modules"]["a"] = {"status": "failed", "error": "ModuleUnavailable: no record"}
    capsys.readouterr()
    assert command.main([str(tmp_path), "--render"]) == 1
    assert "the render pass drew nothing" in capsys.readouterr().out


def test_a_render_rerun_loads_no_evaluator(monkeypatch, synthetic_config, tmp_path):
    from lumina.outputs import command, saved_run
    monkeypatch.setattr(saved_run, "load_run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("evaluator built")))
    monkeypatch.setattr(saved_run, "load_saved_run", lambda run, **k: (synthetic_config, "state", "posterior"))
    monkeypatch.setattr(command, "check_module_paths", lambda config, **k: (_ for _ in ()).throw(AssertionError("paths checked")))
    config, state, evaluator, posterior, audit, final = command.prepare_rerun(tmp_path, mode="render")
    assert (state, evaluator, posterior) == ("state", None, "posterior") and final.root == tmp_path / "final"
    with pytest.raises(ValueError, match="--draws"):
        command.prepare_rerun(tmp_path, mode="render", draws=4)


def test_remap_moves_only_path_fields(synthetic_config, tmp_path, monkeypatch):
    from lumina.outputs.registry import MODULES as registry
    from lumina.outputs.saved_run import catalog_fields, path_fields, remap_paths
    outputs = synthetic_config.outputs
    config = replace(synthetic_config, paths=replace(synthetic_config.paths, figure_prefix="/far/away/prefix",
                                                     run_name="/far/away/name"),
                     outputs=replace(outputs, truth=replace(outputs.truth, kappa_true_fits="/far/away/kappa.fits")))
    covered = set(path_fields(config))
    # Every path a module checks and every catalog the model is built from is a path field.
    declared = {f"outputs.{field}" for module in registry for field in module.path_fields} | set(catalog_fields(config))
    assert declared <= covered, declared - covered
    assert not {"paths.figure_prefix", "paths.run_name"} & covered
    moved = remap_paths(config, [("/far/away", "/near")], log=lambda _: None)
    assert moved.outputs.truth.kappa_true_fits == "/near/kappa.fits"
    assert (moved.paths.figure_prefix, moved.paths.run_name) == ("/far/away/prefix", "/far/away/name")
    # A relative saved path is compared as the absolute path it names from the working directory.
    monkeypatch.chdir(tmp_path)
    relative = replace(config, outputs=replace(config.outputs, rgb=replace(config.outputs.rgb, image_dir="data/imaging/hst")))
    moved = remap_paths(relative, [(str(tmp_path / "data" / "imaging"), "/empty")], log=lambda _: None)
    assert moved.outputs.rgb.image_dir == "/empty/hst"
    for old, match in (("far/away", "must be an absolute path"), ("/", "is the root")):
        with pytest.raises(ValueError, match=match):
            remap_paths(config, [(old, "/x")], log=lambda _: None)


def test_a_module_skipped_after_a_failure_loses_its_numbers(monkeypatch, tmp_path):
    images = SummaryModule("images", "fit")
    truth = SummaryModule("truth", "truth", requires=["images"])
    full = summary_pass(monkeypatch, tmp_path, [images, truth], draws=8)
    assert full["truth"] == {"value": "truth over 8 draws"} and full["modules"]["truth"]["status"] == "done"
    # Pass 2, every module: images fails, so truth is skipped; its old numbers must not survive.
    broken = SummaryModule("images", "fit", fail=True)
    second = summary_pass(monkeypatch, tmp_path, [broken, truth], draws=8)
    assert second["modules"]["images"]["status"] == "failed" and "fit" not in second
    assert second["modules"]["truth"]["status"] == "skipped" and "truth" not in second
    assert second["modules"]["truth"]["reason"] == "requires images, which failed without a product"
    # A module the user skipped keeps its saved record and numbers.
    summary_pass(monkeypatch, tmp_path, [images, truth], draws=8)
    ctx = context(tmp_path)
    ctx.config = toy_config()
    from lumina.reporting.outputs import write_run_summary
    run_modules(ctx, [images, truth], skip=["images"])
    write_run_summary(ctx, partial=True, modules=[images, truth])
    kept = store.read_record(tmp_path, "run_summary")
    assert kept["truth"] == {"value": "truth over 8 draws"} and kept["modules"]["truth"]["status"] == "done"


class Keyed(Phased):
    """A Phased module whose compute uses reporting draws with the key ``draws:<count>``."""

    def compute(self, ctx):
        ctx.draws
        super().compute(ctx)


def test_modules_computed_over_other_draws_are_named(monkeypatch, tmp_path):
    from lumina.reporting import draws as draws_module

    def fake(count):
        class FakeDraws(list):
            seed, has_field, cache_stage, key = 7, False, None, f"draws{count:04d}"
            image_indices = list(range(count))
        return FakeDraws(range(count))

    modules = [Keyed("a"), Keyed("b", requires=["a"])]
    monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: fake(8))
    ctx = context(tmp_path)
    run_modules(ctx, modules)
    assert "warnings" not in ctx.summary
    # a computed again over 4 draws: b is now stale, and a render pass says so too (without blocking it).
    monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: fake(4))
    lines = []
    ctx = context(tmp_path, log=lines.append)
    run_modules(ctx, modules, only=["a"])
    warning = ctx.summary["warnings"][-1]
    assert warning.startswith("modules computed over different reporting draws: a (4 draws, seed 7, key draws000)")
    assert "b (8 draws, seed 7, key draws000)" in warning and "Stale" in warning and "b (requires a)" in warning
    assert any(line.startswith("[outputs] WARNING: modules computed over different reporting draws") for line in lines)
    ctx = context(tmp_path, mode="render")
    records = run_modules(ctx, modules)
    assert records["a"]["status"] == records["b"]["status"] == "done" and ctx.summary["warnings"] == [warning]


def test_render_step_names_are_checked(tmp_path):
    class WrongName(Phased):
        def render(self, ctx):
            ctx.render_step("figur", lambda: None, needs="table")

    class WrongNeed(Phased):
        def render(self, ctx):
            ctx.render_step("figure", lambda: None, needs="tabel")

    run_modules(context(tmp_path), [Phased("a")])
    for module, message in ((WrongName("a"), "render step 'figur' is not in its render_steps"),
                            (WrongNeed("a"), "compute step 'tabel' is not in its compute_steps")):
        records = run_modules(context(tmp_path, mode="render"), [module])
        assert records["a"]["status"] == "failed" and message in records["a"]["error"]


def test_a_skipped_compute_step_skips_only_its_own_figures(tmp_path):
    class Skips(Phased):
        def compute(self, ctx):
            super().compute(ctx)
            ctx.skip_step("nothing", "table failed")

    records = run_modules(context(tmp_path), [Skips("a")])
    assert store.read_record(tmp_path, completion_name("a"))["complete"] is False
    # The figure of the table step is drawn; the one of the skipped step is not, and that fails the module.
    assert records["a"]["render_steps"]["figure"]["status"] == "done"
    assert records["a"]["render_steps"]["unused"] == {"status": "skipped", "seconds": 0.0, "reason": "compute step nothing skipped"}
    assert records["a"]["status"] == "failed" and records["a"]["error"] == "render steps not drawn: unused"
    (tmp_path / "figures" / "a.pdf").unlink()
    records = run_modules(context(tmp_path, mode="render"), [Phased("a")])
    assert (tmp_path / "figures" / "a.pdf").read_text() == "3.0"
    assert records["a"]["status"] == "failed" and records["a"]["error"] == "render steps not drawn: unused"


class Unavailable(Phased):
    """A Phased module whose second step fails, after which its compute raises ModuleUnavailable (like truth after a
    failed truth_maps): its first step's figure must still be drawn, in the full pass and in a later render pass."""

    compute_steps = ("table", "nothing", "maps")
    render_steps = ("figure", "unused", "maps_figure")

    def compute(self, ctx):
        super().compute(ctx)
        ctx.step("maps", lambda: (_ for _ in ()).throw(ValueError("maps broke")))
        raise ModuleUnavailable("no maps (the maps step failed)")

    def render(self, ctx):
        super().render(ctx)
        ctx.render_step("maps_figure", lambda: self.events.append("maps drawn"), needs="maps")


def test_a_module_unavailable_after_some_steps_draws_what_they_saved(tmp_path):
    events = []
    modules = [Unavailable("a", events=events), Phased("b", requires=["a"], events=events)]
    records = run_modules(context(tmp_path), modules)
    assert events == ["a.compute", "a.render"] and (tmp_path / "figures" / "a.pdf").read_text() == "3.0"
    assert records["a"]["status"] == "failed" and records["a"]["error"] == "ModuleUnavailable: no maps (the maps step failed)"
    assert records["a"]["render_steps"]["maps_figure"]["reason"] == "compute step maps failed"
    assert records["b"]["status"] == "skipped" and records["b"]["reason"] == "requires a, which failed without a product"
    saved = store.read_record(tmp_path, completion_name("a"))
    assert saved["unavailable"] == "ModuleUnavailable: no maps (the maps step failed)" and saved["complete"] is False
    # A render pass after it: the figure comes back, the module is failed without a product, b is skipped.
    (tmp_path / "figures" / "a.pdf").unlink()
    events.clear()
    lines = []
    ctx = context(tmp_path, log=lines.append, mode="render")
    records = run_modules(ctx, modules)
    assert events == ["a.render"] and (tmp_path / "figures" / "a.pdf").read_text() == "3.0" and "a" not in ctx.products
    assert records["a"]["status"] == "failed" and records["a"]["error"].startswith(
        "ModuleUnavailable: the last compute of a gave no product (ModuleUnavailable: no maps (the maps step failed))")
    assert records["a"]["render_steps"]["figure"]["status"] == "done" and records["b"]["status"] == "skipped"
    assert any(line == "[outputs] a.maps_figure (render): skipped (compute step maps failed)" for line in lines)
    # Only what a render step needs decides: --only b in a render pass still takes no product from a.
    records = run_modules(context(tmp_path, mode="render"), modules, only=["b"])
    assert records["a"]["status"] == "failed" and records["b"]["status"] == "skipped"


def test_a_render_pass_after_an_interrupted_compute_refuses_the_module(tmp_path):
    run_modules(context(tmp_path), [Phased("a")])
    with pytest.raises(KeyboardInterrupt):
        run_modules(context(tmp_path), [Phased("a", error=KeyboardInterrupt())])
    lines = []
    records = run_modules(context(tmp_path, log=lines.append, mode="render"), [Phased("a")])
    assert records["a"]["not_computed"] is True and "was interrupted before it recorded its steps" in records["a"]["error"]
    assert any(line.startswith("[outputs] WARNING: this render pass drew nothing: nothing was computed for the selected "
                               "modules (a)") for line in lines)


def test_why_a_render_pass_drew_nothing():
    from lumina.outputs.base import nothing_drawn_reason
    assert nothing_drawn_reason({"a": {"status": "disabled"}}).startswith("no selected module ran")
    assert nothing_drawn_reason({"a": {"status": "failed", "not_computed": True}}).startswith(
        "nothing was computed for the selected modules (a)")
    imaging = {"status": "failed", "render_steps": {"imaging:jwst": {"status": "failed", "input": True,
                                                                     "error": "FileNotFoundError: no folder x"}}}
    assert nothing_drawn_reason({"images": {"status": "done"}, "rgb": imaging}) == (
        "the compute is there, but an input needed to draw is missing: rgb.imaging:jwst (FileNotFoundError: no folder x)")
    skipped = {"status": "failed", "render_steps": {"figure": {"status": "skipped", "reason": "compute step table failed"}}}
    assert nothing_drawn_reason({"a": skipped}) == (
        "the compute is there, but no figure could be drawn: a.figure (skipped: compute step table failed)")
    assert nothing_drawn_reason({"a": {"status": "done"}}) == "the last compute of the selected modules saved nothing to draw"


def test_a_step_note_is_logged_and_recorded(tmp_path):
    class Noting(Phased):
        def compute(self, ctx):
            super().compute(ctx)
            ctx.step("nothing", lambda: ctx.note("went without the intervals") or "wrote")

    lines = []
    records = run_modules(context(tmp_path, log=lines.append), [Noting("a")])
    assert records["a"]["steps"]["nothing"]["notes"] == ["went without the intervals"]
    assert "[outputs] a.nothing: went without the intervals" in lines
    assert "notes" not in records["a"]["steps"]["table"]
    with pytest.raises(RuntimeError, match="inside ctx.step"):
        context(tmp_path).note("outside")


def test_list_applies_set_and_remap(synthetic_config, tmp_path, capsys):
    from lumina.io import artifacts
    from lumina.outputs import command
    artifacts.save_config(tmp_path / "final" / "artifacts", synthetic_config)
    assert command.main(["--list", str(tmp_path), "--set", "outputs.mass.enabled=false"]) == 0
    rows = {line.split()[0]: line.split() for line in capsys.readouterr().out.splitlines() if not line.startswith(" ")}
    assert rows["mass"][3] == "no" and rows["fit"][3] == "yes"
    assert command.main(["--list", str(tmp_path), "--set", "outputs.nosuch=1"]) == 2
    with pytest.raises(SystemExit):
        command.main(["--list", "--set", "outputs.mass.enabled=false"])


def test_paths_are_checked_only_for_modules_that_will_run(synthetic_config):
    from lumina.outputs.command import check_module_paths
    outputs = synthetic_config.outputs
    config = replace(synthetic_config, outputs=replace(outputs, images=replace(outputs.images, enabled=False),
                                                       rgb=replace(outputs.rgb, image_dir="/far/away/imaging")))
    check_module_paths(config)                      # rgb requires the disabled images: it will not run
    with pytest.raises(FileNotFoundError, match=r"outputs.rgb.image_dir = /far/away/imaging \(module rgb\)"):
        check_module_paths(config, all_modules=True)


def test_a_module_skipped_for_a_disabled_requirement_replaces_its_saved_record(monkeypatch, tmp_path):
    images = SummaryModule("images", "fit")
    truth = SummaryModule("truth", "truth", requires=["images"])
    summary_pass(monkeypatch, tmp_path, [images, truth], draws=8)
    off = summary_pass(monkeypatch, tmp_path, [images, truth], draws=8, disabled=("images",))
    assert off["modules"]["images"]["status"] == "disabled"
    assert off["modules"]["truth"]["status"] == "skipped" and off["modules"]["truth"]["disabled_requirement"] == "images"
    assert off["modules"]["truth"]["reason"] == "requires images, which was disabled"
    # Nothing was recomputed and nothing failed: the numbers stay.
    assert off["truth"] == {"value": "truth over 8 draws"} and off["fit"] == {"value": "images over 8 draws"}


def test_the_outputs_section_is_validated_field_by_field(synthetic_config):
    from lumina.outputs.saved_run import apply_overrides
    for bad, message in (("outputs.rgb.display.q=abc", "outputs.rgb.display.q must be a finite number or null"),
                         ("outputs.rgb.bands=F1", "outputs.rgb.bands must be a list of strings or null"),
                         ("outputs.rgb.image_scale=99mas", "outputs.rgb.image_scale must be one of auto"),
                         ("outputs.rgb.paper_fold_cutouts=3", "outputs.rgb.paper_fold_cutouts must be a list of strings"),
                         ("outputs.rgb.paper_fold_cutouts=[x]", "outputs.rgb.paper_fold_cutouts: 'x' is not image:ID"),
                         ("outputs.rgb.figure_suffix=3", "outputs.rgb.figure_suffix must be a string"),
                         ("outputs.image_solver.exact_image_lm_max_iter=abc", "outputs.image_solver.exact_image_lm_max_iter must be an integer"),
                         ("outputs.image_solver.match_tolerance_arcsec=-1", "outputs.image_solver.match_tolerance_arcsec must be positive"),
                         ("outputs.rgb.bogus=1", "outputs.rgb.bogus: no such field"),
                         ("outputs.image_solver.bogus=1", "outputs.image_solver.bogus: no such field")):
        with pytest.raises(ValueError, match="^--set: " + re.escape(message)):
            apply_overrides(synthetic_config, [bad])
    assert apply_overrides(synthetic_config, ["outputs.rgb.bands=[F1, F2]"]).outputs.rgb.bands == ("F1", "F2")


# ----------------------------------------------------------------------------- settings of a pass, never persisted
def _digest(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["full", "compute", "render"])
@pytest.mark.parametrize("fails", [False, True])
def test_the_outputs_command_never_writes_the_saved_config(monkeypatch, synthetic_config, tmp_path, mode, fails):
    """--set and --draws hold for one pass: config.json keeps its bytes after any pass mode, successful or failed,
    through the command line and through rerun_outputs."""
    from lumina.outputs import command, saved_run
    from lumina.reporting import outputs
    directory = tmp_path / "final" / "artifacts"
    artifacts.save_config(directory, synthetic_config)
    before = _digest(directory / artifacts.CONFIG_FILE)
    seen = []

    def load(run, *, remap=(), overrides=(), transform=None, log=print):
        return saved_run.apply_overrides(artifacts.load_config(Path(run) / "final" / "artifacts"), overrides), "state", "posterior"

    monkeypatch.setattr(saved_run, "load_saved_run", load)
    monkeypatch.setattr(saved_run, "load_run", lambda run, **k: (*load(run, **k)[:2], "evaluator", "posterior", None))
    monkeypatch.setattr(command, "check_module_paths", lambda config, **k: None)

    def write(config, *args, mode, **kwargs):
        seen.append((config.outputs.rgb.enabled, config.outputs.draws))
        if fails:
            raise ValueError("the pass broke")
        return {"modules": {"fit": {"status": "done", "render_steps": {"figure": {"status": "done"}}}}}

    monkeypatch.setattr(outputs, "write_outputs", write)
    argv = [str(tmp_path), "--set", "outputs.rgb.enabled=false"] + ({"compute": ["--compute", "--draws", "4"],
                                                                     "full": ["--draws", "4"], "render": ["--render"]}[mode])
    assert command.main(argv) == (1 if fails else 0)
    options = dict(overrides=["outputs.rgb.enabled=false"], **({} if mode == "render" else {"draws": 4}))
    if fails:
        with pytest.raises(ValueError, match="the pass broke"):
            command.rerun_outputs(tmp_path, mode=mode, log=lambda _: None, **options)
    else:
        command.rerun_outputs(tmp_path, mode=mode, log=lambda _: None, **options)
    # The pass saw the overrides; the saved configuration did not take them.
    assert seen == [(False, 64 if mode == "render" else 4)] * 2
    assert _digest(directory / artifacts.CONFIG_FILE) == before
    assert not hasattr(command, "save_outputs_settings")


class Setting(Phased):
    """A Phased module that records ``draws`` and its own section, and whose ``value`` only render reads when
    ``shown``."""

    shared_settings = ("draws",)

    def __init__(self, name, requires=(), shown=False):
        super().__init__(name, requires)
        self.render_settings = (f"{name}.value",) if shown else ()


def test_the_record_keeps_the_settings_of_its_compute_and_a_clean_pass_never_warns(tmp_path):
    modules = [Setting("a"), Setting("b", requires=["a"])]
    lines = []
    ctx = context(tmp_path, log=lines.append)
    records = run_modules(ctx, modules)
    for name in ("a", "b"):
        # Its shared fields and its own section, without the enabled switch; never workers or fail_fast.
        assert store.read_record(tmp_path, completion_name(name))["settings"] == {"draws": 8, f"{name}.value": 1}
        assert records[name]["settings"] == {"draws": 8, f"{name}.value": 1}
    assert "warnings" not in ctx.summary and not any("WARNING" in line for line in lines)
    # A render pass with the same settings: none either, and its records are those of the compute.
    ctx = context(tmp_path, log=lines.append, mode="render")
    records = run_modules(ctx, modules)
    assert "warnings" not in ctx.summary and records["a"]["settings"] == {"draws": 8, "a.value": 1}


def test_a_module_computed_with_other_settings_is_named_with_its_fields(tmp_path):
    modules = [Setting("a"), Setting("b", requires=["a"])]
    run_modules(context(tmp_path), modules)
    # --only a --draws 4 --set outputs.a.value=2 (a pass that runs only some modules): b keeps draws 8, value 1.
    lines = []
    ctx = context(tmp_path, log=lines.append, config=toy_config(draws=4, a=ToySection(value=2)))
    run_modules(ctx, modules, only=["a"])
    warnings = [w for w in ctx.summary["warnings"] if w.startswith("outputs settings changed")]
    assert len(warnings) == 1 and warnings[0].startswith("outputs settings changed since the last compute of b (at ")
    assert "outputs.draws 8 -> 4" in warnings[0] and "--only b" in warnings[0] and "a.value" not in warnings[0]
    assert f"[outputs] WARNING: {warnings[0]}" in lines
    # A later plain render pass (the saved settings again): a is now the one computed with other settings.
    lines.clear()
    ctx = context(tmp_path, log=lines.append, mode="render")
    records = run_modules(ctx, modules)
    assert records["a"]["status"] == records["b"]["status"] == "done"             # the note blocks nothing
    warnings = [w for w in ctx.summary["warnings"] if w.startswith("outputs settings changed")]
    assert len(warnings) == 1 and "of a (at " in warnings[0]
    assert "outputs.draws 4 -> 8" in warnings[0] and "outputs.a.value 2 -> 1" in warnings[0]
    # A module disabled in this pass is not compared.
    ctx = context(tmp_path, mode="render")
    run_modules(ctx, modules, disabled=["a", "b"])
    assert "warnings" not in ctx.summary


def test_a_field_only_render_reads_never_warns(tmp_path):
    modules = [Setting("a", shown=True)]
    run_modules(context(tmp_path), modules)
    ctx = context(tmp_path, mode="render", config=toy_config(a=ToySection(value=5)))
    run_modules(ctx, modules)
    assert "warnings" not in ctx.summary
    ctx = context(tmp_path, mode="render", config=toy_config(draws=2, a=ToySection(value=5)))
    run_modules(ctx, modules)
    assert ctx.summary["warnings"] == [w for w in ctx.summary["warnings"] if "outputs.draws 8 -> 2" in w and "a.value" not in w]


def test_settings_warnings_of_an_earlier_pass_are_replaced(tmp_path):
    """The settings note in run_summary is that of the last pass, as the draws warning."""
    from lumina.reporting.outputs import merged_summary
    modules = [Setting("a"), Setting("b", requires=["a"])]
    run_modules(context(tmp_path), modules)
    ctx = context(tmp_path, config=toy_config(draws=4))
    run_modules(ctx, modules, only=["a"])
    saved = merged_summary({}, ctx.summary, modules)
    assert any("of b" in w for w in saved["warnings"])
    ctx = context(tmp_path, config=toy_config(draws=4))
    run_modules(ctx, modules)                                   # both computed again with draws 4: nothing stale
    assert "warnings" not in merged_summary(saved, ctx.summary, modules)


def test_the_snapshot_of_each_real_module():
    from lumina.outputs.base import settings_snapshot
    outputs = OutputsConfig()
    snapshots = {module.name: settings_snapshot(outputs, module) for module in MODULES}
    for name, snapshot in snapshots.items():
        assert not any(key in ("workers", "fail_fast") or key.endswith(".enabled") for key in snapshot)
        assert {"draws", "image_draws"} <= set(snapshot)
    assert "image_solver.match_tolerance_arcsec" not in snapshots["fit"] and "caustic_source_redshift" in snapshots["fit"]
    assert {"image_solver.exact_image_finder", "images.extra_images", "truth.deflection_true_fits"} <= set(snapshots["images"])
    assert "caustic_source_redshift" not in snapshots["images"] and "truth.kappa_true_fits" not in snapshots["images"]
    assert {"truth.kappa_true_fits", "image_solver.exact_image_finder"} <= set(snapshots["mass"])
    assert {"rgb.display.q", "rgb.image_dir", "rgb.paper_fold_cutouts", "truth.kappa_true_fits"} <= set(snapshots["rgb"])
    assert module_named("rgb").render_settings == ("rgb.display",)
    # JSON values: tuples as lists, paths as given.
    path = replace(outputs, rgb=replace(outputs.rgb, image_dir=Path("data/x"), paper_fold_cutouts=("system:1",)))
    snapshot = settings_snapshot(path, module_named("rgb"))
    assert snapshot["rgb.image_dir"] == "data/x" and snapshot["rgb.paper_fold_cutouts"] == ["system:1"]


# ----------------------------------------------------------------------------- render after a compute stopped early
class TwoSteps(Phased):
    """Compute steps ``table`` then ``nothing``; render draws ``figure`` from ``table`` and ``unused`` from
    ``nothing``.  ``stop`` makes ``table`` raise."""

    def __init__(self, name, stop=False):
        super().__init__(name)
        self.stop = stop

    def compute(self, ctx):
        ctx.step("table", lambda: (_ for _ in ()).throw(ValueError("stopped here")) if self.stop else
                 store.write_table(ctx.stage, f"{self.name}_table", pd.DataFrame({"x": [1.0]})) or "written")
        ctx.step("nothing", lambda: "wrote")

    def render(self, ctx):
        ctx.render_step("figure", lambda: "drawn", needs="table")
        ctx.render_step("unused", lambda: "drawn", needs="nothing")


def test_a_render_after_a_fail_fast_compute_reports_the_steps_that_did_not_run(tmp_path):
    with pytest.raises(ValueError, match="stopped here"):
        run_modules(context(tmp_path), [TwoSteps("a", stop=True)], fail_fast=True)
    saved = store.read_record(tmp_path, completion_name("a"))
    assert list(saved["steps"]) == ["table"] and saved["unavailable"] == "ValueError: stopped here"
    lines = []
    records = run_modules(context(tmp_path, log=lines.append, mode="render"), [TwoSteps("a")])
    steps = records["a"]["render_steps"]
    assert steps["figure"] == {"status": "skipped", "seconds": 0.0, "reason": "compute step table failed"}
    assert steps["unused"] == {"status": "skipped", "seconds": 0.0, "reason": "compute step nothing did not run"}
    assert records["a"]["status"] == "failed" and "gave no product" in records["a"]["error"]


def test_why_a_render_pass_drew_nothing_names_modules_failed_outside_their_steps():
    from lumina.outputs.base import nothing_drawn_reason
    records = {"a": {"status": "failed", "error": "ModuleUnavailable: no entry a_table; run its compute first"},
               "b": {"status": "failed", "not_computed": True, "error": "NotComputed: no record"}}
    assert nothing_drawn_reason(records) == ("modules failed before drawing: a (ModuleUnavailable: no entry a_table; "
                                             "run its compute first) (never computed: b)")


def test_the_missing_imaging_of_each_variant_says_what_is_true(monkeypatch, tmp_path):
    from lumina.config import RGBOutputConfig
    from lumina.imaging import cutouts
    monkeypatch.setattr(cutouts, "load_cluster_imaging", lambda image_config, model, run: (_ for _ in ()).throw(
        FileNotFoundError(f"no folder {image_config.image_dir}")))
    configured = RGBOutputConfig(image_dir=Path("data/imaging/m0416/buffalo"))
    config = SimpleNamespace(outputs=SimpleNamespace(rgb=configured), model=None, paths=SimpleNamespace(run_name="m0416"))
    ctx = OutputContext(config, None, None, None, {}, None, log=lambda _: None, mode="render")
    # The configured folder, given as a string equal to the saved Path: the remappable field.
    with pytest.raises(FileNotFoundError, match=r"outputs.rgb.image_dir = data/imaging/m0416/buffalo .*--remap OLD=NEW"):
        ctx.imaging(replace(configured, image_dir="data/imaging/m0416/buffalo"), required=True)
    # The extra JWST variant: the absolute folder looked for, built relative to the working directory.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as raised:
        ctx.imaging(replace(configured, image_dir="data/imaging/m0416/canucs_dr1", figure_suffix="_jwst"), required=True)
    message = str(raised.value)
    assert f"loaded from {Path('data/imaging/m0416/canucs_dr1').absolute()} (" in message
    assert "relative to the working directory" in message and "--remap cannot move it" in message
    assert "run the command from the directory that contains data/imaging/m0416/canucs_dr1" in message
    assert "repository root" not in message


def test_a_bad_set_on_the_run_command_is_a_usage_error(monkeypatch, capsys):
    from lumina import cli
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert cli.main(["catalogs/hera", "--dry-run", "--set", "outputs.rgb.bogus=1"]) == 2
    error = capsys.readouterr().err
    assert error.startswith("lumina: outputs.rgb.bogus: no such field") and "Traceback" not in error
    assert cli.main(["catalogs/hera", "--dry-run", "--set", "outputs.workers=0"]) == 2
    assert "outputs.workers must be a positive integer" in capsys.readouterr().err
