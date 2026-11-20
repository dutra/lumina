"""The outputs stage as modules: ordering, selection, skipping, failures, lazy draws, the config section and the
``kappaccino outputs`` command."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from kappaccino.config import LensClusterSolverConfig, OutputsConfig
from kappaccino.io import artifacts, store
from kappaccino.outputs import MODULES, OutputContext, OutputModule, module_named, order_modules, plan_modules, run_modules
from kappaccino.outputs.base import ModuleUnavailable


class Fake(OutputModule):
    def __init__(self, name, requires=(), after=(), inactive=None, error=None, checkpoint=False, calls=None, draws=False):
        self.name, self.requires, self.after, self.checkpoint = name, tuple(requires), tuple(after), checkpoint
        self.inactive, self.error, self.calls, self.draws = inactive, error, calls if calls is not None else [], draws

    def active(self, ctx):
        return self.inactive

    def run(self, ctx):
        self.calls.append(self.name)
        if self.draws:
            ctx.draws
        if self.error is not None:
            raise self.error
        return f"product of {self.name}"


def context(tmp_path=None, log=None):
    paths = None if tmp_path is None else SimpleNamespace(tables=tmp_path / "tables", figures=tmp_path / "figures", root=tmp_path)
    return OutputContext(None, None, None, None, {}, paths, log=log if log is not None else (lambda _: None))


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
    with pytest.raises(ValueError, match="outputs.only names unknown"):
        plan_modules([Fake("a")], only=["b"])
    with pytest.raises(ValueError, match="outputs.skip names unknown"):
        plan_modules([Fake("a")], skip=["b"])


def test_only_pulls_in_what_is_required():
    lines = []
    calls = []
    modules = [Fake("a", calls=calls), Fake("b", requires=["a"], calls=calls), Fake("c", requires=["b"], calls=calls),
               Fake("d", calls=calls)]
    ctx = context(log=lines.append)
    records = run_modules(ctx, modules, only=["c"])
    assert calls == ["a", "b", "c"] and list(records) == ["a", "b", "c"]
    assert ctx.products == {"a": "product of a", "b": "product of b", "c": "product of c"}
    assert any("added because a selected module requires them: b (by c), a (by b)" in line for line in lines)


def test_skip_cascades_to_dependents():
    calls = []
    modules = [Fake("a", calls=calls), Fake("b", requires=["a"], calls=calls), Fake("c", requires=["b"], calls=calls),
               Fake("d", after=["a"], calls=calls)]
    ctx = context()
    records = run_modules(ctx, modules, skip=["a"])
    assert calls == ["d"]
    assert records["a"] == {"status": "skipped", "seconds": 0.0, "reason": "in outputs.skip"}
    assert records["b"]["status"] == records["c"]["status"] == "skipped"
    assert records["c"]["reason"] == "requires b, which was skipped" and records["d"]["status"] == "done"


def test_inactive_module_gives_its_reason():
    lines, calls = [], []
    modules = [Fake("truth", inactive="no truth maps configured", calls=calls), Fake("plots", requires=["truth"], calls=calls)]
    ctx = context(log=lines.append)
    records = run_modules(ctx, modules)
    assert calls == []
    assert records["truth"] == {"status": "inactive", "seconds": 0.0, "reason": "no truth maps configured"}
    assert records["plots"]["reason"] == "requires truth, which was inactive"
    assert "[outputs] truth: inactive (no truth maps configured)" in lines


def test_a_failing_module_is_isolated_and_recorded():
    lines, calls = [], []
    modules = [Fake("a", error=RuntimeError("broken map"), calls=calls), Fake("b", requires=["a"], calls=calls),
               Fake("c", calls=calls)]
    ctx = context(log=lines.append)
    records = run_modules(ctx, modules)
    assert calls == ["a", "c"]
    assert records["a"]["status"] == "failed" and records["a"]["error"] == "RuntimeError: broken map"
    assert records["b"] == {"status": "skipped", "seconds": 0.0, "reason": "requires a, which failed without a product"}
    assert records["c"]["status"] == "done" and ctx.summary["modules"] is records
    failure = next(line for line in lines if line.startswith("[outputs] a: RuntimeError"))
    assert "broken map" in failure and "Traceback" in failure
    assert "[outputs] a: FAILED (RuntimeError: broken map) in" in "\n".join(lines)
    assert sum("Traceback" in line for line in lines) == 1
    assert lines[-1].startswith("[outputs] failed modules: a")


def test_fail_fast_reraises():
    calls = []
    modules = [Fake("a", error=RuntimeError("broken"), calls=calls), Fake("c", calls=calls)]
    ctx = context()
    with pytest.raises(RuntimeError, match="broken"):
        run_modules(ctx, modules, fail_fast=True)
    assert calls == ["a"] and ctx.summary["modules"]["a"]["status"] == "failed"
    # KeyboardInterrupt is never caught.
    with pytest.raises(KeyboardInterrupt):
        run_modules(context(), [Fake("a", error=KeyboardInterrupt())])


def test_checkpoint_runs_once_before_the_first_imaging_module():
    events = []
    modules = [Fake("a", calls=events), Fake("rgb", checkpoint=True, calls=events), Fake("folds", checkpoint=True, calls=events)]
    # (two checkpoint modules: the checkpoint still runs once)
    run_modules(context(), modules, checkpoint=lambda: events.append("checkpoint"))
    assert events == ["a", "checkpoint", "rgb", "folds"]
    events.clear()
    run_modules(context(), modules, skip=["rgb", "folds"], checkpoint=lambda: events.append("checkpoint"))
    assert events == ["a"]


def test_draws_are_built_lazily(monkeypatch, tmp_path):
    from kappaccino.reporting import draws as draws_module
    built = []

    class FakeDraws(list):
        seed, has_field, cache_stage, image_indices = 7, False, None, [0, 1, 2]

    monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: built.append(1) or FakeDraws([0, 1]))
    ctx = context(tmp_path)
    records = run_modules(ctx, [Fake("a"), Fake("b")])
    assert built == [] and not ctx.draws_built and "reporting_draws" not in ctx.summary
    assert "reporting_draws" not in records["a"]
    records = run_modules(ctx, [Fake("a", draws=True), Fake("b")])
    assert built == [1] and ctx.summary["reporting_draws"] == 2 and ctx.draws.cache_stage == tmp_path
    # Each module records the draws it used.
    assert records["a"]["reporting_draws"] == 2 and records["a"]["image_draws"] == 3 and "reporting_draws" not in records["b"]


def test_a_failed_draws_build_is_not_repeated(monkeypatch):
    from kappaccino.reporting import draws as draws_module
    built = []

    def broken(*a):
        built.append(1)
        raise RuntimeError("no room for the draws")

    monkeypatch.setattr(draws_module, "reporting_draws", broken)
    ctx = context()
    records = run_modules(ctx, [Fake("a", draws=True), Fake("b", draws=True)])
    assert built == [1]
    assert records["a"]["error"] == records["b"]["error"] == "RuntimeError: no room for the draws"


def test_outputs_stage_builds_draws_only_in_the_step_that_needs_them(monkeypatch, synthetic_config, tmp_path):
    from kappaccino.outputs import fit
    from kappaccino.plotting import block_corner, chain_traces, gp, members, plots, potfile
    from kappaccino.reporting import draws as draws_module, outputs, results_tex

    def no_draws(*a):
        raise RuntimeError("draws built")

    monkeypatch.setattr(draws_module, "reporting_draws", no_draws)
    for module, name in ((members, "write_member_diagnostic_plots"), (gp, "write_gp_plots"), (chain_traces, "write_chain_traces"),
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

    def __init__(self, name="stepped", fail=(), product=True):
        self.name, self.fail, self.product, self.calls = name, set(fail), product, []

    requires_table = True

    def _work(self, step):
        self.calls.append(step)
        if step in self.fail:
            raise ValueError(f"{step} broke")
        return step

    def run(self, ctx):
        table = ctx.step("table", self._work, "table")
        if ctx.ok("table"):
            ctx.step("figure", self._work, "figure")
        else:
            ctx.skip_step("figure", "table failed")
        ctx.step("other", self._work, "other")
        if not ctx.ok("table"):
            raise ModuleUnavailable("no table (the table step failed)")
        return table if self.product else None


def test_a_failing_step_does_not_stop_the_module():
    lines = []
    module = Stepped(fail={"table"})
    ctx = context(log=lines.append)
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
    records = run_modules(context(), [Stepped(fail={"other"}), Fake("after", requires=["stepped"])])
    assert records["stepped"]["status"] == "failed" and records["stepped"]["error"] == "failed steps: other"
    assert records["after"]["status"] == "done"
    # A module that returned None satisfies its dependents.
    records = run_modules(context(), [Stepped(product=False), Fake("after", requires=["stepped"])])
    assert records["stepped"]["status"] == records["after"]["status"] == "done"


def test_a_failing_step_with_fail_fast_reraises():
    lines = []
    module = Stepped(fail={"table"})
    ctx = context(log=lines.append)
    with pytest.raises(ValueError, match="table broke"):
        run_modules(ctx, [module], fail_fast=True)
    assert module.calls == ["table"]
    assert sum("Traceback" in line for line in lines) == 1          # logged once, by the step
    record = ctx.summary["modules"]["stepped"]
    assert record["status"] == "failed" and record["steps"]["table"]["status"] == "failed"
    with pytest.raises(RuntimeError, match="outside a module run"):
        context().step("x", print)


def test_imaging_is_loaded_once_per_variant(monkeypatch, synthetic_config):
    from kappaccino.imaging import cutouts
    loads, lines = [], []

    def load(image_config, *args):
        loads.append(image_config.image_dir)
        if image_config.image_dir == "missing":
            raise FileNotFoundError("no such folder")
        return ("imaging", image_config.image_dir)

    monkeypatch.setattr(cutouts, "load_cluster_imaging", load)
    ctx = OutputContext(synthetic_config, None, None, None, {}, None, log=lines.append)
    view = replace(synthetic_config.image_catalog, image_dir="hst")
    assert ctx.imaging(view) is ctx.imaging(replace(view, figure_suffix="_other"))
    missing = replace(view, image_dir="missing")
    assert ctx.imaging(missing) is None and ctx.imaging(missing) is None
    assert loads == ["hst", "missing"]
    assert lines == ["[outputs] cluster mass figure and covariance RGB cutouts skipped: no such folder"]


def test_outputs_config_is_validated(synthetic_config):
    synthetic_config.validate()
    replace(synthetic_config, outputs=OutputsConfig(only=("fit",), skip=("truth", "rgb"), fail_fast=True)).validate()
    for bad, match in ((OutputsConfig(skip=("nope",)), "outputs.skip names unknown"),
                       (OutputsConfig(only=("fit", "nope")), "outputs.only names unknown"),
                       (OutputsConfig(only=()), "at least one"),
                       (OutputsConfig(skip="rgb"), "must be a list"),
                       (OutputsConfig(fail_fast="yes"), "fail_fast")):
        with pytest.raises(ValueError, match=match):
            replace(synthetic_config, outputs=bad).validate()


def test_outputs_settings_leave_the_hash_and_the_resume_alone(synthetic_config, tmp_path):
    from kappaccino import cli
    from kappaccino.pipeline import resume_payload
    base = synthetic_config
    chosen = replace(base, outputs=OutputsConfig(only=("truth",), skip=("rgb",), fail_fast=True)).validate()
    assert cli.config_hash(chosen) == cli.config_hash(base)
    assert resume_payload(chosen.to_json_dict()) == resume_payload(base.to_json_dict())
    assert "outputs" not in resume_payload(base.to_json_dict())
    # The section round-trips through config.json.
    artifacts.save_config(tmp_path, chosen)
    assert artifacts.load_config(tmp_path).outputs == chosen.outputs
    assert LensClusterSolverConfig.from_json_dict(chosen.to_json_dict()).outputs == chosen.outputs
    assert chosen.to_json_dict()["outputs"] == {"only": ["truth"], "skip": ["rgb"], "fail_fast": True}
    # A config.json without the section (written before it existed) loads with the defaults and resumes.
    payload = chosen.to_json_dict()
    del payload["outputs"]
    (tmp_path / artifacts.CONFIG_FILE).write_text(json.dumps(payload))
    saved = artifacts.load_config(tmp_path)
    assert saved.outputs == OutputsConfig()
    assert resume_payload(saved.to_json_dict()) == resume_payload(chosen.to_json_dict())


def test_outputs_section_from_run_files_and_set():
    from kappaccino import cli
    from kappaccino.run_files import load_run
    config = load_run("catalogs/hera", overrides=["outputs.skip=[truth, rgb]", "outputs.fail_fast=true"])
    assert config.outputs == OutputsConfig(skip=("truth", "rgb"), fail_fast=True)
    assert load_run("catalogs/hera", overrides=["outputs.only=[truth]"]).outputs.only == ("truth",)
    assert (cli.build_config("catalogs/hera", overrides=["outputs.skip=[rgb]"]).paths.run_name
            == cli.build_config("catalogs/hera").paths.run_name)


def test_outputs_list_command(capsys, synthetic_config, tmp_path):
    from kappaccino import cli
    assert cli.main(["outputs", "--list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[0] for line in lines[1::2]] == [m.name for m in MODULES]
    assert lines[0].split()[:4] == ["module", "requires", "after", "description"]
    assert lines[5].split()[:3] == ["mass", "images", "fit"]
    assert lines[6] == "    steps: critical_curves, aperture_masses, zeta_masses"
    # With a run: whether each module applies, from its saved configuration.
    run = tmp_path / "run"
    config = replace(synthetic_config, image_catalog=replace(synthetic_config.image_catalog, image_dir=None),
                     truth=replace(synthetic_config.truth, kappa_true_fits=None))
    artifacts.save_config(run / "final" / "artifacts", config)
    assert cli.main(["outputs", "--list", str(run)]) == 0
    rows = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1::2]}
    assert rows["fit"].rstrip().endswith("yes")
    assert rows["truth"].rstrip().endswith("no: no truth convergence map (truth.kappa_true_fits)")
    assert rows["rgb"].rstrip().endswith("no: no imaging (image_catalog.image_dir)")
    with pytest.raises(SystemExit):
        cli.main(["outputs", str(run), "--only", "nope"])
    with pytest.raises(SystemExit):
        cli.main(["outputs"])


# ----------------------------------------------------------------------------- run_summary across passes
class SummaryModule(OutputModule):
    """Writes ``summary[key]`` from the draws, or fails."""

    def __init__(self, name, key, requires=(), fail=False):
        self.name, self.summary_keys, self.requires, self.fail = name, (key,), tuple(requires), fail

    def run(self, ctx):
        draws = ctx.draws
        if self.fail:
            raise RuntimeError(f"{self.name} broke")
        ctx.summary[self.summary_keys[0]] = {"value": f"{self.name} over {len(draws)} draws"}
        return {}


def summary_pass(monkeypatch, tmp_path, modules, *, draws, only=None):
    from kappaccino.reporting import draws as draws_module, results_tex
    from kappaccino.reporting.outputs import write_run_summary

    class FakeDraws(list):
        seed, has_field, cache_stage = 7, False, None
        image_indices = list(range(2 * draws))

    monkeypatch.setattr(draws_module, "reporting_draws", lambda *a: FakeDraws(range(draws)))
    monkeypatch.setattr(results_tex, "write_run_results", lambda *a, **k: None)
    ctx = context(tmp_path)
    ctx.config = SimpleNamespace(paths=SimpleNamespace(figure_prefix="toy"))
    run_modules(ctx, modules, only=only)
    write_run_summary(ctx, partial=only is not None, modules=modules)
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


def test_fail_fast_still_writes_the_summary_and_compacts(monkeypatch, synthetic_config, tmp_path):
    from kappaccino.io import store as store_module
    from kappaccino.outputs import fit
    from kappaccino.reporting import outputs, results_tex
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


# ----------------------------------------------------------------------------- kappaccino outputs on a saved run
def test_module_selection_on_the_command_line(synthetic_config):
    from kappaccino.outputs.command import select_modules
    saved = replace(synthetic_config, outputs=OutputsConfig(only=("truth",), skip=("rgb",)))
    assert select_modules(saved).outputs == saved.outputs                       # nothing given: the saved selection
    assert select_modules(saved, all_modules=True).outputs == OutputsConfig()
    assert select_modules(saved, only=["mass"]).outputs == OutputsConfig(only=("mass",))
    assert select_modules(saved, skip=["truth"]).outputs == OutputsConfig(skip=("truth",))
    assert select_modules(saved, fail_fast=True).outputs.fail_fast


def test_remap_and_report_overrides(synthetic_config, tmp_path):
    from kappaccino.outputs.saved_run import apply_overrides, remap_paths
    config = replace(synthetic_config, truth=replace(synthetic_config.truth, kappa_true_fits="/far/away/data/kappa.fits"),
                     image_catalog=replace(synthetic_config.image_catalog, image_dir="/far/away/data/imaging"))
    lines = []
    moved = remap_paths(config, [("/far/away", str(tmp_path))], log=lines.append)
    assert moved.truth.kappa_true_fits == f"{tmp_path}/data/kappa.fits" and moved.image_catalog.image_dir == f"{tmp_path}/data/imaging"
    assert sorted(line.split()[2].rstrip(":") for line in lines) == ["image_catalog.image_dir", "truth.kappa_true_fits"]
    assert remap_paths(config, [("/far/aw", "/x")], log=lambda _: None).truth.kappa_true_fits == "/far/away/data/kappa.fits"
    # Report settings may change; anything in the run hash may not.
    changed = apply_overrides(config, ["truth.deflection_true_fits=defl.fits", "outputs.skip=[rgb]", "runtime.output_workers=2"])
    assert changed.truth.deflection_true_fits == "defl.fits" and changed.outputs.skip == ("rgb",)
    for bad in ("sampling.warmup=3", "reporting.draws=4", "truth.kappa_true_fits=other.fits"):
        with pytest.raises(ValueError, match="only report settings"):
            apply_overrides(config, [bad])
    with pytest.raises(ValueError, match="unknown configuration sections"):
        apply_overrides(config, ["nosuch.field=1"])
    with pytest.raises(ValueError):
        apply_overrides(config, ["outputs.nosuch=1"])


def test_missing_files_name_the_field_and_suggest_remap(synthetic_config):
    from kappaccino.outputs.command import check_module_paths
    config = replace(synthetic_config, truth=replace(synthetic_config.truth, kappa_true_fits="/far/away/kappa.fits"))
    with pytest.raises(FileNotFoundError, match=r"truth.kappa_true_fits = /far/away/kappa.fits \(module mass\).*--remap"):
        check_module_paths(config)
    check_module_paths(replace(config, outputs=OutputsConfig(only=("fit",))))      # fit reads no such file


def test_outputs_command_options(monkeypatch, tmp_path, capsys):
    from kappaccino.outputs import command
    with pytest.raises(SystemExit):
        command.main([str(tmp_path), "--all", "--only", "fit"])
    seen = {}

    def prepare(run, **options):
        seen.update(options)
        return ("config",)

    monkeypatch.setattr(command, "prepare_rerun", prepare)
    from kappaccino.reporting import outputs
    store.write_record(tmp_path / "final", "run_summary", {"fit": {"rms": 0.25}})
    monkeypatch.setattr(outputs, "write_outputs", lambda *a, **k: {"modules": {"fit": {"status": "done"}}})
    assert command.main([str(tmp_path), "--only", "fit", "--draws", "4", "--remap", "/a=/b", "--set", "outputs.fail_fast=true"]) == 0
    assert seen["only"] == ("fit",) and seen["draws"] == 4 and seen["remap"] == [("/a", "/b")]
    assert seen["overrides"] == ["outputs.fail_fast=true"]
    # The fit numbers printed are the saved record's, which a pass of fit alone does not touch.
    assert json.loads(capsys.readouterr().out) == {"rms": 0.25}
    # --fail-fast: one line, exit code 1.
    monkeypatch.setattr(outputs, "write_outputs", lambda *a, **k: (_ for _ in ()).throw(ValueError("step broke")))
    assert command.main([str(tmp_path), "--fail-fast"]) == 1
    assert capsys.readouterr().out.strip() == "kappaccino outputs: stopped at the first error (ValueError: step broke)"
    monkeypatch.setattr(command, "prepare_rerun", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x not found")))
    assert command.main([str(tmp_path)]) == 2


def test_draws_option_sets_both_draw_counts(monkeypatch, synthetic_config, tmp_path):
    from kappaccino.outputs import command, saved_run
    monkeypatch.setattr(saved_run, "load_run", lambda run, **k: (synthetic_config, None, None, None, None))
    monkeypatch.setattr(command, "check_module_paths", lambda config: None)
    config, *_ = command.prepare_rerun(tmp_path, draws=4)
    assert (config.reporting.draws, config.reporting.image_draws) == (4, 4)
