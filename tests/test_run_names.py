"""Run names assigned once (``<mon><day><letter>_<description>``), a new run folder for every run command, and
``lumina resume RUN_DIR``, the only way back into an existing one."""
from __future__ import annotations

import io
import json
import multiprocessing
import os
import re
import shlex
import shutil
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from lumina import run_lock, run_names
from lumina.config import OutputsConfig, RuntimeConfig, SamplingConfig, SearchConfig
from lumina.io import artifacts
from lumina.run_names import RunRefused, letter_index, letters, new_run, reserve

ROOT = Path(__file__).resolve().parents[1]
NOV21, NOV22 = date(2026, 11, 21), date(2026, 11, 22)


def _cluster_config(config, root):
    """The synthetic config writing to ``<root>/results/toy``, as the command builds it (the name is a stand-in)."""
    return replace(config, paths=replace(config.paths, output_dir=str(Path(root) / "results" / "toy"), run_name="pending"))


def _complete_init(run, config, *, final=None, posterior=False):
    """Make ``run`` hold a complete init stage of ``config`` as the pipeline checks it (its configuration, its
    signature and the files it requires, empty), and with ``final`` the configuration of a started final stage
    (complete with ``posterior``)."""
    init = Path(run) / "init" / "artifacts"
    artifacts.save_config(init, config)
    artifacts.write_json(init / artifacts.INIT_SIGNATURE_FILE, artifacts.init_signature(config))
    for name in (artifacts.STATE_FILE, artifacts.CLASSIFICATION_FILE, artifacts.SEARCH_FILE, artifacts.MODES_FILE,
                 artifacts.START_FILE):
        (init / name).write_text("{}")
    if final is not None:
        stage = Path(run) / "final" / "artifacts"
        artifacts.save_config(stage, final)
        if posterior:
            (stage / artifacts.POSTERIOR_FILE).write_text("")
            (stage / artifacts.AUDIT_FILE).write_text("{}")


def _folders(cluster_dir):
    return sorted(p.name for p in Path(cluster_dir).iterdir() if not p.name.startswith("."))


def test_letters_and_dates():
    assert [letters(i) for i in (0, 1, 25, 26, 27, 51, 52, 701, 702)] == ["a", "b", "z", "aa", "ab", "az", "ba", "zz", "aaa"]
    assert all(letter_index(letters(i)) == i for i in range(2000))
    assert run_names.date_prefix(NOV21) == "nov21" and run_names.date_prefix(date(2027, 1, 5)) == "jan5"


def test_letters_count_every_run_of_the_date(tmp_path):
    cluster = tmp_path / "hera"
    names = [reserve(cluster, f"w{index % 3}_desc", day=NOV21) for index in range(27)]
    assert [name.split("_")[0] for name in names] == [f"nov21{letters(i)}" for i in range(27)]
    assert names[0] == "nov21a_w0_desc" and names[1] == "nov21b_w1_desc" and names[26] == "nov21aa_w2_desc"
    assert all((cluster / name).is_dir() for name in names)
    # Another day starts again at a; folders not named like an assigned name take no letter of the date.
    (cluster / "w500s250t7_allblocks-3f9a1c").mkdir()
    (cluster / "notes").mkdir()
    assert reserve(cluster, "x", day=NOV22) == "nov22a_x"
    assert run_names.next_name(cluster, "x", day=NOV22) == "nov22b_x"
    # A deleted run's letter is not given again; nov2 is not nov21.
    shutil.rmtree(cluster / "nov22a_x")
    assert reserve(cluster, "x", day=NOV22) == "nov22b_x"
    assert reserve(cluster, "x", day=date(2026, 11, 2)) == "nov2a_x"
    # A folder made by hand with a date name takes its letter; hand labels that only start like one take none.
    (cluster / "nov23c_by_hand").mkdir()
    assert reserve(cluster, "x", day=date(2026, 11, 23)) == "nov23d_x"
    for label in ("nov24final", "nov24best_fit", "nov24abc_x", "nov24a"):
        (cluster / label).mkdir()
    assert run_names.next_name(cluster, "x", day=date(2026, 11, 24)) == "nov24a_x"
    assert reserve(cluster, "x", day=date(2026, 11, 24)) == "nov24a_x"
    # The same date of a later year goes on after the letters already given.
    assert reserve(cluster, "x", day=date(2027, 11, 21)) == "nov21ab_x"


def test_two_reservations_at_once_get_different_letters(tmp_path):
    cluster = tmp_path / "hera"
    barrier = threading.Barrier(8)

    def racer(index):
        barrier.wait()
        return reserve(cluster, f"d{index % 2}", day=NOV21)

    with ThreadPoolExecutor(8) as pool:
        names = list(pool.map(racer, range(8)))
    assert sorted(name.split("_")[0] for name in names) == [f"nov21{letters(i)}" for i in range(8)]
    with ProcessPoolExecutor(4) as pool:                # processes too: the reservation is a directory creation
        names = list(pool.map(_reserve_nov21, [str(cluster)] * 8, chunksize=1))
    assert sorted(name.split("_")[0] for name in names) == [f"nov21{letters(i)}" for i in range(8, 16)]


def _reserve_nov21(cluster):
    return reserve(cluster, "p", day=NOV21)


def test_every_new_run_is_a_new_folder(tmp_path):
    cluster = tmp_path / "results" / "toy"
    # A dry run names the folder and creates nothing.
    assert new_run(cluster, "d", dry_run=True, day=NOV21) == (cluster / "nov21a_d", None) and not cluster.exists()
    assert new_run(cluster, "d", run_label="mine", dry_run=True) == (cluster / "mine", None) and not cluster.exists()
    first, lock = new_run(cluster, "d", day=NOV21)
    assert first == cluster / "nov21a_d" and (first / run_lock.RUN_LOCK).is_file()
    assert not (cluster / run_lock.CLUSTER_LOCK).exists()                 # held only while the folder is created
    second, other = new_run(cluster, "d", day=NOV21)                     # the same call again: another run
    assert second == cluster / "nov21b_d"
    lock.release(), other.release()
    assert not list(cluster.rglob("*.lock"))
    # --run-label names a new folder; an existing one is refused with the command that continues it.
    labelled, lock = new_run(cluster, "d", run_label="mine")
    lock.release()
    assert labelled == cluster / "mine" and labelled.is_dir()
    for dry_run in (False, True):
        with pytest.raises(RunRefused, match=rf"{re.escape(str(labelled))} already exists: --run-label names a new run\.  "
                                             rf"Continue that run with: lumina resume {re.escape(str(labelled))}"):
            new_run(cluster, "d", run_label="mine", dry_run=dry_run)
    with pytest.raises(RunRefused, match="already exists"):
        new_run(cluster, "d", run_label=first.name)
    assert _folders(cluster) == ["mine", "nov21a_d", "nov21b_d"] and not list(cluster.rglob("*.lock"))


def test_a_names_file_and_a_results_folder_that_cannot_be_written(tmp_path):
    cluster = tmp_path / "results" / "toy"
    cluster.mkdir(parents=True)
    (cluster / ".names").write_text("")
    for dry_run in (False, True):
        with pytest.raises(RunRefused, match=r"\.names is not a folder"):
            new_run(cluster, "d", day=NOV21, dry_run=dry_run)
    (cluster / ".names").unlink()
    if os.geteuid() != 0:
        os.chmod(cluster, 0o555)
        try:
            with pytest.raises(RunRefused, match="cannot write in the results folder"):
                new_run(cluster, "d", day=NOV21)
        finally:
            os.chmod(cluster, 0o755)
        assert _folders(cluster) == []


def test_resume_payload_leaves_out_what_resume_sets_and_what_changes_no_fit(synthetic_config):
    """``lumina resume`` sets ``runtime.resume`` and the run folder, so neither is compared; WBIC and the outputs
    section change no fit; everything else is compared, the runtime fields included."""
    from lumina.pipeline import resume_payload

    def payload(config):
        return resume_payload(config.to_json_dict())
    base = payload(synthetic_config)
    assert payload(replace(synthetic_config, runtime=replace(synthetic_config.runtime, resume=False))) == base
    assert payload(replace(synthetic_config, paths=replace(synthetic_config.paths, output_dir="x", run_name="y"))) == base
    assert payload(replace(synthetic_config, sampling=replace(synthetic_config.sampling, wbic=True))) == base
    assert payload(replace(synthetic_config, outputs=replace(synthetic_config.outputs, draws=7))) == base
    for field, value in (("seed", 7), ("chains", 2), ("skip_plots", True)):
        assert payload(replace(synthetic_config, runtime=replace(synthetic_config.runtime, **{field: value}))) != base
    assert payload(replace(synthetic_config, paths=replace(synthetic_config.paths, figure_prefix="other"))) != base


def _quiet_command(monkeypatch, fake_run):
    from lumina import cpu_runtime, pipeline
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(run_names, "today", lambda: NOV21)
    monkeypatch.setattr(cpu_runtime, "configure_cpu_runtime", lambda *args: None)
    monkeypatch.setattr(pipeline, "run", fake_run)


def _pasted(line):
    """The ``lumina ...`` command at the end of a printed line, as arguments of ``cli.main``."""
    words = shlex.split(line.split("with: ", 1)[1])
    assert words[0] == "lumina"
    return words[1:]


def test_the_same_command_twice_makes_two_runs(tmp_path, monkeypatch, capsys):
    """``lumina catalogs/hera`` end to end with the fit replaced: every command is a new folder, ``--stage init`` is
    followed by the printed ``lumina resume RUN --stage final``, which runs the final stage there."""
    from lumina import cli
    ran = []

    def fake_run(config, stage="both", saved_form=None):
        run = Path(config.paths.output_dir) / config.paths.run_name
        ran.append((config.paths.run_name, stage, config.sampling.samples))
        assert (run / run_lock.RUN_LOCK).is_file()                       # the run is locked while it runs
        if stage == "init":
            _complete_init(run, config)
        else:
            if not artifacts.is_init_complete(run / "init" / "artifacts"):
                _complete_init(run, config)
            artifacts.save_config(run / "final" / "artifacts", config)

    _quiet_command(monkeypatch, fake_run)
    results = tmp_path / "results"
    hera = results / "hera"
    base = ["catalogs/hera", "--results", str(results)]
    assert cli.main([*base, "--stage", "init", "--samples", "20"]) == 0
    out = capsys.readouterr().out
    first = hera / "nov21a_w500s20t7_allblocks"
    assert f"[run] created {first} (if it stops, continue it with: lumina resume {first})" in out
    nxt = next(line for line in out.splitlines() if line.startswith("[run] init stage done"))
    assert nxt == f"[run] init stage done in {first}; run its final stage with: lumina resume {first} --stage final"
    # The same command again: a second run, the next letter.
    assert cli.main([*base, "--stage", "init", "--samples", "20"]) == 0
    assert ran == [("nov21a_w500s20t7_allblocks", "init", 20), ("nov21b_w500s20t7_allblocks", "init", 20)]
    assert all((hera / name / "inputs" / "command.json").is_file() for name, _, _ in ran)
    capsys.readouterr()
    # The printed command, pasted: the final stage of the first run, with its saved settings.
    assert cli.main(_pasted(nxt)) == 0
    assert ran[-1] == (first.name, "final", 20)
    out = capsys.readouterr().out
    assert f"[run] finished {first}; lumina resume {first} reloads it and reruns the outputs stage" in out
    # --stage final on the run command: a new folder has no init stage, so the resume form is named.
    assert cli.main([*base, "--stage", "final"]) == 2
    assert "lumina resume RUN_DIR --stage final" in capsys.readouterr().err
    # --run-label names a new run; an existing folder is refused with the command that continues it.
    assert cli.main([*base, "--run-label", "mine", "--stage", "init"]) == 0 and ran[-1][0] == "mine"
    capsys.readouterr()
    for extra in ([], ["--dry-run"]):
        assert cli.main([*base, "--run-label", "mine", *extra]) == 2
        assert f"{hera / 'mine'} already exists: --run-label names a new run.  Continue that run with: lumina resume {hera / 'mine'}" in capsys.readouterr().err
    assert cli.main([*base, "--dry-run"]) == 0
    assert f"would create {hera / 'nov21c_w500s250t7_allblocks'}" in capsys.readouterr().out
    assert len(ran) == 4 and _folders(hera) == ["mine", "nov21a_w500s20t7_allblocks", "nov21b_w500s20t7_allblocks"]
    assert not list(results.rglob("*.lock"))


def test_the_lock_is_released_and_the_resume_printed_after_a_normal_end_an_error_and_ctrl_c(tmp_path, monkeypatch, capsys):
    from lumina import cli
    from lumina.resume import main as resume_main
    outcome = {}

    def fake_run(config, stage="both", saved_form=None):
        run = Path(config.paths.output_dir) / config.paths.run_name
        outcome["run"] = run
        assert (run / run_lock.RUN_LOCK).is_file()
        if "raise" in outcome:
            _complete_init(run, config)
            raise outcome["raise"]
        _complete_init(run, config)

    _quiet_command(monkeypatch, fake_run)
    base = ["catalogs/hera", "--results", str(tmp_path / "results"), "--stage", "init"]
    assert cli.main(base) == 0 and not (outcome["run"] / run_lock.RUN_LOCK).exists()
    for error in (RuntimeError("inside the fit"), KeyboardInterrupt()):
        outcome["raise"] = error
        capsys.readouterr()
        with pytest.raises(type(error)):
            cli.main(base)
        assert not (outcome["run"] / run_lock.RUN_LOCK).exists()
        assert not (outcome["run"].parent / run_lock.CLUSTER_LOCK).exists()
        line = capsys.readouterr().out.splitlines()[-1]
        assert line == f"[run] {outcome['run']} stopped before the end; continue it with: lumina resume {outcome['run']}"
        assert _pasted(line) == ["resume", str(outcome["run"])]
    # The same three for lumina resume.
    del outcome["raise"]
    run = outcome["run"]
    assert resume_main([str(run), "--stage", "init"]) == 0 and not (run / run_lock.RUN_LOCK).exists()
    for error in (RuntimeError("inside the fit"), KeyboardInterrupt()):
        outcome["raise"] = error
        capsys.readouterr()
        with pytest.raises(type(error)):
            resume_main([str(run), "--stage", "init"])
        assert not (run / run_lock.RUN_LOCK).exists()
        assert capsys.readouterr().out.splitlines()[-1].endswith(f"continue it with: lumina resume {run}")


def test_a_locked_run_is_refused(tmp_path, monkeypatch, capsys):
    """A run another live process holds: lumina resume and a computing lumina outputs refuse it (exit 2); a dry run
    says who holds it; --render proceeds; a stale lock is taken over."""
    from lumina import cli
    from lumina.outputs.command import main as outputs_main
    from lumina.resume import main as resume_main

    def fake_run(config, stage="both", saved_form=None):
        _complete_init(Path(config.paths.output_dir) / config.paths.run_name, config)

    _quiet_command(monkeypatch, fake_run)
    results = tmp_path / "results"
    assert cli.main(["catalogs/hera", "--results", str(results), "--stage", "init"]) == 0
    run = next(p for p in (results / "hera").iterdir() if not p.name.startswith("."))
    holder = os.getppid()                   # a live process that is not this one
    (run / run_lock.RUN_LOCK).write_text(json.dumps({"host": run_lock.host(), "pid": holder,
                                                     "started": "2026-11-21T10:00:00", "command": "lumina catalogs/hera"}))
    capsys.readouterr()
    assert resume_main([str(run), "--stage", "init"]) == 2
    assert f"the run {run} is in use: locked by host {run_lock.host()} pid {holder} since 2026-11-21T10:00:00" in capsys.readouterr().err
    assert outputs_main([str(run)]) == 2 and "is in use" in capsys.readouterr().out
    assert outputs_main([str(run), "--render"]) == 2 and "is in use" not in capsys.readouterr().out   # no final stage
    assert resume_main([str(run), "--dry-run"]) == 0 and f"currently locked by host {run_lock.host()}" in capsys.readouterr().out
    assert (run / run_lock.RUN_LOCK).is_file()                           # never removed by those who refused
    # Another host's lock cannot be checked: respected, with the file to delete.
    (run / run_lock.RUN_LOCK).write_text(json.dumps({"host": "elsewhere.example", "pid": 1, "started": "2026-11-20T09:00:00"}))
    assert resume_main([str(run), "--stage", "init"]) == 2
    assert f"another host's lock cannot be checked from {run_lock.host()}: if that process is gone, delete {run / run_lock.RUN_LOCK}" in capsys.readouterr().err
    # A stale lock of this host (its process is gone) is taken over, with one line.
    dead = _dead_pid()
    (run / run_lock.RUN_LOCK).write_text(json.dumps({"host": run_lock.host(), "pid": dead, "started": "2026-11-20T09:00:00"}))
    assert resume_main([str(run), "--stage", "init"]) == 0
    assert f"[lock] took over the stale lock {run / run_lock.RUN_LOCK} of host {run_lock.host()} pid {dead}" in capsys.readouterr().out
    assert not (run / run_lock.RUN_LOCK).exists()


def _dead_pid() -> int:
    import subprocess
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _race_child(task):
    """One ``lumina catalogs/hera`` of a race: waits at ``barrier`` with the others, then runs the command with the
    fit replaced."""
    results, barrier = task
    os.chdir(ROOT)
    from lumina import cli, cpu_runtime, pipeline
    run_names.today = lambda: NOV21
    cpu_runtime.configure_cpu_runtime = lambda *args: None
    pipeline.run = lambda config, stage="both": _complete_init(Path(config.paths.output_dir) / config.paths.run_name, config)
    barrier.wait()
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(["catalogs/hera", "--results", results, "--stage", "init"])
    return code, err.getvalue()


def test_identical_commands_launched_at_once_make_one_run_each(tmp_path):
    """Through the real command: N identical commands launched together make N runs, one letter each."""
    count = 4
    results = tmp_path / "results"
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        barrier = manager.Barrier(count)
        with ProcessPoolExecutor(count, mp_context=context) as pool:
            outcomes = list(pool.map(_race_child, [(str(results), barrier)] * count))
    assert [code for code, _ in outcomes] == [0] * count, outcomes
    assert _folders(results / "hera") == [f"nov21{letters(i)}_w500s250t7_allblocks" for i in range(count)]
    assert not list(results.rglob("*.lock"))


def test_an_unreadable_saved_configuration_is_refused_by_resume(synthetic_config, tmp_path, capsys):
    from lumina.resume import load_saved_config, main as resume_main
    config = _cluster_config(synthetic_config, tmp_path).validate()
    cluster = Path(config.paths.output_dir)
    good = config.to_json_dict()
    bad = {"old": ({"model": {}, "no_such_section": {"x": 1}}, "UnknownField: no_such_section: no such field"),
           "broken": ("{not json", "is not valid JSON (JSONDecodeError:"),
           "listed": ([1, 2], "a JSON list, not an object"),
           "runtime": ({**good, "runtime": [1, 2]}, "InvalidSection: runtime is a JSON list where the section RuntimeConfig belongs"),
           "empty": ({}, "it lacks the section(s) model, paths")}
    for name, text in (("os", "os:path"), ("function", "lumina.config:replace"), ("old_name", "kappaccino.config:DPIEHaloConfig")):
        payload = json.loads(json.dumps(good))
        payload["model"]["large_halos"][0]["_class"] = text
        bad[name] = (payload, f"UnknownField: model.large_halos[0]: no such class {text!r}")
    for name, (payload, message) in bad.items():
        folder = cluster / name / "init" / "artifacts"
        folder.mkdir(parents=True)
        (folder / artifacts.CONFIG_FILE).write_text(payload if isinstance(payload, str) else json.dumps(payload))
        capsys.readouterr()
        assert resume_main([str(cluster / name), "--dry-run"]) == 2, name
        err = capsys.readouterr().err
        assert f"{folder / artifacts.CONFIG_FILE} " in err and message in err, (name, err)
    (cluster / "stray").mkdir()
    assert resume_main([str(cluster / "stray")]) == 2 and "is not a run folder" in capsys.readouterr().err
    # A bug in the loader is not an unreadable run: it propagates.
    import lumina.config as config_module
    _complete_init(cluster / "fine", config)

    def broken_loader(payload):
        raise ZeroDivisionError("a bug")
    original = config_module.LensClusterSolverConfig.from_json_dict
    try:
        config_module.LensClusterSolverConfig.from_json_dict = staticmethod(broken_loader)
        with pytest.raises(ZeroDivisionError):
            load_saved_config(cluster / "fine" / "init" / "artifacts" / artifacts.CONFIG_FILE)
    finally:
        config_module.LensClusterSolverConfig.from_json_dict = original


def _tiny(config, root):
    return replace(_cluster_config(config, root), runtime=RuntimeConfig(seed=3, chains=2, nuts_chain_method="sequential",
                                                                         skip_plots=True, quiet=True),
                   search=SearchConfig(n_starts=2, member_starts=1, lbfgs_max_iter=30, polish_max_iter=20),
                   sampling=SamplingConfig(warmup=10, samples=10, max_tree_depth=4), outputs=OutputsConfig(draws=2))


@pytest.mark.slow                       # init stages and short NUTS runs on the toy cluster
def test_lumina_resume_continues_a_run_with_its_own_settings(tmp_path, monkeypatch, capsys):
    from conftest import synthetic_cluster_config
    from lumina import cpu_runtime, pipeline
    from lumina.resume import main as resume_main
    monkeypatch.setattr(cpu_runtime, "configure_cpu_runtime", lambda *args: None)
    root = tmp_path / "here"
    config = _tiny(synthetic_cluster_config(root), root).validate()
    run, lock = new_run(config.paths.output_dir, "w10s10t4", day=NOV21)
    config = replace(config, paths=replace(config.paths, run_name=run.name))
    pipeline.run(config, stage="init")
    lock.release()
    copies = {}
    for name in ("interrupted", "in_init", "other_model", "remapped"):
        copies[name] = run.parent / f"nov21b_{name}"
        shutil.copytree(run, copies[name])
    artifacts.save_config(copies["interrupted"] / "final" / "artifacts", config)      # killed during sampling
    (copies["in_init"] / "init" / "artifacts" / artifacts.START_FILE).unlink()         # killed during the search
    signature = copies["other_model"] / "init" / "artifacts" / artifacts.INIT_SIGNATURE_FILE
    signature.write_text(json.dumps({**json.loads(signature.read_text()), "runtime": {"seed": 99}}))

    # A fit setting is refused before anything runs; a dry run writes nothing.
    assert resume_main([str(run), "--set", "sampling.warmup=3"]) == 2
    assert "may change only the outputs section" in capsys.readouterr().err
    assert resume_main([str(run), "--dry-run"]) == 0
    assert "final stage not started" in capsys.readouterr().out and not (run / "final").exists()
    assert resume_main([str(tmp_path / "nowhere")]) == 2
    # A saved init stage the saved configuration cannot continue is refused by the pipeline, and the lock released.
    with pytest.raises(ValueError, match="saved init stage was made with a different model"):
        resume_main([str(copies["other_model"])])
    assert not (copies["other_model"] / run_lock.RUN_LOCK).exists()
    # Interrupted during the init stage: the init stage runs again from the start.
    assert "init stage not complete" in (resume_main([str(copies["in_init"]), "--dry-run"]) == 0 and capsys.readouterr().out)
    assert resume_main([str(copies["in_init"]), "--stage", "init"]) == 0
    assert artifacts.is_init_complete(copies["in_init"] / "init" / "artifacts")
    assert "complete init stage found" not in (copies["in_init"] / "init" / "logs" / "init.log").read_text()

    # Init complete, final never started: --stage final runs it, with the outputs settings given here saved.
    assert resume_main([str(run), "--stage", "final", "--draws", "3"]) == 0
    assert artifacts.is_final_complete(run / "final" / "artifacts")
    assert artifacts.load_config(run / "final" / "artifacts").outputs.draws == 3
    # Interrupted during sampling: sampled again from the saved start with the saved final settings.
    assert "interrupted in the final stage" in (resume_main([str(copies["interrupted"]), "--dry-run"]) == 0 and capsys.readouterr().out)
    assert resume_main([str(copies["interrupted"])]) == 0 and artifacts.is_final_complete(copies["interrupted"] / "final" / "artifacts")
    # A second final stage from the same init: a copy without final/, resumed with --stage final.  It samples with
    # the init stage's saved settings, and the first run is left as it was.
    second = run.parent / f"{run.name}_second"
    shutil.copytree(run, second)
    shutil.rmtree(second / "final")
    before = (run / "final" / "artifacts" / artifacts.POSTERIOR_FILE).read_bytes()
    assert resume_main([str(second), "--stage", "final"]) == 0
    assert artifacts.is_final_complete(second / "final" / "artifacts")
    assert artifacts.load_config(second / "final" / "artifacts").sampling == config.sampling
    assert (run / "final" / "artifacts" / artifacts.POSTERIOR_FILE).read_bytes() == before

    # Moved to another machine: the catalogs are elsewhere, so it needs --remap.  A final stage that samples there
    # saves the paths as first saved, so the folder stays the same run without the remap.
    moved = tmp_path / "there"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    remapped = moved / "results" / "toy" / copies["remapped"].name
    assert resume_main([str(remapped)]) == 2 and "--remap" in capsys.readouterr().err
    assert resume_main([str(remapped), "--remap", f"{root}={moved}"]) == 0
    assert artifacts.is_final_complete(remapped / "final" / "artifacts")
    for saved in (remapped / "init" / "artifacts" / artifacts.CONFIG_FILE, remapped / "final" / "artifacts" / artifacts.CONFIG_FILE,
                  remapped / "init" / "artifacts" / artifacts.INIT_SIGNATURE_FILE):
        text = saved.read_text()
        assert str(root) in text and str(moved) not in text, saved
    assert resume_main([str(remapped), "--remap", f"{root}={moved}"]) == 0             # a second resume, same remap
    # A remap that moves nothing is a warning, and the exit code is kept.
    assert resume_main([str(remapped), "--stage", "init", "--remap", "/nomatch=/x"]) == 2
    assert "WARNING remap: no saved path starts with /nomatch" in capsys.readouterr().out
    # Back where the catalogs are: resumed without any remap.
    shutil.copytree(moved, root)
    back = root / "results" / "toy" / remapped.name

    # Finished: reloaded, neither searched nor sampled again.
    def forbidden(*args, **kwargs):
        raise AssertionError("a finished run was sampled again")
    monkeypatch.setattr(pipeline, "run_nuts", forbidden)
    monkeypatch.setattr(pipeline, "split_map", forbidden)
    assert resume_main([str(back)]) == 0
    assert resume_main([str(back), "--stage", "init", "--remap", "/nomatch=/x"]) == 0
    assert "WARNING remap: no saved path starts with /nomatch" in capsys.readouterr().out
    assert resume_main([str(run)]) == 0
    # Renamed in the other tree: needs the remap, then resumes as before.
    elsewhere = moved / "results" / "toy" / "renamed"
    (moved / "results" / "toy" / run.name).rename(elsewhere)
    assert resume_main([str(elsewhere), "--remap", f"{root}={moved}"]) == 0
    assert resume_main([str(elsewhere), "--stage", "init", "--remap", f"{root}={moved}"]) == 0
    assert not list(tmp_path.rglob("*.lock"))


def test_the_imaging_cluster_never_comes_from_the_run_name(synthetic_config):
    """With dated run names the run folder says nothing about the cluster: the catalog paths, else the run's figure
    prefix (its name in the run file), name it."""
    from lumina.imaging.cutouts import cluster_from_config
    assert cluster_from_config(synthetic_config.model, "m0416_canucs2025") == "m0416"
    assert cluster_from_config(synthetic_config.model, "tngc014_small") == "tngc014"
    assert cluster_from_config(synthetic_config.model, None) == "cluster"
