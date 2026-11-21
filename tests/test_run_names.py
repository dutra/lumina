"""Run names assigned once (``<mon><day><letter>_<description>``), the saved run the ``lumina`` command resumes, and
``lumina resume RUN_DIR``."""
from __future__ import annotations

import io
import json
import multiprocessing
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from lumina import run_lock, run_names
from lumina.config import OutputsConfig, RuntimeConfig, SamplingConfig, SearchConfig
from lumina.io import artifacts
from lumina.run_names import RunRefused, choose_run, fit_settings, letter_index, letters, reserve

ROOT = Path(__file__).resolve().parents[1]
NOV21, NOV22 = date(2026, 11, 21), date(2026, 11, 22)


def _cluster_config(config, root):
    """The synthetic config writing to ``<root>/results/toy``, as the command builds it (the name is a stand-in)."""
    return replace(config, paths=replace(config.paths, output_dir=str(Path(root) / "results" / "toy"), run_name="pending"))


def _saved_run(cluster_dir, name, config, *, created=None, final=False):
    """A run folder holding the saved configuration of its init stage (and of its final stage)."""
    run = Path(cluster_dir) / name
    artifacts.save_config(run / "init" / "artifacts", config)
    if final:
        artifacts.save_config(run / "final" / "artifacts", config)
    if created is not None:
        os.utime(run / "init" / "artifacts" / artifacts.CONFIG_FILE, (created, created))
    return run


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
    # Another day starts again at a; a hashed name from before and a stray folder take no letter of the date.
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


def test_two_commands_at_once_get_different_letters(tmp_path):
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


def test_a_matching_run_is_resumed_and_a_changed_one_explained(synthetic_config, tmp_path, monkeypatch):
    monkeypatch.setattr(run_names, "today", lambda: NOV21)
    config = _cluster_config(synthetic_config, tmp_path).validate()
    cluster = Path(config.paths.output_dir)
    lines = []
    log = lines.append
    first = choose_run(config, "w10_desc", log=log)
    assert (first.name, first.resume) == ("nov21a_w10_desc", False) and first.path.is_dir() and not lines
    # The new run holds its settings and its lock at once: the identical command finds it, locked, and refuses.
    assert run_names.read_saved_run(first.path).settings == fit_settings(config)
    assert (first.path / run_lock.RUN_LOCK).is_file() and not (cluster / run_lock.CLUSTER_LOCK).exists()
    with pytest.raises(run_lock.RunLocked, match=rf"the run {first.path} is in use: locked by host .* pid {os.getpid()}"):
        choose_run(config, "w10_desc", log=log)
    dry = choose_run(config, "w10_desc", dry_run=True, log=log)
    assert dry.path == first.path and dry.resume and dry.locked_by.pid == os.getpid() and dry.lock is None
    assert _folders(cluster) == ["nov21a_w10_desc"]
    first.release()
    assert not (first.path / run_lock.RUN_LOCK).exists()
    assert choose_run(config, "w10_desc", dry_run=True, log=log).locked_by is None
    _saved_run(cluster, first.name, config, created=1000)
    again = choose_run(config, "w10_desc", log=log)
    assert again.resume and again.path == first.path and _folders(cluster) == ["nov21a_w10_desc"] and not lines
    again.release()
    # Another day, another outputs section, and the runtime fields that change only what is printed or drawn: still
    # the same run.
    monkeypatch.setattr(run_names, "today", lambda: NOV22)
    outputs = replace(config, outputs=replace(config.outputs, draws=7, workers=1),
                      runtime=replace(config.runtime, skip_plots=True, quiet=True, numpyro_print_summary=True))
    choose_run(outputs, "w10_desc", log=log).release()
    assert choose_run(outputs, "w10_desc", dry_run=True, log=log).path == first.path
    # A changed fit setting: the next name of the date, with the settings that differ from the closest run.
    changed = replace(config, sampling=replace(config.sampling, warmup=config.sampling.warmup + 5)).validate()
    created = choose_run(changed, "w15_desc", log=log)
    created.release()
    assert created.name == "nov22a_w15_desc" and not created.resume
    text = "\n".join(lines)
    assert "none of the 1 saved run(s)" in text and f"closest: {first.path}, 1 setting(s) differ" in text
    assert f"sampling.warmup: {config.sampling.warmup} -> {config.sampling.warmup + 5}" in text
    assert f"lumina resume {first.path} continues that run with its own settings" in text
    assert run_names.PATH_PREFIX_HINT not in text
    # Several runs match: the newest is resumed and the others named.  An old hashed name matches like any other.
    _saved_run(cluster, "w10_desc-3f9a1c", config, created=3000)
    _saved_run(cluster, "nov21c_w10_desc", config, created=2000)
    lines.clear()
    newest = choose_run(config, "w10_desc", log=log)
    newest.release()
    assert newest.resume and newest.name == "w10_desc-3f9a1c"
    assert "3 saved runs match these fit settings" in lines[0] and "nov21a_w10_desc" in lines[0] and "nov21c_w10_desc" in lines[0]
    # runtime.resume false never matches, and says nothing about the other runs.
    lines.clear()
    fresh = replace(config, runtime=replace(config.runtime, resume=False))
    made = choose_run(fresh, "w10_desc", log=log)
    made.release()
    assert made.name == "nov22b_w10_desc" and not lines


def test_fit_settings_leave_out_what_changes_no_fit(synthetic_config):
    """The runtime fields that only print or draw leave the comparison; seed, chains and the chain method stay; a
    prior is part of a run's identity, WBIC is not."""
    base = fit_settings(synthetic_config)
    for field, value in (("skip_plots", True), ("quiet", True), ("numpyro_print_summary", True), ("resume", False)):
        assert fit_settings(replace(synthetic_config, runtime=replace(synthetic_config.runtime, **{field: value}))) == base
    for field, value in (("seed", 7), ("chains", 2), ("nuts_chain_method", "vectorized")):
        assert fit_settings(replace(synthetic_config, runtime=replace(synthetic_config.runtime, **{field: value}))) != base
    assert set(base["runtime"]) == {"seed", "chains", "nuts_chain_method"}
    assert fit_settings(replace(synthetic_config, sampling=replace(synthetic_config.sampling, wbic=True))) == base
    assert artifacts.init_signature(synthetic_config)["runtime"] == {"seed": synthetic_config.runtime.seed}


def test_an_unreadable_saved_configuration_is_reported_and_skipped(synthetic_config, tmp_path):
    config = _cluster_config(synthetic_config, tmp_path).validate()
    cluster = Path(config.paths.output_dir)
    old = cluster / "w10_old-abcdef" / "init" / "artifacts"
    old.mkdir(parents=True)
    (old / artifacts.CONFIG_FILE).write_text(json.dumps({"model": {}, "no_such_section": {"x": 1}}))
    broken = cluster / "nov20a_broken" / "init" / "artifacts"
    broken.mkdir(parents=True)
    (broken / artifacts.CONFIG_FILE).write_text("{not json")
    listed = cluster / "nov20b_list" / "init" / "artifacts"
    listed.mkdir(parents=True)
    (listed / artifacts.CONFIG_FILE).write_text("[1, 2]")
    (cluster / "stray").mkdir()
    # Valid JSON that is not a configuration of this code: a section of the wrong type, no sections, or a _class that
    # is not a configuration class (another module, a function, the package's old name).
    good = config.to_json_dict()
    bad = {"nov20c_runtime": {**good, "runtime": [1, 2]}, "nov20d_sampling": {**good, "sampling": "x"},
           "nov20e_paths": {**good, "paths": 3}, "nov20f_empty": {}}
    for name, text in (("nov20g_os", "os:path"), ("nov20h_function", "lumina.config:replace"),
                       ("nov20i_old_name", "kappaccino.config:DPIEHaloConfig")):
        payload = json.loads(json.dumps(good))
        payload["model"]["large_halos"][0]["_class"] = text
        bad[name] = payload
    for name, payload in bad.items():
        folder = cluster / name / "init" / "artifacts"
        folder.mkdir(parents=True)
        (folder / artifacts.CONFIG_FILE).write_text(json.dumps(payload))
    lines = []
    choice = choose_run(config, "w10_desc", day=NOV21, log=lines.append)
    choice.release()
    assert choice.name == "nov21a_w10_desc" and not choice.resume
    text = "\n".join(lines)
    for name in bad:
        assert f"{name}: skipped, {cluster / name / 'init' / 'artifacts' / artifacts.CONFIG_FILE} is not a configuration this code can load" in text, name
    assert "InvalidSection: runtime is a JSON list where the section RuntimeConfig belongs" in text
    assert "it lacks the section(s) model, paths" in text
    assert "UnknownField: model.large_halos[0]: no such class 'kappaccino.config:DPIEHaloConfig'" in text
    from lumina.resume import main as resume_main
    for name in bad:
        assert resume_main([str(cluster / name), "--dry-run"]) == 2, name
    assert f"w10_old-abcdef: skipped, {old / artifacts.CONFIG_FILE} is not a configuration this code can load (UnknownField: no_such_section: no such field" in text
    assert f"nov20a_broken: skipped, {broken / artifacts.CONFIG_FILE} is not valid JSON (JSONDecodeError:" in text
    assert "nov20b_list: skipped" in text and "a JSON list, not an object" in text
    assert "closest" not in text and "stray" not in text
    lines.clear()
    again = choose_run(config, "w10_desc", day=NOV21, log=lines.append)
    again.release()
    assert again.path == choice.path and not lines
    # A bug in the loader is not an unreadable run: it propagates.
    import lumina.config as config_module

    def broken_loader(payload):
        raise ZeroDivisionError("a bug")
    original = config_module.LensClusterSolverConfig.from_json_dict
    try:
        config_module.LensClusterSolverConfig.from_json_dict = staticmethod(broken_loader)
        with pytest.raises(ZeroDivisionError):
            run_names.read_saved_run(choice.path)
    finally:
        config_module.LensClusterSolverConfig.from_json_dict = original


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs file permissions that bind")
def test_an_unreadable_sibling_is_reported_and_skipped(synthetic_config, tmp_path):
    config = _cluster_config(synthetic_config, tmp_path).validate()
    cluster = Path(config.paths.output_dir)
    secret = _saved_run(cluster, "nov20a_secret", config)
    hidden = _saved_run(cluster, "nov20b_hidden", config)
    os.chmod(secret, 0)
    os.chmod(hidden / "init" / "artifacts" / artifacts.CONFIG_FILE, 0)
    try:
        lines = []
        choice = choose_run(config, "w10_desc", day=NOV21, log=lines.append)
        choice.release()
        text = "\n".join(lines)
        assert choice.name == "nov21a_w10_desc"
        assert f"{secret}: skipped, the folder cannot be read (PermissionError: Permission denied)" in text
        assert f"{hidden}: skipped, {hidden / 'init' / 'artifacts' / artifacts.CONFIG_FILE} cannot be read (PermissionError" in text
    finally:
        os.chmod(secret, 0o755)
        os.chmod(hidden / "init" / "artifacts" / artifacts.CONFIG_FILE, 0o644)


def test_the_explanation_shows_absent_settings_and_whole_paths(synthetic_config, tmp_path):
    config = _cluster_config(synthetic_config, tmp_path).validate()
    saved = fit_settings(config)
    current = json.loads(json.dumps(saved))
    del current["sampling"]["warmup"]
    long_path = "/a/very/long/folder/name/that/goes/on/and/on/for/more/than/sixty/characters/images.yaml"
    current["model"]["image_constraints"]["catalog_path"] = long_path
    text = "\n".join(run_names.difference_lines(saved, current))
    assert f"sampling.warmup: {config.sampling.warmup} -> (absent)" in text
    assert f'-> "{long_path}"' in text
    # Paths that differ only by a prefix (a command run from another folder) get the hint.
    lines = []
    moved = json.loads(json.dumps(saved))
    for population in moved["model"]["member_populations"]:
        population["catalog_path"] = population["catalog_path"].split("/", 2)[-1]
    moved["model"]["image_constraints"]["catalog_path"] = saved["model"]["image_constraints"]["catalog_path"].split("/", 2)[-1]
    run_names.explain_no_match([run_names.SavedRun(tmp_path / "nov21a_x", 1.0, saved)], moved, log=lines.append)
    assert f"[run]   these settings {run_names.PATH_PREFIX_HINT}" in lines
    lines.clear()
    moved["runtime"]["seed"] += 1
    run_names.explain_no_match([run_names.SavedRun(tmp_path / "nov21a_x", 1.0, saved)], moved, log=lines.append)
    assert f"[run]   2 of these settings {run_names.PATH_PREFIX_HINT}" in lines


def test_a_names_file_and_a_results_folder_that_cannot_be_written(synthetic_config, tmp_path):
    config = _cluster_config(synthetic_config, tmp_path).validate()
    cluster = Path(config.paths.output_dir)
    cluster.mkdir(parents=True)
    (cluster / ".names").write_text("")
    with pytest.raises(RunRefused, match=r"\.names is not a folder"):
        choose_run(config, "d", day=NOV21)
    with pytest.raises(RunRefused, match=r"\.names is not a folder"):
        choose_run(config, "d", day=NOV21, dry_run=True)
    (cluster / ".names").unlink()
    if os.geteuid() != 0:
        os.chmod(cluster, 0o555)
        try:
            with pytest.raises(RunRefused, match="cannot write in the results folder"):
                choose_run(config, "d", day=NOV21)
        finally:
            os.chmod(cluster, 0o755)
        assert _folders(cluster) == []


def test_run_label_and_dry_run(synthetic_config, tmp_path):
    config = _cluster_config(synthetic_config, tmp_path).validate()
    cluster = Path(config.paths.output_dir)
    # Absent: a dry run creates nothing at all, a run creates exactly that folder.
    assert choose_run(config, "d", run_label="mine", dry_run=True).resume is False and not cluster.exists()
    assert choose_run(config, "d", dry_run=True, day=NOV21).name == "nov21a_d" and not cluster.exists()
    with pytest.raises(RunRefused, match="--stage final needs a run"):
        choose_run(config, "d", stage="final", day=NOV21)
    assert not cluster.exists()
    made = choose_run(config, "d", run_label="mine")
    assert made.path == cluster / "mine" and made.path.is_dir() and not made.resume
    with pytest.raises(run_lock.RunLocked, match="is in use"):          # the folder is in use
        choose_run(config, "d", run_label="mine")
    made.release()
    assert "its init stage is not complete: the init stage runs again" in choose_run(config, "d", run_label="mine",
                                                                                     dry_run=True).reason
    # Present: continued when the pipeline would continue its saved stages, refused with the differences otherwise.
    _complete_init(made.path, config)
    choose_run(config, "d", run_label="mine").release()
    changed = replace(config, runtime=replace(config.runtime, seed=config.runtime.seed + 1))
    with pytest.raises(RunRefused, match=r"other fit settings for --stage both(.|\n)*init stage:(.|\n)*runtime\.seed: (.|\n)*lumina resume"):
        choose_run(changed, "d", run_label="mine")
    assert _folders(cluster) == ["mine"] and not (made.path / run_lock.RUN_LOCK).exists()


def _quiet_command(monkeypatch, fake_run):
    from lumina import cpu_runtime, pipeline
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(run_names, "today", lambda: NOV21)
    monkeypatch.setattr(cpu_runtime, "configure_cpu_runtime", lambda *args: None)
    monkeypatch.setattr(pipeline, "run", fake_run)


def test_the_command_resumes_its_run_and_refuses_a_changed_label(tmp_path, monkeypatch, capsys):
    """``lumina catalogs/hera`` end to end with the fit replaced by saving a complete init stage."""
    from lumina import cli
    ran = []

    def fake_run(config, stage="both"):
        run = Path(config.paths.output_dir) / config.paths.run_name
        ran.append((config.paths.run_name, stage))
        assert (run / run_lock.RUN_LOCK).is_file()                       # the run is locked while it runs
        if stage == "final":
            artifacts.save_config(run / "final" / "artifacts", config)
        else:
            _complete_init(run, config)

    _quiet_command(monkeypatch, fake_run)
    results = tmp_path / "results"
    base = ["catalogs/hera", "--results", str(results)]
    assert cli.main([*base, "--stage", "init"]) == 0
    assert cli.main([*base, "--stage", "final", "--draws", "8", "--skip-plots"]) == 0    # the init-only run
    assert cli.main([*base, "--warmup", "600"]) == 0
    assert ran == [("nov21a_w500s250t7_allblocks", "init"), ("nov21a_w500s250t7_allblocks", "final"),
                   ("nov21b_w600s250t7_allblocks", "both")]
    out = capsys.readouterr().out
    assert "sampling.warmup: 500 -> 600" in out and f"resuming {results / 'hera' / 'nov21a_w500s250t7_allblocks'}" in out
    assert (results / "hera" / "nov21a_w500s250t7_allblocks" / "inputs" / "command.json").is_file()
    assert cli.main([*base, "--run-label", "nov21a_w500s250t7_allblocks", "--seed", "8"]) == 2
    assert "other fit settings" in capsys.readouterr().err
    assert cli.main([*base, "--dry-run", "--warmup", "700"]) == 0
    assert f"would create {results / 'hera' / 'nov21c_w700s250t7_allblocks'}" in capsys.readouterr().out
    assert cli.main([*base, "--dry-run", "--draws", "4"]) == 0
    assert f"would resume {results / 'hera' / 'nov21a_w500s250t7_allblocks'}" in capsys.readouterr().out
    assert sorted(p.name for p in (results / "hera").iterdir()) == [".names", "nov21a_w500s250t7_allblocks",
                                                                   "nov21b_w600s250t7_allblocks"]
    assert not list(results.rglob("*.lock"))


def test_stage_final_runs_from_a_complete_init_with_other_sampling_settings(tmp_path, monkeypatch, capsys):
    """``--stage final --set sampling.samples=...`` on a folder holding a complete init: the final stage runs from
    it, with or without ``--run-label``, as long as the pipeline would continue the saved stages."""
    from lumina import cli
    ran = []

    def fake_run(config, stage="both"):
        ran.append((config.paths.run_name, stage, config.sampling.samples))
        run = Path(config.paths.output_dir) / config.paths.run_name
        if stage == "init":
            _complete_init(run, config)
        else:
            artifacts.save_config(run / "final" / "artifacts", config)

    _quiet_command(monkeypatch, fake_run)
    results = tmp_path / "results"
    base = ["catalogs/hera", "--results", str(results)]
    hera = results / "hera"
    assert cli.main([*base, "--run-label", "L", "--stage", "init", "--samples", "20"]) == 0
    assert cli.main([*base, "--run-label", "L", "--stage", "final", "--set", "sampling.samples=30"]) == 0
    assert ran[-1] == ("L", "final", 30)
    # A complete final stage with other fit settings: the pipeline would refuse it, so the command does.
    (hera / "L" / "final" / "artifacts" / artifacts.POSTERIOR_FILE).write_text("")
    (hera / "L" / "final" / "artifacts" / artifacts.AUDIT_FILE).write_text("{}")
    capsys.readouterr()
    assert cli.main([*base, "--run-label", "L", "--stage", "final", "--set", "sampling.samples=40"]) == 2
    err = capsys.readouterr().err
    assert "other fit settings for --stage final" in err and "final stage:" in err and "sampling.samples: 30 -> 40" in err
    # The same --stage init still works (the init stage alone is compared).
    assert cli.main([*base, "--run-label", "L", "--stage", "init", "--samples", "40"]) == 0
    # Without --run-label: the init-only run matches --stage final whatever its sampling settings...
    assert cli.main([*base, "--stage", "init", "--samples", "20"]) == 0
    first = ran[-1][0]
    assert first.startswith("nov21a_")
    assert cli.main([*base, "--stage", "final", "--samples", "30"]) == 0
    assert ran[-1] == (first, "final", 30)
    # ...but once its final stage started, the final settings must match too.
    capsys.readouterr()
    assert cli.main([*base, "--stage", "final", "--samples", "50"]) == 2
    captured = capsys.readouterr()
    assert (f"--stage final: {hera / first} has the init stage these settings would make, but its final stage already "
            "started with other fit settings (saved -> current):\n    sampling.samples: 30 -> 50") in captured.err
    assert f"this command with --run-label {first} samples it again there" in captured.err       # not finished
    assert "sampling.samples: 30 -> 50" in captured.out
    assert cli.main([*base, "--stage", "final", "--samples", "30"]) == 0 and ran[-1] == (first, "final", 30)
    # Once that final stage is complete, the way that works is the one the message gives: a copy without it.
    (hera / first / "final" / "artifacts" / artifacts.POSTERIOR_FILE).write_text("")
    (hera / first / "final" / "artifacts" / artifacts.AUDIT_FILE).write_text("{}")
    assert cli.main([*base, "--stage", "final", "--samples", "50"]) == 2
    err = capsys.readouterr().err
    assert "its final stage is complete and is kept" in err and f"cp -R {hera / first} {hera}/NEW" in err
    shutil.copytree(hera / first, hera / "NEW")
    shutil.rmtree(hera / "NEW" / "final")
    assert cli.main([*base, "--stage", "final", "--samples", "50", "--run-label", "NEW"]) == 0
    assert ran[-1] == ("NEW", "final", 50)
    shutil.rmtree(hera / "NEW")
    # A changed init setting never matches --stage final.
    assert cli.main([*base, "--stage", "final", "--samples", "30", "--seed", "99"]) == 2
    assert sorted(p.name for p in hera.iterdir() if not p.name.startswith(".")) == ["L", first]


def test_the_lock_is_released_after_a_normal_end_an_error_and_ctrl_c(tmp_path, monkeypatch):
    from lumina import cli
    outcome = {}

    def fake_run(config, stage="both"):
        run = Path(config.paths.output_dir) / config.paths.run_name
        outcome["run"] = run
        assert (run / run_lock.RUN_LOCK).is_file()
        if "raise" in outcome:
            raise outcome["raise"]
        _complete_init(run, config)

    _quiet_command(monkeypatch, fake_run)
    base = ["catalogs/hera", "--results", str(tmp_path / "results"), "--stage", "init"]
    assert cli.main(base) == 0 and not (outcome["run"] / run_lock.RUN_LOCK).exists()
    for error in (RuntimeError("inside the fit"), KeyboardInterrupt()):
        outcome["raise"] = error
        with pytest.raises(type(error)):
            cli.main(base)
        assert not (outcome["run"] / run_lock.RUN_LOCK).exists()
        assert not (outcome["run"].parent / run_lock.CLUSTER_LOCK).exists()
    # The same three for lumina resume.
    from lumina import pipeline
    from lumina.resume import main as resume_main
    monkeypatch.setattr(pipeline, "run", lambda config, stage="both", saved_form=None: fake_run(config, stage))
    del outcome["raise"]
    assert resume_main([str(outcome["run"]), "--stage", "init"]) == 0 and not (outcome["run"] / run_lock.RUN_LOCK).exists()
    for error in (RuntimeError("inside the fit"), KeyboardInterrupt()):
        outcome["raise"] = error
        with pytest.raises(type(error)):
            resume_main([str(outcome["run"]), "--stage", "init"])
        assert not (outcome["run"] / run_lock.RUN_LOCK).exists()


def test_a_locked_run_is_refused_by_every_command(tmp_path, monkeypatch, capsys):
    """A run another live process holds: the identical command, --run-label, lumina resume and a computing
    lumina outputs refuse it (exit 2) and create nothing; a dry run says who holds it; --render proceeds."""
    from lumina import cli
    from lumina.outputs.command import main as outputs_main
    from lumina.resume import main as resume_main

    def fake_run(config, stage="both"):
        _complete_init(Path(config.paths.output_dir) / config.paths.run_name, config)

    _quiet_command(monkeypatch, fake_run)
    results = tmp_path / "results"
    base = ["catalogs/hera", "--results", str(results), "--stage", "init"]
    assert cli.main(base) == 0
    run = next(p for p in (results / "hera").iterdir() if not p.name.startswith("."))
    holder = os.getppid()                   # a live process that is not this one
    (run / run_lock.RUN_LOCK).write_text(json.dumps({"host": run_lock.host(), "pid": holder,
                                                     "started": "2026-11-21T10:00:00", "command": "lumina catalogs/hera"}))
    before = sorted(p.name for p in (results / "hera").iterdir())
    capsys.readouterr()
    for command in (cli.main, ):
        assert command(base) == 2
    err = capsys.readouterr().err
    assert f"the run {run} is in use: locked by host {run_lock.host()} pid {holder} since 2026-11-21T10:00:00" in err
    assert cli.main([*base, "--run-label", run.name]) == 2 and "is in use" in capsys.readouterr().err
    assert resume_main([str(run), "--stage", "init"]) == 2 and "is in use" in capsys.readouterr().err
    assert outputs_main([str(run)]) == 2 and "is in use" in capsys.readouterr().out
    assert outputs_main([str(run), "--render"]) == 2 and "is in use" not in capsys.readouterr().out   # no final stage
    assert cli.main([*base, "--dry-run"]) == 0
    assert f"would resume {run}, currently locked by host {run_lock.host()} pid {holder}" in capsys.readouterr().out
    assert resume_main([str(run), "--dry-run"]) == 0 and f"currently locked by host {run_lock.host()}" in capsys.readouterr().out
    assert sorted(p.name for p in (results / "hera").iterdir()) == before
    assert (run / run_lock.RUN_LOCK).is_file()                           # never removed by those who refused
    # Another host's lock cannot be checked: respected, with the file to delete.
    (run / run_lock.RUN_LOCK).write_text(json.dumps({"host": "elsewhere.example", "pid": 1, "started": "2026-11-20T09:00:00"}))
    assert cli.main(base) == 2
    assert f"another host's lock cannot be checked from {run_lock.host()}: if that process is gone, delete {run / run_lock.RUN_LOCK}" in capsys.readouterr().err
    # A stale lock of this host (its process is gone) is taken over, with one line.
    dead = _dead_pid()
    (run / run_lock.RUN_LOCK).write_text(json.dumps({"host": run_lock.host(), "pid": dead, "started": "2026-11-20T09:00:00"}))
    assert cli.main(base) == 0
    out = capsys.readouterr().out
    assert f"[lock] took over the stale lock {run / run_lock.RUN_LOCK} of host {run_lock.host()} pid {dead}" in out
    assert f"resuming {run}" in out and not (run / run_lock.RUN_LOCK).exists()


def _dead_pid() -> int:
    import subprocess
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _race_child(task):
    """One ``lumina catalogs/hera`` of a race: waits at ``barrier`` with the others, then runs the command with the
    fit replaced; with ``hold`` the fit keeps its run until every other command has ended."""
    results, extra, barrier, done, count, hold = task
    os.chdir(ROOT)
    from lumina import cli, cpu_runtime, pipeline
    run_names.today = lambda: NOV21
    cpu_runtime.configure_cpu_runtime = lambda *args: None

    def fake_run(config, stage="both"):
        _complete_init(Path(config.paths.output_dir) / config.paths.run_name, config)
        deadline = time.monotonic() + 120
        while hold and len(list(Path(done).iterdir())) < count - 1 and time.monotonic() < deadline:
            time.sleep(0.05)

    pipeline.run = fake_run
    barrier.wait()
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(["catalogs/hera", "--results", results, "--stage", "init", *extra])
    (Path(done) / str(os.getpid())).write_text(str(code))
    return code, err.getvalue()


@pytest.mark.parametrize("identical", [True, False], ids=["identical", "different"])
def test_commands_launched_at_once(tmp_path, identical):
    """Through the real command: N identical commands launched together make one run (the others find it locked and
    refuse); N commands with different settings get N letters."""
    count = 4
    results, done = tmp_path / "results", tmp_path / "done"
    done.mkdir()
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        barrier = manager.Barrier(count)
        tasks = [(str(results), [] if identical else ["--seed", str(index)], barrier, str(done), count, identical)
                 for index in range(count)]
        with ProcessPoolExecutor(count, mp_context=context) as pool:
            outcomes = list(pool.map(_race_child, tasks))
    runs = sorted(p.name for p in (results / "hera").iterdir() if not p.name.startswith("."))
    codes = sorted(code for code, _ in outcomes)
    if identical:
        assert runs == ["nov21a_w500s250t7_allblocks"] and codes == [0] + [2] * (count - 1)
        assert all("is in use: locked by" in err for code, err in outcomes if code == 2)
    else:
        assert runs == [f"nov21{letters(i)}_w500s250t7_allblocks" for i in range(count)] and codes == [0] * count
    assert not list(results.rglob("*.lock"))


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
    choice = choose_run(config, "w10s10t4", day=NOV21)
    config = replace(config, paths=replace(config.paths, run_name=choice.name))
    pipeline.run(config, stage="init")
    choice.release()
    run = choice.path
    # The plain command with --stage final later finds this init-only run.
    found = choose_run(config, "w10s10t4", stage="final", day=NOV22)
    found.release()
    assert found.path == run
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
    # A saved init stage the saved configuration cannot continue: a usage error, not a traceback.
    capsys.readouterr()
    assert resume_main([str(copies["other_model"])]) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"lumina resume: {copies['other_model']}: the saved init stage of this run folder was made with another model")
    assert "runtime.seed: 99 -> 3" in err and not (copies["other_model"] / run_lock.RUN_LOCK).exists()
    # Interrupted during the init stage: the init stage runs again from the start.
    assert "init stage not complete" in (resume_main([str(copies["in_init"]), "--dry-run"]) == 0 and capsys.readouterr().out)
    assert resume_main([str(copies["in_init"]), "--stage", "init"]) == 0
    assert artifacts.is_init_complete(copies["in_init"] / "init" / "artifacts")
    assert "complete init stage found" not in (copies["in_init"] / "init" / "logs" / "init.log").read_text()

    # Init complete, final never started: the final stage runs, with the outputs settings given here saved.
    assert resume_main([str(run), "--draws", "3"]) == 0
    assert artifacts.is_final_complete(run / "final" / "artifacts")
    assert artifacts.load_config(run / "final" / "artifacts").outputs.draws == 3
    # Interrupted during sampling: sampled again from the saved start with the saved final settings.
    assert "interrupted in the final stage" in (resume_main([str(copies["interrupted"]), "--dry-run"]) == 0 and capsys.readouterr().out)
    assert resume_main([str(copies["interrupted"])]) == 0 and artifacts.is_final_complete(copies["interrupted"] / "final" / "artifacts")

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
    # Back where the catalogs are: resumed without any remap, and the plain command matches it.
    shutil.copytree(moved, root)
    back = root / "results" / "toy" / remapped.name
    lines = []
    found = choose_run(config, "w10s10t4", dry_run=True, day=NOV22, log=lines.append)
    assert found.resume and back.name in lines[0] and "saved runs match these fit settings" in lines[0]

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
