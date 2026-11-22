"""benchmarks/lenscluster_pr_benchmark.py compare on synthetic per-round records (no timings are run)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "lenscluster_pr_benchmark.py"
SHAS = {"base": "a" * 40, "pr": "b" * 40}
# Values of a run, roughly the PR #85 runner numbers (ms for the evaluations, s for the rest).
BASE = {"log-likelihood": 6.5, "log-posterior value+grad": 30.0, "compile": 14.0, "build_state_seconds": 0.7,
        "evaluator_setup_seconds": 1.2, "lbfgs_seconds": 1.0, "total": 55.0}
EVALUATIONS = ("log-likelihood", "log-posterior value+grad")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("lenscluster_pr_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(label: str, round_index: int, started: float, roots: dict[str, Path], scale: dict[str, float]) -> dict:
    """A run of both likelihoods with the BASE values times ``scale`` (default 1); the median trial is the value."""
    value = lambda name: BASE[name] * scale.get(name, 1.0)
    cases = [{"likelihood": likelihood, "n_members": 340, "n_systems": 19, "n_images": 65, "n_parameters": 745,
              "n_unconstrained": 745,
              "phases": {name: value(name) for name in ("build_state_seconds", "evaluator_setup_seconds", "lbfgs_seconds")},
              "timings": [{"name": name, "trial_ms_per_eval": [1.03 * value(name), 0.98 * value(name), value(name)],
                           "compile_seconds": value("compile")} for name in EVALUATIONS]}
             for likelihood in ("smoothed", "grf_gp")]
    return {"label": label, "round": round_index, "started_unix": started, "sha": SHAS[label],
            "lumina_file": str(roots[label] / "src" / "lumina" / "__init__.py"), "lumina_commit": SHAS[label],
            "repeats": 50, "trials": 3, "lbfgs_iters": 25, "cases": cases, "total_seconds": value("total")}


def write_rounds(tmp_path: Path, pr: list[dict[str, float]], base: list[dict[str, float]] | None = None, *,
                 base_rounds=None) -> tuple[Path, dict[str, Path]]:
    """Runs in the workflow order base, PR, PR, base, ...; round i of each side scaled by ``pr[i]`` / ``base[i]``."""
    roots = {"base": tmp_path / "benchmark-base", "pr": tmp_path / "benchmark-head"}
    runs = tmp_path / "runs"
    base = base or [{}] * len(pr)
    base_rounds = range(1, len(pr) + 1) if base_rounds is None else base_rounds
    started = 0.0
    for round_index in range(1, len(pr) + 1):
        for label in (("base", "pr") if round_index % 2 else ("pr", "base")):
            started += 60.0
            if label == "base" and round_index not in base_rounds:
                continue
            scale = (pr if label == "pr" else base)[round_index - 1]
            path = runs / f"{label}-{round_index}" / "benchmark.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record(label, round_index, started, roots, scale)))
    return runs, roots


def edit(runs: Path, run: str, change) -> None:
    path = runs / run / "benchmark.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def compare(bench, runs: Path, roots: dict[str, Path], rounds: int = 3) -> str:
    return bench.compare(runs, base_root=roots["base"], head_root=roots["pr"], planned_rounds=rounds, workflow_url="local")


def row(text: str, likelihood: str, metric: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {likelihood} | {metric} |"))


def flagged(text: str) -> int:
    return sum(line.endswith(":warning: |") for line in text.splitlines())


def test_a_consistent_slowdown_of_an_evaluation_is_flagged(bench, tmp_path):
    slow = [{"log-likelihood": 1.25}, {"log-likelihood": 1.22}, {"log-likelihood": 1.27}]
    text = compare(bench, *write_rounds(tmp_path, slow))
    assert row(text, "smoothed", "log-likelihood") == \
        "| smoothed | log-likelihood | 6.500 ms | 7.930 ms | +22.0% (rounds +22% to +27%) :warning: |"
    assert flagged(text) == 2  # smoothed and grf_gp
    assert "3 rounds of base and PR, each run a fresh process, in the order base, PR, PR, base, base, PR." in text
    assert "minimum over runs" in text and "never get a warning" in text


def test_one_slow_round_like_pr_85_is_not_flagged(bench, tmp_path):
    burst = {"log-likelihood": 1.263, "compile": 1.24, "build_state_seconds": 1.44, "evaluator_setup_seconds": 1.30,
             "total": 1.055}
    text = compare(bench, *write_rounds(tmp_path, [burst, {name: 1.0 for name in burst}, {name: 0.97 for name in burst}]))
    assert flagged(text) == 0
    assert row(text, "smoothed", "log-likelihood").endswith("| -3.0% (rounds -3% to +26%) |")


def test_a_single_fast_trial_does_not_move_the_value(bench, tmp_path):
    # Seen on the runner: one trial of a run 20% faster than the rest. The median trial ignores it.
    runs, roots = write_rounds(tmp_path, [{}, {}, {}])
    edit(runs, "base-3", lambda data: data["cases"][1]["timings"][1]["trial_ms_per_eval"].__setitem__(1, 0.8 * 30.0))
    assert row(compare(bench, runs, roots), "grf_gp", "log-posterior value+grad").endswith("| +0.0% (rounds +0% to +0%) |")


def test_a_slowdown_below_the_threshold_is_not_flagged(bench, tmp_path):
    # Every round is 7% slower, above the per-round margin, but the delta is below 10%.
    text = compare(bench, *write_rounds(tmp_path, [{"log-likelihood": 1.07}] * 3))
    assert row(text, "smoothed", "log-likelihood").endswith("| +7.0% (rounds +7% to +7%) |") and flagged(text) == 0


def test_a_round_less_than_the_margin_slower_stops_the_warning(bench, tmp_path):
    # The minima differ by 15%, but in round 2 (slow base) the PR is only 3% slower: slower, yet within the margin.
    base = [{}, {"log-likelihood": 1.20}, {}]
    pr = [{"log-likelihood": 1.15}, {"log-likelihood": 1.24}, {"log-likelihood": 1.15}]
    text = compare(bench, *write_rounds(tmp_path, pr, base))
    assert row(text, "smoothed", "log-likelihood").endswith("| +15.0% (rounds +3% to +15%) |") and flagged(text) == 0


@pytest.mark.parametrize("metric,likelihood,label", [("compile", "smoothed", "log-likelihood compile"),
                                                     ("build_state_seconds", "smoothed", "`build_state_seconds`"),
                                                     ("total", "", "total benchmark")])
def test_the_informational_rows_are_never_flagged(bench, tmp_path, metric, likelihood, label):
    text = compare(bench, *write_rounds(tmp_path, [{metric: 1.5}] * 3))
    assert row(text, likelihood, label).endswith("| +50.0% (rounds +50% to +50%) |") and flagged(text) == 0


def test_a_missing_base_shows_pr_timings_only(bench, tmp_path):
    text = compare(bench, *write_rounds(tmp_path, [{}], base_rounds=[]))
    assert "No base result to compare against." in text and "| delta |" not in text and ":warning:" not in text
    assert "| smoothed | log-likelihood | 6.500 ms |" in text and "One PR run." in text


def test_a_single_paired_round_gives_no_deltas(bench, tmp_path):
    text = compare(bench, *write_rounds(tmp_path, [{"log-likelihood": 2.0}, {"log-likelihood": 2.0}], base_rounds=[1]))
    assert "Only 1 of 3 rounds have a base run" in text and "| delta |" not in text and ":warning:" not in text
    assert "2 PR runs" in text


def test_a_nan_or_missing_value_marks_the_row(bench, tmp_path):
    runs, roots = write_rounds(tmp_path, [{}, {}, {}])
    # In first place a NaN would sort into a finite median of the trials, so the check on the trials must catch it.
    edit(runs, "pr-2", lambda data: data["cases"][0]["timings"][0]["trial_ms_per_eval"].__setitem__(0, math.nan))
    edit(runs, "base-3", lambda data: data["cases"][0]["phases"].pop("lbfgs_seconds"))
    text = compare(bench, runs, roots)
    assert row(text, "smoothed", "log-likelihood") == "| smoothed | log-likelihood | 6.500 ms | n/a | n/a |"
    assert row(text, "smoothed", "`lbfgs_seconds`") == "| smoothed | `lbfgs_seconds` | n/a | 1.00 s | n/a |"
    for name in ("base-1", "base-2", "base-3"):
        (runs / name / "benchmark.json").unlink()
    assert row(compare(bench, runs, roots), "smoothed", "log-likelihood") == "| smoothed | log-likelihood | n/a |"


def test_pairs_of_different_lengths_are_an_error(bench):
    with pytest.raises(ValueError, match="2 base values but 3 PR values"):
        bench.paired_delta([1.0, 1.0], [1.0, 1.0, 1.0], can_warn=True)


def test_a_run_of_the_wrong_checkout_is_rejected(bench, tmp_path):
    runs, roots = write_rounds(tmp_path, [{}, {}])
    edit(runs, "base-2", lambda data: data.update(lumina_file=str(roots["pr"] / "src" / "lumina" / "__init__.py")))
    with pytest.raises(SystemExit, match="base run of round 2 imported .* not in"):
        compare(bench, runs, roots)
    edit(runs, "base-2", lambda data: data.update(lumina_file=str(roots["base"] / "src" / "lumina" / "__init__.py"),
                                                  lumina_commit=SHAS["pr"]))
    with pytest.raises(SystemExit, match="base run of round 2 imported lumina at commit b+, not a+"):
        compare(bench, runs, roots)


def test_runs_of_one_side_at_different_commits_are_rejected(bench, tmp_path):
    runs, roots = write_rounds(tmp_path, [{}, {}])
    edit(runs, "pr-2", lambda data: data.update(sha="c" * 40, lumina_commit="c" * 40))
    with pytest.raises(SystemExit, match="The PR runs were made at different commits"):
        compare(bench, runs, roots)


def test_an_unknown_label_is_rejected(bench, tmp_path):
    runs, roots = write_rounds(tmp_path, [{}, {}])
    edit(runs, "pr-2", lambda data: data.update(label="head"))
    with pytest.raises(SystemExit, match="has label 'head'"):
        compare(bench, runs, roots)


def test_the_run_records_the_imported_checkout_and_its_commit(bench):
    lumina_file, commit = bench.imported_lumina()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert Path(lumina_file).is_relative_to(REPO.resolve() / "src") and commit == head
    bench.check_imported(label="pr", round_index=1, sha=head, lumina_file=lumina_file, lumina_commit=commit, root=REPO)


@pytest.mark.parametrize("checkout,sha,message", [("elsewhere", None, "not in"), ("repo", "0" * 40, "not 0+")])
def test_run_refuses_to_time_another_checkout(bench, tmp_path, checkout, sha, message):
    args = argparse.Namespace(checkout=REPO if checkout == "repo" else tmp_path, label="base", round=2, sha=sha,
                              output_dir=tmp_path / "out")
    with pytest.raises(SystemExit, match=f"base run of round 2 imported .*{message}"):
        bench._run_command(args)
    assert not args.output_dir.exists()
