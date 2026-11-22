"""benchmarks/lenscluster_pr_benchmark.py compare on synthetic per-round records (no timings are run)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmarks" / "lenscluster_pr_benchmark.py"
SHAS = {"base": "a" * 40, "pr": "b" * 40}
# Per-round metric values of the base, roughly the PR #85 runner numbers.
BASE = {"log-likelihood": 6.5, "log-posterior value+grad": 30.0, "compile": 14.0, "build_state_seconds": 0.7,
        "evaluator_setup_seconds": 1.2, "lbfgs_seconds": 1.0, "total": 55.0}


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("lenscluster_pr_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(label: str, round_index: int, started: float, roots: dict[str, Path], scale: dict[str, float] | None = None) -> dict:
    """A run of both likelihoods with the BASE values, each metric multiplied by ``scale`` (default 1)."""
    scale = scale or {}
    value = lambda name: BASE[name] * scale.get(name, 1.0)
    cases = [{"likelihood": likelihood, "n_members": 340, "n_systems": 19, "n_images": 65, "n_parameters": 745,
              "n_unconstrained": 745,
              "phases": {name: value(name) for name in ("build_state_seconds", "evaluator_setup_seconds", "lbfgs_seconds")},
              "timings": [{"name": name, "ms_per_eval": value(name), "ms_per_eval_stderr": 0.01 * value(name),
                           "compile_seconds": value("compile")} for name in ("log-likelihood", "log-posterior value+grad")]}
             for likelihood in ("smoothed", "grf_gp")]
    return {"label": label, "round": round_index, "started_unix": started, "sha": SHAS[label],
            "lumina_file": str(roots[label] / "src" / "lumina" / "__init__.py"), "lumina_commit": SHAS[label],
            "repeats": 20, "trials": 3, "lbfgs_iters": 25, "cases": cases, "total_seconds": value("total")}


def write_rounds(tmp_path: Path, pr_scales: list[dict[str, float]], *, base_rounds=None) -> tuple[Path, dict[str, Path]]:
    """Runs in the workflow order base, PR, PR, base, ...; PR round i scaled by ``pr_scales[i]``."""
    roots = {"base": tmp_path / "benchmark-base", "pr": tmp_path / "benchmark-head"}
    runs = tmp_path / "runs"
    base_rounds = range(1, len(pr_scales) + 1) if base_rounds is None else base_rounds
    started = 0.0
    for round_index, scale in enumerate(pr_scales, start=1):
        for label in (("base", "pr") if round_index % 2 else ("pr", "base")):
            started += 60.0
            if label == "base" and round_index not in base_rounds:
                continue
            path = runs / f"{label}-{round_index}" / "benchmark.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(record(label, round_index, started, roots, scale if label == "pr" else None)))
    return runs, roots


def compare(bench, runs: Path, roots: dict[str, Path], rounds: int = 3) -> str:
    return bench.compare(runs, base_root=roots["base"], head_root=roots["pr"], planned_rounds=rounds, workflow_url="local")


def row(text: str, likelihood: str, metric: str) -> str:
    return next(line for line in text.splitlines() if line.startswith(f"| {likelihood} | {metric} |"))


def test_a_consistent_slowdown_is_flagged(bench, tmp_path):
    slow = [{"log-likelihood": 1.25}, {"log-likelihood": 1.22}, {"log-likelihood": 1.27}]
    text = compare(bench, *write_rounds(tmp_path, slow))
    assert ":warning:" in row(text, "smoothed", "log-likelihood")
    assert "+24.6%" in row(text, "smoothed", "log-likelihood")
    assert text.count(":warning:") == 2 + 1  # smoothed and grf_gp log-likelihood rows, and the footnote
    assert "3 rounds of base and PR" in text and "in the order base, PR, PR, base, base, PR." in text
    assert "every round shows" in text


def test_one_slow_round_like_pr_85_is_not_flagged(bench, tmp_path):
    burst = {"log-likelihood": 1.263, "compile": 1.24, "build_state_seconds": 1.44, "evaluator_setup_seconds": 1.30,
             "total": 1.055}
    calm = {name: 1.0 for name in burst}
    faster = {name: 0.97 for name in burst}
    text = compare(bench, *write_rounds(tmp_path, [burst, calm, faster]))
    assert text.count(":warning:") == 1  # the footnote only
    assert row(text, "smoothed", "log-likelihood").endswith("% |")


@pytest.mark.parametrize("ratios", [(1.44, 0.95, 1.20), (1.60, 1.02, 1.05)], ids=["sign-disagreement", "large-scatter"])
def test_a_noisy_single_shot_phase_is_not_flagged(bench, tmp_path, ratios):
    text = compare(bench, *write_rounds(tmp_path, [{"build_state_seconds": ratio} for ratio in ratios]))
    line = row(text, "smoothed", "`build_state_seconds`")
    assert "+/-" in line and ":warning:" not in line


def test_the_uncertainty_adds_the_within_run_spread_to_the_scatter_between_rounds(bench):
    same = bench.paired_delta([(10.0, None), (10.0, None)], [(11.0, None), (11.0, None)])
    assert same["delta"] == pytest.approx(10.0) and same["stderr"] == pytest.approx(0.0)
    timed = bench.paired_delta([(10.0, 0.1), (10.0, 0.1)], [(11.0, 0.11), (11.0, 0.11)])
    assert timed["stderr"] == pytest.approx(110.0 * (2 * 2 * 0.01**2) ** 0.5 / 2)
    assert bench.paired_delta([(0.0, None), (1.0, None)], [(1.0, None), (1.0, None)]) is None


def test_a_missing_base_shows_pr_timings_only(bench, tmp_path):
    text = compare(bench, *write_rounds(tmp_path, [{}], base_rounds=[]), rounds=3)
    assert "No base result to compare against." in text and "| delta |" not in text and ":warning:" not in text
    assert "| smoothed | log-likelihood | 6.500 +/- 0.065 ms |" in text


def test_a_single_paired_round_gives_no_deltas(bench, tmp_path):
    text = compare(bench, *write_rounds(tmp_path, [{"log-likelihood": 2.0}, {"log-likelihood": 2.0}], base_rounds=[1]))
    assert "Only 1 of 3 rounds have a base run" in text and "| delta |" not in text and ":warning:" not in text
    assert "2 PR runs" in text


def test_a_run_of_the_wrong_checkout_is_rejected(bench, tmp_path):
    runs, roots = write_rounds(tmp_path, [{}, {}])
    path = runs / "base-2" / "benchmark.json"
    wrong = json.loads(path.read_text())
    wrong["lumina_file"] = str(roots["pr"] / "src" / "lumina" / "__init__.py")
    path.write_text(json.dumps(wrong))
    with pytest.raises(SystemExit, match="base run of round 2 imported .* not in"):
        compare(bench, runs, roots)
    wrong["lumina_file"], wrong["lumina_commit"] = str(roots["base"] / "src" / "lumina" / "__init__.py"), SHAS["pr"]
    path.write_text(json.dumps(wrong))
    with pytest.raises(SystemExit, match="base run of round 2 imported lumina at commit b+, not a+"):
        compare(bench, runs, roots)


def test_the_run_records_the_imported_checkout_and_its_commit(bench):
    lumina_file, commit = bench.imported_lumina()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert Path(lumina_file).is_relative_to(REPO.resolve() / "src") and commit == head
