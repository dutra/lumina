"""Likelihood and gradient timings on the FF-SIMS Hera model, for the PR benchmark comment.

``run`` times the lumina that Python imports, in one fresh process, and writes ``benchmark.json`` and a markdown
``output``; the record keeps the imported ``lumina.__file__`` and the commit of its checkout.  ``compare`` reads the
``benchmark.json`` of every run under ``--runs-dir`` and renders the base-versus-PR table that
.github/workflows/publish-pr-benchmarks.yml posts on the pull request.

The workflow runs base and PR in alternating rounds (base, PR, then PR, base, ...), each run a fresh process with
``--catalog-root`` on the PR checkout so the inputs are the same.  ``compare`` first checks that every base run imported
lumina from the base checkout and every PR run from the PR checkout, at the commit the run was given.  Load on a shared
runner mostly slows a run down, so each side's value of a row is its minimum over rounds (for the evaluation rows, of
the median trial of each run) and the delta is the ratio of the two minima; the per-round PR/base ratios give its
range.  Only the evaluation rows can get a warning: a delta above 10% with the PR more than 5% slower in every round.
With fewer than two paired rounds the table shows PR timings only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import lumina
from lumina.config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                                LikelihoodConfig, MemberPopulationConfig, ReferenceFrameConfig, RunPathsConfig, ShearHaloConfig)
from lumina.likelihood.evaluator import Evaluator
from lumina.inference.search import Objective
from lumina.core.state import build_state

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- lenscluster benchmark -->"
# Name -> (likelihood_mode, gp_kernel): the library default and the grf_gp model of the cluster run.yaml files.
LIKELIHOODS = {"smoothed": ("smoothed", "image_plane"), "grf_gp": ("gp_image_plane", "knee")}
SIDES = {"base": "base", "pr": "PR"}
MIN_PAIRED_ROUNDS = 2
WARN_PERCENT = 10.0
# Every round must show the PR this much slower; quiet runs of the same code differ by a few percent.
ROUND_MARGIN = 1.05


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _stderr(values: list[float]) -> float:
    return float(statistics.stdev(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _workflow_url() -> str:
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    return f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else "local"


def hera_config(catalog_root: Path, likelihood: str, output_dir: str) -> LensClusterSolverConfig:
    """The Hera model of catalogs/hera/priors.yaml with library defaults elsewhere."""
    mode, kernel = LIKELIHOODS[likelihood]
    return LensClusterSolverConfig(
        model=LensModelConfig(
            reference=ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0),
            cosmology=CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
            large_halos=(DPIEHaloConfig.cluster_scale("1", 19.54080009, 2.37820005, 0.507, v_disp=800.0),
                         DPIEHaloConfig.cluster_scale("2", 0.001, 0.003, 0.507, v_disp=700.0),
                         ShearHaloConfig.free("S1", 0.507)),
            member_populations=(MemberPopulationConfig.bergamini2019("potfile_1", str(catalog_root / "catalogs/hera/members.yaml"), 0.507,
                                                                     mag0=19.82),),
            image_constraints=ImageConstraintsConfig(catalog_path=str(catalog_root / "catalogs/hera/images.yaml"))),
        likelihood=LikelihoodConfig(likelihood_mode=mode, gp_kernel=kernel),
        paths=RunPathsConfig(output_dir=output_dir, run_name="benchmark")).validate()


def _bench_jitted(name: str, fn: Callable[[], Any], repeats: int, trials: int) -> dict[str, Any]:
    """Compile time of the first call, then ``trials`` blocks of ``repeats`` calls."""
    start = time.perf_counter()
    jax.block_until_ready(fn())
    compile_seconds = time.perf_counter() - start
    trial_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(repeats):
            out = fn()
        jax.block_until_ready(out)
        trial_ms.append(1.0e3 * (time.perf_counter() - start) / repeats)
    return {"name": name, "repeats": repeats, "trials": trials, "compile_seconds": float(compile_seconds),
            "trial_ms_per_eval": trial_ms, "ms_per_eval": _mean(trial_ms), "ms_per_eval_stderr": _stderr(trial_ms)}


def run_case(catalog_root: Path, likelihood: str, *, repeats: int, trials: int, lbfgs_iters: int) -> dict[str, Any]:
    phases: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as output_dir:
        start = time.perf_counter()
        config = hera_config(catalog_root, likelihood, output_dir)
        state = build_state(config)
        phases["build_state_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        evaluator = Evaluator(state, config.likelihood.likelihood_mode)
        objective = Objective(state, evaluator, state.initial_theta, log_likelihood=evaluator.log_likelihood)
        phases["evaluator_setup_seconds"] = time.perf_counter() - start

        theta = jnp.asarray(state.initial_theta)
        u0 = objective.base
        log_likelihood = jax.jit(evaluator.log_likelihood)
        rows = [_bench_jitted("log-likelihood", lambda: log_likelihood(theta), repeats, trials),
                _bench_jitted("log-posterior value+grad", lambda: objective.value_and_grad(u0), repeats, trials)]

        # A fixed-length L-BFGS run from the initial model: the per-iteration cost the init search pays.
        def fun(x):
            value, grad = objective.value_and_grad(jnp.asarray(x))
            return float(value), np.asarray(grad, dtype=float)

        start = time.perf_counter()
        result = minimize(fun, np.asarray(u0, dtype=float), jac=True, method="L-BFGS-B",
                          options={"maxiter": lbfgs_iters, "gtol": 0.0, "ftol": 0.0})
        phases["lbfgs_seconds"] = time.perf_counter() - start
    return {"likelihood": likelihood, "n_members": len(state.members.member_ids), "n_systems": len(state.family_data),
            "n_images": int(sum(family.n_images for family in state.family_data)),
            "n_parameters": int(np.size(state.initial_theta)), "n_unconstrained": int(np.size(u0)), "phases": phases,
            "timings": rows, "lbfgs_iterations": int(result.nit), "lbfgs_evaluations": int(result.nfev)}


def imported_lumina() -> tuple[str, str | None]:
    """The imported ``lumina/__init__.py`` and the commit of the git checkout it sits in (None outside a checkout)."""
    path = Path(lumina.__file__).resolve()
    git = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "HEAD"], capture_output=True, text=True)
    return str(path), git.stdout.strip() if git.returncode == 0 else None


def run_benchmark(*, label: str, round_index: int, sha: str | None, catalog_root: Path, likelihoods: list[str], repeats: int,
                  trials: int, lbfgs_iters: int) -> dict[str, Any]:
    lumina_file, lumina_commit = imported_lumina()
    started_unix = time.time()
    start = time.perf_counter()
    cases = [run_case(catalog_root, name, repeats=repeats, trials=trials, lbfgs_iters=lbfgs_iters) for name in likelihoods]
    return {"label": label, "round": round_index, "started_unix": started_unix, "sha": sha or lumina_commit,
            "lumina_file": lumina_file, "lumina_commit": lumina_commit, "repeats": repeats, "trials": trials,
            "lbfgs_iters": lbfgs_iters, "jax_version": jax.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "cases": cases, "total_seconds": time.perf_counter() - start}


def metrics(record: dict[str, Any]) -> dict[tuple[str, str], tuple[float, str, bool]]:
    """(likelihood, metric) -> (value, unit, can warn) for every row of the table.

    An evaluation row takes the median of the run's trials (each the mean time of ``repeats`` evaluations), which a
    single odd trial cannot move: on the runner a trial can also come out 20% faster than the rest of its run.  Compile
    times, phases and the total are single measurements.  Only the evaluation rows can get a warning.
    """
    out: dict[tuple[str, str], tuple[float, str, bool]] = {}
    for case in record["cases"]:
        for row in case["timings"]:
            trials = row["trial_ms_per_eval"]
            median = statistics.median(trials) if trials and all(math.isfinite(value) for value in trials) else math.nan
            out[(case["likelihood"], row["name"])] = (median, "ms", True)
            out[(case["likelihood"], f"{row['name']} compile")] = (row["compile_seconds"], "s", False)
        for name, seconds in case["phases"].items():
            out[(case["likelihood"], f"`{name}`")] = (seconds, "s", False)
    out[("", "total benchmark")] = (record["total_seconds"], "s", False)
    return out


def _valid(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def check_imported(*, label: str, round_index: int, sha: str | None, lumina_file: str, lumina_commit: str | None,
                   root: Path) -> None:
    """Exit unless lumina was imported from ``root`` at commit ``sha`` (any commit when ``sha`` is None)."""
    where = f"{SIDES[label]} run of round {round_index}"
    if not Path(lumina_file).resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"The {where} imported {lumina_file}, which is not in {root}.")
    if lumina_commit is None or (sha is not None and lumina_commit != sha):
        raise SystemExit(f"The {where} imported lumina at commit {lumina_commit}, not {sha}.")


def load_runs(runs_dir: Path, roots: dict[str, Path]) -> dict[str, dict[int, dict[str, Any]]]:
    """label -> round -> record of every ``*/benchmark.json`` under ``runs_dir``, each checked against its checkout."""
    runs: dict[str, dict[int, dict[str, Any]]] = {label: {} for label in SIDES}
    for path in sorted(runs_dir.glob("*/benchmark.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("label") not in SIDES:
            raise SystemExit(f"{path} has label {record.get('label')!r}, not one of {', '.join(SIDES)}.")
        check_imported(label=record["label"], round_index=record["round"], sha=record["sha"], lumina_file=record["lumina_file"],
                       lumina_commit=record["lumina_commit"], root=roots[record["label"]])
        if record["round"] in runs[record["label"]]:
            raise SystemExit(f"Two {SIDES[record['label']]} runs of round {record['round']} under {runs_dir}.")
        runs[record["label"]][record["round"]] = record
    for label, side in runs.items():
        if len({record["sha"] for record in side.values()}) > 1:
            raise SystemExit(f"The {SIDES[label]} runs were made at different commits.")
    return runs


def paired_delta(base: list[float | None], pr: list[float | None], *, can_warn: bool) -> dict[str, Any] | None:
    """Delta of the PR minimum over the base minimum, in percent, from the values of the paired rounds in order.

    Shared runners mostly slow a run down, so the minimum over rounds is the estimate least touched by other load
    (including the cold first run of the job).  ``low``/``high`` are the smallest and largest per-round PR/base ratio,
    as percent changes.  A row that can warn does so when the delta is above WARN_PERCENT and the PR was more than
    ROUND_MARGIN times the base in every round.  None when a value is missing, not finite or not positive.
    """
    if len(base) != len(pr):
        raise ValueError(f"{len(base)} base values but {len(pr)} PR values.")
    if not all(_valid(value) for value in base + pr):
        return None
    ratios = [p / b for b, p in zip(base, pr)]
    delta = 100.0 * (min(pr) / min(base) - 1.0)
    every_round_slower = all(ratio > ROUND_MARGIN for ratio in ratios)
    return {"delta": delta, "low": 100.0 * (min(ratios) - 1.0), "high": 100.0 * (max(ratios) - 1.0),
            "every_round_slower": every_round_slower, "warn": can_warn and delta > WARN_PERCENT and every_round_slower}


def _fmt_value(values: list[float | None], unit: str) -> str:
    """Minimum over runs, or n/a when a run lacks a usable value."""
    if not values or not all(_valid(value) for value in values):
        return "n/a"
    return f"{min(values):.3f} ms" if unit == "ms" else f"{min(values):.2f} s"


def _fmt_delta(delta: dict[str, Any] | None) -> str:
    if delta is None:
        return "n/a"
    return (f"{delta['delta']:+.1f}% (rounds {delta['low']:+.0f}% to {delta['high']:+.0f}%)"
            + (" :warning:" if delta["warn"] else ""))


def _order(records: list[dict[str, Any]]) -> str:
    return ", ".join(SIDES[record["label"]] for record in sorted(records, key=lambda record: record["started_unix"]))


def _rows(records: list[dict[str, Any]]) -> tuple[list[dict], dict[tuple[str, str], tuple[str, bool]]]:
    """metrics() of each record and every row any of them has, in first-seen order, with its unit and warn switch."""
    rows = [metrics(record) for record in records]
    keys: dict[tuple[str, str], tuple[str, bool]] = {}
    for row in rows:
        for key, (_, unit, can_warn) in row.items():
            keys.setdefault(key, (unit, can_warn))
    return rows, keys


def _header(record: dict[str, Any], runs_line: str) -> list[str]:
    case = record["cases"][0]
    return [MARKER, "### lumina PR benchmark", "",
            f"Model: FF-SIMS Hera ({case['n_images']} images in {case['n_systems']} systems, {case['n_members']} members, {case['n_unconstrained']} free parameters), "
            f"`ubuntu-latest` CPU, {record['trials']} trials x {record['repeats']} evaluations per run, L-BFGS for {record['lbfgs_iters']} iterations.",
            "", runs_line + " Each value is the minimum over runs, for the evaluation rows of each run's median trial. "
            "n/a marks a row that a run lacks or timed as not a positive number.", ""]


def render_markdown(records: list[dict[str, Any]], *, note: str, workflow_url: str) -> str:
    """PR timings only: the minimum over ``records``, runs of one commit."""
    runs_line = f"{len(records)} PR runs, each a fresh process." if len(records) > 1 else "One PR run."
    lines = _header(records[0], runs_line) + [f"Commit `{records[0]['sha'][:12]}`. {note}", "",
                                              "| likelihood | metric | value |", "| --- | --- | ---: |"]
    rows, keys = _rows(records)
    for key, (unit, _) in keys.items():
        lines.append(f"| {key[0]} | {key[1]} | {_fmt_value([row.get(key, (None,))[0] for row in rows], unit)} |")
    lines.extend(["", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def render_comparison_markdown(base_runs: dict[int, dict[str, Any]], pr_runs: dict[int, dict[str, Any]], *,
                               planned_rounds: int, workflow_url: str) -> str:
    """Base-versus-PR table over the rounds that have both runs (at least ``MIN_PAIRED_ROUNDS``)."""
    rounds = sorted(set(base_runs) & set(pr_runs))
    records = [base_runs[r] for r in rounds] + [pr_runs[r] for r in rounds]
    planned = f" ({planned_rounds} planned)" if planned_rounds != len(rounds) else ""
    runs_line = f"{len(rounds)} rounds{planned} of base and PR, each run a fresh process, in the order {_order(records)}."
    rows, keys = _rows(records)
    base_rows, pr_rows = rows[:len(rounds)], rows[len(rounds):]
    lines = _header(pr_runs[rounds[0]], runs_line) + [
        "| likelihood | metric | base | PR | delta |", "| --- | --- | ---: | ---: | ---: |",
        f"| | commit | `{base_runs[rounds[0]]['sha'][:12]}` | `{pr_runs[rounds[0]]['sha'][:12]}` | |"]
    for key, (unit, can_warn) in keys.items():
        base = [row.get(key, (None,))[0] for row in base_rows]
        pr = [row.get(key, (None,))[0] for row in pr_rows]
        lines.append(f"| {key[0]} | {key[1]} | {_fmt_value(base, unit)} | {_fmt_value(pr, unit)} | "
                     f"{_fmt_delta(paired_delta(base, pr, can_warn=can_warn))} |")
    lines.extend(["", "Each round times base and PR in fresh processes one after the other, alternating which goes first. "
                      "Other load on a shared runner slows runs down, so the delta compares the PR minimum with the base minimum; the "
                      "range after it is the smallest and largest PR/base ratio of a single round. "
                      f":warning: marks a log-likelihood or value+grad row whose delta is above {WARN_PERCENT:.0f}% and whose PR "
                      f"run was more than {100 * (ROUND_MARGIN - 1):.0f}% slower than the base run in every round. Compile times, "
                      "the setup phases, L-BFGS and the total are for information and never get a warning.",
                  "", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def compare(runs_dir: Path, *, base_root: Path, head_root: Path, planned_rounds: int, workflow_url: str) -> str:
    runs = load_runs(runs_dir, {"base": base_root, "pr": head_root})
    if not runs["pr"]:
        raise SystemExit(f"No PR run under {runs_dir}.")
    paired = set(runs["base"]) & set(runs["pr"])
    if len(paired) >= MIN_PAIRED_ROUNDS:
        return render_comparison_markdown(runs["base"], runs["pr"], planned_rounds=planned_rounds, workflow_url=workflow_url)
    note = ("No base result to compare against." if not runs["base"] else
            f"Only {len(paired)} of {planned_rounds} rounds have a base run; a comparison needs {MIN_PAIRED_ROUNDS}.")
    return render_markdown([runs["pr"][r] for r in sorted(runs["pr"])], note=note, workflow_url=workflow_url)


def _run_command(args: argparse.Namespace) -> None:
    if args.checkout is not None:
        # Fail before the timings, so a side that cannot import its own checkout counts as a failed run.
        lumina_file, lumina_commit = imported_lumina()
        check_imported(label=args.label, round_index=args.round, sha=args.sha, lumina_file=lumina_file,
                       lumina_commit=lumina_commit, root=args.checkout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(label=args.label, round_index=args.round, sha=args.sha, catalog_root=args.catalog_root,
                           likelihoods=args.likelihoods, repeats=args.repeats, trials=args.trials, lbfgs_iters=args.lbfgs_iters)
    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "output").write_text(render_markdown([result], note="No base result to compare against.",
                                                            workflow_url=_workflow_url()), encoding="utf-8")


def _compare_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = compare(args.runs_dir, base_root=args.base_root, head_root=args.head_root, planned_rounds=args.rounds,
                   workflow_url=_workflow_url())
    (args.output_dir / "output").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--label", choices=sorted(SIDES), default="pr")
    run_parser.add_argument("--round", type=int, default=1)
    run_parser.add_argument("--sha", default=None, help="commit the imported lumina must be at (default: whatever it is)")
    run_parser.add_argument("--checkout", type=Path, default=None, help="refuse to run unless lumina is imported from this checkout")
    run_parser.add_argument("--catalog-root", type=Path, default=REPO_ROOT)
    run_parser.add_argument("--likelihoods", nargs="+", choices=sorted(LIKELIHOODS), default=["smoothed", "grf_gp"])
    run_parser.add_argument("--repeats", type=int, default=int(os.getenv("LENSCLUSTER_BENCH_REPEATS", "50")))
    run_parser.add_argument("--trials", type=int, default=int(os.getenv("LENSCLUSTER_BENCH_TRIALS", "3")))
    run_parser.add_argument("--lbfgs-iters", type=int, default=int(os.getenv("LENSCLUSTER_BENCH_LBFGS_ITERS", "25")))
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--runs-dir", type=Path, required=True, help="holds one folder with a benchmark.json per run")
    compare_parser.add_argument("--base-root", type=Path, required=True, help="checkout every base run must import lumina from")
    compare_parser.add_argument("--head-root", type=Path, required=True, help="checkout every PR run must import lumina from")
    compare_parser.add_argument("--rounds", type=int, default=int(os.getenv("LENSCLUSTER_BENCH_ROUNDS", "3")))
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _compare_command(args) if args.command == "compare" else _run_command(args)


if __name__ == "__main__":
    sys.exit(main())
