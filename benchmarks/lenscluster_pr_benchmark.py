"""Likelihood and gradient timings on the FF-SIMS Hera model, for the PR benchmark comment.

``run`` times the lumina that Python imports, in one fresh process, and writes ``benchmark.json`` and a markdown
``output``; the record keeps the imported ``lumina.__file__`` and the commit of its checkout.  ``compare`` reads the
``benchmark.json`` of every run under ``--runs-dir`` and renders the base-versus-PR table that
.github/workflows/publish-pr-benchmarks.yml posts on the pull request.

The workflow runs base and PR in alternating rounds (base, PR, then PR, base, ...), each run a fresh process with
``--catalog-root`` on the PR checkout so the inputs are the same.  ``compare`` first checks that every base run imported
lumina from the base checkout and every PR run from the PR checkout, at the commit the run was given.  Base and PR of
one round form a pair: the delta of a metric is the geometric mean over rounds of the PR/base ratio, its uncertainty the
standard error of the log ratio between rounds combined with the spread of the trials within each run.  A row gets a
warning when the slowdown is above 10%, more than twice its uncertainty, and every round is slower.  With fewer than two
paired rounds the table shows PR timings only.
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


def metrics(record: dict[str, Any]) -> dict[tuple[str, str], tuple[float, float | None, str]]:
    """(likelihood, metric) -> (value, standard error within the run or None, unit) for every row of the table."""
    out: dict[tuple[str, str], tuple[float, float | None, str]] = {}
    for case in record["cases"]:
        for row in case["timings"]:
            out[(case["likelihood"], row["name"])] = (row["ms_per_eval"], row["ms_per_eval_stderr"], "ms")
            out[(case["likelihood"], f"{row['name']} compile")] = (row["compile_seconds"], None, "s")
        for name, seconds in case["phases"].items():
            out[(case["likelihood"], f"`{name}`")] = (seconds, None, "s")
    out[("", "total benchmark")] = (record["total_seconds"], None, "s")
    return out


def verify_checkout(record: dict[str, Any], root: Path) -> None:
    """Exit unless the run imported lumina from ``root`` at the commit the run was given."""
    where = f"{SIDES[record['label']]} run of round {record['round']}"
    if not Path(record["lumina_file"]).resolve().is_relative_to(root.resolve()):
        raise SystemExit(f"The {where} imported {record['lumina_file']}, which is not in {root}.")
    if record["lumina_commit"] is None or record["lumina_commit"] != record["sha"]:
        raise SystemExit(f"The {where} imported lumina at commit {record['lumina_commit']}, not {record['sha']}.")


def load_runs(runs_dir: Path, roots: dict[str, Path]) -> dict[str, dict[int, dict[str, Any]]]:
    """label -> round -> record of every ``*/benchmark.json`` under ``runs_dir``, each checked against its checkout."""
    runs: dict[str, dict[int, dict[str, Any]]] = {label: {} for label in SIDES}
    for path in sorted(runs_dir.glob("*/benchmark.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        verify_checkout(record, roots[record["label"]])
        if record["round"] in runs[record["label"]]:
            raise SystemExit(f"Two {SIDES[record['label']]} runs of round {record['round']} under {runs_dir}.")
        runs[record["label"]][record["round"]] = record
    for label, side in runs.items():
        if len({record["sha"] for record in side.values()}) > 1:
            raise SystemExit(f"The {SIDES[label]} runs were made at different commits.")
    return runs


def paired_delta(base: list[tuple[float, float | None]], pr: list[tuple[float, float | None]]) -> dict[str, Any] | None:
    """Delta in percent of PR over base from per-round pairs of (value, standard error within the run or None).

    The delta is the geometric mean of the per-round ratios; a mean of log ratios treats a slowdown and the matching
    speedup alike, and it keeps a single outlier round visible rather than hiding it as a median of two or three would
    (the every-round rule guards the warning against that round instead).  Its variance is the scatter of the log
    ratios between rounds over n plus the within-run variances over n^2, so a lucky agreement of two rounds cannot shrink
    the uncertainty below the timing noise inside the runs.
    """
    if any(b <= 0.0 or p <= 0.0 for (b, _), (p, _) in zip(base, pr)):
        return None
    logs = [math.log(p / b) for (b, _), (p, _) in zip(base, pr)]
    within = [(pe / p) ** 2 + (be / b) ** 2 for (b, be), (p, pe) in zip(base, pr) if be is not None and pe is not None]
    n = len(logs)
    variance = statistics.variance(logs) / n + sum(within) / n**2
    mean = statistics.fmean(logs)
    delta = 100.0 * math.expm1(mean)
    stderr = 100.0 * math.exp(mean) * math.sqrt(variance)
    every_round_slower = all(value > 0.0 for value in logs)
    return {"delta": delta, "stderr": stderr, "every_round_slower": every_round_slower,
            "warn": delta > WARN_PERCENT and delta > 2.0 * stderr and every_round_slower}


def _fmt_value(values: list[tuple[float, float | None]], unit: str) -> str:
    """Mean over runs; ms rows add the standard error between runs, or the one within the run for a single run."""
    mean = _mean([value for value, _ in values])
    if unit != "ms":
        return f"{mean:.2f} s"
    stderr = _stderr([value for value, _ in values]) if len(values) > 1 else values[0][1] or 0.0
    return f"{mean:.3f} +/- {stderr:.3f} ms"


def _fmt_delta(delta: dict[str, Any] | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta['delta']:+.1f}% +/- {delta['stderr']:.1f}%" + (" :warning:" if delta["warn"] else "")


def _order(records: list[dict[str, Any]]) -> str:
    return ", ".join(SIDES[record["label"]] for record in sorted(records, key=lambda record: record["started_unix"]))


def _header(record: dict[str, Any], runs_line: str) -> list[str]:
    case = record["cases"][0]
    return [MARKER, "### lumina PR benchmark", "",
            f"Model: FF-SIMS Hera ({case['n_images']} images in {case['n_systems']} systems, {case['n_members']} members, {case['n_unconstrained']} free parameters), "
            f"`ubuntu-latest` CPU, {record['trials']} trials x {record['repeats']} evaluations per run, L-BFGS for {record['lbfgs_iters']} iterations.",
            "", runs_line, ""]


def render_markdown(records: list[dict[str, Any]], *, note: str, workflow_url: str) -> str:
    """PR timings only: the mean over ``records``, runs of one commit."""
    runs_line = (f"{len(records)} PR runs, each a fresh process." if len(records) > 1 else "One PR run.")
    lines = _header(records[0], runs_line) + [f"Commit `{records[0]['sha'][:12]}`. {note}", "",
                                              "| likelihood | metric | value |", "| --- | --- | ---: |"]
    rows = [metrics(record) for record in records]
    for key, (_, _, unit) in rows[0].items():
        if all(key in row for row in rows):
            lines.append(f"| {key[0]} | {key[1]} | {_fmt_value([row[key][:2] for row in rows], unit)} |")
    lines.extend(["", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def render_comparison_markdown(base_runs: dict[int, dict[str, Any]], pr_runs: dict[int, dict[str, Any]], *,
                               planned_rounds: int, workflow_url: str) -> str:
    """Base-versus-PR table over the rounds that have both runs (at least ``MIN_PAIRED_ROUNDS``)."""
    rounds = sorted(set(base_runs) & set(pr_runs))
    records = [base_runs[r] for r in rounds] + [pr_runs[r] for r in rounds]
    planned = f" ({planned_rounds} planned)" if planned_rounds != len(rounds) else ""
    runs_line = f"{len(rounds)} rounds{planned} of base and PR, each run a fresh process, in the order {_order(records)}."
    base_rows = [metrics(base_runs[r]) for r in rounds]
    pr_rows = [metrics(pr_runs[r]) for r in rounds]
    lines = _header(pr_runs[rounds[0]], runs_line) + [
        "| likelihood | metric | base | PR | delta |", "| --- | --- | ---: | ---: | ---: |",
        f"| | commit | `{base_runs[rounds[0]]['sha'][:12]}` | `{pr_runs[rounds[0]]['sha'][:12]}` | |"]
    for key, (_, _, unit) in pr_rows[0].items():
        if not all(key in row for row in base_rows + pr_rows):
            continue
        base = [row[key][:2] for row in base_rows]
        pr = [row[key][:2] for row in pr_rows]
        lines.append(f"| {key[0]} | {key[1]} | {_fmt_value(base, unit)} | {_fmt_value(pr, unit)} | {_fmt_delta(paired_delta(base, pr))} |")
    lines.extend(["", "Each round times base and PR in fresh processes one after the other, alternating which goes first. "
                      "The delta is the geometric mean over rounds of the PR/base ratio; its uncertainty combines the scatter of that "
                      "ratio between rounds with the spread of the trials within each run. "
                      f":warning: marks a slowdown above {WARN_PERCENT:.0f}% that is more than twice its uncertainty and that every round shows.",
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
        verify_checkout({"label": args.label, "round": args.round, "sha": args.sha or lumina_commit, "lumina_file": lumina_file,
                         "lumina_commit": lumina_commit}, args.checkout)
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
