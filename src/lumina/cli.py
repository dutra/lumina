"""Run a cluster declared in a catalog folder: ``lumina <folder>`` (or ``python -m lumina <folder>``).

The folder holds ``run.yaml`` and the ``priors.yaml`` it names (:mod:`lumina.run_files`).  Paths in them are
resolved against the working directory, so run from the repository root::

    lumina catalogs/hera --dry-run
    lumina catalogs/m0416/canucs2025 --profile fast
    lumina catalogs/tngc014/large --set sampling.warmup=1000 --set runtime.chains=8

Every run is a new folder, ``results/<name>/<mon><day><letter>_<description>``, e.g.
``results/hera/nov21a_w500s250t7_allblocks``: the date the folder is created, the next letter of that cluster and
date, and :func:`describe` of the run (:mod:`lumina.run_names`).  The same command run twice makes two runs.
``--run-label`` names the new folder by hand (an existing one is refused), and ``--results`` moves the tree.  The
files the run read and the command are copied to the run's ``inputs/`` folder.  The run folder is locked
(``run.lock``, :mod:`lumina.run_lock`) while the command works in it.  The command prints the run folder when it
starts and, when it ends, the ``lumina resume`` command that continues it.

``lumina resume RUN_DIR`` continues a run folder with its own saved settings (:mod:`lumina.resume`): it is the only
way back into an existing run, e.g. ``lumina resume RUN_DIR --stage final`` after ``--stage init``, and
``lumina outputs RUN_DIR`` reruns the outputs stage of a finished run without fitting (:mod:`lumina.outputs.command`);
a cluster folder literally named ``resume`` or ``outputs`` is given as ``./resume`` or ``./outputs``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

# Likelihood (mode, kernel) -> its tag in the run label; the GRF with a knee spectrum (the cluster runs) has none.
LIKELIHOOD_TAGS = {("gp_image_plane", "knee"): "", ("gp_image_plane", "image_plane"): "imagegp", ("smoothed", "image_plane"): "smoothed"}


def describe(config, profiles=()) -> str:
    """The description in a run's name: NUTS warmup, samples and tree depth, then the GP blocks (GRF likelihood) or the likelihood,
    the image catalog when it is not the one of ``priors.yaml``'s folder, ``uncertain`` with uncertain images,
    ``powerlaw`` with cored power-law members (and their slope when it is held fixed), and the profiles.  E.g. ``w500s250t7_allblocks``, ``w500s250t8_systemblocks_fast``, ``w500s250t8_smoothed_trusted_uncertain_fast``."""
    sampling, likelihood = config.sampling, config.likelihood
    parts = [f"w{sampling.warmup}s{sampling.samples}t{sampling.max_tree_depth}"]
    tag = LIKELIHOOD_TAGS.get((likelihood.likelihood_mode, likelihood.gp_kernel), likelihood.likelihood_mode.replace("_", ""))
    parts.append(f"{likelihood.gp_blocks}blocks" if not tag else tag)
    images = Path(config.model.image_constraints.catalog_path)
    if images.name != "images.yaml":
        parts.append(images.stem)
    if config.partition is not None:
        parts.append("uncertain")
    if any(int(population.profile_type) == 102 for population in config.model.member_populations):
        fixed = "slope" in config.scaling.fixed_relation_fields
        parts.append("powerlaw" + (f"{config.model.member_populations[0].slope:g}" if fixed else ""))
    return "_".join([*parts, *profiles])


def _segment(text: str) -> str:
    if not text.strip() or text in {".", ".."} or "/" in text or "\\" in text:
        raise argparse.ArgumentTypeError("must be a nonempty single path segment")
    return text


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _accept(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must lie in (0, 1)")
    return value


# Named options: the fields each sets, after the profiles and before --set (so --set wins).
FIELD_OPTIONS = {"chains": ("runtime.chains",), "seed": ("runtime.seed",), "starts": ("search.n_starts",),
                 "warm_start": ("search.warm_start",), "warmup": ("sampling.warmup",), "samples": ("sampling.samples",),
                 "tree_depth": ("sampling.max_tree_depth",), "target_accept": ("sampling.target_accept",),
                 "draws": ("outputs.draws", "outputs.image_draws")}


def _option_overrides(args) -> list[str]:
    """The ``section.field=value`` assignments of the named options that were given."""
    out = [f"{field}={getattr(args, name)}" for name, targets in FIELD_OPTIONS.items() if getattr(args, name) is not None
           for field in targets]
    return out + (["runtime.skip_plots=true"] if args.skip_plots else [])


def _slope(text: str) -> float:
    value = float(text)
    if not 1.0 < value < 3.0:
        raise argparse.ArgumentTypeError("must lie in (1, 3)")
    return value


def _folder(text: str) -> str:
    if not (Path(text) / "run.yaml").is_file():
        raise argparse.ArgumentTypeError(f"no run.yaml in {text!r}")
    return text


def parser() -> argparse.ArgumentParser:
    """The command-line options."""
    p = argparse.ArgumentParser(prog="lumina", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="lumina resume RUN_DIR [--stage init|final|both] [--remap OLD=NEW] [--set outputs.FIELD=VALUE] "
                                       "[--draws N] [--dry-run] continues a run folder with its own saved settings; see "
                                       "lumina resume --help.  "
                                       "lumina outputs RUN_DIR [--only A,B | --skip C,D | --all] [--remap OLD=NEW] [--set ...] [--draws N] "
                                       "reruns the outputs stage of a finished run without fitting; see lumina outputs --help.  "
                                       "A cluster folder named resume or outputs is given as ./resume or ./outputs.")
    p.add_argument("cluster", type=_folder, help="the cluster folder, e.g. catalogs/hera")
    p.add_argument("--profile", action="append", default=[], metavar="NAME",
                   help="apply a profile of run.yaml (fast; jwst or hst for the figures' imaging), in order; repeatable")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE",
                   help="override one field after the profiles, e.g. sampling.warmup=1000, runtime.skip_plots=true, "
                        "search.warm_start=results/<name>/<run>, sampling.wbic=true; repeatable")
    p.add_argument("--run-label", type=_segment, default=None,
                   help="the name of the new run folder (default <mon><day><letter>_<description>, e.g. "
                        "nov21a_w500s250t7_allblocks); an existing folder is refused: continue it with lumina resume")
    p.add_argument("--results", default="results", metavar="DIR", help="the results tree, DIR/<name>/<run> (default results)")
    p.add_argument("--chains", type=_positive_int, metavar="N", help="NUTS chains, one CPU device each (runtime.chains)")
    p.add_argument("--seed", type=int, metavar="N", help="random seed of the search, the sampler and the reporting draws (runtime.seed)")
    p.add_argument("--starts", type=_positive_int, metavar="N", help="starts of the initial search (search.n_starts)")
    p.add_argument("--warm-start", metavar="RUN_DIR", help="start the search from a previous run's saved start (search.warm_start)")
    p.add_argument("--warmup", type=_positive_int, metavar="N", help="NUTS warmup draws per chain (sampling.warmup)")
    p.add_argument("--samples", type=_positive_int, metavar="N", help="NUTS draws per chain after warmup (sampling.samples)")
    p.add_argument("--tree-depth", type=_positive_int, metavar="N", help="NUTS maximum tree depth (sampling.max_tree_depth)")
    p.add_argument("--target-accept", type=_accept, metavar="P", help="NUTS target acceptance in (0, 1) (sampling.target_accept)")
    p.add_argument("--draws", type=_positive_int, metavar="N",
                   help="posterior draws behind every reported product (outputs.draws and outputs.image_draws)")
    p.add_argument("--skip-plots", action="store_true", help="no figures or tables after sampling (runtime.skip_plots)")
    p.add_argument("--member-profile", choices=("dpie", "power-law"), default=None,
                   help="the member halos: dpie, or power-law, the cored power law (the dPIE with a free 3D slope gamma, "
                        "rho ~ r^-gamma between core and cut, at the dPIE's mass inside the cut), one sampled slope per "
                        "population.  Default: what priors.yaml declares")
    p.add_argument("--member-slope", type=_slope, default=None, metavar="G",
                   help="power-law members: hold the slope at G, in (1, 3), instead of sampling it")
    p.add_argument("--images", default=None, metavar="YAML",
                   help="fit this image catalog instead of the one priors.yaml names (e.g. a degraded catalog). Images of "
                        "unknown family: --set partition.catalog_path=CSV (null for the untrusted images of the catalog), "
                        "partition.redshift_grid=[Z_MIN,Z_MAX,POINTS]")
    p.add_argument("--stage", choices=("init", "final", "both"), default="both",
                   help="the stages to run in the new folder (default both); final is refused, as a new run has no "
                        "init stage: the final stage of an existing run is lumina resume RUN_DIR --stage final")
    p.add_argument("--dry-run", action="store_true",
                   help="print the run folder it would create, classify the members and print the model's size; "
                        "write nothing")
    return p


def build_config(cluster, *, profiles=(), overrides=(), run_label=None, images=None, results="results",
                 member_profile=None, member_slope=None):
    """The configuration of a cluster folder with the profiles and overrides, writing to
    ``<results>/<name>/<run_label>`` (run_label None leaves ``paths.run_name`` None: the command assigns it,
    :func:`lumina.run_names.new_run`), with ``images`` in place of the image catalog of ``priors.yaml``.  ``member_profile`` ("dpie" or "power-law") sets the profile of every member population
    (None keeps what ``priors.yaml`` declares), and ``member_slope`` holds the power-law slope fixed.

    Returns
    -------
    LensClusterSolverConfig
        Not validated.
    """
    from .run_files import load_run
    config = load_run(cluster, profiles, overrides)
    if images is not None:
        config = replace(config, model=replace(config.model, image_constraints=replace(config.model.image_constraints, catalog_path=images)))
    if member_profile is not None or member_slope is not None:
        # Profile 102 is the cored power law (MemberPopulationConfig): one 3D slope per population, sampled or held fixed.
        code = {"dpie": 81, "power-law": 102, None: None}[member_profile]
        populations = tuple(replace(p, **({} if code is None else {"profile_type": code}),
                                    **({} if member_slope is None else {"slope": float(member_slope)}))
                            for p in config.model.member_populations)
        if member_slope is not None and any(int(p.profile_type) != 102 for p in populations):
            raise ValueError("--member-slope needs power-law members (--member-profile power-law, or profile_type 102 in priors.yaml).")
        config = replace(config, model=replace(config.model, member_populations=populations))
        if member_slope is not None and "slope" not in config.scaling.fixed_relation_fields:
            config = replace(config, scaling=replace(config.scaling, fixed_relation_fields=(*config.scaling.fixed_relation_fields, "slope")))
    return replace(config, paths=replace(config.paths, output_dir=str(Path(results) / config.paths.figure_prefix), run_name=run_label))


def _fast_warning(config) -> None:
    """A banner that cannot be missed: a fast run uses exploratory settings."""
    from rich.console import Console
    from rich.panel import Panel
    members = config.members
    cut = members.min_deflection_arcsec
    text = ("This is a FAST run.  Its settings are exploratory:\n"
            + (f"  - members deflecting every image by less than {cut:g}\" are dropped from the model\n" if cut is not None
               else "  - every catalog member is kept (no impact cut)\n")
            + f"  - far-field members are allowed down to {members.far_field_cut_radii:g} cut radii from an image (default 3)\n"
            f"  - {config.search.n_starts} search starts, {config.sampling.warmup} warmup + {config.sampling.samples} draws per chain "
            f"at tree depth {config.sampling.max_tree_depth}, target accept {config.sampling.target_accept:g}\n"
            f"  - GP covariance blocks: {config.likelihood.gp_blocks} (full runs default to all blocks)")
    Console().print(Panel(text, title="[bold]WARNING: fast profile[/bold]", border_style="yellow", style="yellow", expand=False))


def main(argv=None) -> int:
    """Run (or dry-run) one cluster folder in a new run folder (:func:`lumina.run_names.new_run`), or with ``resume``
    first continue a run folder (:func:`lumina.resume.main`), or with ``outputs`` first rerun the outputs stage of a
    finished run (:func:`lumina.outputs.command.main`); returns the exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "outputs":
        from .outputs.command import main as outputs_main
        return outputs_main(argv[1:])
    if argv and argv[0] == "resume":
        from .resume import main as resume_main
        return resume_main(argv[1:])
    args = parser().parse_args(argv)
    if args.stage == "final":
        print("lumina: --stage final needs a completed init stage, and a new run folder has none; run the final stage "
              "of an existing run with: lumina resume RUN_DIR --stage final", file=sys.stderr, flush=True)
        return 2
    args.overrides = [*_option_overrides(args), *args.overrides]
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lumina")
    from .config import UnknownField
    from .cpu_runtime import configure_cpu_runtime, resolve_cpu_plan
    from .run_files import read_run
    from .run_names import new_run
    log = lambda text: print(text, flush=True)
    try:
        # The CPU devices are set before anything starts JAX (building a prior does).
        runtime = read_run(args.cluster, args.profile, args.overrides).get("runtime") or {}
        chains, method = int(runtime.get("chains", 1)), runtime.get("nuts_chain_method", "parallel")
        cpu_plan = resolve_cpu_plan(chains, method)
        if not args.dry_run:
            configure_cpu_runtime(chains, method)
        config = build_config(args.cluster, profiles=args.profile, overrides=args.overrides, results=args.results,
                              member_profile=args.member_profile, member_slope=args.member_slope,
                              run_label=args.run_label, images=args.images)
        description = describe(config, args.profile)
        # Validated before the folder is created (the name is only a stand-in here).
        config = replace(config, paths=replace(config.paths, run_name=args.run_label or description)).validate()
        path, lock = new_run(config.paths.output_dir, description, run_label=args.run_label, dry_run=args.dry_run, log=log)
    except (ValueError, UnknownField) as error:     # a bad run file, profile or --set value, a refused folder: a usage error
        print(f"lumina: {error}", file=sys.stderr, flush=True)
        return 2
    config = replace(config, paths=replace(config.paths, run_name=path.name))
    if args.dry_run:
        from .pipeline import dry_run
        log(f"[run] would create {path}")
        print(json.dumps({"dry_run": True, "run": str(path), "cpu_plan": cpu_plan.to_dict(), **dry_run(config)},
                         indent=2, default=str))
        return 0
    try:
        return _run_in(path, config, args, log)
    finally:
        lock.release()


def _run_in(path, config, args, log) -> int:
    """The run of ``config`` in the new folder ``path`` (whose lock the caller holds)."""
    from .resume import next_step
    from .run_files import input_files
    log(f"[run] created {path} (if it stops, continue it with: lumina resume {shlex.quote(str(path))})")
    inputs = path / "inputs"            # the files the run was declared with, and the command
    inputs.mkdir()
    for index, source in enumerate(input_files(args.cluster, args.profile)):
        shutil.copy2(source, inputs / f"{index}_{source.name}")
    (inputs / "command.json").write_text(json.dumps(
        {"cluster": args.cluster, "profiles": args.profile, "overrides": args.overrides, "images": args.images,
         "member_profile": args.member_profile, "member_slope": args.member_slope}, indent=2))
    from .pipeline import run
    fast = "fast" in args.profile
    if fast:
        _fast_warning(config)
    print(f"[config] cluster={args.cluster} profiles={','.join(args.profile) or '-'} run_name={config.paths.run_name} "
          f"output_dir={config.paths.output_dir} stage={args.stage} chains={config.runtime.chains} "
          f"search={config.search.halo_method}+{config.search.member_method} n_starts={config.search.n_starts} "
          f"likelihood={config.likelihood.likelihood_mode}/{config.likelihood.gp_kernel} gp_blocks={config.likelihood.gp_blocks}",
          flush=True)
    start = time.monotonic()
    finished = False
    try:
        run(config, stage=args.stage)
        finished = True
        if fast:
            _fast_warning(config)
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
        log(next_step(path, args.stage, finished))
    return 0
