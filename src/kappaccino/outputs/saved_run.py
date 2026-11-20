"""A finished run loaded back from its folder: configuration, model state, posterior and (for a compute) the evaluator,
without fitting."""
import json
from pathlib import Path
from types import SimpleNamespace


REMAP_HINT = "for a run made on another machine or moved, give --remap OLD=NEW (a prefix of the saved paths and its replacement)"


def _flatten(payload, prefix=""):
    """``{"section.field[index].field": value}`` of the leaves of a configuration dictionary."""
    if isinstance(payload, dict):
        out = {}
        for key, value in payload.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(payload, list):
        out = {}
        for index, value in enumerate(payload):
            out.update(_flatten(value, f"{prefix}[{index}]"))
        return out
    return {prefix: payload}


def _replace_prefix(value: str, remaps) -> str:
    for old, new in remaps:
        old, new = old.rstrip("/"), new.rstrip("/")
        if old and (value == old or value.startswith(old + "/")):
            return new + value[len(old):]
    return value


def remap_paths(config, remaps, *, log=print):
    """``config`` with each saved path that starts with ``OLD`` (a whole path prefix) starting with ``NEW`` instead,
    for every ``(OLD, NEW)`` of ``remaps`` (the first that matches wins).  Every string field of the configuration is
    considered, so catalog paths, imaging folders, truth and deflection maps and candidate catalogs are all covered;
    each field changed is logged."""
    from ..config import LensClusterSolverConfig
    remaps = [(str(old), str(new)) for old, new in remaps]
    if not remaps:
        return config
    payload = config.to_json_dict()
    changed = []

    def walk(value, where):
        if isinstance(value, dict):
            return {key: walk(item, f"{where}.{key}" if where else key) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item, f"{where}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, str):
            moved = _replace_prefix(value, remaps)
            if moved != value:
                changed.append((where, value, moved))
            return moved
        return value

    payload = walk(payload, "")
    for where, old, new in changed:
        log(f"[outputs] remap {where}: {old} -> {new}")
    if not changed:
        log(f"[outputs] remap: no saved path starts with {', '.join(old for old, _ in remaps)}")
    return LensClusterSolverConfig.from_json_dict(payload)


def apply_overrides(config, overrides):
    """``config`` with the ``section.field=value`` assignments ``overrides`` (the syntax of ``kappaccino --set``,
    ``run_files.read_run``).  Only settings outside the run hash may change (``pipeline.resume_payload`` leaves them
    out: the ``outputs`` section, ``config.REPORT_ONLY_FIELDS``, ``config.PERFORMANCE_FIELDS``, the WBIC settings) and
    the ``paths`` section, which the hash leaves out too: anything else would no longer describe the saved fit.

    Raises
    ------
    ValueError
        For a malformed assignment, an unknown section or field, or a field that is part of the run hash.
    """
    from .. import config as _config
    from ..pipeline import resume_payload
    from ..run_files import _dotted, _merge, parse_value
    overrides = tuple(overrides)
    if not overrides:
        return config
    overlay = _dotted(overrides)
    from dataclasses import fields
    sections = {f.name for f in fields(_config.LensClusterSolverConfig)}
    unknown = sorted(set(overlay) - sections)
    if unknown:
        raise ValueError(f"--set: unknown configuration sections {unknown}")
    before = config.to_json_dict()
    after = _merge(before, _config._jsonable(parse_value(overlay)))
    try:
        updated = _config.LensClusterSolverConfig.from_json_dict(after)
    except TypeError as error:
        raise ValueError(f"--set: {error}") from None

    def fit_part(payload):
        return _flatten({key: value for key, value in resume_payload(payload).items() if key != "paths"})

    old, new = fit_part(before), fit_part(updated.to_json_dict())
    fit_fields = sorted(key for key in set(old) | set(new) if old.get(key) != new.get(key))
    if fit_fields:
        raise ValueError(f"--set may change only report settings (outside the run hash); {', '.join(fit_fields)} "
                         "would change the saved fit's configuration")
    return updated


def config_value(config, field: str):
    """The value of the dotted field ``section.field`` of ``config`` (None when a section is None)."""
    value = config
    for part in field.split("."):
        value = getattr(value, part) if value is not None else None
    return value


def missing_paths(config, fields) -> list[str]:
    """``"field = path"`` of each set path field (dotted names, ``model.member_populations[i].catalog_path`` included)
    whose file or folder does not exist."""
    flat = _flatten(config.to_json_dict())
    return [f"{field} = {flat[field]}" for field in fields if flat.get(field) and not Path(str(flat[field])).exists()]


def catalog_fields(config) -> list[str]:
    """The path fields the model is built from: the image catalog and each member population's catalog."""
    fields = ["model.image_constraints.catalog_path"]
    fields += [f"model.member_populations[{i}].catalog_path" for i in range(len(config.model.member_populations))]
    if config.partition is not None:
        fields.append("partition.catalog_path")
    return fields


def load_saved_run(run_directory, *, remap=(), overrides=(), transform=None, log=print):
    """``(config, state, posterior)`` of the finished run in ``run_directory``, without its evaluator: the
    configuration of ``final/artifacts``, the state rebuilt with the init stage's member classes and checked against
    the saved parameter order, and the saved posterior.  Building the state reads the catalogs and packs the lens; it
    evaluates no lens and compiles nothing, so this is what a render pass loads (:func:`load_run` adds the evaluator).

    The saved paths are as the run wrote them: relative to the folder it was started from, or absolute.  ``remap``
    (``(OLD, NEW)`` pairs, :func:`remap_paths`) relocates them, ``overrides`` (``section.field=value``,
    :func:`apply_overrides`) changes report settings, and ``transform`` (a function of the configuration) is applied
    last, before the model is built.

    Raises
    ------
    FileNotFoundError
        When a catalog the model is built from does not exist; the message names the field.
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    from ..io import artifacts
    from ..core.state import build_state, with_member_classes
    run = Path(run_directory).resolve()
    init, final = run / 'init/artifacts', run / 'final'
    config = apply_overrides(remap_paths(artifacts.load_config(final / 'artifacts'), remap, log=log), overrides)
    if transform is not None:
        config = transform(config)
    missing = missing_paths(config, catalog_fields(config))
    if missing:
        raise FileNotFoundError(f"run {run}: {'; '.join(missing)} not found; {REMAP_HINT}")
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get('dropped_members', ()))
    centered = set(classes.get('centered_members', []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered for m in state.members.member_ids]))
    state.initial_theta = artifacts.load_start(init)
    saved = json.loads((init / artifacts.STATE_FILE).read_text())
    assert [p.name for p in state.parameter_specs] == [p['name'] for p in saved['parameter_specs']]
    assert list(state.members.member_ids) == saved['member_ids']
    return config, state, artifacts.load_posterior(final / 'artifacts')


def load_run(run_directory, *, physical_draw_indices=None, remap=(), overrides=(), transform=None, log=print):
    """``(config, state, evaluator, posterior, paths)`` of the finished run in ``run_directory``: :func:`load_saved_run`
    (``remap``, ``overrides``, ``transform``) and the evaluator, checked to reproduce the saved physical parameters of
    the best draw; ``paths`` has the final ``figures`` and ``tables``.  ``physical_draw_indices`` rebuilds the latent
    vectors of those saved draws from their physical values (checked to round-trip).

    Raises
    ------
    FileNotFoundError
        When a catalog the model is built from does not exist; the message names the field.
    """
    import jax
    import numpy as np
    from ..likelihood.evaluator import Evaluator
    config, state, posterior = load_saved_run(run_directory, remap=remap, overrides=overrides, transform=transform, log=log)
    final = Path(run_directory).resolve() / 'final'
    best = int(np.nanargmax(posterior.log_prob))
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    if physical_draw_indices is not None:
        # Plot saved physical draws explicitly, preserving physical source positions.
        # This is not a fallback or a modification of the posterior on disk.
        indices = np.unique(np.r_[physical_draw_indices, best]).astype(int)
        source_indices = np.r_[state.images.source_x_param_index, state.images.source_y_param_index]
        mass_indices = np.setdiff1d(np.arange(posterior.samples.shape[1]), source_indices)
        if evaluator.gp_knee:
            from ..core.model import convert_theta_to_latent
            evaluator = evaluator._padded_evaluator()
            whitening_fn = jax.jit(lambda theta: evaluator.statistics(theta)['whitening'])
            reported_fn = jax.jit(evaluator.reported_physical)
        for index in indices:
            np.testing.assert_allclose(np.asarray(evaluator.physical(posterior.samples[index]))[mass_indices],
                                       posterior.physical[index, mass_indices], rtol=1e-10, atol=1e-10)
            if evaluator.gp_knee:
                theta = convert_theta_to_latent(posterior.physical[index], state.parameter_specs)
                theta[source_indices] = 0.
                whitening = whitening_fn(theta)
                layout = evaluator.gp
                mask = np.asarray(layout['knot_mask'])
                family = np.asarray(layout['knot_family'])
                source = np.stack((posterior.physical[index, state.images.source_x_param_index],
                                   posterior.physical[index, state.images.source_y_param_index]), -1)
                target = np.where(mask[..., None], source[np.where(mask, family, 0)], 0.).reshape(mask.shape[0], -1)
                offset = np.linalg.solve(np.asarray(whitening['cholesky']),
                                         (target-np.asarray(whitening['centroid']))[..., None])[..., 0].reshape(*mask.shape, 2)
                real = offset[np.asarray(layout['real_block']), np.asarray(layout['real_knot'])]
                real_family = np.asarray(layout['real_family'])
                theta[np.asarray(state.images.source_x_param_index)[real_family]] = real[:, 0]
                theta[np.asarray(state.images.source_y_param_index)[real_family]] = real[:, 1]
                reported = reported_fn(theta)
            else:
                theta = evaluator.latent_from_reported_physical(posterior.physical[index])
                reported = evaluator.reported_physical(theta)
            np.testing.assert_allclose(reported, posterior.physical[index],
                                       rtol=1e-10, atol=1e-10)
            posterior.samples[index] = theta
        print(f'[outputs] verified {len(indices)} saved physical draws including source positions', flush=True)
    np.testing.assert_allclose(evaluator.reported_physical(posterior.samples[best]), posterior.physical[best], rtol=1e-10, atol=1e-10)

    paths = SimpleNamespace(figures=final / 'figures', tables=final / 'tables')
    return config, state, evaluator, posterior, paths
