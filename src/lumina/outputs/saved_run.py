"""A finished run loaded back from its folder: configuration, model state, posterior and (for a compute) the evaluator,
without fitting."""
import json
from pathlib import Path
from types import SimpleNamespace


REMAP_HINT = ("for a run made on another machine or moved, give --remap OLD=NEW (OLD an absolute prefix of the saved "
              "paths, NEW its replacement)")


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


def path_fields(config) -> list[str]:
    """The dotted names (``section.field``, ``model.member_populations[i].catalog_path``, as :func:`_flatten` names
    them) of every path-valued field of ``config``: the fields of its dataclasses whose type admits a ``Path``, found
    by walking the configuration, so a new path field is covered without a list to keep."""
    from dataclasses import fields, is_dataclass
    names = []

    def walk(node, where):
        for field in fields(node):
            value, name = getattr(node, field.name), f"{where}.{field.name}" if where else field.name
            if is_dataclass(value):
                walk(value, name)
            elif isinstance(value, (tuple, list)) and value and all(is_dataclass(item) for item in value):
                for index, item in enumerate(value):
                    walk(item, f"{name}[{index}]")
            elif "Path" in (field.type if isinstance(field.type, str) else repr(field.type)):
                names.append(name)
    walk(config, "")
    return names


def _set(payload, dotted: str, value) -> None:
    """Set the leaf ``dotted`` (a :func:`_flatten` name) of the configuration dictionary ``payload``."""
    import re
    parts = [(key, None if index is None else int(index)) for key, index in
             ((m.group(1), m.group(2)) for m in re.finditer(r"([^.\[\]]+)(?:\[(\d+)\])?", dotted))]
    node = payload
    for position, (key, index) in enumerate(parts):
        last = position == len(parts) - 1
        if index is None:
            if last:
                node[key] = value
            else:
                node = node[key]
        elif last:
            node[key][index] = value
        else:
            node = node[key][index]


def remap_paths(config, remaps, *, log=print):
    """``config`` with each saved path under ``OLD`` (a whole path prefix) moved under ``NEW`` instead, for every
    ``(OLD, NEW)`` of ``remaps`` (the first that matches wins).  Only path-valued fields (:func:`path_fields`) are
    considered: catalogs, imaging folders, truth and deflection maps, candidate catalogs; never a name such as
    ``paths.run_name`` or ``paths.figure_prefix``.  A relative saved path is taken relative to the working directory
    (as the command reads it) and compared as that absolute path.  Each field changed is logged.

    Raises
    ------
    ValueError
        When an ``OLD`` is not an absolute path, or is the root ``/``.
    """
    import os
    from ..config import LensClusterSolverConfig
    remaps = [(str(old), str(new)) for old, new in remaps]
    if not remaps:
        return config
    for old, _ in remaps:
        if not os.path.isabs(old):
            raise ValueError(f"--remap {old}=...: OLD must be an absolute path (a prefix of the saved paths, as "
                             "printed by the error that named the missing file)")
        if not old.rstrip("/"):
            raise ValueError("--remap /=...: OLD is the root, which would move every absolute path; give the saved "
                             "folder that moved")
    payload = config.to_json_dict()
    flat = _flatten(payload)
    changed = []
    for name in path_fields(config):
        value = flat.get(name)
        if not isinstance(value, str) or not value:
            continue
        absolute = value if os.path.isabs(value) else os.path.normpath(os.path.join(os.getcwd(), value))
        for old, new in remaps:
            old, new = os.path.normpath(old), new.rstrip("/") or "/"
            if absolute == old or absolute.startswith(old + "/"):
                moved = new + absolute[len(old):]
                _set(payload, name, moved)
                changed.append((name, value, moved))
                break
    for where, old, new in changed:
        log(f"[outputs] remap {where}: {old} -> {new}")
    if not changed:
        log(f"[outputs] remap: no saved path starts with {', '.join(old for old, _ in remaps)}")
    return LensClusterSolverConfig.from_json_dict(payload)


def apply_overrides(config, overrides):
    """``config`` with the ``section.field=value`` assignments ``overrides`` (the syntax of ``lumina --set``,
    ``run_files.read_run``), each of which must name a field of the ``outputs`` section
    (``config.REPORT_ONLY_SECTIONS``): ``outputs.FIELD`` or ``outputs.MODULE.FIELD``.  Everything else describes the
    saved fit or where its products go (``paths``, which the run hash leaves out but which names the run folder and the
    figure prefix).

    Raises
    ------
    ValueError
        For a malformed assignment, an unknown field, a setting outside ``outputs``, or a value the configuration
        refuses (``config.validate_outputs``).
    """
    from .. import config as _config
    from ..run_files import _dotted, _merge, parse_value
    overrides = tuple(overrides)
    if not overrides:
        return config
    overlay = _dotted(overrides)
    refused = sorted(section for section in overlay if section not in _config.REPORT_ONLY_SECTIONS)
    if refused:
        allowed = ", ".join(f"{section}.*" for section in _config.REPORT_ONLY_SECTIONS)
        raise ValueError(f"--set may change only the outputs section ({allowed}); refused: {', '.join(refused)}")
    try:
        changed = _config.LensClusterSolverConfig.from_json_dict(_merge(config.to_json_dict(), _config._jsonable(parse_value(overlay))))
    except (TypeError, AttributeError) as error:
        raise ValueError(f"--set: {error}") from None
    try:
        _config.validate_outputs(changed)
    except ValueError as error:
        raise ValueError(f"--set: {error}") from None
    return changed


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
    (``(OLD, NEW)`` pairs, :func:`remap_paths`) relocates them, ``overrides`` (``outputs.field=value``,
    :func:`apply_overrides`) changes the outputs settings, and ``transform`` (a function of the configuration) is applied
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
