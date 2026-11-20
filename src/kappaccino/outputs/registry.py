"""The output modules of the stage, in their default order.  A new module is added here (see ``docs/outputs.md``)."""
from __future__ import annotations

from .base import OutputModule
from .fit import FitModule
from .images import ImagesModule
from .mass import MassModule
from .rgb import RGBModule
from .truth import TruthModule

MODULES: tuple[OutputModule, ...] = (FitModule(), ImagesModule(), MassModule(), TruthModule(), RGBModule())


def module_names() -> tuple[str, ...]:
    return tuple(module.name for module in MODULES)


def module_named(name: str) -> OutputModule:
    """The registered module called ``name``.

    Raises
    ------
    KeyError
        If no module has that name; the message lists the known ones.
    """
    for module in MODULES:
        if module.name == name:
            return module
    raise KeyError(f"no output module {name!r}; known: {', '.join(module_names())}")


def check_module_names(label: str, names) -> None:
    """Check that ``names`` (the value of the setting ``label``) is a list of registered module names.

    Raises
    ------
    ValueError
        If ``names`` is not a list or tuple of strings or names an unknown module.
    """
    if isinstance(names, str) or not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{label} must be a list of output module names, e.g. [truth, rgb]; got {names!r}")
    unknown = [n for n in names if n not in module_names()]
    if unknown:
        raise ValueError(f"{label} names unknown output modules {unknown}; known: {', '.join(module_names())}")
