"""The outputs stage: the tables and figures written after the fit, as modules that can be switched on and off.

:mod:`kappaccino.outputs.base` holds the context (:class:`OutputContext`), the module base class
(:class:`OutputModule`: compute to the store, product, render from the store) and the runner (:func:`run_modules`);
:mod:`kappaccino.outputs.registry` lists the modules in their default order.  ``reporting.outputs.write_outputs`` runs the stage after a fit, and ``kappaccino outputs RUN_DIR``
(:mod:`kappaccino.outputs.command`) reruns it on a finished run.  The ``outputs`` section of the configuration
(``config.OutputsConfig``) holds the settings of the stage: the reporting draws, and one section per module with its
``enabled`` switch and its own settings.
"""
from .base import (MODES, ComputeOnly, ModuleUnavailable, OutputContext, OutputModule, disabled_modules, order_modules,
                   plan_modules, run_modules)
from .registry import MODULES, check_module_names, module_named, module_names

__all__ = ["MODES", "MODULES", "ComputeOnly", "ModuleUnavailable", "OutputContext", "OutputModule", "check_module_names",
           "disabled_modules", "module_named", "module_names", "order_modules", "plan_modules", "run_modules"]
