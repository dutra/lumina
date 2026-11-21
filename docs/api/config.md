# Configuration

```{eval-rst}
.. automodule:: lumina.config
   :no-members:

Run files
---------

.. autofunction:: lumina.run_files.load_run
.. autofunction:: lumina.run_files.load_priors
.. autofunction:: lumina.run_files.read_run
.. autofunction:: lumina.run_files.parse_value
.. autofunction:: lumina.cli.main

Top level
---------

.. autoclass:: lumina.config.LensClusterSolverConfig
.. autoclass:: lumina.config.RunPathsConfig
.. autoclass:: lumina.config.RuntimeConfig

Lens model
----------

.. autoclass:: lumina.config.LensModelConfig
.. autoclass:: lumina.config.ReferenceFrameConfig
.. autoclass:: lumina.config.CosmologyConfig
.. autoclass:: lumina.config.DPIEHaloConfig
.. autoclass:: lumina.config.NFWHaloConfig
.. autoclass:: lumina.config.ShearHaloConfig
.. autoclass:: lumina.config.MassSheetConfig
.. autoclass:: lumina.config.MemberPopulationConfig
.. autoclass:: lumina.config.IndependentMemberHaloConfig
.. autoclass:: lumina.config.ImageConstraintsConfig

Members and scaling relation
----------------------------

.. autoclass:: lumina.config.MemberConfig
.. autoclass:: lumina.config.ScalingModelConfig

Likelihood, search and sampling
-------------------------------

.. autoclass:: lumina.config.LikelihoodConfig
.. autoclass:: lumina.config.SearchConfig
.. autoclass:: lumina.config.SamplingConfig

The outputs stage
-----------------

.. autoclass:: lumina.config.OutputsConfig
.. autoclass:: lumina.config.ImageSolverConfig
.. autoclass:: lumina.config.FitOutputConfig
.. autoclass:: lumina.config.ImagesOutputConfig
.. autoclass:: lumina.config.MassOutputConfig
.. autoclass:: lumina.config.TruthOutputConfig
.. autoclass:: lumina.config.RGBOutputConfig
.. autoclass:: lumina.config.RGBDisplayConfig
```

Priors are `numpyro.distributions` objects; see {doc}`priors`. A configuration can also be read from a Lenstool parameter
file ({doc}`lenstool`).
