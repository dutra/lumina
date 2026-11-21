# Model, posterior and analysis

The objects a run returns ({class}`~lumina.pipeline.RunResult`) and the helpers that evaluate a fitted model.
{doc}`../posterior` defines every posterior parameter and its units.

```{eval-rst}
Model state and parameters
--------------------------

.. autofunction:: lumina.core.state.build_state
.. autofunction:: lumina.core.state.with_member_classes
.. autoclass:: lumina.core.model.ModelState
   :no-members:
.. autoclass:: lumina.core.model.ParameterSpec
   :members: latent_prior, log
.. autoclass:: lumina.core.model.FamilyData
   :no-members:
.. autoclass:: lumina.core.model.ImageData
   :no-members:
.. autoclass:: lumina.core.model.MemberModel
   :no-members:

Latent and physical values
--------------------------

.. autofunction:: lumina.core.model.latent_to_physical
.. autofunction:: lumina.core.model.physical_to_latent
.. autofunction:: lumina.core.model.latent_array_to_physical
.. autofunction:: lumina.core.model.convert_theta_to_physical
.. autofunction:: lumina.core.model.convert_theta_to_latent
.. autofunction:: lumina.core.model.convert_sample_matrix_to_physical
.. autofunction:: lumina.core.model.default_theta

Posterior
---------

.. autoclass:: lumina.core.model.PosteriorResults
   :no-members:

Evaluating the model
--------------------

.. autoclass:: lumina.likelihood.evaluator.Evaluator
   :members: physical, packed_state, packed_state_for_z, component_dispersions, fields, total_fields_at, statistics,
             error_scales, log_likelihood, search_log_likelihood, reported_physical, latent_from_reported_physical,
             source_positions, gp_deflection_correction

Lens maps and truth comparison
------------------------------

.. automodule:: lumina.diagnostics.truth_recovery
   :no-members:

.. autofunction:: lumina.diagnostics.truth_recovery.truth_maps_on_grid
.. autofunction:: lumina.diagnostics.truth_recovery.load_truth_fits
.. autofunction:: lumina.diagnostics.truth_recovery.diagnostic_grid
.. autofunction:: lumina.diagnostics.truth_recovery.sample_on_grid
.. autofunction:: lumina.diagnostics.truth_recovery.grid_arcsec
.. autofunction:: lumina.diagnostics.truth_recovery.compute_truth_recovery

Saved artifacts
---------------

.. automodule:: lumina.io.artifacts
   :no-members:

.. autofunction:: lumina.io.artifacts.load_config
.. autofunction:: lumina.io.artifacts.load_posterior
.. autofunction:: lumina.io.artifacts.load_start
.. autofunction:: lumina.io.artifacts.load_search
.. autofunction:: lumina.io.artifacts.load_exact_positions
.. autofunction:: lumina.io.artifacts.save_config
.. autofunction:: lumina.io.artifacts.save_state
.. autofunction:: lumina.io.artifacts.save_search
.. autofunction:: lumina.io.artifacts.save_start
.. autofunction:: lumina.io.artifacts.save_posterior
.. autofunction:: lumina.io.artifacts.spec_to_json
.. autofunction:: lumina.io.artifacts.init_signature
.. autofunction:: lumina.io.artifacts.write_json
.. autofunction:: lumina.io.hdf5.load_arrays
.. autofunction:: lumina.io.hdf5.save_arrays
```
