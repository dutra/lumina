# API reference

Every class and function a user calls, with its parameters, units and return values. The configuration classes
describe a fit; the pipeline runs it; the priors, catalog readers and the Lenstool reader build its inputs; and the
model, posterior and analysis objects are what a finished run returns. The remaining modules (the search and sampler,
the far-field series, the likelihood kernels and the plotting modules) are internal.

The package is organized as:

| module or subpackage | contents |
|---|---|
| `lumina.config`, `lumina.priors` | the configuration of a run and its priors |
| `lumina.pipeline`, `lumina.cpu_runtime` | running a fit: the two stages, and the CPU devices of the chains |
| `lumina.core` | the model: parameter specifications, the model state, inputs and member classification |
| `lumina.lens` | lens physics in JAX: profiles, deflections and Jacobians, cosmology, the far-field series |
| `lumina.likelihood` | the evaluator and the position likelihoods (smoothed, image-plane GP, GRF) |
| `lumina.inference` | the multi-start MAP search, NUTS and model comparison (BIC, WBIC) |
| `lumina.io` | catalogs, Lenstool parameter files, sky coordinates, run artifacts and HDF5 arrays |
| `lumina.imaging` | HST and JWST images: cutouts, RGB display, photometry |
| `lumina.diagnostics` | numerical diagnostics: image recovery, exact images, members, priors, truth recovery |
| `lumina.plotting` | figures and the shared matplotlib style |
| `lumina.reporting` | tables, text reports and publication macros written after a run |
| `lumina.partition` | uncertain images: the sum over their partitions into groups |
| `lumina.mock_validation` | analytic mock clusters, their fits and recovery scores |

```{toctree}
:maxdepth: 1

config
priors
profiles
pipeline
model_comparison
results
catalogs
lenstool
mock_validation
runtime
```
