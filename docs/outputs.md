# Outputs

This page lists the files a run writes; {doc}`posterior` defines each posterior parameter and derived quantity.

A run writes everything under its run root, `<output_dir>/<run_name>/` from
{class}`~kappaccino.config.RunPathsConfig`. Figure names start with `RunPathsConfig.figure_prefix`, for example
`m0416_critical_curves.pdf`.

## Directory layout

`init/artifacts/`
: `config.json`, `init_signature.json`, `state.json`, `classification.json`, `search.h5` and `modes.json`,
  `start.json`, `summary.json`, `model_comparison.json` (BIC, see {doc}`model-comparison`).

`init/results.h5`
: The record `shallow_fold_pairs` (GP likelihoods) and, with member kinematics, the table `member_kinematics`.

`init/tables/`
: `classification_report.txt` and `start_parameters.txt` (plain text).

`final/artifacts/`
: `config.json`, `posterior.h5`, `audit.json`, `summary.json`, and with uncertain images `uncertain_images.h5` (see
  {doc}`likelihoods`), `model_comparison.json` (BIC and, with `SamplingConfig.wbic`, WBIC) and with WBIC
  `wbic_draws.h5` (see {doc}`model-comparison`).

Arrays are saved as HDF5 (`.h5`, one dataset per array; read them with `kappaccino.io.hdf5.load_arrays` or `h5py`) and
everything else as JSON. `config.json` restores the run configuration exactly
(`kappaccino.io.artifacts.load_config`). Nothing is pickled.

### Results and cache files

Each stage folder (`init/`, `final/`) keeps its products in two HDF5 files instead of loose tables:

`results.h5`
: Every table (formerly `tables/<name>.csv`), every record (formerly `tables/<name>.json`), every array bundle (formerly
  `tables/<name>.h5`) and the record of each figure (formerly `figures/<figure>.json`, now stored under the figure's
  name without `.pdf`). Below, a table, record or arrays `<name>` means that entry of `results.h5`.

`cache.h5`
: What only saves time: the lens maps of every grid the products evaluate (`grid_maps_<key>`, float32 maps of the
  reporting draws, and `lens_reference_<key>`, the parametric Jacobians of the overview), and the zeta masses of the
  reporting draws (`zeta_masses_cache`). Deleting it never loses or changes a result; the next output pass recomputes
  what it needs.

The figures (PDF), the FITS maps, the plain-text reports (`run_summary.txt`, `audit_report.txt`,
`classification_report.txt`, `start_parameters.txt`), the `<prefix>_results.tex` macros, the logs, `artifacts/` and
`inputs/` stay separate files. A table reads back exactly as it was written (column order, dtypes, strings, NaN and
None). The dtypes are the ones the table had in memory, not what parsing a CSV gives: a column of numeric-looking ids
such as `family_id` written as strings reads back as strings. `results.h5` is the only source: loose files under
`tables/` or `figures/`, as runs written before this layout have, are not read, so such a run fails with the missing
entry.

`results.h5` is never changed in place. A write builds the new version in a temporary file next to it
(`results.h5.tmp-*`), fsyncs it and renames it over the old one, so a crash at any moment leaves either the old or the
new file, complete. Writers in different processes take turns through the lock file `results.h5.lock` (a writer waits
up to 10 minutes, then fails with "busy"); readers take no lock and always see a complete version. A file that is not
a valid HDF5 file is moved to `results.h5.unreadable-<time>` by the next writer, which logs the entries lost; any
other failure to open or write it stops with an error and leaves the file alone. A write first checks that the disk
(and `ulimit -f`) leaves room for the copy and the new entry, and stops with an error otherwise; if the disk fills up
after that check, the write still fails with an error and `results.h5` keeps its old version, but the HDF5 library may
crash the process later. Rebuilding the file to reclaim the space of replaced entries is given up (logged) rather than
drop an entry that fails to copy. A symlinked `results.h5` is followed: the file it points to is replaced.

Reading a product that was never written raises `store.MissingEntry` (a `FileNotFoundError`). Reading one that is
there but cannot be decoded raises `store.DamagedEntry`, which is not a `FileNotFoundError`, so a step that carries on
without a missing product stops instead of writing a product without it; out-of-memory and I/O errors are raised as
they are. The only way an entry is ever dropped is writing again a name whose stored entry is damaged and cannot be
replaced in place: the writer checks it with a second read and logs an ERROR that names it as LOST. `cache.h5` is
written in place, and any failure to read or write it only costs a recomputation.

List, print or export the entries of a run or stage from the repository root:

```bash
python -m kappaccino.io.store results/<cluster>/<run>                       # every entry with its shape
python -m kappaccino.io.store results/<cluster>/<run> --table image_fit_quality > image_fit_quality.csv
python -m kappaccino.io.store results/<cluster>/<run> --record run_summary
python -m kappaccino.io.store results/<cluster>/<run> --figure-record <prefix>_corner_globals
python -m kappaccino.io.store results/<cluster>/<run> --arrays conditional_field_maps
python -m kappaccino.io.store results/<cluster>/<run> --stage init --record shallow_fold_pairs
python -m kappaccino.io.store results/<cluster>/<run> --export /tmp/<run>_loose  # loose csv/json/h5 files
```

`--cache` makes `--table`, `--record` and `--arrays` read `cache.h5`. The export writes `<stage>/tables/<name>.csv`,
`.json` and `.h5`, `<stage>/figures/<figure>.json` and `<stage>/cache/<name>.*`. An entry that cannot be read is listed as `DAMAGED` (or reported as not exported), the rest is listed or
exported, and the command exits with status 1.

Every reported product (figures, tables and results macros) is computed over one set of reporting draws:
`outputs.draws` random posterior draws over all chains, chosen with `runtime.seed` (64 in a full run, 8 in the
fast profile). Each draw is evaluated once and shared by every product. In runs with a shared GRF, each draw carries one
conditional realization of the field: an unconditional draw of the field corrected to agree with the images (Matheron's
rule), with a seed fixed by `runtime.seed` and the draw's posterior row. The median over the draws is close to the mean
field, and the 16–84% range holds the uncertainty of the lens parameters and of the field together. Maps are
per-pixel medians over the draws with their 16th and 84th percentiles, curves are zero contours of median maps with
16–84% bands, and numbers are medians with 16–84% ranges. No product uses the single best sample.

The image solver follows the same rule: where a draw's lens puts the images, their residuals from the catalog, the RMS
and recovered counts, the images the lens adds to a family, and the magnifications at the images are all solved in the
draw's own lens, parametric plus its realization. The RMS of a draw is therefore that of one posterior sample of the
mass distribution, and its range over the draws includes the uncertainty of the field.

This holds for runs with `gp_blocks: all`. A run with `gp_blocks: system` (the fast profile) gave each system its own
field, and the products rebuild one shared field from its draws. That field is not a posterior sample of the lens, so
the residuals, RMS, position contours and map uncertainties of such a run are not trustworthy and likely inflated (on
an M0416 fast run the RMS came out twice what the fitted image-plane scatter allows). The output pass logs a warning
and writes it to the record `run_summary` under `warnings`.

The lens position of an image is not where its catalog entry is expected, because the likelihood lets the two differ by
the image-plane GP and the measurement error. Each draw's lens position therefore also has a predicted observed
position (`x_predicted_arcsec`, `y_predicted_arcsec`): the lens position plus one draw of that scatter, taken from
its prior with the draw's own `sigma_gp`, `ell_gp` and `sigma_meas` and not conditioned on the catalog. The position
contours of the observed images are the 68.3% and 95.4% regions of these positions, so about 68% of the catalog
positions should fall inside their 68.3% contours. The table `position_coverage` has that check per image (`credible_level`,
`inside_68`, `inside_95`), and the results macros `PositionCoverageOneSigmaPercent` and
`PositionCoverageTwoSigmaPercent` the fractions. The contours of extra images, which have no catalog position, are of
their true positions. The tables also hold a true position per draw (`x_true_arcsec`,
`y_true_arcsec`): the lens position plus the GP displacement drawn given the catalog positions. For a catalog image it
is the catalog position without its measurement error, so its spread is at most `sigma_meas`; where the data do not
constrain `sigma_meas` (no images closer than `ell_gp`) that spread is the prior's.

`final/results.h5`
: Tables `posterior_summary`, `member_classification`, `member_scaling_relation`, `image_fit_quality`,
  `image_count_recovery`, `model_magnification`, `posterior_image_residual_draws`, `posterior_image_rms`,
  `posterior_extra_images`, `image_recovery_extra_images`, `position_coverage`, `chain_convergence` and those of the
  figures below; records `run_summary`, `runtime`, `audit`, the provenance records of the figures and the
  completeness record `output_module_<name>` of each output module (see [Output modules](#output-modules));
  arrays such as `conditional_field_maps` and `member_diagnostic_draws`, and the arrays a figure is drawn from, named
  by the figure without its prefix (`critical_curves`, `subhalo_mass_concentration`, `magnification_recovery`,
  `truth_recovery_residuals`); for simulated clusters the tables `truth_recovery_m2d_aperture_profile` and the like
  and the record `truth_recovery`. With uncertain membership (`MemberConfig.membership`), the table
  `member_membership`: one row per member with `treatment` (`certain`, `enumerated` or `marginalized`), `p_catalog`,
  `p_prior` (after any calibration shift, averaged over the draws), and `p_posterior` with `p_posterior_std` over the
  draws. A marginalized member's posterior equals its prior. This table is written even with `runtime.skip_plots`.
  With uncertain images, the tables `uncertain_images` and `uncertain_pairs`.

`final/cache.h5`
: `grid_maps_<key>` and `lens_reference_<key>` (the lens maps of each grid, keyed by the reporting draws and the
  model they map to, the redshift, the grid and the code version), `zeta_masses_cache`.

### A second output pass

An output pass over a stage that already holds products (a resumed run, or a pass restarted after a crash) reuses
what does not change with the same reporting draws, and logs each reuse or recomputation with its reason.  Every reuse
is keyed by the draws (`ReportingDraws.key`): the seed, the posterior rows, the latent and physical parameter vectors,
and a digest of the model that turns them into a lens (`reporting.draws.lens_digest`: the parameter names, the packed
lens spec with every component's fixed values and every member's position, luminosity ratio and class, the membership
priors, the cosmology, the image catalog, the GP settings and the field's code, `plotting.grf.PLOT_FIELD_VERSION` and
`N_MODES`).  An edited member catalog, priors file or member classification therefore recomputes the products.  The
digest does not see code changes elsewhere (the lens kernels, the image solver); those need their version constants
bumped.

- the image-recovery tables, read back from `results.h5` when the record `image_recovery_fingerprint` saved with them
  matches: the draws (seed, indices, parameter values), every `outputs.image_solver` setting, `outputs.images.extra_images`,
  the image catalog and the solver version (`IMAGE_RECOVERY_VERSION` in `diagnostics/image_recovery.py`, bumped
  whenever the solver can give other numbers). Any difference, or a missing table, recomputes them;
- the lens maps of every grid, from `cache.h5`; at most `reporting.draws.GRID_MEMORY_BYTES` (1 GiB) of maps stay in
  memory during a pass, and a grid asked for again later is read back from the cache;
- the member aperture masses of Hera and Ares (`member_aperture_masses`, written in one commit with a fingerprint in
  the record `aperture_mass_histogram`) and the zeta masses (`zeta_masses_cache`).

Everything else (figures, fold diagnostics, the GRF critical band, contours) is recomputed from these.

`final/tables/`
: `run_summary.txt` and `audit_report.txt` (plain text).

`final/figures/`
: The figures listed below and `<prefix>_results.tex`. Their records are in `final/results.h5`.

`final/fits/`
: Convergence, shear, $\det A$ and magnification grids on the truth WCS: the per-pixel median and 16th and 84th
  percentiles over the reporting draws, each draw's GRF conditioned at that draw. Also the overview convergence maps,
  per-pixel medians over the same draws.

`init/logs/init.log`, `final/logs/final.log`
: The stage logs.

## Output modules

The outputs stage after the fit is five modules ({mod}`kappaccino.outputs`), run in this order. Each module works in
three phases, each a sequence of named steps:

- **compute** evaluates what the module needs (the lens of the reporting draws, the image solver) and writes every
  result to the stage's store: tables, records and arrays in `results.h5`, per-draw grid maps in `cache.h5`. It draws
  no figure.
- **product** loads from the store what the modules that require this one read (`ctx.products[name]`).
- **render** draws the figures from the store alone: the saved tables, records and arrays, the configuration, the
  catalogs (`ctx.state`) and the posterior. It never touches the evaluator or the reporting draws.

A full pass runs, per module, compute, product and render. Render reads back from the store even then, so the figures
of a full pass and of a later render-only pass come from the same code and the same numbers.

| module | compute steps | render steps | requires | runs when |
|---|---|---|---|---|
| `fit` | `tables` (`posterior_summary`, `member_classification`), `chain_traces` (`chain_convergence`, record `chain_traces`), `audit` (the record), `scaling_table` (`member_scaling_relation`), `covariance` (`covariance_errors_both_planes`), `fields` (`gp_diagnostics`, `conditional_field_maps`), `members` (`member_diagnostics` and its companions) | `chain_traces`, `potfiles`, `block_corner` (`corner_combined.pdf`), `cosmology_corner`, `covariance` (covariance corner and errors), `fields` (GP and GRF plots), `members` (the member plots), `scaling_plot`, `globals_corner` | | always |
| `images` | `recovery` (the image-recovery tables, record `image_recovery_fingerprint`, `position_coverage`, `fit` in `run_summary`), `true_lens` (with `outputs.truth.deflection_true_fits` and observed images: `true_lens_residuals`, record `true_lens_rms`; a map that cannot be read or fitted to the catalog fails the step) | `residual_histogram`, `recovery_figure` (both with observed images) | | always |
| `mass` | `critical_curves` (arrays `critical_curves`), `aperture_masses` (`member_aperture_masses`, record `aperture_mass_histogram`), `zeta_masses` (`member_zeta_masses`, record `zeta_masses`, arrays `subhalo_mass_concentration`) | `critical_curves`, `zeta_masses` (the figure and the table `zeta_mass_function`) | `images` | always; the masses with a truth convergence map |
| `truth` | `relative_displacement` (with `outputs.truth.deflection_true_fits`, the true-lens record and a shared field: table and record `relative_displacement`), `truth_maps` (FITS, profile tables, record `truth_recovery`, arrays `truth_recovery_residuals`, the published ratios; with `outputs.truth.kappa_true_fits`), `magnification_recovery`, `cumulative_source_area`, `magnification_scatter` | `relative_displacement`, `magnification_recovery`, `cumulative_source_area`, `truth_residuals`, `mass_profile`, `kappa_mass_residuals` | `images` | with `outputs.truth.kappa_true_fits` or `outputs.truth.deflection_true_fits` |
| `rgb` | per imaging variant `imaging`, `covariance_cutouts`, `rgb_critical`; for the overview variants `cluster_mass` and, once, `fold_data`; for the JWST overview also `position_contours` and `small_lambda` (before the mass figure) and `paper_cutouts` (see [The rgb module](#the-rgb-module)) | per imaging variant `imaging`, `covariance_cutouts`, `rgb_critical`; for the overview variants `cluster_mass` and `fold_cutouts`; for the JWST overview also `small_lambda`, `apj_fold_cutouts` and `paper_cutouts` | `images` | with `outputs.rgb.image_dir` and observed images |

The product of `images` is read from the store, so `mass`, `truth` and `rgb` get the same tables in every pass mode:

- `image_df`: the table `image_fit_quality` (empty when no image was observed; `rgb` is then inactive and `images`
  draws no figure);
- `recovery`: every table that the record `image_recovery_fingerprint` lists (`image_fit_quality`,
  `image_count_recovery`, `model_magnification`, `image_recovery_extra_images`, `posterior_extra_images`,
  `posterior_image_rms`, `posterior_image_residual_draws`, and with a mass field the two `_lens_only` tables), plus
  `position_coverage`, by table name;
- `true_lens`: the record `true_lens_rms` when the last compute's `true_lens` step wrote it, else None. A record left
  by an older compute is never returned.

The tables come back as they were written, with their column order and dtypes, and with a fresh `RangeIndex`.

`mass` and `rgb` also run after `fit` without needing its product (it has none), because their computes read tables
the compute of `fit` writes: `member_diagnostics` (aperture and zeta masses) and `covariance_errors_both_planes`
(covariance cutouts, which save what they draw of it in their own table). The steps of `rgb` are named per variant, for example `fold_cutouts:jwst`. `kappaccino outputs --list`
prints the modules with these relations and their compute and render steps.

### The `fit` module

The compute of `fit` writes every table and record of the sampled parameters; its render draws every figure from
those entries and the posterior. Its compute steps and what they save:

- `tables`: `posterior_summary` and `member_classification`;
- `chain_traces`: `chain_convergence` (split R-hat and ESS of every parameter) and the record `chain_traces`; nothing
  when the posterior has no recorded chains;
- `audit`: the record `audit`, when the fit has one;
- `scaling_table`: `member_scaling_relation`;
- `covariance`: the table and record `covariance_errors_both_planes`, over the reporting draws: per image and
  direction, the median singular value, the median width of each component and the 16th, 50th and 84th percentiles
  of each component and of the total;
- `fields` (GP runs only): the arrays and record `gp_diagnostics` and, with a shared knee-spectrum GRF, the arrays and
  record `conditional_field_maps`. The record `gp_diagnostics` says whether the kernel is the knee kernel (`knee`) and
  whether this compute wrote the field maps (`field_maps`); its arrays hold what the GP figures took from the
  evaluator: for the knee kernel the separation grid `r` and, with a sampled knee wavelength, the `longitudinal` and
  `transverse` correlations of the mass field at the median knee wavelength and slope; otherwise the family `slots`
  and `mask`;
- `members` (runs with population members): the tables `member_diagnostics`, `diagnostic_image_index` and
  `image_field_diagnostics`, the arrays `member_diagnostic_draws` (the fields of each reporting draw and the deviation
  draws `deviations` and `prior_deviations`) and the record `member_diagnostic_provenance`.

The render steps `chain_traces`, `potfiles`, `block_corner`, `cosmology_corner` and `globals_corner` and the covariance
corner read only the posterior (`chain_traces` recomputes its diagnostics from the posterior chains, as the compute
did). The covariance errors are drawn from their table, the GP figures from `gp_diagnostics` and the posterior, the
field figures from `conditional_field_maps` (only when `gp_diagnostics` says this compute wrote them), the member
figures from the member entries above and the scaling relation from `member_scaling_relation` and the posterior. A
render pass of `fit` on Hera took 34 s; the full pass that computed it took 63 s (see [Pass modes](#pass-modes)).

### The rgb module

`rgb` loops over the imaging variants (`imaging.cutouts.imaging_variants`) in both phases. A variant is named by its
`figure_suffix` without the underscore, `primary` without one. Compute evaluates the lens and saves what each figure
draws; render draws every figure from `results.h5` and the imaging pixels, never from `cache.h5`. Render reads the
saved median det A maps with their percentiles and the saved position contours. Two things it still derives from
saved per-draw values, with the same functions compute uses: the KDE regions of the per-draw positions in the
small-lambda, fold and paper cutouts (their density grids would take 3.5 MB per panel), and the percentiles of the
per-draw critical-curve tracks of the fold figures (8 to 64 curves of 97 points each).

Compute needs the imaging FITS files, but only their headers and paths: the reference band's WCS, shape and pixel
scale set the overview grid, the cutout fields and the imaging footprint, and the band paths go into the record of the
covariance cutouts. Loading a variant also reads a subsample of each band for its sky level, which only the display
uses. The command checks that `outputs.rgb.image_dir` exists before a full or compute pass, so on a machine without
the imaging run `--compute --skip rgb`. In a compute, a variant whose band files are missing (the extra JWST variant of
M0416 and AS1063, for example) is logged once and left out, its `imaging:<variant>` step records that it wrote
nothing, and the other variants go on; the render leaves that variant out too. In a render, a variant the compute drew
needs its imaging again: when it cannot be loaded, the render step `imaging:<variant>` fails with a message that names
the folder, the variant's figures are not drawn, the other variants go on, `rgb` ends as failed and the command exits
with 1. For the configured imaging the message names `outputs.rgb.image_dir` and suggests `--remap`; for the extra
JWST variant, whose folder is no configuration field, it says where the relative folder resolves and to run the
command from the repository root. Loading the imaging does not count as drawing a figure (`input_steps` of the
module); a render pass that drew nothing because the imaging is missing says so and names the folder. One failed step
of a variant leaves the figures of the other steps and variants alone.

| step | compute saves | render reads |
|---|---|---|
| `covariance_cutouts:<variant>` | table and record `small_lambda_covariance_cutouts<suffix>`, arrays `..._contours` and `..._ellipses` (the selected images' indices, directions and widths of the linear ellipse), table `..._intervals` (the posterior 16th and 84th percentiles along the stretch of the selected images, from the table `covariance_errors_both_planes` when the last compute of `fit` wrote it, else empty: the figure then has no interval lines, which the log and the step's `notes` say) | those |
| `rgb_critical:<variant>` | arrays `<stem>` and record `<stem>_band` (`hst_rgb_critical_z9` or `jwst_rgb_critical_z9`), record `<stem>_model` (whether the run has a shared GRF and its label), for M0416 the record `<stem>_candidate_catalog` (the path and the candidates of `outputs.rgb.candidate_catalog_path`, default the packaged `m0416_candidates.json`, that the `_candidates` overview draws) and the table `<stem>_marker_photometry` (the F200W magnitudes of the packaged `m0416_image_f200w_photometry.csv` matched to the catalog images by label and position, which size the markers) with the record of the same name (its `path`), the position contours | those; writes the figure records and the marker tables |
| `position_contours:<variant>` | arrays and record `posterior_overview_positions`, new arrays and record `posterior_overview_contours` (the 68.3% and 95.4% curves, `curve_<c>_segment_<k>`) | (no figure) |
| `small_lambda:<variant>` | table and arrays `<prefix>_small_lambda_a_image_posteriors_jwst` (the ranking, with the panels shown and their fields), arrays and record `..._critical`, new record `..._selection` (the figure's settings) | those, `posterior_overview_positions`; writes the record `<prefix>_small_lambda_a_image_posteriors_jwst` |
| `cluster_mass:<variant>` | tables `image_catalog_arc_directions<suffix>`, `..._extra_arc_directions`, `image_catalog_mass_profile<suffix>` and `..._y`, the covariance ellipses table, record `image_catalog_mass_provenance<suffix>`, the FITS, new arrays `image_catalog_family_cluster_mass<suffix>` (the median maps `model`, `model_grf`, `grf`, the resampled `truth` and the covariance curves) | those and, for the JWST overview, the position contours |
| `fold_data` | arrays and record `fold_pairs_source_caustics` | `fold_cutouts`, `apj_fold_cutouts` read them |
| `paper_cutouts:<variant>` | arrays and record `<prefix>_rgb_inset_critical_<figure>` (`<figure>` is `small_lambda_posteriors` or `fold_cutouts`) when a selected image panel of that figure is not in the small-lambda figure of this compute; record `<prefix>_paper_cutouts` (`figures`: per figure prepared, its selectors and the entry and rows that hold the critical fields of its image panels) | that record, the entries it names, the small-lambda ranking, the positions and the fold data |

The image panels of a paper-cutout figure take their critical fields from the small-lambda figure's
`..._critical` only when the small-lambda step of the same compute wrote it and it holds every image panel; otherwise
the figure saves its own inset entry. The render draws only the figures the record names, from the entry and rows it
names, and fails the step when those rows do not hold the figure's panels (same centres, fields and redshifts) or a
recorded figure cannot be drawn.

### Pass modes

`full`
: compute, product and render; the default, and what the run command does after the fit.

`compute`
: compute and product, no figures (`kappaccino outputs RUN --compute`), for example `--compute --skip rgb` on a
  machine without the imaging.

`render`
: product and render only (`kappaccino outputs RUN --render`): every figure drawn again from
  what the last compute saved. The run is loaded without its evaluator (`saved_run.load_saved_run`: configuration,
  state and posterior; on Hera 3 s, of which 0.8 s go to about 40 one-operation JAX kernels the state's arrays need,
  and no model function is compiled), the reporting draws are never built, and the files the computes read (truth
  maps, imaging) are not checked. Measured on 2026-11-20: on Hera (`results/hera/w1000s500t8_allblocks_fast-af374a`,
  8 draws) the render pass of all five modules took 43 s of wall time, including loading the run (`fit` 34 s, `images`
  0.2 s, `mass` 0.4 s, `truth` 1.3 s, `rgb` 4 s), against 232 s for the full pass (`fit` 63 s, `mass` 68 s, `truth`
  56 s, `rgb` 20 s); on M0416 (`results/m0416_canucs2025/w1000s500t8_allblocks_fast-59dfa1`) `--render --only rgb`
  took 54 s (`rgb` 50 s, most of it building the RGB cutouts from the JWST mosaics) against 240 s for
  `--only rgb --draws 16` (`rgb` 199 s). A render pass in which no selected module draws anything says why in the
  log and on the command line: no selected module was computed in this run folder, an input a figure needs is missing
  (the imaging of `rgb`, named with its folder), the render steps could not be drawn, or the last compute saved
  nothing to draw; the command then exits with 1. So does any render pass in which a module failed, for example
  `rgb` when the imaging of a variant its compute drew cannot be loaded.

In a render pass (and in the product and render phases of any pass) `ctx.draws` and `ctx.evaluator` raise
`ComputeOnly`, which names the module and step that asked, so a figure that still needs the lens fails its step
instead of being computed again in secret. The guard covers these two only: nothing stops a render step from
evaluating a lens it builds itself (from `ctx.state`, for example), so a render step must read everything it draws
from the store.

### The completeness record

When a module's compute starts, its record `output_module_<name>` in `results.h5` is set to
`{"complete": false, "started_at": ...}`; when the compute ends it holds the compute steps (`status`, `seconds`,
`wrote`, `notes` when the step noted something it went without, and `error` or `reason`), `complete` (true only when
every compute step ran without failing and the module gave its product; for information), `unavailable` (the error
with which the compute ended without a product, `ModuleUnavailable` or another error, else null), `computed_at`, and
when the compute used the reporting draws `reporting_draws`, `image_draws`, `draws_seed` and `draws_key`.

Completeness is per step. A render step is drawn when the compute step it needs finished and wrote in the module's
last compute, whatever the other steps did; its figure never comes from an older compute. A render pass refuses a
module only when there is no record of a compute with its steps: never computed in this run folder, or interrupted
before the compute recorded them (the record is still the marker). The module then fails with `ModuleUnavailable`
and a message that says to run its compute first (`kappaccino outputs RUN --only <module>`, or `--compute --only
<module>`). Otherwise the render pass draws what it can and reports each render step it could not draw, with the
reason; the module ends as failed when a compute step of its last compute failed or a render step could not be drawn,
and as done otherwise. A module whose last compute gave no product (`unavailable`) draws what it can and is failed
with no product, so the modules that require it are skipped. A module pulled into a render pass by `--only` (because
a selected module requires it) gives only its product.

A compute step is skipped (`ctx.skip_step`) only when it cannot run because another step failed; the render steps
that need it are then skipped too, as not drawn. A step that does not apply to the run (for example the magnification
figures without shear truth) runs and returns None: it wrote nothing and is recorded with `wrote` false. Render steps
are declared against the compute steps they draw (`ctx.render_step`): a render step whose compute step failed, was
skipped or did not run is skipped with the reason (not drawn), and one whose compute step wrote nothing is left out. A
render step whose name is not in `render_steps`, or that needs a step not in `compute_steps`, is an error that fails
the module.

### Steps, failures and the run summary

A step that raises is logged with its traceback and recorded, and the module goes on with its other steps; a step
that needs the result of a failed one is skipped with the reason (no step of `images` runs after a failed `recovery`,
no `truth` step after a failed `truth_maps`, no fold step after a failed `fold_data`, no figure whose compute step
failed). A module with a failed step ends as `failed` but keeps its product, and a full pass still draws the figures
whose compute steps succeeded. A module that cannot give its dependents what they need raises after its steps
(`images` without its recovery, `truth` without its maps) and is failed with no product; its render still runs, in a
full pass and in a later render pass, and draws the figures whose compute steps succeeded (`truth` draws the relative
displacement, which needs no truth map, and reports its other figures as not drawn; `images` has nothing to draw
without its recovery). A module that requires
another runs only when that module gave its product, whatever it is: `mass`, `truth` and `rgb` therefore need a
successful image recovery. The log ends with the list of failed modules. `run_summary` records each module under
`modules`, in run order: `status` (`done`, `disabled`, `inactive`, `skipped` or `failed`), `mode` (the pass mode),
`seconds`, `steps` (each compute step's `status`, `seconds`, `wrote` and `error` or `reason`; in a render pass those of
the completeness record), `render_steps` when the module rendered, `reporting_draws` and `image_draws` when its
compute used the draws, the module's `reason` or `error`, and `failed_requirement` (the module whose failure it
follows) when it was skipped because a module it requires failed in this pass.

The reporting draws are built the first time a step needs them (the `covariance` step of `fit`); if building them
fails, every later step that needs them fails with the same error. The summary and the results macros are written
before `rgb` and again at the end. A pass that runs only part of the modules, a render pass, or a pass in which a
module failed or was skipped, merges its summary into the saved `run_summary`. Each module owns some top-level keys,
written by its compute (`images` owns `fit`, `truth` owns `truth`): for every module that computed in this pass, done
or failed, its keys and its record are this pass's, so a failed module leaves no old numbers (the macros then say "not
available"). The same holds for a module skipped because a module it requires failed in this pass (`truth` after a
failed `images`): it is recorded as skipped with the reason and its keys are dropped. A disabled module keeps its
saved keys and gets this pass's record (`disabled`). So does a module skipped because a module it requires is
disabled in this pass (`truth` with `outputs.images.enabled=false`): its record says it was skipped and why, and its
numbers stay, since nothing was recomputed and nothing failed. A module that did not run for any other reason (not selected, in
the skip list or requiring a module that is, inactive) keeps its saved keys and its saved record, and a module that ran in a render pass gets this pass's record but keeps the keys of its last
compute, so a render pass leaves the numbers and the macros as the last compute wrote them. The top-level
`reporting_draws` and `warnings` are those of the last pass that built the draws.

At the end of every pass the completeness records of all modules are compared: when the saved results of some modules
were computed over other reporting draws than others (a pass with `--only` and another `--draws`, for example), the
log has one `[outputs] WARNING` that names the modules with their draw counts and seeds, and the warning goes to
`warnings` in `run_summary`. A module whose required module was computed again with other draws after it is named
as stale. The warning never blocks a pass; computing the modules together over the same draws clears it.

### The outputs section

Every setting of the stage lives in the `outputs` section of the configuration
({class}`~kappaccino.config.OutputsConfig`): the settings the modules share at its top level, then one section per
module, named after it, with its `enabled` switch and the module's own settings. The section never changes a fit:
it is not part of the run hash, a resume ignores it, and `kappaccino outputs --set` may change any of it.

```yaml
outputs:
  draws: 64               # reporting draws behind the maps, curves and masses
  image_draws: 256        # draws behind the image positions, residuals and RMS (at least draws)
  workers: 4              # threads evaluating the draws; the products are the same for any value
  caustic_source_redshift: 9.0
  fail_fast: false        # stop at the first step that raises
  image_solver: {}        # the exact-image solver, shared by images and rgb (ImageSolverConfig)
  fit: {enabled: true}
  images: {enabled: true, extra_images: true}
  mass: {enabled: true}
  truth:
    enabled: true
    kappa_true_fits: ../../data/imaging/hera/ffsims/kappa_z9_0.fits
    deflection_true_fits: ../../data/imaging/hera/ffsims/deflAnglesHera.fits
  rgb:
    enabled: true
    image_dir: ../../data/imaging/hera/ffsims
    bands: [F435W, F606W, F814W]
```

A module runs when it is enabled and active (its inputs exist: `truth` needs a truth map, `rgb` an imaging folder).
A disabled module is recorded as `disabled` and the modules that require it are skipped with the reason; `--set
outputs.rgb.enabled=false` does the same for one rerun. Steps cannot be switched on and off. Some inputs are read by
more than one module: `outputs.truth.kappa_true_fits` also by `mass` (aperture and zeta masses) and `rgb` (the cluster mass
figure), and `outputs.truth.deflection_true_fits` also by `images` (the true-lens RMS), so a disabled `truth` leaves those
products alone. `runtime.skip_plots` turns the whole stage off.

### Rerunning the outputs of a finished run

```bash
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --only truth,mass
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --render            # figures only, from the store
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --compute --skip rgb
kappaccino outputs results/m0416_canucs2025/<run> --skip rgb --fail-fast --draws 32
kappaccino outputs RUN --remap /home/dutra/sdev/burkelenscluster=$PWD --only truth
kappaccino outputs RUN --all --set outputs.truth.deflection_true_fits=data/imaging/hera/defl.fits
kappaccino outputs RUN --set outputs.rgb.enabled=false --set outputs.workers=8
kappaccino outputs --list results/hera/w500s250t7_allblocks-3f9a1c
```

The command loads the saved run ({mod}`kappaccino.outputs.saved_run`) and runs the stage into its `final/` folder,
without fitting. The saved paths are as the run wrote them: relative to the folder the run was started from (run the
command from the same folder, usually the repository root) or absolute, as in a run made on another machine.
`--remap OLD=NEW` (repeatable) replaces the prefix `OLD` of every saved path, catalogs, imaging folders and truth and
deflection maps included, and logs each field it changes. Only path-valued fields of the configuration are changed
(the fields whose type is a path), never a name such as `paths.run_name` or `paths.figure_prefix`. `OLD` must be an
absolute path other than `/`; a relative saved path is compared as the absolute path it names from the folder the
command runs in, so `--remap $PWD/data/imaging/m0416/buffalo=/elsewhere` also moves a saved
`data/imaging/m0416/buffalo`. The extra JWST variant of M0416 and AS1063 (`imaging.cutouts.imaging_variants`) is not
a configuration field: its folder is built as `data/imaging/<cluster>/canucs_dr1` (M0416) or
`data/imaging/<cluster>/dja_v7.5` (AS1063) relative to the folder the command runs in, and `--remap` does not move
it, so run the command from the repository root for that variant. Before a full or compute pass, the command checks
that the files the selected modules read exist; a missing one is named by its field, with the hint to use `--remap`.
A render pass needs the catalogs (the state is built from them) and, for `rgb`, the imaging pixels of every variant
its last compute drew; the imaging is not checked before the pass, and a variant whose imaging is missing fails its
render step (see [The rgb module](#the-rgb-module)).

`--render` and `--compute` choose the pass mode (see [Pass modes](#pass-modes)); they exclude each other, and
`--draws` cannot go with `--render`. The `enabled` switches of the run's saved configuration (after `--set`) choose
the modules. `--only A,B` runs A, B and the modules they require, enabled or not (an explicit request wins), and
`--all` runs every module, enabled or not; the two exclude each other. `--skip C,D` leaves out C, D and the modules
that require them on top of that: a module disabled in the configuration stays off, and `--all --skip rgb` runs every
module but `rgb`. `--set outputs.FIELD=VALUE` or
`--set outputs.MODULE.FIELD=VALUE` (repeatable, the syntax of the run command) changes a field of the `outputs`
section before the pass. Anything outside that section is refused: settings in the run hash would no longer describe
the saved fit, and the `paths` section, which the hash leaves out, names the run, the figure prefix and the imaging
variants (`paths.figure_prefix=../../X` would write figures outside the run folder). `--draws N` sets
`outputs.draws` and `outputs.image_draws` to N, as the run command's `--draws` does. After a full or compute pass
(also one that failed), `final/artifacts/config.json` takes the `outputs` section of that pass: the saved one with
`--set` and `--draws` applied, its paths as saved (`--remap` only says where the same files are on this machine;
`--fail-fast` is a flag of the command, not a setting). The saved configuration then describes the settings behind the
products, and a later `kappaccino outputs RUN` uses them. A render pass changes nothing, since it computes nothing. A
pass that computes only some modules (`--only truth --draws 4`) still records its settings for the whole section; the
completeness record of each module keeps the draws its own compute used, and the draws check below warns when they
differ. The run command does the same: resuming a finished run (`kappaccino catalogs/hera --draws 8`) saves the new
`outputs` section in `final/artifacts/config.json` before the outputs stage runs, and leaves the rest of the saved
configuration alone. `--list` prints the modules
with their `requires`, `after` and compute and render steps; given a run, it also says whether each is enabled and
whether it applies to it, judged from the configuration alone, after the `--remap` and `--set` given with it. At the end the command prints the fit numbers of the
saved `run_summary`. The exit code is 1 when a module failed, `--fail-fast`
stopped the pass (one line names the error; the step logged its traceback), the pass failed outside the modules
(writing the run summary, for example; the traceback is printed) or a render pass drew nothing (the command says
why), and 2 for a usage
error, a missing file or a refused `--set` or `--remap`. A `--set` value is checked before the pass, and so is the
whole `outputs` section of every run (`config.validate_outputs`, which calls each section's `validate`). An unknown
field anywhere in the configuration is an error that names its dotted path (`outputs.rgb.bogus`). Every field under
`outputs`, nested sections included (`outputs.rgb.display`, `outputs.image_solver`), must have the type it is
declared with: true or false, an integer, a finite number, a string, a path, a list of strings, or null where the
field allows it (`outputs.rgb.bands` a list of strings or null, `outputs.rgb.display.q` a finite number or null).
On top of that: `outputs.draws` and `outputs.workers` positive, `outputs.image_draws` at least `outputs.draws`; the
path fields non-empty; `outputs.truth.grid_size` nonnegative; `outputs.rgb.image_scale` one of `auto`, `20mas`,
`30mas`, `40mas`, `60mas`, `outputs.rgb.bands` not empty, every `paper_*_cutouts` selector `image:ID`, `family:ID`
or `system:ID`, `fold_pair_max_separation_arcsec` positive; `outputs.image_solver.exact_image_finder` one of the
three finders, its pitches, tolerances, separations, thresholds, `match_tolerance_arcsec` and
`exact_image_lm_max_iter` positive, `exact_image_window_margin_arcsec` and `exact_image_adaptive_max_levels`
nonnegative, the fine pitch at most the coarse one and `exact_image_polish_iterations` positive. The trust radius and
the display values are checked for their type only. A bad value is a usage error that names the field.

### Comparing the outputs of two runs

```bash
PYTHONPATH=src python scripts/compare_outputs.py RUN_A RUN_B
PYTHONPATH=src python scripts/compare_outputs.py RUN_A RUN_B --ignore-entry run_summary --dpi 200 --list-figures
```

`scripts/compare_outputs.py` compares the `final` stages of two run folders (or two stage folders): every table,
record, array bundle and figure record of `results.h5` and every entry of `cache.h5` by content (entries only on one
side, and for differing entries what differs and the largest absolute difference), the file lists, text files
(`*_results.tex` and the like) line by line, FITS files by data and header, other files byte by byte, and every
figure PDF rasterized with `pdftoppm` at `--dpi` (default 100) and compared as pixels (the largest and mean absolute
difference per figure). Table columns and indexes, array keys and JSON values are compared with their types (`true` is
not `1`). `--ignore-entry`, `--ignore-key` (record keys, table columns, array keys, FITS cards) and `--ignore-file`
take shell patterns; the record keys named exactly `seconds`, `elapsed_seconds`, `started_at` and `computed_at`
(wall times, at any depth of a record) are ignored unless `--no-default-ignores`, and nothing else is, so a table
column such as `exposure_seconds` is always compared. It exits with 1 on any other difference.

### Writing a new module with compute, product and render

A module is three things: a subclass of {class}`~kappaccino.outputs.base.OutputModule`, a frozen dataclass of its
settings in `config.py`, and one entry in the registry. The class has a `name`, the modules whose products it reads
(`requires`) or that it should follow (`after`), an optional `active` that returns the reason it does not apply, its
`compute_steps` and `render_steps`, and the three methods:

```python
from kappaccino.outputs.base import ModuleUnavailable, OutputModule


class GGSLModule(OutputModule):
    name = "ggsl"
    requires = ("images",)
    description = "galaxy-galaxy strong-lensing probability"
    compute_steps = ("probability",)
    render_steps = ("probability",)

    def active(self, ctx):
        return None if ctx.config.model.member_populations else "no member populations"

    def compute(self, ctx):
        from ..io import store
        from ..plotting.ggsl import ggsl_probability          # heavy imports inside the methods

        def probability():
            store.write_table(ctx.stage, "ggsl_probability",
                              ggsl_probability(ctx.state, ctx.draws, ctx.settings.min_einstein_radius_arcsec))
            return "ggsl_probability"                          # None would mean: nothing written
        ctx.step("probability", probability)
        if not ctx.ok("probability"):
            raise ModuleUnavailable("no GGSL probability (the probability step failed)")

    def product(self, ctx):                                    # what dependents read, from the store
        from ..io import store
        return store.read_table(ctx.stage, "ggsl_probability")

    def render(self, ctx):
        from ..io import store
        from ..plotting.ggsl import plot_ggsl
        ctx.render_step("probability", lambda: plot_ggsl(store.read_table(ctx.stage, "ggsl_probability"),
                                                         ctx.figures, ctx.prefix))
```

Its settings are a frozen dataclass in `src/kappaccino/config.py`, documented like its neighbours, with an `enabled`
switch and a `validate(where)` method that names `outputs.<module>.<field>` in its errors (it starts with
`_check_field_types`, which checks every field against its annotation, then checks ranges and choices), added as one
field of
{class}`~kappaccino.config.OutputsConfig` under the module's name:

```python
@dataclass(frozen=True)
class GGSLOutputConfig:
    """The ``ggsl`` output module: the galaxy-galaxy strong-lensing probability.

    Parameters
    ----------
    enabled : bool, optional
        Run the module.  Default True.
    min_einstein_radius_arcsec : float, optional
        Smallest member Einstein radius counted, in arcsec.  Positive.  Default ``0.5``.
    """
    enabled: bool = True
    min_einstein_radius_arcsec: float = 0.5

    def validate(self, where: str = "outputs.ggsl") -> None:
        _check_field_types(self, where)                       # every field has its declared type
        _boolean(f"{where}.enabled", self.enabled)
        _positive(f"{where}.min_einstein_radius_arcsec", self.min_einstein_radius_arcsec)


# in OutputsConfig:
    ggsl: GGSLOutputConfig = field(default_factory=GGSLOutputConfig)
```

`OutputsConfig.validate` calls the section's `validate`, the runner reads its `enabled`, and the module reads the
section as `ctx.settings`; a file the compute reads is a path field of the section, named in the module's
`path_fields` as a dotted name under `outputs` (`"ggsl.catalog_path"`), which `kappaccino outputs` checks before a
pass and `--remap` moves. Put the module class in a file of `src/kappaccino/outputs/` and add an instance to
`MODULES` in `outputs/registry.py`, where its position sets its place among modules with no ordering between them.
Nothing else needs to change: the section is outside the run hash with the rest of `outputs`, and `--set
outputs.ggsl.FIELD=VALUE` already accepts it. The context (`ctx`) holds `config`, `state`,
`posterior`, `audit`, `paths`, `stage`, `tables`, `figures`, `prefix`, `log`, `summary`, `products`, `mode`, `settings`,
`completion` (product and render: the completeness record), `imaging(image_config)`, which loads each imaging variant
once per pass, and for compute steps only `evaluator` and the reporting draws `draws`. `active` should decide from
`ctx.settings` (or `ctx.config`) and the products of the modules it requires: `--list` calls it with only the saved
configuration (no state, no products). Keep the imports of JAX and the plotting code inside the methods: the command
line imports the registry to list and check module names. Set `checkpoint = True` on a slow imaging module so that the run summary is written
before it. The instances in `MODULES` are process-wide singletons, shared by every pass of the process: keep no state
on `self` between compute and render (or between passes); what render needs goes through the store.

1. **Steps.** A step is one product or a closely tied set of products, and every step name (its part before any `:`,
   for steps named per variant) is declared in `compute_steps` or `render_steps`; an undeclared name is an error.
   `ctx.ok(step)` tells whether a step of the current phase ran without an error (a step may return None on success),
   and `ctx.computed(step)` whether a compute step ran and wrote something. A step that does not apply to the run
   returns None (`wrote` false); `ctx.skip_step` is only for a step that cannot run because another failed, and the
   render steps that need it are then not drawn. A step that goes on without something it would have used says so
   with `ctx.note(message)`, which logs the message and adds it to the step's `notes`. An input the run configures
   that cannot be read (an unreadable or malformed map) is an error that fails the step, not a skip.
2. **Split the computation from the drawing.** Keep the computation in a function that writes to the store and
   returns something that is not None when it wrote (`save_magnification_recovery`, `save_zeta_masses`,
   `critical_curve_maps` plus one `store.write_arrays`), and the drawing in one that takes the stage and the figure
   path and reads the store (`render_magnification_recovery`, `render_zeta_masses`, `render_critical_curves`).
3. **Decide what to save.** Save what the figure draws, not what was convenient to compute: the per-pixel percentiles
   of the critical-curve eigenvalues, not the per-draw maps; the masked pixel values of the magnification figure, not
   the whole maps; the members actually shown in the subhalo figure, with the selection made in compute. Selections,
   fits, contour levels that depend on the data and anything else that is analysis belong in compute; render only
   lays out. Large per-draw maps stay where they are (`cache.h5` through `ReportingDraws.grid_maps`); render must not
   need them, because `cache.h5` may be deleted. Inputs the figure reads from beside a configured file (the published
   ratios beside the truth map) are saved as tables too, so render needs no data file and no `--remap`. Save with
   full precision and the dtypes of the computation: a float32 array saved as float64 or a float rounded through text
   changes pixels.
4. **Name new entries** after the figure they draw, without the prefix (arrays `critical_curves`,
   `subhalo_mass_concentration`, `magnification_recovery`, `truth_recovery_residuals`), or after the existing table or
   record family (`truth_recovery`, `truth_literature_m2d_ratios`). Reuse an existing entry whenever it already holds
   the data, and write the entries of one product in one `store.batch`. Document each new entry here.
5. **Compute.** Render steps reuse the name of the compute step they draw, so `render_step(name, ...)` finds its
   compute step by default, and get `needs="other_step"` when they draw another (`truth_residuals` needs
   `truth_maps`). Numbers for `run_summary` (`summary_keys`) are set in compute. A compute may pass in-memory results
   between its own steps (`truth` passes the truth maps to the magnification steps); only product and render must
   read the store. If a figure is drawn only under a condition on the data (`truth` draws the profiles only with a
   profile centre), save that condition (here `center` in the record `truth_recovery`) and test it in render.
6. **Write `product`** as a store read of exactly what the dependents use, for example the image table for `images`:
   `return {"image_df": store.read_table(ctx.stage, "image_fit_quality"), ...}`. Never return an object built in
   compute. A `store.MissingEntry` in product fails the module with `ModuleUnavailable` and the hint to run its
   compute, which is what a render pass needs. A module that requires another reads that module's product (`mass`
   and `truth` take `ctx.products["images"]["image_df"]`), not the store.
7. **Test.** In `tests/test_output_modules.py` the fake `Phased` module shows the pattern: run a full pass, delete the
   figure, run a render pass (`OutputContext(..., mode="render")`) and check that the figure comes back without
   compute; any use of `ctx.draws` or `ctx.evaluator` there raises `ComputeOnly` and fails the step, which is the
   guard. On a copy of a real run (Hera: `results/hera/w1000s500t8_allblocks_fast-af374a`, with
   `--remap /home/dutra/sdev/burkelenscluster=$PWD`), run the code before the change on one copy and after it on
   another, then `scripts/compare_outputs.py` the two: every old entry, every FITS and text file and every figure must
   be identical, with only new entries and the module records of `run_summary` added. Then copy the new run, delete
   its figures, run `--render` on the copy and compare again: the figures must come back pixel-identical. Do the
   render pass with `Evaluator.__init__`, `reporting.draws.reporting_draws`, `ReportingDraws.__init__` and
   `saved_run.load_run` patched to raise, to prove it evaluates nothing. `tests/test_pipeline.py` does the same on
   the toy run (where `truth` and `rgb` are inactive), and `tests/test_outputs_rgb.py` and
   `tests/test_outputs_truth.py` do it for `rgb` (synthetic HST imaging) and `truth` (synthetic truth maps).

Pitfalls:

- A figure that redraws "the same thing" from a recomputed quantity is not the same figure. Render recomputes only
  what is deterministic and cheap from saved arrays (the zero contours of a saved median map, through the same
  helper compute uses, `plots.critical_curve_contours`); everything else is saved.
- `abs(median(mu))` is not `median(abs(mu))`: the magnification figure uses the first, the residual map the second.
  Check which quantity each figure really draws before saving one in place of the other.
- Writers that return None after logging a skip must keep doing so: render relies on `wrote` false to leave the
  figure out, and a stale entry from an older compute must never be drawn.
- A table that a render step writes (`zeta_mass_function`, the counts behind the subhalo figure) is presentational and
  is written again by every render pass; a table a later module or the macros read belongs to compute.
- Entries are never deleted from the store, so a render step must not decide from `store.has_*` (or from a
  `store.MissingEntry` caught) whether to draw; use `ctx.computed`, `render_step` and the module's own records.
- A render step draws only what the last compute of its own module wrote. A table of another module (`fit`'s
  `covariance_errors_both_planes` in the covariance cutouts) is read in compute and what the figure draws of it is
  saved under the module's own name; an entry that one step reuses from another (the small-lambda critical fields in
  the paper cutouts) is named in a record that the same compute writes, and render reads that record.
- Render reads no external file: inputs such as the M0416 candidate catalog are saved in compute.

## First checks

`final/tables/run_summary.txt`
: Images recovered out of observed, rms and median image-plane residuals, split R-hat, ESS, divergences and run times.

`<prefix>_chain_traces.pdf`
: Traces of the sampler diagnostics, all global parameters and the twelve other parameters with the highest split
  R-hat. The table `chain_convergence` holds split R-hat and ESS for every parameter.

`final/artifacts/audit.json`
: Whether the far-field approximation held over the posterior.

## Standard figures

| figure | content |
|---|---|
| `image_residual_histogram.pdf` | image-plane residuals; with `outputs.truth.deflection_true_fits`, the true-lens RMS: the rms of the true deflection map on the catalog with a free source per family and one free amplitude, a reference value and not a lower bound on the RMS (band from a bootstrap over families; table `true_lens_residuals`, record `true_lens_rms`) |
| `image_recovery.pdf` | observed and model images per family |
| `corner_globals.pdf` | posterior of the global parameters |
| `scaling_relation.pdf` | member scaling relation and deviations |
| `critical_curves.pdf` | critical curves and caustics (drawn from the arrays `critical_curves`: the eigenvalue percentiles, the caustics and the sources) |
| `image_catalog_family_cluster_mass.pdf` | cluster mass overview on the imaging, with the image catalog |
| `member_deviations.pdf`, `member_constraints.pdf` | per-member deviations and how well the data constrain them |
| `whitened_image_residuals.pdf`, `far_field_accuracy.pdf` | residuals in units of their covariance; far-field error |
| `<population>_scaling_correlations.pdf` | correlations among the relation parameters |

Corner plot axes keep log10 coordinates for log-sampled parameters; the summaries above the histograms report physical
values with asymmetric uncertainties.

## GP and GRF diagnostics

GP runs write, with plotting on:

- `gp_kernel_sky.pdf`: positions and links colored by total displacement correlation;
- `gp_shared_vs_white.pdf`: relative-displacement noise and a joint uncertainty ellipse;
- `gp_scale_posteriors.pdf`: shared and white amplitude posteriors (mas).

Shared-field GRF runs (`gp_blocks="all"`) also write `total_grf_dimensionless_power.pdf` and
`gp_correction_correlation.pdf` (per-pixel medians over the reporting draws, and the median and 16–84% of the
knee spectrum), with the maps in the arrays and record `conditional_field_maps`. For these runs the truth maps, magnification recovery, aperture profiles,
critical curves and caustics include each posterior draw's realization of the GRF. System- and family-block runs do
not define one cluster-wide field and skip these figures.

## Simulated clusters

With truth maps (`outputs.truth`, {class}`~kappaccino.config.TruthOutputConfig`, set for Hera and Ares), the run also writes the following. Every figure is
drawn from the store (`kappaccino outputs RUN --render` draws them again): the residual maps from the arrays
`truth_recovery_residuals`, the profiles from the profile tables, the record `truth_recovery` (centre, mass per unit
convergence per pixel) and the published ratios saved as `truth_literature_m2d_ratios` and
`truth_literature_kappa_ratios`, the magnification figure from the arrays `magnification_recovery` (the true and model
|μ| of the pixels drawn) and its two tables, the source area from its table and provenance record.

- `truth_recovery_residuals.pdf` and `mass_profile.pdf` (enclosed projected mass of model and truth, and their ratio);
- `kappa_mass_residuals.pdf`: model over truth for the enclosed mass and the azimuthally averaged convergence, with the
  published CATS and GLAFIC ratios when `meneghetti2017_ratios.csv` sits beside the truth map;
- `magnification_recovery.pdf`: truth against model magnification at the observed images and over the image-position
  convex hull at the truth-map source redshift (9). The orange error bars are spatial 16–84 percentiles, not posterior
  uncertainties;
- `cumulative_source_area.pdf`: source-plane area magnified above μ at the truth-map source redshift,
  A(>μ) = Σ A_pix/|μ| over the truth-map pixels with |μ| > μ, for the truth, the parametric model and the parametric
  + GRF model (each draw's realization of the field), with the comoving volume of the shell z ± 0.5 on the right axis and the
  ratio to the truth below. Lines and bands are the median and 16–84% range over the reporting draws. Thresholds below
  μ = 3 depend on the size of the map and are shaded. Numbers in the table `cumulative_source_area`;
- the table `magnification_scatter` (no figure): scatter and bias of the model magnification about the truth at the map
  pixels inside the image hull whose true magnification is within 10% of 3 or 10, over all reporting draws, for the
  parametric model and the parametric + GRF model (after Raney et al. 2020). `scripts/make_scatter_summary.py` draws
  these for several clusters as `scatter_summary.pdf`;
- `relative_displacement.pdf` (with `outputs.truth.deflection_true_fits` and a shared field): the rms difference of a
  displacement between two points against their separation, along the separation (blue) and across it (orange).
  Dashed is the true map minus each draw's parametric lens, solid the draw's mass-field realization, both at the
  field's reference redshift with the modes the likelihood projects out (`gp_project_modes`) removed by least squares;
  green is the image-plane GP (each draw's true minus lens position on the left, its analytic curve on the right).
  The left panel uses every pair of catalog images, the right panel random pairs of a regular grid over the whole true
  map (its step the smallest that keeps it within 60,000 points: 1" on Hera, 1.5" on Ares). The open circles are the
  image position scatter of the true lens (its residuals on the catalog); they are measured over image pairs in both
  panels. Lines are the median over the reporting draws, bands (mass field and GP) the 16–84% range. The compute step
  `relative_displacement` of `truth` writes the table and record `relative_displacement` (every curve per draw and
  bin, and the grid, bins and draws used) from the `images` tables `posterior_image_residual_draws` and
  `true_lens_residuals` and the record `true_lens_rms` of the same pass; the render step draws the figure from the
  table alone. The step writes nothing, and the figure is left out, when the run has no deflection map, the `images`
  compute wrote no true-lens record, the run has no shared field or no image was observed; the log then says
  `[outputs] relative displacement skipped: <reason>`. A deflection map that cannot be read, or is malformed, fails the
  step (the other steps of `truth` go on), as it fails the `true_lens` step of `images`. With a deflection map and no
  `outputs.truth.kappa_true_fits`, `truth` runs this step alone. The step runs before the truth maps, so its figure is
  drawn even when the `truth_maps` step fails;
- `subhalo_mass_concentration.pdf`: one row of four panels for the members on the true map and the parametric + GRF
  model map (evaluated on the truth-map pixels). Mass is the zeta statistic (Fahlman et al. 1994) in 10 kpc, the mass
  inside 10 kpc above the mean of the 10–20 kpc annulus; concentration is κ̄(<R_e)/κ̄(R_e–2R_e) − 1. Panels: number of
  subhalos per 0.2 dex of mass, model against true mass, concentration against mass, model against true concentration.
  The two brightest members and members whose 16–84% posterior range of mass or concentration exceeds twice the median
  are left out. Tables `member_zeta_masses` (every member, with the parametric-only model too) and `zeta_mass_function`
  (the counts of the first panel, written when the figure is drawn), the record `zeta_masses`, the arrays
  `subhalo_mass_concentration` (the members shown, their true values and the full model's value in every draw) and, in `cache.h5`, `zeta_masses_cache` (reused when the draws match;
  `regenerate_zeta_masses.py --recompute` ignores it); the record `aperture_mass_histogram` keeps the 2R_e aperture-mass ratios of Meneghetti et al. (2017), each member's model
  mass the median over the same posterior draws as the truth maps.

## Imaging figures

When imaging is configured and present under `data/imaging/`:

- `small_lambda_covariance_cutouts.pdf`: RGB cutouts, linear covariance ellipses and second-order lens-map contours for
  the twelve images with the smallest Jacobian eigenvalues;
- `fold_pairs_source_caustics.pdf`: model-identified fold pairs (mutual-nearest opposite-parity images within
  `outputs.rgb.fold_pair_max_separation_arcsec`), with posterior critical-curve bands and source panels;
- for M0416 and AS1063, JWST versions with an `_jwst` suffix, including contours of the predicted observed image positions (68.3% solid,
  95.4% dashed).

A compute reports missing imaging and leaves that variant out. `kappaccino outputs RUN --render --only rgb` draws all
of these again from the store and the imaging, and fails for a variant whose imaging the compute had but the render
cannot load (see [The rgb module](#the-rgb-module)). The marker sizes of the M0416 overview come from the F200W
photometry that the compute saved (`<stem>_marker_photometry`), so the render reads no packaged data file.

The det A maps of the fold pairs, fold systems and small cutouts are not evaluated over the whole lens on their own
grids. Each draw's parametric lens on those grids is the overview's (512 x 512 over the cluster, evaluated anyway)
interpolated with a bicubic spline, with the cluster members and halos close to the grid evaluated exactly, and the
draw's GRF realization is evaluated exactly on the grid (`reporting.local_maps`). Every draw is checked against the
exact lens at up to 24 x 24 points of the grid and evaluated directly when it misses by more than 1e-5. On Hera and
AS1063 the zero contours of det A moved by at most 6e-6 arcsec and no fold pair changed. Setting
`ReportingDraws.maps_from_reference = False` evaluates every component on every grid as before.

The RGB critical overview (`<prefix>_hst_rgb_critical_z9.pdf` or `<prefix>_jwst_rgb_critical_z9.pdf`) shows the
z=9 critical curve and its 16–84% band, the observed images (red when not recovered), the extra images
and the 68.3% contours of the predicted observed positions, each enlarged four times about the mean of its
image's positions. A finished run's overview can be drawn again from its saved results, without the posterior or the
determinant maps, from the repository root:

```bash
python -m kappaccino.plotting.jwst_rgb results/<cluster>/<run> [results/<cluster>/<run> ...]
```

It reads the band from the arrays `<stem>`, the catalog from the table `image_fit_quality` and the contours from
the saved image recovery (all in `final/results.h5`), and rebuilds the configuration
from `inputs/command.json`. It refuses when the imaging settings, the image catalog or the reference of that
configuration differ from `final/artifacts/config.json`, and, before writing anything, when the run has no completed
image recovery or a draw of `<stem>` is missing from `posterior_image_residual_draws`. The model description in the
figure record comes from `final/artifacts/config.json`, which also records how many images got a contour
(`recovered_position_contours`). For M0416 it draws the candidates saved in the record `<stem>_candidate_catalog`
and sizes the markers from the table `<stem>_marker_photometry`.
It rewrites the figures, their records, the marker tables, `posterior_overview_positions` and
`posterior_overview_contours` (arrays and records) in `final/results.h5`, never the arrays `<stem>`.

## Reading results in Python

```python
from kappaccino.io import store
from kappaccino.io.hdf5 import load_arrays

run = "results/demo/hera"
summary = store.read_table(f"{run}/final", "posterior_summary")       # a DataFrame
fit = store.read_record(f"{run}/final", "run_summary")["fit"]
maps = store.read_arrays(f"{run}/final", "conditional_field_maps")    # dict of arrays
record = store.read_figure_record(f"{run}/final/figures/hera_corner_globals.pdf")
posterior = load_arrays(f"{run}/final/artifacts/posterior.h5")
```

`store.entries(stage)` lists the names. Write only from the main thread
of a process: the writes are serialized by a lock, but the worker threads of the output pass never write. Writes made
in a row go inside `with store.batch(stage):`, which commits them as one new version of `results.h5` (one copy of the
file instead of one per entry). An exception that leaves the block, a nested `with store.batch(stage):` or a write
inside it discards all of them; if it was caught inside the block, later writes and the end of the block raise
`store.BatchFailed`, so a batch is never committed in part. A table or record the store refuses (`TypeError`) can be
caught without failing the batch. A damaged entry raises `store.DamagedEntry`, which is not a `store.MissingEntry`.

## Manuscript macros

Every final run writes `final/figures/<prefix>_results.tex`, and mock aggregates write
`aggregate_plots/mock_results.tex`. Include one with `\input{...}` and use commands such as `\HeraNImages`,
`\HeraNRecovered`, `\HeraRMS`, `\HeraMedianResidual` and `\HeraRuntime`. Residuals are in arcseconds; add the units
in the text. Names contain letters only: M0416 uses `MJZeroFourOneSix`, AS1063 uses `ASOneZeroSixThree`.

`\HeraRMS`, `\HeraMedianResidual`, `\HeraLensOnlyRMS` and `\HeraLensOnlyMedianResidual` are posterior medians with
16–84% ranges over the saved draws (the table `posterior_image_residual_draws`, and `..._lens_only` for the parametric
lens without the GRF), written for math mode, e.g. `$\HeraRMS''$`. The image counts (`\HeraNRecovered`,
`\HeraNMissed`, `\HeraNExtraImages`, `\HeraRecoveredPercent`) are medians over the reporting draws, with
`...Interval` macros for the 16–84% range. In the image tables an image is recovered when at least half the draws
recover it; its model position, residual and magnification are medians over the draws that do. Extra images are
those found in at least half the draws.

The extra images of a draw come from a global search of its lens (parametric plus its GRF realization). Every family's
window, the span of its images plus `outputs.image_solver.exact_image_window_margin_arcsec`, is ray-shot on a coarse grid
(`exact_image_grid_pitch_arcsec`) with a finer sub-grid (`exact_image_fine_pitch_arcsec`) near critical curves, and
every candidate is refined to a root of the lens equation. With a single lens plane and a fixed cosmology the lens of
any source redshift is the lens at one reference redshift times the ratio of lensing efficiencies, plus the external
shear, which does not scale. The search then puts all families on one lattice, each window snapped outward onto it,
and refines the candidates of all families together; each draw checks the scaling at a few points first and, if it
does not hold, searches each source redshift on its own grid. The two searches place their grid points differently,
so on a few draws one of them finds a faint image near the limit of the coarse grid that the other misses.

`\HeraSigmaTotal` is the quadrature sum of `\HeraSigmaGRF`, `\HeraSigmaGP` and `\HeraSigmaMeas` taken per posterior draw
(arcsec, per coordinate, for an image at `\HeraGRFReferenceRedshift`), and `\HeraGRFVariancePercent` the share of the GRF in
that variance; both are medians over the draws with `...Interval` macros for the 16–84% range.

For a mock campaign, `\MockRMS` is the median and 16–84% range of the RMS residual over the mocks and their draws, and
`\MockInjectedOffset` the same over the mocks of the rms separation between the observed and true image positions
(the displacement and measurement noise added to each mock catalog), both in arcsec for math mode.
The same file holds what the mocks inject. `\MockInjectedSigmaGRF`, `\MockInjectedSigmaGP`, `\MockInjectedSigmaMeas`,
`\MockInjectedLambdaB`, `\MockInjectedEllGP` and `\MockInjectedGammaKappa` are medians over the mocks, and
`\Mock<Name>Min` and `\Mock<Name>Max` give the range each is drawn from (`SigmaGRF`, at `\MockFieldReferenceRedshift`,
`LambdaB`, `GammaKappa`, `SigmaGP`, `EllGP`, `SigmaMeas`). `\MockSchechterAlpha`, `\MockSchechterMagnitude`,
`\MockPivotMagnitude`, `\MockVelocityScatterDex` and `\MockVelocityScatterPercent` describe the member population.
`\MockNSystemsPerMock`, `\MockNFamiliesPerMock` and `\MockNImagesPerMock` are the median numbers of lensed systems, image
families (a knot of a system is its own family) and images per mock, each with `...Min` and `...Max`.

Simulated clusters with a truth map also get the truth recovery within the image footprint: `\HeraFootprintMassRatio`
and `\HeraFootprintKappaRatio` (mean model/true ratio of the enclosed mass and of the azimuthally averaged
convergence, uniform in log R from `\HeraFootprintInnerArcsec` to `\HeraFootprintOuterArcsec`, the outermost
image cut at the edge of the published figure), `\HeraFootprintMassMeanAbsPercent`, `\HeraFootprintKappaMeanAbsPercent`, `\HeraFootprintMassRMSPercent` and
`\HeraFootprintKappaRMSPercent` (mean |model/true − 1| and rms of model/true − 1 over the same radii, in per cent; unlike the mean ratio it
does not let high and low stretches cancel, and it weights the large excursions), and the same with a model name appended (`...RatioCATS`,
`...RatioGLAFIC`, `...RatioJohnsonSharon`, `...RatioZitrinLTMgauss`, `...RatioZitrinNFW`) for the digitized
parametric reconstructions of Meneghetti et al. (2017). Our numbers are medians over the saved posterior draws
(the tables `truth_recovery_{m2d_aperture,kappa_annulus}_ratio_draws`, `\HeraFootprintNDraws` of them, the
reporting draws), and each has an `...Interval` macro for math mode with the 16-84 per cent
range, e.g. `$\HeraFootprintMassRMSPercentInterval\%$`, and `...Minus` and `...Plus` macros with the lower and upper error alone.

Simulated clusters with shear truth also get the magnification against the truth:
`\HeraMagnificationScatterMuThreePercent`, `\HeraMagnificationScatterMuTenPercent` and
`\HeraMagnificationBiasMu...Percent` (from the table `magnification_scatter`, in per cent), and `\HeraSourceAreaRatioMuThree`,
`...MuTen` and `...MuThirty` (median source-plane area magnified above that threshold over the true area, from
the table `cumulative_source_area`). These use the lens of each draw, parametric plus its realization of the GRF; the same
names with `LensOnly` after the cluster prefix (`\HeraLensOnlyMagnificationScatterMuTenPercent`) are the parametric
lens alone. They are "not available" until the run has those tables.

Regenerate from saved artifacts, without inference or plotting:

```bash
PYTHONPATH=src python -m kappaccino.reporting.results_tex path/to/run/final
```

To regenerate individual figures without rerunning inference, see {doc}`scripts`.
