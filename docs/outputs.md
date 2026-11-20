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
such as `family_id` written as strings reads back as strings, where the old `tables/<name>.csv` read it as integers.
Runs written before this layout keep their loose files, and every reader falls back to them.

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
python -m kappaccino.io.store results/<cluster>/<run> --export /tmp/<run>_loose  # the old csv/json/h5 files
```

`--cache` makes `--table`, `--record` and `--arrays` read `cache.h5`. The export writes `<stage>/tables/<name>.csv`,
`.json` and `.h5`, `<stage>/figures/<figure>.json` and `<stage>/cache/<name>.*`. All of this also works on runs in the
old layout. An entry that cannot be read is listed as `DAMAGED` (or reported as not exported), the rest is listed or
exported, and the command exits with status 1.

Every reported product (figures, tables and results macros) is computed over one set of reporting draws:
`reporting.draws` random posterior draws over all chains, chosen with `runtime.seed` (64 in a full run, 8 in the
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
  completeness record `output_module_<name>` of each converted output module (see [Output modules](#output-modules));
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
  matches: the draws (seed, indices, parameter values), every `image_diagnostics` setting, `reporting.extra_images`,
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
| `fit` | not converted yet: `tables` (`posterior_summary`, `member_classification`), `chain_traces`, `audit` (the record), `scaling_table` (`member_scaling_relation`), `potfiles`, `block_corner` (`corner_combined.pdf`), `cosmology_corner`, `covariance` (covariance corner and `covariance_errors_both_planes`), `fields` (GP and GRF plots), `members` (`member_diagnostics` and the member plots), `scaling_plot`, `globals_corner` | | | always |
| `images` | `recovery` (the image-recovery tables, record `image_recovery_fingerprint`, `position_coverage`, `fit` in `run_summary`), `true_lens` (with `truth.deflection_true_fits` and observed images: `true_lens_residuals`, record `true_lens_rms`) | `residual_histogram`, `recovery_figure` (both with observed images) | | always |
| `mass` | `critical_curves` (arrays `critical_curves`), `aperture_masses` (`member_aperture_masses`, record `aperture_mass_histogram`), `zeta_masses` (`member_zeta_masses`, record `zeta_masses`, arrays `subhalo_mass_concentration`) | `critical_curves`, `zeta_masses` (the figure and the table `zeta_mass_function`) | `images` | always; the masses with a truth convergence map |
| `truth` | `truth_maps` (FITS, profile tables, record `truth_recovery`, arrays `truth_recovery_residuals`, the published ratios), `magnification_recovery`, `cumulative_source_area`, `magnification_scatter` | `magnification_recovery`, `cumulative_source_area`, `truth_residuals`, `mass_profile`, `kappa_mass_residuals` | `images` | with `truth.kappa_true_fits` |
| `rgb` | not converted yet: per imaging variant `imaging`, `covariance_cutouts`, `rgb_critical`; for the overview variants `cluster_mass` and, computed once, `fold_data`, then `fold_cutouts`; for the JWST overview also `position_contours` and `small_lambda` (before the mass figure) and `apj_fold_cutouts` and `paper_cutouts` (after the folds) | | `images` | with `image_catalog.image_dir` and observed images |

`fit` and `rgb` are being converted. Until then each still computes and draws in one `run`: a full or compute pass
runs it whole (a compute pass then draws its figures too, because its dependents need what it computes), and a render
pass reports it `unavailable` ("not converted to compute/render yet") and gives no product.

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

`mass` and `rgb` also run after `fit` without needing its product, because they read tables it writes:
`member_diagnostics` (aperture and zeta masses) and `covariance_errors_both_planes` (covariance cutouts). The steps of
`rgb` are named per variant, for example `fold_cutouts:jwst`. `kappaccino outputs --list` prints the modules with
these relations and their compute and render steps.

### Pass modes

`full`
: compute, product and render; the default, and what the run command does after the fit.

`compute`
: compute and product, no figures (`kappaccino outputs RUN --compute`), for example on a machine without the
  imaging.

`render`
: product and render only (`kappaccino outputs RUN --render`): every figure of a converted module drawn again from
  what the last compute saved. The run is loaded without its evaluator (`saved_run.load_saved_run`: configuration,
  state and posterior; on Hera 3 s, of which 0.8 s go to about 40 one-operation JAX kernels the state's arrays need,
  and no model function is compiled), the reporting draws are never built, and the files the computes read (truth
  maps, imaging) are not checked. A render pass of `mass` and `truth` on Hera takes 5 s; the full pass that computed
  them took 3 min 40 s.

In a render pass (and in the product and render phases of any pass) `ctx.draws` and `ctx.evaluator` raise
`ComputeOnly`, which names the module and step that asked, so a figure that still needs the lens fails its step
instead of being computed again in secret.

### The completeness record

When a converted module's compute starts, its record `output_module_<name>` in `results.h5` is set to
`{"complete": false, "started_at": ...}`; when the compute ends it holds `complete` (true only when every compute step
ran without failing), `computed_at`, the compute steps (`status`, `seconds`, `wrote`, and `error` or `reason`), and
when the compute used the reporting draws `reporting_draws`, `image_draws`, `draws_seed` and `draws_key`. A compute
that fails, raises or is interrupted therefore never leaves a record that claims completeness. A render pass renders a
module only when its record is complete; otherwise the module fails with `ModuleUnavailable` and a message that says
to run its compute first (`kappaccino outputs RUN --only <module>`, or `--compute --only <module>`). A module pulled
into a render pass by `--only` (because a selected module requires it) gives only its product.

A compute step that returns None wrote nothing (it did not apply, for example the magnification figures without shear
truth) and is recorded with `wrote` false. Render steps are declared against the compute steps they draw
(`ctx.render_step`): a render step whose compute step failed is skipped with the reason, and one whose compute step
wrote nothing is left out.

### Steps, failures and the run summary

A step that raises is logged with its traceback and recorded, and the module goes on with its other steps; a step
that needs the result of a failed one is skipped with the reason (no step of `images` runs after a failed `recovery`,
no `truth` step after a failed `truth_maps`, no fold step after a failed `fold_data`, no figure whose compute step
failed). A module with a failed step ends as `failed` but keeps its product, and a full pass still draws the figures
whose compute steps succeeded. A module that cannot give its dependents what they need raises after its steps
(`images` without its recovery, `truth` without its maps) and is failed with no product. A module that requires
another runs only when that module gave its product, whatever it is: `mass`, `truth` and `rgb` therefore need a
successful image recovery. The log ends with the list of failed modules. `run_summary` records each module under
`modules`, in run order: `status` (`done`, `inactive`, `skipped`, `unavailable` or `failed`), `mode` (the pass mode),
`seconds`, `steps` (each compute step's `status`, `seconds`, `wrote` and `error` or `reason`; in a render pass those of
the completeness record), `render_steps` when the module rendered, `reporting_draws` and `image_draws` when its
compute used the draws, and the module's `reason` or `error`.

The reporting draws are built the first time a step needs them (the `covariance` step of `fit`); if building them
fails, every later step that needs them fails with the same error. The summary and the results macros are written
before `rgb` and again at the end. A pass that runs only part of the modules, a render pass, or a pass in which a
module failed or was skipped, merges its summary into the saved `run_summary`. Each module owns some top-level keys,
written by its compute (`images` owns `fit`, `truth` owns `truth`): for every module that computed in this pass, done
or failed, its keys and its record are this pass's, so a failed module leaves no old numbers (the macros then say "not
available"). A module that did not run keeps its saved keys and its saved record, and a module that ran in a render
pass gets this pass's record but keeps the keys of its last compute, so a render pass leaves the numbers and the
macros as the last compute wrote them. The top-level `reporting_draws` and `warnings` are those of the last pass that
built the draws; the module records say which draws each module used, so a mixed record shows.

### Choosing modules

The `outputs` section of the configuration ({class}`~kappaccino.config.OutputsConfig`) selects them:

```yaml
outputs:
  skip: [rgb]             # no imaging products
  # only: [truth]         # these and what they require (here images); null for all
  # fail_fast: true       # stop at the first step that raises
```

or `--set outputs.skip=[rgb]` on the command line. Steps cannot be selected. The section changes only what is
reported, so it is not part of the run hash and a resume ignores it. `runtime.skip_plots` turns the whole stage off.

### Rerunning the outputs of a finished run

```bash
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --only truth,mass
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --render            # figures only, from the store
kappaccino outputs results/hera/w500s250t7_allblocks-3f9a1c --compute --skip rgb
kappaccino outputs results/m0416_canucs2025/<run> --skip rgb --fail-fast --draws 32
kappaccino outputs RUN --remap /home/dutra/sdev/burkelenscluster=$PWD --only truth
kappaccino outputs RUN --all --set truth.deflection_true_fits=data/imaging/hera/defl.fits
kappaccino outputs --list results/hera/w500s250t7_allblocks-3f9a1c
```

The command loads the saved run ({mod}`kappaccino.outputs.saved_run`) and runs the stage into its `final/` folder,
without fitting. The saved paths are as the run wrote them: relative to the folder the run was started from (run the
command from the same folder, usually the repository root) or absolute, as in a run made on another machine.
`--remap OLD=NEW` (repeatable) replaces the prefix `OLD` of every saved path, catalogs, imaging folders and truth and
deflection maps included, and logs each field it changes. Before a full or compute pass, the command checks that the
files the selected modules read exist; a missing one is named by its field, with the hint to use `--remap`. A render
pass needs only the catalogs (the state is built from them).

`--render` and `--compute` choose the pass mode (see [Pass modes](#pass-modes)); they exclude each other, and
`--draws` cannot go with `--render`. `--only` and `--skip` replace the run's saved `outputs.only` and `outputs.skip`
together (the one not given is empty), `--all` runs every module whatever the saved selection, and with none of them
the saved selection applies. `--set SECTION.FIELD=VALUE` (repeatable, the syntax of the run command) changes a report
setting before the pass: only settings that the run hash leaves out are accepted (the `outputs` section,
`truth.deflection_true_fits`, `runtime.output_workers`, the WBIC settings and the `paths` section), because any other
change would no longer describe the saved fit. `--draws N` sets `reporting.draws` and `reporting.image_draws` to N, as
the run command's `--draws` does; both are in the run hash, and this is the one change of that kind a rerun allows.
`--list` prints the modules with their `requires`, `after` and compute and render steps; given a run, it also says
whether each applies to it, judged from the configuration alone. At the end the command prints the fit numbers of the
saved `run_summary`. The exit code is 1 when a module failed (an unavailable unconverted module does not count) or
`--fail-fast` stopped the pass (one line names the error; the step logged its traceback), and 2 for a usage error, a
missing file or a refused `--set`.

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
difference per figure). `--ignore-entry`, `--ignore-key` (record keys, table columns, array keys, FITS cards) and
`--ignore-file` take shell patterns; the timing keys `seconds`, `*_seconds`, `started_at` and `computed_at` are
ignored unless `--no-default-ignores`. It exits with 1 on any other difference.

### Adding a module

A module is a subclass of {class}`~kappaccino.outputs.base.OutputModule` with a `name`, the modules whose products it
reads (`requires`) or that it should follow (`after`), an optional `active` that returns the reason it does not apply,
its `compute_steps` and `render_steps`, and the three methods:

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
            store.write_table(ctx.stage, "ggsl_probability", ggsl_probability(ctx.state, ctx.draws))
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

Put it in a file of `src/kappaccino/outputs/` and add an instance to `MODULES` in `outputs/registry.py`, where its
position sets its place among modules with no ordering between them. A step is one product or a closely tied set of
products; `ctx.ok(step)` tells whether a step of the current phase ran without an error (a step may return None on
success), and `ctx.computed(step)` whether a compute step ran and wrote something. The context (`ctx`) holds
`config`, `state`, `posterior`, `audit`, `paths`, `stage`, `tables`, `figures`, `prefix`, `log`, `summary`,
`products`, `mode`, `completion` (product and render: the completeness record), `imaging(image_config)`, which loads
each imaging variant once per pass, and for compute steps only `evaluator` and the reporting draws `draws`. `active`
should decide from `ctx.config` and the products of the modules it requires: `--list` calls it with only the saved
configuration (no state, no products). Keep the imports of JAX and the plotting code inside the methods: the
configuration imports the registry to check module names. Set `checkpoint = True` on a slow imaging module so that the
run summary is written before it.

### Converting a module to compute and render

This is how `mass`, `truth` and `images` were converted, for the modules still on `run` (`fit`, `rgb`).

1. **List every product.** For each step of `run`, write down what it writes (tables, records, arrays, FITS, figures)
   and what each figure function reads. Most writers already save part of what their figure needs (a table, a
   provenance record); the gap is usually an in-memory array or a number passed straight from the computation to the
   plot.
2. **Split each writer in two.** Keep the computation in a function that writes to the store and returns something
   that is not None when it wrote (`save_magnification_recovery`, `save_zeta_masses`, `critical_curve_maps` plus one
   `store.write_arrays`), and the drawing in one that takes the stage and the figure path and reads the store
   (`render_magnification_recovery`, `render_zeta_masses`, `render_critical_curves`). Leave the drawing code itself as
   it is (move it, do not reformat it), so the figure does not change. Where scripts or tests call the old
   `write_<name>`, keep it as the two in a row; it is not a shim, it is the same function.
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
5. **Write the module.** `compute_steps` keep the old step names; render steps reuse the name of the compute step they
   draw, so `render_step(name, ...)` finds its compute step by default, and get `needs="other_step"` when they draw
   another (`truth_residuals` needs `truth_maps`). Numbers for `run_summary` (`summary_keys`) are set in compute. A
   compute may still pass in-memory results between its own steps (`truth` passes the truth maps to the
   magnification steps); only product and render must read the store. If a figure was drawn only under a condition
   on the data (`truth` draws the profiles only with a profile centre), save that condition (here `center` in the
   record `truth_recovery`) and test it in render.
6. **Write `product`** as a store read of exactly what the dependents use, for example the image table for `images`:
   `return {"image_df": store.read_table(ctx.stage, "image_fit_quality"), ...}`. Never return an object built in
   compute. A `store.MissingEntry` in product makes the module unavailable with the hint to run its compute, which is
   what a render pass needs. A module that requires another reads that module's product (`mass` and `truth` take
   `ctx.products["images"]["image_df"]`), not the store. When all five modules are converted, delete the
   transitional path (`run`, `steps`, `converted`, `NOT_CONVERTED` in `outputs/base.py`, the unconverted branch of
   `--list`, and the transitional test).
7. **Test.** In `tests/test_output_modules.py` the fake `Phased` module shows the pattern: run a full pass, delete the
   figure, run a render pass (`OutputContext(..., mode="render")`) and check that the figure comes back without
   compute; any use of `ctx.draws` or `ctx.evaluator` there raises `ComputeOnly` and fails the step, which is the
   guard. On a copy of a real run (Hera: `results/hera/w1000s500t8_allblocks_fast-af374a`, with
   `--remap /home/dutra/sdev/burkelenscluster=$PWD`), run the code before the change on one copy and after it on
   another, then `scripts/compare_outputs.py` the two: every old entry, every FITS and text file and every figure must
   be identical, with only new entries and the module records of `run_summary` added. Then copy the new run, delete
   the module's figures, run `--render` on the copy and compare again: the figures must come back pixel-identical.
   Do the render pass with `Evaluator.__init__`, `reporting.draws.reporting_draws` and `saved_run.load_run` patched
   to raise, to prove it evaluates nothing.

Pitfalls met in `mass` and `truth`:

- A figure that redraws "the same thing" from a recomputed quantity is not the same figure. Render recomputes only
  what is deterministic and cheap from saved arrays (the zero contours of a saved median map, through the same
  helper compute uses, `plots.critical_curve_contours`); everything else is saved.
- `abs(median(mu))` is not `median(abs(mu))`: the magnification figure uses the first, the residual map the second.
  Check which quantity each figure really draws before saving one in place of the other.
- Writers that return None after logging a skip must keep doing so: render relies on `wrote` false to leave the
  figure out, and a stale entry from an older compute must never be drawn.
- A table that a render step writes (`zeta_mass_function`, the counts behind the subhalo figure) is presentational and
  is written again by every render pass; a table a later module or the macros read belongs to compute.
- Entries are never deleted from the store, so a render step must not decide from `store.has_*` whether to draw;
  use `ctx.computed`, `render_step` and the module's own records.

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
| `image_residual_histogram.pdf` | image-plane residuals; with `truth.deflection_true_fits`, the true-lens RMS: the rms of the true deflection map on the catalog with a free source per family and one free amplitude, a reference value and not a lower bound on the RMS (band from a bootstrap over families; table `true_lens_residuals`, record `true_lens_rms`) |
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

With truth maps (`TruthRecoveryConfig`, set for Hera and Ares), the run also writes the following. Every figure is
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
  `image_diagnostics.fold_pair_max_separation_arcsec`), with posterior critical-curve bands and source panels;
- for M0416 and AS1063, JWST versions with an `_jwst` suffix, including contours of the predicted observed image positions (68.3% solid,
  95.4% dashed).

Missing imaging is reported and skipped.

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
the saved image recovery (all in `final/results.h5`, or the loose files of an old run), and rebuilds the configuration
from `inputs/command.json`. It refuses when the imaging settings, the image catalog or the reference of that
configuration differ from `final/artifacts/config.json`, and, before writing anything, when the run has no completed
image recovery or a draw of `<stem>` is missing from `posterior_image_residual_draws`. The model description in the
figure record comes from `final/artifacts/config.json`, which also records how many images got a contour
(`recovered_position_contours`). It rewrites the figures, their records, the marker tables and
`posterior_overview_positions` in `final/results.h5`, never the arrays `<stem>`.

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

The same calls read an old run's loose files. `store.entries(stage)` lists the names. Write only from the main thread
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
window, the span of its images plus `image_diagnostics.exact_image_window_margin_arcsec`, is ray-shot on a coarse grid
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
