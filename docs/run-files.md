# Priors and run files

A fit of a catalogued cluster is declared in two YAML files in its catalog folder, next to `images.yaml` and the
member catalog:

- `priors.yaml`: what is fitted. The reference point, cosmology, lens redshift, image catalog, cluster-scale halos,
  member populations and the priors of the member scaling relation.
- `run.yaml`: how it is fitted. The likelihood, search, sampler, member classification, reported products and
  figures, with named profiles layered on top.

{func}`lumina.run_files.load_run` reads both into a {class}`~lumina.config.LensClusterSolverConfig`. The
`lumina` command (or `python -m lumina`, {mod}`lumina.cli`) runs a folder from the repository root:

```bash
lumina catalogs/hera --dry-run
lumina catalogs/m0416/canucs2025 --profile fast --profile jwst
lumina catalogs/as1063/bergamini2019 --set sampling.warmup=1000 --set runtime.skip_plots=true
lumina catalogs/hera --set outputs.rgb.enabled=false
lumina resume results/hera/nov21a_w500s250t7_allblocks
lumina outputs results/hera/nov21a_w500s250t7_allblocks --only truth
```

`lumina resume RUN_DIR` continues a run folder with its own saved settings, and `lumina outputs RUN_DIR` reruns the
outputs stage of a finished run without fitting (see {doc}`outputs`). A cluster folder that is literally named
`resume` or `outputs` must be given as `./resume` or `./outputs`.

## Run names

Every run command starts a new run folder, `results/<name>/<mon><day><letter>_<description>`, for example
`results/hera/nov21a_w500s250t7_allblocks`. Running the same command again makes a second run,
`results/hera/nov21b_w500s250t7_allblocks`. `<mon><day>` is the local date the folder is created (lowercase
three-letter month, day without a leading zero) and `<letter>` the next letter of that cluster and date (`a` to `z`,
then `aa`, `ab`, ...), counting every run of the date whatever its description. The description gives the NUTS warmup,
samples and tree depth, the GP blocks (or the likelihood when it is not the GRF), the image catalog when `--images`
replaces it, `uncertain` with uncertain images, `powerlaw` with power-law members, and the profiles. The name is given
once and never changes.

A letter is reserved by creating `results/<name>/.names/<mon><day><letter>` (an atomic directory creation), so fits
launched together get different letters, and a letter once given is not given again, even when its run folder is
deleted. Folders named like an assigned name (`<mon><day>`, one or two letters, `_` and a description) take their
letter too; a label made by hand such as `nov21final` takes none. Names carry no year, so on the same date of a later
year the letters go on after those already given.

`--run-label NAME` names the new folder by hand. A folder that already exists is refused (exit code 2) with the
`lumina resume` command that continues it. `--results DIR` moves the tree to `DIR/<name>/...`. `--dry-run` prints the
folder it would create and creates nothing, not even a reserved letter. The files the run read and the command are
copied to the run's `inputs/` folder.

The command prints the run folder when it starts and, when it ends, the command to run next:

```bash
lumina catalogs/hera --stage init
# [run] init stage done in results/hera/nov21a_w500s250t7_allblocks; run its final stage with:
lumina resume results/hera/nov21a_w500s250t7_allblocks --stage final
```

A run that stops early (an error, Ctrl-C) prints `lumina resume RUN_DIR`, and a finished run the `lumina resume` and
`lumina outputs` commands that rerun its outputs. `--stage final` on the run command is refused, since a new folder
has no init stage to start from.

## Resuming a run folder

`lumina resume RUN_DIR` is the only way back into an existing run. It continues the run with its own saved
configuration (`final/artifacts/config.json` once the final stage started, else `init/artifacts/config.json`), so it
works whatever the defaults or the run files say now:

```bash
lumina resume results/hera/nov21a_w500s250t7_allblocks                 # whatever stage is unfinished
lumina resume results/hera/nov21a_w500s250t7_allblocks --stage final   # the final stage after --stage init
lumina resume results/hera/nov21a_w500s250t7_allblocks --draws 64      # a finished run: its outputs again
```

A run whose init stage is complete and whose final stage never started runs its final stage. A run stopped during
sampling samples again from the saved chain start, since sampling has no checkpoint. A run stopped in the init stage
runs it again from the start. A finished run reloads its posterior and reruns the outputs stage; `lumina outputs
RUN_DIR` reruns the outputs alone and saves no settings (see {doc}`outputs`). `--dry-run` prints the state of the run
and what would run.

`--set` and `--draws` change only the `outputs` section, checked as for `lumina outputs`; any fit setting is refused.
Unlike `lumina outputs`, a resumed final stage saves the outputs settings it used into `final/artifacts/config.json`.
`--remap OLD=NEW` relocates the saved paths of a run made on another machine, as for `lumina outputs`, for that command
only: it is never saved. Every file the run writes (`config.json`, `init_signature.json`) keeps the paths as first
saved, and the saved stages are compared in those terms, so later commands treat the folder the same with or without
the remap. The outputs records of `final/results.h5` are the exception: the outputs stage of a remapped resume records
the remapped truth-map paths it read (in `output_module_mass`, `output_module_truth`, `run_summary` and
`magnification_recovery_provenance`), as `lumina outputs --remap` does, so a later pass without the remap warns that the
outputs settings changed; nothing else follows from it. A remap that moves no saved path is a warning. The run writes
where its folder is now, so a run folder can be moved or renamed.

Since the fit settings come from the saved configuration, a run folder holds one fit. A second final stage from the
same init stage, keeping the first, goes in a copy of the folder without its `final/`:

```bash
cp -R results/hera/nov21a_w500s250t7_allblocks results/hera/nov21a_w500s250t7_allblocks_second
rm -rf results/hera/nov21a_w500s250t7_allblocks_second/final
lumina resume results/hera/nov21a_w500s250t7_allblocks_second --stage final
```

The copy samples with the sampling settings of the init stage's saved configuration, those of the command that made
the run, so it repeats the first final stage with the same seed. No command samples a saved init stage with other
sampling settings (more draws, another tree depth or seed): that is a new run, which runs its init stage again.
Turning on WBIC for a finished run is the same: the command line has no route to it, since `lumina resume --set`
changes only `outputs` (see {doc}`model-comparison` for the Python route).

## Locks

A process working in a run folder holds its lock, the file `run.lock` in that folder: `lumina <cluster>`, `lumina
resume` and a `lumina outputs` pass that computes, from start to end. The file is written whole and published
atomically (a temporary file linked into place), and holds the host name, the process id, the time it was taken and
the command; it is removed when the command ends, also after an error or Ctrl-C. `lumina resume` and `lumina outputs`
refuse a run locked by a running process with exit code 2, and name the host, the pid and since when. `lumina outputs
--render` takes no lock and proceeds, since it only reads results already committed and draws figures. While a fit is
writing the same run, a render pass can report a module as not computed yet (its compute has not committed), wait
behind the fit for the store, or write the same PDF as the fit's outputs stage (the last one written stays).

Reserving a new name and creating and locking its folder happen under a second, short-lived lock,
`results/<name>/.names.lock`, which the command holds for that only (never while it fits) and waits for up to two
minutes before it gives up with the holder's name; the wait is logged when it starts, with the holder, and every 30
seconds.

A lock whose process is gone (stopped with SIGTERM or SIGKILL, a crash, a reboot) is stale when it names this host
(the same short host name, the part before the first dot, in any case, so `mac` and `mac.local` are one host) and that
process no longer runs: the next command takes it over and logs one line. A lock written on another host cannot be
checked (the results tree may be mirrored between machines by a file-sync tool, and lock files travel with it), so it
counts as held; the message names the file to delete when that process is known to be gone. An empty or cut-short lock
file (which this code never writes) is reported as an incomplete lock file, with the file to delete. A process id
reused by an unrelated process makes a stale lock look held: delete the file then. A copy of a run folder made while a
command works in it carries that command's `run.lock`; delete it in the copy.

Locks and letters are per machine. Two commands started on two machines that mirror the same results tree do not see
each other's lock or reserved letter until the sync tool copies them, so within that delay they can reserve the same
name; start the runs of one cluster on one machine.

## Options

`--images YAML` fits another image catalog in place of the one `priors.yaml` names (a degraded catalog, say).
Images of unknown family are a `partition` section: `--set partition.catalog_path=uncertain.csv`, or
`partition.catalog_path=null` for the untrusted images of the image catalog, with
`partition.redshift_grid=[z_min,z_max,points]` for their redshift quadrature.

The common fields have named options, applied after the profiles and before `--set`: `--chains`, `--seed`,
`--starts`, `--warm-start RUN_DIR`, `--warmup`, `--samples`, `--tree-depth`, `--target-accept`, `--draws`
(`outputs.draws` and `outputs.image_draws`) and `--skip-plots`. `lumina --help` lists them with the field each sets.

`--member-profile power-law` fits the members as cored power laws, the dPIE with a free 3D slope (one sampled slope
per population), and `--member-slope G` holds that slope at G. In `priors.yaml` the same is `profile_type: 102` on a
member population, with `slope` or `slope_prior`.

| folder | name | profiles |
|---|---|---|
| `catalogs/hera` | hera | fast, cosmography |
| `catalogs/ares` | ares | fast, cosmography |
| `catalogs/tngc014/small` | tngc014_small | fast, cosmography |
| `catalogs/tngc014/medium` | tngc014_medium | fast, cosmography |
| `catalogs/tngc014/large` | tngc014 | fast, cosmography |
| `catalogs/tngc014_dm/small` | tngc014_dm_small | fast, cosmography |
| `catalogs/tngc014_dm/medium` | tngc014_dm_medium | fast, cosmography |
| `catalogs/tngc014_dm/large` | tngc014_dm | fast, cosmography |
| `catalogs/as1063/bergamini2019` | as1063 | fast, jwst, cosmography |
| `catalogs/as1063/diego2026` | as1063_jwst | fast, hst, cosmography |
| `catalogs/m0416/bergamini2023` | m0416_bergamini2023 | fast, jwst, cosmography |
| `catalogs/m0416/canucs2025` | m0416_canucs2025 | fast, jwst, cosmography |

## Values

A number is fixed. A string of arithmetic is evaluated, with `inf` and, in `priors.yaml`, `arcsec` (the kpc per arcsec
at the lens redshift): `2.3 / 0.72`, `13.5871 * arcsec`. A string that calls a prior function is a prior on the
physical value:

| prior | sampled in |
|---|---|
| `uniform(low, high)` | the value |
| `loguniform(low, high)` | ln of the value |
| `normal(loc, scale)`, `normal(loc, scale) in [low, high]` | the value |
| `student_t(df, loc, scale)` | the value |
| `lognormal(median, width)`, `lognormal(mean=m, std=s)`, either `in [low, high]` | ln of the value |

`width` is the standard deviation of ln of the value; `mean` and `std` are the moments of the untruncated
log-normal. `inf` leaves a side of `in [low, high]` open. Any other string stays a string. A prior's commas and
brackets end a YAML flow mapping (`{...}`), so give a parameter with a prior its own line.

Paths are relative to the file that names them. A file may start with `extends: <file>`: its mappings merge key by key
into that file's, and every other value replaces the base's. `catalogs/m0416/canucs2025/run.yaml` extends the Bergamini
run and changes only its name, priors and candidate catalog. Unknown keys raise.

## priors.yaml

```yaml
reference: {ra0_deg: 64.03821668520617, dec0_deg: -24.06750122335985}
cosmology: {H0: 70.0, Om0: 0.3, Ode0: 0.7}
z_lens: 0.396
images: images.yaml

halos:
  "1":
    x_centre: normal(0.2, 1.5) in [-5.8, 6.2]
    y_centre: normal(0.0, 1.5) in [-6.0, 6.0]
    shape: {ellipticity: [0.2, 0.94], angle: [100.0, 180.0], soft_cap: 0.9}
    core_radius_kpc: loguniform(10.0, 100.0)
    cut_radius_kpc: 2000.0 * arcsec
    v_disp:
      start: 650.0
      prior: lognormal(mean=650.0, std=150.0) in [400.0, 1000.0]
  shear: {profile: shear, preset: free, gamma_upper: 0.3}

members:
  potfile_1:
    catalog_path: members.yaml
    preset: neutral
    mag0: 18.66
    independent:
      "2526":
        id: bcg_2526
        x_centre: fixed
        v_disp: lognormal(mean=250.0, std=50.0) in [100.0, 400.0]

scaling:
  preset: neutral
  alpha_sigma_prior: normal(0.25, 0.05) in [0.1, 0.4]
```

`reference` and `cosmology` take the fields of {class}`~lumina.config.ReferenceFrameConfig` and
{class}`~lumina.config.CosmologyConfig`. `z_lens` is the redshift of every halo and population.

`halos` are keyed by id. `profile` is `dpie` (default), `nfw`, `shear` or `sheet`. `preset` builds the component with a
constructor of its class, {meth}`~lumina.config.DPIEHaloConfig.cluster_scale` or
{meth}`~lumina.config.ShearHaloConfig.free`, from the keys it takes; a `cluster_scale` halo takes its softening
from the centre of `scaling.softening_length_prior`. Every other key is a parameter of the component, named as in its
class. A number fixes it. A prior samples it from the prior's centre; `start` and `prior` sample it from `start`.
`e1`, `e2`, `cut_excess_kpc`, `gamma1` and `gamma2` take priors only. `shape` samples the shape uniformly in the
(e1, e2) box of {func}`~lumina.priors.ellipticity_box` over the `ellipticity` (Lenstool `ellipticite`) and
`angle` ranges, starting from the middle of the box, below `soft_cap` when one is given, unless `ellipticite` and
`angle_pos` are set.

`members` are keyed by population id: `catalog_path`, an optional `preset` (`bergamini2019` or `neutral`, see
{class}`~lumina.config.MemberPopulationConfig`) with the keys it takes, and any other field of the population.
`independent` takes members off the relation, keyed by catalog id; each entry's priors replace the defaults of
{class}`~lumina.config.IndependentMemberHaloConfig`, and `fixed` holds a parameter at its start.

`scaling` takes an optional `preset` (`bergamini2019` or `neutral`) and the fields of
{class}`~lumina.config.ScalingModelConfig`.

(cosmography)=
### Cosmography

A prior on `Om0` or `w0` under `cosmology` samples that parameter of the flat wCDM cosmology with the lens:

```yaml
cosmology:
  H0: 70.0
  Om0:
    start: 0.3
    prior: uniform(0.05, 0.95)
  w0:
    start: -1.0
    prior: uniform(-2.0, -0.2)
```

The multiple images measure the ratios of $D_{ls}/D_s$ between families at different source redshifts, which depend
on $\Omega_m$ and $w$ and not on $H_0$, so `H0` stays a number. `Ode0` must be absent or null (the cosmology is flat).
At every likelihood evaluation the fit computes $D_{ls}/D_s$ of each source redshift, the kpc per arcsec at the lens
and the NFW critical density from the sampled values; the distances are a 64-point Gauss–Legendre integral that agrees
with astropy to $10^{-11}$ (`tests/test_cosmology.py`). The parameters are `cosmology.Om0` and `cosmology.w0`, in the
block of the cluster-scale halos: the search draws them from their priors and optimizes them with the halos.

Each cluster folder, the TNG mocks included, has a `priors_cosmography.yaml` that extends its `priors.yaml` with these priors, and a
`cosmography` profile that selects it: `lumina catalogs/hera --profile cosmography` (Hera was simulated with
$\Omega_m = 0.24$, Ares with 0.272 and the TNG mocks were built with 0.3, all with $w = -1$).

Three things stay at the start values: the redshift weights of the GRF and of the image-plane GP (they change by
less than 2.5% over these priors), the member classification, and the quantities reported in physical units (masses,
kpc). A family with a photometric redshift sits at its catalog redshift, so cosmography should use spectroscopic
families. Uncertain images (`partition`) are not supported with a sampled cosmology.

## run.yaml

```yaml
name: hera
priors: priors.yaml
likelihood:
  gp_blocks: all
  lambda_b_prior: lognormal(50.0, 1.0) in [0.5, 200.0]
sampling: {warmup: 500, samples: 250, max_tree_depth: 7, target_accept: 0.8}
outputs:
  draws: 64
  image_draws: 64
  fit: {enabled: true}
  images: {enabled: true, extra_images: true}
  mass: {enabled: true}
  truth:
    enabled: true
    kappa_true_fits: ../../data/imaging/hera/ffsims/kappa_z9_0.fits
  rgb:
    enabled: true
    image_dir: ../../data/imaging/hera/ffsims
profiles:
  fast:
    likelihood: {gp_blocks: system, gp_knee_project_modes: []}
    outputs: {draws: 8, image_draws: 8, images: {extra_images: false}}
```

`name` sets the figure prefix and the output folder, and `priors` names the priors file. Every other top-level key is a
section of {class}`~lumina.config.LensClusterSolverConfig` with its field names; `scaling` here changes what
`priors.yaml` set. `outputs` lists what the run produces, one section per output module with its `enabled` switch and
its settings (see {doc}`outputs`); paths in it (`outputs.truth.*_true_fits`, `outputs.rgb.image_dir`,
`outputs.rgb.candidate_catalog_path`) are relative to the file's folder, as `partition.catalog_path` is. `profiles` are overlays of the same sections, applied in the order given on the command line, and
`--set section.field=value` overrides come last. The defaults of every field are the library defaults.

The fast profile of each cluster is exploratory: far-field members down to 2 cut radii, classified at the current
model with 3x headroom on the cut radius, system GP blocks, member shapes fixed at the catalog, tree depth 8, 8
reporting draws and no extra-image search.
