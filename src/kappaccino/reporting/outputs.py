"""Tables and figures written after a run: posterior summary, image recovery, scaling relation, truth recovery."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..io import store
from ..plotting import plots
from ..plotting.chain_traces import write_chain_traces
from ..plotting.block_corner import write_posterior_block_corner, write_cosmology_corner
from ..config import LensClusterSolverConfig
from ..diagnostics.image_recovery import fit_summary, image_recovery_tables, image_recovery_with_reuse
from ..likelihood.evaluator import Evaluator
from ..core.model import CORED_POWER_LAW_PROFILE, ModelState, PosteriorResults
from ..plotting.potfile import write_potfile_plots
from ..plotting.members import write_member_diagnostic_plots
from ..diagnostics.truth_recovery import compute_truth_recovery
from ..plotting.aperture_mass_histogram import write_aperture_mass_histogram
from ..plotting.zeta_masses import write_zeta_masses
from .results_tex import write_run_results
from ..plotting.gp import write_gp_plots
from ..plotting.covariance import write_covariance_plots
from ..plotting.covariance_cutouts import write_covariance_cutouts
from ..plotting.fold_pairs import compute_fold_diagnostics, save_fold_data, render_fold_diagnostics
from .draws import reporting_draws


REBUILT_FIELD_WARNING = (
    "gp_blocks = {blocks}: the posterior gave each block its own mass field, and the products rebuild one shared field "
    "from it. That field is not a posterior sample of the lens, so the image residuals, RMS, position contours and map "
    "uncertainties in the draws' own lenses are not trustworthy and likely inflated. Use an all-block run for reported numbers.")


def posterior_summary_table(state: ModelState, posterior: PosteriorResults) -> pd.DataFrame:
    physical = np.asarray(posterior.physical, dtype=float)
    latent = np.asarray(posterior.samples, dtype=float)
    diagnostics = posterior.diagnostics or {}
    rows = []
    for index, spec in enumerate(state.parameter_specs):
        q = np.percentile(physical[:, index], [16, 50, 84])
        rows.append({"parameter": spec.name, "site": spec.sample_name, "family": spec.component_family, "field": spec.field,
                     "median": q[1], "p16": q[0], "p84": q[2], "mean": float(physical[:, index].mean()), "std": float(physical[:, index].std()),
                     "latent_median": float(np.median(latent[:, index])),
                     "r_hat": float(np.asarray(diagnostics.get("r_hat", np.full(latent.shape[1], np.nan)))[index]),
                     "n_eff": float(np.asarray(diagnostics.get("n_eff", np.full(latent.shape[1], np.nan)))[index])})
    return pd.DataFrame(rows)


def classification_table(state: ModelState) -> pd.DataFrame:
    rankings = (state.members.classification or {}).get("rankings", [])
    return pd.DataFrame([{"member_id": r["member"], "member_class": "exact" if r["exact"] else "far-field", **{k: v for k, v in r.items() if k not in {"member", "exact"}}}
                         for r in rankings])


def write_true_lens_rms(stage: Path, image_df: pd.DataFrame, deflection_fits, cosmo_config: dict, *, log=print) -> dict | None:
    """The true-lens RMS of the image catalog (``diagnostics.true_lens``): the per-image table ``true_lens_residuals``
    and the record ``true_lens_rms`` (rms, bootstrap band over families, amplitude, images left out) in results.h5.
    Report-only: an unreadable map or no usable image logs the reason and returns None."""
    from ..diagnostics.exact_images import astropy_cosmology
    from ..diagnostics.true_lens import true_lens_residuals, true_lens_rms
    try:
        table, amplitude = true_lens_residuals(image_df, deflection_fits, astropy_cosmology(cosmo_config))
        record = {**true_lens_rms(table), "amplitude": amplitude, "deflection_fits": str(deflection_fits)}
    except (OSError, KeyError, IndexError, ValueError) as error:
        log(f"[outputs] true-lens RMS skipped: {type(error).__name__}: {error}")
        return None
    with store.batch(stage):
        store.write_table(stage, "true_lens_residuals", table)
        store.write_record(stage, "true_lens_rms", record)
    left_out = ", ".join(f"{v} {k[2:].replace('_', ' ')}" for k, v in record.items()
                         if k.startswith("n_") and k not in ("n_images", "n_catalog_images", "n_families") and v)
    log(f"[outputs] true-lens RMS {record['rms_arcsec']:.3f}\" ({record['p16_arcsec']:.3f}-{record['p84_arcsec']:.3f}) on "
        f"{record['n_images']}/{record['n_catalog_images']} images of {record['n_families']} families, amplitude {amplitude:.4f}"
        f"; left out: {left_out or 'none'}")
    return record


def write_outputs(config: LensClusterSolverConfig, state: ModelState, evaluator: Evaluator, posterior: PosteriorResults,
                  audit: dict[str, Any], paths, *, log=print) -> dict[str, Any]:
    """``paths`` is the final StagePaths.  Every product is computed over one set of reporting draws
    (``reporting.draws.reporting_draws``), each draw evaluated once and shared by all of them; none uses the single best
    sample.

    A second pass over the same stage (a resumed run) reuses what the first one saved: the image-recovery tables from
    results.h5 when their fingerprint matches (``diagnostics.image_recovery.image_recovery_with_reuse``), the lens maps
    of every grid from cache.h5 (``ReportingDraws.grid_maps``), and the aperture masses; each reuse or recomputation is
    logged with its reason."""
    tables, figures, prefix = paths.tables, paths.figures, config.paths.figure_prefix
    stage = store.stage_dir(paths.tables)
    draws = reporting_draws(config, evaluator, posterior)
    draws.cache_stage = stage
    log(f"[outputs] {len(draws)} reporting draws (seed {draws.seed})")
    summary = {"reporting_draws": len(draws)}
    if draws.has_field and draws.field_is_rebuilt:
        summary["warnings"] = [REBUILT_FIELD_WARNING.format(blocks=state.gp_blocks)]
        log(f"[outputs] WARNING: {summary['warnings'][0]}")

    with store.batch(stage):
        store.write_table(stage, "posterior_summary", posterior_summary_table(state, posterior))
        store.write_table(stage, "member_classification", classification_table(state))
    write_chain_traces(posterior, state.parameter_specs, figures, tables, prefix, log=log)
    if audit:
        store.write_record(stage, "audit", audit, default=float)

    member_table = plots.member_scaling_table(state, posterior)
    store.write_table(stage, "member_scaling_relation", member_table)
    write_potfile_plots(posterior.physical, state.parameter_specs, figures, prefix, posterior=posterior, log=log)
    write_posterior_block_corner(posterior.physical, state.parameter_specs,
                                plots.figure_path(figures, "corner_combined.pdf", prefix))
    write_cosmology_corner(posterior.physical, state.parameter_specs,
                          plots.figure_path(figures, "corner_cosmology.pdf", prefix))
    write_covariance_plots(config, state, draws, posterior, paths, log=log)
    write_gp_plots(config, state, evaluator, posterior, paths, log=log)
    write_member_diagnostic_plots(config, state, evaluator, posterior, paths, log=log)
    if not member_table.empty:
        plots.plot_scaling_relation(member_table, np.asarray(posterior.physical), state.parameter_specs,
                                    plots.figure_path(figures, "scaling_relation.pdf", prefix),
                                    fixed=fixed_relation_values(state, member_table))

    recovery = image_recovery_with_reuse(stage, state, evaluator, draws, config.image_diagnostics,
                                         extra_images=config.reporting.extra_images, compute=image_recovery_tables, log=log)
    from ..diagnostics.position_contours import position_coverage
    recovery["position_coverage"] = position_coverage(recovery["posterior_image_residual_draws"])
    store.write_table(stage, "position_coverage", recovery["position_coverage"])
    image_df = recovery["image_fit_quality"]
    summary["fit"] = fit_summary(state, recovery, extra_image_draws=len(draws) if config.reporting.extra_images else 0)
    fit = summary["fit"]
    if "recovered_image_count" in fit:
        log(f"[outputs] recovered {fit['recovered_image_count']['median']:g}/{fit['observed_image_count']} images (median over "
            f"{len(draws.image_indices)} draws), rms {fit['image_rms_arcsec']['median']:.3f}\"")
    true_lens = None
    if config.truth.deflection_true_fits and not image_df.empty:
        true_lens = write_true_lens_rms(stage, image_df, config.truth.deflection_true_fits, state.cosmo_config, log=log)
    if not image_df.empty:
        plots.plot_image_residual_histogram(image_df, plots.figure_path(figures, "image_residual_histogram.pdf", prefix),
                                            draws=recovery.get("posterior_image_residual_draws"), true_lens=true_lens, log=log)
        plots.plot_image_recovery(image_df, recovery["image_recovery_extra_images"], plots.figure_path(figures, "image_recovery.pdf", prefix))

    from ..plotting.chain_traces import trace_data
    traces = trace_data(posterior, state.parameter_specs)
    corner_path = plots.plot_corner(np.asarray(posterior.physical), state.parameter_specs,
                                   plots.figure_path(figures, "corner_globals.pdf", prefix),
                                   grouped_physical=None if traces is None else traces['chains'])
    if corner_path is None:
        log("[outputs] corner skipped: fewer than two global parameters with posterior spread")

    z_crit = float(config.truth.caustic_source_redshift)
    plots.plot_critical_curves(state, draws, z_crit if z_crit > state.z_lens else max(state.z_source_values),
                               plots.figure_path(figures, "critical_curves.pdf", prefix), log=log)

    truth = compute_truth_recovery(state, draws, config.truth, paths.fits, tables, log=log)
    write_aperture_mass_histogram(config, state, draws, paths, image_df, log=log)
    write_zeta_masses(config, state, draws, paths, image_df, log=log)
    if truth is not None:
        from ..diagnostics.exact_images import astropy_cosmology
        from ..plotting.magnification_recovery import write_magnification_recovery
        write_magnification_recovery(truth, image_df, state.reference, state.z_lens,
                                     astropy_cosmology(state.cosmo_config), config.truth,
                                     figures, tables, prefix, log=log)
        from ..plotting.cumulative_source_area import write_cumulative_source_area
        write_cumulative_source_area(truth, draws, astropy_cosmology(state.cosmo_config), figures, tables, prefix, log=log)
        from ..plotting.scatter_summary import write_magnification_scatter
        write_magnification_scatter(truth, draws, image_df, tables, log=log)
        plots.plot_truth_recovery_residuals(truth, image_df, plots.figure_path(figures, "truth_recovery_residuals.pdf", prefix))
        if truth["profile"] is not None:
            plots.plot_m2d_aperture_ratio(truth["profile"], truth["center"], image_df, truth["mass_per_kappa_pixel_msun"],
                                          plots.figure_path(figures, "mass_profile.pdf", prefix),
                                          literature=truth["literature_m2d"])
            plots.plot_truth_ratio_panels(truth["profile"], truth["kappa_profile"], truth["center"], image_df,
                                          plots.figure_path(figures, "kappa_mass_residuals.pdf", prefix),
                                          literature_mass=truth["literature_m2d"], literature_kappa=truth["literature_kappa"])
            ratio = truth["profile"]
            summary["truth"] = {"m2d_ratio_at_max_radius": float(ratio["m2d_ratio_median"].iloc[-1]), "n_radii": int(len(ratio))}
    # The headline numbers and the results macros are written here and again at the end: the imaging products below
    # (overview, fold cutouts, RGB critical curves) are the slowest and most failure-prone stage, and the Ares run of
    # 2026-11-08 ended inside it with no run_summary.json and "not available" macros.
    store.write_record(stage, "run_summary", summary, default=float)
    write_run_results(paths.root, name=prefix)
    if config.image_catalog.image_dir and not image_df.empty:
        from ..imaging.cutouts import imaging_variants, load_cluster_imaging
        from ..plotting.mass_figure import render_cluster_mass_figure
        fold_data = None
        overview_configs = imaging_variants(config, overview_only=True)
        for image_config in imaging_variants(config):
            try:
                imaging = load_cluster_imaging(image_config, config.model, config.paths.run_name)
            except FileNotFoundError as error:
                # Missing external imaging must not prevent the other outputs.
                log(f"[outputs] cluster mass figure and covariance RGB cutouts skipped: {error}")
                continue
            write_covariance_cutouts(replace(config, image_catalog=image_config), state, draws, paths, imaging, log=log)
            from ..plotting.jwst_rgb import render_rgb_critical, rgb_overview_stem
            render_rgb_critical(replace(config, image_catalog=image_config), state, draws, image_df,
                recovery['image_recovery_extra_images'], imaging,
                plots.figure_path(figures, rgb_overview_stem(image_config, imaging[1]) + '.pdf', prefix), tables, log=log)
            overview_config = next((view for view in overview_configs
                                    if replace(view, figure_suffix=image_config.figure_suffix) == image_config), None)
            if overview_config is None:
                continue
            image_config = overview_config
            position_contours = None
            if image_config.figure_suffix == '_jwst':
                from ..diagnostics.position_contours import load_or_compute_contours
                position_contours = load_or_compute_contours(replace(config, image_catalog=image_config), state, draws,
                    recovery['image_recovery_extra_images'], tables, log=log)
                from ..plotting.small_lambda_jwst import write_small_lambda_jwst
                write_small_lambda_jwst(replace(config, image_catalog=image_config), state, draws,
                    paths, imaging, position_contours, log=log)
            render_cluster_mass_figure(draws, image_df, recovery["image_recovery_extra_images"], imaging,
                                       z_source=z_crit, kappa_true_fits=config.truth.kappa_true_fits,
                                       figure_path=plots.figure_path(figures, f"image_catalog_family_cluster_mass{image_config.figure_suffix}.pdf", prefix),
                                       tables_dir=tables, fits_dir=paths.fits, artifact_suffix=image_config.figure_suffix,
                                       position_contours=position_contours, log=log)
            log(f"[outputs] cluster mass figure rendered from {imaging[3]} imaging ({image_config.image_dir})")
            if fold_data is None:
                fold_data = compute_fold_diagnostics(config, state, draws, tables, log=log)
                save_fold_data(fold_data, tables)
            render_fold_diagnostics(replace(config, image_catalog=image_config), state, fold_data, paths, imaging, log=log)
            if image_config.figure_suffix == '_jwst':
                from ..plotting.apj_fold_cutouts import render_apj_fold_cutouts
                output = render_apj_fold_cutouts(config, fold_data, imaging,
                    plots.figure_path(figures, 'fold_cutouts_apj_jwst.pdf', prefix),
                    run_directory=paths.root.parent,
                    source_cache=f"{store.results_path(stage)} (arrays and record fold_pairs_source_caustics)")
                if output is not None:
                    log(f'[outputs] ApJ fold cutouts: {output}')
                from ..plotting.paper_cutouts import write_paper_cutouts
                write_paper_cutouts(replace(config, image_catalog=image_config), state, draws, tables, figures, imaging, log=log)
    store.write_record(stage, "run_summary", summary, default=float)
    write_run_results(paths.root, name=prefix)
    store.compact(stage)
    return summary


def fixed_relation_values(state, member_table) -> dict[str, float]:
    """``{"<population>.<field>": value}`` of the scaling-relation fields that are not sampled: the base values of the
    packed lens spec (``scaling.fixed_relation_fields``, or a slope a Lenstool file holds fixed)."""
    sampled = {spec.name for spec in state.parameter_specs}
    base = {"sigma": "sigma_ref_base", "cutkpc": "cut_ref_base", "alpha_sigma": "alpha_sigma_base", "beta_radius": "beta_radius_base"}
    members = np.asarray(state.members.component_indices, dtype=int)
    out = {}
    if members.size == 0:
        return out
    profile = np.asarray(state.packed_lens_spec.profile_type)
    for population in member_table["potfile_id"].unique():
        for field, attribute in base.items():
            if f"{population}.{field}" not in sampled:
                out[f"{population}.{field}"] = float(np.asarray(getattr(state.packed_lens_spec, attribute), dtype=float)[members[0]])
        # A cored power-law population with its 3D slope held fixed (scaling.fixed_relation_fields).
        component = [int(r["component_index"]) for r in state.scaling_component_records if r["potfile_id"] == population]
        if component and profile[component[0]] == CORED_POWER_LAW_PROFILE and f"{population}.slope" not in sampled:
            out[f"{population}.slope"] = float(np.asarray(state.packed_lens_spec.slope_base, dtype=float)[component[0]])
    return out
