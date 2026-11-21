"""The products drawn on the cluster imaging, in one loop over the imaging variants (``imaging.cutouts.imaging_variants``).

For every variant whose imaging loads: the covariance cutouts and the RGB critical overview.  For the overview
variants (``imaging_variants(config, overview_only=True)``) also the cluster mass figure and the fold diagnostics
(computed once, at the first overview), with first the position contours and the small-lambda figures for the JWST
overview (``figure_suffix == "_jwst"``) and after them its ApJ fold cutouts and paper cutouts.  The steps of a variant
are named ``<step>:<variant>``, the variant being its ``figure_suffix`` without the underscore (``primary`` without
one).  Compute evaluates the lens and saves what each figure draws; render draws the figures from the store and the
imaging pixels.  A variant whose imaging folder is missing in the compute is logged once and left out of both phases;
a variant the compute drew whose imaging cannot be loaded in the render fails its ``imaging`` render step.
"""
from __future__ import annotations

from dataclasses import replace

from .base import OutputContext, OutputModule


def overview_variant(image_config, overview_configs):
    """The overview variant drawn from the imaging of ``image_config`` (same imaging, possibly another
    ``figure_suffix``), or None when ``image_config`` has no overview."""
    return next((view for view in overview_configs if replace(view, figure_suffix=image_config.figure_suffix) == image_config), None)


def variant_label(image_config) -> str:
    """The name of an imaging variant in its step names: its ``figure_suffix`` without the underscore, ``primary``
    without one."""
    return image_config.figure_suffix.lstrip('_') or 'primary'


def fit_covariance_errors(stage, *, note):
    """The table ``covariance_errors_both_planes`` when the last compute of ``fit`` wrote it (its completeness
    record), else None, after ``note`` (``OutputContext.note``) is called with why: the covariance cutouts are then
    drawn without their posterior interval lines."""
    from ..io import store
    from .base import completion_name
    try:
        record = store.read_record(stage, completion_name("fit"))
    except store.MissingEntry:
        why = "fit was never computed here"
    else:
        step = record["steps"].get("covariance") if "steps" in record else None    # no steps: interrupted compute
        if step is not None and step["status"] == "done" and step["wrote"]:
            return store.read_table(stage, "covariance_errors_both_planes")
        why = ("the last compute of fit did not finish" if "steps" not in record else
               "its covariance step did not run" if step is None else
               f"its covariance step {step['status']}" if step["status"] != "done" else "its covariance step wrote nothing")
    note(f"no posterior interval lines in the covariance cutouts: no table covariance_errors_both_planes from the last "
         f"compute of fit ({why})")
    return None


def image_tables(ctx: OutputContext):
    """``(image_df, extra)``: the tables ``image_fit_quality`` and ``image_recovery_extra_images`` of the images
    product."""
    images = ctx.products["images"]
    return images["image_df"], images["recovery"]["image_recovery_extra_images"]


class RGBModule(OutputModule):
    """The covariance cutouts read the table ``covariance_errors_both_planes`` (``fit``) in compute, hence ``after``.

    Compute, per imaging variant whose imaging loads (``imaging``): the covariance cutouts
    (``covariance_cutouts.save_covariance_cutouts``) and the band of the RGB critical overview
    (``jwst_rgb.save_rgb_critical``); per overview variant the position contours
    (``position_contours.load_or_compute_contours``) and the small-lambda ranking (``save_small_lambda_jwst``) for the
    JWST overview, the cluster mass maps (``mass_figure.save_cluster_mass_figure``), the fold diagnostics once
    (``fold_pairs.compute_fold_diagnostics`` and ``save_fold_data``) and the paper-cutout critical fields
    (``paper_cutouts.prepare_paper_cutouts``).  Of the imaging the compute reads only the band files' WCS, shape and
    paths.  Render, per variant: every figure from the store and the imaging pixels (``imaging``, an input step: it
    draws nothing).  No product: no module reads it."""

    name = "rgb"
    requires = ("images",)
    after = ("fit",)
    checkpoint = True
    description = ("covariance RGB cutouts, RGB critical overview, position contours, small-lambda figures, cluster mass "
                   "figure, fold diagnostics, ApJ fold cutouts, paper cutouts (with imaging)")
    compute_steps = ("imaging", "covariance_cutouts", "rgb_critical", "position_contours", "small_lambda", "cluster_mass",
                     "fold_data", "paper_cutouts")
    render_steps = ("imaging", "covariance_cutouts", "rgb_critical", "small_lambda", "cluster_mass", "fold_cutouts",
                    "apj_fold_cutouts", "paper_cutouts")
    input_steps = ("imaging",)
    path_fields = ("rgb.image_dir", "rgb.candidate_catalog_path", "truth.kappa_true_fits")

    def active(self, ctx: OutputContext) -> str | None:
        if not ctx.settings.image_dir:
            return "no imaging (outputs.rgb.image_dir)"
        images = ctx.products.get("images")
        if images is not None and images["image_df"].empty:
            return "no observed image in the image recovery"
        return None

    def compute(self, ctx: OutputContext) -> None:
        from ..diagnostics.position_contours import load_or_compute_contours
        from ..imaging.cutouts import imaging_variants, variant_config
        from ..plotting.covariance_cutouts import save_covariance_cutouts
        from ..plotting.fold_pairs import compute_fold_diagnostics, save_fold_data
        from ..plotting.jwst_rgb import save_rgb_critical
        from ..plotting.mass_figure import save_cluster_mass_figure
        from ..plotting.paper_cutouts import prepare_paper_cutouts
        from ..plotting.small_lambda_jwst import save_small_lambda_jwst
        config, state, paths, tables, log = ctx.config, ctx.state, ctx.paths, ctx.tables, ctx.log
        image_df, extra = image_tables(ctx)
        z_crit = float(config.outputs.caustic_source_redshift)

        def fold_data():
            save_fold_data(compute_fold_diagnostics(config, state, ctx.draws, tables, log=log), tables)
            return "fold_pairs_source_caustics"

        overview_configs = imaging_variants(config, overview_only=True)
        folds_computed = False
        for image_config in imaging_variants(config):
            label = variant_label(image_config)
            imaging = ctx.step(f"imaging:{label}", ctx.imaging, image_config)
            if imaging is None:
                continue
            variant = variant_config(config, image_config)
            ctx.step(f"covariance_cutouts:{label}", lambda: save_covariance_cutouts(
                variant, state, ctx.draws, paths, imaging, intervals=fit_covariance_errors(ctx.stage, note=ctx.note), log=log))
            ctx.step(f"rgb_critical:{label}", lambda: save_rgb_critical(variant, state, ctx.draws, image_df, extra, imaging, tables, log=log))
            view = overview_variant(image_config, overview_configs)
            if view is None:
                continue
            label, variant, jwst = variant_label(view), variant_config(config, view), view.figure_suffix == '_jwst'
            position_contours, contours_ready, small_lambda = None, True, None
            if jwst:
                position_contours = ctx.step(f"position_contours:{label}", lambda: load_or_compute_contours(
                    variant, state, ctx.draws, extra, tables, log=log))
                contours_ready = ctx.ok(f"position_contours:{label}")
                if contours_ready:
                    small_lambda = ctx.step(f"small_lambda:{label}", lambda: save_small_lambda_jwst(
                        variant, state, ctx.draws, paths, imaging, position_contours, log=log))
                else:
                    ctx.skip_step(f"small_lambda:{label}", f"position_contours:{label} failed")
            if contours_ready:
                ctx.step(f"cluster_mass:{label}", lambda: save_cluster_mass_figure(
                    ctx.draws, image_df, extra, imaging, z_source=z_crit, kappa_true_fits=config.outputs.truth.kappa_true_fits,
                    tables_dir=tables, fits_dir=paths.fits, artifact_suffix=view.figure_suffix,
                    position_contours=position_contours, log=log))
            else:
                ctx.skip_step(f"cluster_mass:{label}", f"position_contours:{label} failed")
            if not folds_computed:                      # once, at the first overview
                folds_computed = True
                ctx.step("fold_data", fold_data)
            if not jwst:
                continue
            if not ctx.ok("fold_data"):
                ctx.skip_step(f"paper_cutouts:{label}", "fold_data failed")
            elif ctx.ok(f"small_lambda:{label}"):
                # The paper cutouts read the saved position contours and small-lambda ranking of this pass, and the
                # small-lambda critical fields only when this pass wrote them.
                ctx.step(f"paper_cutouts:{label}", lambda: prepare_paper_cutouts(
                    variant, tables, ctx.draws, small_lambda_critical=small_lambda is not None, log=log))
            else:
                ctx.skip_step(f"paper_cutouts:{label}", f"small_lambda:{label} did not run")

    def render(self, ctx: OutputContext) -> None:
        from functools import cache
        from ..diagnostics.position_contours import load_position_contours
        from ..imaging.cutouts import imaging_variants, variant_config
        from ..io import store
        from ..plotting import plots
        from ..plotting.apj_fold_cutouts import render_apj_fold_cutouts
        from ..plotting.covariance_cutouts import render_covariance_cutouts
        from ..plotting.fold_pairs import load_fold_data, render_fold_diagnostics
        from ..plotting.jwst_rgb import render_saved_rgb_critical, rgb_overview_stem
        from ..plotting.paper_cutouts import render_paper_cutouts
        from ..plotting.small_lambda_jwst import render_small_lambda_jwst
        config, state, paths, tables, figures, prefix, log = (ctx.config, ctx.state, ctx.paths, ctx.tables, ctx.figures,
                                                              ctx.prefix, ctx.log)
        image_df, extra = image_tables(ctx)
        contours = cache(lambda: load_position_contours(tables))        # read once, when a figure first needs them
        folds = cache(lambda: load_fold_data(tables))
        overview_configs = imaging_variants(config, overview_only=True)
        for image_config in imaging_variants(config):
            label = variant_label(image_config)
            # Left out when the compute had no imaging; a failed step when the compute drew it and it cannot be loaded.
            imaging = ctx.render_step(f"imaging:{label}", ctx.imaging, image_config, required=True)
            if imaging is None:
                continue
            variant = variant_config(config, image_config)
            ctx.render_step(f"covariance_cutouts:{label}", render_covariance_cutouts, variant, state, paths, imaging, log=log)
            ctx.render_step(f"rgb_critical:{label}", lambda: render_saved_rgb_critical(
                variant, state, image_df, extra, imaging,
                plots.figure_path(figures, rgb_overview_stem(image_config, imaging[1]) + '.pdf', prefix), tables, contours(), log=log))
            view = overview_variant(image_config, overview_configs)
            if view is None:
                continue
            label, variant, jwst = variant_label(view), variant_config(config, view), view.figure_suffix == '_jwst'
            if jwst:
                ctx.render_step(f"small_lambda:{label}", render_small_lambda_jwst, variant, state, paths, imaging, log=log)
            ctx.render_step(f"cluster_mass:{label}", lambda: self._cluster_mass(ctx, image_df, extra, imaging, view,
                                                                               contours() if jwst else None))
            ctx.render_step(f"fold_cutouts:{label}", lambda: render_fold_diagnostics(variant, state, folds(), paths, imaging, log=log),
                            needs="fold_data")
            if not jwst:
                continue
            output = ctx.render_step(f"apj_fold_cutouts:{label}", lambda: render_apj_fold_cutouts(
                config, folds(), imaging, plots.figure_path(figures, 'fold_cutouts_apj_jwst.pdf', prefix),
                run_directory=paths.root.parent,
                source_cache=f"{store.results_path(ctx.stage)} (arrays and record fold_pairs_source_caustics)"), needs="fold_data")
            if output is not None:
                log(f'[outputs] ApJ fold cutouts: {output}')
            ctx.render_step(f"paper_cutouts:{label}", render_paper_cutouts, variant, state, tables, figures, imaging, log=log)

    @staticmethod
    def _cluster_mass(ctx, image_df, extra, imaging, view, position_contours):
        from ..plotting import plots
        from ..plotting.mass_figure import draw_cluster_mass_figure
        draw_cluster_mass_figure(image_df, extra, imaging, reference=ctx.state.reference,
                                 figure_path=plots.figure_path(ctx.figures, f"image_catalog_family_cluster_mass{view.figure_suffix}.pdf", ctx.prefix),
                                 tables_dir=ctx.tables, artifact_suffix=view.figure_suffix, position_contours=position_contours)
        ctx.log(f"[outputs] cluster mass figure rendered from {imaging[3]} imaging ({view.image_dir})")
