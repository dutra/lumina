"""The products drawn on the cluster imaging, in one loop over the imaging variants (``imaging.cutouts.imaging_variants``).

For every variant whose imaging loads: the covariance cutouts and the RGB critical overview.  For the overview
variants (``imaging_variants(config, overview_only=True)``) also the cluster mass figure and the fold diagnostics
(computed once, at the first overview), with first the position contours and the small-lambda figures for the JWST
overview (``figure_suffix == "_jwst"``) and after them its ApJ fold cutouts and paper cutouts.  The steps of a variant
are named ``<step>:<variant>``, the variant being its ``figure_suffix`` without the underscore (``primary`` without
one).
"""
from __future__ import annotations

from dataclasses import replace

from .base import OutputContext, OutputModule


def overview_variant(image_config, overview_configs):
    """The overview variant drawn from the imaging of ``image_config`` (same imaging, possibly another
    ``figure_suffix``), or None when ``image_config`` has no overview."""
    return next((view for view in overview_configs if replace(view, figure_suffix=image_config.figure_suffix) == image_config), None)


class RGBModule(OutputModule):
    """The covariance cutouts read the table ``covariance_errors_both_planes`` (``fit``), hence ``after``.  Product:
    ``{"position_contours": {figure_suffix: contours}, "fold_data": the fold diagnostics or None}``."""

    name = "rgb"
    requires = ("images",)
    after = ("fit",)
    checkpoint = True
    description = ("covariance RGB cutouts, RGB critical overview, position contours, small-lambda figures, cluster mass "
                   "figure, fold diagnostics, ApJ fold cutouts, paper cutouts (with imaging)")
    steps = ("imaging", "covariance_cutouts", "rgb_critical", "position_contours", "small_lambda", "cluster_mass", "fold_data",
             "fold_cutouts", "apj_fold_cutouts", "paper_cutouts")
    path_fields = ("image_catalog.image_dir", "image_catalog.candidate_catalog_path", "truth.kappa_true_fits")

    def active(self, ctx: OutputContext) -> str | None:
        if not ctx.config.image_catalog.image_dir:
            return "no imaging (image_catalog.image_dir)"
        images = ctx.products.get("images")
        if images is not None and images["image_df"].empty:
            return "no observed image in the image recovery"
        return None

    def run(self, ctx: OutputContext):
        from ..diagnostics.position_contours import load_or_compute_contours
        from ..imaging.cutouts import imaging_variants
        from ..io import store
        from ..plotting import plots
        from ..plotting.apj_fold_cutouts import render_apj_fold_cutouts
        from ..plotting.covariance_cutouts import write_covariance_cutouts
        from ..plotting.fold_pairs import compute_fold_diagnostics, render_fold_diagnostics, save_fold_data
        from ..plotting.jwst_rgb import render_rgb_critical, rgb_overview_stem
        from ..plotting.mass_figure import render_cluster_mass_figure
        from ..plotting.paper_cutouts import write_paper_cutouts
        from ..plotting.small_lambda_jwst import write_small_lambda_jwst
        config, state, draws, paths, figures, tables, prefix, log = (ctx.config, ctx.state, ctx.draws, ctx.paths, ctx.figures,
                                                                     ctx.tables, ctx.prefix, ctx.log)
        images = ctx.products["images"]
        image_df, extra = images["image_df"], images["recovery"]["image_recovery_extra_images"]
        z_crit = float(config.truth.caustic_source_redshift)
        fold_data, contours_by_view = None, {}
        overview_configs = imaging_variants(config, overview_only=True)
        for image_config in imaging_variants(config):
            label = image_config.figure_suffix.lstrip('_') or 'primary'
            imaging = ctx.step(f"imaging:{label}", ctx.imaging, image_config)
            if imaging is None:
                continue
            ctx.step(f"covariance_cutouts:{label}", write_covariance_cutouts, replace(config, image_catalog=image_config), state,
                     draws, paths, imaging, log=log)
            ctx.step(f"rgb_critical:{label}", lambda: render_rgb_critical(
                replace(config, image_catalog=image_config), state, draws, image_df, extra, imaging,
                plots.figure_path(figures, rgb_overview_stem(image_config, imaging[1]) + '.pdf', prefix), tables, log=log))
            view = overview_variant(image_config, overview_configs)
            if view is None:
                continue
            image_config = view
            label = image_config.figure_suffix.lstrip('_') or 'primary'
            position_contours, contours_ready = None, True
            if image_config.figure_suffix == '_jwst':
                position_contours = ctx.step(f"position_contours:{label}", load_or_compute_contours,
                                             replace(config, image_catalog=image_config), state, draws, extra, tables, log=log)
                contours_ready = ctx.ok(f"position_contours:{label}")
                if contours_ready:
                    contours_by_view[image_config.figure_suffix] = position_contours
                    ctx.step(f"small_lambda:{label}", write_small_lambda_jwst, replace(config, image_catalog=image_config), state,
                             draws, paths, imaging, position_contours, log=log)
                else:
                    ctx.skip_step(f"small_lambda:{label}", f"position_contours:{label} failed")
            if contours_ready:
                ctx.step(f"cluster_mass:{label}", self._cluster_mass, ctx, render_cluster_mass_figure, image_df, extra, imaging,
                         image_config, z_crit, position_contours)
            else:
                ctx.skip_step(f"cluster_mass:{label}", f"position_contours:{label} failed")
            if fold_data is None and "fold_data" not in ctx._steps:
                fold_data = ctx.step("fold_data", self._fold_data, ctx, compute_fold_diagnostics, save_fold_data)
            if not ctx.ok("fold_data"):
                for step in ("fold_cutouts", *(("apj_fold_cutouts", "paper_cutouts") if image_config.figure_suffix == '_jwst' else ())):
                    ctx.skip_step(f"{step}:{label}", "fold_data failed")
                continue
            ctx.step(f"fold_cutouts:{label}", render_fold_diagnostics, replace(config, image_catalog=image_config), state, fold_data,
                     paths, imaging, log=log)
            if image_config.figure_suffix == '_jwst':
                output = ctx.step(f"apj_fold_cutouts:{label}", render_apj_fold_cutouts, config, fold_data, imaging,
                                  plots.figure_path(figures, 'fold_cutouts_apj_jwst.pdf', prefix),
                                  run_directory=paths.root.parent,
                                  source_cache=f"{store.results_path(ctx.stage)} (arrays and record fold_pairs_source_caustics)")
                if output is not None:
                    log(f'[outputs] ApJ fold cutouts: {output}')
                # The paper cutouts read the saved position contours and small-lambda ranking of this pass.
                if ctx.ok(f"small_lambda:{label}"):
                    ctx.step(f"paper_cutouts:{label}", write_paper_cutouts, replace(config, image_catalog=image_config), state, draws,
                             tables, figures, imaging, log=log)
                else:
                    ctx.skip_step(f"paper_cutouts:{label}", f"small_lambda:{label} did not run")
        return {"position_contours": contours_by_view, "fold_data": fold_data}

    @staticmethod
    def _cluster_mass(ctx, render, image_df, extra, imaging, view, z_crit, position_contours):
        from ..plotting import plots
        config, paths = ctx.config, ctx.paths
        render(ctx.draws, image_df, extra, imaging, z_source=z_crit, kappa_true_fits=config.truth.kappa_true_fits,
               figure_path=plots.figure_path(ctx.figures, f"image_catalog_family_cluster_mass{view.figure_suffix}.pdf", ctx.prefix),
               tables_dir=ctx.tables, fits_dir=paths.fits, artifact_suffix=view.figure_suffix,
               position_contours=position_contours, log=ctx.log)
        ctx.log(f"[outputs] cluster mass figure rendered from {imaging[3]} imaging ({view.image_dir})")

    @staticmethod
    def _fold_data(ctx, compute, save):
        data = compute(ctx.config, ctx.state, ctx.draws, ctx.tables, log=ctx.log)
        save(data, ctx.tables)
        return data
