"""The comparison with the true lens of a simulated cluster (``truth.kappa_true_fits``)."""
from __future__ import annotations

from .base import ModuleUnavailable, OutputContext, OutputModule


def _once(load):
    """``load`` called at most once: a render step reads the saved truth recovery when it first needs it."""
    cached = []

    def get():
        if not cached:
            cached.append(load())
        return cached[0]
    return get


class TruthModule(OutputModule):
    """Compute: the truth maps (``diagnostics.truth_recovery.compute_truth_recovery``: FITS, profile tables, the record
    ``truth_recovery`` and the arrays ``truth_recovery_residuals``), the magnification recovery, the cumulative source
    area, the magnification scatter and ``truth`` in ``run_summary``; raises ModuleUnavailable when the truth_maps
    step failed.  Render: every figure, from the store (``truth_recovery.load_truth_recovery`` and the tables of each
    figure).  No product: no module reads it."""

    name = "truth"
    requires = ("images",)
    description = "truth-recovery maps and profiles and every figure made from them (with a truth convergence map)"
    compute_steps = ("truth_maps", "magnification_recovery", "cumulative_source_area", "magnification_scatter")
    render_steps = ("magnification_recovery", "cumulative_source_area", "truth_residuals", "mass_profile", "kappa_mass_residuals")
    summary_keys = ("truth",)
    path_fields = ("truth.kappa_true_fits", "truth.gammax_true_fits", "truth.gammay_true_fits")

    def active(self, ctx: OutputContext) -> str | None:
        return None if ctx.config.truth.kappa_true_fits else "no truth convergence map (truth.kappa_true_fits)"

    def compute(self, ctx: OutputContext) -> None:
        from ..diagnostics.exact_images import astropy_cosmology
        from ..diagnostics.truth_recovery import compute_truth_recovery
        from ..plotting.cumulative_source_area import save_cumulative_source_area
        from ..plotting.magnification_recovery import save_magnification_recovery
        from ..plotting.scatter_summary import write_magnification_scatter
        from .images import stored_image_table
        config, state, tables, log = ctx.config, ctx.state, ctx.tables, ctx.log
        truth = ctx.step("truth_maps", lambda: compute_truth_recovery(state, ctx.draws, config.truth, ctx.paths.fits, tables, log=log))
        if not ctx.ok("truth_maps"):
            for name in self.compute_steps[1:]:
                ctx.skip_step(name, "truth_maps failed")
            raise ModuleUnavailable("no truth maps (the truth_maps step failed)")
        if truth is None:
            return
        image_df = stored_image_table(ctx)          # TRANSITIONAL: ctx.products["images"] once images is converted
        cosmology = astropy_cosmology(state.cosmo_config)
        ctx.step("magnification_recovery", save_magnification_recovery, truth, image_df, state.reference, state.z_lens,
                 cosmology, config.truth, tables, log=log)
        ctx.step("cumulative_source_area", lambda: save_cumulative_source_area(truth, ctx.draws, cosmology, tables, log=log))
        ctx.step("magnification_scatter", lambda: write_magnification_scatter(truth, ctx.draws, image_df, tables, log=log))
        if truth["profile"] is not None:
            ratio = truth["profile"]
            ctx.summary["truth"] = {"m2d_ratio_at_max_radius": float(ratio["m2d_ratio_median"].iloc[-1]), "n_radii": int(len(ratio))}

    def render(self, ctx: OutputContext) -> None:
        from ..diagnostics.truth_recovery import load_truth_recovery
        from ..plotting import plots
        from ..plotting.cumulative_source_area import render_cumulative_source_area
        from ..plotting.magnification_recovery import render_magnification_recovery
        from .images import stored_image_table
        if not ctx.computed("truth_maps"):          # the source redshift is not behind the lens: nothing was computed
            return
        figures, prefix, stage = ctx.figures, ctx.prefix, ctx.stage
        truth = _once(lambda: load_truth_recovery(stage))
        image_df = _once(lambda: stored_image_table(ctx))      # TRANSITIONAL: the images product once converted
        ctx.render_step("magnification_recovery", lambda: render_magnification_recovery(
            stage, figures / f'{prefix + "_" if prefix else ""}magnification_recovery.pdf', log=ctx.log))
        ctx.render_step("cumulative_source_area", lambda: render_cumulative_source_area(
            stage, figures / f'{prefix + "_" if prefix else ""}cumulative_source_area.pdf', log=ctx.log))
        ctx.render_step("truth_residuals", lambda: plots.plot_truth_recovery_residuals(
            truth(), image_df(), plots.figure_path(figures, "truth_recovery_residuals.pdf", prefix)), needs="truth_maps")
        ctx.render_step("mass_profile", lambda: None if truth()["profile"] is None else plots.plot_m2d_aperture_ratio(
            truth()["profile"], truth()["center"], image_df(), truth()["mass_per_kappa_pixel_msun"],
            plots.figure_path(figures, "mass_profile.pdf", prefix), literature=truth()["literature_m2d"]), needs="truth_maps")
        ctx.render_step("kappa_mass_residuals", lambda: None if truth()["profile"] is None else plots.plot_truth_ratio_panels(
            truth()["profile"], truth()["kappa_profile"], truth()["center"], image_df(),
            plots.figure_path(figures, "kappa_mass_residuals.pdf", prefix), literature_mass=truth()["literature_m2d"],
            literature_kappa=truth()["literature_kappa"]), needs="truth_maps")
