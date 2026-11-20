"""The comparison with the true lens of a simulated cluster (``truth.kappa_true_fits``)."""
from __future__ import annotations

from .base import ModuleUnavailable, OutputContext, OutputModule


class TruthModule(OutputModule):
    """Product: the result of ``diagnostics.truth_recovery.compute_truth_recovery`` (None when the source redshift
    is not behind the lens, which it logs).  Raises ModuleUnavailable when the truth_maps step failed."""

    name = "truth"
    requires = ("images",)
    description = "truth-recovery maps and profiles and every figure made from them (with a truth convergence map)"
    steps = ("truth_maps", "magnification_recovery", "cumulative_source_area", "magnification_scatter", "truth_residuals",
             "mass_profile", "kappa_mass_residuals")
    summary_keys = ("truth",)
    path_fields = ("truth.kappa_true_fits", "truth.gammax_true_fits", "truth.gammay_true_fits")

    def active(self, ctx: OutputContext) -> str | None:
        return None if ctx.config.truth.kappa_true_fits else "no truth convergence map (truth.kappa_true_fits)"

    def run(self, ctx: OutputContext):
        from ..diagnostics.exact_images import astropy_cosmology
        from ..diagnostics.truth_recovery import compute_truth_recovery
        from ..plotting import plots
        from ..plotting.cumulative_source_area import write_cumulative_source_area
        from ..plotting.magnification_recovery import write_magnification_recovery
        from ..plotting.scatter_summary import write_magnification_scatter
        config, state, figures, tables, prefix, log = ctx.config, ctx.state, ctx.figures, ctx.tables, ctx.prefix, ctx.log
        truth = ctx.step("truth_maps", lambda: compute_truth_recovery(state, ctx.draws, config.truth, ctx.paths.fits, tables, log=log))
        if not ctx.ok("truth_maps"):
            for name in self.steps[1:]:
                ctx.skip_step(name, "truth_maps failed")
            raise ModuleUnavailable("no truth maps (the truth_maps step failed)")
        if truth is None:
            return None
        image_df = ctx.products["images"]["image_df"]
        cosmology = astropy_cosmology(state.cosmo_config)
        ctx.step("magnification_recovery", write_magnification_recovery, truth, image_df, state.reference, state.z_lens,
                 cosmology, config.truth, figures, tables, prefix, log=log)
        ctx.step("cumulative_source_area", write_cumulative_source_area, truth, ctx.draws, cosmology, figures, tables, prefix, log=log)
        ctx.step("magnification_scatter", write_magnification_scatter, truth, ctx.draws, image_df, tables, log=log)
        ctx.step("truth_residuals", plots.plot_truth_recovery_residuals, truth, image_df,
                 plots.figure_path(figures, "truth_recovery_residuals.pdf", prefix))
        if truth["profile"] is not None:
            ctx.step("mass_profile", plots.plot_m2d_aperture_ratio, truth["profile"], truth["center"], image_df,
                     truth["mass_per_kappa_pixel_msun"], plots.figure_path(figures, "mass_profile.pdf", prefix),
                     literature=truth["literature_m2d"])
            ctx.step("kappa_mass_residuals", plots.plot_truth_ratio_panels, truth["profile"], truth["kappa_profile"], truth["center"],
                     image_df, plots.figure_path(figures, "kappa_mass_residuals.pdf", prefix),
                     literature_mass=truth["literature_m2d"], literature_kappa=truth["literature_kappa"])
            ratio = truth["profile"]
            ctx.summary["truth"] = {"m2d_ratio_at_max_radius": float(ratio["m2d_ratio_median"].iloc[-1]), "n_radii": int(len(ratio))}
        return truth
