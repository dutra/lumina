"""The comparison with the true lens of a simulated cluster (``outputs.truth.kappa_true_fits``,
``outputs.truth.deflection_true_fits``)."""
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
    """Compute: the relative displacement against separation (``plotting.relative_displacement``, with a true
    deflection map: the table and record ``relative_displacement``), the truth maps
    (``diagnostics.truth_recovery.compute_truth_recovery``: FITS, profile tables, the record ``truth_recovery`` and the
    arrays ``truth_recovery_residuals``), the magnification recovery, the cumulative source area, the magnification
    scatter and ``truth`` in ``run_summary``; raises ModuleUnavailable when the truth_maps step failed, after which the
    relative displacement figure is still drawn (its step needs no truth map) and the figures of the truth maps are
    reported as not drawn.  Render: every figure, from the store (``truth_recovery.load_truth_recovery`` and the tables
    of each figure).  No product: no module reads it."""

    name = "truth"
    requires = ("images",)
    description = ("truth-recovery maps and profiles and every figure made from them (with a truth convergence map), "
                   "relative displacement against separation (with a true deflection map)")
    compute_steps = ("relative_displacement", "truth_maps", "magnification_recovery", "cumulative_source_area",
                     "magnification_scatter")
    render_steps = ("relative_displacement", "magnification_recovery", "cumulative_source_area", "truth_residuals",
                    "mass_profile", "kappa_mass_residuals")
    summary_keys = ("truth",)
    path_fields = ("truth.kappa_true_fits", "truth.gammax_true_fits", "truth.gammay_true_fits",
                   "truth.deflection_true_fits")

    def active(self, ctx: OutputContext) -> str | None:
        truth = ctx.settings
        return None if truth.kappa_true_fits or truth.deflection_true_fits else (
            "no truth convergence or deflection map (outputs.truth.kappa_true_fits, outputs.truth.deflection_true_fits)")

    def compute(self, ctx: OutputContext) -> None:
        from ..diagnostics.exact_images import astropy_cosmology
        from ..diagnostics.truth_recovery import compute_truth_recovery
        from ..plotting.cumulative_source_area import save_cumulative_source_area
        from ..plotting.magnification_recovery import save_magnification_recovery
        from ..plotting.relative_displacement import save_relative_displacement
        from ..plotting.scatter_summary import write_magnification_scatter
        config, state, tables, log = ctx.config, ctx.state, ctx.tables, ctx.log
        images = ctx.products["images"]
        # First: it needs no truth maps, so its table is saved and its figure drawn even when truth_maps fails.
        ctx.step("relative_displacement", lambda: save_relative_displacement(
            config, state, ctx.evaluator, ctx.draws, images["recovery"], images["true_lens"], ctx.stage, log=log))
        truth = ctx.step("truth_maps", lambda: compute_truth_recovery(state, ctx.draws, config.outputs, ctx.paths.fits, tables, log=log))
        if not ctx.ok("truth_maps"):
            for name in self.compute_steps[2:]:
                ctx.skip_step(name, "truth_maps failed")
            raise ModuleUnavailable("no truth maps (the truth_maps step failed)")
        if truth is None:
            return
        image_df = images["image_df"]
        cosmology = astropy_cosmology(state.cosmo_config)
        ctx.step("magnification_recovery", save_magnification_recovery, truth, image_df, state.reference, state.z_lens,
                 cosmology, config.outputs, tables, log=log)
        ctx.step("cumulative_source_area", lambda: save_cumulative_source_area(truth, ctx.draws, cosmology, tables, log=log))
        ctx.step("magnification_scatter", lambda: write_magnification_scatter(truth, ctx.draws, image_df, tables, log=log))
        if truth["profile"] is not None:
            ratio = truth["profile"]
            ctx.summary["truth"] = {"m2d_ratio_at_max_radius": float(ratio["m2d_ratio_median"].iloc[-1]), "n_radii": int(len(ratio))}

    def render(self, ctx: OutputContext) -> None:
        from ..diagnostics.truth_recovery import load_truth_recovery
        from ..io import store
        from ..plotting import plots
        from ..plotting.cumulative_source_area import render_cumulative_source_area
        from ..plotting.magnification_recovery import render_magnification_recovery
        from ..plotting.relative_displacement import render_relative_displacement
        ctx.render_step("relative_displacement", lambda: render_relative_displacement(
            store.read_table(ctx.stage, "relative_displacement"),
            plots.figure_path(ctx.figures, "relative_displacement.pdf", ctx.prefix)))
        maps = ctx.completion["steps"]["truth_maps"]
        if maps["status"] == "done" and not maps["wrote"]:   # no truth convergence map, or no source behind the lens
            return
        figures, prefix, stage = ctx.figures, ctx.prefix, ctx.stage
        truth = _once(lambda: load_truth_recovery(stage))
        image_df = ctx.products["images"]["image_df"]
        ctx.render_step("magnification_recovery", lambda: render_magnification_recovery(
            stage, figures / f'{prefix + "_" if prefix else ""}magnification_recovery.pdf', log=ctx.log))
        ctx.render_step("cumulative_source_area", lambda: render_cumulative_source_area(
            stage, figures / f'{prefix + "_" if prefix else ""}cumulative_source_area.pdf', log=ctx.log))
        ctx.render_step("truth_residuals", lambda: plots.plot_truth_recovery_residuals(
            truth(), image_df, plots.figure_path(figures, "truth_recovery_residuals.pdf", prefix)), needs="truth_maps")
        ctx.render_step("mass_profile", lambda: None if truth()["profile"] is None else plots.plot_m2d_aperture_ratio(
            truth()["profile"], truth()["center"], image_df, truth()["mass_per_kappa_pixel_msun"],
            plots.figure_path(figures, "mass_profile.pdf", prefix), literature=truth()["literature_m2d"]), needs="truth_maps")
        ctx.render_step("kappa_mass_residuals", lambda: None if truth()["profile"] is None else plots.plot_truth_ratio_panels(
            truth()["profile"], truth()["kappa_profile"], truth()["center"], image_df,
            plots.figure_path(figures, "kappa_mass_residuals.pdf", prefix), literature_mass=truth()["literature_m2d"],
            literature_kappa=truth()["literature_kappa"]), needs="truth_maps")
