"""Critical curves and the member masses (aperture-mass histogram and zeta masses of the simulated clusters)."""
from __future__ import annotations

from .base import OutputContext, OutputModule

CRITICAL_CURVES = "critical_curves"          # the arrays of the figure critical_curves.pdf


class MassModule(OutputModule):
    """The aperture masses and the zeta masses read the table ``member_diagnostics`` (``fit``), hence ``after``; each
    decides by itself whether it applies (a truth convergence map and population members).  Compute: the arrays
    ``critical_curves`` (``plots.critical_curve_maps``), the aperture masses (``member_aperture_masses`` and the record
    ``aperture_mass_histogram``; no figure) and the zeta masses (``zeta_masses.save_zeta_masses``).  Render: the
    critical curves and the subhalo mass and concentration figure.  No product."""

    name = "mass"
    requires = ("images",)
    after = ("fit",)
    description = "critical curves, aperture-mass histogram and zeta masses (the last two with a truth convergence map)"
    compute_steps = ("critical_curves", "aperture_masses", "zeta_masses")
    render_steps = ("critical_curves", "zeta_masses")
    path_fields = ("truth.kappa_true_fits",)

    def compute(self, ctx: OutputContext) -> None:
        from ..io import store
        from ..plotting import plots
        from ..plotting.aperture_mass_histogram import write_aperture_mass_histogram
        from ..plotting.zeta_masses import save_zeta_masses
        from .images import stored_image_table
        config, state, log = ctx.config, ctx.state, ctx.log
        image_df = stored_image_table(ctx)          # TRANSITIONAL: ctx.products["images"] once images is converted
        z_crit = float(config.truth.caustic_source_redshift)

        def critical_curves():
            maps = plots.critical_curve_maps(state, ctx.draws, z_crit if z_crit > state.z_lens else max(state.z_source_values), log=log)
            store.write_arrays(ctx.stage, CRITICAL_CURVES, maps)
            return CRITICAL_CURVES

        ctx.step("critical_curves", critical_curves)
        ctx.step("aperture_masses", lambda: write_aperture_mass_histogram(config, state, ctx.draws, ctx.paths, image_df, log=log))
        ctx.step("zeta_masses", lambda: save_zeta_masses(config, state, ctx.draws, ctx.paths, image_df, log=log))

    def render(self, ctx: OutputContext) -> None:
        from ..io import store
        from ..plotting import plots
        from ..plotting.zeta_masses import render_zeta_masses
        ctx.render_step("critical_curves", lambda: plots.render_critical_curves(
            ctx.state, store.read_arrays(ctx.stage, CRITICAL_CURVES), plots.figure_path(ctx.figures, "critical_curves.pdf", ctx.prefix)))
        ctx.render_step("zeta_masses", render_zeta_masses, ctx.stage,
                        ctx.figures / f"{ctx.prefix or ''}_subhalo_mass_concentration.pdf", log=ctx.log)
