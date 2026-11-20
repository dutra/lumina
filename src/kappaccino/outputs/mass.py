"""Critical curves and the member masses (aperture-mass histogram and zeta masses of the simulated clusters)."""
from __future__ import annotations

from .base import OutputContext, OutputModule


class MassModule(OutputModule):
    """The aperture-mass histogram and the zeta masses read the table ``member_diagnostics`` (``fit``), hence
    ``after``; each decides by itself whether it applies (a truth convergence map and population members)."""

    name = "mass"
    requires = ("images",)
    after = ("fit",)
    description = "critical curves, aperture-mass histogram and zeta masses (the last two with a truth convergence map)"
    steps = ("critical_curves", "aperture_masses", "zeta_masses")
    path_fields = ("truth.kappa_true_fits",)

    def run(self, ctx: OutputContext):
        from ..plotting import plots
        from ..plotting.aperture_mass_histogram import write_aperture_mass_histogram
        from ..plotting.zeta_masses import write_zeta_masses
        config, state, log = ctx.config, ctx.state, ctx.log
        image_df = ctx.products["images"]["image_df"]
        z_crit = float(config.truth.caustic_source_redshift)
        ctx.step("critical_curves", lambda: plots.plot_critical_curves(
            state, ctx.draws, z_crit if z_crit > state.z_lens else max(state.z_source_values),
            plots.figure_path(ctx.figures, "critical_curves.pdf", ctx.prefix), log=log))
        ctx.step("aperture_masses", lambda: write_aperture_mass_histogram(config, state, ctx.draws, ctx.paths, image_df, log=log))
        ctx.step("zeta_masses", lambda: write_zeta_masses(config, state, ctx.draws, ctx.paths, image_df, log=log))
        return None
