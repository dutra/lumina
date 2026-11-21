"""The products of the sampled parameters: posterior tables, chain traces, audit, scaling relation, corners, covariance,
field and member diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import OutputContext, OutputModule


def posterior_summary_table(state, posterior) -> pd.DataFrame:
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


def classification_table(state) -> pd.DataFrame:
    rankings = (state.members.classification or {}).get("rankings", [])
    return pd.DataFrame([{"member_id": r["member"], "member_class": "exact" if r["exact"] else "far-field", **{k: v for k, v in r.items() if k not in {"member", "exact"}}}
                         for r in rankings])


def fixed_relation_values(state, member_table) -> dict[str, float]:
    """``{"<population>.<field>": value}`` of the scaling-relation fields that are not sampled: the base values of the
    packed lens spec (``scaling.fixed_relation_fields``, or a slope a Lenstool file holds fixed)."""
    from ..core.model import CORED_POWER_LAW_PROFILE
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


class FitModule(OutputModule):
    """Compute: the tables ``posterior_summary`` and ``member_classification``, the chain diagnostics
    (``chain_convergence``, record ``chain_traces``), the record ``audit``, ``member_scaling_relation``, the covariance
    errors (``covariance_errors_both_planes``, over the reporting draws), the GP diagnostics (``gp_diagnostics`` and, with
    a shared GRF, ``conditional_field_maps``) and the member diagnostics (``member_diagnostics`` and its companions).
    Render: every figure, from those entries and the posterior; the potfile corners, the block, cosmology, covariance and
    global corners are drawn from the posterior alone.  No product: ``mass`` and ``rgb`` read the tables
    ``member_diagnostics`` and ``covariance_errors_both_planes`` from the store."""

    name = "fit"
    description = ("posterior tables, chain traces, audit, scaling relation, potfile plots, corners, covariance, GP/GRF "
                   "plots, member diagnostics")
    compute_steps = ("tables", "chain_traces", "audit", "scaling_table", "covariance", "fields", "members")
    shared_settings = ("draws", "image_draws", "caustic_source_redshift")
    render_steps = ("chain_traces", "potfiles", "block_corner", "cosmology_corner", "covariance", "fields", "members",
                    "scaling_plot", "globals_corner")

    def compute(self, ctx: OutputContext) -> None:
        from ..io import store
        from ..plotting import plots
        from ..plotting.chain_traces import save_chain_traces
        from ..plotting.covariance import save_covariance_errors
        from ..plotting.gp import save_gp_diagnostics
        from ..plotting.members import save_member_diagnostics
        config, state, posterior, stage, log = ctx.config, ctx.state, ctx.posterior, ctx.stage, ctx.log

        def tables():
            with store.batch(stage):
                store.write_table(stage, "posterior_summary", posterior_summary_table(state, posterior))
                store.write_table(stage, "member_classification", classification_table(state))
            return "posterior_summary"

        def scaling_table():
            store.write_table(stage, "member_scaling_relation", plots.member_scaling_table(state, posterior))
            return "member_scaling_relation"

        def audit():
            store.write_record(stage, "audit", ctx.audit, default=float)
            return "audit"

        ctx.step("tables", tables)
        ctx.step("chain_traces", save_chain_traces, posterior, state.parameter_specs, ctx.tables, log=log)
        if ctx.audit:
            ctx.step("audit", audit)
        ctx.step("scaling_table", scaling_table)
        ctx.step("covariance", lambda: save_covariance_errors(config, state, ctx.draws, ctx.paths))
        ctx.step("fields", lambda: save_gp_diagnostics(config, state, ctx.evaluator, posterior, ctx.paths, log=log))
        ctx.step("members", lambda: save_member_diagnostics(config, state, ctx.evaluator, posterior, ctx.paths, log=log))

    def render(self, ctx: OutputContext) -> None:
        from ..io import store
        from ..plotting import plots
        from ..plotting.block_corner import write_cosmology_corner, write_posterior_block_corner
        from ..plotting.chain_traces import render_chain_traces, trace_data
        from ..plotting.covariance import render_covariance_plots
        from ..plotting.gp import render_gp_diagnostics
        from ..plotting.members import render_member_diagnostics
        from ..plotting.potfile import write_potfile_plots
        config, state, posterior, stage, figures, prefix, log = ctx.config, ctx.state, ctx.posterior, ctx.stage, ctx.figures, ctx.prefix, ctx.log

        def scaling_plot():
            member_table = store.read_table(stage, "member_scaling_relation")
            if member_table.empty:
                return None
            return plots.plot_scaling_relation(
                member_table, np.asarray(posterior.physical), state.parameter_specs,
                plots.figure_path(figures, "scaling_relation.pdf", prefix), fixed=fixed_relation_values(state, member_table))

        def globals_corner():
            traces = trace_data(posterior, state.parameter_specs)
            corner_path = plots.plot_corner(np.asarray(posterior.physical), state.parameter_specs,
                                            plots.figure_path(figures, "corner_globals.pdf", prefix),
                                            grouped_physical=None if traces is None else traces['chains'])
            if corner_path is None:
                log("[outputs] corner skipped: fewer than two global parameters with posterior spread")

        ctx.render_step("chain_traces", render_chain_traces, posterior, state.parameter_specs, figures, prefix, log=log)
        ctx.step("potfiles", write_potfile_plots, posterior.physical, state.parameter_specs, figures, prefix, posterior=posterior, log=log)
        ctx.step("block_corner", write_posterior_block_corner, posterior.physical, state.parameter_specs,
                 plots.figure_path(figures, "corner_combined.pdf", prefix))
        ctx.step("cosmology_corner", write_cosmology_corner, posterior.physical, state.parameter_specs,
                 plots.figure_path(figures, "corner_cosmology.pdf", prefix))
        ctx.render_step("covariance", render_covariance_plots, config, state, posterior, ctx.paths, log=log)
        ctx.render_step("fields", render_gp_diagnostics, config, state, posterior, ctx.paths, log=log)
        ctx.render_step("members", render_member_diagnostics, config, ctx.paths, log=log)
        ctx.render_step("scaling_plot", scaling_plot, needs="scaling_table")
        ctx.step("globals_corner", globals_corner)
