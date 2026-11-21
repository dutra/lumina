"""The image recovery of the reporting draws: its tables, the fit numbers of ``run_summary`` and the true-lens RMS."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import ModuleUnavailable, OutputContext, OutputModule

TRUE_LENS_RECORD = "true_lens_rms"       # the record of write_true_lens_rms
COVERAGE_TABLE = "position_coverage"


def write_true_lens_rms(stage: Path, image_df: pd.DataFrame, deflection_fits, cosmo_config: dict, *, log=print) -> dict | None:
    """The true-lens RMS of the image catalog (``diagnostics.true_lens``): the per-image table ``true_lens_residuals``
    and the record ``true_lens_rms`` (rms, bootstrap band over families, amplitude, images left out) in results.h5.
    Report-only: an unreadable map or no usable image logs the reason and returns None."""
    from ..diagnostics.exact_images import astropy_cosmology
    from ..diagnostics.true_lens import true_lens_residuals, true_lens_rms
    from ..io import store
    try:
        table, amplitude = true_lens_residuals(image_df, deflection_fits, astropy_cosmology(cosmo_config))
        record = {**true_lens_rms(table), "amplitude": amplitude, "deflection_fits": str(deflection_fits)}
    except (OSError, KeyError, IndexError, ValueError) as error:
        log(f"[outputs] true-lens RMS skipped: {type(error).__name__}: {error}")
        return None
    with store.batch(stage):
        store.write_table(stage, "true_lens_residuals", table)
        store.write_record(stage, TRUE_LENS_RECORD, record)
    left_out = ", ".join(f"{v} {k[2:].replace('_', ' ')}" for k, v in record.items()
                         if k.startswith("n_") and k not in ("n_images", "n_catalog_images", "n_families") and v)
    log(f"[outputs] true-lens RMS {record['rms_arcsec']:.3f}\" ({record['p16_arcsec']:.3f}-{record['p84_arcsec']:.3f}) on "
        f"{record['n_images']}/{record['n_catalog_images']} images of {record['n_families']} families, amplitude {amplitude:.4f}"
        f"; left out: {left_out or 'none'}")
    return record


class ImagesModule(OutputModule):
    """Compute: the image recovery (``diagnostics.image_recovery.image_recovery_with_reuse``: its tables and the
    record ``image_recovery_fingerprint``, reused when saved for the same draws), the table ``position_coverage``,
    ``fit`` in ``run_summary`` and, with ``outputs.truth.deflection_true_fits``, the true-lens RMS (``write_true_lens_rms``);
    raises ModuleUnavailable when the recovery step failed, so the modules that need the recovery are skipped.
    Product, read from the store: ``{"image_df": the table image_fit_quality, "recovery": {name: table} for every
    table the fingerprint record lists plus position_coverage, "true_lens": the record true_lens_rms when the last
    compute wrote it, else None}``.  Render: the residual histogram and the image recovery figure, both left out
    when no image was observed."""

    name = "images"
    description = ("image recovery (reused when saved for the same draws), position coverage, fit numbers, true-lens "
                   "RMS, residual histogram, image recovery figure")
    compute_steps = ("recovery", "true_lens")
    render_steps = ("residual_histogram", "recovery_figure")
    summary_keys = ("fit",)
    path_fields = ("truth.deflection_true_fits",)

    def compute(self, ctx: OutputContext) -> None:
        from ..diagnostics.image_recovery import fit_summary, image_recovery_tables, image_recovery_with_reuse
        from ..diagnostics.position_contours import position_coverage
        from ..io import store
        config, settings, state, stage, log = ctx.config, ctx.settings, ctx.state, ctx.stage, ctx.log
        deflection = config.outputs.truth.deflection_true_fits

        def recover():
            draws = ctx.draws
            recovery = image_recovery_with_reuse(stage, state, ctx.evaluator, draws, config.outputs.image_solver,
                                                 extra_images=settings.extra_images, compute=image_recovery_tables, log=log)
            recovery[COVERAGE_TABLE] = position_coverage(recovery["posterior_image_residual_draws"])
            store.write_table(stage, COVERAGE_TABLE, recovery[COVERAGE_TABLE])
            ctx.summary["fit"] = fit_summary(state, recovery, extra_image_draws=len(draws) if settings.extra_images else 0)
            fit = ctx.summary["fit"]
            if "recovered_image_count" in fit:
                log(f"[outputs] recovered {fit['recovered_image_count']['median']:g}/{fit['observed_image_count']} images (median over "
                    f"{len(draws.image_indices)} draws), rms {fit['image_rms_arcsec']['median']:.3f}\"")
            return recovery

        recovery = ctx.step("recovery", recover)
        if not ctx.ok("recovery"):
            ctx.skip_step("true_lens", "recovery failed")
            raise ModuleUnavailable("no image recovery (the recovery step failed)")
        image_df = recovery["image_fit_quality"]
        if deflection and not image_df.empty:
            ctx.step("true_lens", write_true_lens_rms, stage, image_df, deflection,
                     state.cosmo_config, log=log)

    def product(self, ctx: OutputContext) -> dict:
        from ..diagnostics.image_recovery import RECOVERY_RECORD
        from ..io import store
        names = [*store.read_record(ctx.stage, RECOVERY_RECORD)["tables"], COVERAGE_TABLE]
        recovery = {name: store.read_table(ctx.stage, name) for name in names}
        # Only the record of the last compute: an older one stays in the store when that compute wrote none.
        true_lens = store.read_record(ctx.stage, TRUE_LENS_RECORD) if ctx.computed("true_lens") else None
        return {"image_df": recovery["image_fit_quality"], "recovery": recovery, "true_lens": true_lens}

    def render(self, ctx: OutputContext) -> None:
        from ..plotting import plots
        images = ctx.products[self.name]
        image_df, recovery = images["image_df"], images["recovery"]
        if image_df.empty:
            return
        ctx.render_step("residual_histogram", plots.plot_image_residual_histogram, image_df,
                        plots.figure_path(ctx.figures, "image_residual_histogram.pdf", ctx.prefix),
                        draws=recovery.get("posterior_image_residual_draws"), true_lens=images["true_lens"], log=ctx.log,
                        needs="recovery")
        ctx.render_step("recovery_figure", plots.plot_image_recovery, image_df, recovery["image_recovery_extra_images"],
                        plots.figure_path(ctx.figures, "image_recovery.pdf", ctx.prefix), needs="recovery")
