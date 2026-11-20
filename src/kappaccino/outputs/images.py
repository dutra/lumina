"""The image recovery of the reporting draws: its tables, the fit numbers of ``run_summary`` and the true-lens RMS."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import ModuleUnavailable, OutputContext, OutputModule


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
        store.write_record(stage, "true_lens_rms", record)
    left_out = ", ".join(f"{v} {k[2:].replace('_', ' ')}" for k, v in record.items()
                         if k.startswith("n_") and k not in ("n_images", "n_catalog_images", "n_families") and v)
    log(f"[outputs] true-lens RMS {record['rms_arcsec']:.3f}\" ({record['p16_arcsec']:.3f}-{record['p84_arcsec']:.3f}) on "
        f"{record['n_images']}/{record['n_catalog_images']} images of {record['n_families']} families, amplitude {amplitude:.4f}"
        f"; left out: {left_out or 'none'}")
    return record


class ImagesModule(OutputModule):
    """Product: ``{"recovery": the tables of image_recovery_with_reuse plus position_coverage, "image_df": the table
    image_fit_quality, "true_lens": the true_lens_rms record or None}``.  Raises ModuleUnavailable when the recovery
    step failed, so the modules that need the recovery are skipped."""

    name = "images"
    description = ("image recovery (reused when saved for the same draws), position coverage, fit numbers, true-lens "
                   "RMS, residual histogram, image recovery figure")
    steps = ("recovery", "true_lens", "residual_histogram", "recovery_figure")
    _dependent = ("true_lens", "residual_histogram", "recovery_figure")
    summary_keys = ("fit",)
    path_fields = ("truth.deflection_true_fits",)

    def run(self, ctx: OutputContext):
        from ..diagnostics.image_recovery import fit_summary, image_recovery_tables, image_recovery_with_reuse
        from ..diagnostics.position_contours import position_coverage
        from ..io import store
        from ..plotting import plots
        config, state, stage, figures, prefix, log = ctx.config, ctx.state, ctx.stage, ctx.figures, ctx.prefix, ctx.log

        def recover():
            draws = ctx.draws
            recovery = image_recovery_with_reuse(stage, state, ctx.evaluator, draws, config.image_diagnostics,
                                                 extra_images=config.reporting.extra_images, compute=image_recovery_tables, log=log)
            recovery["position_coverage"] = position_coverage(recovery["posterior_image_residual_draws"])
            store.write_table(stage, "position_coverage", recovery["position_coverage"])
            ctx.summary["fit"] = fit_summary(state, recovery, extra_image_draws=len(draws) if config.reporting.extra_images else 0)
            fit = ctx.summary["fit"]
            if "recovered_image_count" in fit:
                log(f"[outputs] recovered {fit['recovered_image_count']['median']:g}/{fit['observed_image_count']} images (median over "
                    f"{len(draws.image_indices)} draws), rms {fit['image_rms_arcsec']['median']:.3f}\"")
            return recovery

        recovery = ctx.step("recovery", recover)
        if not ctx.ok("recovery"):
            for name in self._dependent:
                ctx.skip_step(name, "recovery failed")
            raise ModuleUnavailable("no image recovery (the recovery step failed)")
        image_df = recovery["image_fit_quality"]
        true_lens = None
        if config.truth.deflection_true_fits and not image_df.empty:
            true_lens = ctx.step("true_lens", write_true_lens_rms, stage, image_df, config.truth.deflection_true_fits,
                                 state.cosmo_config, log=log)
        if not image_df.empty:
            ctx.step("residual_histogram", plots.plot_image_residual_histogram, image_df,
                     plots.figure_path(figures, "image_residual_histogram.pdf", prefix),
                     draws=recovery.get("posterior_image_residual_draws"), true_lens=true_lens, log=log)
            ctx.step("recovery_figure", plots.plot_image_recovery, image_df, recovery["image_recovery_extra_images"],
                     plots.figure_path(figures, "image_recovery.pdf", prefix))
        return {"recovery": recovery, "image_df": image_df, "true_lens": true_lens}
