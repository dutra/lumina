"""Recompute the zeta masses and contrasts of cluster members (true and model convergence maps) and their two figures
for saved Hera or Ares runs.  Uses the saved posterior; no inference.  See lumina.plotting.zeta_masses.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    from _covariance_plot_cli import add_catalogs_argument, redirect
    add_catalogs_argument(parser)
    parser.add_argument('--draws', type=int, default=None,
                        help='reporting draws behind the model masses (default: the run config)')
    parser.add_argument('--recompute', action='store_true', help='ignore the cached masses (zeta_masses_cache.h5)')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from lumina.outputs.saved_run import load_run
    from dataclasses import replace
    from lumina.plotting.zeta_masses import write_zeta_masses
    from lumina.reporting.draws import reporting_draws
    for run in args.runs:
        config, state, evaluator, posterior, paths = load_run(run, transform=redirect(args.catalogs, args.ffsims_dir))
        images = pd.read_csv(paths.tables/'image_fit_quality.csv', dtype={'family_id': str, 'image_label': str})
        if args.draws is not None:
            config = replace(config, reporting=replace(config.reporting, draws=args.draws))
        print(write_zeta_masses(config, state, reporting_draws(config, evaluator, posterior), paths, images, recompute=args.recompute,
                                log=lambda text: print(text, flush=True)), flush=True)


if __name__ == '__main__':
    main()
