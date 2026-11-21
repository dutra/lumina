"""Rerun the whole output pass of saved runs over their reporting draws; no inference.

Every figure, table and results macro is rebuilt from one set of posterior draws (``outputs.draws`` of the run, or
--draws), each draw evaluated once and shared by all products.  The same as ``lumina outputs RUN_DIR`` for each run,
plus the options for moved catalog and FF-SIMS files.
"""
import argparse
import json
from pathlib import Path

from _covariance_plot_cli import add_catalogs_argument, redirect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    add_catalogs_argument(parser)
    parser.add_argument('--draws', type=int, default=None, help='outputs.draws and outputs.image_draws (default: the run config)')
    parser.add_argument('--only', default=None, help='comma-separated output modules to run (lumina outputs --list)')
    parser.add_argument('--skip', default=None, help='comma-separated output modules not to run')
    args = parser.parse_args()
    from lumina.io import store
    from lumina.outputs.command import rerun_outputs
    names = lambda text: None if text is None else tuple(n.strip() for n in text.split(',') if n.strip())
    for run in args.runs:
        rerun_outputs(run, only=names(args.only), skip=names(args.skip), draws=args.draws,
                      transform=redirect(args.catalogs, args.ffsims_dir), log=lambda text: print(text, flush=True))
        print(json.dumps(store.read_record(run / 'final', 'run_summary').get('fit', {}), indent=2, default=float), flush=True)


if __name__ == '__main__':
    main()
