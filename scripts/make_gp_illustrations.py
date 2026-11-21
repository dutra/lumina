"""Regenerate standard GP diagnostics, including shared-field maps, from a saved run."""
import argparse

import _covariance_plot_cli  # noqa: F401  (puts src/ on sys.path)
from lumina.outputs.saved_run import load_run
from lumina.plotting.gp import render_gp_diagnostics, save_gp_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gp_run')
    args = parser.parse_args()
    config, state, evaluator, posterior, paths = load_run(args.gp_run)
    if not evaluator.gp_mode:
        parser.error('the run must use likelihood_mode gp_image_plane')
    if save_gp_diagnostics(config, state, evaluator, posterior, paths) is not None:
        render_gp_diagnostics(config, state, posterior, paths)


if __name__ == '__main__':
    main()
