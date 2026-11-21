"""Shared by the scripts that work on a saved run: puts ``src/`` on ``sys.path`` (import this module before
``kappaccino``) and redirects the catalog and FF-SIMS paths of runs saved before the data layout of 2026-11-08.  The
loading itself is :mod:`kappaccino.outputs.saved_run`."""
from dataclasses import replace
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def with_catalogs(config, catalogs):
    """``config`` reading the image and member catalogs from ``catalogs`` = (images.yaml, members.yaml): for a saved run
    whose catalog files have since moved (the FF-SIMS catalogs went from data/ff_sims to catalogs/<cluster>, 2026-11-08).
    The catalog contents must be the ones the run used; the rebuilt parameter order is checked against the saved state."""
    if not catalogs:
        return config
    images, members = catalogs
    model = replace(config.model, image_constraints=replace(config.model.image_constraints, catalog_path=images),
                    member_populations=tuple(replace(p, catalog_path=members) for p in config.model.member_populations))
    return replace(config, model=model)


def with_ffsims_dir(config, directory):
    """``config`` reading the FF-SIMS truth maps and simulated HST frames from ``directory`` (data/imaging/<cluster>/ffsims)
    instead of the paths saved in the run (data/ff_sims before 2026-11-08)."""
    if not directory:
        return config
    directory = str(directory).rstrip('/')
    outputs = config.outputs
    truth = replace(outputs.truth, kappa_true_fits=f'{directory}/kappa_z9_0.fits', gammax_true_fits=f'{directory}/gammax_z9_0.fits',
                    gammay_true_fits=f'{directory}/gammay_z9_0.fits')
    return replace(config, outputs=replace(outputs, truth=truth, rgb=replace(outputs.rgb, image_dir=directory)))


def redirect(catalogs=None, ffsims_dir=None):
    """The ``transform`` of ``kappaccino.outputs.saved_run.load_run`` that applies :func:`with_catalogs` and
    :func:`with_ffsims_dir`."""
    return lambda config: with_ffsims_dir(with_catalogs(config, catalogs), ffsims_dir)


def add_catalogs_argument(parser):
    parser.add_argument('--catalogs', nargs=2, metavar=('IMAGES_YAML', 'MEMBERS_YAML'),
                        help='read the catalogs from these files instead of the paths saved in the run (moved catalog files)')
    parser.add_argument('--ffsims-dir', metavar='DIR',
                        help='read the FF-SIMS truth maps and simulated frames from DIR (data/imaging/<cluster>/ffsims) instead of the saved paths')
