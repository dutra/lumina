"""Median critical curves at each cutout's source redshift, over the reporting draws."""
from pathlib import Path
import numpy as np
from ..io import store
from ..likelihood.grf_fields import shared_grf_label


def cutout_grids(centers, fovs, resolution=96):
    """The ``resolution``-square grids (cutout, row, column, 2) in arcsec of the cutouts of :func:`median_cutout_determinants`."""
    axis = np.linspace(-.5, .5, resolution)
    return np.array([np.stack(np.meshgrid(c[0] + axis * f, c[1] + axis * f), axis=-1) for c, f in zip(centers, fovs)])


def median_cutout_determinants(draws, centers, fovs, redshifts, path, *, resolution=96, log=print):
    """Per-pixel 16th, 50th and 84th percentiles over the reporting draws (``reporting.draws.ReportingDraws``) of
    det A on a ``resolution``-square grid per cutout (centre, field of view in arcsec, source redshift), each draw with
    its own GRF realization.  Returns ``(grids, median, lower, upper)``; also stored, as arrays and a record named
    after the stem of ``path`` (``<stage>/tables/<name>.h5`` names the entry ``<name>`` of that stage's results)."""
    centers, fovs, redshifts = map(np.asarray, (centers, fovs, redshifts))
    grids = cutout_grids(centers, fovs, resolution)
    determinants = np.stack([draws.grid_maps(f'cutout {i}', grid[..., 0], grid[..., 1], float(z), quantities=('detA',),
                                             from_reference=True, log=log)['detA']
                             for i, (grid, z) in enumerate(zip(grids, redshifts))], axis=1)
    lower, median, upper = np.quantile(determinants, [.16, .5, .84], axis=0)
    path = Path(path)
    store.write_arrays(path.parent, path.stem, median_determinant=median, lower=lower, upper=upper, draw_indices=draws.indices, redshifts=redshifts,
                centers=centers, fovs=fovs, fingerprint=draws.key)
    store.write_record(path.parent, path.stem, dict(
        definition='Zero contour of pointwise posterior median determinant; band where the same draws 16-84% determinant interval contains zero',
        model='parametric + conditional GRF realization of each draw' if draws.has_field else 'parametric',
        grf_field=shared_grf_label(draws.evaluator), draw_indices=draws.indices.tolist(),
        redshifts=redshifts.tolist(), resolution=resolution))
    return grids, median, lower, upper
