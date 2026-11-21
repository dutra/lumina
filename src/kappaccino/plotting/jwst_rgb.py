"""RGB overviews with matched posterior z=9 critical curves and bands."""
from pathlib import Path
from .style import CRITICAL_COLOR
NOT_RECOVERED_COLOR = '#FF3030'     # an observed image found in fewer than half the draws
EXTRA_IMAGE_COLOR = '#C77DFF'       # a predicted image with no catalog entry, found in at least half the draws
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .grf import PLOT_FIELD_VERSION
from ..io import store
from ..likelihood.grf_fields import shared_field_evaluator, shared_grf_label
from ..diagnostics.position_contours import PROBABILITIES

# The recovered-position contours: the 68.3% region of each observed image, enlarged about the mean of its per-draw
# positions so that its shape and orientation show at the scale of the overview.
RECOVERED_CONTOUR_PROBABILITY = PROBABILITIES[0]
RECOVERED_CONTOUR_SCALE = 4.0
# The legend names the contours only: their level and enlargement belong to the caption (and the figure's record).
RECOVERED_CONTOUR_LABEL = 'Recovered positions'
# The F200W photometry of the M0416 images that sizes the overview's markers (read in compute, saved per overview).
MARKER_PHOTOMETRY = Path(__file__).parents[1] / 'data' / 'm0416_image_f200w_photometry.csv'


def marker_photometry(catalog, reference, photometry=None):
    """Use measured F200W magnitudes only for matching IDs and sky positions: the rows of the table ``photometry``
    (:data:`MARKER_PHOTOMETRY` as read) whose label and position match an image of ``catalog``; with None, no image
    has a measured magnitude."""
    from ..imaging.cutouts import arcsec_to_radec
    ra,dec=arcsec_to_radec(catalog.x_obs_arcsec.to_numpy(),catalog.y_obs_arcsec.to_numpy(),reference)
    result=pd.DataFrame({'image_label':catalog.image_label.to_numpy(),'phot_id':-1,
                         'match_separation_arcsec':np.nan,'m_F200W_AB':np.nan})
    if photometry is not None:
        saved=photometry.set_index('image_label').reindex(result.image_label)
        sep=np.hypot((saved.ra.to_numpy()-ra)*3600*np.cos(np.deg2rad(dec)),(saved.dec.to_numpy()-dec)*3600)
        valid=(sep<.05)&saved.valid_photometry.fillna(False).to_numpy(dtype=bool)&np.isfinite(saved.m_F200W_AB.to_numpy())
        for key in ('phot_id','match_separation_arcsec','m_F200W_AB'):
            result.loc[valid,key]=saved[key].to_numpy()[valid]
    valid=np.isfinite(result.m_F200W_AB.to_numpy())
    result['valid_photometry']=valid
    result['marker_area_pt2']=9.
    result.loc[valid,'marker_area_pt2']=np.clip(16*10**(-.2*(result.loc[valid,'m_F200W_AB']-26)),9,64)
    return result


def rgb_overview_stem(image_config, bands):
    """Name the imaging product by its instrument, independently of cluster."""
    jwst = image_config.figure_suffix.lower() == '_jwst'
    jwst |= any(str(b).lower().split('_')[-1] in {'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w'} for b in bands)
    return ('jwst' if jwst else 'hst') + '_rgb_critical_z9'


def overview_marker_photometry(config, catalog, matched, stem):
    """Attach catalog photometry by ID; prefer the matched F200W measurements ``matched`` (:func:`marker_photometry`
    of ``catalog``, as compute saved it)."""
    result = matched.copy()
    result['magnitude_AB'] = result['m_F200W_AB']
    result['magnitude_band'] = np.where(result.valid_photometry, 'F200W', '')
    path = config.model.image_constraints.catalog_path
    if path:
        from ..io.image_catalog import load_image_catalog
        phot = load_image_catalog(path).set_index('image_label').reindex(catalog.image_label)
    else:
        phot = catalog
    preferred = ('f200w', 'f814w') if stem.startswith('jwst') else ('f814w', 'f200w')
    columns = [c for band in preferred for c in phot if c.lower() in {f'mag_{band}', f'mag_hst_{band}', f'mag_jwst_{band}'}]
    columns += [c for c in phot if c.startswith('mag_') and not c.startswith('mag_err_') and c not in columns]
    if 'catalog_mag' in phot:
        columns.append('catalog_mag')
    for column in columns:
        values = pd.to_numeric(phot[column], errors='coerce').to_numpy()
        use = ~result.valid_photometry.to_numpy() & np.isfinite(values)
        result.loc[use, 'magnitude_AB'] = values[use]
        result.loc[use, 'magnitude_band'] = column.removeprefix('mag_')
        result.loc[use, 'valid_photometry'] = True
    valid = result.valid_photometry.to_numpy()
    result.loc[valid, 'marker_area_pt2'] = np.clip(16 * 10**(-.2 * (result.loc[valid, 'magnitude_AB'] - 26)), 9, 64)
    return result


def enlarged_contour_segments(segments, positions, scale=RECOVERED_CONTOUR_SCALE):
    """The contour ``segments`` (arcsec) of one image enlarged by ``scale`` about the mean of its finite per-draw
    ``positions`` (draw, 2), so that its shape and orientation show on the overview.  All segments share that centre."""
    points = np.asarray(positions, dtype=float).reshape(-1, 2)
    points = points[np.isfinite(points).all(1)]
    if not len(points):
        raise ValueError('No finite position to enlarge the contour about')
    centre = points.mean(0)
    return [centre + scale * (np.asarray(segment, dtype=float) - centre) for segment in segments]


def recovered_contour_segments(contours):
    """``(identity, segments)`` of the recovered-position contours at ``RECOVERED_CONTOUR_PROBABILITY``, each enlarged
    by ``RECOVERED_CONTOUR_SCALE`` about the mean of its own image's per-draw positions (arcsec)."""
    curves = [curve for curve in contours['curves']
              if curve['kind'] == 'recovered' and np.isclose(curve['probability'], RECOVERED_CONTOUR_PROBABILITY)]
    if not curves:
        return []
    data = contours['position_data']
    labels = [str(label) for label in data['image_labels']]
    return [(curve['identity'], enlarged_contour_segments(curve['segments'], data['recovered'][:, labels.index(str(curve['identity']))]))
            for curve in curves]


def candidate_catalog(config):
    """The M0416 candidate catalog the ``_candidates`` overview draws: ``{"path": ..., "candidates": [...]}``, read from
    ``outputs.rgb.candidate_catalog_path`` (default: the packaged ``data/m0416_candidates.json``)."""
    path = config.outputs.rgb.candidate_catalog_path
    path = Path(path) if path else Path(__file__).parents[1] / 'data' / 'm0416_candidates.json'
    return dict(path=str(path), candidates=json.loads(path.read_text())['candidates'])


def candidate_record(stem):
    """The record of the candidates of the overview ``stem`` (:func:`candidate_catalog`)."""
    return f'{stem}_candidate_catalog'


def photometry_entry(stem):
    """The table and record of the matched marker photometry of the overview ``stem`` (:func:`marker_photometry`)."""
    return f'{stem}_marker_photometry'


def read_marker_photometry(tables, stem):
    """``{"table": ..., "path": ...}``: the matched marker photometry of the overview ``stem`` and the file it was read
    from, as :func:`save_rgb_critical` saved them."""
    return dict(table=store.read_table(tables, photometry_entry(stem)),
                path=store.read_record(tables, photometry_entry(stem))['path'])


def save_rgb_critical(config, state, draws, catalog, extra, imaging, tables, *, log=print):
    """Compute and save what the RGB critical overview draws, in the stage's results (``tables`` is the stage folder or
    its ``tables``): the arrays ``<stem>`` (median det A over the reporting draws on the overview grid, its 16th and
    84th percentiles), the records ``<stem>_band`` and ``<stem>_model`` (whether the run has a shared GRF and its
    label), for M0416 the candidates of its ``_candidates`` overview (record ``<stem>_candidate_catalog``,
    :func:`candidate_catalog`) and the F200W photometry that sizes its markers (table ``<stem>_marker_photometry``,
    :func:`marker_photometry` of :data:`MARKER_PHOTOMETRY`, and the record of the same name with that ``path``), and
    the position contours
    (:func:`~kappaccino.diagnostics.position_contours.load_or_compute_contours`).  Of the imaging only the reference
    band's WCS and shape (the overview grid) are read.  Returns the stem."""
    tables=Path(tables)
    from .mass_figure import overview_geometry
    from ..imaging import cutouts

    images,bands,display,cluster=imaging
    stem=rgb_overview_stem(config.outputs.rgb, bands)
    reference_image=images[cutouts.rgb_reference_band(bands)]
    band_geometry=overview_geometry(reference_image,catalog,extra,state.reference)
    from ..reporting.draws import OVERVIEW_QUANTITIES
    determinants=draws.grid_maps('overview',band_geometry['x'],band_geometry['y'],9.,quantities=OVERVIEW_QUANTITIES,
                               reference=True,log=log)['detA']
    lower,det,upper=critical_quantile_summary(determinants)
    band=(lower<0)&(upper>0)
    fingerprint=draws.key
    candidates=candidate_catalog(config) if cluster.lower()=='m0416' else None
    photometry=marker_photometry(catalog,state.reference,pd.read_csv(MARKER_PHOTOMETRY)) if cluster.lower()=='m0416' else None
    band_info=dict(draw_indices=draws.indices.tolist(),resolution=int(det.shape[0]),
                   definition='Zero contour of pointwise median determinant; shaded where its 16-84% interval contains zero; parametric + conditional GRF realization of each draw',
                   selection='The reporting draws: random posterior draws over all chains (run seed)')
    with store.batch(tables):
        store.write_record(tables,f'{stem}_band',band_info)
        store.write_arrays(tables,stem,determinant=det,lower=lower,upper=upper,band=band,
                            fingerprint=fingerprint,z_source=9.,draw_indices=draws.indices,
                            central_statistic='pointwise posterior median determinant')
        store.write_record(tables,f'{stem}_model',dict(shared_grf=shared_field_evaluator(draws.evaluator) is not None,
                                                       grf_field=shared_grf_label(draws.evaluator)))
        if candidates is not None:
            store.write_record(tables,candidate_record(stem),candidates)
        if photometry is not None:
            store.write_table(tables,photometry_entry(stem),photometry)
            store.write_record(tables,photometry_entry(stem),dict(path=str(MARKER_PHOTOMETRY)))
    from ..diagnostics.position_contours import load_or_compute_contours
    load_or_compute_contours(config, state, draws, extra, tables, log=log)
    return stem


def render_saved_rgb_critical(config, state, catalog, extra, imaging, output, tables, contours=None, *, log=print):
    """Draw the RGB critical overview (:func:`draw_rgb_critical`) from what :func:`save_rgb_critical` saved; the
    position ``contours`` are read from the store unless given (as
    :func:`~kappaccino.diagnostics.position_contours.load_position_contours` returns them).  Returns the PDF paths."""
    from ..diagnostics.position_contours import load_position_contours
    stem=rgb_overview_stem(config.outputs.rgb, imaging[1])
    saved=store.read_arrays(tables,stem)
    model=store.read_record(tables,f'{stem}_model')
    m0416=imaging[3].lower()=='m0416'
    return draw_rgb_critical(config, state.reference, catalog, extra, imaging, output, tables, determinant=saved['determinant'],
                             lower=saved['lower'], upper=saved['upper'],
                             contours=load_position_contours(tables) if contours is None else contours,
                             fingerprint=np.asarray(saved['fingerprint']).item(), reporting_draws=len(saved['draw_indices']),
                             shared_grf=model['shared_grf'], grf_field=model['grf_field'],
                             candidates=store.read_record(tables,candidate_record(stem)) if m0416 else None,
                             photometry=read_marker_photometry(tables,stem) if m0416 else None, log=log)


@plt.rc_context()
def draw_rgb_critical(config, reference, catalog, extra, imaging, output, tables, *, determinant, lower, upper, contours,
                      fingerprint, reporting_draws, shared_grf, grf_field, candidates=None, photometry=None, redrawn=False,
                      log=print):
    """Draw the RGB critical overview (and, for M0416, its ``_candidates`` variant) from computed inputs: the
    median ``determinant`` of det A with its 16th/84th percentiles ``lower``/``upper``, the position ``contours``
    (:func:`~kappaccino.diagnostics.position_contours.load_or_compute_contours`), the draws' ``fingerprint`` and
    count, whether the run has a ``shared_grf`` (a knee mass field shared by every image) and the ``grf_field``
    provenance label (:func:`~kappaccino.likelihood.grf_fields.shared_grf_label`) and, for M0416, the saved
    ``candidates`` (:func:`candidate_catalog`) of the ``_candidates`` variant and the saved marker ``photometry``
    (:func:`read_marker_photometry`).  Writes the PDFs, their figure records and the marker tables (in the results of
    the stage of ``tables``); returns the PDF paths."""
    output,tables=Path(output),Path(tables)
    from .mass_figure import overview_geometry, aligned_rgb, grid_pixels
    from ..imaging import cutouts

    images,bands,display,cluster=imaging
    stem=rgb_overview_stem(config.outputs.rgb, bands)
    m0416=cluster.lower()=='m0416'
    if m0416 and (candidates is None or photometry is None):
        raise ValueError('The M0416 overview needs the saved candidates of its _candidates variant (candidate_catalog) '
                         'and the saved marker photometry (read_marker_photometry)')
    reference_image=images[cutouts.rgb_reference_band(bands)]
    geometry=overview_geometry(reference_image,catalog,extra,reference,resolution=768)
    output.parent.mkdir(parents=True, exist_ok=True)
    shared_grf=bool(shared_grf)
    det=np.asarray(determinant)
    band=(lower<0)&(upper>0)
    log(f'[outputs] rendering {cluster} RGB critical overview')
    rgb=cutouts.rgb_cutout(images,bands,display,geometry['center'],geometry['size'])
    rgb=aligned_rgb(rgb,reference_image,geometry)
    plt.style.use(str(Path(__file__).with_name('style.mplstyle')))
    position_segments = recovered_contour_segments(contours)
    contoured = len({str(identity) for identity, _ in position_segments})
    written = []
    main_output = output
    for show_candidates in ((False, True) if m0416 else (False,)):
        output = main_output.with_name(main_output.stem + '_candidates.pdf') if show_candidates else main_output
        # Match the M0416 candidate overview's page and map footprint for
        # side-by-side paper figures. Expand the sky crop, never stretch pixels.
        matched_page = cluster.lower() == 'as1063'
        page_width, page_height = 612.8157348632812, 551.6261596679688
        map_left, map_top, map_right, map_bottom = 113.94043731689453, 6.9825439453125, 608.6768798828125, 506.11749267578125
        if matched_page:
            fig=plt.figure(figsize=(page_width/72.,page_height/72.))
            ax=fig.add_axes([map_left/page_width, (page_height-map_bottom)/page_height,
                             (map_right-map_left)/page_width, (map_bottom-map_top)/page_height],
                            projection=geometry['wcs'])
        else:
            fig=plt.figure(figsize=(9,9))
            ax=fig.add_subplot(111,projection=geometry['wcs'])
        n=geometry['resolution']
        ax.imshow(rgb,origin='lower',extent=(-.5,n-.5,-.5,n-.5),interpolation='none')
        if band is not None:
            draw_critical_interval(ax,lower,upper,CRITICAL_COLOR,(-.5,n-.5,-.5,n-.5))
        ax.contour(det,levels=[0.],colors=[CRITICAL_COLOR],linewidths=.65,
                   origin='lower',extent=(-.5,n-.5,-.5,n-.5))
        px,py=grid_pixels(geometry,catalog.x_obs_arcsec.to_numpy(),catalog.y_obs_arcsec.to_numpy())
        matched=photometry['table'] if m0416 else marker_photometry(catalog, reference)
        marker_table=overview_marker_photometry(config, catalog, matched, stem)
        sizes=marker_table.marker_area_pt2.to_numpy()
        valid=marker_table.valid_photometry.to_numpy()
        candidate_path=Path(candidates['path']) if show_candidates else None
        shown_candidates=candidates['candidates'] if show_candidates else []
        family_colors={'K66':'#40DFFF','K67':'#FF70C5','K20':'#FF6060',
                       'K28':'#BD9AFF','K34':'#609FFF','K115':'#FFFFFF','A':'#32E875'}
        systems=catalog.family_id.astype(str).str.extract(r'^(K\d+|C\d+)',expand=False)
        selected={r['system'] for r in shown_candidates if r['kind']=='counterimage'}
        colors=[family_colors.get(system,'#FF9500') if system in selected else '#FF9500' for system in systems]
        missed = ((catalog['image_recovery_status'].astype(str) != 'recovered').to_numpy()
                  if 'image_recovery_status' in catalog else np.zeros(len(catalog), bool))
        colors=[NOT_RECOVERED_COLOR if miss else color for miss, color in zip(missed, colors)]
        ax.scatter(px,py,s=sizes,marker='s',
                   facecolors='none',edgecolors=colors,linewidths=.8,zorder=25)
        has_extra = extra is not None and not extra.empty
        if has_extra:
            ex, ey = grid_pixels(geometry, extra['x_model_arcsec'].to_numpy(float), extra['y_model_arcsec'].to_numpy(float))
            ax.scatter(ex, ey, s=16, marker='s', facecolors='none', edgecolors=EXTRA_IMAGE_COLOR, linewidths=.8, zorder=25)
        for _, segments in position_segments:
            for segment in segments:
                sx, sy = grid_pixels(geometry, segment[:, 0], segment[:, 1])
                line, = ax.plot(sx, sy, color='cyan', linewidth=.45,
                                antialiased=True, solid_joinstyle='round', zorder=20)
                line.get_path().should_simplify = False
        marker_table['system']=systems.to_numpy()
        marker_table['color']=colors
        overlay_rows=[]
        all_px,all_py=list(px),list(py)
        for system in dict.fromkeys(r['system'] for r in shown_candidates):
            color=family_colors.get(system,'#42A5FF')
            label='New family A' if system=='A' else f'{system} candidates'
            rows=[r for r in shown_candidates if r['system']==system]
            if not rows:
                continue
            ra0,dec0=reference[1:]
            x=-(np.array([r['ra'] for r in rows])-ra0)*3600*np.cos(np.deg2rad(dec0))
            y=(np.array([r['dec'] for r in rows])-dec0)*3600
            cx,cy=grid_pixels(geometry,x,y)
            if not np.all((cx>=0)&(cx<n)&(cy>=0)&(cy<n)):
                raise ValueError('Candidate overlay extends beyond the RGB grid')
            ax.scatter(cx,cy,s=48,marker='D',facecolors='none',edgecolors=color,linewidths=1.,zorder=8,label=label)
            all_px.extend(cx);all_py.extend(cy)
            overlay_rows.extend(rows)
        if overlay_rows:
            store.write_table(tables,f'{stem}_candidate_overlays',pd.DataFrame(overlay_rows))
        # Include both fitted images and proposed candidates, with 3 arcsec padding.
        pad=3./geometry['size']*n
        limits=[float(min(all_px)-pad),float(max(all_px)+pad),float(min(all_py)-pad),float(max(all_py)+pad)]
        if matched_page:
            target_ratio=(map_right-map_left)/(map_bottom-map_top)
            width,height=limits[1]-limits[0],limits[3]-limits[2]
            cx,cy=(limits[0]+limits[1])/2.,(limits[2]+limits[3])/2.
            width=max(width,height*target_ratio)
            height=width/target_ratio
            limits=[cx-width/2.,cx+width/2.,cy-height/2.,cy+height/2.]
        ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:])
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.legend_handler import HandlerTuple
        handles = [
            Line2D([], [], marker='s', linestyle='none', markerfacecolor='none',
                   markeredgecolor='#FF9500', markeredgewidth=.8, markersize=5),
            Line2D([], [], marker='o', linestyle='none', markerfacecolor='none',
                   markeredgecolor='cyan', markeredgewidth=.45, markersize=5),
            (Patch(facecolor=CRITICAL_COLOR, alpha=.25, edgecolor='none'),
             Line2D([], [], color=CRITICAL_COLOR, linewidth=.65)),
        ]
        labels = ['Observed images', RECOVERED_CONTOUR_LABEL, 'Critical curve']
        for present, color, label in ((bool(missed.any()), NOT_RECOVERED_COLOR, 'Not recovered'), (has_extra, EXTRA_IMAGE_COLOR, 'Extra images')):
            if present:
                handles.insert(1, Line2D([], [], marker='s', linestyle='none', markerfacecolor='none',
                                         markeredgecolor=color, markeredgewidth=.8, markersize=5))
                labels.insert(1, label)
        candidate_handles, candidate_labels = ax.get_legend_handles_labels()
        if show_candidates:
            candidate_handles = [
                Line2D([], [], marker='D', linestyle='none', markerfacecolor='none',
                       markeredgecolor='white', markersize=5),
                Line2D([], [], marker='D', linestyle='none', markerfacecolor='none',
                       markeredgecolor=family_colors['A'], markersize=5),
            ]
            candidate_labels = ['Candidate counterimages', 'New family A']
        legend = ax.legend(handles + candidate_handles, labels + candidate_labels,
                           handler_map={tuple: HandlerTuple(ndivide=1)}, loc='lower left',
                           frameon=True, facecolor='black', edgecolor='none',
                           framealpha=.7, labelcolor='white', borderpad=.6)
        legend.set_zorder(40)
        bar_length = 5. * n / abs(geometry['xlim'][1] - geometry['xlim'][0])
        bar_right = limits[0] + .05 * (limits[1] - limits[0]) + bar_length
        bar_y = limits[3] - .07 * (limits[3] - limits[2])
        ax.plot([bar_right - bar_length, bar_right], [bar_y, bar_y],
                color='white', linewidth=2.5, solid_capstyle='butt', zorder=30)
        ax.annotate(r'$5^{\prime\prime}$', (bar_right - bar_length / 2, bar_y),
                    xytext=(0, 5), textcoords='offset points', ha='center',
                    va='bottom', color='white', fontsize=12, zorder=30)
        assert np.all((px>limits[0])&(px<limits[1])&(py>limits[2])&(py<limits[3]))
        store.write_table(tables,('hst_rgb_markers' if stem.startswith('hst') else 'jwst_rgb_marker_magnitudes') + ('_candidates' if show_candidates else ''),marker_table)
        log(f'[outputs] magnitude-scaled RGB markers: {valid.sum()}/{len(valid)}')
        ra,dec=ax.coords
        ra.set_axislabel('R.A. (J2000)');dec.set_axislabel('Dec. (J2000)')
        ra.set_major_formatter('hh:mm:ss');dec.set_major_formatter('dd:mm:ss')
        ra.set_ticks(number=5,color='white',direction='in',size=6,width=1)
        dec.set_ticks(number=5,color='white',direction='in',size=6,width=1)
        ra.set_ticks_position('bt');dec.set_ticks_position('lr')
        ra.set_ticklabel_position('b');dec.set_ticklabel_position('l')
        ra.display_minor_ticks(True);dec.display_minor_ticks(True)
        inset_info=None
        if matched_page:
            from matplotlib.transforms import Bbox
            fig.savefig(output,dpi=300,bbox_inches=Bbox.from_bounds(0,0,page_width/72.,page_height/72.),pad_inches=0)
        else:
            fig.savefig(output,dpi=300,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
        store.write_figure_record(output, dict(run=config.paths.run_name,image_position_marker='square',recovered_position_probability=RECOVERED_CONTOUR_PROBABILITY,recovered_position_contour_scale=RECOVERED_CONTOUR_SCALE,recovered_position_contour_centre='mean of the finite per-draw positions of the image',recovered_position_contours=contoured,not_recovered_images=int(missed.sum()),extra_images=int(len(extra)) if has_extra else 0,redrawn_from_tables=bool(redrawn),insets=inset_info,candidate_marker='diamond' if show_candidates else None,central_statistic='zero contour of pointwise posterior median determinant; same draws as band',z_source=9.,reporting_draws=int(reporting_draws),model='parametric + conditional GRF realization of each draw; no GP displacement' if shared_grf else 'parametric only; no shared GRF available',grf_field=grf_field,grf_derivatives=PLOT_FIELD_VERSION,bands=list(bands),catalog_images=len(catalog),resolution=n,determinant_resolution=int(det.shape[0]),critical_definition='median over draws of det(I - Hessian_draw) = 0',cache_fingerprint=str(fingerprint),marker_photometry=photometry['path'] if m0416 else None,magnitude_band=sorted(set(marker_table.loc[valid, 'magnitude_band'])),marker_photometry_catalog=str(config.model.image_constraints.catalog_path),matched_magnitudes=int(valid.sum()),missing_magnitude_policy='minimum marker area 9 pt^2',marker_area='clip(16*10**(-0.2*(m-26)),9,64) pt^2',crop_padding_arcsec=3.,crop_pixel_limits=limits,candidate_catalog=str(candidate_path) if candidate_path else None,candidate_overlays=[r['id'] for r in overlay_rows]))
        log(f"[outputs] RGB critical overview written: {output}")
        written.append(output)
    return written


# The redraw from a finished run's saved tables: no posterior, evaluator or determinant maps.

def read_saved_extra_images(tables):
    """The table ``image_recovery_extra_images`` as the pipeline holds it; the empty table if missing."""
    from ..diagnostics.image_recovery import common_extra_images
    try:
        return store.read_table(tables, 'image_recovery_extra_images')
    except store.MissingEntry:
        return common_extra_images(pd.DataFrame(), 1)


def saved_section(saved, name, path):
    """The section ``name`` (dotted, e.g. ``model.image_constraints``) of the saved configuration ``saved`` read from
    ``path``; a ValueError naming both when it is missing."""
    value = saved
    for key in name.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f'{path} has no {name}; the redraw cannot check or describe the run\'s picture without it')
        value = value[key]
    return value


def read_saved_config(run_dir):
    """The run's own configuration, ``final/artifacts/config.json``, as the dict it was saved as."""
    return json.loads((Path(run_dir) / 'final' / 'artifacts' / 'config.json').read_text())


def saved_grf_field(saved, path='final/artifacts/config.json'):
    """``(shared, label)`` of the run saved as ``saved`` (its ``config.json``): whether its evaluator has a knee mass
    field shared by every image, and :func:`~kappaccino.likelihood.grf_fields.shared_grf_label` of that evaluator.
    Decided by :func:`~kappaccino.likelihood.grf_fields.shared_field_evaluator` on a stand-in evaluator with the
    attributes ``Evaluator`` derives from the likelihood; a system-block run's all-block twin is never rebuilt."""
    from types import SimpleNamespace
    mode = saved_section(saved, 'likelihood.likelihood_mode', path)
    knee = mode == 'gp_image_plane' and saved_section(saved, 'likelihood.gp_kernel', path) == 'knee'
    likelihood = saved['likelihood']
    stand_in = SimpleNamespace(
        gp_knee=knee, gp_knee_spectrum=str(likelihood.get('gp_knee_spectrum', 'knee')),
        sigma_grf_index=0 if knee and saved_section(saved, 'likelihood.sigma_grf_prior', path) is not None else -1,
        state=SimpleNamespace(gp_blocks=str(saved_section(saved, 'likelihood.gp_blocks', path)) if knee else 'all'),
        _shared_field_twin=object())          # the cached all-block twin, so that it is not built
    return shared_field_evaluator(stand_in) is not None, shared_grf_label(stand_in)


def rebuild_run_config(run_dir):
    """The configuration of the finished run in ``run_dir``, rebuilt from ``inputs/command.json`` so that its paths
    point at ``run_dir``.  Refuses when the picture's inputs (the ``outputs.rgb`` section, ``model.image_constraints``,
    ``model.reference`` and ``paths.figure_prefix``) differ from the run's saved ``final/artifacts/config.json``."""
    from ..cli import build_config
    run_dir = Path(run_dir).resolve()
    command = json.loads((run_dir / 'inputs' / 'command.json').read_text())
    config = build_config(command['cluster'], profiles=tuple(command.get('profiles') or ()),
                          overrides=tuple(command.get('overrides') or ()), images=command.get('images'),
                          member_profile=command.get('member_profile'), member_slope=command.get('member_slope'),
                          run_label=run_dir.name, results=run_dir.parent.parent)
    folder = (Path(config.paths.output_dir) / config.paths.run_name).resolve()
    if folder != run_dir:
        raise ValueError(f'The rebuilt configuration writes to {folder}, not to the run folder {run_dir}')
    saved_path = run_dir / 'final' / 'artifacts' / 'config.json'
    saved = read_saved_config(run_dir)
    rebuilt = json.loads(json.dumps(config.to_json_dict()))
    for name in ('outputs.rgb', 'model.image_constraints', 'model.reference', 'paths.figure_prefix'):
        old = saved_section(saved, name, saved_path)
        new = saved_section(rebuilt, name, 'the rebuilt configuration')
        if old != new:
            raise ValueError(f'{name} of the rebuilt configuration differs from the run\'s final/artifacts/config.json '
                             f'(saved {old!r}, rebuilt {new!r}); the redraw would not draw the run\'s picture')
    return config


class SavedDraws:
    """The reporting draws as the saved tables know them: their posterior rows and fingerprint (all that
    :func:`~kappaccino.diagnostics.position_contours.load_or_compute_contours` reads)."""

    def __init__(self, indices, key):
        self.indices, self.key = np.asarray(indices, dtype=int), str(key)

    def __len__(self):
        return len(self.indices)


# What the redraw reads besides the band: the run's inputs (files) and its completed image recovery (final tables).
REDRAW_INPUTS = ('inputs/command.json', 'final/artifacts/config.json')
REDRAW_TABLES = ('image_fit_quality', 'posterior_image_residual_draws')


def check_saved_draws(indices, tables, stem):
    """Refuse band draw ``indices`` (arrays ``<stem>``) that the saved image recovery (table
    ``posterior_image_residual_draws``) does not hold: their position contours would be empty."""
    draws = store.read_table(tables, 'posterior_image_residual_draws')
    recovered = set(draws['posterior_index'].astype(int))
    missing = sorted(set(np.asarray(indices, dtype=int).tolist()) - recovered)
    if missing:
        shown = ', '.join(map(str, missing[:10])) + (', ...' if len(missing) > 10 else '')
        raise ValueError(f'{len(missing)} of the {len(indices)} draw indices of {stem} are not a posterior_index of '
                         f'posterior_image_residual_draws in {store.stage_dir(tables)} ({shown}); the band and the saved '
                         f'image recovery come from different draws')


def redraw_rgb_critical(run_dir, *, log=print):
    """Draw the RGB critical overviews of a finished run again from its saved results (``final/results.h5``): the band of the arrays ``<stem>``, the catalog ``image_fit_quality``, the extra
    images and the position contours of the saved image recovery.  Run from the repository root (the run's catalog and imaging paths are
    relative to it).  The model description of the records comes from the run's ``final/artifacts/config.json``.
    Refuses, before writing anything, a run without a completed image recovery and a band whose draws the image
    recovery does not hold.  Does not rewrite the arrays ``<stem>`` or the record ``<stem>_band``; rewrites the
    figures, their records, the marker tables and ``posterior_overview_positions`` (all in ``final/results.h5``).
    Returns the written PDF paths."""
    from types import SimpleNamespace
    from ..imaging import cutouts
    from ..imaging.cutouts import imaging_variants, load_cluster_imaging, variant_config, DEFAULT_HFF_RGB_BANDS
    from ..diagnostics.position_contours import load_or_compute_contours
    from .mass_figure import overview_geometry
    from . import plots
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f'{run_dir} is not a run folder')
    for name in REDRAW_INPUTS:
        if not (run_dir / name).exists():
            raise FileNotFoundError(f'Run {run_dir}: no {name}; the redraw needs a finished run with a completed image recovery')
    for name in REDRAW_TABLES:
        if not store.has_table(run_dir / 'final', name):
            raise FileNotFoundError(f'Run {run_dir}: no final table {name}; the redraw needs a finished run with a completed image recovery')
    config = rebuild_run_config(run_dir)
    shared_grf, grf_field = saved_grf_field(read_saved_config(run_dir), run_dir / 'final' / 'artifacts' / 'config.json')
    tables, figures = run_dir / 'final' / 'tables', run_dir / 'final' / 'figures'
    catalog = store.read_table(tables, 'image_fit_quality')
    extra = read_saved_extra_images(tables)
    reference = config.model.reference
    # ModelState.reference, as core.inputs.declared_model_inputs builds it.
    state = SimpleNamespace(reference=(int(reference.reference), float(reference.ra0_deg), float(reference.dec0_deg)))
    # Every band is read and its draws checked before anything is written.
    variants = []
    for image_config in imaging_variants(config):
        bands = tuple(image_config.bands) if image_config.bands else tuple(DEFAULT_HFF_RGB_BANDS)
        stem = rgb_overview_stem(image_config, bands)
        if not store.has_arrays(tables, stem):
            log(f'[redraw] {run_dir.name}: no {stem} arrays, {image_config.image_dir} skipped')
            continue
        saved = store.read_arrays(tables, stem)
        check_saved_draws(saved['draw_indices'], tables, stem)
        variants.append((image_config, bands, stem, saved))
    recovered = (int((catalog['image_recovery_status'].astype(str) == 'recovered').sum())
                 if 'image_recovery_status' in catalog else len(catalog))
    written = []
    for image_config, bands, stem, saved in variants:
        imaging = load_cluster_imaging(image_config, config.model, config.paths.run_name)
        if imaging[1] != bands or rgb_overview_stem(image_config, imaging[1]) != stem:
            raise ValueError(f'The imaging bands {imaging[1]} do not give the stem {stem}')
        reference_image = imaging[0][cutouts.rgb_reference_band(imaging[1])]
        expected = overview_geometry(reference_image, catalog, extra, state.reference)['x'].shape
        if tuple(saved['determinant'].shape) != tuple(expected):
            raise ValueError(f'{stem} holds a {saved["determinant"].shape} determinant, the band geometry is {expected}')
        draws = SavedDraws(saved['draw_indices'], np.asarray(saved['fingerprint']).item())
        view = variant_config(config, image_config)
        contours = load_or_compute_contours(view, state, draws, extra, tables, log=log)
        paths = draw_rgb_critical(view, state.reference, catalog, extra, imaging,
                                  plots.figure_path(figures, stem + '.pdf', config.paths.figure_prefix), tables,
                                  determinant=saved['determinant'], lower=saved['lower'], upper=saved['upper'],
                                  contours=contours,
                                  fingerprint=draws.key, reporting_draws=len(draws), shared_grf=shared_grf,
                                  grf_field=grf_field, redrawn=True, log=log,
                                  candidates=store.read_record(tables, candidate_record(stem)) if imaging[3].lower() == 'm0416' else None,
                                  photometry=read_marker_photometry(tables, stem) if imaging[3].lower() == 'm0416' else None)
        contoured = store.read_figure_record(paths[0])['recovered_position_contours']
        if recovered and not contoured:
            log(f'[redraw] {stem}: no recovered-position contour drawn although {recovered} images are recovered')
        written += paths
    return written


def main(argv=None):
    """``python -m kappaccino.plotting.jwst_rgb RUN_DIR [RUN_DIR ...]``: :func:`redraw_rgb_critical` of each run."""
    import argparse
    parser = argparse.ArgumentParser(prog='python -m kappaccino.plotting.jwst_rgb',
                                     description='Redraw the RGB critical overviews of finished runs from their saved tables.')
    parser.add_argument('run_dirs', nargs='+', metavar='RUN_DIR', help='a run folder, e.g. results/hera/<run name>')
    run_dirs = parser.parse_args(argv).run_dirs
    missing = [run_dir for run_dir in run_dirs if not Path(run_dir).is_dir()]
    if missing:
        parser.error('not a run folder: ' + ', '.join(missing))
    for run_dir in run_dirs:
        for path in redraw_rgb_critical(run_dir):
            print(f'[redraw] written: {path}')
    return 0


def draw_critical_interval(ax, lower, upper, color, extent=None, *, coordinates=None):
    """Shade the intersection lower<=0 and upper>=0, preserving subpixel bands.

    Interpolating min(-lower, upper) first can erase a narrow interval whose
    two boundaries cross a pixel. Intersect their separately contoured regions.
    """
    style=dict(origin='lower',extent=extent) if coordinates is None else {}
    args=() if coordinates is None else tuple(coordinates)
    if np.min(lower)>=0 or np.max(upper)<=0:
        return None
    bound=max(float(np.max(np.abs(lower))),float(np.max(np.abs(upper))),1.)+1.
    fill=ax.contourf(*args,lower,levels=[-bound,0.],colors=[color],alpha=.25,zorder=2,**style)
    clip=ax.contourf(*args,upper,levels=[0.,bound],colors='none',**style)
    fill.set_clip_path(clip.get_paths()[0],ax.transData)
    fill.set_clip_box(ax.bbox)  # Keep the interval clipped to the displayed RGB crop too.
    clip.remove()
    return fill


def critical_quantile_summary(determinants):
    """Matched 16/50/84 percentiles of the same determinant draws."""
    return np.quantile(determinants, [.16, .5, .84], axis=0)


def critical_quantile_band(determinants):
    """Pointwise region where the central 68% determinant interval straddles zero."""
    lower,_,upper=critical_quantile_summary(determinants)
    return lower,upper,(lower<0)&(upper>0)



if __name__ == '__main__':
    raise SystemExit(main())
