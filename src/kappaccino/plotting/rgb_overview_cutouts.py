"""Selected MACS J0416 posterior cutouts for the RGB critical overview."""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

from ..imaging.cutouts import arcsec_to_skycoord, arcsec_to_radec, cutout_window, rgb_cutout, rgb_reference_band
from .fold_pairs import _credible_density_grid, load_fold_data, critical_curve_quantiles
from .style import CRITICAL_COLOR
from ..io import store



def overview_cutout_data(tables, prefix, *, image_labels=(), system_ids=(), draws=None, small_lambda_critical=False,
                         critical=None, neighbors=False, group=None, log=print):
    """Reuse saved image draws and fold geometry, never run an image search.  ``neighbors`` adds to each image panel
    the other catalog images inside its field (``item['neighbors']``: label, family, observed position, density grid
    or None), which :func:`render_overview_cutouts` then draws and labels.

    The median det A of the image panels comes from one saved entry, named with its rows in ``critical_source`` of the
    result (``entry`` and ``index``; ``entry`` None without image panels).  With the reporting ``draws`` (a compute):
    the small-lambda figure's ``<ranking>_critical`` when ``small_lambda_critical`` (this compute wrote it) and it
    holds every image panel, else computed from the draws for all of them and saved as ``<prefix>_rgb_inset_critical``
    (with ``_<group>`` appended for a named group of panels, so that each group keeps its own).  Without draws (a
    render): the entry and rows ``critical`` (a ``critical_source`` the compute recorded), which must hold exactly these
    panels; a ValueError otherwise."""
    image_labels,system_ids=tuple(image_labels),tuple(system_ids)
    if not image_labels and not system_ids:
        return None
    tables=Path(tables)
    ranking_name=f'{prefix}_small_lambda_a_image_posteriors_jwst'
    ranking=store.read_table(tables,ranking_name).set_index('image_label')
    data = store.read_arrays(tables,'posterior_overview_positions')
    labels=data['image_labels'].astype(str).tolist()
    positions=data['recovered'].copy()
    folds=load_fold_data(tables)
    systems=[]
    for identity in system_ids:
        matches=[s for s in folds['systems']+folds['partial_systems'] if s['system_id']==identity]
        if len(matches)!=1:
            log(f'[outputs] RGB inset panels unavailable: expected one fold system {identity}, found {len(matches)}')
            return None
        systems.append(matches[0])
    if any(label not in labels or label not in ranking.index for label in image_labels):
        log('[outputs] RGB inset panels unavailable: selected image IDs absent from catalog')
        return None
    images=[]
    for label in image_labels:
        row=ranking.loc[label]
        grid=_credible_density_grid(positions[:,labels.index(label)])
        if grid is None:
            log(f'[outputs] RGB inset panels unavailable: no resolved posterior for {label}')
            return None
        # Derive framing even when a selected image lies beyond the ranking plot's limit.
        import contourpy
        curves=contourpy.contour_generator(x=grid['x'],y=grid['y'],z=grid['density']).lines(grid['levels'][0])
        if not curves:
            return None
        center=np.array([row.x_arcsec,row.y_arcsec])
        radius=np.max(np.abs(np.concatenate(curves)-center))
        fov=float(.5*np.ceil(max(3.,2.3*radius)/.5))
        item=dict(label=label,center=center,fov=fov,z=row.z_source,grid=grid)
        if neighbors:
            inside=ranking[(np.abs(ranking[['x_arcsec','y_arcsec']].to_numpy(float)-center).max(axis=1)<fov/2)
                           &(ranking.index!=label)&ranking.index.isin(labels)]
            item['family']=str(row.family_id)
            item['neighbors']=[dict(label=other,family=str(r.family_id),center=np.array([r.x_arcsec,r.y_arcsec]),
                                    grid=_credible_density_grid(positions[:,labels.index(other)]))
                               for other,r in inside.iterrows()]
        images.append(item)
    small_lambda=f'{ranking_name}_critical'
    inset=f'{prefix}_rgb_inset_critical'+(f'_{group}' if group else '')
    if not images:
        source=dict(entry=None,index=[])
    elif draws is None:
        source=critical
        if source is None or source['entry'] not in (small_lambda,inset) or len(source['index'])!=len(images):
            raise ValueError(f'The compute recorded no critical fields for the RGB cutout panels '
                             f'{[item["label"] for item in images]} ({source}); compute them with the reporting draws first')
        saved=store.read_arrays(tables,source['entry'])
        index=np.asarray(source['index'],dtype=int)
        matched=all(np.allclose(saved[key][index],[item[name] for item in images],rtol=0,atol=1e-10)
                    for key,name in (('centers','center'),('fovs','fov'),('redshifts','z')))
        if not matched:
            raise ValueError(f'{source["entry"]} does not hold the panels {[item["label"] for item in images]} at the rows '
                             f'the compute recorded ({source["index"]}); compute them with the reporting draws again')
        median,lower,upper=(saved[key][index] for key in ('median_determinant','lower','upper'))
    else:
        index=None
        if small_lambda_critical:
            saved=store.read_arrays(tables,small_lambda)
            index=[np.flatnonzero(np.all(np.isclose(saved['centers'],item['center'],rtol=0,atol=1e-10),axis=1)
                                  &np.isclose(saved['redshifts'],item['z'],rtol=0,atol=1e-10)
                                  &np.isclose(saved['fovs'],item['fov'],rtol=0,atol=1e-10)) for item in images]
            index=[int(match[0]) for match in index] if all(len(match)==1 for match in index) else None
        if index is not None:
            source=dict(entry=small_lambda,index=index)
            median,lower,upper=(saved[key][index] for key in ('median_determinant','lower','upper'))
        else:
            # Explicit launcher selections are independent of the standalone uncertainty cuts.
            from .critical_fields import median_cutout_determinants
            _,median,lower,upper=median_cutout_determinants(draws,
                [item['center'] for item in images],[item['fov'] for item in images],
                [item['z'] for item in images],tables/f'{inset}.h5',log=log)
            source=dict(entry=inset,index=list(range(len(images))))
    for i,item in enumerate(images):
        item['critical']=dict(median_determinant=median[i],lower=lower[i],upper=upper[i])
    return dict(images=images,systems=systems,critical_source=source)


def render_overview_cutouts(fig, data, imaging, frame, *, image_rects=None, system_rects=None):
    """Four panels left-to-right above; four folds top-to-bottom at right."""
    images,bands,display,_=imaging
    reference=images[rgb_reference_band(bands)]
    palette=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#F0E442','#7B61A8']
    def panel(rect,center,fov,label,z):
        ax=fig.add_axes(rect)
        sky=arcsec_to_skycoord(*center,frame)
        window=cutout_window(reference,sky,fov)
        if window is None:
            raise ValueError(f'No imaging coverage for RGB inset {label}')
        rgb=rgb_cutout(images,bands,display,sky,fov)
        ax.imshow(rgb,origin='lower',interpolation='nearest')
        def pixels(x,y):
            ra,dec=arcsec_to_radec(x,y,frame)
            px,py=reference.wcs.world_to_pixel_values(ra,dec)
            return px-window.x0,py-window.y0
        ax.text(.03,.97,label+'\n'+rf'$z={z:g}$',transform=ax.transAxes,
                va='top',color='white',fontsize=7,
                bbox=dict(facecolor='black',alpha=.5,edgecolor='none',pad=2))
        size=rgb.shape[0];bar=.5/reference.pixel_scale_arcsec
        ax.plot([.06*size,.06*size+bar],[.07*size]*2,color='white',lw=.8)
        ax.text(.06*size+bar/2,.10*size,r'$0.5^{\prime\prime}$',ha='center',color='white',fontsize=6)
        ax.set(xlim=(-.5,rgb.shape[1]-.5),ylim=(-.5,size-.5),xticks=[],yticks=[])
        ax.grid(False)
        return ax,pixels
    def contour(ax,grid,pixels,color):
        if grid is not None:
            ax.contour(*pixels(grid['x'],grid['y']),grid['density'],
                       levels=[grid['levels'][0]],colors=[color],linewidths=.65,linestyles='-')
    for i,item in enumerate(data['images']):
        rect = image_rects[i] if image_rects is not None else [.07+i*.18,.80,.18,.18]
        # 'frame_center' (optional) frames the panel elsewhere than on the image, e.g. on a galaxy-scale deflector.
        frame_center=item.get('frame_center',item['center'])
        ax,pixels=panel(rect,frame_center,item['fov'],item['label'],item['z'])
        from .jwst_rgb import draw_critical_interval
        critical=item['critical'];det=critical['median_determinant']
        axis=np.linspace(-.5,.5,det.shape[0])
        x,y=np.meshgrid(frame_center[0]+axis*item['fov'],frame_center[1]+axis*item['fov'])
        coordinates=pixels(x,y)
        draw_critical_interval(ax,critical['lower'],critical['upper'],CRITICAL_COLOR,coordinates=coordinates)
        if det.min()<0<det.max():
            ax.contour(*coordinates,det,levels=[0],colors=[CRITICAL_COLOR],linewidths=.65)
        # The panel's image and its family in orange; each other family in the field takes its own color.
        others=iter(['#009E73','#CC79A7','#F0E442','#56B4E9','#7B61A8'])
        colors={item.get('family'):'tab:orange'}
        label_points=[]
        for image in [item,*item.get('neighbors',[])]:
            if image.get('family') not in colors:
                colors[image['family']]=next(others,'#FFFFFF')
            color=colors[image.get('family')]
            contour(ax,image['grid'],pixels,color)
            point=pixels(*image['center'])
            ax.plot(*point, marker='s', linestyle='none', markerfacecolor='none', markeredgecolor=color,
                    markersize=4, markeredgewidth=.8, zorder=8)
            label_points.append((image['label'],point,color))
        if 'neighbors' in item:
            from .apj_fold_cutouts import label_catalog_images
            label_catalog_images(ax,label_points)
    for i,s in enumerate(data['systems']):
        rect = system_rects[i] if system_rects is not None else [.80,.61-i*.18,.18,.18]
        ax,pixels=panel(rect,s['center'],s['fov'],s.get('panel_label',f"System {s['system_id']}"),s['z_source'])
        if 'partial_reason' in s:  # no fold anchor: median det A = 0 line, band along its normal (median_critical_band)
            points,normals,band=(np.asarray(s[k]) for k in ('critical_points','critical_normals','critical_band'))
            for segment in np.unique(s['critical_segments']):
                index=np.flatnonzero(np.asarray(s['critical_segments'])==segment)
                ax.plot(*pixels(*points[index].T),color=CRITICAL_COLOR,lw=.65)
                shown=index[np.isfinite(band[index]).all(1)]
                for run in np.split(shown,np.flatnonzero(np.diff(shown)>1)+1):
                    if len(run)>1:
                        vertices=np.vstack((points[run]+band[run,:1]*normals[run],(points[run]+band[run,1:]*normals[run])[::-1]))
                        ax.add_patch(Polygon(np.column_stack(pixels(*vertices.T)),facecolor=CRITICAL_COLOR,alpha=.18,edgecolor='none'))
        else:
            curves=s.get('critical_grf_local_draws',s['critical'])
            for low,median,high in critical_curve_quantiles(curves,s['seeds'],s['origin'],s['normal']):
                vertices=np.vstack((low,high[::-1]))
                ax.add_patch(Polygon(np.column_stack(pixels(*vertices.T)),facecolor=CRITICAL_COLOR,alpha=.18,edgecolor='none'))
                ax.plot(*pixels(*median.T),color=CRITICAL_COLOR,lw=.65)
        label_points=[]
        for j,label in enumerate(s['labels']):
            color=palette[s['image_family_index'][j]%len(palette)]
            contour(ax,_credible_density_grid(s['positions'][:,j]),pixels,color)
            point=pixels(*s['observed'][j])
            ax.plot(*point,marker='s',linestyle='none',color=color,
                    markerfacecolor='none',ms=4,mew=.8,zorder=8)
            label_points.append((label,point,color))
        from .apj_fold_cutouts import label_catalog_images
        label_catalog_images(ax,label_points)
    return dict(top=[item['label'] for item in data['images']],right=[s['system_id'] for s in data['systems']],
                position_probability=.954499736,position_contours='predicted observed positions; 2 sigma only; conditional on recovery')
