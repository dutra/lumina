from dataclasses import asdict
from types import SimpleNamespace as NS
import json
from kappaccino.io import store

import numpy as np
import pytest

from kappaccino.plotting import paper_cutouts as pc
from kappaccino.config import ImageCatalogCutoutConfig


def system():
    return dict(system_id='K6', family_ids=['K6A', 'K6B'],
                labels=['K6A.a','K6B.a','K6A.b','K6B.b'],
                image_family_index=np.array([0,1,0,1]),
                observed=np.arange(8).reshape(4,2), positions=np.arange(80).reshape(10,4,2))


def test_selectors_preserve_order_and_select_only_family():
    original = system()
    kind, identity, selected = pc.resolve_panel('family:K6B', [original])
    assert (kind, identity) == ('family', 'K6')
    assert selected['labels'] == ['K6B.a','K6B.b']
    np.testing.assert_array_equal(selected['positions'], original['positions'][:,[1,3]])
    assert original['positions'].shape == (10,4,2)
    assert pc.resolve_panel('image:K34.b', [original]) == ('image', 'K34.b', None)
    assert pc.resolve_panel('system:K6', [original])[2]['labels'] == original['labels']
    for selector in ['K6', 'family:missing', 'system:missing', 'image:']:
        with pytest.raises(ValueError):
            pc.resolve_panel(selector, [original])


def test_writer_uses_saved_panels_and_preserves_selector_order(tmp_path, monkeypatch):
    selections = ('family:K6B','image:K34.b','system:K6')
    cfg = ImageCatalogCutoutConfig(paper_small_lambda_cutouts=selections)
    assert asdict(cfg)['paper_small_lambda_cutouts'] == selections
    config = NS(image_catalog=cfg, paths=NS(figure_prefix='test',run_name='test'))
    monkeypatch.setattr(pc,'load_fold_data',lambda _:dict(systems=[system()]))
    calls=[]
    def cached(*args, **kwargs):
        assert kwargs['image_labels'] == ('K34.b',)
        assert kwargs['system_ids'] == ('K6',)
        assert kwargs['group'] == 'small_lambda_posteriors'      # each figure saves its own inset fields
        return dict(images=[dict(label='K34.b')],systems=[system()])
    monkeypatch.setattr(pc,'overview_cutout_data',cached)
    def render(fig,data,*args,**kwargs):
        calls.append((data,kwargs['image_rects'][0]))
        fig.add_axes(kwargs['image_rects'][0])
    monkeypatch.setattr(pc,'render_overview_cutouts',render)
    paths=pc.write_paper_cutouts(config,NS(reference=(3,0,0)),None,tmp_path,tmp_path,
                                ({},(),None,'m0416'),log=lambda _:None)
    assert len(paths)==1 and paths[0].name=='m0416_small_lambda_posteriors.pdf'
    assert calls[0][0]['systems'][0]['labels']==['K6B.a','K6B.b']
    assert calls[1][0]['images'][0]['label']=='K34.b'
    np.testing.assert_allclose([entry[1][0] for entry in calls],[0,1/3,2/3])
    assert store.read_figure_record(paths[0])['selectors']==list(selections)


def test_empty_selections_do_not_load_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(pc,'load_fold_data',lambda _:pytest.fail('disabled figure read diagnostics'))
    assert pc.write_paper_cutouts(NS(image_catalog=ImageCatalogCutoutConfig()),None,None,
                                 tmp_path,tmp_path,None)==[]


def test_figure_without_a_saved_posterior_is_skipped_not_fatal(tmp_path, monkeypatch):
    cfg = ImageCatalogCutoutConfig(paper_small_lambda_cutouts=('image:K34.b',), paper_fold_cutouts=('system:K6',))
    config = NS(image_catalog=cfg, paths=NS(figure_prefix='test',run_name='test'))
    monkeypatch.setattr(pc,'load_fold_data',lambda _:dict(systems=[system()]))
    monkeypatch.setattr(pc,'overview_cutout_data',
                        lambda *a, **k: None if k['image_labels'] else dict(images=[],systems=[system()]))
    monkeypatch.setattr(pc,'render_overview_cutouts',lambda fig,*a,**k: fig.add_axes(k['image_rects'][0]))
    lines=[]
    paths=pc.write_paper_cutouts(config,NS(reference=(3,0,0)),None,tmp_path,tmp_path,
                                ({},(),None,'m0416'),log=lines.append)
    assert [path.name for path in paths]==['m0416_fold_cutouts.pdf']
    assert '[outputs] paper cutouts skipped: small_lambda_posteriors' in lines


def partial(identity='K8'):
    return dict(system(), system_id=identity, family_ids=[f'{identity}A', f'{identity}B'], partial_reason='changes branch')


def test_partial_system_is_drawn_reduced_and_unknown_selector_skips_figure(tmp_path, monkeypatch):
    cfg = ImageCatalogCutoutConfig(paper_small_lambda_cutouts=('system:K6','system:K9'),
                                   paper_fold_cutouts=('system:K6','system:K8','family:K8B'))
    config = NS(image_catalog=cfg, paths=NS(figure_prefix='test',run_name='test'))
    monkeypatch.setattr(pc,'load_fold_data',lambda _:dict(systems=[system()],partial_systems=[partial()]))
    monkeypatch.setattr(pc,'overview_cutout_data',lambda *a, **k: dict(images=[],systems=[]))
    drawn=[]
    def render(fig,data,*a,**k):
        drawn.append(data['systems'][0]); fig.add_axes(k['image_rects'][0])
    monkeypatch.setattr(pc,'render_overview_cutouts',render)
    lines=[]
    paths=pc.write_paper_cutouts(config,NS(reference=(3,0,0)),None,tmp_path,tmp_path,
                                ({},(),None,'m0416'),log=lines.append)
    assert [path.name for path in paths]==['m0416_fold_cutouts.pdf']
    assert "[outputs] paper cutouts skipped: small_lambda_posteriors (Paper cutout 'system:K9' has no saved fold region)" in lines
    assert [s['system_id'] for s in drawn]==['K6','K8','K8'] and drawn[2]['labels']==['K6B.a','K6B.b']
    assert "[outputs] paper cutouts fold_cutouts: system:K8 drawn without its fold anchor (changes branch)" in lines
    assert "[outputs] paper cutouts fold_cutouts: family:K8B drawn without its fold anchor (changes branch)" in lines
    record=store.read_figure_record(paths[0])
    assert record['panels_without_fold_anchor']==[dict(selector='system:K8',system='K8',reason='changes branch'),
                                                  dict(selector='family:K8B',system='K8',reason='changes branch')]
    assert record['partial_critical_band']==pc.PARTIAL_CRITICAL_BAND
    with pytest.raises(ValueError, match='exactly one'):
        pc.resolve_panel('system:K6', [system(), system()])
