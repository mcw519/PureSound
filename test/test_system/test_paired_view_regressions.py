"""Regression checks for v20 collation and acceptance gates; no training or assets."""
from types import SimpleNamespace
import sys

import pytest
import torch

from puresound.task.paired_views import collate_paired_views
from puresound.system.paired_views import paired_view_loss, PairedViewConsistencyConfig
from egs.voice_isolate.scripts import preflight_ckpt_recipe as preflight


def test_short_auxiliary_uses_primary_frame_grid_and_backpropagates():
    class Objective:
        paired_output = 'enhanced'
        def paired_consistency(self, primary, secondary, batch, view):
            return (primary[view['source_indices']] - secondary).square().mean(), 1, 1
    class Module:
        backbone = SimpleNamespace()
        loss_func_list = [Objective()]
        loss_func_list_w = [1.]
        def forward(self, x):
            return x * 2
        def _loss_providers(self, *, enhanced, **kwargs):
            return {'enhanced': lambda: enhanced}
        def log(self, *args, **kwargs):
            pass
    x = torch.ones(2, 10, requires_grad=True)
    rows = [{'row_source_id': 1, 'paired_view': {'row_source_id': 1,
             'noisy_speech': torch.ones(8), 'clean_speech': torch.ones(8)}}, {}]
    batch = collate_paired_views(rows, {'noisy_speech': x, 'clean_speech': x.detach()})
    assert batch['paired_view']['noisy_speech'].shape == (1, 10)
    assert torch.equal(batch['paired_view']['noisy_speech'][0, 8:], torch.zeros(2))
    value = paired_view_loss(Module(), batch, x, PairedViewConsistencyConfig(enabled=True))
    value.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('allowed, mismatch, expected', [([], False, 1),
    (['vad_head'], False, 1), (['vad_head', 'proximity_head'], False, 0),
    (['vad_head', 'proximity_head'], True, 1)])
def test_preflight_requires_explicit_missing_heads(monkeypatch, allowed, mismatch, expected):
    state = {'backbone.encoder.weight': torch.zeros(2),
             'backbone.vad_head.weight': torch.zeros(2),
             'backbone.proximity_head.weight': torch.zeros(2)}
    checkpoint = {'backbone.encoder.weight': torch.zeros(3 if mismatch else 2)}
    argv = ['preflight', '--ckpt', 'checkpoint', 'recipe']
    for head in allowed:
        argv += ['--allow-missing-head', head]
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(preflight, 'load_recipe', lambda *a, **kw: SimpleNamespace(model={}))
    monkeypatch.setattr(preflight, 'init_siso_model', lambda *a: SimpleNamespace(state_dict=lambda: state))
    monkeypatch.setattr(preflight.torch, 'load', lambda *a, **kw: checkpoint)
    assert preflight.main() == expected


