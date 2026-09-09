"""Regression checks for v20 collation and acceptance gates; no training or assets."""
from types import SimpleNamespace
import sys

import pytest
import torch

from puresound.task.paired_views import collate_paired_views
from puresound.system.paired_views import paired_view_loss, PairedViewConsistencyConfig
from egs.voice_isolate.scripts import preflight_ckpt_recipe as preflight
from egs.voice_isolate.benchmarks.probes.v20_session_rows import no_update_memory_smoke as smoke
from egs.voice_isolate.benchmarks.probes.v20_session_rows import audit_300_rows as audit


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


def test_smoke_rejects_unexercised_paired_path():
    with pytest.raises(RuntimeError, match='no effective'):
        smoke.validate_coverage('paired', 0)
    smoke.validate_coverage('paired', 1)
    smoke.validate_coverage('baseline', 0)


def test_audit_preserves_schedule_and_clips_final_batch(monkeypatch, tmp_path):
    import egs.voice_isolate.main as main
    from puresound.config import load_recipe
    root = audit.ROOT
    path = root / 'egs/voice_isolate/config/exp/train_dpcrn_v20_r1a.yaml'
    original = load_recipe(path, expected_task='voice_isolation', expected_purpose='train')
    expected = [b.model_dump() for b in original.trainer.length_schedule]
    def loader(recipe):
        assert [b.model_dump() for b in recipe.trainer.length_schedule] == expected
        assert recipe.trainer.train_iter_per_epoch >= 3
        def batches():
            for _ in range(3):
                yield {'noisy_speech': torch.zeros(3, 10), 'length': torch.full((3,), 480000),
                       'paired_view': {'source_indices': torch.tensor([0, 2])}}
        return batches(), None
    monkeypatch.setattr(main, 'init_dataloader', loader)
    monkeypatch.chdir(tmp_path)  # restores cwd after the audit changes it
    result = audit.run(str(path), rows=5, out=None, num_workers=0)
    assert result['rows'] == 5
    assert result['paired_rows'] == 3
    assert result['bucket_schedule'] == expected


def test_smoke_forces_coverage_without_changing_source_recipe():
    from puresound.config import load_recipe
    path = audit.ROOT / 'egs/voice_isolate/config/exp/train_dpcrn_v20_r1a.yaml'
    recipe = load_recipe(path, expected_task='voice_isolation', expected_purpose='train')
    forced = smoke.coverage_recipe(recipe, 12)
    assert forced.augmentation_session_rows.prob == 1
    assert forced.augmentation_session_rows.paired_view_prob == 1
    assert recipe.augmentation_session_rows.prob == .5
    assert forced.augmentation_session_rows.shape_probs == recipe.augmentation_session_rows.shape_probs
    with pytest.raises(ValueError, match='duration'):
        smoke.coverage_recipe(recipe, 6)
