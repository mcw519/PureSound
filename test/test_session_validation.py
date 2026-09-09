"""Numerical acceptance definitions and fixed artifact replay (CPU only)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puresound.evaluation.session_validation import (
    assert_disjoint, digest, evaluate_session_manifest, persist_batch,
    presence_metrics, proximity_metrics, require_checkpoint_heads, target_projection,
)

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT/'egs/voice_isolate/benchmarks/probes'


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def labels():
    return {'turn_id': torch.tensor([[1, 1, 2, 2, 3, 3, 4, 4]]),
            'turn_role': torch.tensor([[7, 8, 8, 0]]),
            'turn_distance': torch.tensor([[2., .5, 2.1, float('nan')]]),
            'user_active': torch.zeros(1, 8), 'bystander_active': torch.zeros(1, 8)}


def test_proximity_true_distance_generic_roles_gap_and_chain_offset():
    lab = labels()
    p = torch.tensor([1., 1., 4., 4., 0., 0., 100., 100.])
    result = proximity_metrics(p, lab, p+123)
    assert result['ordering_pairs'] == 1  # near-equal2m vs2.1m excluded; padding excluded
    assert result['ordering_accuracy'] == 1
    assert result['margins'] == [3.]
    assert result['cross_chain_gap_absolute_error'] == [0.]
    assert result['saturation_fraction'] is None
    assert proximity_metrics(p, lab, pair_selection='all')['ordering_pairs'] == 2
    assert proximity_metrics(torch.zeros(8), lab)['ordering_correct'] == 0
    # Same role is excluded even if it is neither known voice role1 nor role2.
    lab['turn_role'][0, 1] = 7
    assert proximity_metrics(p, lab)['ordering_pairs'] == 1  # .5m(role7) vs2.1m(role8)


def test_proximity_overlap_grid_alignment_and_invalid_distance():
    lab = labels()
    lab['user_active'][0, 1] = lab['bystander_active'][0, 1] = 1
    p = torch.tensor([1., 100., 4., 4., 0., 0., 10., 10.])
    assert proximity_metrics(p, lab)['margins'] == [3.]
    assert proximity_metrics(p, lab, min_turn_frames=2)['ordering_pairs'] == 0
    lab['turn_distance'][0, 1] = -1
    assert proximity_metrics(p, lab)['ordering_pairs'] == 0
    with pytest.raises(ValueError, match='non-finite'):
        proximity_metrics(p*float('nan'), lab)


def test_presence_alignment_denominators_delays_and_censoring():
    target = [0, 1, 1, 1, 0, 1, 1, 0]
    # One-frame algorithmic delay is removed before scoring.
    scores = [-5, 5, -5, 5, 5, -5, -5, -5, -5]
    result = presence_metrics(scores, target, fps=10, delay_frames=1)
    assert result['positive_frames'] == 5
    assert result['negative_frames'] == 3
    assert result['false_negative_frames'] == 3
    assert result['false_positive_frames'] == 1
    assert result['onset_delay_seconds'] == [.1]
    assert result['censored_run_seconds'] == [.2]
    assert presence_metrics([-1, -1], [0, 0])['fnr'] is None


def test_projection_rejects_mixture_energy_as_preservation():
    target = np.array([1., -1., 1., -1.])
    orthogonal_noise = np.array([10., 10., 10., 10.])
    result = target_projection(.5*target+orthogonal_noise, target)
    assert result['gain'] == pytest.approx(.5)
    assert result['gain_db'] == pytest.approx(-6.0206, abs=.001)
    assert result['residual_to_target_db'] == pytest.approx(20.)
    assert target_projection(orthogonal_noise, np.zeros(4)) is None


class EvalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleNamespace()

    def forward(self, x):
        # A sample is a frame in this tiny deterministic numerical fixture.
        self.backbone.last_proximity = x
        self.backbone.last_vad_logits = torch.where(x > 0, 10., -10.)
        # Context-sensitive attenuation makes contextual-vs-fresh comparison observable.
        return x * (.5 if x.shape[-1] > 5 else 1.)


def fixture_manifest(tmp_path):
    signal = torch.tensor([[-1., -1., 1., 1., 0., 0., 1., 1.]])
    lab = {'turn_id': torch.tensor([[1, 1, 2, 2, 0, 0, 3, 3]]),
           'turn_role': torch.tensor([[2, 1, 1]]),
           'turn_distance': torch.tensor([[3., .5, .5]]),
           'user_active': (signal > 0).float(), 'bystander_active': (signal < 0).float()}
    batch = dict(lab, noisy_speech=signal, clean_speech=signal.clamp_min(0),
                 vad_target=(signal > 0).float(), session_shape=torch.tensor([3.]),
                 row_source_id=torch.tensor([9]), turn_chain=torch.tensor([[1, 1, 1]]))
    batch['paired_view'] = dict(noisy_speech=signal+1, clean_speech=signal.clamp_min(0),
                               source_indices=torch.tensor([0]), row_source_id=torch.tensor([9]),
                               turn_chain=torch.tensor([[2, 2, 2]]))
    entry = persist_batch(tmp_path, batch, row_id=0, seed=9, seconds=12.)
    manifest = dict(rows=[entry], sample_rate=2, fps=2,
                    speaker_pools={'train':['a'], 'validation':['b']})
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(manifest))
    return path, batch


def test_fixed_artifact_replay_reference_pair_and_mode_restoration(tmp_path):
    path, batch = fixture_manifest(tmp_path)
    model = EvalModel().train()
    first = evaluate_session_manifest(model, path)
    second = evaluate_session_manifest(model, path)
    assert first == second
    assert model.training
    assert first['summary']['all']['proximity']['ordering_accuracy'] == 1
    onsets = first['rows'][0]['onsets']
    assert [r['kind'] for r in onsets] == ['bystander_first', 'reentry']
    assert onsets[1]['gain_delta_db'] == pytest.approx(-6.0206, abs=.001)
    assert onsets[0]['samples'] == 2  # default one-second window
    json.dumps(first, allow_nan=False)
    with pytest.raises(FileExistsError):
        persist_batch(tmp_path, batch, row_id=0, seed=9, seconds=12.)
    (tmp_path/'row_00000.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='corrupt'):
        evaluate_session_manifest(model, path)
    assert model.training


def test_split_assertion_and_checkpoint_heads():
    assert_disjoint(['train'], ['valid'])
    with pytest.raises(ValueError, match='overlap'):
        assert_disjoint(['same'], ['same'])
    with pytest.raises(ValueError, match='empty'):
        assert_disjoint(['train'], [])
    model = torch.nn.Module()
    model.proximity_head = torch.nn.Linear(2, 1)
    model.vad_head = torch.nn.Linear(2, 1)
    require_checkpoint_heads(model, model.state_dict())
    with pytest.raises(ValueError, match='vad_head'):
        require_checkpoint_heads(model, {k:v for k,v in model.state_dict().items() if 'proximity' in k})
    malformed = model.state_dict(); malformed['proximity_head.weight'] = torch.ones(3, 3)
    with pytest.raises(ValueError, match='mismatched'):
        require_checkpoint_heads(model, malformed)


def test_readability_direct_proximity_and_legacy_head_math(tmp_path):
    sim = load_script('test_v20_anchor_sim', PROBES/'anchor_gate_sim.py')
    values = np.r_[np.full(100, 3.), np.full(100, -2.)]
    est = sim.sliding_proximity(values, np.ones(200, dtype=bool), [.5])[.5]
    assert est[49] == 3. and est[-1] == -2.
    # Existing MLP path remains a separate explicit computation.
    h = (np.ones((1, 1)), np.zeros(1), np.ones((3, 1)), np.zeros(3))
    assert np.allclose(sim.head_apply(h, np.ones((1, 1))), 1/(1+np.exp(-1)))
    tag = tmp_path/'v20'; tag.mkdir()
    meta = dict(spans={'keep':[[0, 1]],'suppress':[[1, 2]]}, offset_s=0., floor_dbfs=-60.)
    artifact = tag/'cache.npz'
    np.savez(artifact, proximity=values, mix=np.ones(32000)*.1, meta=json.dumps(meta))
    entry = dict(file=str(artifact), clip='cold_90D', side='near', condition='none')
    (tag/'field_index.jsonl').write_text(json.dumps(entry)+'\n')
    out = tmp_path/'readability.json'
    sim.cmd_readability(SimpleNamespace(cache=str(tmp_path), tags=['v20'], out=str(out), head='proximity'))
    report = json.loads(out.read_text())
    assert report['head'] == 'proximity'
    assert report['tags']['v20']['auc'][0]['auc_proximity'] > .9


def test_materialize_production_generator_fixed_seeds_and_disjoint_pool(tmp_path, monkeypatch, write_puresound_metafile):
    import puresound.config
    from puresound.config.augmentation import SessionRowsConfig, SpeechAugmentation, VadLabelConfig
    from puresound.task.device_chain import ChainResult, DeviceChain
    from puresound.config.base import with_overrides
    cli = load_script('test_v20_session_cli', PROBES/'v20_session_rows/session_validation.py')
    recipe_path = ROOT/'egs/voice_isolate/config/exp/train_dpcrn_v16_lengthmix.yaml'
    recipe = puresound.config.load_recipe(str(recipe_path))
    valid = write_puresound_metafile(tmp_path/'valid.csv', speakers=4, duration=4.)
    train = tmp_path/'train.csv'
    train.write_text(valid.read_text().replace(', corpus_spk', ', train_spk'))
    overrides = {key: None for key in type(recipe).model_fields if key.startswith('augmentation_')}
    overrides.update(dataset=dict(train_metafile=str(train), valid_metafile=str(valid), filter_min_utterance_length=1.),
        augmentation_speech=dict(used=True, is_target=False, prob=1., add_n_cases=[1, 2], snr_range=[-10., 10.]),
        augmentation_session_rows=SessionRowsConfig(enabled=True, prob=.5),
        vad_label=VadLabelConfig(used=True, backend='energy'))
    # Replace speech entirely so physical mix modes from v16 do not require an RIR.
    overrides['augmentation_speech'] = SpeechAugmentation.model_validate(overrides['augmentation_speech'])
    recipe = with_overrides(recipe, **overrides)
    monkeypatch.setattr(puresound.config, 'load_recipe', lambda *a, **kw: recipe)
    def chain(self, noisy, target, *, sample_rate):
        gain = .1+float(torch.rand(()))
        return ChainResult(noisy*gain, target*gain, {})
    monkeypatch.setattr(DeviceChain, 'apply', chain)
    first = cli.materialize(recipe_path, tmp_path/'first', rows_per_bucket=1)
    second = cli.materialize(recipe_path, tmp_path/'second', rows_per_bucket=1)
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert a == b
    assert [r['seconds'] for r in a['rows']] == [12., 30.]
    assert recipe.augmentation_session_rows.prob == .5
    assert recipe.augmentation_session_rows.paired_view_prob == 0.
    for entry in a['rows']:
        batch = torch.load(first.parent/entry['file'], weights_only=True)
        assert batch['noisy_speech'].shape[-1] == int(entry['seconds']*16000)
        assert batch['target_present'].item() == 1
        assert 'turn_distance' in batch and 'paired_view' in batch
        assert digest(first.parent/entry['file']) == entry['sha256']
    recipe = with_overrides(recipe, dataset=dict(train_metafile=str(valid)))
    with pytest.raises(ValueError, match='overlap'):
        cli.materialize(recipe_path, tmp_path/'overlap', rows_per_bucket=1)


def test_r6_cache_selects_direct_new_head_and_preserves_legacy(tmp_path):
    cache = load_script('test_v20_anchor_cache', PROBES/'anchor_gate_cache.py')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = SimpleNamespace(dist_head=SimpleNamespace(net=torch.nn.Linear(2, 3)))
        def forward(self, x):
            self.backbone.last_bottleneck = torch.ones(1, 2, 1, 4)
            self.backbone.last_proximity = torch.tensor([[7., 8., 9., 10.]])
            self.backbone.last_vad_logits = torch.ones(1, 4)
            return x
    model = Model()
    legacy = cache.run(model, torch.ones(1, 640), 'cpu')
    assert legacy['dist'].shape == (4, 3) and 'proximity' not in legacy
    model._anchor_cache_head = 'proximity'
    direct = cache.run(model, torch.ones(1, 640), 'cpu')
    assert direct['proximity'].tolist() == [7., 8., 9., 10.]
    assert 'dist' not in direct
    cache.save_head(model, tmp_path)
    assert (tmp_path/'proximity_head.json').exists()
    assert not (tmp_path/'dist_head_net.pt').exists()
