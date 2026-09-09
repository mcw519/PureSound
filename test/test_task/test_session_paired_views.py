"""Explicit chain views share all source randomness and do not duplicate rows."""
import random

import numpy as np
import pytest
import torch

from puresound.config.augmentation import SessionRowsConfig
from puresound.task.device_chain import ChainResult
from puresound.task.voice_isolation import VoiceIsolationCollateFunc
from test.test_task.test_session_rows import corpus as _corpus, _dataset, _row, _session


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return _corpus.__wrapped__(tmp_path_factory)


def _capture_chain(dataset, monkeypatch):
    inputs = []

    def apply(noisy, target, *, sample_rate):
        inputs.append((noisy.clone(), target.clone()))
        gain = 0.5 if len(inputs) % 2 else 0.25
        # Exercise in-place stages: view two must not receive view one's edits.
        noisy.mul_(gain)
        target.mul_(gain)
        return ChainResult(noisy, target, {})

    monkeypatch.setattr(dataset.device_chain, 'apply', apply)
    return inputs


def test_pair_uses_exact_prechain_inputs_and_preserves_primary_batch(corpus, monkeypatch):
    ds = _dataset(corpus, _session(paired_view_prob=1.0), reverb=False)
    inputs = _capture_chain(ds, monkeypatch)
    row = _row(ds, 15551)
    assert len(inputs) == 2
    assert all(torch.equal(a, b) for a, b in zip(inputs[0], inputs[1]))
    pair = row['paired_view']
    assert torch.equal(pair['noisy_speech'], row['noisy_speech'] * 0.5)
    assert torch.equal(pair['clean_speech'], row['clean_speech'] * 0.5)
    assert int(pair['row_source_id']) == int(row['row_source_id']) >= 0
    assert bool((pair['turn_chain'] != row['turn_chain']).all())
    assert 'turn_id' not in pair  # index primary time labels; never regenerate
    plain = _row(ds, 15552, seconds=6.0)
    batch = VoiceIsolationCollateFunc()([plain, row])
    assert batch['noisy_speech'].shape[0] == 2
    assert batch['paired_view']['noisy_speech'].shape[0] == 1
    assert batch['paired_view']['source_indices'].tolist() == [1]
    assert torch.equal(batch['turn_id'][1], row['turn_id'])


def test_identical_chain_draw_is_not_an_effective_pair(corpus):
    ds = _dataset(corpus, _session(paired_view_prob=1.0), reverb=False)
    row = _row(ds, 15553)
    assert 'paired_view' not in row
    assert int(row['row_source_id']) == -1
    assert 'paired_view' not in VoiceIsolationCollateFunc()([row])


def test_default_off_preserves_waveforms_labels_and_all_rng_streams(corpus):
    implicit = _dataset(corpus, _session(), reverb=False)
    explicit = _dataset(corpus, _session(paired_view_prob=0.0), reverb=False)
    first = _row(implicit, 15554)
    after_first = (torch.rand(4), np.random.rand(4), random.random())
    second = _row(explicit, 15554)
    after_second = (torch.rand(4), np.random.rand(4), random.random())
    assert first.keys() == second.keys()
    for key, value in first.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, second[key], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(after_first[0], after_second[0])
    assert np.array_equal(after_first[1], after_second[1])
    assert after_first[2] == after_second[2]


def test_short_bucket_does_not_draw_or_apply_extra_chain(corpus, monkeypatch):
    ds = _dataset(corpus, _session(paired_view_prob=1.0, paired_view_min_seconds=30.0),
                  reverb=False)
    inputs = _capture_chain(ds, monkeypatch)
    row = _row(ds, 15555)
    assert len(inputs) == 1
    assert 'paired_view' not in row


def test_paired_view_configuration_rejects_ambiguous_or_invalid_contract():
    with pytest.raises(ValueError):
        SessionRowsConfig(enabled=True, pair_prob=1.0, paired_view_prob=1.0)
    with pytest.raises(ValueError):
        SessionRowsConfig(paired_view_min_seconds=0)
    with pytest.raises(ValueError):
        SessionRowsConfig(paired_view_prob=1.1)


def test_pair_turn_chain_padding_uses_primary_batch_turn_width():
    from puresound.task.session_rows import collate_session_labels

    primary = {
        'row_source_id': torch.tensor(-1),
        'turn_chain': torch.full((5,), 7, dtype=torch.long),
    }
    paired = {
        'row_source_id': torch.tensor(42),
        'turn_chain': torch.full((2,), 8, dtype=torch.long),
        'paired_view': {
            'row_source_id': torch.tensor(42),
            'turn_chain': torch.full((2,), 9, dtype=torch.long),
            'noisy_speech': torch.ones(1, 32),
            'clean_speech': torch.zeros(1, 32),
        },
    }
    batch = collate_session_labels([primary, paired], {})
    assert batch['paired_view']['turn_chain'].tolist() == [[9, 9, 0, 0, 0]]


def test_noise_suppression_subclass_pairs_without_session_labels(corpus, monkeypatch):
    from puresound.task.ns import NoiseSuppressionDataset, NoiseSuppressionCollateFunc

    class PairedNoiseDataset(NoiseSuppressionDataset):
        def _auxiliary_chain_view_probability(self, plan):
            return 1.0

    ds = PairedNoiseDataset(
        metafile_path=str(corpus), min_utt_length_in_seconds=1.0,
        min_utts_in_each_speaker=1, target_sr=16000,
        training_sample_length_in_seconds=2.0, audio_gain_normalized_to=-28,
        vad_label_args={'used': True, 'backend': 'energy', 'args': {'frame_length': 400, 'hop_length': 160}},
    )
    inputs = _capture_chain(ds, monkeypatch)
    row = _row(ds, 18801)
    assert len(inputs) == 2
    assert all(torch.equal(a, b) for a, b in zip(*inputs))
    assert 'turn_chain' not in row['paired_view']
    assert 'turn_role' not in row
    plain = {key: value for key, value in row.items()
             if key not in ('paired_view', 'row_source_id')}
    batch = NoiseSuppressionCollateFunc()([plain, row])
    assert batch['noisy_speech'].shape[0] == 2
    assert batch['paired_view']['source_indices'].tolist() == [1]
    assert batch['row_source_id'][0].item() == -1
    assert torch.equal(batch['row_source_id'][1:], batch['paired_view']['row_source_id'])


def test_generic_disabled_pairing_preserves_rng_and_primary_inplace_result():
    from puresound.task.paired_views import apply_chain_views

    class Chain:
        def apply(self, noisy, target, *, sample_rate):
            gain = torch.rand(())
            return ChainResult(noisy.mul_(gain), target.mul_(gain), {})

    torch.manual_seed(18802)
    expected = Chain().apply(torch.ones(1, 32), torch.ones(1, 32), sample_rate=16000)
    state = torch.get_rng_state()
    torch.manual_seed(18802)
    actual, paired = apply_chain_views(
        Chain(), torch.ones(1, 32), torch.ones(1, 32),
        sample_rate=16000, sample_length=32,
    )
    assert paired is None
    assert torch.equal(expected.noisy, actual.noisy)
    assert torch.equal(state, torch.get_rng_state())
