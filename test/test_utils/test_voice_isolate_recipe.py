import torch

from puresound.audio import augmentaion as aug_mod
from puresound.audio.augmentaion import AudioEffectAugmentor
from puresound.task.ns import sample_query_distance_overrides
from puresound.task.sampler import SpeakerSampler


def test_query_distance_override_respects_simulator_role_ranges():
    # Stage-A v2 query distances live around 1 m, but the simulator curriculum
    # still wants far-field interferers to start at 2 m.
    qd_cfg = {
        "used": True,
        "range": [1.0, 1.0],
        "peak_prob": 0.0,
        "near_floor": 0.3,
        "far_ceiling": 5.0,
    }
    sim_cfg = {
        "foreground_distance_range": [0.3, 1.0],
        "interferer_distance_range": [2.0, 5.0],
    }

    qd, fg_override, itf_override, near_floor = sample_query_distance_overrides(
        qd_cfg,
        sim_cfg,
    )

    assert qd == 1.0
    assert near_floor == 0.3
    assert fg_override == [0.3, 1.0]
    assert itf_override == [2.0, 5.0]


def test_speaker_sampler_uses_distinct_seeded_streams_per_rank():
    data = {
        f"spk{i}": {"utts": {f"utt{i}": {"sr": 16000}}}
        for i in range(8)
    }
    rank0 = list(
        SpeakerSampler(data, total_batch=4, n_spks=2, n_per=1, seed=123, rank=0, world_size=2)
    )
    rank1 = list(
        SpeakerSampler(data, total_batch=4, n_spks=2, n_per=1, seed=123, rank=1, world_size=2)
    )
    rank0_again = list(
        SpeakerSampler(data, total_batch=4, n_spks=2, n_per=1, seed=123, rank=0, world_size=2)
    )

    assert rank0 == rank0_again
    assert rank0 != rank1
    assert {item[2] for batch in rank0 for item in batch}.isdisjoint(
        {item[2] for batch in rank1 for item in batch}
    )


def test_augmentor_resamples_background_noise_waveform(monkeypatch):
    captured = {}

    def fake_open(f_path, normalized=False):
        return torch.ones(1, 48000), 48000

    def fake_resampling(wav, origin_sr, target_sr, backend):
        assert origin_sr == 48000
        assert target_sr == 16000
        return torch.full((1, 16000), 2.0), 16000

    def fake_add_bg_noise(wav, noise, snr_list):
        captured["noise_shape"] = noise[0].shape
        captured["noise_mean"] = noise[0].mean().item()
        return [wav], noise

    monkeypatch.setattr(aug_mod.AudioIO, "open", staticmethod(fake_open))
    monkeypatch.setattr(aug_mod, "wav_resampling", fake_resampling)
    monkeypatch.setattr(aug_mod, "add_bg_noise", fake_add_bg_noise)

    augmentor = AudioEffectAugmentor()
    augmentor.bg_noise = {"noise": {"wav_path": "dummy.wav"}}
    augmentor.add_bg_noise(torch.zeros(1, 16000), snr_list=[10], sr=16000)

    assert captured["noise_shape"] == torch.Size([1, 16000])
    assert captured["noise_mean"] == 2.0
