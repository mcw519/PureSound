"""Two noises at once: ``add_bg_noise(overlay=...)`` and ``augmentation_noise.prob_overlay``.

The "dynamic" type plays two noises one after the other; an overlay is heard at
the same time as the first noise, at ``overlay_db`` relative to it, and the
requested SNR refers to the sum. Off, it draws nothing.
"""
import math

import pytest
import torch
from pydantic import ValidationError

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.noise import add_bg_noise
from puresound.config.augmentation import NoiseAugmentation
from puresound.task.noise_stage import NoiseStage

SR = 16000
N = 1600


def _tone(freq, length=N // 4):
    t = torch.arange(length) / SR
    return torch.sin(2 * math.pi * freq * t).view(1, -1)


def _speech():
    torch.manual_seed(0)
    return torch.randn(1, N) * 0.1


def _rms(x):
    return float(torch.sqrt(torch.mean(x**2)))


def _level_of(added, freq):
    """RMS of the component of ``added`` at ``freq`` (the tones are orthogonal over N)."""
    t = torch.arange(N) / SR
    c, s = torch.cos(2 * math.pi * freq * t), torch.sin(2 * math.pi * freq * t)
    amp = math.hypot(float(added[0] @ c), float(added[0] @ s)) * 2 / N
    return amp / math.sqrt(2)


@pytest.mark.parametrize("reference", ["clip", "segment"])
def test_the_overlay_sits_at_its_level_and_the_snr_refers_to_the_sum(reference):
    speech = _speech()
    _, (added,) = add_bg_noise(
        speech, [_tone(400.0)], [5.0], level_reference=reference, overlay=[_tone(1000.0)], overlay_db=-6.0
    )
    ratio_db = 20 * math.log10(_level_of(added, 1000.0) / _level_of(added, 400.0))
    assert ratio_db == pytest.approx(-6.0, abs=0.05)
    assert 20 * math.log10(_rms(speech) / _rms(added)) == pytest.approx(5.0, abs=0.05)


def test_without_an_overlay_the_mixture_and_the_draws_are_unchanged():
    speech, noise = _speech(), [torch.randn(1, 10 * N)]
    torch.manual_seed(5)
    _, (plain,) = add_bg_noise(speech, noise, [0.0])
    state = torch.get_rng_state()
    torch.manual_seed(5)
    _, (explicit,) = add_bg_noise(speech, noise, [0.0], overlay=None, overlay_db=-3.0)
    assert torch.equal(plain, explicit)
    assert torch.equal(torch.get_rng_state(), state)


def test_the_overlay_cuts_its_own_window_after_the_first():
    speech = _speech()
    bed, top = torch.randn(1, 10 * N), torch.randn(1, 10 * N)
    torch.manual_seed(2)
    _, (added,) = add_bg_noise(speech, [bed], [0.0], level_reference="segment", overlay=[top], overlay_db=-10.0)
    torch.manual_seed(2)
    s_bed = int(torch.randint(0, 9 * N, (1,)))
    s_top = int(torch.randint(0, 9 * N, (1,)))
    a = bed[:, s_bed : s_bed + N] / _rms(bed[:, s_bed : s_bed + N])
    b = top[:, s_top : s_top + N] / _rms(top[:, s_top : s_top + N])
    mix = a + 10 ** (-10 / 20) * b
    assert torch.allclose(added, mix / _rms(mix) * _rms(speech), atol=1e-5)


def _config(**extra):
    return NoiseAugmentation(
        used=True, prob=1.0, noise_folder="/x", snr_range=[0, 10],
        prob_white_noise=0.0, white_noise_snr_range=[10, 30], **extra,
    )


def test_the_recipe_block_needs_a_level_range_with_an_overlay():
    assert _config().prob_overlay is None
    with pytest.raises(ValidationError, match="overlay_level_db"):
        _config(prob_overlay=0.3)
    with pytest.raises(ValidationError):
        _config(prob_overlay=0.3, overlay_level_db=[0.0, -10.0])
    assert _config(prob_overlay=0.3, overlay_level_db=[-15, 0]).overlay_level_db == (-15.0, 0.0)


class _Recorder:
    def __init__(self):
        self.calls = []

    def add_bg_noise(self, wav, snr_list, **kwargs):
        self.calls.append(kwargs)
        return [wav], None


def test_an_unset_overlay_draws_nothing_in_the_noise_stage():
    states = []
    for config in (_config(), _config(prob_overlay=0.0, overlay_level_db=[-15, 0])):
        torch.manual_seed(11)
        recorder = _Recorder()
        NoiseStage(recorder, config).apply(torch.zeros(1, N), sample_rate=SR)
        assert recorder.calls[0]["overlay_db"] is None
        states.append(torch.get_rng_state())
    assert torch.equal(states[0], states[1])


def test_overlay_rows_get_a_level_in_range_and_dynamic_rows_none():
    recorder = _Recorder()
    stage = NoiseStage(recorder, _config(prob_overlay=1.0, overlay_level_db=[-15, -5]))
    torch.manual_seed(0)
    for _ in range(200):
        stage.apply(torch.zeros(1, N), sample_rate=SR)
    dynamic = [c for c in recorder.calls if c["dynamic_type"]]
    single = [c for c in recorder.calls if not c["dynamic_type"]]
    assert dynamic and single
    assert all(c["overlay_db"] is None for c in dynamic)
    assert all(-15 <= c["overlay_db"] <= -5 for c in single)


def test_the_augmentor_loads_a_second_file_for_the_overlay(tmp_path, write_tone_wav):
    noise_dir = tmp_path / "noise"
    write_tone_wav(noise_dir / "a.wav", duration=0.5, freq=400.0)
    write_tone_wav(noise_dir / "b.wav", duration=0.5, freq=1000.0)
    aug = AudioEffectAugmentor()
    aug.load_bg_noise_sources([("src", str(noise_dir), 1.0)])
    draws = iter(sorted(aug.bg_noise))  # first noise a, overlay b
    aug._draw_noise_id = lambda: next(draws)
    speech = _speech()
    _, (added, noise_id, _) = aug.add_bg_noise(speech, snr_list=[0.0], sr=SR, overlay_db=-6.0)
    assert noise_id == sorted(aug.bg_noise)[0]
    assert 20 * math.log10(_level_of(added[0], 1000.0) / _level_of(added[0], 400.0)) == pytest.approx(-6.0, abs=0.1)
    assert 20 * math.log10(_rms(speech) / _rms(added[0])) == pytest.approx(0.0, abs=0.1)
