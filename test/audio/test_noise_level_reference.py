"""``add_bg_noise`` level reference: where the level the SNR refers to is measured.

A noise longer than the speech is cut to a window. ``clip`` (the default) scales
by the whole noise; ``segment`` by the window mixed in. ``min_segment_level_db``
redraws windows far below the whole noise's level. Off, neither may draw from the
RNG, so a recipe that does not set them regenerates the same mixtures.
"""
import pytest
import torch
from pydantic import ValidationError

from puresound.audio.noise import add_bg_noise
from puresound.config.augmentation import NoiseAugmentation
from puresound.task.noise_stage import NoiseStage

N = 1600  # speech length


def _speech():
    torch.manual_seed(0)
    return torch.randn(1, N) * 0.1


def _bursty(length=10 * N, burst_at=8 * N):
    """Silence except one loud burst one window long near the end."""
    noise = torch.zeros(1, length)
    noise[0, burst_at : burst_at + N] = torch.randn(N)
    return noise


def _rms(x):
    return float(torch.sqrt(torch.mean(x**2)))


def _snr_db(speech, added):
    return 20 * torch.log10(torch.tensor(_rms(speech) / _rms(added))).item()


def test_the_default_draws_one_window_and_scales_by_the_whole_noise():
    speech, noise = _speech(), _bursty()
    torch.manual_seed(7)
    (noisy,), (added,) = add_bg_noise(speech, [noise], [5.0])
    after_default = torch.get_rng_state()
    torch.manual_seed(7)
    s = int(torch.randint(0, noise.shape[-1] - N, (1,)))
    assert torch.equal(torch.get_rng_state(), after_default)
    whole = noise / torch.sqrt(torch.mean(noise**2))
    expected = whole[:, s : s + N] * (_rms(speech) / 10 ** (5.0 / 20))
    assert torch.allclose(added, expected, atol=1e-6)
    assert torch.allclose(noisy, speech + added)


def test_segment_makes_the_requested_snr_the_mixtures():
    speech = _speech()
    noise = torch.randn(1, 10 * N) * torch.linspace(0.01, 1.0, 10 * N)  # level varies along the clip
    for seed in range(5):
        torch.manual_seed(seed)
        _, (added,) = add_bg_noise(speech, [noise], [3.0], level_reference="segment")
        assert _snr_db(speech, added) == pytest.approx(3.0, abs=1e-3)


def test_under_clip_a_window_between_bursts_is_far_quieter_than_requested():
    speech, noise = _speech(), _bursty(burst_at=9 * N - 1)
    torch.manual_seed(0)
    _, (added,) = add_bg_noise(speech, [noise], [0.0])
    assert _rms(added) < 1e-6  # the window missed the burst: the row is clean


def _offsets(monkeypatch, offsets):
    """Make the window draws return ``offsets`` in turn; record how many were taken."""
    taken = []

    def randint(low, high, size):
        taken.append(offsets[len(taken)])
        return torch.tensor([taken[-1]])

    monkeypatch.setattr(torch, "randint", randint)
    return taken


def _window(noise, s):
    return noise[:, s : s + N]


def test_a_window_below_the_floor_is_redrawn(monkeypatch):
    speech = _speech()
    noise = torch.full((1, 10 * N), 1e-3)  # a faint floor ...
    noise[0, 8 * N : 9 * N] = torch.randn(N)  # ... and one burst
    taken = _offsets(monkeypatch, [N, 2 * N, 8 * N, 0])
    _, (added,) = add_bg_noise(
        speech, [noise], [0.0], level_reference="segment", min_segment_level_db=-10.0
    )
    assert taken == [N, 2 * N, 8 * N]  # two quiet windows refused, the burst kept
    expected = _window(noise, 8 * N) / _rms(_window(noise, 8 * N)) * _rms(speech)
    assert torch.allclose(added, expected, atol=1e-6)


def test_without_the_floor_a_quiet_window_is_raised_to_the_requested_level(monkeypatch):
    speech = _speech()
    noise = torch.full((1, 10 * N), 1e-3)
    noise[0, 8 * N : 9 * N] = torch.randn(N)
    _offsets(monkeypatch, [N])
    _, (added,) = add_bg_noise(speech, [noise], [0.0], level_reference="segment")
    assert _snr_db(speech, added) == pytest.approx(0.0, abs=1e-3)  # the floor, now as loud as the speech


def test_redraws_are_capped_and_the_last_window_is_kept(monkeypatch):
    speech, noise = _speech(), _bursty()
    taken = _offsets(monkeypatch, [0, N, 2 * N, 3 * N, 8 * N])
    _, (added,) = add_bg_noise(speech, [noise], [0.0], min_segment_level_db=-10.0, segment_redraws=4)
    assert taken == [0, N, 2 * N, 3 * N]
    assert _rms(added) < 1e-6  # the fourth window, silent, kept


def test_a_silent_window_keeps_the_clip_scale_instead_of_raising_rounding_noise():
    speech, noise = _speech(), _bursty(burst_at=9 * N - 1)
    torch.manual_seed(0)
    _, (added,) = add_bg_noise(speech, [noise], [0.0], level_reference="segment")
    assert torch.isfinite(added).all()
    assert _rms(added) < 1e-6


def test_a_noise_shorter_than_the_speech_is_tiled_under_either_reference():
    speech = _speech()
    noise = torch.randn(1, N // 3)
    for reference in ("clip", "segment"):
        _, (added,) = add_bg_noise(speech, [noise], [10.0], level_reference=reference)
        assert added.shape == speech.shape
        assert _snr_db(speech, added) == pytest.approx(10.0, abs=0.5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"level_reference": "file"}, "level_reference"),
        ({"min_segment_level_db": 3.0}, "min_segment_level_db"),
        ({"segment_redraws": 0}, "segment_redraws"),
    ],
)
def test_bad_arguments_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        add_bg_noise(_speech(), [_bursty()], [0.0], **kwargs)


def _config(**extra):
    return NoiseAugmentation(
        used=True, prob=1.0, noise_folder="/x", snr_range=[0, 10],
        prob_white_noise=0.0, white_noise_snr_range=[10, 30], **extra,
    )


def test_the_recipe_block_defaults_to_the_old_behaviour_and_validates():
    config = _config()
    assert (config.level_reference, config.min_segment_level_db, config.segment_redraws) == ("clip", None, 10)
    with pytest.raises(ValidationError):
        _config(level_reference="file")
    with pytest.raises(ValidationError):
        _config(min_segment_level_db=1.0)
    with pytest.raises(ValidationError):
        _config(segment_redraws=0)


def test_the_noise_stage_hands_the_block_to_the_augmentor():
    seen = {}

    class Augmentor:
        def add_bg_noise(self, wav, snr_list, **kwargs):
            seen.update(kwargs)
            return [wav], None

    config = _config(level_reference="segment", min_segment_level_db=-12.0, segment_redraws=6)
    NoiseStage(Augmentor(), config).apply(torch.zeros(1, N), sample_rate=16000)
    assert (seen["level_reference"], seen["min_segment_level_db"], seen["segment_redraws"]) == ("segment", -12.0, 6)
