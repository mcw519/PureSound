"""`AudioEffectAugmentor`: what reaches the noise mixing step, and how its noise and
RIR folders are listed."""

import pytest
import torch

from puresound.audio import augmentation as aug_mod
from puresound.audio.augmentation import AudioEffectAugmentor


def test_background_noise_is_resampled_to_the_speech_rate(monkeypatch):
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


def test_the_noise_transform_is_applied_before_mixing(tmp_path, write_tone_wav):
    """Room colouring hooks in here; a transform applied after the mix would
    colour the speech as well."""
    noise_dir = tmp_path / "noise"
    write_tone_wav(noise_dir / "n0.wav", duration=0.5, freq=500.0)

    aug = AudioEffectAugmentor()
    aug.load_bg_noise_from_folder(str(noise_dir))

    torch.manual_seed(0)
    wav = 0.05 * torch.randn(1, 8000)
    seen = {}

    def transform(n):
        seen["called"] = True
        return torch.zeros_like(n)  # silence the noise entirely

    _, (added_dry, _, _) = aug.add_bg_noise(wav=wav.clone(), snr_list=[0.0], sr=16000)
    _, (added_silenced, _, _) = aug.add_bg_noise(
        wav=wav.clone(), snr_list=[0.0], sr=16000, noise_transform=transform
    )
    assert seen.get("called")
    assert float(added_dry[0].abs().sum()) > 0
    assert float(added_silenced[0].abs().sum()) == 0.0


def test_a_float32_speed_resamples_at_the_rate_it_names(monkeypatch):
    """The task draws speeds from a float32 grid, so 0.95 arrives as 0.94999999.
    Truncating ``16000 * speed`` gives 15199 Hz, which shares no factor with
    16000: torchaudio then builds a 16000-phase sinc kernel, about 3 s and 4 GB
    for one 6 s clip, which is what grew dataloader workers past the host's
    memory. Rounded, the rate is 15200 Hz and the kernel is 20 phases."""
    seen = []
    real_resample = aug_mod.torchaudio.functional.resample

    def spy(wav, orig_freq, new_freq, **kwargs):
        seen.append((orig_freq, new_freq))
        return real_resample(wav, orig_freq=orig_freq, new_freq=new_freq, **kwargs)

    monkeypatch.setattr(aug_mod.torchaudio.functional, "resample", spy)
    augmentor = AudioEffectAugmentor()
    for speed in torch.arange(0.9, 1.1 + 0.025, 0.05):
        augmentor.sox_speed_perturbed(torch.zeros(1, 1600), speed=speed.item(), sr=16000)
    assert seen == [(14400, 16000), (15200, 16000), (16000, 16000), (16800, 16000), (17600, 16000)]


def test_folders_with_spaces_in_their_names_are_listed_by_their_real_paths(tmp_path, write_tone_wav):
    """A noise or RIR corpus under a directory or file name with a space must
    resolve to files that exist, keyed by the file's own stem."""
    write_tone_wav(tmp_path / "plain.wav")
    write_tone_wav(tmp_path / "my noise dir" / "babble 01.wav")
    write_tone_wav(tmp_path / "my noise dir" / "inner" / "plain2.wav")

    augmentor = AudioEffectAugmentor()
    augmentor.load_bg_noise_from_folder(str(tmp_path))

    assert {key: value["wav_path"] for key, value in augmentor.bg_noise.items()} == {
        "plain": str(tmp_path / "plain.wav"),
        "babble 01": str(tmp_path / "my noise dir" / "babble 01.wav"),
        "plain2": str(tmp_path / "my noise dir" / "inner" / "plain2.wav"),
    }


def test_apply_rir_without_a_mode_convolves_the_whole_impulse_response(tmp_path, write_tone_wav):
    write_tone_wav(tmp_path / "rir.wav")
    augmentor = AudioEffectAugmentor()
    augmentor.load_rir_from_folder(str(tmp_path))
    wav = torch.randn(1, 4000)

    default = augmentor.apply_rir(wav, rir_id="rir")
    explicit = augmentor.apply_rir(wav, rir_mode="full", rir_id="rir")

    assert default.wav.shape == wav.shape
    assert torch.equal(default.wav, explicit.wav)
    assert default.detail.info["mode"] == "full"


@pytest.mark.parametrize("as_list", [False, True])
def test_a_returned_noise_id_can_be_replayed(tmp_path, write_tone_wav, as_list):
    for name in ("one", "two", "three"):
        write_tone_wav(tmp_path / f"{name}.wav")
    augmentor = AudioEffectAugmentor()
    augmentor.load_bg_noise_from_folder(str(tmp_path))
    wav = torch.zeros(1, 4000) + 0.1

    first, (_, noise_id, _) = augmentor.add_bg_noise(wav, snr_list=[5.0], sr=16000)
    replay, (_, replayed_id, _) = augmentor.add_bg_noise(
        wav, snr_list=[5.0], sr=16000, noise_id=[noise_id] if as_list else noise_id
    )

    assert replayed_id in (noise_id, [noise_id])
    assert replay[0].shape == first[0].shape
