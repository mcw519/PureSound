"""Audio I/O and level normalisation, the STFT round trip, and simulated room
reverb for a foreground/interferer scene."""

from pathlib import Path

import pytest
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.io import AudioIO
from puresound.audio.spectrum import (
    cpx_stft_as_mag_and_phase,
    mag_and_phase_as_cpx_stft,
    stft_to_wav,
    wav_to_stft,
)
from puresound.audio.volume import calculate_rms

TEST_AUDIO_PATH = str(Path(__file__).resolve().parents[1] / "test_case" / "1272-141231-0008.flac")


@pytest.mark.audio_func
@pytest.mark.parametrize("norm_gain", [-22, -28, -40])
def test_audio_io_normalises_the_level_and_saves_what_it_opened(norm_gain, tmp_path):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=norm_gain, verbose=True
    )
    assert torch.allclose(
        calculate_rms(wav=wav, to_log=True),
        torch.as_tensor(norm_gain, dtype=torch.float32),
    )
    path = str(tmp_path / "saved.wav")
    AudioIO.save(wav=wav, f_path=path, sr=sr, subtype="FLOAT")
    reopened, reopened_sr = AudioIO.open(f_path=path)
    assert reopened_sr == sr
    assert torch.allclose(reopened, wav, atol=1e-6)


@pytest.mark.audio_func
@pytest.mark.parametrize("target_lvl", [None, -20.0])
@pytest.mark.parametrize("verbose", [False, True])
def test_audio_io_normalized_scales_to_unit_average_amplitude(target_lvl, verbose, tmp_path):
    path = str(tmp_path / "quiet.wav")
    quiet = 0.05 * torch.sin(torch.linspace(0, 300, 8000)).view(1, -1)
    AudioIO.save(torch.cat([quiet, 2 * quiet]), path, 16000, subtype="FLOAT")

    wav, _ = AudioIO.open(path, normalized=True, target_lvl=target_lvl, verbose=verbose)

    assert torch.allclose(wav.abs().mean(dim=-1), torch.ones(2), atol=1e-4)


@pytest.mark.audio_func
@pytest.mark.parametrize(
    "length_s, expected_samples", [(0.5, 8000), (2.5, 40000), (1, 16000)]
)
def test_audio_cut_accepts_fractional_seconds(length_s, expected_samples):
    wav, (offset, end_offset) = AudioIO.audio_cut(torch.randn(1, 16000), 16000, length_s)

    assert wav.shape == (1, expected_samples)
    assert end_offset - offset == expected_samples


@pytest.mark.audio_func
@pytest.mark.parametrize(
    "nfft, win_size, hop_size, win_type",
    [[512, 512, 128, "hann_window"], [1024, 512, 160, "hamming_window"]],
)
def test_audio_to_spectrum_func(nfft, win_size, hop_size, win_type):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    cpx_stft, stft_info = wav_to_stft(
        wav=wav,
        nfft=nfft,
        win_size=win_size,
        hop_size=hop_size,
        window_type=win_type,
        stft_normalized=False,
    )
    mag, phase = cpx_stft_as_mag_and_phase(x=cpx_stft, eps=None)
    wav_gen = stft_to_wav(x=cpx_stft, **stft_info)

    cpx_stft2 = mag_and_phase_as_cpx_stft(mag=mag, phase=phase)
    wav_gen2 = stft_to_wav(x=cpx_stft2, **stft_info)

    assert torch.allclose(
        wav[..., : wav_gen.shape[-1]],
        wav_gen,
        atol=1e-7,
    ), torch.nn.functional.l1_loss(wav[..., : wav_gen.shape[-1]], wav_gen)
    assert torch.allclose(
        wav[..., : wav_gen2.shape[-1]],
        wav_gen2,
        atol=1e-7,
    ), torch.nn.functional.l1_loss(wav[..., : wav_gen2.shape[-1]], wav_gen2)


@pytest.mark.audio_func
def test_audio_simulated_reverb_func():
    augmentation = AudioEffectAugmentor()
    augmentation.init_room_simulator(
        {
            "room_dim_range": [[6.0, 6.0], [6.0, 6.0], [2.8, 2.8]],
            "rt60_range": [0.2, 0.4],
            "source_receiver_distance_range": [0.5, 2.5],
            "foreground_distance_range": [0.5, 1.0],
            "interferer_distance_range": [1.5, 2.5],
            "receiver_margin": 0.4,
            "source_margin": 0.4,
            "nsample": 2048,
            "order": 4,
        }
    )
    wav = torch.zeros(1, 4000)
    wav[:, 200] = 1.0
    scene = augmentation.sample_room_scene()
    reverb_wav, (rir_id, rir_info) = augmentation.apply_rir(
        wav=wav,
        rir_mode="full",
        sr=8000,
        rir_id=None,
        room_scene=scene,
        source_role="foreground",
    )
    early_wav, _ = augmentation.apply_rir(
        wav=wav, rir_mode="early", sr=8000, rir_id=rir_id
    )
    interferer_wav, (_, interferer_info) = augmentation.apply_rir(
        wav=wav,
        rir_mode="full",
        sr=8000,
        rir_id=None,
        room_scene=scene,
        source_role="interferer",
    )
    assert rir_id.startswith("simulated-")
    assert rir_info["metadata"]["rt60"] >= 0.2
    assert rir_info["metadata"]["room_dim"] == interferer_info["metadata"]["room_dim"]
    assert rir_info["metadata"]["receiver"] == interferer_info["metadata"]["receiver"]
    assert rir_info["metadata"]["source_receiver_distance"] <= 1.0
    assert interferer_info["metadata"]["source_receiver_distance"] >= 1.5
    assert reverb_wav.shape == wav.shape
    assert early_wav.shape == wav.shape
    assert interferer_wav.shape == wav.shape
    assert reverb_wav.abs().sum() > wav.abs().sum()


