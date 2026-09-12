import sys
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
from puresound.audio.volume import (
    calculate_rms,
)
from puresound.utils import create_folder

sys.path.insert(0, "./")

TEST_CASE_DIR = Path(__file__).resolve().parents[1] / "test_case"
TEST_AUDIO_PATH = str(TEST_CASE_DIR / "1272-141231-0008.flac")
TEST_NOISE_FOLDER = str(TEST_CASE_DIR / "noise")
TEST_RIR_FOLDER = str(TEST_CASE_DIR / "rir")
TEST_NOISE_PATH = str(TEST_CASE_DIR / "noise" / "zzpQAtOmMhQ.wav")
TEST_RIR_PATH = str(TEST_CASE_DIR / "rir" / "Room042-00093.wav")
OUT_TEST_FOLDER = str(TEST_CASE_DIR / "outputs")
SAVE_TEST_AUDIO = True

create_folder(OUT_TEST_FOLDER)


def align_and_stack(wav1: torch.Tensor, wav2: torch.Tensor):
    if wav1.shape[-1] > wav2.shape[-1]:
        wav1 = wav1[..., : wav2.shape[-1]]
    else:
        wav2 = wav2[..., : wav1.shape[-1]]

    return torch.cat([wav1, wav2], dim=0)


@pytest.mark.audio_func
@pytest.mark.parametrize("norm_gain", [-22, -28, -40])
def test_audio_io_and_norm_func(norm_gain):
    wav, _ = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=norm_gain, verbose=True
    )
    assert torch.allclose(
        calculate_rms(wav=wav, to_log=True),
        torch.as_tensor(norm_gain, dtype=torch.float32),
    )


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
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=wav_gen),
            f_path=f"{OUT_TEST_FOLDER}/istft_gen.wav",
            sr=sr,
        )
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=wav_gen2),
            f_path=f"{OUT_TEST_FOLDER}/istft_gen2.wav",
            sr=sr,
        )


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


