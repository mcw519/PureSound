import sys

import pytest
import torch

from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise, add_bg_white_noise
from puresound.audio.spectrum import (cpx_stft_as_mag_and_phase, mag_and_phase_as_cpx_stft,
                                      stft_to_wav, wav_to_stft)
from puresound.audio.volume import calculate_rms, rand_gain_distortion, wav_fade_in, wav_fade_out

sys.path.insert(0, "./")


TEST_AUDIO_PATH = "./test_case/1272-141231-0008.flac"
TEST_NOISE_PATH = "./test_case/zzpQAtOmMhQ.wav"
OUT_TEST_FOLDER = "./test_case"


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
@pytest.mark.parametrize("start_time, duration", [(None, None), (0.3, 2), (1, None)])
def test_audio_gain_distortion_func(start_time, duration):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    distored_wav = rand_gain_distortion(
        wav=wav,
        sample_rate=sr,
        start_time=start_time,
        duration=duration,
        return_info=False,
    )
    assert distored_wav.shape == wav.shape
    AudioIO.save(
        wav=distored_wav,
        f_path=f"{OUT_TEST_FOLDER}/gain_distored_start_{start_time}_dur_{duration}.wav",
        sr=sr,
    )


@pytest.mark.audio_func
@pytest.mark.parametrize(
    "fade_len, fade_start, fade_type",
    [(3, 0.5, "linear"), (3, 0.5, "exponential"), (3, 0.5, "logarithmic")],
)
def test_audio_fade_func(fade_len, fade_start, fade_type):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    distored_wav = wav_fade_in(
        wav=wav,
        sr=sr,
        fade_len_s=fade_len,
        fade_begin_s=fade_start,
        fade_shape=fade_type,
    )
    assert distored_wav.shape == wav.shape
    AudioIO.save(
        wav=distored_wav,
        f_path=f"{OUT_TEST_FOLDER}/fadein_distored_{fade_type}_start_{fade_start}_dur_{fade_len}.wav",
        sr=sr,
    )

    distored_wav = wav_fade_out(
        wav=wav,
        sr=sr,
        fade_len_s=fade_len,
        fade_begin_s=fade_start,
        fade_shape=fade_type,
    )
    assert distored_wav.shape == wav.shape
    AudioIO.save(
        wav=distored_wav,
        f_path=f"{OUT_TEST_FOLDER}/fadeout_distored_{fade_type}_start_{fade_start}_dur_{fade_len}.wav",
        sr=sr,
    )


@pytest.mark.audio_func
@pytest.mark.parametrize("snr_list", [[-10], [0], [10]])
def test_audio_add_white_noise_func(snr_list):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    noisy_wav = add_bg_white_noise(wav=wav, snr_list=snr_list)
    for wav in noisy_wav:
        AudioIO.save(
            wav=wav,
            f_path=f"{OUT_TEST_FOLDER}/add_white_noise_{snr_list[0]}_dB.wav",
            sr=sr,
        )


@pytest.mark.audio_func
@pytest.mark.parametrize("snr_list", [[-10], [0], [10]])
def test_audio_add_bg_noise_func(snr_list):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    noise, sr = AudioIO.open(
        f_path=TEST_NOISE_PATH, normalized=False, target_lvl=None, verbose=True
    )
    noisy_wav = add_bg_noise(wav=wav, noise=[noise], snr_list=snr_list)
    for wav in noisy_wav:
        AudioIO.save(
            wav=wav,
            f_path=f"{OUT_TEST_FOLDER}/add_bg_noise_{snr_list[0]}_dB.wav",
            sr=sr,
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
        wav[..., : wav_gen.shape[-1]], wav_gen, atol=1e-7,
    ), torch.nn.functional.l1_loss(wav[..., : wav_gen.shape[-1]], wav_gen)
    assert torch.allclose(
        wav[..., : wav_gen2.shape[-1]], wav_gen2, atol=1e-7,
    ), torch.nn.functional.l1_loss(wav[..., : wav_gen2.shape[-1]], wav_gen2)
    AudioIO.save(wav=wav_gen, f_path=f"{OUT_TEST_FOLDER}/istft_gen.wav", sr=sr)
    AudioIO.save(wav=wav_gen2, f_path=f"{OUT_TEST_FOLDER}/istft_gen2.wav", sr=sr)
