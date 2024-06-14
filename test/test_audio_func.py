import sys

import pytest
import torch

from puresound.audio.augmentaion import AudioEffectAugmentor
from puresound.audio.dsp import (ParametricEQ, get_biquad_params, wav_apply_biquad_filter,
                                 wav_resampling)
from puresound.audio.impluse_response import rand_add_2nd_filter_response, wav_apply_rir
from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise, add_bg_white_noise
from puresound.audio.spectrum import (cpx_stft_as_mag_and_phase, mag_and_phase_as_cpx_stft,
                                      stft_to_wav, wav_to_stft)
from puresound.audio.volume import calculate_rms, rand_gain_distortion, wav_fade_in, wav_fade_out

sys.path.insert(0, "./")


TEST_AUDIO_PATH = "./test_case/1272-141231-0008.flac"
TEST_NOISE_PATH = "./test_case/noise/zzpQAtOmMhQ.wav"
TEST_RIR_PATH = "./test_case/rir/Room042-00093.wav"
OUT_TEST_FOLDER = "./test_case"
SAVE_TEST_AUDIO = True


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
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=distored_wav),
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
        wav=align_and_stack(wav1=wav, wav2=distored_wav),
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
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=distored_wav),
            f_path=f"{OUT_TEST_FOLDER}/fadeout_distored_{fade_type}_start_{fade_start}_dur_{fade_len}.wav",
            sr=sr,
        )


@pytest.mark.audio_func
@pytest.mark.parametrize("snr_list", [[-10], [0], [10]])
def test_audio_add_white_noise_func(snr_list):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    noisy_wav, _ = add_bg_white_noise(wav=wav, snr_list=snr_list)
    if SAVE_TEST_AUDIO:
        for n_wav in noisy_wav:
            AudioIO.save(
                wav=align_and_stack(wav1=wav, wav2=n_wav),
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
    noisy_wav, _ = add_bg_noise(wav=wav, noise=[noise], snr_list=snr_list)
    if SAVE_TEST_AUDIO:
        for idx, n_wav in enumerate(noisy_wav):
            AudioIO.save(
                wav=align_and_stack(wav1=wav, wav2=n_wav),
                f_path=f"{OUT_TEST_FOLDER}/add_bg_noise_{snr_list[idx]}_dB.wav",
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
@pytest.mark.parametrize("rir_type", ["early", "direct", "full"])
def test_audio_reverb_func(rir_type):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    rir, sr = AudioIO.open(
        f_path=TEST_RIR_PATH, normalized=False, target_lvl=None, verbose=True
    )
    reverb_wav = wav_apply_rir(wav=wav, impaulse=rir, sample_rate=sr, rir_mode=rir_type)
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=reverb_wav),
            f_path=f"{OUT_TEST_FOLDER}/reverb_{rir_type}_wav_gen.wav",
            sr=sr,
        )


@pytest.mark.audio_func
def test_audio_add_2nd_rand_response_func():
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    out_wav, a, b = rand_add_2nd_filter_response(wav=wav, a=None, b=None)
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=out_wav),
            f_path=f"{OUT_TEST_FOLDER}/2nd_response_wav.wav",
            sr=sr,
        )


@pytest.mark.audio_func
@pytest.mark.parametrize("target_sr, backend", [[8000, "sox"], [24000, "torchaudio"]])
def test_aduio_resampling_func(target_sr, backend):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    resample_wav, *others = wav_resampling(
        wav=wav, origin_sr=sr, target_sr=target_sr, backend=backend
    )
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=resample_wav,
            f_path=f"{OUT_TEST_FOLDER}/resampling_to_{target_sr}_{backend}_wav.wav",
            sr=others[0],
        )


@pytest.mark.audio_func
@pytest.mark.parametrize("cutoff, filter_type", [[4000, "lpf"], [100, "hpf"]])
def test_biquad_lpf_hpf_func(cutoff, filter_type):
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    b, a = get_biquad_params(
        gain_dB=0,
        cutoff_freq=cutoff,
        q_factor=0.707,
        sample_rate=sr,
        filter_type=filter_type,
    )
    filterd_wav = wav_apply_biquad_filter(wav=wav, b_coeff=b, a_coeff=a)
    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=filterd_wav),
            f_path=f"{OUT_TEST_FOLDER}/{filter_type}_filtered_wav.wav",
            sr=sr,
        )


@pytest.mark.audio_func
def test_biquad_filter_and_eq_func():
    EQ_param = {
        "sample_rate": 16000,
        "eq_band_gain": (0.5, 5.5, -3.25, -2.5, -4, -4, -4.5),
        "eq_band_cutoff": (500, 1000, 1500, 2500, 3500, 5500, 6000),
        "eq_band_q_factor": (0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707),
        "low_shelf_gain_dB": 0.0,
        "low_shelf_cutoff_freq": 80.0,
        "low_shelf_q_factor": 0.707,
        "high_shelf_gain_dB": 0.0,
        "high_shelf_cutoff_freq": 7800,
        "high_shelf_q_factor": 0.707,
    }
    peq = ParametricEQ(**EQ_param)
    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )
    eq_wav = peq.forward(wav=wav)
    if SAVE_TEST_AUDIO:
        peq.plot_eq(
            savefig=f"{OUT_TEST_FOLDER}/paramteric_EQ.png",
        )
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=eq_wav),
            f_path=f"{OUT_TEST_FOLDER}/paramteric_EQ_wav.wav",
            sr=sr,
        )


@pytest.mark.audio_func
def test_audio_effect_augmentor():
    augmentation = AudioEffectAugmentor()
    augmentation.load_bg_noise_from_folder("./test_case/noise")
    augmentation.load_rir_from_folder("./test_case/rir")

    wav, sr = AudioIO.open(
        f_path=TEST_AUDIO_PATH, normalized=False, target_lvl=-28, verbose=True
    )

    # Gain
    distroyed_wav, _ = augmentation.apply_gain_distortion(wav=wav, sr=sr)

    # Sox effects
    speed_wav, _ = augmentation.sox_speed_perturbed(wav=wav, speed=1.2, sr=sr)
    pitch_wav, _ = augmentation.sox_pitch_perturbed(wav=wav, shift_ratio=-100, sr=sr)
    volume_wav, _ = augmentation.sox_volume_perturbed(wav=wav, vol_ratio=0.3, sr=sr)

    # Reverb and Inject noise
    noisy_wav, _ = augmentation.apply_rir(wav=wav, rir_mode="full", sr=sr, rir_id=None)
    noisy_wav, _ = augmentation.add_bg_noise(wav=noisy_wav, snr_list=[-10], sr=sr, dynamic_type=False, noise_id=None)

    # SRC and Filters
    filtered_wav, _ = augmentation.apply_src_effect(wav=wav, sr=sr, src_sr=8000, src_backend="sox")
    filtered_wav, _ = augmentation.apply_2nd_iir_response(wav=filtered_wav, a_coeffs=None, b_coeffs=None)

    if SAVE_TEST_AUDIO:
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=distroyed_wav),
            f_path=f"{OUT_TEST_FOLDER}/aug_distroyed_audio.wav",
            sr=sr,
        )
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=speed_wav),
            f_path=f"{OUT_TEST_FOLDER}/aug_speed_change_audio.wav",
            sr=sr,
        )
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=pitch_wav),
            f_path=f"{OUT_TEST_FOLDER}/aug_pitch_change_audio.wav",
            sr=sr,
        )
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=volume_wav),
            f_path=f"{OUT_TEST_FOLDER}/aug_volume_change_audio.wav",
            sr=sr,
        )
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=noisy_wav[0]),
            f_path=f"{OUT_TEST_FOLDER}/aug_reverb_noise_audio.wav",
            sr=sr,
        )
        AudioIO.save(
            wav=align_and_stack(wav1=wav, wav2=filtered_wav),
            f_path=f"{OUT_TEST_FOLDER}/aug_filtered_audio.wav",
            sr=sr,
        )