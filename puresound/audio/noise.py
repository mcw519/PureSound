from typing import List

import torch

from .volume import calculate_rms, normalize_waveform


def add_bg_noise(wav: torch.Tensor, noise: List[torch.Tensor], snr_list: List[float]) -> List:
    """
    Injected additive background noises with a SNR list.\n
    Numbers of augmented outputs must same as length of SNR list.

    Noisy = clean_wav + scale*noise,
    Here, we change the noise's scale to math the target SNR.

    Args:
        wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
        noise: bg noises with time dimension in the last tensor shape, i.e., [..., L]
        snr_list: list of SNR ratio in dB

    Returns:
        List of waveform has been addedd noise background
    """
    cat_noises = []
    for n in noise:
        if n.shape[0] != 1:
            n = n[0].view(1, -1)

        cat_noises.append(normalize_waveform(wav=n, amp_type="rms"))
    cat_noises = torch.cat(cat_noises, dim=-1)
    cat_noises = normalize_waveform(wav=cat_noises, amp_type="rms")

    # check shape
    wav_length = wav.shape[-1]
    noise_length = cat_noises.shape[-1]

    if wav_length <= noise_length:
        s = int(torch.randint(0, noise_length - wav_length, (1,)))
        cat_noises = cat_noises[:, s : s + wav_length]
    else:
        cat_noises = cat_noises.repeat(1, round(wav_length / noise_length) + 1)
        cat_noises = cat_noises[:, :wav_length]

    wav_rms = calculate_rms(wav=wav)
    noisy_speech = []
    for snr_db in snr_list:
        snr = 10 ** (torch.Tensor([snr_db / 20]))
        bg_rms = (wav_rms / snr).unsqueeze(-1)
        noisy_speech.append((wav + bg_rms * cat_noises))

    return noisy_speech


def add_bg_white_noise(wav: torch.Tensor, snr_list: List[float]):
    """
    Simple Gaussian noise injection

    Args:
        wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
        snr_list: list of SNR ratio in dB

    Returns:
        List of waveform has been addedd white noise in background
    """
    noisy_speech = []
    for snr_db in snr_list:
        RMS_s = calculate_rms(wav=wav)
        snr = 10 ** (torch.Tensor([snr_db / 20]))
        RMS_n = RMS_s / snr
        STD_n = float(RMS_n)
        noise = torch.FloatTensor(wav.shape[-1]).normal_(mean=0, std=STD_n).view(1, -1)
        noisy_speech.append(wav + noise)

    return noisy_speech


# TODO: Color noise type