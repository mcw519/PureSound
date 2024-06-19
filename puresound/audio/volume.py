import random
from typing import Optional

import torch


def calculate_rms(wav: torch.Tensor, to_log: bool = False):
    """
    Calculates the root mean square.

    Based on https://github.com/iver56/audiomentations/blob/master/audiomentations/core/utils.py
    """
    rms = torch.sqrt(torch.mean(torch.square(wav), dim=-1, keepdim=False))
    if to_log:
        return 20 * torch.log10(rms)
    return rms


def normalize_waveform(wav: torch.Tensor, amp_type: str = "avg") -> torch.Tensor:
    """
    This function normalizes a signal to unitary average or peak amplitude

    Args:
        wav (Tensor): The waveform used for computing amplitude. Shape should be [..., L]
        amp_type: Whether to compute "avg" average or "peak" amplitude. Choose between ["rms", "avg", "peak"]

    Returns:
        Normalized level waveform, with same shape as input
    """
    amp_type = amp_type.lower()
    eps = 1e-14
    assert amp_type in ["rms", "avg", "peak"]

    if amp_type == "avg":
        den = torch.mean(torch.abs(wav), dim=-1, keepdim=True)
    elif amp_type == "peak":
        den = torch.max(torch.abs(wav), dim=-1, keepdim=True)[0]
    elif amp_type == "rms":
        den = torch.sqrt(torch.mean(torch.square(wav), dim=-1, keepdim=False))

    den = den + eps

    return wav / den


def rescale_waveform(
    wav: torch.Tensor, target_lvl: float, amp_type: str = "avg", scale: str = "linear"
) -> torch.Tensor:
    """
    This functions performs signal rescaling to a target level

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        target_lvl: Target lvl in dB or linear scale
        amp_type: Whether to compute "avg" average or "peak" amplitude. Choose between ["rms", "avg", "peak"]
        scale: whether target_lvl belongs to linear or dB scale. Choose between ["linear", "dB"]

    Returns:
        Rescaled waveform, with same shape as input
    """
    amp_type = amp_type.lower()
    scale = scale.lower()

    assert scale in ["linear", "db"]

    wav = normalize_waveform(wav=wav, amp_type=amp_type)

    if scale == "linear":
        out = target_lvl * wav
    elif scale == "db":
        target_lvl = 10 ** (target_lvl / 20)
        out = target_lvl * wav

    return out


def rand_gain_distortion(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
    return_info: bool = False,
):
    """
    This function simulated the audio gain distortion.

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        sample_rate: audio sample rate
        start_time: distortion start time
        duration: distortion length
        return_info: if true, return details

    Returns:
        distorted waveform, with same shape as input
    """
    if start_time is None:
        distortion_start_time = random.randint(0, wav.shape[-1])
    else:
        # add some dither
        distortion_start_time = int(
            (start_time + random.random() * start_time) * sample_rate
        )

    if duration is None:
        distortion_duration = random.randint(0, wav.shape[-1] - distortion_start_time)
    else:
        # add some dither
        distortion_duration = int((duration + random.random() * duration) * sample_rate)

    distortion_gain = 4 ** random.gauss(0, 1)

    distortion_mask = torch.ones_like(wav)
    distortion_mask[
        ..., distortion_start_time : distortion_start_time + distortion_duration
    ] = distortion_gain
    distorted_wav = torch.clip(wav * distortion_mask, min=-1, max=1)

    if return_info:
        return distorted_wav, (
            distortion_start_time,
            distortion_duration,
            distortion_gain,
        )
    else:
        return distorted_wav


def wav_fade_in(
    wav: torch.Tensor,
    sr: int,
    fade_len_s: int,
    fade_begin_s: int = 0,
    fade_shape: str = "linear",
) -> torch.Tensor:
    """
    This function simulated the fade in effect.

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        sample_rate: audio sample rate
        fade_len_s: distortion length in second
        fade_begin_s: distortion start time in second
        fade_shape: the fade changin types, choose in ["linear", "exponential", "logarithmic"]

    Returns:
        distorted waveform, with same shape as input
    """
    wav_len = wav.shape[-1]
    fade_in_len = int(sr * fade_len_s)
    fade_begin = int(sr * fade_begin_s)
    fade = torch.linspace(0, 1, fade_in_len)
    ones_before = torch.ones(fade_begin) if fade_begin != 0 else None
    ones_after = (
        torch.ones(wav_len - fade_in_len - fade_begin)
        if wav_len - fade_in_len - fade_begin > 0
        else None
    )

    if fade_shape == "linear":
        fade = fade

    if fade_shape == "exponential":
        fade = torch.pow(2, (fade - 1)) * fade

    if fade_shape == "logarithmic":
        fade = torch.log10(0.1 + fade) + 1

    output = []
    for x in [ones_before, fade, ones_after]:
        if x is not None:
            output.append(x)

    fade = torch.cat(output).clamp_(0, 1)
    assert fade.shape[-1] == wav_len
    return fade * wav


def wav_fade_out(
    wav: torch.Tensor,
    sr: int,
    fade_len_s: int,
    fade_begin_s: int = 0,
    fade_shape: str = "linear",
) -> torch.Tensor:
    """
    This function simulated the fade out effect.

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        sample_rate: audio sample rate
        fade_len_s: distortion length in second
        fade_begin_s: distortion start time in second
        fade_shape: the fade changin types, choose in ["linear", "exponential", "logarithmic"]

    Returns:
        distorted waveform, with same shape as input
    """
    wav_len = wav.shape[-1]
    fade_out_len = int(sr * fade_len_s)
    fade_begin = int(sr * fade_begin_s)

    fade = torch.linspace(0, 1, fade_out_len)
    ones_before = torch.ones(fade_begin) if fade_begin != 0 else None
    ones_after = (
        torch.ones(wav_len - fade_out_len - fade_begin)
        if wav_len - fade_out_len - fade_begin > 0
        else None
    )

    if fade_shape == "linear":
        fade = -fade + 1

    if fade_shape == "exponential":
        fade = torch.pow(2, -fade) * (1 - fade)

    if fade_shape == "logarithmic":
        fade = torch.log10(1.1 - fade) + 1

    output = []
    for x in [ones_before, fade, ones_after]:
        if x is not None:
            output.append(x)

    fade = torch.cat(output).clamp_(0, 1)
    assert fade.shape[-1] == wav_len
    return fade * wav


def wav_clipping(
    wav: torch.Tensor, min_quantile: float = 0.0, max_quantile: float = 0.9
):
    """
    Apply the clipping distortion to the input signal.
    Referes: https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/simulate_data_from_param.py#L89

    Args:
        speech_sample: a single speech sample (..., L)
        min_quantile: lower bound on the quantile of samples to be clipped
        max_quantile: upper bound on the quantile of samples to be clipped

    Returns:
        clipped speech sample (..., L)
    """
    q = torch.Tensor([min_quantile, max_quantile])
    min_, max_ = torch.quantile(wav, q, dim=-1)
    wav = torch.clip(input=wav, min=min_, max=max_)
    return wav
