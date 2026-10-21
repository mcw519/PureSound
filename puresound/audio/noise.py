import math
from typing import List, Optional

import torch

from .volume import calculate_rms, normalize_waveform

# Below this RMS (about -140 dBFS) a window is treated as digital silence.
SILENT_SEGMENT_RMS = 1e-7


LEVEL_REFERENCES = ("clip", "segment")


def add_bg_noise(
    wav: torch.Tensor,
    noise: List[torch.Tensor],
    snr_list: List[float],
    level_reference: str = "clip",
    min_segment_level_db: Optional[float] = None,
    segment_redraws: int = 10,
    overlay: Optional[List[torch.Tensor]] = None,
    overlay_db: float = 0.0,
) -> List:
    """
    Injected additive background noises with a SNR list.\n
    Numbers of augmented outputs must same as length of SNR list.

    Noisy = clean_wav + scale*noise,
    Here, we change the noise's scale to math the target SNR.

    Several tensors in ``noise`` play one after another (the "dynamic" type); a
    noise longer than ``wav`` is cut to a random window of ``wav``'s length.
    Where the noise level that the SNR refers to is measured:

    * ``"clip"`` -- over the whole noise before the window is cut. For a steady
      noise the window has the clip's level; for a bursty one (a bark, a door)
      the window can fall between the events and the mixture's real SNR is then
      far above the requested one, or catch them all and sit below it.
    * ``"segment"`` -- over the window actually mixed in, so the requested SNR is
      the mixture's. This is how the paired evaluation sets are mixed
      (``puresound.evaluation.tools.mix_paired_set`` crops before calling here).

    ``min_segment_level_db`` redraws a window whose level is more than that many
    dB below the whole noise's (at most ``segment_redraws`` draws, then the last
    one is kept). Under ``"segment"`` it keeps a quiet stretch between a bursty
    noise's events from being raised to the requested level as a steady hiss.

    ``overlay`` is a second noise heard at the same time as the first, not after
    it: its own window, ``overlay_db`` relative to the first, and the requested
    SNR then refers to the sum.

    All of these draw nothing unless set, so the default regenerates the same
    mixtures.

    Args:
        wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
        noise: bg noises with time dimension in the last tensor shape, i.e., [..., L]
        snr_list: list of SNR ratio in dB
        level_reference: "clip" or "segment", see above
        min_segment_level_db: lowest window level relative to the whole noise, in dB (<= 0); None = any window
        segment_redraws: window draws allowed by ``min_segment_level_db``
        overlay: noises laid over ``noise``, same layout; None = none
        overlay_db: the overlay's level relative to ``noise``'s, in dB

    Returns:
        List of waveform has been addedd noise background
    """
    if level_reference not in LEVEL_REFERENCES:
        raise ValueError(f"level_reference must be one of {LEVEL_REFERENCES}, got {level_reference!r}")
    if min_segment_level_db is not None and min_segment_level_db > 0:
        raise ValueError(f"min_segment_level_db must be <= 0, got {min_segment_level_db}")
    if segment_redraws < 1:
        raise ValueError(f"segment_redraws must be >= 1, got {segment_redraws}")
    window = dict(
        length=wav.shape[-1],
        level_reference=level_reference,
        min_segment_level_db=min_segment_level_db,
        segment_redraws=segment_redraws,
    )
    cat_noises = _noise_window(noise, **window)
    if overlay is not None:
        gain = 10 ** (overlay_db / 20)
        cat_noises = cat_noises + gain * _noise_window(overlay, **window)
        if level_reference == "segment" and calculate_rms(cat_noises).max() > SILENT_SEGMENT_RMS:
            cat_noises = normalize_waveform(wav=cat_noises, amp_type="rms")
        else:
            # Unit-level layers that do not correlate sum to this level.
            cat_noises = cat_noises / math.sqrt(1 + gain**2)

    wav_rms = calculate_rms(wav=wav)
    noisy_speech = []
    noise = []
    for snr_db in snr_list:
        snr = 10 ** (torch.Tensor([snr_db / 20]))
        bg_rms = (wav_rms / snr).unsqueeze(-1)
        noisy_speech.append(wav + bg_rms * cat_noises)
        noise.append(bg_rms * cat_noises)

    return noisy_speech, noise


def _noise_window(
    noise: List[torch.Tensor],
    length: int,
    level_reference: str,
    min_segment_level_db: Optional[float],
    segment_redraws: int,
) -> torch.Tensor:
    """``noise`` played in turn and cut or tiled to ``length``, at unit level.

    The level is unit over the whole noise or over the window, as
    ``level_reference`` says (see ``add_bg_noise``).
    """
    cat_noises = []
    for n in noise:
        if n.shape[0] != 1:
            n = n[0].view(1, -1)

        cat_noises.append(normalize_waveform(wav=n, amp_type="rms"))
    cat_noises = torch.cat(cat_noises, dim=-1)
    cat_noises = normalize_waveform(wav=cat_noises, amp_type="rms")

    noise_length = cat_noises.shape[-1]
    if length < noise_length:
        s = int(torch.randint(0, noise_length - length, (1,)))
        if min_segment_level_db is not None:
            whole = torch.mean(torch.square(cat_noises))
            floor = whole * 10 ** (min_segment_level_db / 10)
            for _ in range(segment_redraws - 1):
                if torch.mean(torch.square(cat_noises[:, s : s + length])) >= floor:
                    break
                s = int(torch.randint(0, noise_length - length, (1,)))
        cat_noises = cat_noises[:, s : s + length]
    else:
        cat_noises = cat_noises.repeat(1, round(length / noise_length) + 1)
        cat_noises = cat_noises[:, :length]

    if level_reference == "segment" and calculate_rms(cat_noises).max() > SILENT_SEGMENT_RMS:
        # A window of digital silence keeps the clip's scale: normalising it would
        # only raise rounding noise to the requested level.
        cat_noises = normalize_waveform(wav=cat_noises, amp_type="rms")
    return cat_noises


def add_bg_white_noise(wav: torch.Tensor, snr_list: List[float]):
    """
    Simple Gaussian noise injection

    Args:
        wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
        snr_list: list of SNR ratio in dB

    Returns:
        List of waveform has been addedd white noise in background
    """
    white_noise = []
    noisy_speech = []
    for snr_db in snr_list:
        RMS_s = calculate_rms(wav=wav)
        snr = 10 ** (torch.Tensor([snr_db / 20]))
        RMS_n = RMS_s / snr
        STD_n = float(RMS_n)
        noise = torch.FloatTensor(wav.shape[-1]).normal_(mean=0, std=STD_n).view(1, -1)
        noisy_speech.append(wav + noise)
        white_noise.append(noise)

    return noisy_speech, white_noise


# TODO: Color noise type