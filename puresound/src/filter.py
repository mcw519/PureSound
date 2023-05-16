import numpy as np
from typing import Optional


class Filter:
    def __init__(self) -> None:
        pass

    @staticmethod
    def lowpass_filter(cutoff: float, win_width: Optional[int] = None) -> np.array:
        """
        Args:
            cutoff: cutoff frequencies, in [0, 1] expressed as f/f_s where f_s is the samplerate.
            win_width: width of the filters (i.e. kernel_size=2 * width + 1).
                Default to 2/cutoffs. Longer filters will have better attenuation but more side effects.
        
        Returns:
            lowpass filter coefficient
        """
        if win_width is None:
            win_width = int(2 / cutoff)
        window = np.blackman(2 * win_width + 1)
        t = np.arange(-win_width, win_width + 1, dtype=np.float32)
        sinc = np.sinc(2 * cutoff * t)
        filter = 2 * cutoff * sinc * window

        return filter

    @staticmethod
    def get_bandpass_filter(
        cutoff_low: float, cutoff_high: float, win_width: Optional[int] = None
    ) -> np.array:
        """
        Args:
            cutoff_low: cutoff lowest frequencies, in [0, 1] expressed as f/f_s where f_s is the samplerate.
            cutoff_high: cutoff highest frequencies, in [0, 1] expressed as f/f_s where f_s is the samplerate.
            win_width: width of the filters (i.e. kernel_size=2 * width + 1).
                Default to 2/cutoffs. Longer filters will have better attenuation but more side effects.

        Returns:
            bandpass filter coefficient
        """
        if win_width is None:
            win_width = int(2 / (min(cutoff_low, cutoff_high)))
        low_filter = Filter.get_lowpass_filter(cutoff_low, win_width)
        high_filter = Filter.get_lowpass_filter(cutoff_high, win_width)
        return high_filter - low_filter

    @staticmethod
    def get_notch_filter(
        cutoff: float, notch_width: float = 0.05, win_width: Optional[int] = None
    ) -> np.array:
        """
        Args:
            cutoff: cutoff lowest frequencies, in [0, 1] expressed as f/f_s where f_s is the samplerate.
            notch_width: notch filter range, in [0, 1] expressed as f/f_s where f_s is the samplerate.
            win_width: width of the filters (i.e. kernel_size=2 * width + 1).
                Default to 2/cutoffs. Longer filters will have better attenuation but more side effects.
        
        Returns:
            notch filter coefficient
        """
        if win_width is None:
            win_width = int(2 / cutoff)
        pad = win_width // 2
        inputs = np.arange(win_width) - pad

        # Avoid frequencies that are too low
        cutoff += notch_width

        # Compute a low-pass filter with cutoff frequency notch_freq.
        hlpf = np.sinc(2 * (cutoff - notch_width) * inputs)
        hlpf *= np.blackman(win_width)
        hlpf /= np.sum(hlpf)

        # Compute a high-pass filter with cutoff frequency notch_freq.
        hhpf = np.sinc(2 * (cutoff + notch_width) * inputs)
        hhpf *= np.blackman(win_width)
        hhpf /= -np.sum(hhpf)
        hhpf[pad] += 1

        # Adding filters creates notch filter
        return (hlpf + hhpf).reshape(-1)


def wav_drop_frequency(
    sig: np.array, sr: int, cutoff_hz: float, drop_width_hz: float, win_width: int = 512
):
    """
    Apples frequency drop by a notch filter and time domain convolution.
    
    Args:
        sig: 1D input signal (np.array) has shape [L]
        sr: sampling rate
        cutoff_hz: cutoff frequency (Hz)
        drop_width_hz: width of drop frequencies (Hz)
        win_width: width of the filters (i.e. kernel_size=2 * width + 1).
            Default to 2/cutoffs. Longer filters will have better attenuation but more side effects.
    
    Returns:
        waveform
    """
    if sig.ndim == 2:
        sig = sig.squeeze()
    assert sig.ndim == 1

    nyquist_fs = sr / 2

    # avoid over nyquist range
    assert cutoff_hz < nyquist_fs
    if cutoff_hz + drop_width_hz > nyquist_fs:
        drop_width_hz = nyquist_fs - cutoff_hz

    # params
    cutoff = cutoff_hz / sr
    notch_width = drop_width_hz / sr

    notch_filter = Filter.get_notch_filter(cutoff, notch_width, win_width)
    out = np.convolve(sig, notch_filter, mode="same")

    return out.unsqueeze(0)


def wav_drop_chunk(sig: np.array, drop_start: float, drop_width: float):
    """
    Applies frame drop

    Args:
        sig: 1D input signal (np.array) has shape [1, L]
        drop_start: in [0, 1] expressed as drop_start_idx/wav_length
        drop_width: in [0, 1] expressed as drop_length/wav_length
    
    Returns:
        waveform
    """
    assert drop_start < 1.0
    assert 0 < drop_width < 1.0

    if sig.ndim == 2:
        sig = sig.squeeze()
    assert sig.ndim == 1

    if drop_start + drop_width > 1:
        drop_width = 1 - drop_start

    # params
    wav_len = sig.size
    start_idx = int(drop_start * wav_len)
    drop_len = int(drop_width * wav_len)
    mask = np.ones_like(sig)
    mask[start_idx : start_idx + drop_len] = 0.0
    out = sig * mask

    return out.unsqueeze(0)
