import random
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torchaudio


def wav_resampling(
    wav: torch.Tensor,
    origin_sr: int,
    target_sr: int,
    backend: str = "sox",
    torch_backend_params: Optional[Dict] = None,
):
    """
    Audio sample rate resaple.

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        origin_sr: original wav's sample rate
        tartget_sr: target sample rate
        backend: choose in ["sox", "torchaudio"]
        torch_backend_params: specific the torchaudio setting

    Returns:
        resampling wav and its target sample rate
    """
    backend = backend.lower()
    assert backend in ["sox", "torchaudio"]

    if backend == "torch":
        """downsample and upsample back by TorchAudio"""
        lp_width = None
        rolloff = None
        window = None

        if torch_backend_params is not None:
            lp_width = torch_backend_params["lp_width"]
            rolloff = torch_backend_params["rolloff"]
            window = torch_backend_params["window"]

        if lp_width is None:
            lp_width = random.choice((6, 16, 32, 64, 128))

        if rolloff is None:
            rolloff = random.uniform(0.8, 0.99)

        if window is None:
            window = random.choice(("sinc_interp_hann", "sinc_interp_kaiser"))

        wav = torchaudio.transforms.Resample(
            orig_freq=origin_sr,
            new_freq=target_sr,
            lowpass_filter_width=lp_width,
            resampling_method=window,
            rolloff=rolloff,
        )(wav)

        torch_backend_params["lp_width"] = lp_width
        torch_backend_params["rolloff"] = rolloff
        torch_backend_params["window"] = window

        return wav, target_sr, torch_backend_params

    else:
        """downsample and upsample back by Sox command"""
        effects1 = [
            ["rate", str(target_sr)],
        ]
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(wav, origin_sr, effects1)

        return wav, target_sr


def get_biquad_params(
    gain_dB: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str,
):
    """
    Use design parameters to generate coefficients for a specific filter type.

    Args:
        gain_dB (float): Shelving filter gain in dB
        cutoff_freq (float): Cutoff frequency in Hz.
        q_factor (float): Q factor.
        sample_rate (float): Sample rate in Hz.
        filter_type (str): Filter type. One of ["high_shelf", "low_shelf", "peaking", "lpf", "hpf", "bpf", "notch"]

    Returns:
        b : Numerator filter coefficients stored as [b0, b1, b2]
        a : Denominator filter coefficients stored as [a0, a1, a2]
    """
    filter_type = filter_type.lower()
    assert filter_type in [
        "high_shelf",
        "low_shelf",
        "peaking",
        "lpf",
        "hpf",
        "bpf",
        "notch",
    ]

    A = 10 ** (gain_dB / 40.0)  # log to linear gain
    w0 = 2.0 * np.pi * (cutoff_freq / sample_rate)
    alpha = np.sin(w0) / (2.0 * q_factor)

    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    elif filter_type == "lpf":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "hpf":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "bpf":
        b0 = sin_w0 / 2
        b1 = 0
        b2 = -(sin_w0 / 2)
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "notch":
        b0 = 1
        b1 = -2 * cos_w0
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a


def wav_apply_biquad_filter(
    wav: torch.Tensor, b_coeff: np.ndarray, a_coeff: np.ndarray
):
    """
    Applies the Biquad-Filter

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
    """
    proc_wav = wav.clone()
    if isinstance(wav, torch.Tensor):
        proc_wav = proc_wav.numpy()

    if proc_wav.ndim == 1:
        proc_wav = np.expand_dims(proc_wav, axis=0)

    ch = proc_wav.shape[0]

    for c in range(ch):
        proc_wav[c] = scipy.signal.lfilter(b_coeff, a_coeff, proc_wav[c])

    return torch.from_numpy(proc_wav)


class ParametricEQ:
    """
    Parametric EQ by series Biquad Filter

    Args:
        sample_rate: waveform sampling rate
        eq_band_gain: series of gain at each band
        eq_band_cutoff: series of cutoff frequency at each band
        eq_band_q_factor: series of Q factor at each band filter
        low_shelf_gain_dB: gain of low shelf filter
        low_shelf_cutoff_freq: cutoff frequency of low shelf filter
        low_shelf_q_factor: Q factor of low shelf filter
        high_shelf_gain_dB: gain of high shelf filter
        high_shelf_cutoff_freq: cutoff frequency of high shelf filter
        high_shelf_q_factor: Q factor of high shelf filter
        dtype: default data type in numpy
    """
    def __init__(
        self,
        sample_rate: float,
        eq_band_gain: Tuple[float],
        eq_band_cutoff: Tuple[float],
        eq_band_q_factor: Tuple[float],
        low_shelf_gain_dB: float = 0.0,
        low_shelf_cutoff_freq: float = 80,
        low_shelf_q_factor: float = 0.707,
        high_shelf_gain_dB: float = 0.0,
        high_shelf_cutoff_freq: float = 1000,
        high_shelf_q_factor: float = 0.707,
        dtype=np.float32,
    ):
        assert len(eq_band_gain) == len(eq_band_cutoff) == len(eq_band_q_factor)
        self.dtype = dtype
        self.sr = sample_rate

        self.b_list = []
        self.a_list = []

        b, a = get_biquad_params(
            gain_dB=low_shelf_gain_dB,
            cutoff_freq=low_shelf_cutoff_freq,
            q_factor=low_shelf_q_factor,
            sample_rate=sample_rate,
            filter_type="low_shelf",
        )
        self.b_list.append(b)
        self.a_list.append(a)

        for i in range(len(eq_band_gain)):
            b, a = get_biquad_params(
                gain_dB=eq_band_gain[i],
                cutoff_freq=eq_band_cutoff[i],
                q_factor=eq_band_q_factor[i],
                sample_rate=sample_rate,
                filter_type="peaking",
            )
            self.b_list.append(b)
            self.a_list.append(a)

        b, a = get_biquad_params(
            gain_dB=high_shelf_gain_dB,
            cutoff_freq=high_shelf_cutoff_freq,
            q_factor=high_shelf_q_factor,
            sample_rate=sample_rate,
            filter_type="high_shelf",
        )
        self.b_list.append(b)
        self.a_list.append(a)

        self.n_eq = len(self.a_list)

    def forward(self, wav: torch.Tensor):
        """
        Applies series of Filters

        Args:
            wav: The waveform used for computing amplitude. Shape should be [..., L]
        
        Returns:
            filtered wav has same shape of input
        """
        for i in range(self.n_eq):
            wav = wav_apply_biquad_filter(
                wav=wav, b_coeff=self.b_list[i], a_coeff=self.a_list[i]
            )

        return wav

    def plot_eq(self, savefig: Optional[str] = None):
        """Plotting the EQ curves"""
        if self.sr == 16000:
            nfft = 512
        elif self.sr == 32000:
            nfft = 1024
        else:
            raise ValueError

        b = torch.stack([torch.from_numpy(x) for x in self.b_list])
        a = torch.stack([torch.from_numpy(x) for x in self.a_list])

        B = torch.fft.rfft(b, nfft)
        A = torch.fft.rfft(a, nfft)

        H = B / A
        H = torch.prod(H, dim=0).view(-1)
        H = H.abs()
        faxis = torch.linspace(0, self.sr // 2, steps=(nfft // 2) + 1)
        plt.plot(faxis, H)
        plt.xlabel("Hz")
        plt.title("Parametric EQ")
        if savefig is not None:
            plt.savefig(savefig)
