from typing import Optional

import torch
import torchaudio

from puresound.utils import fftconvolve


def wav_apply_rir(
    wav: torch.Tensor,
    impaulse: torch.Tensor,
    sample_rate: int,
    rir_mode: str = "full",
):
    """
    Simulate reverberation data by convolue RIR in waveform.\n

    Args:
        wav: input waveform tensor with time dimension in the last tensor shape, i.e., [..., L]
        impaulse: rir tensor with shpae as  [chaneels, rir length]
        sample_rate: speech sample rate
        rir_mode: select in ["full", "direct", "early"]

    Returns:
        waveform has been convolved with RIR
    """
    rir_mode = rir_mode.lower()
    wav_ch, _ = wav.shape
    rir_ch, _ = impaulse.shape

    assert rir_mode in ["full", "direct", "early"]

    if rir_mode == "full":
        pass

    elif rir_mode == "direct":
        peak_idx = impaulse.argmax().item()
        direct_range = peak_idx + int(sample_rate * 0.006)  # 6ms range
        impaulse = impaulse[:, : int(direct_range)]

    elif rir_mode == "early":
        peak_idx = impaulse.argmax().item()
        early_range = peak_idx + int(sample_rate * 0.05)  # 50ms range
        impaulse = impaulse[:, : int(early_range)]

    impaulse = impaulse / torch.norm(impaulse, p=2)
    out = []
    if rir_ch == 1:
        for i in range(wav_ch):
            tmp_wav = fftconvolve(wav[i].view(1, -1), impaulse, mode="full")
            propagation_delays = impaulse.abs().argmax(dim=-1, keepdim=False)[0]
            tmp_wav = tmp_wav[
                ..., propagation_delays : propagation_delays + wav.shape[-1]
            ]
            out.append(tmp_wav)

    else:
        assert wav.shape[0] == 1, f"when rir chaneels not equal to 1 (ex: mic array inpaulse), wav must be single channel case."
        for i in range(rir_ch):
            tmp_wav = fftconvolve(wav, impaulse[i].view(1, -1), mode="full")
            propagation_delays = impaulse.abs().argmax(dim=-1, keepdim=False)[0]
            tmp_wav = tmp_wav[
                ..., propagation_delays : propagation_delays + wav.shape[-1]
            ]
            out.append(tmp_wav)

    out = torch.cat(out, dim=0)
    assert wav.shape[-1] == out.shape[-1]

    return out


def rand_add_2nd_filter_response(
    wav: torch.Tensor,
    a: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
):
    """
    Reference:
        [1] A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement
    """
    if a is None or b is None:
        r = torch.Tensor(4).uniform_(-3 / 8, 3 / 8)
        a = torch.Tensor([1, r[0], r[1]])
        b = torch.Tensor([1, r[2], r[3]])

    wav = torchaudio.functional.lfilter(
        wav, a_coeffs=a.to(wav.device), b_coeffs=b.to(wav.device)
    )

    return wav, a, b
