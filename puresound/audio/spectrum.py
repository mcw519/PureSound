import torch


def tensor_as_complex(x: torch.Tensor):
    """Check and convert input tensor to complex tensor."""
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(
            f"Last dimension need to be of length 2 (re + im), but got {x.shape}"
        )
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def cpx_stft_as_mag_and_phase(x: torch.Tensor, eps: float = 1e-8):
    """Check and convert input complex stft tensor to tensor with magnitude and phase."""
    if not torch.is_complex(x):
        x = tensor_as_complex(x)

    if eps is not None:
        mag = torch.sqrt(torch.pow(x.real, 2) + torch.pow(x.imag, 2) + eps)
    else:
        mag = torch.sqrt(torch.pow(x.real, 2) + torch.pow(x.imag, 2))
    phase = torch.atan2(x.imag, x.real)
    return mag, phase


def mag_and_phase_as_cpx_stft(mag: torch.Tensor, phase: torch.Tensor):
    """Check and convert magnitude and phase back to complex stft."""
    assert mag.shape == phase.shape
    cpx_stft = mag * torch.exp(1j * phase)
    return cpx_stft


def wav_to_stft(
    wav: torch.Tensor,
    nfft: int = 512,
    win_size: int = 512,
    hop_size: int = 128,
    window_type: str = "hann_window",
    stft_normalized: bool = False,
):
    """
    Convert a wavform to STFT

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        nfft: numbers of FFT size
        win_size: window length
        hop_size: window shifting length
        window_type: most use type is "hann_window" or "hamming_window"
        stft_normalized: the function returns the normalized STFT results, i.e., multiplied by win_size^-0.5

    Returns:
        complex stft tensor with shape [batch, nfft, nframes]
    """
    win = getattr(torch, window_type)(window_length=win_size, device=wav.device)
    cpx_stft = torch.stft(
        input=wav,
        n_fft=nfft,
        win_length=win_size,
        hop_length=hop_size,
        window=win,
        normalized=stft_normalized,
        return_complex=True,
    )
    stft_info = {
        "nfft": nfft,
        "win_size": win_size,
        "hop_size": hop_size,
        "window_type": window_type,
        "stft_normalized": stft_normalized,
    }

    return cpx_stft, stft_info


def stft_to_wav(
    x: torch.Tensor,
    nfft: int = 512,
    win_size: int = 512,
    hop_size: int = 128,
    window_type: str = "hann_window",
    stft_normalized: bool = False,
):
    """
    Convert a STFT back to wavform

    Args:
        wav: The waveform used for computing amplitude. Shape should be [..., L]
        nfft: numbers of FFT size
        win_size: window length
        hop_size: window shifting length
        window_type: most use type is "hann_window" or "hamming_window"
        stft_normalized: the function returns the normalized STFT results, i.e., multiplied by win_size^-0.5

    Returns:
        waveform tensor with shape [..., L]
    """
    x = tensor_as_complex(x)
    win = getattr(torch, window_type)(window_length=win_size, device=x.device)
    wav = torch.istft(
        x,
        n_fft=nfft,
        win_length=win_size,
        hop_length=hop_size,
        window=win,
        normalized=stft_normalized,
    )
    return wav
