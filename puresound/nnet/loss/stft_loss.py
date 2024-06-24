"""
Refers:
    https://github.com/facebookresearch/denoiser/blob/f98f16ce55fbf23e60cfd12e0cc3f5964f5b8dba/denoiser/stft_loss.py
    https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/loss.py#L137
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
    )
    real = x_stft.real
    imag = x_stft.imag

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-8)).transpose(2, 1)


def as_complex(x: torch.Tensor):
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(
            f"Last dimension need to be of length 2 (re + im), but got {x.shape}"
        )
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(
            torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1)
        )


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(
        self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"
    ):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        factor_sc=0.1,
        factor_mag=0.1,
    ):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss


"""
def over_suppression_loss(enh, ref, p=0.5, fft_size=512, hop_size=128, win_length=512):
    window = torch.hann_window(win_length)
    enh_mag = stft(enh, fft_size, hop_size, win_length, window)
    ref_mag = stft(ref, fft_size, hop_size, win_length, window)
    loss = ref_mag.pow(p) - enh_mag.pow(p)
    mask = loss > 0
    mask = mask.float()
    loss = torch.mean((loss * mask).pow(2))
    return loss
"""


class OverSuppressionLoss(torch.nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        fft_size: int = 512,
        hop_size: int = 128,
        win_length: int = 512,
    ):
        super().__init__()
        self.p = p
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_len = win_length

        window = torch.hann_window(window_length=win_length)
        self.register_buffer("hann_window", window)

    def forward(self, enh: torch.Tensor, ref: torch.Tensor):
        enh_mag = stft(
            enh, self.fft_size, self.hop_size, self.win_len, self.hann_window
        )
        ref_mag = stft(
            ref, self.fft_size, self.hop_size, self.win_len, self.hann_window
        )
        loss = ref_mag.pow(self.p) - enh_mag.pow(self.p)
        mask = loss > 0
        mask = mask.float()
        loss = torch.mean((loss * mask).pow(2))
        return loss


class SpectralLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 1,
        factor_magnitude: float = 1,
        factor_complex: float = 1,
        factor_under: float = 1,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, enh: torch.Tensor, ref: torch.Tensor):
        """
        Args:
            enh (Tensor or ComplexTensor): enhanced spectral has shape [N, *, C, T, 2] or [N, *, C, T]
            ref (Tensor or ComplexTensor): target spectral shape [N, *, C, T, 2] or [N, *, C, T]

        Return:
            loss (Tensor): scalar
        """
        enh = as_complex(enh)
        ref = as_complex(ref)
        enh_abs = enh.abs()
        ref_abs = ref.abs()
        if self.gamma != 1:
            enh_abs = enh_abs.clamp_min(1e-12).pow(self.gamma)
            ref_abs = ref_abs.clamp_min(1e-12).pow(self.gamma)
        tmp = (enh_abs - ref_abs).pow(2)
        if self.f_u != 1:
            # Weighting if predicted abs is too low
            tmp *= torch.where(enh_abs < ref_abs, self.f_u, 1.0)
        loss = torch.mean(tmp) * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                enh = enh_abs * torch.exp(1j * angle.apply(enh))
                ref = ref_abs * torch.exp(1j * angle.apply(ref))
            loss_c = (
                F.mse_loss(torch.view_as_real(enh), torch.view_as_real(ref)) * self.f_c
            )
            loss = loss + loss_c

        return loss
