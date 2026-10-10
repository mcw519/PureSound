"""The STFT magnitude floor: an absolute clamp on |X|^2 is a magnitude floor with
zero gradient below it, so it pins more of a quiet row's bins than a loud
row's. ``relative_floor`` ties the floor to the row's own peak; the default
keeps the absolute clamp bit for bit."""

import math

import torch

from puresound.nnet.loss.stft_loss import MultiResolutionSTFTLoss, stft


def test_the_default_floor_is_the_absolute_clamp():
    torch.manual_seed(0)
    x = torch.randn(2, 16000) * 1e-3
    w = torch.hann_window(512)
    ref = torch.sqrt(
        torch.clamp(torch.stft(x, 512, 128, 512, w, return_complex=True).abs() ** 2, min=1e-8)
    ).transpose(2, 1)
    assert torch.allclose(stft(x, 512, 128, 512, w), ref, atol=0, rtol=1e-6)

    m = MultiResolutionSTFTLoss(fft_sizes=[256, 512], hop_sizes=[64, 128],
                                win_lengths=[256, 512], power_floor=1e-12, relative_floor=True)
    assert all(l.power_floor == 1e-12 and l.relative_floor for l in m.stft_losses), (
        "the floor flags must reach every resolution"
    )


def test_the_relative_floor_pins_the_same_fraction_of_bins_at_any_level():
    """Signal = harmonics with a broadband floor 50 dB below the peak (speech-like valleys)."""
    g = torch.Generator().manual_seed(0)
    t = torch.arange(32000) / 16000.0
    harm = sum(torch.sin(2 * math.pi * 140 * k * t) / k for k in range(1, 12))
    harm = harm / harm.abs().max()
    sig = harm + torch.randn(32000, generator=g) * 10 ** (-50 / 20)
    w = torch.hann_window(512)

    def pinned(x, relative):
        mag = stft(x.view(1, -1), 512, 128, 512, w, power_floor=1e-8, relative_floor=relative)
        floor = math.sqrt(1e-8) * (mag.max() if relative else 1.0)
        return (mag <= floor * (1 + 1e-6)).float().mean().item()

    loud, quiet = sig * 10 ** (-20 / 20), sig * 10 ** (-75 / 20)
    assert abs(pinned(loud, True) - pinned(quiet, True)) < 1e-3, "relative floor must be level-invariant"
    assert pinned(quiet, False) > pinned(loud, False) + 0.10, "absolute floor must bite harder on the quiet row"
    assert pinned(quiet, True) < pinned(quiet, False), "relative floor frees bins the absolute one pins"
