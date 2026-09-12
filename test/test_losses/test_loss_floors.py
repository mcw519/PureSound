"""The three gradient/label floors found by the v17 investigation, each behind a
flag whose default reproduces the historical behaviour bit for bit.

* MultiResolutionSTFTLoss: |X|^2 clamped at 1e-8 = a magnitude floor with zero
  gradient below it (32% of a -53 dBFS clean target's bins pinned).
* SDRLoss inactive path: absolute-energy objective on a SUM -> eps floor moves
  10 dB between 3 s and 30 s rows; a quieter input earns a free discount.
* EnergyVADLabeler: fixed 1e-8 clamp -> all-frames-active below -40 dBFS peak.
"""
import math

import pytest
import torch

from puresound.audio.vad import EnergyVADLabeler
from puresound.nnet.loss.sdr import SDRLoss, inactive_sdr_loss
from puresound.nnet.loss.stft_loss import MultiResolutionSTFTLoss, stft


# ---------------------------------------------------------------- STFT floor
def test_stft_default_is_the_historical_clamp():
    torch.manual_seed(0)
    x = torch.randn(2, 16000) * 1e-3
    w = torch.hann_window(512)
    new = stft(x, 512, 128, 512, w)
    ref = torch.sqrt(torch.clamp((torch.stft(x, 512, 128, 512, w, return_complex=True).abs() ** 2),
                                 min=1e-8)).transpose(2, 1)
    assert torch.allclose(new, ref, atol=0, rtol=1e-6)


def test_relative_floor_follows_the_row_level():
    """The contract: with relative_floor the pinned-bin fraction is a property of the
    SIGNAL, not of its level; with the absolute clamp it grows as the row gets quieter.
    Signal = harmonics with a broadband floor 50 dB below the peak (speech-like valleys)."""
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


def test_mrstft_flags_reach_every_resolution():
    m = MultiResolutionSTFTLoss(fft_sizes=[256, 512], hop_sizes=[64, 128], win_lengths=[256, 512],
                                power_floor=1e-12, relative_floor=True)
    assert all(l.power_floor == 1e-12 and l.relative_floor for l in m.stft_losses)


# ---------------------------------------------------------- inactive SDR modes
def test_inactive_absolute_is_unchanged_and_length_dependent():
    torch.manual_seed(0)
    for n in (3 * 16000, 30 * 16000):
        s1 = torch.zeros(1, n); s2 = torch.zeros(1, n)
        v = inactive_sdr_loss(s1, s2, reduction=False).item()
        assert math.isclose(v, -80.0, abs_tol=1e-3)           # the 1e-8 floor
    # a real residual: absolute value shifts with length because it is a SUM
    g = torch.Generator().manual_seed(1)
    r3 = inactive_sdr_loss(torch.randn(1, 3 * 16000, generator=g) * 1e-3, torch.zeros(1, 3 * 16000), reduction=False)
    r30 = inactive_sdr_loss(torch.randn(1, 30 * 16000, generator=g) * 1e-3, torch.zeros(1, 30 * 16000), reduction=False)
    assert abs((r30 - r3).item() - 10.0) < 0.1


def test_inactive_mean_is_length_invariant():
    g = torch.Generator().manual_seed(1)
    r3 = inactive_sdr_loss(torch.randn(1, 3 * 16000, generator=g) * 1e-3, torch.zeros(1, 3 * 16000), reduction=False, mode="mean")
    r30 = inactive_sdr_loss(torch.randn(1, 30 * 16000, generator=g) * 1e-3, torch.zeros(1, 30 * 16000), reduction=False, mode="mean")
    assert abs((r30 - r3).item()) < 0.1


def test_inactive_relative_scores_the_reduction_not_the_level():
    torch.manual_seed(0)
    noisy = torch.randn(1, 16000)
    enh = noisy * 10 ** (-20 / 20)                               # 20 dB of suppression
    for gain in (1.0, 10 ** (-30 / 20)):                         # loud and quiet rows
        v = inactive_sdr_loss(enh * gain, torch.zeros_like(enh), reduction=False,
                              mode="relative", noisy=noisy * gain).item()
        assert abs(v + 20.0) < 0.2


def test_sdrloss_relative_mode_declares_and_uses_the_batch():
    loss = SDRLoss(scaled=False, scale_dependent=True, zero_mean=True, inactive_mode="relative")
    assert "batch" in loss.required_inputs
    torch.manual_seed(0)
    noisy = torch.randn(2, 16000)
    enh = torch.stack([noisy[0] * 0.1, noisy[1]])                 # row0 suppressed 20 dB, row1 untouched
    target = torch.zeros_like(enh)
    labels = torch.tensor([True, True])
    v = loss(enh, target, inactive_labels=labels, batch={"noisy_speech": noisy})
    assert -12 < v.item() < -8                                     # mean of (-20, 0)
    default = SDRLoss(scaled=False, scale_dependent=True, zero_mean=True)
    assert "batch" not in default.required_inputs
    with pytest.raises(ValueError):
        SDRLoss(inactive_mode="bogus")


# ------------------------------------------------------------- energy VAD eps
def _speech_like():
    torch.manual_seed(0)
    x = torch.zeros(1, 16000 * 2)
    x[:, 4000:12000] = torch.randn(8000)                          # 0.5 s burst, rest silence
    return x


def test_energy_vad_absolute_saturates_below_minus_40_dbfs_peak():
    lab = EnergyVADLabeler()
    loud = lab(_speech_like()).mean().item()
    quiet = lab(_speech_like() * 10 ** (-60 / 20)).mean().item()
    assert loud < 0.6
    assert quiet > 0.95, "historical labeler must go all-active on a quiet row (the documented bug)"


def test_energy_vad_relative_holds_the_label_at_any_level():
    lab = EnergyVADLabeler(eps_mode="relative")
    loud = lab(_speech_like()).mean().item()
    quiet = lab(_speech_like() * 10 ** (-60 / 20)).mean().item()
    assert abs(loud - quiet) < 0.02
    with pytest.raises(ValueError):
        EnergyVADLabeler(eps_mode="bogus")
