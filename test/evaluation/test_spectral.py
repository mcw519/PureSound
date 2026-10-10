"""What the two bottleneck measures must be able to tell apart."""

import math

import pytest
import torch

from puresound.evaluation.spectral import (
    harmonic_contrast_db,
    harmonic_contrast_gap_db,
    transient_correlation,
)

SR = 16000


@pytest.fixture
def comb():
    """Voiced-like: F0 125 Hz with 24 harmonics -- peaks and valleys to measure."""
    t = torch.arange(SR * 2) / SR
    return sum(torch.sin(2 * torch.pi * 125 * k * t) / k for k in range(1, 25)).view(1, -1) * 0.05


@pytest.fixture
def clicks():
    wav = torch.zeros(1, SR)
    wav[0, ::1600] = 1.0
    return wav


def smear_along_frequency(wav, width=5, n_fft=512, hop=160):
    """Blur the spectrum across frequency and resynthesise.

    This is what a bottleneck too coarse in frequency does. A time-domain
    low-pass is NOT the same thing -- it removes high frequencies while leaving
    the comb's peaks and valleys exactly where they were.
    """
    window = torch.hann_window(n_fft)
    spec = torch.stft(wav.reshape(-1), n_fft, hop, n_fft, window, return_complex=True)
    kernel = torch.ones(1, 1, width) / width
    blurred = torch.nn.functional.conv1d(
        spec.abs().transpose(0, 1).unsqueeze(1), kernel, padding=width // 2
    ).squeeze(1).transpose(0, 1)
    return torch.istft(
        blurred * torch.exp(1j * spec.angle()), n_fft, hop, n_fft, window
    ).view(1, -1)


def smear_along_time(wav):
    return torch.nn.functional.avg_pool1d(
        wav.view(1, 1, -1), kernel_size=129, stride=1, padding=64
    ).view(1, -1)


def test_contrast_ranks_a_comb_above_the_same_comb_with_filled_valleys_above_noise(comb):
    """Energy in the valleys reads as a flatter comb, which a single energy ratio
    cannot distinguish from less speech."""
    noise = torch.randn(1, SR * 2) * 0.02
    noisy = comb + torch.randn_like(comb) * 0.02
    assert harmonic_contrast_db(comb) > harmonic_contrast_db(noisy)
    assert harmonic_contrast_db(comb) > 3 * harmonic_contrast_db(noise)


@pytest.mark.parametrize("gain, tolerance", [(1.0, 1e-6), (0.3, 0.5)])
def test_gain_is_not_reported_as_structure(comb, clicks, gain, tolerance):
    """Peaks and valleys move together under a gain, and a constant level offset
    leaves the log-energy slope unchanged."""
    assert harmonic_contrast_gap_db(comb * gain, comb) == pytest.approx(0.0, abs=tolerance)
    assert transient_correlation(clicks * gain, clicks) == pytest.approx(1.0, abs=1e-3)


def test_each_smear_is_caught_by_its_own_measure(comb, clicks):
    assert harmonic_contrast_gap_db(smear_along_frequency(comb), comb) < -1.0
    assert transient_correlation(smear_along_time(clicks), clicks) < 0.2


def test_the_measures_answer_different_questions(comb, clicks):
    """A frequency smear and a time smear must not look like the same defect.

    Both are applied to one signal carrying both structures -- a comb and the
    impulses -- because that is the case the probe actually reads.
    """
    signal = comb[:, : clicks.shape[-1]] + clicks * 0.3

    freq_smeared = smear_along_frequency(signal)
    time_smeared = smear_along_time(signal)

    freq_gap = harmonic_contrast_gap_db(freq_smeared, signal)
    freq_corr = transient_correlation(freq_smeared, signal)
    time_gap = harmonic_contrast_gap_db(time_smeared, signal)
    time_corr = transient_correlation(time_smeared, signal)

    # Each defect shows up mostly in its own measure, not in both.
    assert freq_gap < time_gap, (freq_gap, time_gap)
    assert time_corr < freq_corr, (time_corr, freq_corr)


@pytest.mark.parametrize(
    "measure",
    [
        lambda: harmonic_contrast_db(torch.zeros(1, SR)),
        lambda: harmonic_contrast_db(torch.randn(1, 100)),
        lambda: transient_correlation(torch.ones(1, SR) * 0.1, torch.ones(1, SR) * 0.1),
    ],
    ids=["silence", "too-short", "constant-signal"],
)
def test_an_unmeasurable_signal_reports_nan_rather_than_a_confident_number(measure):
    assert math.isnan(measure())
