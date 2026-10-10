"""Two measures for what a bottleneck throws away.

A U-net's decoder upsamples back to full resolution, so "the output is still 256
bins" says nothing about whether the bottleneck could carry the structure that had
to be there. These two ask that question directly, and they ask it **against the
target on the same item**, so the numbers compare across architectures rather than
across content.

:func:`harmonic_contrast_db`
    Voiced speech is a comb: peaks at multiples of F0, valleys between. A bottleneck
    too coarse in frequency produces a mask that cannot follow the comb, and the
    enhanced signal's peaks and valleys flatten toward each other. This separates
    "suppressed the noise" from "smeared the speech"; a single energy ratio cannot,
    because it adds the two together.

:func:`transient_correlation`
    Keyboard, mouse and door noise are milliseconds long, and they are a large
    share of the DNS dev set. A bottleneck too coarse in time smears their edges.
    Measured as the correlation between the enhanced and target log-energy slopes at
    a time resolution finer than any frame the model sees.
"""

from __future__ import annotations

import torch


def _log_magnitude(
    wav: torch.Tensor, n_fft: int, hop: int, eps: float = 1e-8
) -> torch.Tensor:
    """``[frames, bins]`` log-magnitude, batch collapsed to mono."""
    wav = wav.reshape(-1).float()
    if wav.numel() < n_fft:
        # torch.stft pads by n_fft // 2 and refuses a signal shorter than that.
        # A clip too short to hold a frame has no structure to measure.
        return torch.zeros(0, n_fft // 2 + 1, device=wav.device)
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        window=window, return_complex=True, center=True,
    )
    return torch.log10(spec.abs().transpose(0, 1) + eps) * 20.0


def harmonic_contrast_db(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    *,
    n_fft: int = 512,
    hop: int = 160,
    band_hz: tuple[float, float] = (100.0, 2000.0),
    active_percentile: float = 0.6,
) -> float:
    """Mean peak-minus-valley contrast of the spectrum, in dB.

    Computed only on the loudest ``1 - active_percentile`` of frames: silence has
    no comb to measure, and including it would report the noise floor's flatness
    as if it were the speech's.

    Peaks and valleys are local extrema along frequency inside ``band_hz`` -- low
    enough that the harmonics are resolved by a 512-point FFT, high enough to hold
    several of them.
    """
    magnitude = _log_magnitude(wav, n_fft, hop)
    if magnitude.shape[0] < 3:
        return float("nan")

    frame_energy = magnitude.mean(dim=1)
    threshold = torch.quantile(frame_energy, active_percentile)
    voiced = magnitude[frame_energy >= threshold]
    if voiced.shape[0] == 0:
        return float("nan")

    bin_hz = sample_rate / n_fft
    low = max(1, int(band_hz[0] / bin_hz))
    high = min(voiced.shape[1] - 1, int(band_hz[1] / bin_hz))
    if high - low < 5:
        return float("nan")

    band = voiced[:, low - 1 : high + 2]
    centre, left, right = band[:, 1:-1], band[:, :-2], band[:, 2:]
    is_peak = (centre > left) & (centre > right)
    is_valley = (centre < left) & (centre < right)

    contrasts = []
    for frame in range(centre.shape[0]):
        peaks, valleys = centre[frame][is_peak[frame]], centre[frame][is_valley[frame]]
        if peaks.numel() and valleys.numel():
            contrasts.append(float(peaks.mean() - valleys.mean()))
    if not contrasts:
        return float("nan")
    return sum(contrasts) / len(contrasts)


def transient_correlation(
    enhanced: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int = 16000,
    *,
    n_fft: int = 128,
    hop: int = 32,
) -> float:
    """Correlation of enhanced vs target log-energy *slope*, in ``[-1, 1]``.

    The slope, not the envelope: two signals with the same average loudness and
    different attack sharpness correlate highly on the envelope and poorly here,
    and attack sharpness is what a coarse time axis destroys.

    The 8 ms window / 2 ms hop is deliberately finer than any frame the model
    computes on, so the measure can see a smear the model's own frame rate cannot.
    """
    left = _log_magnitude(enhanced, n_fft, hop).mean(dim=1)
    right = _log_magnitude(target, n_fft, hop).mean(dim=1)
    length = min(left.shape[0], right.shape[0])
    if length < 4:
        return float("nan")

    left_slope = left[1:length] - left[: length - 1]
    right_slope = right[1:length] - right[: length - 1]
    left_slope = left_slope - left_slope.mean()
    right_slope = right_slope - right_slope.mean()

    denominator = left_slope.norm() * right_slope.norm()
    if float(denominator) < 1e-8:
        return float("nan")
    return float((left_slope @ right_slope) / denominator)


def harmonic_contrast_gap_db(
    enhanced: torch.Tensor, target: torch.Tensor, sample_rate: int = 16000, **kwargs
) -> float:
    """``contrast(enhanced) - contrast(target)``; negative means smeared.

    The gap rather than the absolute value, because the absolute number depends on
    the speaker and the utterance -- comparing two architectures needs the part
    that does not.
    """
    return harmonic_contrast_db(enhanced, sample_rate, **kwargs) - harmonic_contrast_db(
        target, sample_rate, **kwargs
    )
