"""Perceptual contrast stretching (PCS): a per-bin gamma on the log1p magnitude.

Chao et al., "Perceptual Contrast Stretching on Target Feature for Speech
Enhancement" (Interspeech 2022). Each bin's magnitude becomes
``expm1(gamma * log1p(|X|))``, phase untouched; ``gamma`` rises from 1.0 at the
band edges to 1.4 over 400-4400 Hz, following the critical-band importance
function, so formant peaks come out sharper where hearing is most sensitive.
The band table reproduces the authors' 512-point/16 kHz bin assignment exactly
and extends it to any STFT size by frequency.

The transform is memoryless per frame, so it adds no latency to a streaming
runtime, and it is level-dependent: the gammas were tuned on the unnormalised
STFT magnitude of a full-scale ``[-1, 1]`` waveform (sum-of-window scale), which
is what every STFT in this package produces.
"""

from __future__ import annotations

import numpy as np
import torch

#: ``(upper edge in Hz, gamma)`` from the lowest band up; bins at or above the
#: last edge keep gamma 1. Edges are the authors' 512-point bin boundaries
#: (bins 3, 6, 9, 12, 138, 166, 200, 241, 256 at 16 kHz) expressed in Hz.
PCS_BANDS: tuple[tuple[float, float], ...] = (
    (93.75, 1.0),
    (187.5, 1.070175439),
    (281.25, 1.182456140),
    (375.0, 1.287719298),
    (4312.5, 1.4),
    (5187.5, 1.322807018),
    (6250.0, 1.238596491),
    (7531.25, 1.161403509),
    (8000.0, 1.077192982),
)


def pcs_gamma(n_fft: int, sample_rate: int) -> np.ndarray:
    """Per-bin gamma for an ``n_fft``-point STFT at ``sample_rate``, ``[n_fft//2 + 1]``."""
    freqs = np.fft.rfftfreq(int(n_fft), 1.0 / float(sample_rate))
    gamma = np.ones_like(freqs)
    lower = 0.0
    for upper, value in PCS_BANDS:
        gamma[(freqs >= lower) & (freqs < upper)] = value
        lower = upper
    return gamma


def stretch_magnitude(magnitude: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """``expm1(gamma * log1p(magnitude))``; ``gamma`` broadcasts over the bin axis."""
    return torch.expm1(gamma * torch.log1p(magnitude))


def stretch_magnitude_np(magnitude: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    return np.expm1(gamma * np.log1p(magnitude))
