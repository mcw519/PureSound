"""Spectral tilt and octave-band filtering.

``valid_octave_centers`` is the gate that decides which octave bands a sample
rate can carry: a band survives only when its whole nominal width stays below
Nyquist, so 16 kHz stops at the 4 kHz band.  Moved out of
``puresound.audio.rir.metrics`` in R4 of ``RIR_MODULARIZATION_PLAN.md``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Iterable, Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from puresound.audio.rir.metrics.core import _as_mono_numpy


DEFAULT_OCTAVE_CENTERS_HZ = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)


def spectral_tilt_db_per_octave(
    rir: Any,
    sample_rate: int,
    low_hz: float = 200.0,
    high_hz: float = 4000.0,
) -> Optional[float]:
    """Fit the RIR log-magnitude slope over a frequency interval."""
    x = _as_mono_numpy(rir)
    if x.size < 2:
        return None
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    nyquist = 0.5 * float(sample_rate)
    low = max(float(low_hz), float(sample_rate) / max(x.size, 1))
    high = min(float(high_hz), nyquist)
    if not 0.0 < low < high:
        return None
    fft_size = 1 << int(math.ceil(math.log2(max(2, x.size))))
    magnitude = np.abs(np.fft.rfft(x, n=fft_size))
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / float(sample_rate))
    mask = (frequencies >= low) & (frequencies <= high)
    if int(mask.sum()) < 8:
        return None
    log_frequency = np.log2(frequencies[mask])
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude[mask], 1e-12))
    centered = log_frequency - float(log_frequency.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return None
    return float(
        np.dot(centered, magnitude_db - float(magnitude_db.mean())) / denominator
    )


def valid_octave_centers(
    sample_rate: int,
    centers_hz: Iterable[float] = DEFAULT_OCTAVE_CENTERS_HZ,
) -> list[float]:
    """Return octave centers whose full nominal band lies below Nyquist."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    nyquist = 0.5 * float(sample_rate)
    return [
        float(center)
        for center in centers_hz
        if center > 0.0 and center * math.sqrt(2.0) < nyquist * 0.99
    ]


def octave_band_rir(
    rir: Any,
    sample_rate: int,
    center_hz: float,
    filter_order: int = 4,
) -> np.ndarray:
    """Causally filter one RIR into a nominal one-octave band."""
    x = _as_mono_numpy(rir)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if filter_order < 1:
        raise ValueError("filter_order must be positive")
    sos = _octave_band_sos(
        sample_rate=int(sample_rate),
        center_hz=float(center_hz),
        filter_order=int(filter_order),
    )
    return np.asarray(sosfilt(sos, x), dtype=np.float64)


@lru_cache(maxsize=128)
def _octave_band_sos(
    sample_rate: int,
    center_hz: float,
    filter_order: int,
) -> np.ndarray:
    """Design and cache an octave filter shared by every analyzed RIR."""
    low = center_hz / math.sqrt(2.0)
    high = center_hz * math.sqrt(2.0)
    nyquist = 0.5 * float(sample_rate)
    if not 0.0 < low < high < nyquist:
        raise ValueError(
            f"octave band centered at {center_hz:g} Hz is invalid for {sample_rate} Hz"
        )
    return butter(
        filter_order,
        [low, high],
        btype="bandpass",
        fs=float(sample_rate),
        output="sos",
    )
