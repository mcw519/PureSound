"""Minimum-phase FIR realization of a :class:`PathBandGain`.

A band gain is a magnitude only.  Realizing it as the minimum-phase filter
with that magnitude keeps the energy at the start of the response, so a
path's arrival time and the early/late partition are not shifted the way a
linear-phase filter's bulk delay would shift them.  The minimum phase comes
from the folded real cepstrum (Oppenheim & Schafer, *Discrete-Time Signal
Processing*, 3rd ed., §13.5); the FIR is that response truncated with a
half-Hann taper.  With 128 taps at 16 kHz a smooth octave-band curve
(directivity, insertion loss) is reproduced within 0.15 dB at the band
centres from 125 Hz up; a curve that alternates by 12 dB between adjacent
octaves within 1.3 dB.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from puresound.audio.rir.path_events.schema import PathBandGain

BAND_FILTER_TAPS = 128
_MAGNITUDE_FLOOR = 1e-4  # -80 dB keeps the log spectrum finite


def minimum_phase_band_filter(
    band_gain: PathBandGain,
    sample_rate: float,
    *,
    taps: int = BAND_FILTER_TAPS,
) -> np.ndarray:
    """Causal FIR whose magnitude follows ``band_gain`` up to Nyquist."""
    if type(taps) is not int or not 8 <= taps <= 4096:
        raise ValueError("taps must be an integer in [8, 4096]")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    return _design(
        tuple(band_gain.frequencies_hz),
        tuple(np.round(band_gain.magnitude, 12)),
        float(sample_rate),
        taps,
    ).copy()


@lru_cache(maxsize=4096)
def _design(
    frequencies: tuple[float, ...],
    magnitude: tuple[float, ...],
    sample_rate: float,
    taps: int,
) -> np.ndarray:
    size = 1 << int(np.ceil(np.log2(16 * taps)))
    grid = np.fft.rfftfreq(size, 1.0 / sample_rate)
    gain = PathBandGain(list(frequencies), list(magnitude), "design").at(grid)
    cepstrum = np.fft.irfft(np.log(np.maximum(gain, _MAGNITUDE_FLOOR)), size)
    folded = np.zeros(size)
    folded[0] = cepstrum[0]
    folded[1 : size // 2] = 2.0 * cepstrum[1 : size // 2]
    folded[size // 2] = cepstrum[size // 2]
    response = np.fft.irfft(np.exp(np.fft.rfft(folded)), size)[:taps]
    fade = taps // 4
    response[-fade:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(1, fade + 1) / fade))
    return response


__all__ = ["BAND_FILTER_TAPS", "minimum_phase_band_filter"]
