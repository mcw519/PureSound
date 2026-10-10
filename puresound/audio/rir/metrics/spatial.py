"""Binaural IACC and array / diffuse-field coherence.

These require genuinely synchronized receivers.  Source-indexed multichannel
banks must report them as not-applicable rather than correlating unrelated
source channels.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np
from scipy.signal import csd, welch

from puresound.audio.rir.metrics.core import _as_mono_numpy, direct_sample
from puresound.audio.rir.metrics.spectral import (
    octave_band_rir,
    valid_octave_centers,
)


IACC_POLICY = "puresound.binaural_iacc.v1"

DIFFUSE_FIELD_COHERENCE_POLICY = "puresound.diffuse_field_coherence.v1"


def _pair_window(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate: int,
    direct_index: int,
    start_ms: float,
    end_ms: Optional[float],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if not np.isfinite(start_ms) or start_ms < 0.0:
        raise ValueError("start_ms must be finite and non-negative")
    if end_ms is not None and (
        not np.isfinite(end_ms) or end_ms <= start_ms
    ):
        raise ValueError("end_ms must be finite and greater than start_ms")
    length = min(first.size, second.size)
    start = direct_index + int(round(start_ms * 1e-3 * sample_rate))
    end = (
        length
        if end_ms is None
        else direct_index + int(round(end_ms * 1e-3 * sample_rate))
    )
    start = int(np.clip(start, 0, length))
    end = int(np.clip(end, start, length))
    return first[start:end], second[start:end], start, end


def interaural_cross_correlation(
    left_rir: Any,
    right_rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    start_ms: float = 0.0,
    end_ms: Optional[float] = 80.0,
    maximum_lag_ms: float = 1.0,
) -> dict[str, Any]:
    """Return maximum absolute normalized inter-channel correlation.

    This is the time-domain IACC primitive. The normalization uses the fixed
    energy of both selected windows; lagged products outside the window are
    treated as zero. A signed coefficient and its lag are retained alongside
    the conventional non-negative maximum absolute value.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(maximum_lag_ms) or maximum_lag_ms < 0.0:
        raise ValueError("maximum_lag_ms must be finite and non-negative")
    left = _as_mono_numpy(left_rir)
    right = _as_mono_numpy(right_rir)
    if direct_index is None:
        direct_index = min(direct_sample(left), direct_sample(right))
    length = min(left.size, right.size)
    direct_index = int(np.clip(direct_index, 0, max(0, length - 1)))
    left_window, right_window, start, end = _pair_window(
        left,
        right,
        sample_rate,
        direct_index,
        start_ms,
        end_ms,
    )
    maximum_lag_samples = max(
        0,
        int(round(maximum_lag_ms * 1e-3 * sample_rate)),
    )
    energy_product = float(
        np.dot(left_window, left_window) * np.dot(right_window, right_window)
    )
    if left_window.size == 0 or energy_product <= np.finfo(np.float64).tiny:
        return {
            "policy": IACC_POLICY,
            "valid": False,
            "reason": "empty_or_zero_energy_window",
            "iacc": None,
            "signed_correlation": None,
            "lag_samples": None,
            "lag_ms": None,
            "start_ms": float(start_ms),
            "end_ms": float(end_ms) if end_ms is not None else None,
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "maximum_lag_ms": float(maximum_lag_ms),
        }

    normalizer = math.sqrt(energy_product)
    best_correlation = 0.0
    best_lag = 0
    for lag in range(-maximum_lag_samples, maximum_lag_samples + 1):
        if lag < 0:
            first = left_window[:lag]
            second = right_window[-lag:]
        elif lag > 0:
            first = left_window[lag:]
            second = right_window[:-lag]
        else:
            first = left_window
            second = right_window
        correlation = float(np.dot(first, second) / normalizer)
        if abs(correlation) > abs(best_correlation):
            best_correlation = correlation
            best_lag = lag
    return {
        "policy": IACC_POLICY,
        "valid": True,
        "reason": "ok",
        "iacc": float(abs(best_correlation)),
        "signed_correlation": float(best_correlation),
        "lag_samples": int(best_lag),
        "lag_ms": float(1000.0 * best_lag / sample_rate),
        "start_ms": float(start_ms),
        "end_ms": float(end_ms) if end_ms is not None else None,
        "window_start_sample": int(start),
        "window_end_sample": int(end),
        "maximum_lag_ms": float(maximum_lag_ms),
    }


def analyze_binaural_iacc(
    left_rir: Any,
    right_rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    early_end_ms: float = 80.0,
    late_end_ms: Optional[float] = None,
    maximum_lag_ms: float = 1.0,
    centers_hz: Iterable[float] = (500.0, 1000.0, 2000.0, 4000.0),
    filter_order: int = 4,
) -> dict[str, Any]:
    """Analyze early/late broadband and four-band binaural IACC."""

    left = _as_mono_numpy(left_rir)
    right = _as_mono_numpy(right_rir)
    length = min(left.size, right.size)
    if direct_index is None:
        direct_index = min(direct_sample(left), direct_sample(right))
        direct_policy = "earliest_absolute_peak"
    else:
        direct_policy = "provided_shared_anchor"
    direct_index = int(np.clip(direct_index, 0, max(0, length - 1)))

    early = interaural_cross_correlation(
        left,
        right,
        sample_rate,
        direct_index=direct_index,
        start_ms=0.0,
        end_ms=early_end_ms,
        maximum_lag_ms=maximum_lag_ms,
    )
    late = interaural_cross_correlation(
        left,
        right,
        sample_rate,
        direct_index=direct_index,
        start_ms=early_end_ms,
        end_ms=late_end_ms,
        maximum_lag_ms=maximum_lag_ms,
    )
    octave_bands: dict[str, Any] = {}
    for center_hz in valid_octave_centers(sample_rate, centers_hz):
        band_left = octave_band_rir(
            left,
            sample_rate,
            center_hz,
            filter_order=filter_order,
        )
        band_right = octave_band_rir(
            right,
            sample_rate,
            center_hz,
            filter_order=filter_order,
        )
        octave_bands[f"{center_hz:g}"] = {
            "center_hz": float(center_hz),
            "early": interaural_cross_correlation(
                band_left,
                band_right,
                sample_rate,
                direct_index=direct_index,
                start_ms=0.0,
                end_ms=early_end_ms,
                maximum_lag_ms=maximum_lag_ms,
            ),
            "late": interaural_cross_correlation(
                band_left,
                band_right,
                sample_rate,
                direct_index=direct_index,
                start_ms=early_end_ms,
                end_ms=late_end_ms,
                maximum_lag_ms=maximum_lag_ms,
            ),
        }

    def _mean_valid(period: str) -> Optional[float]:
        values = [
            band[period]["iacc"]
            for band in octave_bands.values()
            if band[period]["valid"]
        ]
        return float(np.mean(values)) if values else None

    return {
        "policy": IACC_POLICY,
        "direct_sample": int(direct_index),
        "direct_policy": direct_policy,
        "early_end_ms": float(early_end_ms),
        "late_end_ms": float(late_end_ms) if late_end_ms is not None else None,
        "maximum_lag_ms": float(maximum_lag_ms),
        "broadband": {"early": early, "late": late},
        "octave_bands": octave_bands,
        "iacc_e4": _mean_valid("early"),
        "iacc_l4": _mean_valid("late"),
    }


def diffuse_field_coherence(
    frequencies_hz: Any,
    microphone_spacing_m: float,
    sound_speed_m_s: float = 343.0,
) -> np.ndarray:
    """Return the isotropic 3D diffuse-field coherence sinc(2 f d / c)."""

    if not np.isfinite(microphone_spacing_m) or microphone_spacing_m < 0.0:
        raise ValueError("microphone_spacing_m must be finite and non-negative")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be finite and positive")
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        raise ValueError("frequencies_hz must be finite and non-negative")
    return np.sinc(
        2.0 * frequencies * float(microphone_spacing_m) / float(sound_speed_m_s)
    )


def analyze_array_spatial_coherence(
    first_rir: Any,
    second_rir: Any,
    sample_rate: int,
    *,
    microphone_spacing_m: float,
    sound_speed_m_s: float = 343.0,
    direct_index: Optional[int] = None,
    start_ms: float = 80.0,
    end_ms: Optional[float] = None,
    nperseg: int = 512,
    centers_hz: Iterable[float] = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0),
) -> dict[str, Any]:
    """Compare late pair coherence with the isotropic diffuse-field target."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if nperseg < 16:
        raise ValueError("nperseg must be at least 16")
    if not np.isfinite(microphone_spacing_m) or microphone_spacing_m < 0.0:
        raise ValueError("microphone_spacing_m must be finite and non-negative")
    if not np.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be finite and positive")
    first = _as_mono_numpy(first_rir)
    second = _as_mono_numpy(second_rir)
    length = min(first.size, second.size)
    if direct_index is None:
        direct_index = min(direct_sample(first), direct_sample(second))
        direct_policy = "earliest_absolute_peak"
    else:
        direct_policy = "provided_shared_anchor"
    direct_index = int(np.clip(direct_index, 0, max(0, length - 1)))
    first_window, second_window, start, end = _pair_window(
        first,
        second,
        sample_rate,
        direct_index,
        start_ms,
        end_ms,
    )
    if first_window.size < 16:
        return {
            "policy": DIFFUSE_FIELD_COHERENCE_POLICY,
            "valid": False,
            "reason": "insufficient_late_samples",
            "direct_sample": int(direct_index),
            "direct_policy": direct_policy,
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "microphone_spacing_m": float(microphone_spacing_m),
            "sound_speed_m_s": float(sound_speed_m_s),
            "bands": {},
        }

    segment_size = min(int(nperseg), int(first_window.size))
    overlap = segment_size // 2
    frequencies, first_psd = welch(
        first_window,
        fs=float(sample_rate),
        window="hann",
        nperseg=segment_size,
        noverlap=overlap,
        detrend="constant",
    )
    _, second_psd = welch(
        second_window,
        fs=float(sample_rate),
        window="hann",
        nperseg=segment_size,
        noverlap=overlap,
        detrend="constant",
    )
    _, cross_psd = csd(
        first_window,
        second_window,
        fs=float(sample_rate),
        window="hann",
        nperseg=segment_size,
        noverlap=overlap,
        detrend="constant",
    )
    denominator = np.sqrt(np.maximum(first_psd * second_psd, 0.0))
    usable = denominator > np.finfo(np.float64).tiny
    measured = np.zeros(cross_psd.shape, dtype=np.complex128)
    measured[usable] = cross_psd[usable] / denominator[usable]
    target = diffuse_field_coherence(
        frequencies,
        microphone_spacing_m,
        sound_speed_m_s,
    )

    bands: dict[str, Any] = {}
    for center_hz in valid_octave_centers(sample_rate, centers_hz):
        low = center_hz / math.sqrt(2.0)
        high = center_hz * math.sqrt(2.0)
        mask = usable & (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            continue
        error = measured[mask] - target[mask]
        bands[f"{center_hz:g}"] = {
            "center_hz": float(center_hz),
            "frequency_bin_count": int(np.count_nonzero(mask)),
            "measured_real_mean": float(np.mean(measured[mask].real)),
            "measured_imaginary_rms": float(
                np.sqrt(np.mean(np.square(measured[mask].imag)))
            ),
            "measured_magnitude_mean": float(np.mean(np.abs(measured[mask]))),
            "diffuse_target_mean": float(np.mean(target[mask])),
            "complex_rmse": float(np.sqrt(np.mean(np.square(np.abs(error))))),
        }

    all_mask = usable & (frequencies > 0.0)
    if not np.any(all_mask):
        return {
            "policy": DIFFUSE_FIELD_COHERENCE_POLICY,
            "valid": False,
            "reason": "zero_pair_spectrum",
            "direct_sample": int(direct_index),
            "direct_policy": direct_policy,
            "window_start_sample": int(start),
            "window_end_sample": int(end),
            "microphone_spacing_m": float(microphone_spacing_m),
            "sound_speed_m_s": float(sound_speed_m_s),
            "bands": bands,
        }
    all_error = measured[all_mask] - target[all_mask]
    hop = max(1, segment_size - overlap)
    segment_count = 1 + max(0, (first_window.size - segment_size) // hop)
    return {
        "policy": DIFFUSE_FIELD_COHERENCE_POLICY,
        "valid": True,
        "reason": "ok",
        "direct_sample": int(direct_index),
        "direct_policy": direct_policy,
        "start_ms": float(start_ms),
        "end_ms": float(end_ms) if end_ms is not None else None,
        "window_start_sample": int(start),
        "window_end_sample": int(end),
        "microphone_spacing_m": float(microphone_spacing_m),
        "sound_speed_m_s": float(sound_speed_m_s),
        "nperseg": int(segment_size),
        "welch_segment_count": int(segment_count),
        "complex_rmse": float(np.sqrt(np.mean(np.square(np.abs(all_error))))),
        "measured_imaginary_rms": float(
            np.sqrt(np.mean(np.square(measured[all_mask].imag)))
        ),
        "bands": bands,
    }
