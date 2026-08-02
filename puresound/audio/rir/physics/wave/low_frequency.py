"""Low-frequency resonance metrics for room impulse responses.

The estimator intentionally works on the measured pressure response rather
than scene metadata. It finds prominent peaks in a gated RIR spectrum and
estimates each resonance's half-power bandwidth and Q factor. The result is
suited to simulator/reference comparisons and bank-level distribution checks;
overlapping or unresolved modes are reported without a Q estimate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class ModePeakEstimate:
    frequency_hz: float
    magnitude_db_relative: float
    prominence_db: float
    bandwidth_hz: Optional[float]
    q_factor: Optional[float]
    lower_half_power_hz: Optional[float]
    upper_half_power_hz: Optional[float]

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True)
class LowFrequencyModeAnalysis:
    sample_rate_hz: float
    analysis_start_s: float
    analysis_duration_s: float
    min_frequency_hz: float
    max_frequency_hz: float
    native_resolution_hz: float
    fft_bin_spacing_hz: float
    peaks: tuple[ModePeakEstimate, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["peaks"] = [peak.to_dict() for peak in self.peaks]
        data["resolved_q_count"] = sum(
            peak.q_factor is not None for peak in self.peaks
        )
        return data


@dataclass(frozen=True)
class RigidRoomMode:
    indices: tuple[int, int, int]
    frequency_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": list(self.indices),
            "frequency_hz": float(self.frequency_hz),
        }


def _mono_float64(signal: Any) -> np.ndarray:
    if hasattr(signal, "detach"):
        signal = signal.detach()
    if hasattr(signal, "cpu"):
        signal = signal.cpu()
    if hasattr(signal, "numpy"):
        signal = signal.numpy()
    values = np.squeeze(np.asarray(signal, dtype=np.float64))
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(
            "modal analysis expects one channel, "
            f"got array with shape {np.shape(signal)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("modal analysis input contains NaN or infinite samples")
    return values


def _interpolated_crossing_hz(
    frequencies_hz: np.ndarray,
    magnitude: np.ndarray,
    first_index: int,
    second_index: int,
    target: float,
) -> float:
    first = float(magnitude[first_index])
    second = float(magnitude[second_index])
    if math.isclose(first, second):
        return 0.5 * float(
            frequencies_hz[first_index] + frequencies_hz[second_index]
        )
    fraction = float(np.clip((target - first) / (second - first), 0.0, 1.0))
    return float(
        frequencies_hz[first_index]
        + fraction
        * (frequencies_hz[second_index] - frequencies_hz[first_index])
    )


def _half_power_bandwidth(
    frequencies_hz: np.ndarray,
    magnitude: np.ndarray,
    peak_index: int,
    lower_limit_index: int,
    upper_limit_index: int,
) -> tuple[float, float, float] | None:
    target = float(magnitude[peak_index]) / math.sqrt(2.0)
    left = int(peak_index)
    while left > lower_limit_index and magnitude[left] > target:
        left -= 1
    right = int(peak_index)
    while right < upper_limit_index and magnitude[right] > target:
        right += 1
    if (
        left == peak_index
        or right == peak_index
        or magnitude[left] > target
        or magnitude[right] > target
    ):
        return None
    lower_hz = _interpolated_crossing_hz(
        frequencies_hz,
        magnitude,
        left,
        left + 1,
        target,
    )
    upper_hz = _interpolated_crossing_hz(
        frequencies_hz,
        magnitude,
        right - 1,
        right,
        target,
    )
    bandwidth_hz = upper_hz - lower_hz
    if bandwidth_hz <= 0.0:
        return None
    return lower_hz, upper_hz, bandwidth_hz


def estimate_low_frequency_modes(
    rir: Any,
    sample_rate_hz: float,
    *,
    min_frequency_hz: float = 20.0,
    max_frequency_hz: float = 300.0,
    analysis_start_s: float = 0.0,
    analysis_duration_s: Optional[float] = 2.0,
    min_prominence_db: float = 6.0,
    min_separation_hz: float = 3.0,
    zero_pad_factor: int = 8,
    max_peaks: Optional[int] = None,
) -> LowFrequencyModeAnalysis:
    """Find low-frequency modal peaks and estimate spectral Q.

    Q is ``peak frequency / -3 dB bandwidth``. Zero padding improves crossing
    interpolation but ``native_resolution_hz`` still records the information
    limit imposed by the gated response length.
    """
    values = _mono_float64(rir)
    sample_rate = float(sample_rate_hz)
    if sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not 0.0 <= min_frequency_hz < max_frequency_hz < 0.5 * sample_rate:
        raise ValueError("frequency range must be ordered and below Nyquist")
    if analysis_start_s < 0.0:
        raise ValueError("analysis_start_s cannot be negative")
    if analysis_duration_s is not None and analysis_duration_s <= 0.0:
        raise ValueError("analysis_duration_s must be positive when supplied")
    if min_prominence_db <= 0.0 or min_separation_hz <= 0.0:
        raise ValueError("peak prominence and separation must be positive")
    if zero_pad_factor < 1:
        raise ValueError("zero_pad_factor must be at least one")

    start = int(round(float(analysis_start_s) * sample_rate))
    stop = values.size
    if analysis_duration_s is not None:
        stop = min(
            stop,
            start + int(round(float(analysis_duration_s) * sample_rate)),
        )
    segment = np.asarray(values[start:stop], dtype=np.float64)
    if segment.size < 16:
        raise ValueError("modal analysis gate must contain at least 16 samples")
    segment = segment - float(np.mean(segment))

    requested_fft_size = max(segment.size, segment.size * int(zero_pad_factor))
    fft_size = 1 << int(math.ceil(math.log2(requested_fft_size)))
    frequencies_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    magnitude = np.abs(np.fft.rfft(segment, n=fft_size))
    reference = float(np.max(magnitude))
    if reference <= np.finfo(np.float64).tiny:
        return LowFrequencyModeAnalysis(
            sample_rate_hz=sample_rate,
            analysis_start_s=start / sample_rate,
            analysis_duration_s=segment.size / sample_rate,
            min_frequency_hz=float(min_frequency_hz),
            max_frequency_hz=float(max_frequency_hz),
            native_resolution_hz=sample_rate / segment.size,
            fft_bin_spacing_hz=sample_rate / fft_size,
            peaks=(),
        )

    magnitude_db = 20.0 * np.log10(
        np.maximum(magnitude, reference * 1e-15) / reference
    )
    in_band = np.flatnonzero(
        (frequencies_hz >= float(min_frequency_hz))
        & (frequencies_hz <= float(max_frequency_hz))
    )
    distance_bins = max(
        1,
        int(math.ceil(float(min_separation_hz) / (sample_rate / fft_size))),
    )
    local_peaks, properties = find_peaks(
        magnitude_db[in_band],
        prominence=float(min_prominence_db),
        distance=distance_bins,
    )
    peak_indices = in_band[local_peaks]
    prominences = np.asarray(properties["prominences"], dtype=np.float64)
    if max_peaks is not None and peak_indices.size > max_peaks:
        keep = np.argsort(prominences)[-int(max_peaks) :]
        keep.sort()
        peak_indices = peak_indices[keep]
        prominences = prominences[keep]

    estimates: list[ModePeakEstimate] = []
    lower_limit = int(in_band[0])
    upper_limit = int(in_band[-1])
    for peak_index, prominence_db in zip(peak_indices, prominences):
        bandwidth = _half_power_bandwidth(
            frequencies_hz,
            magnitude,
            int(peak_index),
            lower_limit,
            upper_limit,
        )
        frequency_hz = float(frequencies_hz[peak_index])
        if bandwidth is None:
            lower_hz = upper_hz = bandwidth_hz = q_factor = None
        else:
            lower_hz, upper_hz, bandwidth_hz = bandwidth
            q_factor = frequency_hz / bandwidth_hz
        estimates.append(
            ModePeakEstimate(
                frequency_hz=frequency_hz,
                magnitude_db_relative=float(magnitude_db[peak_index]),
                prominence_db=float(prominence_db),
                bandwidth_hz=bandwidth_hz,
                q_factor=q_factor,
                lower_half_power_hz=lower_hz,
                upper_half_power_hz=upper_hz,
            )
        )
    return LowFrequencyModeAnalysis(
        sample_rate_hz=sample_rate,
        analysis_start_s=start / sample_rate,
        analysis_duration_s=segment.size / sample_rate,
        min_frequency_hz=float(min_frequency_hz),
        max_frequency_hz=float(max_frequency_hz),
        native_resolution_hz=sample_rate / segment.size,
        fft_bin_spacing_hz=sample_rate / fft_size,
        peaks=tuple(estimates),
    )


def rigid_rectangular_room_modes(
    room_dim_m: tuple[float, float, float],
    *,
    max_frequency_hz: float,
    sound_speed_m_s: float = 343.0,
) -> tuple[RigidRoomMode, ...]:
    """Enumerate non-DC rigid-wall modes of a rectangular room."""
    if (
        len(room_dim_m) != 3
        or any(float(length) <= 0.0 for length in room_dim_m)
        or max_frequency_hz <= 0.0
        or sound_speed_m_s <= 0.0
    ):
        raise ValueError("room, frequency limit, and sound speed must be positive")
    lx, ly, lz = (float(length) for length in room_dim_m)
    limits = [
        int(math.floor(2.0 * float(max_frequency_hz) * length / sound_speed_m_s))
        for length in (lx, ly, lz)
    ]
    modes: list[RigidRoomMode] = []
    for nx in range(limits[0] + 1):
        for ny in range(limits[1] + 1):
            for nz in range(limits[2] + 1):
                if nx == ny == nz == 0:
                    continue
                frequency_hz = 0.5 * float(sound_speed_m_s) * math.sqrt(
                    (nx / lx) ** 2 + (ny / ly) ** 2 + (nz / lz) ** 2
                )
                if frequency_hz <= float(max_frequency_hz):
                    modes.append(
                        RigidRoomMode(
                            indices=(nx, ny, nz),
                            frequency_hz=frequency_hz,
                        )
                    )
    return tuple(sorted(modes, key=lambda mode: (mode.frequency_hz, mode.indices)))


__all__ = [
    "LowFrequencyModeAnalysis",
    "ModePeakEstimate",
    "RigidRoomMode",
    "estimate_low_frequency_modes",
    "rigid_rectangular_room_modes",
]
