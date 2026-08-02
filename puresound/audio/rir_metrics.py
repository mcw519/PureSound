"""Room-impulse-response metrics shared by generators and bank benchmarks.

The functions in this module operate on one RIR channel at a time.  They accept
NumPy arrays, CPU/GPU torch tensors, and singleton-channel arrays, but reject
genuinely multi-channel input so channels cannot accidentally be concatenated
into a physically meaningless response.

Decay estimates use Schroeder backward integration followed by a least-squares
line fit.  A Lundeby-style block-energy analysis estimates the stationary noise
floor and its intersection with the fitted decay.  When the estimate is
reliable, the integration is truncated and expected noise energy is removed.
The implementation reports its dynamic range, intersection, correction status,
fit quality, and failure reason; it remains an engineering benchmark rather
than a claim of full ISO 3382 conformance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional

import numpy as np
from scipy.signal import butter, csd, sosfilt, welch


DEFAULT_OCTAVE_CENTERS_HZ = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

# A zero-mean Gaussian process places this fraction of its samples more than
# one standard deviation away from zero. Abel and Huang use it to normalize
# local echo density so that fully diffuse Gaussian noise has expectation one.
GAUSSIAN_OUTSIDE_ONE_STD_FRACTION = math.erfc(1.0 / math.sqrt(2.0))
ABEL_ECHO_DENSITY_POLICY = "puresound.abel_huang_echo_density.v1"
MULTIBAND_LATE_FIELD_POLICY = "puresound.multiband_late_field.v1"
IACC_POLICY = "puresound.binaural_iacc.v1"
DIFFUSE_FIELD_COHERENCE_POLICY = "puresound.diffuse_field_coherence.v1"


@dataclass(frozen=True)
class DecayEstimate:
    """Linear fit to a section of a Schroeder energy-decay curve."""

    rt60_s: float
    slope_db_per_s: float
    intercept_db: float
    r_squared: float
    start_db: float
    end_db: float
    fit_start_s: float
    fit_end_s: float
    sample_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class NoiseFloorEstimate:
    """Stationary tail-noise estimate and Lundeby decay intersection."""

    noise_power: float
    noise_level_db_relative: Optional[float]
    dynamic_range_db: Optional[float]
    intersection_time_s: Optional[float]
    truncation_sample: int
    block_size_samples: int
    decay_slope_db_per_s: Optional[float]
    decay_fit_r_squared: Optional[float]
    correction_applied: bool
    reason: str
    iterations: int

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        data = asdict(self)
        for key in (
            "noise_power",
            "noise_level_db_relative",
            "dynamic_range_db",
            "intersection_time_s",
            "decay_slope_db_per_s",
            "decay_fit_r_squared",
        ):
            value = data[key]
            data[key] = (
                float(value)
                if value is not None and np.isfinite(float(value))
                else None
            )
        return data


def _linear_fit(
    time_s: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, float] | None:
    if time_s.size < 2 or time_s.size != values.size:
        return None
    centered = time_s - float(time_s.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return None
    slope = float(np.dot(centered, values - float(values.mean())) / denominator)
    intercept = float(values.mean() - slope * time_s.mean())
    predicted = slope * time_s + intercept
    residual = float(np.square(values - predicted).sum())
    total = float(np.square(values - values.mean()).sum())
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return slope, intercept, float(r_squared)


def _as_mono_numpy(rir: Any) -> np.ndarray:
    """Convert one RIR channel to a finite float64 NumPy vector."""
    if hasattr(rir, "detach"):
        rir = rir.detach()
    if hasattr(rir, "cpu"):
        rir = rir.cpu()
    if hasattr(rir, "numpy"):
        rir = rir.numpy()
    x = np.asarray(rir, dtype=np.float64)
    x = np.squeeze(x)
    if x.ndim == 0:
        x = x.reshape(1)
    if x.ndim != 1:
        raise ValueError(
            f"RIR metrics expect one channel, got array with shape {np.shape(rir)}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("RIR contains NaN or infinite samples")
    return x


def direct_sample(rir: Any) -> int:
    """Return the absolute-peak sample used as the default direct arrival."""
    x = _as_mono_numpy(rir)
    return int(np.argmax(np.abs(x))) if x.size else 0


def abel_normalized_echo_density_profile(
    rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    window_ms: float = 20.0,
    hop_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Abel-Huang normalized echo-density profile.

    Times are seconds relative to the detected or provided direct sample. Only
    full analysis windows after that sample are evaluated so pre-arrival
    silence cannot bias the first estimates. Values are intentionally not
    clipped: a finite statistical estimate can be greater than one.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if window_ms <= 0.0:
        raise ValueError("window_ms must be positive")
    if hop_ms <= 0.0:
        raise ValueError("hop_ms must be positive")

    x = _as_mono_numpy(rir)
    if x.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty.copy()
    if direct_index is None:
        direct_index = direct_sample(x)
    direct_index = int(np.clip(direct_index, 0, x.size - 1))

    half_window = max(
        1,
        int(round(0.5 * window_ms * 1e-3 * sample_rate)),
    )
    window_size = 2 * half_window + 1
    hop_size = max(1, int(round(hop_ms * 1e-3 * sample_rate)))
    first_center = direct_index + half_window
    last_center = x.size - half_window
    if first_center >= last_center:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty.copy()

    centers = np.arange(
        first_center,
        last_center,
        hop_size,
        dtype=np.int64,
    )
    density = np.empty(centers.size, dtype=np.float64)
    normalizer = window_size * GAUSSIAN_OUTSIDE_ONE_STD_FRACTION
    tiny = np.finfo(np.float64).tiny

    for output_index, center in enumerate(centers):
        frame = x[center - half_window : center + half_window + 1]
        sigma = float(np.std(frame))
        if sigma <= tiny:
            density[output_index] = 0.0
            continue
        exceedances = int(np.count_nonzero(np.abs(frame) > sigma))
        density[output_index] = exceedances / normalizer

    times_s = (centers.astype(np.float64) - direct_index) / sample_rate
    return times_s, density


def estimate_abel_mixing_time(
    times_s: Any,
    normalized_density: Any,
    *,
    threshold: float = 0.9,
    minimum_sustain_ms: float = 0.0,
) -> Optional[float]:
    """Estimate direct-relative mixing time from normalized echo density.

    With ``minimum_sustain_ms=0`` this implements the original first threshold
    crossing. A positive sustain time is a PureSound robustness extension for
    rejecting isolated crossings in a strongly fluctuating finite-window
    profile.
    """

    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    if not np.isfinite(minimum_sustain_ms) or minimum_sustain_ms < 0.0:
        raise ValueError("minimum_sustain_ms must be finite and non-negative")

    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    density = np.asarray(normalized_density, dtype=np.float64).reshape(-1)
    if times.size != density.size:
        raise ValueError("times_s and normalized_density must have equal length")
    if times.size == 0:
        return None
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(density)):
        raise ValueError("profile values must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    above = density >= threshold
    if minimum_sustain_ms == 0.0 or times.size == 1:
        crossings = np.flatnonzero(above)
        return None if crossings.size == 0 else float(times[crossings[0]])

    hop_s = float(np.median(np.diff(times)))
    required = max(1, int(math.ceil(minimum_sustain_ms * 1e-3 / hop_s)))
    run_start = 0
    run_length = 0
    for index, is_above in enumerate(above):
        if is_above:
            if run_length == 0:
                run_start = index
            run_length += 1
            if run_length >= required:
                return float(times[run_start])
        else:
            run_length = 0
    return None


def analyze_echo_density(
    rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    window_ms: float = 20.0,
    hop_ms: float = 1.0,
    threshold: float = 0.9,
    minimum_sustain_ms: float = 10.0,
    probe_times_ms: Iterable[float] = (20.0, 50.0, 100.0, 200.0),
) -> dict[str, Any]:
    """Summarize monophonic echo density and direct-relative mixing time."""

    times_s, density = abel_normalized_echo_density_profile(
        rir,
        sample_rate,
        direct_index=direct_index,
        window_ms=window_ms,
        hop_ms=hop_ms,
    )
    mixing_time_s = estimate_abel_mixing_time(
        times_s,
        density,
        threshold=threshold,
        minimum_sustain_ms=minimum_sustain_ms,
    )

    probes: dict[str, Optional[float]] = {}
    for probe_time_ms in probe_times_ms:
        probe_time_ms = float(probe_time_ms)
        if not np.isfinite(probe_time_ms) or probe_time_ms < 0.0:
            raise ValueError("probe_times_ms must be finite and non-negative")
        key = f"{probe_time_ms:g}"
        if (
            times_s.size == 0
            or probe_time_ms * 1e-3 < times_s[0]
            or probe_time_ms * 1e-3 > times_s[-1]
        ):
            probes[key] = None
            continue
        nearest = int(np.argmin(np.abs(times_s - probe_time_ms * 1e-3)))
        probes[key] = float(density[nearest])

    if density.size == 0:
        late_median = None
        profile_max = None
    else:
        late_start = max(0, int(math.floor(0.75 * density.size)))
        late_median = float(np.median(density[late_start:]))
        profile_max = float(np.max(density))

    return {
        "policy": ABEL_ECHO_DENSITY_POLICY,
        "time_reference": "seconds_after_direct_sample",
        "window_ms": float(window_ms),
        "hop_ms": float(hop_ms),
        "threshold": float(threshold),
        "minimum_sustain_ms": float(minimum_sustain_ms),
        "mixing_time_s": mixing_time_s,
        "profile_sample_count": int(density.size),
        "profile_max_normalized_density": profile_max,
        "late_median_normalized_density": late_median,
        "normalized_density_at_ms": probes,
    }


def estimate_noise_floor_lundeby(
    rir: Any,
    sample_rate: int,
    direct_index: Optional[int] = None,
    block_ms: float = 10.0,
    tail_fraction: float = 0.2,
    margin_db: float = 10.0,
    min_dynamic_range_db: float = 15.0,
    max_iterations: int = 5,
) -> NoiseFloorEstimate:
    """Estimate stationary noise and the decay/noise intersection.

    This follows the practical structure of the Lundeby method: average squared
    pressure in short blocks, estimate a stationary tail floor, fit the decay
    sufficiently above that floor, solve their intersection, and refine the
    floor using blocks after the intersection. The truncation sample is an
    exclusive index in the original RIR.
    """
    x = _as_mono_numpy(rir)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if block_ms <= 0.0:
        raise ValueError("block_ms must be positive")
    if not 0.05 <= tail_fraction <= 0.5:
        raise ValueError("tail_fraction must be in [0.05, 0.5]")
    if margin_db <= 0.0 or min_dynamic_range_db <= 0.0:
        raise ValueError("noise margin and dynamic range must be positive")

    if x.size == 0:
        return NoiseFloorEstimate(
            noise_power=0.0,
            noise_level_db_relative=None,
            dynamic_range_db=None,
            intersection_time_s=None,
            truncation_sample=0,
            block_size_samples=0,
            decay_slope_db_per_s=None,
            decay_fit_r_squared=None,
            correction_applied=False,
            reason="empty_rir",
            iterations=0,
        )
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, x.size - 1))
    segment = np.asarray(x[peak:], dtype=np.float64)
    block_size = max(8, int(round(float(block_ms) * 1e-3 * sample_rate)))
    n_blocks = int(math.ceil(segment.size / block_size))
    if n_blocks < 8:
        return NoiseFloorEstimate(
            noise_power=0.0,
            noise_level_db_relative=None,
            dynamic_range_db=None,
            intersection_time_s=None,
            truncation_sample=int(x.size),
            block_size_samples=block_size,
            decay_slope_db_per_s=None,
            decay_fit_r_squared=None,
            correction_applied=False,
            reason="insufficient_blocks",
            iterations=0,
        )

    block_power = np.empty(n_blocks, dtype=np.float64)
    block_time_s = np.empty(n_blocks, dtype=np.float64)
    energy = np.square(segment, dtype=np.float64)
    for block in range(n_blocks):
        start = block * block_size
        end = min(start + block_size, segment.size)
        block_power[block] = float(energy[start:end].mean())
        block_time_s[block] = 0.5 * float(start + end) / float(sample_rate)
    peak_power = float(np.max(block_power))
    numeric_floor = max(peak_power * 1e-15, np.finfo(np.float64).tiny)
    if peak_power <= numeric_floor:
        return NoiseFloorEstimate(
            noise_power=0.0,
            noise_level_db_relative=None,
            dynamic_range_db=None,
            intersection_time_s=None,
            truncation_sample=int(x.size),
            block_size_samples=block_size,
            decay_slope_db_per_s=None,
            decay_fit_r_squared=None,
            correction_applied=False,
            reason="zero_energy",
            iterations=0,
        )

    relative_db = 10.0 * np.log10(np.maximum(block_power, numeric_floor) / peak_power)
    tail_count = max(6, int(math.ceil(n_blocks * float(tail_fraction))))
    tail_count = min(tail_count, max(2, n_blocks // 2))
    tail_slice = slice(n_blocks - tail_count, n_blocks)
    noise_power = float(np.median(block_power[tail_slice]))
    if noise_power <= numeric_floor:
        return NoiseFloorEstimate(
            noise_power=max(noise_power, 0.0),
            noise_level_db_relative=None,
            dynamic_range_db=None,
            intersection_time_s=None,
            truncation_sample=int(x.size),
            block_size_samples=block_size,
            decay_slope_db_per_s=None,
            decay_fit_r_squared=None,
            correction_applied=False,
            reason="tail_below_numeric_floor",
            iterations=0,
        )

    noise_db = float(10.0 * math.log10(noise_power / peak_power))
    fit_mask = (relative_db <= -5.0) & (relative_db >= noise_db + margin_db)
    if int(fit_mask.sum()) < 4:
        fit_mask = (relative_db <= -1.0) & (relative_db >= noise_db + margin_db)
    initial_fit = _linear_fit(block_time_s[fit_mask], relative_db[fit_mask])
    if initial_fit is None or initial_fit[0] >= 0.0:
        return NoiseFloorEstimate(
            noise_power=noise_power,
            noise_level_db_relative=noise_db,
            dynamic_range_db=float(-noise_db),
            intersection_time_s=None,
            truncation_sample=int(x.size),
            block_size_samples=block_size,
            decay_slope_db_per_s=None,
            decay_fit_r_squared=None,
            correction_applied=False,
            reason="insufficient_decay_above_noise",
            iterations=0,
        )

    tail_fit = _linear_fit(
        block_time_s[tail_slice],
        relative_db[tail_slice],
    )
    if tail_fit is not None:
        flatness_limit = max(3.0, 0.15 * abs(float(initial_fit[0])))
        if abs(float(tail_fit[0])) > flatness_limit:
            return NoiseFloorEstimate(
                noise_power=noise_power,
                noise_level_db_relative=None,
                dynamic_range_db=None,
                intersection_time_s=None,
                truncation_sample=int(x.size),
                block_size_samples=block_size,
                decay_slope_db_per_s=float(initial_fit[0]),
                decay_fit_r_squared=float(initial_fit[2]),
                correction_applied=False,
                reason="tail_is_not_stationary_noise",
                iterations=0,
            )

    decay_fit = initial_fit
    intersection = float((noise_db - decay_fit[1]) / decay_fit[0])
    iterations = 0
    duration_s = float(segment.size) / float(sample_rate)
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        if not 0.0 < intersection < duration_s:
            break
        noise_mask = block_time_s >= intersection + max(0.02, 2.0 * block_ms * 1e-3)
        if int(noise_mask.sum()) >= 3:
            noise_power = float(np.median(block_power[noise_mask]))
            noise_db = float(10.0 * math.log10(max(noise_power, numeric_floor) / peak_power))
        fit_mask = (
            (relative_db <= -5.0)
            & (relative_db >= noise_db + margin_db)
            & (block_time_s < intersection)
        )
        if int(fit_mask.sum()) < 4:
            break
        refined = _linear_fit(block_time_s[fit_mask], relative_db[fit_mask])
        if refined is None or refined[0] >= 0.0:
            break
        new_intersection = float((noise_db - refined[1]) / refined[0])
        decay_fit = refined
        if abs(new_intersection - intersection) <= block_size / float(sample_rate):
            intersection = new_intersection
            break
        intersection = new_intersection

    dynamic_range = float(-noise_db)
    valid_intersection = (
        2.0 * block_size / float(sample_rate)
        < intersection
        < duration_s - block_size / float(sample_rate)
    )
    correction_applied = bool(
        valid_intersection and dynamic_range >= float(min_dynamic_range_db)
    )
    if correction_applied:
        reason = "applied"
        truncation = peak + int(round(intersection * sample_rate))
        truncation = int(np.clip(truncation, peak + 2, x.size))
    elif dynamic_range < float(min_dynamic_range_db):
        reason = "insufficient_dynamic_range"
        truncation = int(x.size)
    else:
        reason = "intersection_outside_rir"
        truncation = int(x.size)
    return NoiseFloorEstimate(
        noise_power=float(max(noise_power, 0.0)),
        noise_level_db_relative=noise_db,
        dynamic_range_db=dynamic_range,
        intersection_time_s=(float(intersection) if np.isfinite(intersection) else None),
        truncation_sample=truncation,
        block_size_samples=block_size,
        decay_slope_db_per_s=float(decay_fit[0]),
        decay_fit_r_squared=float(decay_fit[2]),
        correction_applied=correction_applied,
        reason=reason,
        iterations=int(iterations),
    )


def compute_drr_db(
    rir: Any,
    sample_rate: int,
    direct_window_ms: float = 2.5,
    direct_index: Optional[int] = None,
) -> float:
    """Direct-to-reverberant energy ratio in dB.

    The direct window begins at ``direct_index`` (the absolute peak by default)
    and extends for ``direct_window_ms``.  This deliberately matches PureSound's
    existing training-bank cue.  Energy before the direct arrival is ignored.
    """
    x = _as_mono_numpy(rir)
    if x.size == 0:
        return float("inf")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if direct_window_ms <= 0:
        raise ValueError("direct_window_ms must be positive")
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, x.size - 1))
    window = max(1, int(round(direct_window_ms * 1e-3 * float(sample_rate))))
    direct_end = min(peak + window, x.size)
    direct_energy = float(np.square(x[peak:direct_end], dtype=np.float64).sum())
    reverb_energy = float(np.square(x[direct_end:], dtype=np.float64).sum())
    if reverb_energy <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max(direct_energy, 1e-12) / reverb_energy)


def clarity_db(
    rir: Any,
    sample_rate: int,
    boundary_ms: float = 50.0,
    direct_index: Optional[int] = None,
) -> float:
    """Early-to-late energy ratio at ``boundary_ms`` (for example C50/C80)."""
    x = _as_mono_numpy(rir)
    if x.size == 0:
        return float("inf")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if boundary_ms <= 0:
        raise ValueError("boundary_ms must be positive")
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, x.size - 1))
    boundary = min(
        peak + max(1, int(round(boundary_ms * 1e-3 * float(sample_rate)))),
        x.size,
    )
    early_energy = float(np.square(x[peak:boundary], dtype=np.float64).sum())
    late_energy = float(np.square(x[boundary:], dtype=np.float64).sum())
    if late_energy <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max(early_energy, 1e-12) / late_energy)


def schroeder_decay_db(
    rir: Any,
    direct_index: Optional[int] = None,
    floor_db: float = -120.0,
) -> np.ndarray:
    """Return the normalized backward-integrated energy curve after the direct path."""
    x = _as_mono_numpy(rir)
    if x.size == 0:
        return np.empty(0, dtype=np.float64)
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, x.size - 1))
    energy = np.square(x[peak:], dtype=np.float64)
    decay = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    total = float(decay[0]) if decay.size else 0.0
    if total <= 0.0:
        return np.full(decay.shape, np.nan, dtype=np.float64)
    normalized = np.maximum(decay / total, 10.0 ** (float(floor_db) / 10.0))
    return 10.0 * np.log10(normalized)


def noise_compensated_schroeder_decay_db(
    rir: Any,
    sample_rate: int,
    direct_index: Optional[int] = None,
    floor_db: float = -120.0,
) -> tuple[np.ndarray, NoiseFloorEstimate]:
    """Return a Schroeder curve with reliable stationary noise removed.

    If Lundeby analysis cannot identify a reliable stationary floor and an
    in-record intersection, the raw full-length Schroeder curve is returned and
    ``correction_applied`` is false.
    """
    x = _as_mono_numpy(rir)
    noise = estimate_noise_floor_lundeby(
        x,
        sample_rate=sample_rate,
        direct_index=direct_index,
    )
    if x.size == 0:
        return np.empty(0, dtype=np.float64), noise
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, x.size - 1))
    if not noise.correction_applied:
        return schroeder_decay_db(x, direct_index=peak, floor_db=floor_db), noise

    end = int(np.clip(noise.truncation_sample - peak, 2, x.size - peak))
    energy = np.square(x[peak : peak + end], dtype=np.float64)
    raw_integral = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    remaining = np.arange(end, 0, -1, dtype=np.float64)
    corrected = raw_integral - float(noise.noise_power) * remaining
    corrected = np.maximum(corrected, 0.0)
    # Expected-noise subtraction can make adjacent samples locally rise. A
    # physical integrated decay must be monotonic.
    corrected = np.minimum.accumulate(corrected)
    total = float(corrected[0]) if corrected.size else 0.0
    if total <= 0.0:
        return np.full(corrected.shape, np.nan, dtype=np.float64), noise
    normalized = np.maximum(
        corrected / total,
        10.0 ** (float(floor_db) / 10.0),
    )
    return 10.0 * np.log10(normalized), noise


def estimate_decay_time(
    rir: Any,
    sample_rate: int,
    start_db: float,
    end_db: float,
    direct_index: Optional[int] = None,
    min_fit_ms: float = 10.0,
    min_points: int = 8,
    noise_compensation: bool = True,
) -> Optional[DecayEstimate]:
    """Extrapolate RT60 from a decay-curve interval.

    Common intervals are ``(0, -10)`` for EDT, ``(-5, -25)`` for T20, and
    ``(-5, -35)`` for T30.  ``None`` means the response does not contain enough
    usable decay or that the fitted decay is non-negative.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not (start_db > end_db and start_db <= 0.0):
        raise ValueError("decay range must satisfy 0 >= start_db > end_db")
    if min_fit_ms < 0:
        raise ValueError("min_fit_ms cannot be negative")

    curve = (
        noise_compensated_schroeder_decay_db(
            rir,
            sample_rate=sample_rate,
            direct_index=direct_index,
        )[0]
        if noise_compensation
        else schroeder_decay_db(rir, direct_index=direct_index)
    )
    return _fit_decay_curve(
        curve=curve,
        sample_rate=sample_rate,
        start_db=start_db,
        end_db=end_db,
        min_fit_ms=min_fit_ms,
        min_points=min_points,
    )


def _fit_decay_curve(
    curve: np.ndarray,
    sample_rate: int,
    start_db: float,
    end_db: float,
    min_fit_ms: float = 10.0,
    min_points: int = 8,
) -> Optional[DecayEstimate]:
    """Fit one interval of an already-computed Schroeder curve."""
    if curve.size == 0 or not np.any(np.isfinite(curve)):
        return None
    indices = np.flatnonzero(
        np.isfinite(curve) & (curve <= float(start_db)) & (curve >= float(end_db))
    )
    if indices.size < max(2, int(min_points)):
        return None
    fit_duration_s = float(indices[-1] - indices[0]) / float(sample_rate)
    if fit_duration_s < float(min_fit_ms) * 1e-3:
        return None

    time_s = indices.astype(np.float64) / float(sample_rate)
    values_db = curve[indices]
    centered_t = time_s - float(time_s.mean())
    denominator = float(np.dot(centered_t, centered_t))
    if denominator <= 0.0:
        return None
    slope = float(np.dot(centered_t, values_db - float(values_db.mean())) / denominator)
    if not np.isfinite(slope) or slope >= 0.0:
        return None
    intercept = float(values_db.mean() - slope * time_s.mean())
    predicted = slope * time_s + intercept
    residual = float(np.square(values_db - predicted).sum())
    total_variation = float(np.square(values_db - values_db.mean()).sum())
    r_squared = 1.0 - residual / total_variation if total_variation > 0.0 else 1.0

    return DecayEstimate(
        rt60_s=float(-60.0 / slope),
        slope_db_per_s=slope,
        intercept_db=intercept,
        r_squared=float(r_squared),
        start_db=float(start_db),
        end_db=float(end_db),
        fit_start_s=float(time_s[0]),
        fit_end_s=float(time_s[-1]),
        sample_count=int(indices.size),
    )


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


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _decay_dict(estimate: Optional[DecayEstimate]) -> dict[str, float | int] | None:
    return estimate.to_dict() if estimate is not None else None


def analyze_multiband_late_field(
    rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    centers_hz: Iterable[float] = DEFAULT_OCTAVE_CENTERS_HZ,
    filter_order: int = 4,
    base_window_ms: float = 20.0,
    minimum_window_cycles: float = 4.0,
    hop_ms: float = 2.0,
    threshold: float = 0.9,
    minimum_sustain_ms: float = 10.0,
    probe_times_ms: Iterable[float] = (50.0, 100.0, 200.0, 400.0),
) -> dict[str, Any]:
    """Analyze noise-aware octave-band decay and echo density.

    Every band keeps the broadband direct sample as its physical time anchor.
    If Lundeby analysis finds a reliable stationary-noise intersection, echo
    density is evaluated only up to that exclusive sample so measurement noise
    is not mislabeled as a physical diffuse tail.

    The local echo-density window is at least ``minimum_window_cycles`` at the
    nominal lower octave edge. This makes low-band estimates explicit rather
    than applying a sub-cycle 20 ms window at every center frequency.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if filter_order < 1:
        raise ValueError("filter_order must be positive")
    if base_window_ms <= 0.0:
        raise ValueError("base_window_ms must be positive")
    if not np.isfinite(minimum_window_cycles) or minimum_window_cycles <= 0.0:
        raise ValueError("minimum_window_cycles must be finite and positive")

    x = _as_mono_numpy(rir)
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, max(0, x.size - 1)))
    if x.size == 0:
        return {
            "policy": MULTIBAND_LATE_FIELD_POLICY,
            "filter": "causal_butterworth_nominal_octave",
            "filter_order": int(filter_order),
            "direct_sample": 0,
            "direct_policy": (
                "absolute_peak"
                if direct_index is None
                else "provided_broadband_anchor"
            ),
            "window_policy": "max_base_or_minimum_cycles_at_lower_octave_edge",
            "base_window_ms": float(base_window_ms),
            "minimum_window_cycles": float(minimum_window_cycles),
            "hop_ms": float(hop_ms),
            "threshold": float(threshold),
            "minimum_sustain_ms": float(minimum_sustain_ms),
            "bands": {},
        }
    bands: dict[str, Any] = {}
    for center_hz in valid_octave_centers(sample_rate, centers_hz):
        band = octave_band_rir(
            x,
            sample_rate,
            center_hz,
            filter_order=filter_order,
        )
        decay_curve, noise_floor = noise_compensated_schroeder_decay_db(
            band,
            sample_rate=sample_rate,
            direct_index=peak,
        )
        analysis_end = (
            int(noise_floor.truncation_sample)
            if noise_floor.correction_applied
            else int(band.size)
        )
        analysis_end = int(np.clip(analysis_end, peak + 1, band.size))
        lower_edge_hz = center_hz / math.sqrt(2.0)
        cycle_window_ms = 1000.0 * minimum_window_cycles / lower_edge_hz
        window_ms = max(float(base_window_ms), float(cycle_window_ms))
        echo_density = analyze_echo_density(
            band[:analysis_end],
            sample_rate,
            direct_index=peak,
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold=threshold,
            minimum_sustain_ms=minimum_sustain_ms,
            probe_times_ms=probe_times_ms,
        )
        edt = _fit_decay_curve(decay_curve, sample_rate, 0.0, -10.0)
        t20 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -25.0)
        t30 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -35.0)
        bands[f"{center_hz:g}"] = {
            "center_hz": float(center_hz),
            "lower_edge_hz": float(lower_edge_hz),
            "analysis_window_ms": float(window_ms),
            "analysis_end_sample": int(analysis_end),
            "analysis_end_s_after_direct": float(
                max(0, analysis_end - peak) / sample_rate
            ),
            "echo_density_truncated_at_noise_intersection": bool(
                noise_floor.correction_applied
            ),
            "echo_density": echo_density,
            "noise_floor": noise_floor.to_dict(),
            "edt": _decay_dict(edt),
            "t20": _decay_dict(t20),
            "t30": _decay_dict(t30),
            "edt_s": edt.rt60_s if edt is not None else None,
            "t20_s": t20.rt60_s if t20 is not None else None,
            "t30_s": t30.rt60_s if t30 is not None else None,
        }
    return {
        "policy": MULTIBAND_LATE_FIELD_POLICY,
        "filter": "causal_butterworth_nominal_octave",
        "filter_order": int(filter_order),
        "direct_sample": int(peak),
        "direct_policy": (
            "absolute_peak" if direct_index is None else "provided_broadband_anchor"
        ),
        "window_policy": "max_base_or_minimum_cycles_at_lower_octave_edge",
        "base_window_ms": float(base_window_ms),
        "minimum_window_cycles": float(minimum_window_cycles),
        "hop_ms": float(hop_ms),
        "threshold": float(threshold),
        "minimum_sustain_ms": float(minimum_sustain_ms),
        "bands": bands,
    }


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


def analyze_rir(
    rir: Any,
    sample_rate: int,
    direct_window_ms: float = 2.5,
    direct_index: Optional[int] = None,
    octave_centers_hz: Optional[Iterable[float]] = None,
    noise_compensation: bool = True,
    echo_density: bool = False,
    echo_density_window_ms: float = 20.0,
    echo_density_hop_ms: float = 1.0,
    echo_density_threshold: float = 0.9,
    echo_density_minimum_sustain_ms: float = 10.0,
) -> dict[str, Any]:
    """Compute a JSON-safe set of broadband and optional octave-band metrics."""
    x = _as_mono_numpy(rir)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, max(0, x.size - 1)))
    if noise_compensation:
        decay_curve, noise_floor = noise_compensated_schroeder_decay_db(
            x,
            sample_rate=sample_rate,
            direct_index=peak,
        )
    else:
        decay_curve = schroeder_decay_db(x, direct_index=peak)
        noise_floor = estimate_noise_floor_lundeby(
            x,
            sample_rate=sample_rate,
            direct_index=peak,
        )
    edt = _fit_decay_curve(decay_curve, sample_rate, 0.0, -10.0)
    t20 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -25.0)
    t30 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -35.0)
    result: dict[str, Any] = {
        "sample_rate": int(sample_rate),
        "num_samples": int(x.size),
        "direct_sample": int(peak),
        "direct_delay_ms": float(peak / float(sample_rate) * 1000.0),
        "peak_abs": float(np.max(np.abs(x))) if x.size else 0.0,
        "energy": float(np.square(x, dtype=np.float64).sum()),
        "drr_db": _finite_or_none(
            compute_drr_db(
                x,
                sample_rate,
                direct_window_ms=direct_window_ms,
                direct_index=peak,
            )
        ),
        "c50_db": _finite_or_none(clarity_db(x, sample_rate, 50.0, peak)),
        "c80_db": _finite_or_none(clarity_db(x, sample_rate, 80.0, peak)),
        "spectral_tilt_db_per_octave": _finite_or_none(
            spectral_tilt_db_per_octave(x, sample_rate)
        ),
        "noise_floor": noise_floor.to_dict(),
        "edt": _decay_dict(edt),
        "t20": _decay_dict(t20),
        "t30": _decay_dict(t30),
        "edt_s": edt.rt60_s if edt is not None else None,
        "t20_s": t20.rt60_s if t20 is not None else None,
        "t30_s": t30.rt60_s if t30 is not None else None,
    }

    if octave_centers_hz is not None:
        octave_metrics: dict[str, Any] = {}
        for center in valid_octave_centers(sample_rate, octave_centers_hz):
            band = octave_band_rir(x, sample_rate, center)
            band_peak = direct_sample(band)
            if noise_compensation:
                (
                    band_curve,
                    band_noise_floor,
                ) = noise_compensated_schroeder_decay_db(
                    band,
                    sample_rate=sample_rate,
                    direct_index=band_peak,
                )
            else:
                band_curve = schroeder_decay_db(band, direct_index=band_peak)
                band_noise_floor = estimate_noise_floor_lundeby(
                    band,
                    sample_rate=sample_rate,
                    direct_index=band_peak,
                )
            band_edt = _fit_decay_curve(band_curve, sample_rate, 0.0, -10.0)
            band_t20 = _fit_decay_curve(band_curve, sample_rate, -5.0, -25.0)
            band_t30 = _fit_decay_curve(band_curve, sample_rate, -5.0, -35.0)
            octave_metrics[f"{center:g}"] = {
                "center_hz": center,
                "drr_db": _finite_or_none(
                    compute_drr_db(
                        band,
                        sample_rate,
                        direct_window_ms=direct_window_ms,
                        direct_index=band_peak,
                    )
                ),
                "c50_db": _finite_or_none(
                    clarity_db(band, sample_rate, 50.0, band_peak)
                ),
                "edt_s": band_edt.rt60_s if band_edt is not None else None,
                "t20_s": band_t20.rt60_s if band_t20 is not None else None,
                "t30_s": band_t30.rt60_s if band_t30 is not None else None,
                "decay_fit_r2": {
                    "edt": band_edt.r_squared if band_edt is not None else None,
                    "t20": band_t20.r_squared if band_t20 is not None else None,
                    "t30": band_t30.r_squared if band_t30 is not None else None,
                },
                "noise_floor": band_noise_floor.to_dict(),
            }
        result["octave_bands"] = octave_metrics
    if echo_density:
        result["echo_density"] = analyze_echo_density(
            x,
            sample_rate,
            direct_index=peak,
            window_ms=echo_density_window_ms,
            hop_ms=echo_density_hop_ms,
            threshold=echo_density_threshold,
            minimum_sustain_ms=echo_density_minimum_sustain_ms,
        )
    return result


__all__ = [
    "ABEL_ECHO_DENSITY_POLICY",
    "DEFAULT_OCTAVE_CENTERS_HZ",
    "DIFFUSE_FIELD_COHERENCE_POLICY",
    "GAUSSIAN_OUTSIDE_ONE_STD_FRACTION",
    "IACC_POLICY",
    "MULTIBAND_LATE_FIELD_POLICY",
    "DecayEstimate",
    "NoiseFloorEstimate",
    "abel_normalized_echo_density_profile",
    "analyze_array_spatial_coherence",
    "analyze_binaural_iacc",
    "analyze_echo_density",
    "analyze_multiband_late_field",
    "analyze_rir",
    "clarity_db",
    "compute_drr_db",
    "diffuse_field_coherence",
    "direct_sample",
    "estimate_abel_mixing_time",
    "estimate_decay_time",
    "estimate_noise_floor_lundeby",
    "interaural_cross_correlation",
    "noise_compensated_schroeder_decay_db",
    "octave_band_rir",
    "schroeder_decay_db",
    "spectral_tilt_db_per_octave",
    "valid_octave_centers",
]
