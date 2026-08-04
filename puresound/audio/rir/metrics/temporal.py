"""Time-domain metrics: DRR, clarity, Schroeder decay, and noise floor.

Moved out of ``puresound.audio.rir.metrics`` in R4 of
``RIR_EXP_LOG.md``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from puresound.audio.rir.metrics.core import (
    DecayEstimate,
    NoiseFloorEstimate,
    _as_mono_numpy,
    _linear_fit,
    direct_sample,
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
