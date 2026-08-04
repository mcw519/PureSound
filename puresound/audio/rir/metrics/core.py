"""Shared primitives for every RIR metric.

Mono coercion, the geometric direct-sample convention, linear fits and the
decay/noise-floor result types.  Moved out of ``puresound.audio.rir.metrics``
in R4 of ``RIR_EXP_LOG.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np



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


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _decay_dict(estimate: Optional[DecayEstimate]) -> dict[str, float | int] | None:
    return estimate.to_dict() if estimate is not None else None
