"""Auditable direct/early/later decomposition for band-limited RIR probes."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


ATTRIBUTION_SCHEMA_VERSION = "puresound.rir_attribution.v1"


def complementary_early_late_masks(
    *,
    num_samples: int,
    sample_rate_hz: float,
    split_center_s: float,
    transition_width_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complementary early/later masks with a raised-cosine crossfade.

    The early mask is one before the transition, falls smoothly from one to
    zero across ``transition_width_s``, and is zero afterwards. The later mask
    is defined as ``1 - early`` so their sum is exactly one sample by sample.
    """
    length = int(num_samples)
    sample_rate = float(sample_rate_hz)
    center = float(split_center_s)
    width = float(transition_width_s)
    if length != num_samples or length <= 0:
        raise ValueError("num_samples must be a positive integer")
    if not math.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if not math.isfinite(center) or center < 0.0:
        raise ValueError("split_center_s must be finite and non-negative")
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("transition_width_s must be finite and positive")
    start = center - 0.5 * width
    stop = center + 0.5 * width
    time_s = np.arange(length, dtype=np.float64) / sample_rate
    progress = np.clip((time_s - start) / width, 0.0, 1.0)
    early = 0.5 + 0.5 * np.cos(math.pi * progress)
    early[time_s <= start] = 1.0
    early[time_s >= stop] = 0.0
    later = 1.0 - early
    return early, later


def decompose_direct_early_later(
    full_output: np.ndarray,
    direct_anchor_output: np.ndarray,
    *,
    sample_rate_hz: float,
    split_center_s: float,
    transition_width_s: float,
) -> dict[str, np.ndarray]:
    """Split an output response around a fixed direct-path anchor.

    The direct component is supplied rather than inferred from a time window.
    This is necessary when the source pulse is wider than the direct-to-first-
    reflection delay. The residual ``full - direct`` is split by complementary
    masks, making the decomposition exactly reconstructive by construction.
    """
    full = np.asarray(full_output, dtype=np.float64)
    direct = np.asarray(direct_anchor_output, dtype=np.float64)
    if full.ndim != 1 or direct.ndim != 1 or full.shape != direct.shape:
        raise ValueError("full and direct outputs must be equal-length 1D arrays")
    if full.size == 0 or not np.all(np.isfinite(full)):
        raise ValueError("full output must be non-empty and finite")
    if not np.all(np.isfinite(direct)):
        raise ValueError("direct anchor output must be finite")
    early_mask, later_mask = complementary_early_late_masks(
        num_samples=full.size,
        sample_rate_hz=sample_rate_hz,
        split_center_s=split_center_s,
        transition_width_s=transition_width_s,
    )
    residual = full - direct
    early = residual * early_mask
    later = residual * later_mask
    return {
        "direct": direct.copy(),
        "early_reflections": early,
        "later_reflections": later,
        "early_cumulative": direct + early,
        "full": full.copy(),
    }


def reconstruction_error(
    components: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Measure reconstruction of ``full = direct + early + later``."""
    required = {
        "direct",
        "early_reflections",
        "later_reflections",
        "full",
    }
    if not required.issubset(components):
        raise ValueError(f"components must contain {sorted(required)}")
    full = np.asarray(components["full"], dtype=np.float64)
    reconstructed = sum(
        (
            np.asarray(components[name], dtype=np.float64)
            for name in (
                "direct",
                "early_reflections",
                "later_reflections",
            )
        ),
        np.zeros_like(full),
    )
    difference = reconstructed - full
    return {
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "nrmse": float(
            np.linalg.norm(difference)
            / max(float(np.linalg.norm(full)), 1e-30)
        ),
    }


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "complementary_early_late_masks",
    "decompose_direct_early_later",
    "reconstruction_error",
]
