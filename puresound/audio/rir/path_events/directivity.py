"""First-order source and receiver directivity.

Moved out of ``puresound.audio.rir.path_events`` in R3 of
``RIR_EXP_LOG.md``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from puresound.audio.rir.path_events.schema import _finite_float_list, _unit_vector3


def orientation_forward_unit(orientation_ypr_deg: Iterable[float]) -> np.ndarray:
    """Return the forward axis for yaw/pitch/roll Euler metadata."""

    orientation = _finite_float_list("orientation", orientation_ypr_deg)
    if len(orientation) != 3:
        raise ValueError("orientation_ypr_deg must contain yaw, pitch, and roll")
    yaw = math.radians(orientation[0])
    pitch = math.radians(orientation[1])
    return np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )


def directivity_pressure_gain(
    directivity_id: str,
    orientation_ypr_deg: Iterable[float],
    direction_from_transducer_unit: Iterable[float],
) -> float:
    """Evaluate supported real first-order source/receiver pressure patterns."""

    pattern = str(directivity_id)
    if pattern == "omnidirectional":
        return 1.0
    alpha_by_pattern = {
        "speech_cardioid": 0.5,
        "cardioid": 0.5,
        "hypercardioid": 0.25,
        "figure_eight": 0.0,
    }
    if pattern not in alpha_by_pattern:
        raise NotImplementedError(f"unsupported directivity: {pattern}")
    forward = orientation_forward_unit(orientation_ypr_deg)
    direction = np.asarray(
        _unit_vector3("directivity direction", direction_from_transducer_unit),
        dtype=np.float64,
    )
    alpha = alpha_by_pattern[pattern]
    return float(alpha + (1.0 - alpha) * np.dot(forward, direction))


__all__ = [
    "directivity_pressure_gain",
    "orientation_forward_unit",
]
