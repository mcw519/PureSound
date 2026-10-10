"""Source and receiver directivity.

First-order patterns (``cardioid`` family) are one real gain per direction.
``speech_human`` is a measured, frequency-dependent talker pattern: a person
radiates almost omnidirectionally at 125 Hz and is 19-26 dB quieter behind
the head at 4-8 kHz, which a single first-order gain cannot express (the
``speech_cardioid`` alias of ``cardioid`` has a null directly behind at every
frequency).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from puresound.audio.rir.path_events.schema import (
    PathBandGain,
    _finite_float_list,
    _unit_vector3,
)

#: Directivity IDs whose gain depends on frequency; they produce a
#: :class:`PathBandGain` through :func:`directivity_band_gain`.
FREQUENCY_DEPENDENT_DIRECTIVITIES = frozenset({"speech_human"})

SPEECH_DIRECTIVITY_ANGLES_DEG = tuple(range(0, 181, 15))
SPEECH_DIRECTIVITY_BANDS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
# Horizontal-plane octave-band levels of normal-level speech, dB SPL at 1 m,
# one row per angle from the mouth axis: Monson, Hunter & Story, "Horizontal
# directivity of low- and high-frequency energy in speech and singing",
# JASA 132(1), 433-441 (2012), Table I.
_SPEECH_LEVELS_DB = np.array(
    [
        [50.2, 56.0, 57.4, 55.0, 51.2, 46.3, 46.4],
        [51.0, 56.7, 58.0, 54.8, 50.2, 44.6, 45.4],
        [50.1, 55.8, 57.5, 54.3, 51.1, 45.6, 44.7],
        [49.8, 55.5, 57.4, 53.6, 50.4, 44.7, 43.3],
        [49.6, 55.2, 57.3, 53.5, 48.7, 42.4, 40.6],
        [49.1, 54.4, 56.7, 53.6, 46.5, 41.0, 39.4],
        [48.6, 53.7, 55.8, 53.4, 44.2, 40.0, 37.7],
        [48.1, 52.9, 54.6, 52.5, 43.6, 37.3, 34.9],
        [47.5, 52.1, 53.0, 50.4, 43.3, 33.6, 31.4],
        [47.1, 51.5, 51.9, 48.0, 41.9, 31.8, 28.6],
        [46.7, 51.1, 51.4, 47.1, 38.2, 29.6, 24.9],
        [46.6, 51.0, 51.4, 48.1, 36.2, 25.3, 21.4],
        [46.6, 50.9, 51.5, 48.7, 38.1, 27.4, 20.1],
    ]
)
SPEECH_DIRECTIVITY_RELATIVE_DB = _SPEECH_LEVELS_DB - _SPEECH_LEVELS_DB[0]


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
    if pattern in FREQUENCY_DEPENDENT_DIRECTIVITIES:
        raise ValueError(
            f"{pattern} depends on frequency; use directivity_band_gain"
        )
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


def directivity_band_gain(
    directivity_id: str,
    orientation_ypr_deg: Iterable[float],
    direction_from_transducer_unit: Iterable[float],
) -> PathBandGain:
    """Octave-band pressure magnitude of a frequency-dependent pattern.

    ``speech_human`` treats the measured horizontal pattern as symmetric
    about the mouth axis: the gain depends only on the angle between the
    forward axis and the departure direction, interpolated linearly in dB
    between the 15-degree table rows.  Elevation therefore reuses the
    horizontal data, the usual simplification when only one plane is
    measured.
    """

    if directivity_id != "speech_human":
        raise NotImplementedError(
            f"no frequency-dependent model for directivity: {directivity_id}"
        )
    forward = orientation_forward_unit(orientation_ypr_deg)
    direction = np.asarray(
        _unit_vector3("directivity direction", direction_from_transducer_unit),
        dtype=np.float64,
    )
    angle = math.degrees(math.acos(float(np.clip(np.dot(forward, direction), -1, 1))))
    relative_db = [
        np.interp(angle, SPEECH_DIRECTIVITY_ANGLES_DEG, column)
        for column in SPEECH_DIRECTIVITY_RELATIVE_DB.T
    ]
    return PathBandGain(
        frequencies_hz=list(SPEECH_DIRECTIVITY_BANDS_HZ),
        magnitude=[10.0 ** (value / 20.0) for value in relative_db],
        provenance="speech_human_monson2012_horizontal",
    )


__all__ = [
    "FREQUENCY_DEPENDENT_DIRECTIVITIES",
    "SPEECH_DIRECTIVITY_ANGLES_DEG",
    "SPEECH_DIRECTIVITY_BANDS_HZ",
    "SPEECH_DIRECTIVITY_RELATIVE_DB",
    "directivity_band_gain",
    "directivity_pressure_gain",
    "orientation_forward_unit",
]
