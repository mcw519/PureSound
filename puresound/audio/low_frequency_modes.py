"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.wave.low_frequency` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.wave.low_frequency import (
    LowFrequencyModeAnalysis,
    ModePeakEstimate,
    RigidRoomMode,
    estimate_low_frequency_modes,
    rigid_rectangular_room_modes,
)

__all__ = [
    "LowFrequencyModeAnalysis",
    "ModePeakEstimate",
    "RigidRoomMode",
    "estimate_low_frequency_modes",
    "rigid_rectangular_room_modes",
]
