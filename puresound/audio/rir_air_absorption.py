"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.propagation` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.propagation import (
    AIR_ABSORPTION_POLICY,
    air_adjusted_rt60_s,
    apply_air_absorption,
    atmospheric_absorption_db_per_m,
    minimum_phase_air_absorption_filter,
)

__all__ = [
    "AIR_ABSORPTION_POLICY",
    "air_adjusted_rt60_s",
    "apply_air_absorption",
    "atmospheric_absorption_db_per_m",
    "minimum_phase_air_absorption_filter",
]
