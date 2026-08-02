"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.wave.fdtd` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.wave.fdtd import (
    BOUNDARIES,
    FDTDDiscretePlaneWaveReflection,
    FDTDReferenceConfig,
    FDTDReferenceResult,
    FOURTH_ORDER_CLOSURE_CFL_LIMIT,
    FOURTH_ORDER_QUADRATIC_BOUNDARY_COURANT_LIMIT,
    absorption_to_impedance,
    fdtd_discrete_plane_wave_reflection,
    impedance_to_absorption,
    ricker_source,
    simulate_fdtd_reference,
    simulate_reciprocal_fdtd_reference,
)

__all__ = [
    "BOUNDARIES",
    "FDTDDiscretePlaneWaveReflection",
    "FDTDReferenceConfig",
    "FDTDReferenceResult",
    "FOURTH_ORDER_CLOSURE_CFL_LIMIT",
    "FOURTH_ORDER_QUADRATIC_BOUNDARY_COURANT_LIMIT",
    "absorption_to_impedance",
    "fdtd_discrete_plane_wave_reflection",
    "impedance_to_absorption",
    "ricker_source",
    "simulate_fdtd_reference",
    "simulate_reciprocal_fdtd_reference",
]
