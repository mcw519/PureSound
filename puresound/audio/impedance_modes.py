"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.impedance.modes` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.impedance.modes import (
    AdmittanceModel,
    ImpedanceCavityMode1D,
    ImpedanceRoomMode3D,
    RECTANGULAR_BOUNDARIES,
    RECTANGULAR_IMPEDANCE_BOUNDARY_SCHEMA_VERSION,
    RectangularImpedanceBoundaryConfig,
    admittance_model_from_metadata,
    solve_1d_impedance_cavity_modes,
    solve_rectangular_impedance_modes,
)

__all__ = [
    "AdmittanceModel",
    "ImpedanceCavityMode1D",
    "ImpedanceRoomMode3D",
    "RECTANGULAR_BOUNDARIES",
    "RECTANGULAR_IMPEDANCE_BOUNDARY_SCHEMA_VERSION",
    "RectangularImpedanceBoundaryConfig",
    "admittance_model_from_metadata",
    "solve_1d_impedance_cavity_modes",
    "solve_rectangular_impedance_modes",
]
