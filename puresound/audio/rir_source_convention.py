"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.wave.source_convention` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.wave.source_convention import (
    FDTD_PRESSURE_CELL_SOURCE_CONVENTION,
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
    PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM,
    convert_pressure_state_modal_residue,
    fdtd_cell_center_position,
    fdtd_pressure_cell_free_field_direct,
    fdtd_pressure_cell_to_free_field_input,
    pressure_state_modal_residue_conversion_factor,
)

__all__ = [
    "FDTD_PRESSURE_CELL_SOURCE_CONVENTION",
    "FREE_FIELD_1_OVER_R_RIR_CONVENTION",
    "PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM",
    "convert_pressure_state_modal_residue",
    "fdtd_cell_center_position",
    "fdtd_pressure_cell_free_field_direct",
    "fdtd_pressure_cell_to_free_field_input",
    "pressure_state_modal_residue_conversion_factor",
]
