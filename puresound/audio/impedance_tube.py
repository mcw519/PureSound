"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.impedance.tube` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.impedance.tube import (
    IMPEDANCE_TUBE_TRANSFER_MEASUREMENT_SCHEMA_VERSION,
    ImpedanceTubeReduction,
    MICROPHONE_SWITCH_CSV_COLUMNS,
    RAW_TRANSFER_CSV_COLUMNS,
    TwoMicrophoneTubeGeometry,
    load_microphone_switch_csv,
    load_transfer_repeats_csv,
    microphone_switch_calibration_factor,
    reduce_two_microphone_repeats,
    reflection_from_two_microphone_transfer,
    transfer_from_surface_reflection,
)

__all__ = [
    "IMPEDANCE_TUBE_TRANSFER_MEASUREMENT_SCHEMA_VERSION",
    "ImpedanceTubeReduction",
    "MICROPHONE_SWITCH_CSV_COLUMNS",
    "RAW_TRANSFER_CSV_COLUMNS",
    "TwoMicrophoneTubeGeometry",
    "load_microphone_switch_csv",
    "load_transfer_repeats_csv",
    "microphone_switch_calibration_factor",
    "reduce_two_microphone_repeats",
    "reflection_from_two_microphone_transfer",
    "transfer_from_surface_reflection",
]
