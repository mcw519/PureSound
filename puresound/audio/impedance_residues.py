"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.impedance.residues` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.impedance.residues import (
    FixedPoleResidueCase,
    IMPEDANCE_MODAL_RESIDUE_REPORT_SCHEMA_VERSION,
    IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
    ImpedanceModalResidueCalibration,
    LEGACY_IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION,
    ModalResidueFitResult,
    fit_fixed_pole_modal_residues,
)

__all__ = [
    "FixedPoleResidueCase",
    "IMPEDANCE_MODAL_RESIDUE_REPORT_SCHEMA_VERSION",
    "IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION",
    "ImpedanceModalResidueCalibration",
    "LEGACY_IMPEDANCE_MODAL_RESIDUE_SCHEMA_VERSION",
    "ModalResidueFitResult",
    "fit_fixed_pole_modal_residues",
]
