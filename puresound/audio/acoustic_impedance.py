"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.impedance.admittance` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.impedance.admittance import (
    DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION,
    DigitalBoundaryReflectionFilter,
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
    RationalAdmittanceModel,
    characteristic_impedance_pa_s_m,
    digital_locally_reacting_reflection_filter,
    digital_normalized_admittance_filter,
    impedance_from_absorption_and_phase,
    impedance_from_normal_incidence_reflection,
    normal_incidence_absorption_coefficient,
    normal_incidence_reflection_coefficient,
)

__all__ = [
    "DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION",
    "DigitalBoundaryReflectionFilter",
    "FirstOrderRelaxationAdmittance",
    "PassiveMultiPoleAdmittance",
    "PassiveResonantAdmittance",
    "RationalAdmittanceModel",
    "characteristic_impedance_pa_s_m",
    "digital_locally_reacting_reflection_filter",
    "digital_normalized_admittance_filter",
    "impedance_from_absorption_and_phase",
    "impedance_from_normal_incidence_reflection",
    "normal_incidence_absorption_coefficient",
    "normal_incidence_reflection_coefficient",
]
