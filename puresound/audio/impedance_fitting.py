"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.impedance.fitting` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.impedance.fitting import (
    ComplexReflectionFitMetrics,
    PassiveMultiPoleFit,
    PassiveResonantFit,
    fit_complex_impedance_measurement,
    fit_normalized_complex_impedance_measurement,
    fit_passive_multi_pole_admittance,
    fit_passive_single_resonance_admittance,
)

__all__ = [
    "ComplexReflectionFitMetrics",
    "PassiveMultiPoleFit",
    "PassiveResonantFit",
    "fit_complex_impedance_measurement",
    "fit_normalized_complex_impedance_measurement",
    "fit_passive_multi_pole_admittance",
    "fit_passive_single_resonance_admittance",
]
