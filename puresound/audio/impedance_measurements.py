"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.impedance.measurements` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.impedance.measurements import (
    COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION,
    ComplexImpedanceMeasurement,
    NORMALIZED_COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION,
    NormalizedComplexImpedanceMeasurement,
    OPTIONAL_NORMALIZED_STD_COLUMNS,
    OPTIONAL_STD_COLUMNS,
    REQUIRED_CSV_COLUMNS,
    REQUIRED_NORMALIZED_CSV_COLUMNS,
)

__all__ = [
    "COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION",
    "ComplexImpedanceMeasurement",
    "NORMALIZED_COMPLEX_IMPEDANCE_MEASUREMENT_SCHEMA_VERSION",
    "NormalizedComplexImpedanceMeasurement",
    "OPTIONAL_NORMALIZED_STD_COLUMNS",
    "OPTIONAL_STD_COLUMNS",
    "REQUIRED_CSV_COLUMNS",
    "REQUIRED_NORMALIZED_CSV_COLUMNS",
]
