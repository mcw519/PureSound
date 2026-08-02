"""Compatibility shim.

The implementation moved to :mod:`puresound.audio.rir.physics.impedance.priors` during R7 of
``RIR_MODULARIZATION_PLAN.md``.  This module re-exports it unchanged; prefer
the new path in new code.
"""

from puresound.audio.rir.physics.impedance.priors import (
    EVIDENCE_TIER_MEASURED_PARAMETER_MODEL,
    IMPEDANCE_PRIOR_CATALOG_VERSION,
    ImpedancePriorFit,
    MIKI_1990_DOI,
    MikiPorousLayerPrior,
    TARNOW_2002_DOI,
    fit_first_order_relaxation,
    reference_impedance_priors,
)

__all__ = [
    "EVIDENCE_TIER_MEASURED_PARAMETER_MODEL",
    "IMPEDANCE_PRIOR_CATALOG_VERSION",
    "ImpedancePriorFit",
    "MIKI_1990_DOI",
    "MikiPorousLayerPrior",
    "TARNOW_2002_DOI",
    "fit_first_order_relaxation",
    "reference_impedance_priors",
]
